//! `kv-residency`: which of each call's prompt blocks the server held when
//! the call was sent and when its first token arrived, from a replay log and
//! the KV event log recorded with it (`[replay.kv_events]`).
//!
//! A block is held while the server has stored more copies of its hash than
//! it has removed. Each KV cache group is tracked separately. A group whose
//! stored events cover more tokens than their blocks skips blocks: Mamba
//! groups in vLLM's "align" prefix caching, and sliding-window groups. A
//! prefix of `k` blocks counts as reusable if every other group holds blocks
//! `0..k` and every skipping group holds block `k - 1`, the state at the end
//! of the prefix. That rule is right for Mamba groups and overstates what a
//! sliding-window group can serve.
//!
//! vLLM stores a call's prompt blocks when it schedules the call, before its
//! first token. At the first token, a block counts as held only if it has
//! been held without a break since the call was sent, so the call's own
//! store of an evicted prefix does not count.

use super::kv_events::{KvEventKind, KvEventRecord};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{BufRead, Write};
use std::path::Path;

/// The fields of a replay log line this analysis reads.
#[derive(Debug, Clone, Deserialize)]
pub struct CallLine {
    pub session_id: String,
    pub call: usize,
    pub warmup: bool,
    pub sent_ms: f64,
    #[serde(default)]
    pub ttft_ms: Option<f64>,
    #[serde(default)]
    pub rendered_tokens: Option<u64>,
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    #[serde(default)]
    pub cached_tokens: Option<u64>,
    pub reuse: u64,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub block_fps: Option<Vec<String>>,
}

/// Why a call with reusable blocks had none reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MissClass {
    /// The server had never stored the call's first block when it was sent.
    NeverStored,
    /// Stored earlier, and no reusable prefix was held when the call was
    /// sent because the attention blocks were removed.
    EvictedBeforeSend,
    /// Stored earlier; the attention blocks were held when the call was sent
    /// but no skipping group held the state at the end of any held prefix.
    StateEvictedBeforeSend,
    /// A reusable prefix was held when the call was sent, and none was held
    /// without a break from the send to the first token.
    EvictedWhileQueued,
    /// A reusable prefix was held without a break from the send to the first
    /// token.
    HeldAtFirstToken,
}

/// One line of `--calls-out`.
#[derive(Debug, Clone, Serialize)]
pub struct CallResidency {
    pub session_id: String,
    pub call: usize,
    pub sent_ms: f64,
    pub prompt_blocks: usize,
    /// `reuse / block_size`: blocks the trace shares with the previous call.
    pub reuse_blocks: u64,
    /// `cached_tokens / block_size`.
    pub cached_blocks: u64,
    /// Leading blocks every non-skipping group held when the call was sent.
    pub held_prefix_at_send: usize,
    /// Longest prefix every group could have served when the call was sent.
    pub reusable_at_send: usize,
    /// As `held_prefix_at_send`, counting only blocks held without a break
    /// from the send to the first token.
    pub held_prefix_at_first_token: usize,
    pub reusable_at_first_token: usize,
    /// Every full block of the prompt appears in a stored event with a known
    /// fingerprint.
    pub all_blocks_stored: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub miss: Option<MissClass>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GroupSummary {
    pub group: u32,
    pub skips_blocks: bool,
    pub stored_blocks: u64,
    pub removed_blocks: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResidencySummary {
    pub block_size: usize,
    pub groups: Vec<GroupSummary>,
    /// Stored blocks with no fingerprint: the event's parent block was never
    /// given one (stored before the capture started, or in a batch the
    /// capture did not receive) and no other event gave the hash one.
    pub stored_unknown: u64,
    /// Removed blocks whose hash no stored event gave a fingerprint.
    pub removed_unknown: u64,
    /// Successful non-warmup calls with block fingerprints.
    pub calls: usize,
    /// Of those, calls whose every full block appears in a stored event with
    /// a known fingerprint. A call is missing from this count when the
    /// server's tokens differ from replay's rendered prompt, or when its
    /// blocks have no fingerprint (`stored_unknown`).
    pub calls_all_blocks_stored: usize,
    /// Calls that reused more blocks than were reusable when they were sent.
    pub calls_cached_above_reusable_at_send: usize,
    /// Calls that reused fewer blocks than were reusable, held without a
    /// break, at their first token.
    pub calls_cached_below_reusable_at_first_token: usize,
    /// Calls with at least one reusable block in the trace and none reused.
    pub misses: usize,
    pub miss_classes: BTreeMap<MissClass, usize>,
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>> {
    let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut out = Vec::new();
    for (n, line) in std::io::BufReader::new(f).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(
            serde_json::from_str(&line)
                .with_context(|| format!("{}: line {}", path.display(), n + 1))?,
        );
    }
    Ok(out)
}

fn parse_fp(s: &str) -> Result<u64> {
    u64::from_str_radix(s, 16).with_context(|| format!("fingerprint {s:?}"))
}

fn group(groups: &mut BTreeMap<u32, GroupSummary>, g: u32) -> &mut GroupSummary {
    groups.entry(g).or_insert(GroupSummary {
        group: g,
        skips_blocks: false,
        stored_blocks: 0,
        removed_blocks: 0,
    })
}

/// One change to the held blocks, in event order.
enum Change {
    Add(u32, u64),
    Remove(u32, u64),
    Clear,
}

struct Query {
    t_ms: f64,
    call: usize,
    at_first_token: bool,
}

/// Blocks held at a point in the event log.
#[derive(Default)]
struct Held {
    /// Copies held, and since when at least one has been held.
    copies: HashMap<(u32, u64), (u64, f64)>,
    /// When each fingerprint was first stored in a non-skipping group.
    first_stored: HashMap<u64, f64>,
}

impl Held {
    fn add(&mut self, group: u32, fp: u64, t_ms: f64) {
        let c = self.copies.entry((group, fp)).or_insert((0, t_ms));
        if c.0 == 0 {
            c.1 = t_ms;
        }
        c.0 += 1;
    }

    /// A removal of a block with no copies held (its store was not captured)
    /// is ignored.
    fn remove(&mut self, group: u32, fp: u64) {
        if let Some(c) = self.copies.get_mut(&(group, fp)) {
            c.0 = c.0.saturating_sub(1);
        }
    }

    /// Whether a copy is held and has been held without a break since
    /// `since` (or at all, with `None`).
    fn holds(&self, group: u32, fp: u64, since: Option<f64>) -> bool {
        self.copies
            .get(&(group, fp))
            .is_some_and(|&(n, from)| n > 0 && since.is_none_or(|s| from <= s))
    }

    /// Leading blocks every group in `full` holds, and the longest of those
    /// prefixes whose last block every group in `skipping` holds.
    fn prefix(
        &self,
        fps: &[u64],
        full: &[u32],
        skipping: &[u32],
        since: Option<f64>,
    ) -> (usize, usize) {
        let held = fps
            .iter()
            .take_while(|&&fp| full.iter().all(|&g| self.holds(g, fp, since)))
            .count();
        let reusable = (1..=held)
            .rev()
            .find(|&k| skipping.iter().all(|&g| self.holds(g, fps[k - 1], since)))
            .unwrap_or(0);
        (held, reusable)
    }
}

/// Match `events` against `calls`, whose `block_fps` were computed at
/// `block_size` (`replay.kv_events.block_size`). Returns the summary and one
/// line per analysed call.
pub fn analyse(
    calls: &[CallLine],
    events: &[KvEventRecord],
    block_size: usize,
) -> Result<(ResidencySummary, Vec<CallResidency>)> {
    if block_size == 0 {
        bail!("the block size must be greater than 0");
    }
    if !events.iter().any(|e| e.kind == KvEventKind::Stored) {
        bail!("the KV event log has no stored events");
    }

    // Fingerprints by hash, from every stored event that determined one.
    let mut by_hash: HashMap<&str, u64> = HashMap::new();
    for e in events.iter().filter(|e| e.kind == KvEventKind::Stored) {
        for (h, fp) in e.hashes.iter().zip(&e.fps) {
            if let Some(fp) = fp {
                by_hash.insert(h, parse_fp(fp)?);
            }
        }
    }

    let mut groups: BTreeMap<u32, GroupSummary> = BTreeMap::new();
    let mut changes: Vec<(f64, Change)> = Vec::new();
    let (mut stored_unknown, mut removed_unknown) = (0, 0);
    for e in events {
        let g = e.group.unwrap_or(0);
        match e.kind {
            KvEventKind::Stored => {
                let gs = group(&mut groups, g);
                gs.stored_blocks += e.hashes.len() as u64;
                if e.block_size == Some(block_size)
                    && e.n_tokens.unwrap_or(0) > e.hashes.len() * block_size
                {
                    gs.skips_blocks = true;
                }
                for h in &e.hashes {
                    match by_hash.get(h.as_str()) {
                        Some(&fp) => changes.push((e.t_ms, Change::Add(g, fp))),
                        None => stored_unknown += 1,
                    }
                }
            }
            KvEventKind::Removed => {
                group(&mut groups, g).removed_blocks += e.hashes.len() as u64;
                for h in &e.hashes {
                    match by_hash.get(h.as_str()) {
                        Some(&fp) => changes.push((e.t_ms, Change::Remove(g, fp))),
                        None => removed_unknown += 1,
                    }
                }
            }
            KvEventKind::Cleared => changes.push((e.t_ms, Change::Clear)),
        }
    }
    let full: Vec<u32> = groups
        .values()
        .filter(|g| !g.skips_blocks)
        .map(|g| g.group)
        .collect();
    let skipping: Vec<u32> = groups
        .values()
        .filter(|g| g.skips_blocks)
        .map(|g| g.group)
        .collect();
    if full.is_empty() {
        bail!("every KV cache group in the event log skips blocks");
    }
    let stored_fps: BTreeSet<u64> = changes
        .iter()
        .filter_map(|(_, c)| match c {
            Change::Add(g, fp) if full.contains(g) => Some(*fp),
            _ => None,
        })
        .collect();

    let analysed: Vec<(&CallLine, Vec<u64>)> = calls
        .iter()
        .filter(|c| !c.warmup && c.error.is_none() && c.cached_tokens.is_some())
        .filter_map(|c| c.block_fps.as_ref().map(|f| (c, f)))
        .map(|(c, f)| Ok((c, f.iter().map(|s| parse_fp(s)).collect::<Result<_>>()?)))
        .collect::<Result<_>>()?;
    if analysed.is_empty() {
        bail!("the replay log has no successful non-warmup calls with block_fps");
    }
    if let Some((c, fps)) = analysed.iter().find(|(c, fps)| {
        c.rendered_tokens
            .is_some_and(|r| r as usize / block_size != fps.len())
    }) {
        bail!(
            "session {} call {}: {} block fingerprints for {} rendered tokens; the replay log was not written with block size {block_size}",
            c.session_id,
            c.call,
            fps.len(),
            c.rendered_tokens.unwrap_or(0)
        );
    }

    let mut queries: Vec<Query> = Vec::with_capacity(analysed.len() * 2);
    for (i, (c, _)) in analysed.iter().enumerate() {
        queries.push(Query {
            t_ms: c.sent_ms,
            call: i,
            at_first_token: false,
        });
        queries.push(Query {
            t_ms: c.sent_ms + c.ttft_ms.unwrap_or(0.0),
            call: i,
            at_first_token: true,
        });
    }
    queries.sort_by(|a, b| a.t_ms.total_cmp(&b.t_ms));

    // (held, reusable) at send and at first token, and whether block 0 had
    // been stored before the call was sent.
    let mut at_send = vec![(0, 0); analysed.len()];
    let mut at_first = vec![(0, 0); analysed.len()];
    let mut seen_before_send = vec![false; analysed.len()];
    let mut held = Held::default();
    let mut next = 0;
    for q in &queries {
        while next < changes.len() && changes[next].0 <= q.t_ms {
            let (t, c) = &changes[next];
            match c {
                Change::Add(g, fp) => {
                    held.add(*g, *fp, *t);
                    if full.contains(g) {
                        held.first_stored.entry(*fp).or_insert(*t);
                    }
                }
                Change::Remove(g, fp) => held.remove(*g, *fp),
                Change::Clear => held.copies.clear(),
            }
            next += 1;
        }
        let (c, fps) = &analysed[q.call];
        if q.at_first_token {
            at_first[q.call] = held.prefix(fps, &full, &skipping, Some(c.sent_ms));
        } else {
            at_send[q.call] = held.prefix(fps, &full, &skipping, None);
            seen_before_send[q.call] = fps
                .first()
                .is_some_and(|fp| held.first_stored.contains_key(fp));
        }
    }

    let mut lines = Vec::with_capacity(analysed.len());
    let mut summary = ResidencySummary {
        block_size,
        groups: groups.into_values().collect(),
        stored_unknown,
        removed_unknown,
        calls: analysed.len(),
        calls_all_blocks_stored: 0,
        calls_cached_above_reusable_at_send: 0,
        calls_cached_below_reusable_at_first_token: 0,
        misses: 0,
        miss_classes: BTreeMap::new(),
    };
    for (i, (c, fps)) in analysed.iter().enumerate() {
        let cached_blocks = c.cached_tokens.unwrap_or(0) / block_size as u64;
        let reuse_blocks = c.reuse / block_size as u64;
        let all_blocks_stored = fps.iter().all(|fp| stored_fps.contains(fp));
        let (held_send, reusable_send) = at_send[i];
        let (held_first, reusable_first) = at_first[i];
        summary.calls_all_blocks_stored += all_blocks_stored as usize;
        summary.calls_cached_above_reusable_at_send +=
            (cached_blocks > reusable_send as u64) as usize;
        summary.calls_cached_below_reusable_at_first_token +=
            (cached_blocks < reusable_first as u64) as usize;
        let miss = (reuse_blocks > 0 && cached_blocks == 0).then(|| {
            if reusable_first > 0 {
                MissClass::HeldAtFirstToken
            } else if reusable_send > 0 {
                MissClass::EvictedWhileQueued
            } else if !seen_before_send[i] {
                MissClass::NeverStored
            } else if held_send > 0 {
                MissClass::StateEvictedBeforeSend
            } else {
                MissClass::EvictedBeforeSend
            }
        });
        if let Some(m) = miss {
            summary.misses += 1;
            *summary.miss_classes.entry(m).or_default() += 1;
        }
        lines.push(CallResidency {
            session_id: c.session_id.clone(),
            call: c.call,
            sent_ms: c.sent_ms,
            prompt_blocks: fps.len(),
            reuse_blocks,
            cached_blocks,
            held_prefix_at_send: held_send,
            reusable_at_send: reusable_send,
            held_prefix_at_first_token: held_first,
            reusable_at_first_token: reusable_first,
            all_blocks_stored,
            miss,
        });
    }
    Ok((summary, lines))
}

fn console(s: &ResidencySummary) -> String {
    let mut out = format!("KV residency (block size {} tokens)\n", s.block_size);
    for g in &s.groups {
        out += &format!(
            "  group {}: {} blocks stored, {} removed{}\n",
            g.group,
            g.stored_blocks,
            g.removed_blocks,
            if g.skips_blocks { ", skips blocks" } else { "" }
        );
    }
    out += &format!(
        "  blocks with no known fingerprint: {} stored, {} removed\n\
         \x20 calls: {} analysed, {} with every block stored by the server\n\
         \x20 calls reusing more than was reusable at send: {}\n\
         \x20 calls reusing less than was held from send to first token: {}\n\
         \x20 misses (reusable blocks in the trace, none reused): {}\n",
        s.stored_unknown,
        s.removed_unknown,
        s.calls,
        s.calls_all_blocks_stored,
        s.calls_cached_above_reusable_at_send,
        s.calls_cached_below_reusable_at_first_token,
        s.misses,
    );
    for (class, n) in &s.miss_classes {
        let name = serde_json::to_value(class).unwrap();
        out += &format!("    {}: {n}\n", name.as_str().unwrap_or("?"));
    }
    out
}

/// Run `kv-residency`.
pub fn run(
    replay_log: &Path,
    events_log: &Path,
    block_size: usize,
    format: &str,
    output: Option<&Path>,
    calls_out: Option<&Path>,
) -> Result<()> {
    let calls: Vec<CallLine> = read_jsonl(replay_log)?;
    let events: Vec<KvEventRecord> = read_jsonl(events_log)?;
    let (summary, lines) = analyse(&calls, &events, block_size)?;
    if let Some(path) = calls_out {
        let f =
            std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let mut w = std::io::BufWriter::new(f);
        for l in &lines {
            serde_json::to_writer(&mut w, l)?;
            w.write_all(b"\n")?;
        }
        w.flush()?;
    }
    let text = match format {
        "json" => serde_json::to_string_pretty(&summary)? + "\n",
        "console" => console(&summary),
        other => bail!("--format must be console or json, got {other:?}"),
    };
    match output {
        Some(path) => std::fs::write(path, text)?,
        None => print!("{text}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::kv_events::{Fingerprints, block_fingerprints, fp_hex};
    use super::*;

    const BS: usize = 2;

    /// Builds event logs the way the capture writes them.
    #[derive(Default)]
    struct Log {
        fps: Fingerprints,
        events: Vec<KvEventRecord>,
    }

    impl Log {
        fn record(&mut self, t_ms: f64, kind: KvEventKind, group: u32) -> &mut KvEventRecord {
            self.events.push(KvEventRecord {
                t_ms,
                seq: self.events.len() as u64,
                ts: 0.0,
                kind,
                group: Some(group),
                hashes: Vec::new(),
                parent: None,
                block_size: None,
                n_tokens: None,
                fps: Vec::new(),
            });
            self.events.last_mut().unwrap()
        }

        fn stored(
            &mut self,
            t_ms: f64,
            group: u32,
            hashes: &[&str],
            parent: Option<&str>,
            tokens: &[u32],
        ) {
            let hashes: Vec<String> = hashes.iter().map(|h| h.to_string()).collect();
            let fps = self.fps.stored(&hashes, parent, tokens, BS);
            let r = self.record(t_ms, KvEventKind::Stored, group);
            r.fps = fps.into_iter().map(|f| f.map(fp_hex)).collect();
            r.hashes = hashes;
            r.parent = parent.map(str::to_string);
            r.block_size = Some(BS);
            r.n_tokens = Some(tokens.len());
        }

        fn removed(&mut self, t_ms: f64, group: u32, hashes: &[&str]) {
            self.record(t_ms, KvEventKind::Removed, group).hashes =
                hashes.iter().map(|h| h.to_string()).collect();
        }
    }

    fn call(n: usize, tokens: &[u32], sent_ms: f64, ttft_ms: f64, cached: u64) -> CallLine {
        CallLine {
            session_id: "s".to_string(),
            call: n,
            warmup: false,
            sent_ms,
            ttft_ms: Some(ttft_ms),
            rendered_tokens: Some(tokens.len() as u64),
            prompt_tokens: Some(tokens.len() as u64),
            cached_tokens: Some(cached),
            reuse: 4,
            error: None,
            block_fps: Some(
                block_fingerprints(tokens, BS)
                    .into_iter()
                    .map(fp_hex)
                    .collect(),
            ),
        }
    }

    const PROMPT: [u32; 6] = [1, 2, 3, 4, 5, 6];

    /// The prompt's three blocks stored at t=0 in attention group 0, and the
    /// state at block 2 (index 1) in skipping group 1.
    fn stored_prompt() -> Log {
        let mut log = Log::default();
        log.stored(0.0, 0, &["a", "b", "c"], None, &PROMPT);
        log.stored(0.0, 1, &["b"], None, &PROMPT[..4]);
        log
    }

    fn classes(calls: &[CallLine], log: &Log) -> (ResidencySummary, Vec<CallResidency>) {
        analyse(calls, &log.events, BS).unwrap()
    }

    #[test]
    fn a_held_prefix_is_reusable_up_to_the_skipping_groups_state() {
        let log = stored_prompt();
        let (s, lines) = classes(&[call(1, &PROMPT, 10.0, 5.0, 4)], &log);
        assert_eq!(s.block_size, BS);
        assert_eq!(
            s.groups
                .iter()
                .map(|g| (g.group, g.skips_blocks))
                .collect::<Vec<_>>(),
            vec![(0, false), (1, true)]
        );
        let l = &lines[0];
        // Group 0 holds all three blocks; group 1 has state only at block 2.
        assert_eq!((l.held_prefix_at_send, l.reusable_at_send), (3, 2));
        assert_eq!(l.cached_blocks, 2);
        assert!(l.all_blocks_stored);
        assert_eq!(l.miss, None);
        assert_eq!(s.calls_cached_above_reusable_at_send, 0);
        assert_eq!(s.calls_cached_below_reusable_at_first_token, 0);
    }

    #[test]
    fn misses_are_classified_by_when_the_prefix_went() {
        let mut log = stored_prompt();
        // Everything removed at t=50: a call sent at 100 finds nothing.
        log.removed(50.0, 0, &["a", "b", "c"]);
        log.removed(50.0, 1, &["b"]);
        // Stored again at 200, the state removed at 250.
        log.stored(200.0, 0, &["a", "b", "c"], None, &PROMPT);
        log.stored(200.0, 1, &["b"], None, &PROMPT[..4]);
        log.removed(250.0, 1, &["b"]);
        // State stored again at 300, attention removed at 320.
        log.stored(300.0, 1, &["b"], None, &PROMPT[..4]);
        log.removed(320.0, 0, &["a"]);
        let calls = [
            call(1, &PROMPT, 100.0, 5.0, 0),
            call(2, &PROMPT, 260.0, 5.0, 0),
            call(3, &PROMPT, 310.0, 20.0, 0),
            call(4, &PROMPT, 305.0, 5.0, 0),
            call(5, &[9, 9, 9, 9], 400.0, 5.0, 0),
        ];
        let (s, lines) = classes(&calls, &log);
        let got: Vec<_> = lines.iter().map(|l| l.miss).collect();
        assert_eq!(
            got,
            vec![
                Some(MissClass::EvictedBeforeSend),
                Some(MissClass::StateEvictedBeforeSend),
                Some(MissClass::EvictedWhileQueued),
                Some(MissClass::HeldAtFirstToken),
                Some(MissClass::NeverStored),
            ]
        );
        assert_eq!(s.misses, 5);
        assert_eq!(s.calls_cached_below_reusable_at_first_token, 1);
        assert_eq!(s.calls_all_blocks_stored, 4);
    }

    #[test]
    fn a_hash_is_held_until_every_copy_is_removed() {
        let mut log = stored_prompt();
        log.stored(1.0, 0, &["a", "b", "c"], None, &PROMPT);
        log.removed(2.0, 0, &["a"]);
        let (s, lines) = classes(&[call(1, &PROMPT, 10.0, 1.0, 4)], &log);
        assert_eq!(lines[0].held_prefix_at_send, 3);
        // A removal of a hash no stored event named is counted, not applied.
        let mut log = stored_prompt();
        log.removed(2.0, 0, &["zz"]);
        let (s2, _) = classes(&[call(1, &PROMPT, 10.0, 1.0, 4)], &log);
        assert_eq!((s.removed_unknown, s2.removed_unknown), (0, 1));
    }

    #[test]
    fn reuse_above_what_was_held_at_send_is_counted() {
        let log = stored_prompt();
        // The server reported 3 blocks; only 2 were reusable.
        let (s, _) = classes(&[call(1, &PROMPT, 10.0, 1.0, 6)], &log);
        assert_eq!(s.calls_cached_above_reusable_at_send, 1);
    }

    #[test]
    fn warmup_failed_and_unfingerprinted_calls_are_skipped() {
        let log = stored_prompt();
        let mut w = call(1, &PROMPT, 10.0, 1.0, 4);
        w.warmup = true;
        let mut f = call(2, &PROMPT, 10.0, 1.0, 4);
        f.error = Some("x".to_string());
        let mut n = call(3, &PROMPT, 10.0, 1.0, 4);
        n.block_fps = None;
        assert!(analyse(&[w.clone(), f.clone(), n.clone()], &log.events, BS).is_err());
        let (s, _) = classes(&[w, f, n, call(4, &PROMPT, 10.0, 1.0, 4)], &log);
        assert_eq!(s.calls, 1);
    }

    /// vLLM stores a call's prompt blocks when it schedules the call, before
    /// its first token, so a prefix evicted before the send reads as held
    /// again by then. Only a block held without a break since the send counts
    /// at the first token.
    #[test]
    fn the_calls_own_store_does_not_count_as_held_at_first_token() {
        let mut log = stored_prompt();
        log.removed(50.0, 0, &["a", "b", "c"]);
        log.removed(50.0, 1, &["b"]);
        log.stored(101.0, 0, &["a", "b", "c"], None, &PROMPT);
        log.stored(101.0, 1, &["b"], None, &PROMPT[..4]);
        let mut queued = stored_prompt();
        queued.removed(105.0, 0, &["a", "b", "c"]);
        queued.stored(110.0, 0, &["a", "b", "c"], None, &PROMPT);
        let (_, before) = classes(&[call(1, &PROMPT, 100.0, 30.0, 0)], &log);
        let (_, during) = classes(&[call(1, &PROMPT, 100.0, 30.0, 0)], &queued);
        assert_eq!(before[0].miss, Some(MissClass::EvictedBeforeSend));
        assert_eq!(before[0].reusable_at_first_token, 0);
        assert_eq!(during[0].miss, Some(MissClass::EvictedWhileQueued));
    }

    /// A removal whose store the capture did not see must not make a later
    /// store read as absent.
    #[test]
    fn a_removal_without_a_store_does_not_go_below_zero() {
        let mut log = Log::default();
        log.stored(0.0, 0, &["a", "b", "c"], None, &PROMPT);
        let fp_source = log.events.remove(0);
        log.removed(10.0, 0, &["a", "b", "c"]);
        log.stored(20.0, 0, &["a", "b", "c"], None, &PROMPT);
        log.events.push(KvEventRecord {
            t_ms: 500.0,
            ..fp_source
        });
        let (_, lines) = classes(&[call(1, &PROMPT, 100.0, 1.0, 6)], &log);
        assert_eq!(lines[0].held_prefix_at_send, 3);
    }

    #[test]
    fn all_blocks_cleared_drops_every_held_block() {
        let mut log = stored_prompt();
        log.record(50.0, KvEventKind::Cleared, 0).group = None;
        let (_, lines) = classes(&[call(1, &PROMPT, 100.0, 1.0, 0)], &log);
        assert_eq!(lines[0].held_prefix_at_send, 0);
        assert_eq!(lines[0].miss, Some(MissClass::EvictedBeforeSend));
    }

    #[test]
    fn fingerprints_at_another_block_size_are_refused() {
        let log = stored_prompt();
        let err = analyse(&[call(1, &PROMPT, 10.0, 1.0, 4)], &log.events, 3).unwrap_err();
        assert!(err.to_string().contains("block size 3"), "{err}");
    }

    #[test]
    fn stored_blocks_with_no_fingerprint_are_counted() {
        let mut log = stored_prompt();
        log.stored(5.0, 0, &["x"], Some("never-seen"), &[7, 8]);
        let (s, _) = classes(&[call(1, &PROMPT, 10.0, 1.0, 4)], &log);
        assert_eq!(s.stored_unknown, 1);
    }
}
