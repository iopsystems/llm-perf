//! Recording vLLM's KV cache events (`--kv-events-config`) during a replay.
//!
//! vLLM publishes each batch of events on a ZMQ PUB socket as three frames:
//! the topic, an 8-byte big-endian sequence number, and a msgpack
//! `[ts, events]` array in which each event is a map tagged by its `type`
//! (`BlockStored`, `BlockRemoved` or `AllBlocksCleared`). Replay writes one
//! JSON line per event, with the time it was received on the same clock as
//! the per-call log.
//!
//! Blocks are identified by vLLM's block hash, which is the same in every KV
//! cache group for the same prefix. Replay adds a fingerprint of the prefix a
//! block covers, computed from the block's token ids with
//! [`block_fingerprints`], so the blocks can be matched to the calls whose
//! prompts contain them.

use crate::config::KvEventsConfig;
use anyhow::{Context, Result, bail};
use log::warn;
use rmpv::Value;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::time::Duration;
use tokio::sync::{mpsc, watch};
use tokio::time::Instant;
use zeromq::{Socket, SocketRecv};

/// After the run stops, the capture keeps receiving until no batch has
/// arrived for this long.
const DRAIN_QUIET: Duration = Duration::from_secs(2);
/// Longest the capture keeps receiving after the run stops.
const DRAIN_MAX: Duration = Duration::from_secs(10);

/// Fingerprint chained from the fingerprint of the preceding prefix and the
/// token ids of one block (64-bit FNV-1a over the little-endian bytes).
fn chain(parent: u64, tokens: &[u32]) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in parent
        .to_le_bytes()
        .into_iter()
        .chain(tokens.iter().flat_map(|t| t.to_le_bytes()))
    {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// Fingerprint of the prefix before the first block.
const ROOT: u64 = 0;

/// Fingerprint of each full `block_size` block of `tokens`. Block `i`'s
/// fingerprint covers tokens `[0, (i + 1) * block_size)`, so two prompts have
/// the same fingerprint at block `i` only if they agree on all of those
/// tokens.
pub fn block_fingerprints(tokens: &[u32], block_size: usize) -> Vec<u64> {
    let mut fp = ROOT;
    tokens
        .chunks_exact(block_size)
        .map(|b| {
            fp = chain(fp, b);
            fp
        })
        .collect()
}

/// A fingerprint as written to the logs: 16 hex digits. JSON numbers above
/// 2^53 lose precision in common readers.
pub fn fp_hex(fp: u64) -> String {
    format!("{fp:016x}")
}

/// One event, as decoded from vLLM's msgpack.
#[derive(Debug, Clone, PartialEq)]
pub enum KvEvent {
    Stored {
        hashes: Vec<String>,
        parent: Option<String>,
        token_ids: Vec<u32>,
        block_size: usize,
        group: u32,
    },
    Removed {
        hashes: Vec<String>,
        group: u32,
    },
    Cleared,
}

/// One published batch.
#[derive(Debug, Clone, PartialEq)]
pub struct KvBatch {
    /// The server's `time.time()` when it built the batch.
    pub ts: f64,
    pub events: Vec<KvEvent>,
}

/// vLLM sends block hashes as bytes (SHA-256, the default) or as integers
/// (`VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES`).
fn hash_text(v: &Value) -> Result<String> {
    match v {
        Value::Binary(b) => Ok(b.iter().map(|x| format!("{x:02x}")).collect()),
        Value::Integer(i) => Ok(i.to_string()),
        other => bail!("block hash is neither bytes nor an integer: {other}"),
    }
}

fn field<'a>(map: &'a [(Value, Value)], name: &str) -> Option<&'a Value> {
    map.iter()
        .find(|(k, _)| k.as_str() == Some(name))
        .map(|(_, v)| v)
        .filter(|v| !v.is_nil())
}

fn hashes(map: &[(Value, Value)]) -> Result<Vec<String>> {
    field(map, "block_hashes")
        .and_then(Value::as_array)
        .context("event has no block_hashes")?
        .iter()
        .map(hash_text)
        .collect()
}

/// `group_idx` is omitted when it has the default, `None`; vLLM's single
/// group is then group 0.
fn group(map: &[(Value, Value)]) -> u32 {
    field(map, "group_idx").and_then(Value::as_u64).unwrap_or(0) as u32
}

fn decode_event(v: &Value) -> Result<Option<KvEvent>> {
    let map = v.as_map().context("event is not a map")?;
    let kind = field(map, "type")
        .and_then(Value::as_str)
        .context("event has no type")?;
    Ok(Some(match kind {
        "BlockStored" => KvEvent::Stored {
            hashes: hashes(map)?,
            parent: field(map, "parent_block_hash").map(hash_text).transpose()?,
            token_ids: field(map, "token_ids")
                .and_then(Value::as_array)
                .context("BlockStored has no token_ids")?
                .iter()
                .map(|t| t.as_u64().map(|t| t as u32).context("token id"))
                .collect::<Result<_>>()?,
            block_size: field(map, "block_size")
                .and_then(Value::as_u64)
                .context("BlockStored has no block_size")? as usize,
            group: group(map),
        },
        "BlockRemoved" => KvEvent::Removed {
            hashes: hashes(map)?,
            group: group(map),
        },
        "AllBlocksCleared" => KvEvent::Cleared,
        _ => return Ok(None),
    }))
}

/// Decode one batch payload. Events of a type replay does not know are
/// skipped.
pub fn decode_batch(payload: &[u8]) -> Result<KvBatch> {
    let v = rmpv::decode::read_value(&mut &payload[..]).context("decoding msgpack")?;
    let a = v.as_array().context("batch is not an array")?;
    let ts = a
        .first()
        .and_then(Value::as_f64)
        .context("batch has no timestamp")?;
    let events = a
        .get(1)
        .and_then(Value::as_array)
        .context("batch has no event list")?
        .iter()
        .filter_map(|e| decode_event(e).transpose())
        .collect::<Result<_>>()?;
    Ok(KvBatch { ts, events })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KvEventKind {
    Stored,
    Removed,
    Cleared,
}

/// One line of the KV event log.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvEventRecord {
    /// When replay received the batch, in milliseconds from the run start.
    pub t_ms: f64,
    /// The batch's sequence number.
    pub seq: u64,
    /// The server's timestamp for the batch, in seconds since the epoch.
    pub ts: f64,
    pub kind: KvEventKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hashes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_size: Option<usize>,
    /// Token ids the stored event covers. More than `hashes.len() *
    /// block_size` when the group skips blocks (Mamba groups in vLLM's
    /// "align" prefix caching).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_tokens: Option<usize>,
    /// Fingerprint of each stored block (hex), where it is known: the event
    /// covers each block's tokens and its parent's fingerprint is known, or
    /// another event already gave the same hash a fingerprint.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fps: Vec<Option<String>>,
}

/// Fingerprints by block hash, learned from stored events.
#[derive(Debug, Default)]
pub struct Fingerprints {
    by_hash: HashMap<String, u64>,
}

impl Fingerprints {
    /// The fingerprint of each block of a stored event, learning those the
    /// event's tokens determine.
    pub fn stored(
        &mut self,
        hashes: &[String],
        parent: Option<&str>,
        token_ids: &[u32],
        block_size: usize,
    ) -> Vec<Option<u64>> {
        let contiguous = block_size > 0 && token_ids.len() == hashes.len() * block_size;
        let parent_fp = match parent {
            None => Some(ROOT),
            Some(p) => self.by_hash.get(p).copied(),
        };
        if let (true, Some(mut fp)) = (contiguous, parent_fp) {
            return hashes
                .iter()
                .zip(token_ids.chunks_exact(block_size))
                .map(|(h, tokens)| {
                    fp = chain(fp, tokens);
                    self.by_hash.insert(h.clone(), fp);
                    Some(fp)
                })
                .collect();
        }
        hashes
            .iter()
            .map(|h| self.by_hash.get(h).copied())
            .collect()
    }

    pub fn get(&self, hash: &str) -> Option<u64> {
        self.by_hash.get(hash).copied()
    }
}

/// Totals for the run summary.
#[derive(Debug, Default, Clone, Serialize)]
pub struct KvEventStats {
    pub batches: u64,
    pub stored_events: u64,
    pub removed_events: u64,
    pub cleared_events: u64,
    /// Sequence numbers skipped between received batches: batches the
    /// publisher sent that replay did not receive.
    pub batches_missed: u64,
    /// Stored events whose `block_size` differs from
    /// `replay.kv_events.block_size`.
    pub block_size_mismatch: u64,
}

fn records(
    t_ms: f64,
    seq: u64,
    batch: KvBatch,
    fps: &mut Fingerprints,
    block_size: usize,
    stats: &mut KvEventStats,
) -> Vec<KvEventRecord> {
    batch
        .events
        .into_iter()
        .map(|e| {
            let mut r = KvEventRecord {
                t_ms,
                seq,
                ts: batch.ts,
                kind: KvEventKind::Cleared,
                group: None,
                hashes: Vec::new(),
                parent: None,
                block_size: None,
                n_tokens: None,
                fps: Vec::new(),
            };
            match e {
                KvEvent::Stored {
                    hashes,
                    parent,
                    token_ids,
                    block_size: bs,
                    group,
                } => {
                    stats.stored_events += 1;
                    stats.block_size_mismatch += (bs != block_size) as u64;
                    r.kind = KvEventKind::Stored;
                    r.fps = fps
                        .stored(&hashes, parent.as_deref(), &token_ids, bs)
                        .into_iter()
                        .map(|f| f.map(fp_hex))
                        .collect();
                    r.group = Some(group);
                    r.hashes = hashes;
                    r.parent = parent;
                    r.block_size = Some(bs);
                    r.n_tokens = Some(token_ids.len());
                }
                KvEvent::Removed { hashes, group } => {
                    stats.removed_events += 1;
                    r.kind = KvEventKind::Removed;
                    r.group = Some(group);
                    r.hashes = hashes;
                }
                KvEvent::Cleared => stats.cleared_events += 1,
            }
            r
        })
        .collect()
}

/// A running capture. [`KvCapture::finish`] stops it and returns its totals.
pub struct KvCapture {
    stop: watch::Sender<bool>,
    task: tokio::task::JoinHandle<Result<KvEventStats>>,
}

impl KvCapture {
    /// Subscribe to the server's events and start writing them to
    /// `config.log`. `start` is the run's time origin.
    pub async fn start(config: &KvEventsConfig, start: Instant) -> Result<Self> {
        let mut socket = zeromq::SubSocket::new();
        socket
            .subscribe(&config.topic)
            .await
            .context("subscribing to KV events")?;
        socket
            .connect(&config.endpoint)
            .await
            .with_context(|| format!("connecting to KV events at {}", config.endpoint))?;
        let file = std::fs::File::create(&config.log)
            .with_context(|| format!("creating {}", config.log.display()))?;
        let (tx, mut rx) = mpsc::unbounded_channel::<Vec<KvEventRecord>>();
        let writer = tokio::task::spawn_blocking(move || -> Result<()> {
            let mut w = std::io::BufWriter::new(file);
            while let Some(batch) = rx.blocking_recv() {
                for r in batch {
                    serde_json::to_writer(&mut w, &r)?;
                    w.write_all(b"\n")?;
                }
            }
            w.flush()?;
            Ok(())
        });
        let (stop, stop_rx) = watch::channel(false);
        let block_size = config.block_size;
        let task = tokio::spawn(async move {
            let stats = receive(socket, start, block_size, stop_rx, tx).await;
            writer.await??;
            stats
        });
        Ok(Self { stop, task })
    }

    /// Stop receiving once the server has gone quiet, and wait for the log to
    /// be written.
    pub async fn finish(self) -> Result<KvEventStats> {
        self.stop.send_replace(true);
        self.task.await?
    }
}

async fn receive(
    mut socket: zeromq::SubSocket,
    start: Instant,
    block_size: usize,
    mut stop: watch::Receiver<bool>,
    tx: mpsc::UnboundedSender<Vec<KvEventRecord>>,
) -> Result<KvEventStats> {
    let mut stats = KvEventStats::default();
    let mut fps = Fingerprints::default();
    let mut last_seq: Option<u64> = None;
    let mut drain_until: Option<Instant> = None;
    loop {
        let quiet = drain_until.map(|d| d.min(Instant::now() + DRAIN_QUIET));
        let msg = tokio::select! {
            m = socket.recv() => m,
            _ = stop.wait_for(|s| *s), if drain_until.is_none() => {
                drain_until = Some(Instant::now() + DRAIN_MAX);
                continue;
            }
            _ = async { tokio::time::sleep_until(quiet.unwrap()).await }, if quiet.is_some() => break,
        };
        let msg = match msg {
            Ok(m) => m,
            Err(e) => {
                warn!("KV events: receiving: {e}");
                break;
            }
        };
        let t_ms = start.elapsed().as_secs_f64() * 1000.0;
        let (Some(seq), Some(payload)) = (msg.get(1), msg.get(2)) else {
            warn!("KV events: message with {} frames, expected 3", msg.len());
            continue;
        };
        let Ok(seq) = <[u8; 8]>::try_from(&seq[..]).map(u64::from_be_bytes) else {
            warn!("KV events: sequence frame of {} bytes", seq.len());
            continue;
        };
        if let Some(last) = last_seq
            && seq > last + 1
        {
            stats.batches_missed += seq - last - 1;
        }
        last_seq = Some(seq);
        let batch = match decode_batch(payload) {
            Ok(b) => b,
            Err(e) => {
                warn!("KV events: batch {seq}: {e:#}");
                continue;
            }
        };
        stats.batches += 1;
        let _ = tx.send(records(t_ms, seq, batch, &mut fps, block_size, &mut stats));
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v_map(fields: Vec<(&str, Value)>) -> Value {
        Value::Map(
            fields
                .into_iter()
                .map(|(k, v)| (Value::from(k), v))
                .collect(),
        )
    }

    fn encode(v: &Value) -> Vec<u8> {
        let mut out = Vec::new();
        rmpv::encode::write_value(&mut out, v).unwrap();
        out
    }

    /// A batch shaped as msgspec encodes vLLM's `KVEventBatch`: the batch is
    /// an array, each event a tagged map with defaults omitted.
    #[test]
    fn decodes_a_vllm_batch() {
        let stored = v_map(vec![
            ("type", Value::from("BlockStored")),
            (
                "block_hashes",
                Value::Array(vec![
                    Value::Binary(vec![0xab, 0x01]),
                    Value::Binary(vec![0xcd]),
                ]),
            ),
            ("parent_block_hash", Value::Nil),
            (
                "token_ids",
                Value::Array((1..=4).map(Value::from).collect()),
            ),
            ("block_size", Value::from(2)),
            ("lora_id", Value::Nil),
            ("medium", Value::from("GPU")),
            ("lora_name", Value::Nil),
            ("group_idx", Value::from(1)),
        ]);
        let removed = v_map(vec![
            ("type", Value::from("BlockRemoved")),
            ("block_hashes", Value::Array(vec![Value::from(7u64)])),
            ("medium", Value::from("GPU")),
        ]);
        let cleared = v_map(vec![("type", Value::from("AllBlocksCleared"))]);
        let other = v_map(vec![("type", Value::from("SomethingNew"))]);
        let batch = Value::Array(vec![
            Value::from(1.5),
            Value::Array(vec![stored, removed, cleared, other]),
        ]);
        let b = decode_batch(&encode(&batch)).unwrap();
        assert_eq!(b.ts, 1.5);
        assert_eq!(
            b.events,
            vec![
                KvEvent::Stored {
                    hashes: vec!["ab01".to_string(), "cd".to_string()],
                    parent: None,
                    token_ids: vec![1, 2, 3, 4],
                    block_size: 2,
                    group: 1,
                },
                KvEvent::Removed {
                    hashes: vec!["7".to_string()],
                    group: 0,
                },
                KvEvent::Cleared,
            ]
        );
        assert!(decode_batch(&encode(&Value::from(3))).is_err());
    }

    #[test]
    fn fingerprints_chain_over_the_whole_prefix() {
        let a = block_fingerprints(&[1, 2, 3, 4, 5], 2);
        assert_eq!(a.len(), 2, "the partial last block has no fingerprint");
        let b = block_fingerprints(&[1, 2, 9, 4], 2);
        assert_eq!(a[0], b[0]);
        assert_ne!(a[1], b[1]);
        // The same block after a different prefix has a different fingerprint.
        let c = block_fingerprints(&[9, 9, 3, 4], 2);
        assert_ne!(a[1], c[1]);
    }

    #[test]
    fn stored_events_learn_fingerprints_that_match_the_prompt() {
        let prompt = [10, 11, 12, 13, 14, 15];
        let want: Vec<Option<u64>> = block_fingerprints(&prompt, 2)
            .into_iter()
            .map(Some)
            .collect();
        let mut fps = Fingerprints::default();
        let h = |s: &str| s.to_string();
        // The first two blocks, then the third with its parent.
        assert_eq!(
            fps.stored(&[h("a"), h("b")], None, &prompt[..4], 2),
            want[..2]
        );
        assert_eq!(fps.stored(&[h("c")], Some("b"), &prompt[4..], 2), want[2..]);
        // A parent never seen leaves the fingerprint unknown.
        assert_eq!(
            fps.stored(&[h("x")], Some("zz"), &prompt[4..], 2),
            vec![None]
        );
        // A group that skips blocks covers more tokens than its hashes; its
        // blocks take the fingerprint another group gave the same hash.
        assert_eq!(fps.stored(&[h("c")], None, &prompt, 2), vec![want[2]]);
        assert_eq!(fps.get("c"), want[2]);
    }

    /// A batch published on a ZMQ PUB socket reaches the log through
    /// `KvCapture`, with skipped sequence numbers counted.
    #[tokio::test]
    async fn capture_writes_published_batches() {
        use zeromq::{SocketSend, ZmqMessage};
        let mut publisher = zeromq::PubSocket::new();
        let endpoint = publisher.bind("tcp://127.0.0.1:0").await.unwrap();
        let dir = tempfile::tempdir().unwrap();
        let config = KvEventsConfig {
            endpoint: endpoint.to_string(),
            topic: String::new(),
            block_size: 2,
            log: dir.path().join("kv.jsonl"),
        };
        let capture = KvCapture::start(&config, Instant::now()).await.unwrap();
        let stored = v_map(vec![
            ("type", Value::from("BlockStored")),
            ("block_hashes", Value::Array(vec![Value::Binary(vec![1])])),
            ("parent_block_hash", Value::Nil),
            (
                "token_ids",
                Value::Array(vec![Value::from(5), Value::from(6)]),
            ),
            ("block_size", Value::from(2)),
        ]);
        let payload = encode(&Value::Array(vec![
            Value::from(1.0),
            Value::Array(vec![stored]),
        ]));
        // The subscription reaches the publisher asynchronously; publish
        // until the capture has had time to receive.
        for seq in [0u64, 0, 0, 3] {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let mut m = ZmqMessage::from(Vec::<u8>::new());
            m.push_back(seq.to_be_bytes().to_vec().into());
            m.push_back(payload.clone().into());
            publisher.send(m).await.unwrap();
        }
        let stats = capture.finish().await.unwrap();
        assert!(stats.batches >= 2, "{stats:?}");
        assert_eq!(stats.batches_missed, 2);
        let text = std::fs::read_to_string(&config.log).unwrap();
        let first: KvEventRecord = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(first.kind, KvEventKind::Stored);
        assert_eq!(first.hashes, vec!["01".to_string()]);
        assert_eq!(
            first.fps,
            vec![Some(fp_hex(block_fingerprints(&[5, 6], 2)[0]))]
        );
    }
}
