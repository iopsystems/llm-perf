//! Running a replay: select sessions, schedule them on the scaled source
//! timeline, send each call, and record what the server cached.

use super::filler::{FillerPool, call_rng};
use super::kv_events::{KvCapture, KvEventStats, block_fingerprints, fp_hex};
use super::prompt::SessionPrompt;
use super::render::ServerRenderer;
use super::sample::selected;
use super::schedule::{session_slots, session_start};
use super::server::probe;
use crate::benchmark::{classify_error, decode_tpot, effective_output_tokens};
use crate::client::{ClientConfig, Message, OpenAIClient};
use crate::config::{Config, OutputFormat, ReplayConfig, ReplayServer};
use crate::metrics::{
    ErrorType, InflightGuard, Metrics, Phase, REPLAY_CACHED_PERMILLE, REPLAY_REUSE,
    REPLAY_REUSE_CACHED, REPLAY_REUSE_PROMPT, REPLAY_SESSION_COMPLETED, REPLAY_SESSION_FAILED,
    REPLAY_SESSION_STARTED, REPLAY_SESSION_TRUNCATED, REPLAY_SESSIONS, RequestStatus,
};
use crate::trace::TraceSession;
use anyhow::{Context, Result, bail};
use log::{debug, info, warn};
use serde::Serialize;
use std::io::{BufRead, Write};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, watch};
use tokio::time::Instant;

/// The prompt-size check runs each time this many more calls have reported
/// `prompt_tokens`.
const SIZE_CHECK_EVERY: usize = 100;
/// Largest allowed median of |prompt_tokens - target| / target.
const SIZE_CHECK_MAX_ERROR: f64 = 0.01;

/// One line of the per-call log. `scheduled_ms` and `sent_ms` are milliseconds
/// from the run start; `lag_ms`, `ttft_ms` and `e2e_ms` are durations in
/// milliseconds.
#[derive(Debug, Clone, Serialize)]
pub struct CallRecord {
    pub session_id: String,
    pub call: usize,
    pub model: Option<String>,
    pub scheduled_ms: f64,
    pub sent_ms: f64,
    /// `sent_ms - scheduled_ms`, or 0 if the call was sent on time.
    pub lag_ms: f64,
    pub gap_capped: bool,
    pub warmup: bool,
    pub target_prompt: u64,
    /// Length of the prompt as replay rendered it with the server's
    /// tokenize endpoint; `None` when building the prompt failed.
    pub rendered_tokens: Option<u64>,
    pub prompt_tokens: Option<u64>,
    pub overshoot: bool,
    /// From the trace.
    pub reuse: u64,
    pub reuse_inferred: bool,
    pub shortfall: u64,
    pub cached_tokens: Option<u64>,
    pub max_tokens: u32,
    pub completion_tokens: Option<u64>,
    pub finish_reason: Option<String>,
    pub ttft_ms: Option<f64>,
    pub e2e_ms: Option<f64>,
    pub error: Option<String>,
    /// With `[replay.kv_events]`: the fingerprint of each full block of the
    /// rendered prompt, matching the `fps` of the KV event log.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_fps: Option<Vec<String>>,
}

/// Totals printed or written at the end of a run. `calls_sent`,
/// `calls_failed`, `calls_render_mismatch`, `render_mismatch_max_tokens` and
/// `prompt_size_error_median` include warmup calls; every other total and
/// percentile covers non-warmup calls that succeeded.
#[derive(Debug, Default, Serialize)]
pub struct ReplaySummary {
    pub server: ReplayServer,
    /// llama-server's `build_info`, or vLLM's version.
    pub server_build: Option<String>,
    /// Longest prompt plus completion one request may have: llama-server's
    /// per-slot context, or vLLM's `max_model_len`.
    pub server_n_ctx: u64,
    pub sessions_in_trace: usize,
    pub sessions_selected: usize,
    pub sessions_completed: usize,
    pub sessions_failed: usize,
    pub sessions_truncated: usize,
    pub calls_sent: usize,
    pub calls_failed: usize,
    pub calls_overshoot: usize,
    pub calls_shortfall: usize,
    pub calls_eos_not_ignored: usize,
    /// Calls whose reported `prompt_tokens` differ from `rendered_tokens`:
    /// the server built a different prompt from the one replay rendered.
    pub calls_render_mismatch: usize,
    /// Largest `|prompt_tokens - rendered_tokens|` over those calls; 0 when
    /// there are none.
    pub render_mismatch_max_tokens: u64,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    /// `prompt_tokens - cached_tokens`.
    pub prefill_tokens: u64,
    /// Calls the cached permille percentiles cover.
    pub calls_measured: usize,
    /// Percentiles of 1000 * cached_tokens / prompt_tokens over calls whose
    /// traced reuse is > 0. A call's value is about `1000 * reuse /
    /// prompt_tokens` at most, and this bound differs between calls.
    pub cached_permille_p50: Option<f64>,
    pub cached_permille_p10: Option<f64>,
    pub lag_ms_p50: Option<f64>,
    pub lag_ms_p99: Option<f64>,
    pub lag_ms_max: Option<f64>,
    pub ttft_ms_p50: Option<f64>,
    pub ttft_ms_p99: Option<f64>,
    pub prompt_size_error_median: Option<f64>,
    /// With `[replay.kv_events]`: what the event capture received.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_events: Option<KvEventStats>,
}

#[derive(Default)]
struct Stats {
    calls_sent: usize,
    calls_failed: usize,
    calls_overshoot: usize,
    calls_shortfall: usize,
    calls_eos_not_ignored: usize,
    calls_render_mismatch: usize,
    render_mismatch_max_tokens: u64,
    prompt_tokens: u64,
    cached_tokens: u64,
    prefill_tokens: u64,
    permille: Vec<f64>,
    lag_ms: Vec<f64>,
    ttft_ms: Vec<f64>,
    size_errors: Vec<f64>,
}

impl Stats {
    /// Count a successful call whose reported prompt length differs from the
    /// rendered one. Returns both lengths for the first such call, so the
    /// caller can log it.
    fn add_render(&mut self, r: &CallRecord) -> Option<(u64, u64)> {
        let (Some(reported), Some(rendered)) = (r.prompt_tokens, r.rendered_tokens) else {
            return None;
        };
        if reported == rendered {
            return None;
        }
        self.calls_render_mismatch += 1;
        self.render_mismatch_max_tokens = self
            .render_mismatch_max_tokens
            .max(reported.abs_diff(rendered));
        (self.calls_render_mismatch == 1).then_some((reported, rendered))
    }

    /// Add a successful non-warmup call to the measured totals.
    fn add_measured(&mut self, r: &CallRecord) {
        self.calls_overshoot += r.overshoot as usize;
        self.calls_shortfall += (r.shortfall > 0) as usize;
        self.calls_eos_not_ignored +=
            r.completion_tokens.is_some_and(|c| c < r.max_tokens as u64) as usize;
        let cached = r.cached_tokens.unwrap_or(0);
        let prompt = r.prompt_tokens.unwrap_or(0);
        self.prompt_tokens += prompt;
        self.cached_tokens += cached;
        self.prefill_tokens += prompt.saturating_sub(cached);
        if r.reuse > 0 && prompt > 0 {
            self.permille.push(1000.0 * cached as f64 / prompt as f64);
        }
        self.lag_ms.push(r.lag_ms);
        if let Some(t) = r.ttft_ms {
            self.ttft_ms.push(t);
        }
    }
}

enum Outcome {
    Completed,
    Failed,
    Truncated,
}

struct Shared {
    client: Arc<OpenAIClient>,
    renderer: ServerRenderer,
    pool: FillerPool,
    system: Option<Message>,
    replay: ReplayConfig,
    start: Instant,
    deadline: Option<Instant>,
    warmup_until: Option<Instant>,
    log: Option<mpsc::UnboundedSender<CallRecord>>,
    stats: Mutex<Stats>,
    /// Set to true to stop the run; sessions waiting for a start time or a
    /// gap wake on it.
    stop: watch::Sender<bool>,
    abort_reason: Mutex<Option<String>>,
}

impl Shared {
    fn ms(&self, t: Instant) -> f64 {
        t.saturating_duration_since(self.start).as_secs_f64() * 1000.0
    }

    fn fail_run(&self, reason: String) {
        let mut r = self.abort_reason.lock().unwrap();
        if r.is_none() {
            *r = Some(reason);
        }
        self.stop.send_replace(true);
    }

    fn past_deadline(&self) -> bool {
        self.deadline.is_some_and(|d| Instant::now() >= d)
    }

    fn emit(&self, record: CallRecord) {
        if let Some(tx) = &self.log {
            let _ = tx.send(record);
        }
    }
}

/// Sleep until `t`, waking early at `deadline` or when `stop` is set. Returns
/// whether the caller should go on: false once the run is stopped or past its
/// deadline.
async fn wait_until(deadline: Option<Instant>, stop: &watch::Sender<bool>, t: Instant) -> bool {
    let wake = deadline.map_or(t, |d| t.min(d));
    let mut rx = stop.subscribe();
    tokio::select! {
        _ = tokio::time::sleep_until(wake) => {}
        _ = rx.wait_for(|stopped| *stopped) => {}
    }
    !(*stop.borrow() || deadline.is_some_and(|d| Instant::now() >= d))
}

fn percentile(v: &mut [f64], p: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let idx = ((p / 100.0) * (v.len() - 1) as f64).round() as usize;
    Some(v[idx])
}

/// Read the trace and keep the sessions `replay` selects.
fn load_sessions(replay: &ReplayConfig) -> Result<(usize, Vec<TraceSession>)> {
    let f = std::fs::File::open(&replay.trace)
        .with_context(|| format!("opening {}", replay.trace.display()))?;
    let mut total = 0;
    let mut kept = Vec::new();
    for (n, line) in std::io::BufReader::new(f).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let s: TraceSession = serde_json::from_str(&line)
            .with_context(|| format!("{}: line {}", replay.trace.display(), n + 1))?;
        total += 1;
        if selected(&s.session_id, replay.sample, replay.sample_seed)? && !s.calls.is_empty() {
            kept.push(s);
        }
    }
    Ok((total, kept))
}

/// Run a replay to completion and report it.
pub async fn run(mut config: Config) -> Result<()> {
    let replay = config.replay.clone().context("run requires [replay]")?;
    Metrics::init();

    if config.endpoint.health_check_timeout > 0 {
        crate::client::check_server_ready(
            &config.endpoint.base_url,
            config.endpoint.api_key.as_deref(),
            Duration::from_secs(config.endpoint.health_check_timeout),
            Duration::from_secs(config.endpoint.health_check_interval),
        )
        .await?;
    }
    let model = match config.endpoint.model.clone() {
        Some(m) => m,
        None => {
            let m = crate::client::detect_model(
                &config.endpoint.base_url,
                config.endpoint.api_key.as_deref(),
                Duration::from_secs(config.endpoint.timeout),
            )
            .await?;
            config.endpoint.model = Some(m.clone());
            m
        }
    };

    let (sessions_in_trace, sessions) = load_sessions(&replay)?;
    if sessions.is_empty() {
        bail!(
            "no sessions selected from {} ({} in trace, sample {})",
            replay.trace.display(),
            sessions_in_trace,
            replay.sample
        );
    }
    info!(
        "selected {} of {} sessions ({:.4})",
        sessions.len(),
        sessions_in_trace,
        sessions.len() as f64 / sessions_in_trace as f64
    );

    // Generation requests are never retried (a retry resends a prompt the
    // server may have cached) and never reuse an idle connection, so no
    // generation request is sent on a connection the server may have closed.
    let client_config = |max_retries: u32, pool_size: usize| ClientConfig {
        base_url: config.endpoint.base_url.clone(),
        api_key: config.endpoint.api_key.clone(),
        model: model.clone(),
        timeout: Duration::from_secs(config.endpoint.timeout),
        max_retries,
        retry_initial_delay_ms: config.endpoint.retry_initial_delay_ms,
        retry_max_delay_ms: config.endpoint.retry_max_delay_ms,
        pool_size,
        stream_idle_timeout: (config.endpoint.stream_idle_timeout > 0)
            .then(|| Duration::from_secs(config.endpoint.stream_idle_timeout)),
        retry_on_timeout: false,
        chat_template_kwargs: config.endpoint.chat_template_kwargs.clone(),
        ignore_eos: Some(true),
        pool_idle_timeout: Duration::from_millis(config.endpoint.pool_idle_timeout_ms),
    };
    let client = Arc::new(OpenAIClient::new(client_config(0, 0))?);
    // Render, tokenize and detokenize requests are idempotent and may retry.
    let aux_client = Arc::new(OpenAIClient::new(client_config(3, 32))?);

    // Every call must fit the server's per-request context.
    let server = probe(&aux_client, replay.server, &model).await?;
    let n_ctx = server.n_ctx;
    info!(
        "server build {}, per-request context {n_ctx}",
        server.build.as_deref().unwrap_or("unknown")
    );
    let too_long = sessions
        .iter()
        .filter(|s| s.calls.iter().any(|c| c.prompt + c.completion > n_ctx))
        .count();
    if too_long > 0 {
        bail!(
            "{too_long} selected sessions have a call longer than the server's per-request context ({n_ctx} tokens); convert the trace with --max-context {n_ctx}"
        );
    }

    let renderer = ServerRenderer::new(
        replay.server,
        aux_client.clone(),
        config.endpoint.chat_template_kwargs.clone(),
    );
    let pool = FillerPool::build(&renderer)
        .await
        .context("building the filler pool")?;
    info!("filler pool: {} single-token words", pool.len());
    let system = (replay.system_prompt_tokens > 0).then(|| Message {
        role: "system".to_string(),
        content: pool.text(
            &mut call_rng(replay.seed, "system", 0),
            replay.system_prompt_tokens,
        ),
    });

    let (log_tx, log_task) = match &replay.log {
        Some(path) => {
            let file = std::fs::File::create(path)
                .with_context(|| format!("creating {}", path.display()))?;
            let (tx, mut rx) = mpsc::unbounded_channel::<CallRecord>();
            let task = tokio::task::spawn_blocking(move || -> Result<()> {
                let mut w = std::io::BufWriter::new(file);
                while let Some(r) = rx.blocking_recv() {
                    serde_json::to_writer(&mut w, &r)?;
                    w.write_all(b"\n")?;
                }
                w.flush()?;
                Ok(())
            });
            (Some(tx), Some(task))
        }
        None => (None, None),
    };

    let t0 = sessions.iter().map(|s| s.start_ms).min().unwrap_or(0);
    let start = Instant::now();
    let kv_capture = match &replay.kv_events {
        Some(kv) => Some(KvCapture::start(kv, start).await?),
        None => None,
    };
    let shared = Arc::new(Shared {
        client,
        renderer,
        pool,
        system,
        start,
        deadline: config
            .load
            .duration_seconds
            .map(|d| start + Duration::from_secs(d)),
        warmup_until: config
            .load
            .warmup_duration
            .map(|d| start + Duration::from_secs(d)),
        log: log_tx,
        stats: Mutex::new(Stats::default()),
        stop: watch::Sender::new(false),
        abort_reason: Mutex::new(None),
        replay: replay.clone(),
    });

    crate::metrics::RUNNING.store(true, Ordering::SeqCst);
    let selected_count = sessions.len();
    let mut handles = Vec::with_capacity(selected_count);
    for s in sessions {
        let offset = session_start(s.start_ms, t0, replay.speedup);
        let sh = shared.clone();
        handles.push(tokio::spawn(run_session(sh, s, offset)));
    }
    let (mut completed, mut failed, mut truncated) = (0, 0, 0);
    for h in handles {
        match h.await {
            Ok(Outcome::Completed) => completed += 1,
            Ok(Outcome::Failed) => failed += 1,
            Ok(Outcome::Truncated) => truncated += 1,
            // Stop the other sessions, keep joining them, and report the panic
            // as the run's error.
            Err(e) => {
                shared.fail_run(format!("a session task panicked: {e}"));
                failed += 1;
            }
        }
    }
    crate::metrics::RUNNING.store(false, Ordering::SeqCst);
    let kv_events = match kv_capture {
        Some(c) => Some(c.finish().await.context("capturing KV events")?),
        None => None,
    };

    // Close the log channel so the writer finishes.
    let shared = Arc::into_inner(shared).context("a session task outlived the run")?;
    drop(shared.log);
    if let Some(task) = log_task {
        task.await??;
    }
    if let Some(reason) = shared.abort_reason.into_inner().unwrap() {
        bail!(reason);
    }

    let mut st = shared.stats.into_inner().unwrap();
    let summary = ReplaySummary {
        server: replay.server,
        server_build: server.build,
        server_n_ctx: n_ctx,
        sessions_in_trace,
        sessions_selected: selected_count,
        sessions_completed: completed,
        sessions_failed: failed,
        sessions_truncated: truncated,
        calls_sent: st.calls_sent,
        calls_failed: st.calls_failed,
        calls_overshoot: st.calls_overshoot,
        calls_shortfall: st.calls_shortfall,
        calls_eos_not_ignored: st.calls_eos_not_ignored,
        calls_render_mismatch: st.calls_render_mismatch,
        render_mismatch_max_tokens: st.render_mismatch_max_tokens,
        prompt_tokens: st.prompt_tokens,
        cached_tokens: st.cached_tokens,
        prefill_tokens: st.prefill_tokens,
        calls_measured: st.permille.len(),
        cached_permille_p50: percentile(&mut st.permille, 50.0),
        cached_permille_p10: percentile(&mut st.permille, 10.0),
        lag_ms_p50: percentile(&mut st.lag_ms, 50.0),
        lag_ms_p99: percentile(&mut st.lag_ms, 99.0),
        lag_ms_max: percentile(&mut st.lag_ms, 100.0),
        ttft_ms_p50: percentile(&mut st.ttft_ms, 50.0),
        ttft_ms_p99: percentile(&mut st.ttft_ms, 99.0),
        prompt_size_error_median: percentile(&mut st.size_errors, 50.0),
        kv_events,
    };
    report(&config, &summary)
}

fn report(config: &Config, s: &ReplaySummary) -> Result<()> {
    let text = match config.output.format {
        OutputFormat::Json => serde_json::to_string_pretty(s)?,
        OutputFormat::Console => {
            let f = |v: Option<f64>| v.map_or("-".to_string(), |v| format!("{v:.1}"));
            format!(
                "Trace replay\n\
                 \x20 server: {} build {}, per-request context {} tokens\n\
                 \x20 sessions: {} selected of {} ({} completed, {} failed, {} truncated)\n\
                 \x20 calls: {} sent, {} failed, {} overshoot, {} shortfall, {} EOS not ignored, {} render mismatch (max {} tokens)\n\
                 \x20 tokens: prompt {}, cached {}, prefill computed {}\n\
                 \x20 cached/prompt permille ({} calls with traced reuse): p50 {}, p10 {}\n\
                 \x20 lag ms: p50 {}, p99 {}, max {}\n\
                 \x20 ttft ms: p50 {}, p99 {}\n\
                 \x20 prompt size error (median): {}\n",
                match s.server {
                    ReplayServer::LlamaServer => "llama-server",
                    ReplayServer::Vllm => "vllm",
                },
                s.server_build.as_deref().unwrap_or("unknown"),
                s.server_n_ctx,
                s.sessions_selected,
                s.sessions_in_trace,
                s.sessions_completed,
                s.sessions_failed,
                s.sessions_truncated,
                s.calls_sent,
                s.calls_failed,
                s.calls_overshoot,
                s.calls_shortfall,
                s.calls_eos_not_ignored,
                s.calls_render_mismatch,
                s.render_mismatch_max_tokens,
                s.prompt_tokens,
                s.cached_tokens,
                s.prefill_tokens,
                s.calls_measured,
                f(s.cached_permille_p50),
                f(s.cached_permille_p10),
                f(s.lag_ms_p50),
                f(s.lag_ms_p99),
                f(s.lag_ms_max),
                f(s.ttft_ms_p50),
                f(s.ttft_ms_p99),
                s.prompt_size_error_median
                    .map_or("-".to_string(), |v| format!("{:.4}", v)),
            ) + &s.kv_events.as_ref().map_or(String::new(), |k| {
                format!(
                    "\x20 kv events: {} batches ({} missed), {} stored, {} removed, {} cleared, {} with another block size\n",
                    k.batches,
                    k.batches_missed,
                    k.stored_events,
                    k.removed_events,
                    k.cleared_events,
                    k.block_size_mismatch,
                )
            })
        }
    };
    match &config.output.file {
        Some(path) => std::fs::write(path, text)?,
        None => print!("{text}"),
    }
    Ok(())
}

async fn run_session(sh: Arc<Shared>, s: TraceSession, offset: Duration) -> Outcome {
    let session_t0 = sh.start + offset;
    if !wait_until(sh.deadline, &sh.stop, session_t0).await {
        REPLAY_SESSIONS.increment(REPLAY_SESSION_TRUNCATED);
        return Outcome::Truncated;
    }
    REPLAY_SESSIONS.increment(REPLAY_SESSION_STARTED);
    let slots = session_slots(
        &s.calls,
        sh.replay.speedup,
        sh.replay.max_gap_ms.map(Duration::from_millis),
    );
    let mut prompt = SessionPrompt::default();
    let outcome = 'calls: {
        for (i, c) in s.calls.iter().enumerate() {
            if *sh.stop.borrow() || sh.past_deadline() {
                break 'calls Outcome::Truncated;
            }
            let model = c.model.clone().or_else(|| s.models.first().cloned());
            let mut record = CallRecord {
                session_id: s.session_id.clone(),
                call: i,
                model,
                scheduled_ms: sh.ms(session_t0 + slots[i].offset),
                sent_ms: 0.0,
                lag_ms: 0.0,
                gap_capped: slots[i].gap_capped,
                warmup: false,
                target_prompt: c.prompt,
                rendered_tokens: None,
                prompt_tokens: None,
                overshoot: false,
                reuse: c.reuse,
                reuse_inferred: c.reuse_inferred,
                shortfall: 0,
                cached_tokens: None,
                max_tokens: c.completion.clamp(1, u32::MAX as u64) as u32,
                completion_tokens: None,
                finish_reason: None,
                ttft_ms: None,
                e2e_ms: None,
                error: None,
                block_fps: None,
            };

            let mut rng = call_rng(sh.replay.seed, &s.session_id, i);
            let built = match prompt
                .build(
                    &sh.renderer,
                    sh.system.as_ref(),
                    c.reuse as usize,
                    c.prompt as usize,
                    &sh.pool,
                    &mut rng,
                )
                .await
            {
                Ok(b) => b,
                Err(e) => {
                    record.error = Some(format!("building prompt: {e:#}"));
                    sh.emit(record);
                    break 'calls Outcome::Failed;
                }
            };
            record.rendered_tokens = Some(built.prompt_tokens as u64);
            if let Some(kv) = &sh.replay.kv_events {
                record.block_fps = Some(
                    block_fingerprints(prompt.rendered(), kv.block_size)
                        .into_iter()
                        .map(fp_hex)
                        .collect(),
                );
            }
            record.shortfall = built.shortfall as u64;
            record.overshoot = built.overshoot;

            if !wait_until(sh.deadline, &sh.stop, session_t0 + slots[i].offset).await {
                break 'calls Outcome::Truncated;
            }
            let sent = Instant::now();
            record.sent_ms = sh.ms(sent);
            record.lag_ms = (record.sent_ms - record.scheduled_ms).max(0.0);
            record.warmup = sh.warmup_until.is_some_and(|w| sent < w);

            let reply = send_call(&sh, &built.messages, &mut record).await;
            account(&sh, &record);
            sh.emit(record.clone());
            match reply {
                Some(content) => {
                    if let Err(e) = prompt.record_reply(&sh.renderer, content).await {
                        warn!("session {}: tokenizing reply: {e:#}", s.session_id);
                        break 'calls Outcome::Failed;
                    }
                }
                None => break 'calls Outcome::Failed,
            }
        }
        Outcome::Completed
    };
    REPLAY_SESSIONS.increment(match outcome {
        Outcome::Completed => REPLAY_SESSION_COMPLETED,
        Outcome::Failed => REPLAY_SESSION_FAILED,
        Outcome::Truncated => REPLAY_SESSION_TRUNCATED,
    });
    outcome
}

/// Send one call and fill in `record`. Returns the reply's content, or `None`
/// if the call failed.
async fn send_call(sh: &Shared, messages: &[Message], record: &mut CallRecord) -> Option<String> {
    let guard = InflightGuard::new(!record.warmup);
    let request = sh
        .client
        .create_messages_request(messages, Some(record.max_tokens), None, None);
    let started = std::time::Instant::now();

    let mut stream = match sh.client.chat_completion_stream(request).await {
        Ok(s) => s,
        Err(e) => {
            guard.complete(RequestStatus::Failed(classify_error(&e)));
            record.error = Some(format!("{e:#}"));
            return None;
        }
    };
    let mut content = String::new();
    loop {
        match stream.next_chunk().await {
            Ok(Some(chunk)) => {
                for choice in chunk.choices {
                    if let Some(c) = choice.delta.content {
                        content.push_str(&c);
                    }
                    if let Some(f) = choice.finish_reason {
                        record.finish_reason = Some(f);
                    }
                }
            }
            Ok(None) => break,
            Err(e) => {
                guard.complete(RequestStatus::Failed(ErrorType::Stream));
                record.error = Some(format!("{e:#}"));
                return None;
            }
        }
    }
    let elapsed = started.elapsed();
    record.e2e_ms = Some(elapsed.as_secs_f64() * 1000.0);
    record.ttft_ms = stream
        .time_to_first_token()
        .map(|t| t.as_secs_f64() * 1000.0);
    let usage = stream.server_usage().cloned();
    if let Some(u) = &usage {
        record.prompt_tokens = Some(u.prompt_tokens as u64);
        record.completion_tokens = Some(u.completion_tokens as u64);
        record.cached_tokens = u
            .prompt_tokens_details
            .as_ref()
            .map(|d| d.cached_tokens as u64);
    }

    if !record.warmup {
        let input = record.prompt_tokens.unwrap_or(0);
        if let Some(ttft) = stream.time_to_first_token() {
            Metrics::record_ttft(ttft, input);
        }
        if let Some(ttft) = stream.time_to_first_content_token() {
            Metrics::record_ttft_content(ttft, input);
        }
        for itl in stream.content_inter_token_latencies() {
            Metrics::record_itl(*itl, input, Phase::Content);
        }
        for itl in stream.reasoning_inter_token_latencies() {
            Metrics::record_itl(*itl, input, Phase::Reasoning);
        }
        Metrics::record_latency(elapsed);
        Metrics::record_schedule_slip(Duration::from_secs_f64(record.lag_ms / 1000.0));
        let (out_reasoning, out_content) = effective_output_tokens(
            usage.as_ref(),
            stream.reasoning_tokens() as u64,
            stream.content_tokens() as u64,
            stream.has_reasoning(),
        );
        Metrics::record_tokens(input, out_reasoning, out_content);
        if let Some(content_ttft) = stream.time_to_first_content_token()
            && let Some(tpot) = decode_tpot(elapsed.saturating_sub(content_ttft), out_content)
        {
            Metrics::record_tpot(tpot, Phase::Content);
        }
        let cached = record.cached_tokens.unwrap_or(0);
        REPLAY_REUSE.add(REPLAY_REUSE_PROMPT, input);
        REPLAY_REUSE.add(REPLAY_REUSE_CACHED, cached);
        if record.reuse > 0 && input > 0 {
            let _ = REPLAY_CACHED_PERMILLE.increment(1000 * cached / input);
        }
    }
    guard.complete(RequestStatus::Success);
    debug!(
        "session {} call {}: prompt {:?} cached {:?}",
        record.session_id, record.call, record.prompt_tokens, record.cached_tokens
    );
    Some(content)
}

/// Why a successful call cannot be used to measure reuse, if it cannot: the
/// server did not report prompt or cached token counts.
fn missing_usage(r: &CallRecord) -> Option<String> {
    let missing = match (r.prompt_tokens, r.cached_tokens) {
        (None, _) => "usage",
        (Some(_), None) => "usage.prompt_tokens_details",
        _ => return None,
    };
    Some(format!(
        "session {} call {}: the response has no {missing}; replay needs the server's prompt and cached token counts",
        r.session_id, r.call
    ))
}

/// Add a finished call to the run totals and apply the prompt-size check.
fn account(sh: &Shared, r: &CallRecord) {
    let mut st = sh.stats.lock().unwrap();
    st.calls_sent += 1;
    if r.error.is_some() {
        st.calls_failed += 1;
        return;
    }
    if let Some(reason) = missing_usage(r) {
        drop(st);
        sh.fail_run(reason);
        return;
    }
    if let Some(p) = r.prompt_tokens {
        st.size_errors
            .push((p as f64 - r.target_prompt as f64).abs() / r.target_prompt.max(1) as f64);
    }
    if let Some((reported, rendered)) = st.add_render(r) {
        warn!(
            "session {} call {}: the server reported {reported} prompt tokens, replay rendered {rendered}; the server built a different prompt from the one replay rendered",
            r.session_id, r.call
        );
    }
    if !r.warmup {
        st.add_measured(r);
    }
    let n = st.size_errors.len();
    if n > 0 && n.is_multiple_of(SIZE_CHECK_EVERY) {
        let mut v = st.size_errors.clone();
        drop(st);
        let median = percentile(&mut v, 50.0).unwrap_or(0.0);
        if median > SIZE_CHECK_MAX_ERROR {
            sh.fail_run(format!(
                "median prompt size error {median:.4} over {n} calls exceeds {SIZE_CHECK_MAX_ERROR}; prompts are not being sized correctly"
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(prompt: Option<u64>, cached: Option<u64>) -> CallRecord {
        CallRecord {
            session_id: "s".to_string(),
            call: 3,
            model: None,
            scheduled_ms: 0.0,
            sent_ms: 0.0,
            lag_ms: 0.0,
            gap_capped: false,
            warmup: false,
            target_prompt: 100,
            rendered_tokens: Some(100),
            prompt_tokens: prompt,
            overshoot: false,
            reuse: 50,
            reuse_inferred: false,
            shortfall: 0,
            cached_tokens: cached,
            max_tokens: 10,
            completion_tokens: Some(10),
            finish_reason: None,
            ttft_ms: None,
            e2e_ms: None,
            error: None,
            block_fps: None,
        }
    }

    #[test]
    fn cached_permille_is_cached_over_prompt_tokens() {
        // The captured vLLM next turn: 7100 prompt tokens, 6864 cached.
        let mut r = record(Some(7100), Some(6864));
        let mut st = Stats::default();
        st.add_measured(&r);
        assert_eq!(st.permille, vec![1000.0 * 6864.0 / 7100.0]);
        assert_eq!(st.prompt_tokens, 7100);
        assert_eq!(st.cached_tokens, 6864);
        assert_eq!(st.prefill_tokens, 7100 - 6864);
        // A call the trace built with no reuse counts in the totals but not in
        // the percentiles.
        r.reuse = 0;
        st.add_measured(&r);
        assert_eq!(st.permille.len(), 1);
        assert_eq!(st.prompt_tokens, 14200);
    }

    #[test]
    fn render_mismatch_is_counted_and_the_first_returned() {
        let mut st = Stats::default();
        let mut r = record(Some(100), Some(0));
        assert_eq!(st.add_render(&r), None);
        r.rendered_tokens = Some(99);
        assert_eq!(st.add_render(&r), Some((100, 99)));
        assert_eq!(st.add_render(&r), None);
        assert_eq!(st.calls_render_mismatch, 2);
        assert_eq!(st.render_mismatch_max_tokens, 1);
        // The largest difference in either direction is kept.
        r.rendered_tokens = Some(130);
        st.add_render(&r);
        r.rendered_tokens = Some(90);
        st.add_render(&r);
        assert_eq!(st.render_mismatch_max_tokens, 30);
        assert_eq!(st.calls_render_mismatch, 4);
        // No reported count is a missing-usage failure, and no rendered count
        // a build failure; neither is a mismatch.
        assert_eq!(st.add_render(&record(None, None)), None);
        r.rendered_tokens = None;
        assert_eq!(st.add_render(&r), None);
        assert_eq!(st.calls_render_mismatch, 4);
    }

    #[test]
    fn missing_cached_tokens_is_reported_not_counted_as_zero() {
        assert_eq!(missing_usage(&record(Some(100), Some(0))), None);
        let no_details = missing_usage(&record(Some(100), None)).unwrap();
        assert!(
            no_details.contains("usage.prompt_tokens_details"),
            "{no_details}"
        );
        assert!(no_details.contains("session s call 3"), "{no_details}");
        let no_usage = missing_usage(&record(None, None)).unwrap();
        assert!(no_usage.contains("no usage;"), "{no_usage}");
    }

    #[tokio::test]
    async fn wait_until_returns_at_the_target_time() {
        let stop = watch::Sender::new(false);
        let t = Instant::now() + Duration::from_millis(50);
        assert!(wait_until(None, &stop, t).await);
        assert!(Instant::now() >= t);
    }

    #[tokio::test]
    async fn wait_until_wakes_at_the_deadline() {
        let stop = watch::Sender::new(false);
        let start = Instant::now();
        let deadline = Some(start + Duration::from_millis(50));
        let go = wait_until(deadline, &stop, start + Duration::from_secs(30)).await;
        assert!(!go);
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "{:?}",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn wait_until_wakes_when_stopped() {
        let stop = Arc::new(watch::Sender::new(false));
        let s = stop.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            s.send_replace(true);
        });
        let start = Instant::now();
        assert!(!wait_until(None, &stop, start + Duration::from_secs(30)).await);
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "{:?}",
            start.elapsed()
        );
    }
}
