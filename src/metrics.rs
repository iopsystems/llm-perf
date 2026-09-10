use metriken::{AtomicHistogram, Counter, CounterGroup, Gauge, HistogramGroup, LazyGauge, metric};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

#[derive(Debug, Clone, Copy)]
pub enum RequestStatus {
    Success,
    Failed(ErrorType),
    Canceled,
}

#[derive(Debug, Clone, Copy)]
pub enum ErrorType {
    Connection,
    Http4xx(u16),
    Http5xx(u16),
    Parse,
    Timeout,
    Stream,
    Other,
}

/// Generation phase for reasoning models.
#[derive(Debug, Clone, Copy)]
pub enum Phase {
    /// Reasoning/thinking tokens (e.g., Qwen3 `reasoning_content`, DeepSeek-R1)
    Reasoning,
    /// Visible content tokens
    Content,
}

// Global running flag for background tasks
pub static RUNNING: AtomicBool = AtomicBool::new(false);

impl Metrics {
    /// Gets the RUNNING flag value
    /// This provides a simple read-only interface to check if the system is running
    ///
    /// # Returns
    ///
    /// Returns true if the benchmark is currently running, false otherwise
    #[inline]
    pub fn is_running() -> bool {
        RUNNING.load(Ordering::Relaxed)
    }
}

/// Gets the RUNNING flag value (non-const version)
/// This provides a simple read-only interface to check if the system is running
///
/// # Returns
///
/// Returns true if the benchmark is currently running, false otherwise
#[inline]
pub fn is_running() -> bool {
    RUNNING.load(Ordering::Relaxed)
}

// Request counters — indexed by status
pub const REQ_SENT: usize = 0;
pub const REQ_SUCCESS: usize = 1;
pub const REQ_ERROR: usize = 2;
pub const REQ_TIMEOUT: usize = 3;
pub const REQ_CANCELED: usize = 4;
pub const REQ_RETRIED: usize = 5;
/// Requests the QPS generator declined to send because the outstanding-request
/// cap was reached (the server couldn't sustain the offered rate).
pub const REQ_DROPPED: usize = 6;

#[metric(name = "requests")]
pub static REQUESTS: CounterGroup = CounterGroup::new(7);

// Error counters — indexed by error type
pub const ERR_CONNECTION: usize = 0;
pub const ERR_HTTP_4XX: usize = 1;
pub const ERR_HTTP_5XX: usize = 2;
pub const ERR_PARSE: usize = 3;
pub const ERR_STREAM: usize = 4;
pub const ERR_OTHER: usize = 5;

#[metric(name = "errors")]
pub static ERRORS: CounterGroup = CounterGroup::new(6);

// Token counters — indexed by direction/phase
pub const TOK_INPUT: usize = 0;
pub const TOK_OUTPUT_REASONING: usize = 1;
pub const TOK_OUTPUT_CONTENT: usize = 2;

#[metric(name = "tokens")]
pub static TOKENS: CounterGroup = CounterGroup::new(3);

// Concurrency metrics
#[metric(name = "requests_inflight")]
pub static REQUESTS_INFLIGHT: LazyGauge = LazyGauge::new(Gauge::default);

// Conversation counters — indexed by status
pub const CONV_SENT: usize = 0;
pub const CONV_SUCCESS: usize = 1;
pub const CONV_FAILED: usize = 2;

#[metric(name = "conversations")]
pub static CONVERSATIONS: CounterGroup = CounterGroup::new(3);

#[metric(name = "turns")]
pub static TURNS: Counter = Counter::new();

// Malformed streaming chunks — `data:` SSE lines that could not be decoded or
// parsed and were skipped. A non-zero value means some tokens/usage were lost.
#[metric(name = "malformed_chunks")]
pub static MALFORMED_CHUNKS: Counter = Counter::new();

// Number of streamed content chunks (SSE events carrying content). Compared to the
// content token total it reveals how many tokens the server packed per chunk: when
// that ratio is >1, inter-token latency is chunk-granular (one gap per chunk covers
// several tokens) rather than truly per-token.
#[metric(name = "streamed_content_chunks")]
pub static STREAMED_CONTENT_CHUNKS: Counter = Counter::new();

// Counted requests whose TTFT came from the first stream event rather than from a
// delta carrying text — see `StreamResponse::used_ttft_fallback`. Non-zero means
// the TTFT histogram is a prefill proxy for those requests, not token-anchored.
#[metric(name = "ttft_fallback_requests")]
pub static TTFT_FALLBACK_REQUESTS: Counter = Counter::new();

#[metric(name = "conversation_latency", metadata = { unit = "nanoseconds" })]
pub static CONVERSATION_LATENCY: AtomicHistogram = AtomicHistogram::new(7, 64);

// Latency metrics (in nanoseconds)
// Histogram parameters: (grouping_power=7, max_value_power=64)
// 128 buckets per power of 2 (~0.54% relative precision), covering the full 64-bit range

// Context size indices for histogram groups
pub const CTX_SMALL: usize = 0;
pub const CTX_MEDIUM: usize = 1;
pub const CTX_LARGE: usize = 2;
pub const CTX_XLARGE: usize = 3;
pub const CTX_XXLARGE: usize = 4;
pub const CTX_COUNT: usize = 5;

// Phase indices for histogram groups
pub const PHASE_REASONING: usize = 0;
pub const PHASE_CONTENT: usize = 1;
pub const PHASE_COUNT: usize = 2;

// TTFT — first token of any kind (prefill latency), context-size bucketed
#[metric(name = "ttft", metadata = { unit = "nanoseconds" })]
pub static TTFT: HistogramGroup = HistogramGroup::new(CTX_COUNT, 7, 64);

// TTFT content — first visible content token (user-perceived latency), context-size bucketed
#[metric(name = "ttft_content", metadata = { unit = "nanoseconds" })]
pub static TTFT_CONTENT: HistogramGroup = HistogramGroup::new(CTX_COUNT, 7, 64);

#[metric(name = "request_latency", metadata = { unit = "nanoseconds" })]
pub static REQUEST_LATENCY: AtomicHistogram = AtomicHistogram::new(7, 64);

// Schedule slip — in QPS (open-loop) mode, the delay between a request's scheduled
// arrival and when it was actually sent (i.e. time spent waiting for an in-flight
// slot). Near zero means the generator kept up; a growing distribution means the
// run was effectively closed-loop and latency percentiles include real queueing.
#[metric(name = "schedule_slip", metadata = { unit = "nanoseconds" })]
pub static SCHEDULE_SLIP: AtomicHistogram = AtomicHistogram::new(7, 64);

// TPOT — per phase
#[metric(name = "tpot", metadata = { unit = "nanoseconds" })]
pub static TPOT: HistogramGroup = HistogramGroup::new(PHASE_COUNT, 7, 64);

// Think duration — time from first reasoning token to first content token
#[metric(name = "think_duration", metadata = { unit = "nanoseconds" })]
pub static THINK_DURATION: AtomicHistogram = AtomicHistogram::new(7, 64);

// ITL — per phase × context size
// Layout: [reasoning_small..reasoning_xxlarge, content_small..content_xxlarge]
pub const ITL_ENTRIES: usize = PHASE_COUNT * CTX_COUNT;
#[metric(name = "itl", metadata = { unit = "nanoseconds" })]
pub static ITL: HistogramGroup = HistogramGroup::new(ITL_ENTRIES, 7, 64);

// Cache outcome counters — indexed by expected×actual behavior
pub const CACHE_HIT_CONFIRMED: usize = 0; // expected hit, actual hit
pub const CACHE_HIT_EVICTED: usize = 1; // expected hit, actual miss (eviction signal)
pub const CACHE_MISS_CONFIRMED: usize = 2; // expected miss, actual miss
pub const CACHE_MISS_SPURIOUS: usize = 3; // expected miss, actual hit

#[metric(name = "cache")]
pub static CACHE: CounterGroup = CounterGroup::new(4);

// TTFT split by expected cache behavior
pub const CACHE_EXPECTED_HIT_IDX: usize = 0;
pub const CACHE_EXPECTED_MISS_IDX: usize = 1;

#[metric(name = "ttft_by_cache", metadata = { unit = "nanoseconds" })]
pub static TTFT_BY_CACHE: HistogramGroup = HistogramGroup::new(2, 7, 64);

/// Map input token count to a TTFT context-size index.
fn ttft_context_index(input_tokens: u64) -> usize {
    match input_tokens {
        0..=200 => CTX_SMALL,
        201..=500 => CTX_MEDIUM,
        501..=2000 => CTX_LARGE,
        2001..=8000 => CTX_XLARGE,
        _ => CTX_XXLARGE,
    }
}

/// Map input token count to an ITL context-size index.
fn itl_context_index(input_tokens: u64) -> usize {
    match input_tokens {
        0..=200 => CTX_SMALL,
        201..=500 => CTX_MEDIUM,
        501..=1000 => CTX_LARGE,
        1001..=2000 => CTX_XLARGE,
        _ => CTX_XXLARGE,
    }
}

/// Map phase to a phase offset.
fn phase_index(phase: Phase) -> usize {
    match phase {
        Phase::Reasoning => PHASE_REASONING,
        Phase::Content => PHASE_CONTENT,
    }
}

pub struct Metrics;

impl Metrics {
    pub fn init() {
        let ctx_names = ["small", "medium", "large", "xlarge", "xxlarge"];
        let phase_names = ["reasoning", "content"];

        // Request counter metadata
        for (idx, status) in ["sent", "success", "error", "timeout", "canceled", "retried"]
            .iter()
            .enumerate()
        {
            REQUESTS.set_metadata(
                idx,
                HashMap::from([("status".to_string(), status.to_string())]),
            );
        }

        // Error counter metadata
        for (idx, error_type) in [
            "connection",
            "http_4xx",
            "http_5xx",
            "parse",
            "stream",
            "other",
        ]
        .iter()
        .enumerate()
        {
            ERRORS.set_metadata(
                idx,
                HashMap::from([("type".to_string(), error_type.to_string())]),
            );
        }

        // Token counter metadata
        TOKENS.set_metadata(
            TOK_INPUT,
            HashMap::from([("direction".to_string(), "input".to_string())]),
        );
        TOKENS.set_metadata(
            TOK_OUTPUT_REASONING,
            HashMap::from([
                ("direction".to_string(), "output".to_string()),
                ("phase".to_string(), "reasoning".to_string()),
            ]),
        );
        TOKENS.set_metadata(
            TOK_OUTPUT_CONTENT,
            HashMap::from([
                ("direction".to_string(), "output".to_string()),
                ("phase".to_string(), "content".to_string()),
            ]),
        );

        // Conversation counter metadata
        for (idx, status) in ["sent", "success", "failed"].iter().enumerate() {
            CONVERSATIONS.set_metadata(
                idx,
                HashMap::from([("status".to_string(), status.to_string())]),
            );
        }

        // TTFT and TTFT_CONTENT per-index metadata
        for (idx, name) in ctx_names.iter().enumerate() {
            let meta = HashMap::from([("context_size".to_string(), name.to_string())]);
            TTFT.set_metadata(idx, meta.clone());
            TTFT_CONTENT.set_metadata(idx, meta);
        }

        // TPOT per-index metadata
        for (idx, name) in phase_names.iter().enumerate() {
            TPOT.set_metadata(
                idx,
                HashMap::from([("phase".to_string(), name.to_string())]),
            );
        }

        // ITL per-index metadata (phase × context_size)
        for (p_idx, p_name) in phase_names.iter().enumerate() {
            for (c_idx, c_name) in ctx_names.iter().enumerate() {
                ITL.set_metadata(
                    p_idx * CTX_COUNT + c_idx,
                    HashMap::from([
                        ("phase".to_string(), p_name.to_string()),
                        ("context_size".to_string(), c_name.to_string()),
                    ]),
                );
            }
        }

        // Cache outcome counter metadata
        for (idx, outcome) in [
            "hit_confirmed",
            "hit_evicted",
            "miss_confirmed",
            "miss_spurious",
        ]
        .iter()
        .enumerate()
        {
            CACHE.set_metadata(
                idx,
                HashMap::from([("outcome".to_string(), outcome.to_string())]),
            );
        }

        // TTFT_BY_CACHE histogram metadata
        for (idx, expected) in ["hit", "miss"].iter().enumerate() {
            TTFT_BY_CACHE.set_metadata(
                idx,
                HashMap::from([("expected".to_string(), expected.to_string())]),
            );
        }
    }

    fn record_status(status: RequestStatus) {
        match status {
            RequestStatus::Success => {
                REQUESTS.increment(REQ_SUCCESS);
            }
            RequestStatus::Failed(error_type) => match error_type {
                ErrorType::Timeout => {
                    REQUESTS.increment(REQ_TIMEOUT);
                }
                _ => {
                    REQUESTS.increment(REQ_ERROR);
                    match error_type {
                        ErrorType::Connection => ERRORS.increment(ERR_CONNECTION),
                        ErrorType::Http4xx(_) => ERRORS.increment(ERR_HTTP_4XX),
                        ErrorType::Http5xx(_) => ERRORS.increment(ERR_HTTP_5XX),
                        ErrorType::Parse => ERRORS.increment(ERR_PARSE),
                        ErrorType::Stream => ERRORS.increment(ERR_STREAM),
                        ErrorType::Other => ERRORS.increment(ERR_OTHER),
                        ErrorType::Timeout => unreachable!(),
                    };
                }
            },
            RequestStatus::Canceled => {
                REQUESTS.increment(REQ_CANCELED);
            }
        }
    }
}

/// RAII guard that maintains the `requests_inflight` gauge and the `requests`
/// counter group. Increments on construction; on drop decrements the gauge and
/// — if `complete` was not called — records the request as `Canceled`.
///
/// When `active` is false (e.g. warmup), the guard is a no-op: no counters or
/// gauges are touched at any point in its lifetime.
#[must_use = "InflightGuard records Canceled on drop unless complete() is called"]
pub struct InflightGuard {
    active: bool,
    completed: bool,
}

impl InflightGuard {
    pub fn new(active: bool) -> Self {
        if active {
            REQUESTS.increment(REQ_SENT);
            REQUESTS_INFLIGHT.increment();
        }
        Self {
            active,
            completed: false,
        }
    }

    /// Record terminal status and consume the guard. Drop will still decrement
    /// the inflight gauge but will not re-record status.
    pub fn complete(mut self, status: RequestStatus) {
        if self.active {
            Metrics::record_status(status);
        }
        self.completed = true;
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        REQUESTS_INFLIGHT.decrement();
        if !self.completed {
            REQUESTS.increment(REQ_CANCELED);
        }
    }
}

impl Metrics {
    pub fn record_tokens(input: u64, output_reasoning: u64, output_content: u64) {
        TOKENS.add(TOK_INPUT, input);
        TOKENS.add(TOK_OUTPUT_REASONING, output_reasoning);
        TOKENS.add(TOK_OUTPUT_CONTENT, output_content);
    }

    /// Record TTFT — first token of any kind (prefill latency).
    pub fn record_ttft(duration: Duration, input_tokens: u64) {
        let _ = TTFT.increment(ttft_context_index(input_tokens), duration.as_nanos() as u64);
    }

    /// Record TTFT content — first visible content token.
    pub fn record_ttft_content(duration: Duration, input_tokens: u64) {
        let _ =
            TTFT_CONTENT.increment(ttft_context_index(input_tokens), duration.as_nanos() as u64);
    }

    pub fn record_tpot(duration: Duration, phase: Phase) {
        let _ = TPOT.increment(phase_index(phase), duration.as_nanos() as u64);
    }

    pub fn record_think_duration(duration: Duration) {
        let _ = THINK_DURATION.increment(duration.as_nanos() as u64);
    }

    pub fn record_itl(duration: Duration, input_tokens: u64, phase: Phase) {
        let idx = phase_index(phase) * CTX_COUNT + itl_context_index(input_tokens);
        let _ = ITL.increment(idx, duration.as_nanos() as u64);
    }

    pub fn record_latency(duration: Duration) {
        let _ = REQUEST_LATENCY.increment(duration.as_nanos() as u64);
    }

    /// Record schedule slip: how long a request waited between its scheduled
    /// arrival and actually being sent (QPS/open-loop mode).
    pub fn record_schedule_slip(duration: Duration) {
        let _ = SCHEDULE_SLIP.increment(duration.as_nanos() as u64);
    }

    pub fn record_retry() {
        REQUESTS.increment(REQ_RETRIED);
    }

    /// Record a request the QPS generator declined to send due to the
    /// outstanding-request cap (overload shedding).
    pub fn record_dropped() {
        REQUESTS.increment(REQ_DROPPED);
    }

    /// Record how many content chunks a streamed response delivered (for the
    /// tokens-per-chunk / ITL-granularity note).
    pub fn record_content_chunks(chunks: u64) {
        STREAMED_CONTENT_CHUNKS.add(chunks);
    }

    /// Record a streaming `data:` line that could not be decoded/parsed and was skipped.
    pub fn record_malformed_chunk() {
        MALFORMED_CHUNKS.increment();
    }

    pub fn record_conversation_sent() {
        CONVERSATIONS.increment(CONV_SENT);
    }

    pub fn record_conversation_complete(success: bool) {
        if success {
            CONVERSATIONS.increment(CONV_SUCCESS);
        } else {
            CONVERSATIONS.increment(CONV_FAILED);
        }
    }

    pub fn record_turn() {
        TURNS.increment();
    }

    pub fn record_conversation_latency(duration: Duration) {
        let _ = CONVERSATION_LATENCY.increment(duration.as_nanos() as u64);
    }

    /// Record a cache outcome and split TTFT.
    /// `expected_hit`: true if request was sent with `[shared]` prefix (expected cache hit).
    /// `actual_hit`: true if server reported cached_tokens > 0. None if server did not report.
    pub fn record_cache_outcome(
        expected_hit: bool,
        actual_hit: Option<bool>,
        ttft: std::time::Duration,
    ) {
        let ttft_idx = if expected_hit {
            CACHE_EXPECTED_HIT_IDX
        } else {
            CACHE_EXPECTED_MISS_IDX
        };
        let _ = TTFT_BY_CACHE.increment(ttft_idx, ttft.as_nanos() as u64);

        if let Some(hit) = actual_hit {
            let idx = match (expected_hit, hit) {
                (true, true) => CACHE_HIT_CONFIRMED,
                (true, false) => CACHE_HIT_EVICTED,
                (false, true) => CACHE_MISS_SPURIOUS,
                (false, false) => CACHE_MISS_CONFIRMED,
            };
            let _ = CACHE.increment(idx);
        }
    }
}

/// Returns true if a request should be designated a cache miss, based on miss_rate.
pub fn should_miss(miss_rate: f64) -> bool {
    if miss_rate <= 0.0 {
        return false;
    }
    if miss_rate >= 1.0 {
        return true;
    }
    rand::random::<f64>() < miss_rate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_counter_indices_are_distinct() {
        assert_ne!(CACHE_HIT_CONFIRMED, CACHE_HIT_EVICTED);
        assert_ne!(CACHE_HIT_CONFIRMED, CACHE_MISS_CONFIRMED);
        assert_ne!(CACHE_HIT_CONFIRMED, CACHE_MISS_SPURIOUS);
        assert_ne!(CACHE_HIT_EVICTED, CACHE_MISS_CONFIRMED);
        assert_ne!(CACHE_HIT_EVICTED, CACHE_MISS_SPURIOUS);
        assert_ne!(CACHE_MISS_CONFIRMED, CACHE_MISS_SPURIOUS);
    }

    #[test]
    fn ttft_by_cache_hit_and_miss_indices_are_distinct() {
        assert_ne!(CACHE_EXPECTED_HIT_IDX, CACHE_EXPECTED_MISS_IDX);
    }

    #[test]
    fn record_schedule_slip_writes_to_histogram() {
        // The schedule-slip histogram captures how long a QPS-mode request waited
        // between its scheduled arrival and actually being sent.
        Metrics::record_schedule_slip(std::time::Duration::from_millis(5));
        let h = SCHEDULE_SLIP.load().expect("histogram should load");
        let total: u64 = h.iter().map(|b| b.count()).sum();
        assert!(total >= 1, "recorded slip sample should be present");
    }
}
