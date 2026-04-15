use metriken::{AtomicHistogram, Counter, Gauge, HistogramGroup, LazyCounter, LazyGauge, metric};
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
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

// Request metrics
#[metric(name = "requests", metadata = { status = "sent" })]
pub static REQUESTS_SENT: LazyCounter = LazyCounter::new(Counter::default);
#[metric(name = "requests", metadata = { status = "success" })]
pub static REQUESTS_SUCCESS: LazyCounter = LazyCounter::new(Counter::default);
#[metric(name = "requests", metadata = { status = "error" })]
pub static REQUESTS_ERROR: LazyCounter = LazyCounter::new(Counter::default);
#[metric(name = "requests", metadata = { status = "timeout" })]
pub static REQUESTS_TIMEOUT: LazyCounter = LazyCounter::new(Counter::default);
#[metric(name = "requests", metadata = { status = "canceled" })]
pub static REQUESTS_CANCELED: LazyCounter = LazyCounter::new(Counter::default);
#[metric(name = "requests", metadata = { status = "retried" })]
pub static REQUESTS_RETRIED: LazyCounter = LazyCounter::new(Counter::default);

// Error category metrics
#[metric(name = "errors", metadata = { "type" = "connection" })]
pub static ERRORS_CONNECTION: LazyCounter = LazyCounter::new(Counter::default);
#[metric(name = "errors", metadata = { "type" = "http_4xx" })]
pub static ERRORS_HTTP_4XX: LazyCounter = LazyCounter::new(Counter::default);
#[metric(name = "errors", metadata = { "type" = "http_5xx" })]
pub static ERRORS_HTTP_5XX: LazyCounter = LazyCounter::new(Counter::default);
#[metric(name = "errors", metadata = { "type" = "parse" })]
pub static ERRORS_PARSE: LazyCounter = LazyCounter::new(Counter::default);
#[metric(name = "errors", metadata = { "type" = "other" })]
pub static ERRORS_OTHER: LazyCounter = LazyCounter::new(Counter::default);

// Token metrics — input has no phase, output is split by phase
#[metric(name = "tokens", metadata = { direction = "input" })]
pub static TOKENS_INPUT: LazyCounter = LazyCounter::new(Counter::default);
#[metric(name = "tokens", metadata = { direction = "output", phase = "reasoning" })]
pub static TOKENS_OUTPUT_REASONING: LazyCounter = LazyCounter::new(Counter::default);
#[metric(name = "tokens", metadata = { direction = "output", phase = "content" })]
pub static TOKENS_OUTPUT_CONTENT: LazyCounter = LazyCounter::new(Counter::default);

// Concurrency metrics
#[metric(name = "requests_inflight")]
pub static REQUESTS_INFLIGHT: LazyGauge = LazyGauge::new(Gauge::default);

// Conversation metrics (multi-turn)
#[metric(name = "conversations", metadata = { status = "sent" })]
pub static CONVERSATIONS_SENT: LazyCounter = LazyCounter::new(Counter::default);
#[metric(name = "conversations", metadata = { status = "success" })]
pub static CONVERSATIONS_SUCCESS: LazyCounter = LazyCounter::new(Counter::default);
#[metric(name = "conversations", metadata = { status = "failed" })]
pub static CONVERSATIONS_FAILED: LazyCounter = LazyCounter::new(Counter::default);
#[metric(name = "turns")]
pub static TURNS_TOTAL: LazyCounter = LazyCounter::new(Counter::default);
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
    }

    pub fn record_request_sent() {
        REQUESTS_SENT.increment();
        REQUESTS_INFLIGHT.increment();
    }

    pub fn record_request_complete(status: RequestStatus) {
        REQUESTS_INFLIGHT.decrement();
        match status {
            RequestStatus::Success => {
                REQUESTS_SUCCESS.increment();
            }
            RequestStatus::Failed(error_type) => match error_type {
                ErrorType::Timeout => {
                    REQUESTS_TIMEOUT.increment();
                }
                _ => {
                    REQUESTS_ERROR.increment();
                    match error_type {
                        ErrorType::Connection => ERRORS_CONNECTION.increment(),
                        ErrorType::Http4xx(_) => ERRORS_HTTP_4XX.increment(),
                        ErrorType::Http5xx(_) => ERRORS_HTTP_5XX.increment(),
                        ErrorType::Parse => ERRORS_PARSE.increment(),
                        ErrorType::Other => ERRORS_OTHER.increment(),
                        ErrorType::Timeout => unreachable!(),
                    };
                }
            },
            RequestStatus::Canceled => {
                REQUESTS_CANCELED.increment();
            }
        }
    }

    pub fn record_tokens(input: u64, output_reasoning: u64, output_content: u64) {
        TOKENS_INPUT.add(input);
        TOKENS_OUTPUT_REASONING.add(output_reasoning);
        TOKENS_OUTPUT_CONTENT.add(output_content);
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

    pub fn record_retry() {
        REQUESTS_RETRIED.increment();
    }

    pub fn record_conversation_sent() {
        CONVERSATIONS_SENT.increment();
    }

    pub fn record_conversation_complete(success: bool) {
        if success {
            CONVERSATIONS_SUCCESS.increment();
        } else {
            CONVERSATIONS_FAILED.increment();
        }
    }

    pub fn record_turn() {
        TURNS_TOTAL.increment();
    }

    pub fn record_conversation_latency(duration: Duration) {
        let _ = CONVERSATION_LATENCY.increment(duration.as_nanos() as u64);
    }
}
