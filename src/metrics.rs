use metriken::{AtomicHistogram, Counter, Gauge, LazyCounter, LazyGauge, metric};
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
#[metric(name = "errors", metadata = { "type" = "stream" })]
pub static ERRORS_STREAM: LazyCounter = LazyCounter::new(Counter::default);
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

// TTFT — first token of any kind (prefill latency), context-size bucketed
#[metric(name = "ttft", metadata = { unit = "nanoseconds", context_size = "small" })]
pub static TTFT_SMALL: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "ttft", metadata = { unit = "nanoseconds", context_size = "medium" })]
pub static TTFT_MEDIUM: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "ttft", metadata = { unit = "nanoseconds", context_size = "large" })]
pub static TTFT_LARGE: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "ttft", metadata = { unit = "nanoseconds", context_size = "xlarge" })]
pub static TTFT_XLARGE: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "ttft", metadata = { unit = "nanoseconds", context_size = "xxlarge" })]
pub static TTFT_XXLARGE: AtomicHistogram = AtomicHistogram::new(7, 64);

// TTFT content — first visible content token (user-perceived latency), context-size bucketed
#[metric(name = "ttft_content", metadata = { unit = "nanoseconds", context_size = "small" })]
pub static TTFT_CONTENT_SMALL: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "ttft_content", metadata = { unit = "nanoseconds", context_size = "medium" })]
pub static TTFT_CONTENT_MEDIUM: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "ttft_content", metadata = { unit = "nanoseconds", context_size = "large" })]
pub static TTFT_CONTENT_LARGE: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "ttft_content", metadata = { unit = "nanoseconds", context_size = "xlarge" })]
pub static TTFT_CONTENT_XLARGE: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "ttft_content", metadata = { unit = "nanoseconds", context_size = "xxlarge" })]
pub static TTFT_CONTENT_XXLARGE: AtomicHistogram = AtomicHistogram::new(7, 64);

#[metric(name = "request_latency", metadata = { unit = "nanoseconds" })]
pub static REQUEST_LATENCY: AtomicHistogram = AtomicHistogram::new(7, 64);

// TPOT — per phase
#[metric(name = "tpot", metadata = { unit = "nanoseconds", phase = "reasoning" })]
pub static TPOT_REASONING: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "tpot", metadata = { unit = "nanoseconds", phase = "content" })]
pub static TPOT_CONTENT: AtomicHistogram = AtomicHistogram::new(7, 64);

// Think duration — time from first reasoning token to first content token
#[metric(name = "think_duration", metadata = { unit = "nanoseconds" })]
pub static THINK_DURATION: AtomicHistogram = AtomicHistogram::new(7, 64);

// ITL — per phase, context-size bucketed
#[metric(name = "itl", metadata = { unit = "nanoseconds", context_size = "small", phase = "reasoning" })]
pub static ITL_REASONING_SMALL: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "itl", metadata = { unit = "nanoseconds", context_size = "medium", phase = "reasoning" })]
pub static ITL_REASONING_MEDIUM: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "itl", metadata = { unit = "nanoseconds", context_size = "large", phase = "reasoning" })]
pub static ITL_REASONING_LARGE: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "itl", metadata = { unit = "nanoseconds", context_size = "xlarge", phase = "reasoning" })]
pub static ITL_REASONING_XLARGE: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "itl", metadata = { unit = "nanoseconds", context_size = "xxlarge", phase = "reasoning" })]
pub static ITL_REASONING_XXLARGE: AtomicHistogram = AtomicHistogram::new(7, 64);

#[metric(name = "itl", metadata = { unit = "nanoseconds", context_size = "small", phase = "content" })]
pub static ITL_CONTENT_SMALL: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "itl", metadata = { unit = "nanoseconds", context_size = "medium", phase = "content" })]
pub static ITL_CONTENT_MEDIUM: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "itl", metadata = { unit = "nanoseconds", context_size = "large", phase = "content" })]
pub static ITL_CONTENT_LARGE: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "itl", metadata = { unit = "nanoseconds", context_size = "xlarge", phase = "content" })]
pub static ITL_CONTENT_XLARGE: AtomicHistogram = AtomicHistogram::new(7, 64);
#[metric(name = "itl", metadata = { unit = "nanoseconds", context_size = "xxlarge", phase = "content" })]
pub static ITL_CONTENT_XXLARGE: AtomicHistogram = AtomicHistogram::new(7, 64);

// Aggregation arrays
pub static ALL_TTFT: [&AtomicHistogram; 5] = [
    &TTFT_SMALL,
    &TTFT_MEDIUM,
    &TTFT_LARGE,
    &TTFT_XLARGE,
    &TTFT_XXLARGE,
];

pub static ALL_TTFT_CONTENT: [&AtomicHistogram; 5] = [
    &TTFT_CONTENT_SMALL,
    &TTFT_CONTENT_MEDIUM,
    &TTFT_CONTENT_LARGE,
    &TTFT_CONTENT_XLARGE,
    &TTFT_CONTENT_XXLARGE,
];

pub static ALL_ITL: [&AtomicHistogram; 10] = [
    &ITL_REASONING_SMALL,
    &ITL_REASONING_MEDIUM,
    &ITL_REASONING_LARGE,
    &ITL_REASONING_XLARGE,
    &ITL_REASONING_XXLARGE,
    &ITL_CONTENT_SMALL,
    &ITL_CONTENT_MEDIUM,
    &ITL_CONTENT_LARGE,
    &ITL_CONTENT_XLARGE,
    &ITL_CONTENT_XXLARGE,
];

pub static ALL_TPOT: [&AtomicHistogram; 2] = [&TPOT_REASONING, &TPOT_CONTENT];

pub struct Metrics;

impl Metrics {
    pub fn init() {
        // Metriken metrics are automatically registered via the #[metric] attribute
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
                        ErrorType::Stream => ERRORS_STREAM.increment(),
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
        let nanos = duration.as_nanos() as u64;
        let histogram = match input_tokens {
            0..=200 => &TTFT_SMALL,
            201..=500 => &TTFT_MEDIUM,
            501..=2000 => &TTFT_LARGE,
            2001..=8000 => &TTFT_XLARGE,
            _ => &TTFT_XXLARGE,
        };
        let _ = histogram.increment(nanos);
    }

    /// Record TTFT content — first visible content token.
    pub fn record_ttft_content(duration: Duration, input_tokens: u64) {
        let nanos = duration.as_nanos() as u64;
        let histogram = match input_tokens {
            0..=200 => &TTFT_CONTENT_SMALL,
            201..=500 => &TTFT_CONTENT_MEDIUM,
            501..=2000 => &TTFT_CONTENT_LARGE,
            2001..=8000 => &TTFT_CONTENT_XLARGE,
            _ => &TTFT_CONTENT_XXLARGE,
        };
        let _ = histogram.increment(nanos);
    }

    pub fn record_tpot(duration: Duration, phase: Phase) {
        let histogram = match phase {
            Phase::Reasoning => &TPOT_REASONING,
            Phase::Content => &TPOT_CONTENT,
        };
        let _ = histogram.increment(duration.as_nanos() as u64);
    }

    pub fn record_think_duration(duration: Duration) {
        let _ = THINK_DURATION.increment(duration.as_nanos() as u64);
    }

    pub fn record_itl(duration: Duration, input_tokens: u64, phase: Phase) {
        let nanos = duration.as_nanos() as u64;
        let histogram = match (phase, input_tokens) {
            (Phase::Reasoning, 0..=200) => &ITL_REASONING_SMALL,
            (Phase::Reasoning, 201..=500) => &ITL_REASONING_MEDIUM,
            (Phase::Reasoning, 501..=1000) => &ITL_REASONING_LARGE,
            (Phase::Reasoning, 1001..=2000) => &ITL_REASONING_XLARGE,
            (Phase::Reasoning, _) => &ITL_REASONING_XXLARGE,
            (Phase::Content, 0..=200) => &ITL_CONTENT_SMALL,
            (Phase::Content, 201..=500) => &ITL_CONTENT_MEDIUM,
            (Phase::Content, 501..=1000) => &ITL_CONTENT_LARGE,
            (Phase::Content, 1001..=2000) => &ITL_CONTENT_XLARGE,
            (Phase::Content, _) => &ITL_CONTENT_XXLARGE,
        };
        let _ = histogram.increment(nanos);
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
