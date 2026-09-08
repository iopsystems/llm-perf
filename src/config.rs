use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Resolve the effective `max_tokens` for a request.
///
/// Endpoint-level configuration takes precedence over workload-level values so
/// a single config file can enforce a consistent response cap across modes.
pub fn resolve_max_tokens(
    endpoint_max_tokens: Option<u32>,
    workload_max_tokens: Option<u32>,
) -> Option<u32> {
    endpoint_max_tokens.or(workload_max_tokens)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub endpoint: EndpointConfig,
    pub load: LoadConfig,
    pub input: InputConfig,
    pub output: OutputConfig,
    #[serde(default)]
    pub runtime: RuntimeConfig,
    #[serde(default)]
    pub log: LogConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<MetricsConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub admin: Option<AdminConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogprobsConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub saturation: Option<SaturationConfig>,
    /// Inter-turn delay configuration for multi-turn conversations.
    /// If absent, behavior is identical to back-to-back turns (no delay).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<ConversationConfig>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConversationConfig {
    /// Mean delay between consecutive turns of a conversation, in milliseconds.
    #[serde(default)]
    pub turn_delay_ms: u64,
    /// Standard deviation for Gaussian sampling of the delay, in milliseconds.
    /// 0 (default) makes the delay deterministic.
    #[serde(default)]
    pub turn_delay_stdev_ms: u64,
    /// Lower clamp for the sampled delay, in milliseconds.
    #[serde(default)]
    pub turn_delay_min_ms: u64,
    /// Upper clamp for the sampled delay, in milliseconds.
    #[serde(default = "default_turn_delay_max_ms")]
    pub turn_delay_max_ms: u64,
}

impl Default for ConversationConfig {
    fn default() -> Self {
        Self {
            turn_delay_ms: 0,
            turn_delay_stdev_ms: 0,
            turn_delay_min_ms: 0,
            turn_delay_max_ms: default_turn_delay_max_ms(),
        }
    }
}

fn default_turn_delay_max_ms() -> u64 {
    60_000
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointConfig {
    pub base_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>, // If not provided, will auto-detect from server
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default)]
    pub max_retries: u32,
    #[serde(default = "default_retry_initial_delay_ms")]
    pub retry_initial_delay_ms: u64,
    #[serde(default = "default_retry_max_delay_ms")]
    pub retry_max_delay_ms: u64,
    /// Per-read idle timeout for streaming responses, in seconds (0 = disabled).
    /// Detects a stalled-but-open stream that the total request timeout may miss
    /// under HTTP/2 keep-alive.
    #[serde(default = "default_stream_idle_timeout")]
    pub stream_idle_timeout: u64,
    /// Retry requests that timed out. Off by default: a timed-out chat completion
    /// may still be running server-side, so retrying re-fires an expensive,
    /// non-idempotent generation.
    #[serde(default)]
    pub retry_on_timeout: bool,
    #[serde(default = "default_health_check_timeout")]
    pub health_check_timeout: u64, // Total time to wait for server readiness in seconds (0 = disabled)
    #[serde(default = "default_health_check_interval")]
    pub health_check_interval: u64, // Interval between readiness check retries in seconds
    /// Additional kwargs passed to the model's chat template for every request.
    /// For example, set `{enable_thinking = false}` to disable thinking mode on Qwen3.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<serde_json::Value>,
    /// Suppress the end-of-sequence token so generation always runs to
    /// `max_tokens`. Makes token-generation throughput comparable across models
    /// and quants, which otherwise stop at different natural lengths and blend
    /// TTFT into `output_tokens_per_second`.
    ///
    /// Supported by llama.cpp (as a logit bias over the EOG tokens) and vLLM.
    /// Servers that ignore unknown fields accept it silently and keep stopping
    /// early, so the run warns if generation came up short — see
    /// `warn_if_eos_not_ignored`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore_eos: Option<bool>,
    /// Exact tokenizer for prompt sizing: a local `tokenizer.json` path OR a
    /// HuggingFace model id (downloaded/cached). When unset, the tool calibrates a
    /// tiktoken estimate against the server's `/tokenize` if available, else uses a
    /// rough tiktoken estimate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ArrivalDistribution {
    #[default]
    Uniform, // Fixed intervals (deterministic)
    Poisson, // Exponential inter-arrival times (stochastic)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoadConfig {
    #[serde(default = "default_concurrent_requests")]
    pub concurrent_requests: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_requests: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qps: Option<f64>, // If set, uses fixed QPS mode; otherwise uses concurrent mode
    #[serde(default)]
    pub arrival_distribution: ArrivalDistribution, // Request arrival pattern (uniform or poisson)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmup_requests: Option<usize>, // Number of warmup requests to exclude from metrics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmup_duration: Option<u64>, // Warmup duration in seconds (alternative to warmup_requests)
    /// QPS mode only: maximum outstanding (in-flight + queued) requests. When the
    /// server can't sustain the offered rate, requests beyond this are dropped and
    /// counted as overload rather than queued unbounded. `None` uses a generous
    /// default; set explicitly to tune (or very high to effectively disable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_outstanding_requests: Option<usize>,
}

fn default_common_prefix_sample_ratio() -> f64 {
    0.0
}

fn default_common_prefix_tokens() -> usize {
    0
}

fn default_turns() -> usize {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticConfig {
    /// Average number of tokens in generated prompts
    pub prompt_tokens: usize,
    /// Standard deviation for prompt token count (Gaussian distribution)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_stdev: Option<usize>,
    /// Minimum prompt token count (hard limit)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_min: Option<usize>,
    /// Maximum prompt token count (hard limit)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_max: Option<usize>,
    /// Ratio of samples that share a common prefix (0.0 to 1.0).
    /// Used to test prefix caching effectiveness. 0.0 = all unique, 1.0 = all share common prefix.
    /// Selection is deterministic and strided (e.g. 0.5 picks every other request), not random.
    /// Default: 0.0
    #[serde(default = "default_common_prefix_sample_ratio")]
    pub common_prefix_sample_ratio: f64,
    /// Token length of the common prefix shared by common_prefix_sample_ratio of samples
    /// If equals prompt_tokens, the whole prompt is the common prefix
    /// Default: 0
    #[serde(default = "default_common_prefix_tokens")]
    pub common_prefix_tokens: usize,
    /// Number of turns per synthetic conversation (1 = single-turn, unchanged behavior)
    #[serde(default = "default_turns")]
    pub turns: usize,
    /// Token count per subsequent turn (defaults to prompt_tokens if not set)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_prompt_tokens: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SystemPromptConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<usize>,
}

impl SystemPromptConfig {
    fn source_count(&self) -> usize {
        [
            self.content.is_some(),
            self.file.is_some(),
            self.tokens.is_some(),
        ]
        .iter()
        .filter(|&&x| x)
        .count()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SharedPrefixConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<usize>,
    #[serde(default)]
    pub miss_rate: f64,
}

impl SharedPrefixConfig {
    fn source_count(&self) -> usize {
        [
            self.content.is_some(),
            self.file.is_some(),
            self.tokens.is_some(),
        ]
        .iter()
        .filter(|&&x| x)
        .count()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputConfig {
    pub file: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synthetic: Option<SyntheticConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<SystemPromptConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shared_prefix: Option<SharedPrefixConfig>,
}

impl InputConfig {
    pub fn is_synthetic(&self) -> bool {
        self.file.to_str() == Some("synthetic")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputConfig {
    #[serde(default = "default_output_format")]
    pub format: OutputFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<PathBuf>,
    #[serde(default)]
    pub quiet: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_log: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfig {
    #[serde(default = "default_worker_threads")]
    pub worker_threads: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogConfig {
    #[serde(default = "default_log_level")]
    pub level: LogLevel,
    /// Per-module log level overrides (e.g., ["hyper=info", "h2=warn"])
    #[serde(default)]
    pub filter: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Error => "error",
            LogLevel::Warn => "warn",
            LogLevel::Info => "info",
            LogLevel::Debug => "debug",
            LogLevel::Trace => "trace",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    Console,
    Json,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricsConfig {
    /// File for parquet metrics output
    pub output: PathBuf,
    /// The snapshot interval (e.g., "1s", "500ms")
    #[serde(default = "default_metrics_interval")]
    pub interval: String,
    /// Batch size for parquet files
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<usize>,
    /// Prometheus /metrics URL of the server under test. None => feature off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_metrics_url: Option<String>,
    /// Output parquet for server metrics. Defaults to `<output-stem>.server.parquet`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_metrics_output: Option<PathBuf>,
    /// Binary to invoke for server-metrics capture. Default `"rezolus"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rezolus_bin: Option<String>,
}

impl MetricsConfig {
    /// Resolved server-metrics parquet path: the explicit `server_metrics_output`,
    /// else the client `output` with its extension replaced by `.server.parquet`.
    pub fn resolved_server_output(&self) -> std::path::PathBuf {
        if let Some(p) = &self.server_metrics_output {
            return p.clone();
        }
        self.output.with_extension("server.parquet")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdminConfig {
    #[serde(default = "default_admin_listen")]
    pub listen: String,
    #[serde(default = "default_admin_enabled")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogprobsConfig {
    /// Whether to request logprobs from the server
    #[serde(default)]
    pub enabled: bool,
    /// Number of top token log probabilities to request (1-20)
    #[serde(default = "default_top_logprobs")]
    pub top_logprobs: u8,
    /// Path to write logprobs JSONL output
    pub output: PathBuf,
}

fn default_top_logprobs() -> u8 {
    5
}

impl Default for AdminConfig {
    fn default() -> Self {
        Self {
            listen: default_admin_listen(),
            enabled: default_admin_enabled(),
        }
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            worker_threads: default_worker_threads(),
        }
    }
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            filter: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SaturationConfig {
    /// SLO thresholds — at least one metric/percentile must be specified
    pub slo: SloThresholds,
    /// Starting concurrency level
    #[serde(default = "default_start_concurrency")]
    pub start_concurrency: usize,
    /// Multiplier for each concurrency step (must be > 1.0)
    #[serde(default = "default_step_multiplier")]
    pub step_multiplier: f64,
    /// Duration to sample at each concurrency level (e.g. "60s", "2m")
    #[serde(default = "default_sample_window")]
    pub sample_window: String,
    /// Deprecated and ignored: the search now bisects and confirms rather than
    /// stopping after N consecutive failures. Accepted for backward compatibility;
    /// a warning is logged if set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_after_failures: Option<u32>,
    /// Maximum concurrency to try
    #[serde(default = "default_max_concurrency")]
    pub max_concurrency: usize,
    /// Minimum throughput-efficiency ratio (0.0–1.0). A rung trips the throughput
    /// gate when its output tokens/s falls below this fraction of the throughput
    /// linearly projected from the previous rung (marginal, one step back) — i.e.
    /// it detects the plateau where adding concurrency stops adding throughput.
    #[serde(default = "default_min_throughput_ratio")]
    pub min_throughput_ratio: f64,
    /// Windows used to confirm the final boundary (M-of-N); a higher value is more
    /// robust to noise but slower. Default 3.
    #[serde(default = "default_confirm_windows")]
    pub confirm_windows: u32,
    /// Settle time (ms) after draining before measuring a rung, letting the new
    /// steady state establish. Default 500.
    #[serde(default = "default_drain_settle_ms")]
    pub drain_settle_ms: u64,
    /// Max time (s) to wait for in-flight requests to drain before measuring a
    /// down-stepped rung. `None` uses the sample window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drain_timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct SloThresholds {
    /// TTFT (time to first token) thresholds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft: Option<SloPercentiles>,
    /// ITL (inter-token latency) thresholds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub itl: Option<SloPercentiles>,
    /// TPOT (time per output token) thresholds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tpot: Option<SloPercentiles>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct SloPercentiles {
    /// Maximum acceptable p50 in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p50_ms: Option<f64>,
    /// Maximum acceptable p99 in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p99_ms: Option<f64>,
    /// Maximum acceptable p999 in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p999_ms: Option<f64>,
}

impl SloThresholds {
    /// Returns true if at least one threshold is configured.
    pub fn has_any(&self) -> bool {
        let has = |p: &Option<SloPercentiles>| {
            p.as_ref()
                .is_some_and(|s| s.p50_ms.is_some() || s.p99_ms.is_some() || s.p999_ms.is_some())
        };
        has(&self.ttft) || has(&self.itl) || has(&self.tpot)
    }
}

fn default_start_concurrency() -> usize {
    1
}

fn default_step_multiplier() -> f64 {
    1.5
}

fn default_sample_window() -> String {
    "60s".to_string()
}

fn default_confirm_windows() -> u32 {
    3
}

fn default_drain_settle_ms() -> u64 {
    500
}

fn default_max_concurrency() -> usize {
    512
}

fn default_min_throughput_ratio() -> f64 {
    0.9
}

fn default_timeout() -> u64 {
    60
}

fn default_retry_initial_delay_ms() -> u64 {
    100
}

fn default_retry_max_delay_ms() -> u64 {
    10000 // 10 seconds
}

fn default_stream_idle_timeout() -> u64 {
    0 // disabled by default to avoid surprising timeouts on legitimately slow models
}

fn default_health_check_timeout() -> u64 {
    0 // Disabled by default
}

fn default_health_check_interval() -> u64 {
    5 // 5 seconds
}

fn default_concurrent_requests() -> usize {
    10
}

fn default_output_format() -> OutputFormat {
    OutputFormat::Console
}

fn default_worker_threads() -> usize {
    num_cpus::get()
}

fn default_log_level() -> LogLevel {
    LogLevel::Info
}

fn default_metrics_interval() -> String {
    "1m".to_string()
}

fn default_admin_listen() -> String {
    "127.0.0.1:9090".to_string()
}

fn default_admin_enabled() -> bool {
    true
}

impl Config {
    pub fn load(path: &PathBuf) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&contents)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.load.total_requests.is_none() && self.load.duration_seconds.is_none() {
            anyhow::bail!("Either total_requests or duration_seconds must be specified");
        }

        if self.load.total_requests.is_some() && self.load.duration_seconds.is_some() {
            anyhow::bail!("Only one of total_requests or duration_seconds can be specified");
        }

        if self.load.warmup_requests.is_some() && self.load.warmup_duration.is_some() {
            anyhow::bail!(
                "Only one of load.warmup_requests or load.warmup_duration can be specified"
            );
        }

        // warmup_requests is only meaningful in total_requests concurrent mode or QPS mode;
        // in duration-based concurrent mode and saturation mode it would be silently ignored
        if self.load.warmup_requests.is_some() {
            if self.load.duration_seconds.is_some() && self.load.qps.is_none() {
                anyhow::bail!(
                    "load.warmup_requests cannot be used with duration_seconds in concurrent mode; \
                     use load.warmup_duration instead"
                );
            }
            if self.saturation.is_some() {
                anyhow::bail!(
                    "load.warmup_requests cannot be used in saturation mode; \
                     use load.warmup_duration instead"
                );
            }
        }

        if self.load.concurrent_requests == 0 {
            anyhow::bail!("concurrent_requests must be greater than 0");
        }

        // If qps is specified, we're in fixed QPS mode
        if let Some(qps) = self.load.qps
            && qps <= 0.0
        {
            anyhow::bail!("qps must be greater than 0");
        }

        if self.runtime.worker_threads == 0 {
            anyhow::bail!("worker_threads must be greater than 0");
        }

        if let Some(ref logprobs) = self.logprobs
            && logprobs.enabled
            && (logprobs.top_logprobs == 0 || logprobs.top_logprobs > 20)
        {
            anyhow::bail!("logprobs.top_logprobs must be between 1 and 20");
        }

        if let Some(max_tokens) = self.endpoint.max_tokens
            && max_tokens == 0
        {
            anyhow::bail!("endpoint.max_tokens must be greater than 0");
        }

        if let Some(ref sat) = self.saturation {
            if !sat.slo.has_any() {
                anyhow::bail!(
                    "saturation.slo must have at least one threshold (ttft, itl, or tpot with p50_ms, p99_ms, or p999_ms)"
                );
            }
            if sat.step_multiplier <= 1.0 {
                anyhow::bail!("saturation.step_multiplier must be greater than 1.0");
            }
            if sat.start_concurrency < 1 {
                anyhow::bail!("saturation.start_concurrency must be at least 1");
            }
            if sat.max_concurrency < sat.start_concurrency {
                anyhow::bail!("saturation.max_concurrency must be >= start_concurrency");
            }
            if !(0.0..=1.0).contains(&sat.min_throughput_ratio) {
                anyhow::bail!("saturation.min_throughput_ratio must be between 0.0 and 1.0");
            }
            // Validate sample_window parses
            humantime::parse_duration(&sat.sample_window)
                .map_err(|e| anyhow::anyhow!("saturation.sample_window: {}", e))?;
        }

        if let Some(ref conv) = self.conversation
            && conv.turn_delay_min_ms > conv.turn_delay_max_ms
        {
            anyhow::bail!(
                "conversation.turn_delay_min_ms ({}) must be <= turn_delay_max_ms ({})",
                conv.turn_delay_min_ms,
                conv.turn_delay_max_ms
            );
        }

        // Validate synthetic mode configuration
        if self.input.is_synthetic() {
            // Require endpoint.max_tokens to be set
            if self.endpoint.max_tokens.is_none() {
                anyhow::bail!(
                    "Synthetic mode (file = \"synthetic\") requires endpoint.max_tokens to be set"
                );
            }

            // Require [input.synthetic] section and validate its fields
            let synthetic = self.input.synthetic.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "Synthetic mode (file = \"synthetic\") requires [input.synthetic] configuration"
                )
            })?;

            if synthetic.prompt_tokens == 0 {
                anyhow::bail!("input.synthetic.prompt_tokens must be greater than 0");
            }

            // Validate prompt token bounds
            if let Some(min) = synthetic.prompt_tokens_min
                && min > synthetic.prompt_tokens
            {
                anyhow::bail!(
                    "input.synthetic.prompt_tokens_min ({}) must be <= prompt_tokens ({})",
                    min,
                    synthetic.prompt_tokens
                );
            }
            if let Some(max) = synthetic.prompt_tokens_max
                && max < synthetic.prompt_tokens
            {
                anyhow::bail!(
                    "input.synthetic.prompt_tokens_max ({}) must be >= prompt_tokens ({})",
                    max,
                    synthetic.prompt_tokens
                );
            }
            if let Some(stdev) = synthetic.prompt_tokens_stdev
                && stdev == 0
            {
                anyhow::bail!(
                    "input.synthetic.prompt_tokens_stdev must be greater than 0 if specified"
                );
            }

            // Validate common prefix fields
            if !(0.0..=1.0).contains(&synthetic.common_prefix_sample_ratio) {
                anyhow::bail!(
                    "input.synthetic.common_prefix_sample_ratio must be between 0.0 and 1.0"
                );
            }

            if synthetic.common_prefix_tokens > synthetic.prompt_tokens {
                anyhow::bail!(
                    "input.synthetic.common_prefix_tokens ({}) cannot exceed prompt_tokens ({})",
                    synthetic.common_prefix_tokens,
                    synthetic.prompt_tokens
                );
            }

            // Validate turn fields
            if synthetic.turns == 0 {
                anyhow::bail!("input.synthetic.turns must be greater than 0");
            }
            if let Some(turn_tokens) = synthetic.turn_prompt_tokens {
                if turn_tokens == 0 {
                    anyhow::bail!(
                        "input.synthetic.turn_prompt_tokens must be greater than 0 if specified"
                    );
                }
                // Effective upper bound matches TokenDistribution: explicit max, or
                // prompt_tokens + 5*stdev (the Gaussian tail clamp), or prompt_tokens when
                // no stdev is set.
                let effective_max = synthetic.prompt_tokens_max.unwrap_or_else(|| {
                    synthetic.prompt_tokens + 5 * synthetic.prompt_tokens_stdev.unwrap_or(0)
                });
                if turn_tokens > effective_max {
                    anyhow::bail!(
                        "input.synthetic.turn_prompt_tokens ({}) cannot exceed effective maximum \
                         prompt tokens ({})",
                        turn_tokens,
                        effective_max,
                    );
                }
            }
        }

        if let Some(ref sp) = self.input.system_prompt {
            if sp.source_count() == 0 {
                anyhow::bail!(
                    "input.system_prompt must have exactly one of: content, file, tokens"
                );
            }
            if sp.source_count() > 1 {
                anyhow::bail!(
                    "input.system_prompt must have exactly one of: content, file, tokens"
                );
            }
        }

        if let Some(ref pfx) = self.input.shared_prefix {
            if pfx.source_count() != 1 {
                anyhow::bail!(
                    "input.shared_prefix must have exactly one of: content, file, tokens"
                );
            }
            if !(0.0..=1.0).contains(&pfx.miss_rate) {
                anyhow::bail!(
                    "input.shared_prefix.miss_rate must be between 0.0 and 1.0, got {}",
                    pfx.miss_rate
                );
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_max_tokens_overrides_workload_value() {
        assert_eq!(resolve_max_tokens(Some(128), Some(32)), Some(128));
    }

    #[test]
    fn workload_max_tokens_is_used_when_endpoint_override_is_absent() {
        assert_eq!(resolve_max_tokens(None, Some(32)), Some(32));
    }

    #[test]
    fn no_max_tokens_when_neither_source_sets_it() {
        assert_eq!(resolve_max_tokens(None, None), None);
    }

    #[test]
    fn config_without_conversation_section_omits_field_on_serialize() {
        // Backward-compat acceptance: a config that omits [conversation] must
        // not introduce the field into serialized output.
        let toml = r#"
[endpoint]
base_url = "http://localhost:8000/v1"

[load]
total_requests = 1
concurrent_requests = 1

[input]
file = "examples/prompts/openorca-10000.jsonl"

[output]
format = "json"
"#;
        let config: Config = toml::from_str(toml).expect("toml parses");
        assert!(config.conversation.is_none());
        let json = serde_json::to_string(&config).expect("json serializes");
        assert!(
            !json.contains("conversation"),
            "serialized config should not mention the conversation field: {}",
            json
        );
        assert!(!json.contains("turn_delay"));
    }

    #[test]
    fn conversation_section_parses_with_defaults() {
        let toml = r#"
[endpoint]
base_url = "http://localhost:8000/v1"

[load]
total_requests = 1
concurrent_requests = 1

[input]
file = "examples/prompts/openorca-10000.jsonl"

[output]
format = "json"

[conversation]
turn_delay_ms = 1500
"#;
        let config: Config = toml::from_str(toml).expect("toml parses");
        let conv = config.conversation.expect("[conversation] present");
        assert_eq!(conv.turn_delay_ms, 1500);
        assert_eq!(conv.turn_delay_stdev_ms, 0);
        assert_eq!(conv.turn_delay_min_ms, 0);
        assert_eq!(conv.turn_delay_max_ms, 60_000);
    }

    #[test]
    fn conversation_min_max_inversion_is_rejected() {
        let toml = r#"
[endpoint]
base_url = "http://localhost:8000/v1"

[load]
total_requests = 1
concurrent_requests = 1

[input]
file = "examples/prompts/openorca-10000.jsonl"

[output]
format = "json"

[conversation]
turn_delay_min_ms = 5000
turn_delay_max_ms = 1000
"#;
        let config: Config = toml::from_str(toml).expect("toml parses");
        assert!(config.validate().is_err());
    }
}

#[cfg(test)]
mod server_metrics_config_tests {
    use super::*;

    #[test]
    fn derives_default_server_output_from_client_output() {
        let toml = r#"
            output = "run.parquet"
            interval = "1s"
            server_metrics_url = "http://localhost:4242/metrics"
        "#;
        let cfg: MetricsConfig = toml::from_str(toml).unwrap();
        assert_eq!(
            cfg.server_metrics_url.as_deref(),
            Some("http://localhost:4242/metrics")
        );
        assert_eq!(
            cfg.resolved_server_output(),
            std::path::PathBuf::from("run.server.parquet")
        );
    }

    #[test]
    fn explicit_server_output_overrides_default() {
        let toml = r#"
            output = "run.parquet"
            server_metrics_url = "http://x/metrics"
            server_metrics_output = "srv.parquet"
        "#;
        let cfg: MetricsConfig = toml::from_str(toml).unwrap();
        assert_eq!(
            cfg.resolved_server_output(),
            std::path::PathBuf::from("srv.parquet")
        );
    }

    #[test]
    fn rezolus_bin_defaults_to_none_and_parses() {
        let toml = r#"
            output = "run.parquet"
            server_metrics_url = "http://x/metrics"
        "#;
        let cfg: MetricsConfig = toml::from_str(toml).unwrap();
        assert!(cfg.rezolus_bin.is_none());

        let toml2 = r#"
            output = "run.parquet"
            rezolus_bin = "/opt/rezolus"
        "#;
        let cfg2: MetricsConfig = toml::from_str(toml2).unwrap();
        assert_eq!(cfg2.rezolus_bin.as_deref(), Some("/opt/rezolus"));
    }
}
