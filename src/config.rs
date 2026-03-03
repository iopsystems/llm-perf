use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointConfig {
    pub base_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>, // If not provided, will auto-detect from server
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default)]
    pub max_retries: u32,
    #[serde(default = "default_retry_initial_delay_ms")]
    pub retry_initial_delay_ms: u64,
    #[serde(default = "default_retry_max_delay_ms")]
    pub retry_max_delay_ms: u64,
    #[serde(default = "default_health_check_timeout")]
    pub health_check_timeout: u64, // Total time to wait for server readiness in seconds (0 = disabled)
    #[serde(default = "default_health_check_interval")]
    pub health_check_interval: u64, // Interval between readiness check retries in seconds
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ArrivalDistribution {
    #[default]
    Uniform, // Fixed intervals (deterministic)
    Poisson, // Exponential inter-arrival times (stochastic)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputConfig {
    pub file: PathBuf,
    #[serde(default)]
    pub shuffle: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_size: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
pub struct RuntimeConfig {
    #[serde(default = "default_worker_threads")]
    pub worker_threads: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub fn to_level_filter(&self) -> log::LevelFilter {
        match self {
            LogLevel::Error => log::LevelFilter::Error,
            LogLevel::Warn => log::LevelFilter::Warn,
            LogLevel::Info => log::LevelFilter::Info,
            LogLevel::Debug => log::LevelFilter::Debug,
            LogLevel::Trace => log::LevelFilter::Trace,
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
pub struct MetricsConfig {
    /// File for parquet metrics output
    pub output: PathBuf,
    /// The snapshot interval (e.g., "1s", "500ms")
    #[serde(default = "default_metrics_interval")]
    pub interval: String,
    /// Batch size for parquet files
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminConfig {
    #[serde(default = "default_admin_listen")]
    pub listen: String,
    #[serde(default = "default_admin_enabled")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

fn default_timeout() -> u64 {
    60
}

fn default_retry_initial_delay_ms() -> u64 {
    100
}

fn default_retry_max_delay_ms() -> u64 {
    10000 // 10 seconds
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

        Ok(())
    }
}
