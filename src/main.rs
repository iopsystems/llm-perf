use anyhow::Result;
use llm_bench::cli::Command;
use llm_bench::{Cli, Config};
use log::{LevelFilter, Metadata, Record, debug, info, warn};
use ringlog::{File, LogBuilder, MultiLogBuilder, Output, Stderr};
use std::collections::HashMap;
use std::io::Write;
use std::sync::Mutex;

/// Maximum log file size before rotation (10MB)
const LOG_FILE_MAX_SIZE: u64 = 1024 * 1024 * 10;

/// Parse log filter strings like "hyper=info" into a map of module prefix to level filter
fn parse_log_filters(filters: &[String]) -> HashMap<String, LevelFilter> {
    let mut map = HashMap::new();
    for filter in filters {
        if let Some((module, level)) = filter.split_once('=') {
            let level_filter = match level.to_lowercase().as_str() {
                "error" => LevelFilter::Error,
                "warn" => LevelFilter::Warn,
                "info" => LevelFilter::Info,
                "debug" => LevelFilter::Debug,
                "trace" => LevelFilter::Trace,
                "off" => LevelFilter::Off,
                _ => continue,
            };
            map.insert(module.to_string(), level_filter);
        }
    }
    map
}

/// Check if a log record should be filtered based on per-module filters
fn should_log(metadata: &Metadata, filters: &HashMap<String, LevelFilter>) -> bool {
    let target = metadata.target();

    // Check each filter to see if it matches this target
    for (module_prefix, level_filter) in filters {
        if target.starts_with(module_prefix) {
            return metadata.level() <= *level_filter;
        }
    }

    // If no filter matched, allow the log (will be caught by global level filter)
    true
}

/// Custom logger with per-module filtering that wraps ringlog
struct FilteredLogger {
    output: Mutex<Box<dyn Output>>,
    max_level: LevelFilter,
    filters: HashMap<String, LevelFilter>,
}

impl log::Log for FilteredLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.max_level && should_log(metadata, &self.filters)
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata())
            && let Ok(mut output) = self.output.lock()
        {
            let message = format!("{}\n", record.args());
            let _ = output.write_all(message.as_bytes());
        }
    }

    fn flush(&self) {
        if let Ok(mut output) = self.output.lock() {
            let _ = output.flush();
        }
    }
}

fn main() -> Result<()> {
    // Parse CLI arguments
    let cli = Cli::parse_args();

    match cli.command {
        Command::Bench { ref config } => run_bench_mode(config),
        Command::Logprobs { ref config } => run_logprobs_mode(config),
        Command::KlDivergence {
            ref baseline,
            ref candidate,
            ref format,
            ref output,
        } => llm_bench::kl_divergence::run_kl_divergence(
            baseline,
            candidate,
            format,
            output.as_deref(),
        ),
    }
}

fn run_bench_mode(config_path: &std::path::Path) -> Result<()> {
    // Load configuration first to check verbosity setting
    let config = Config::load(&config_path.to_path_buf())?;

    // Set up logging with ringlog and per-module filtering
    let log_level = config.log.level.to_level_filter();

    // Configure output destination
    let output: Box<dyn Output> = if let Some(ref log_file) = config.output.trace_log {
        // Log to file with rotation
        let backup_file = log_file.with_extension("old");
        Box::new(File::new(log_file.clone(), backup_file, LOG_FILE_MAX_SIZE)?)
    } else {
        // Log to stderr
        Box::new(Stderr::new())
    };

    // Parse per-module filters from config
    let filters = parse_log_filters(&config.log.filter);

    // Create logger with per-module filtering if configured
    if filters.is_empty() {
        // No filters - use ringlog directly
        let base_log = LogBuilder::new()
            .output(output)
            .build()
            .expect("failed to initialize logger");

        let _drain = MultiLogBuilder::new()
            .level_filter(log_level)
            .default(base_log)
            .build()
            .start();
    } else {
        // Use custom filtered logger
        let logger = FilteredLogger {
            output: Mutex::new(output),
            max_level: log_level,
            filters,
        };

        log::set_boxed_logger(Box::new(logger)).expect("failed to set logger");
        log::set_max_level(log_level);
    }

    // Print clean startup message
    if !config.output.quiet {
        println!("LLM Benchmark Tool");
        println!("   Config: {}", config_path.display());
        println!("   Target: {}", config.endpoint.base_url);

        if let Some(qps) = config.load.qps {
            println!("   Mode: Fixed QPS ({:.1} req/s)", qps);
        } else {
            println!(
                "   Mode: Concurrent ({} workers)",
                config.load.concurrent_requests
            );
        }

        if let Some(total) = config.load.total_requests {
            println!("   Requests: {}", total);
        } else if let Some(duration) = config.load.duration_seconds {
            println!("   Duration: {}s", duration);
        }
        println!();
    }

    // Build custom tokio runtime with specified worker threads
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(config.runtime.worker_threads)
        .enable_all()
        .build()?;

    // Run the benchmark
    runtime.block_on(async { run_benchmark(config).await })
}

async fn run_benchmark(config: Config) -> Result<()> {
    // Start admin server if configured
    if let Some(ref admin_config) = config.admin
        && admin_config.enabled
    {
        let addr: std::net::SocketAddr = admin_config
            .listen
            .parse()
            .expect("Invalid admin listen address");

        info!("Starting metrics server on {}", addr);
        tokio::spawn(async move {
            llm_bench::admin::start_server(addr).await;
        });
    }

    debug!("Initializing benchmark runner");
    let runner = llm_bench::BenchmarkRunner::new(config).await?;
    info!("Starting benchmark run");
    runner.run().await?;
    info!("Benchmark completed successfully");
    Ok(())
}

fn run_logprobs_mode(config_path: &std::path::Path) -> Result<()> {
    let config = Config::load(&config_path.to_path_buf())?;

    // Require [logprobs] section
    let lp_config = config
        .logprobs
        .as_ref()
        .filter(|lp| lp.enabled)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "The logprobs subcommand requires a [logprobs] section with enabled = true in the config"
            )
        })?
        .clone();

    // Set up logging (reuse bench mode pattern)
    let log_level = config.log.level.to_level_filter();

    let output: Box<dyn Output> = if let Some(ref log_file) = config.output.trace_log {
        let backup_file = log_file.with_extension("old");
        Box::new(File::new(log_file.clone(), backup_file, LOG_FILE_MAX_SIZE)?)
    } else {
        Box::new(Stderr::new())
    };

    let filters = parse_log_filters(&config.log.filter);

    if filters.is_empty() {
        let base_log = LogBuilder::new()
            .output(output)
            .build()
            .expect("failed to initialize logger");

        let _drain = MultiLogBuilder::new()
            .level_filter(log_level)
            .default(base_log)
            .build()
            .start();
    } else {
        let logger = FilteredLogger {
            output: Mutex::new(output),
            max_level: log_level,
            filters,
        };

        log::set_boxed_logger(Box::new(logger)).expect("failed to set logger");
        log::set_max_level(log_level);
    }

    if !config.output.quiet {
        println!("LLM Logprobs Collection");
        println!("   Config: {}", config_path.display());
        println!("   Target: {}", config.endpoint.base_url);
        println!("   Top logprobs: {}", lp_config.top_logprobs);
        println!("   Output: {}", lp_config.output.display());
        println!();
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    runtime.block_on(run_logprobs_collection(config, lp_config))
}

async fn run_logprobs_collection(
    config: Config,
    lp_config: llm_bench::config::LogprobsConfig,
) -> Result<()> {
    use llm_bench::benchmark::Prompt;
    use llm_bench::logprobs::{LogprobRecord, LogprobWriter};
    use tokio::io::AsyncBufReadExt;

    // Wait for server readiness if configured
    if config.endpoint.health_check_timeout > 0 {
        llm_bench::client::check_server_ready(
            &config.endpoint.base_url,
            config.endpoint.api_key.as_deref(),
            std::time::Duration::from_secs(config.endpoint.health_check_timeout),
            std::time::Duration::from_secs(config.endpoint.health_check_interval),
        )
        .await?;
    }

    // Detect model
    let model = if let Some(model) = config.endpoint.model.clone() {
        model
    } else {
        info!("Model not specified, querying server for available models");
        llm_bench::client::detect_model(
            &config.endpoint.base_url,
            config.endpoint.api_key.as_deref(),
            std::time::Duration::from_secs(config.endpoint.timeout),
        )
        .await?
    };

    // Create client with pool_size=1 (sequential)
    let client = llm_bench::OpenAIClient::new(llm_bench::ClientConfig {
        base_url: config.endpoint.base_url.clone(),
        api_key: config.endpoint.api_key.clone(),
        model,
        timeout: std::time::Duration::from_secs(config.endpoint.timeout),
        max_retries: config.endpoint.max_retries,
        retry_initial_delay_ms: config.endpoint.retry_initial_delay_ms,
        retry_max_delay_ms: config.endpoint.retry_max_delay_ms,
        pool_size: 1,
    })?;

    // Load prompts
    let file = tokio::fs::File::open(&config.input.file).await?;
    let reader = tokio::io::BufReader::new(file);
    let mut lines = reader.lines();
    let mut prompts: Vec<Prompt> = Vec::new();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Prompt>(&line) {
            Ok(prompt) => prompts.push(prompt),
            Err(e) => warn!("Failed to parse prompt line: {}", e),
        }
    }

    if let Some(sample_size) = config.input.sample_size {
        prompts.truncate(sample_size);
    }

    info!("Loaded {} prompts", prompts.len());

    // Set up writer channel
    let (tx, writer) = LogprobWriter::new(lp_config.output.clone(), 256);
    let writer_handle = tokio::spawn(async move {
        if let Err(e) = writer.run().await {
            log::error!("Logprob writer error: {}", e);
        }
    });

    // Process prompts sequentially
    let total = prompts.len();
    for (idx, prompt) in prompts.iter().enumerate() {
        debug!("Processing prompt {}/{}", idx + 1, total);

        let request = client.create_request(
            &prompt.prompt,
            prompt.max_tokens,
            Some(true),
            Some(lp_config.top_logprobs),
        );

        match client.chat_completion_stream(request).await {
            Ok(mut stream) => {
                // Consume the stream
                while let Some(_chunk) = stream.next_chunk().await? {}

                let collected = stream.logprobs();
                if !collected.is_empty() {
                    let record = LogprobRecord {
                        prompt_index: idx,
                        prompt: prompt.prompt.clone(),
                        tokens: collected.to_vec(),
                    };
                    if let Err(e) = tx.send(record).await {
                        warn!("Failed to send logprob record: {}", e);
                    }
                } else {
                    warn!("No logprobs returned for prompt {}", idx);
                }
            }
            Err(e) => {
                warn!("Request failed for prompt {}: {}", idx, e);
            }
        }

        if !config.output.quiet && (idx + 1) % 10 == 0 {
            println!("   Progress: {}/{} prompts", idx + 1, total);
        }
    }

    // Close sender to signal writer to finish
    drop(tx);
    let _ = writer_handle.await;

    if !config.output.quiet {
        println!();
        println!("Logprobs collection complete");
        println!("   Prompts processed: {}", total);
        println!("   Output: {}", lp_config.output.display());
    }

    Ok(())
}
