use anyhow::Result;
use llm_bench::{Cli, Config};
use log::{LevelFilter, Metadata, Record, debug, info};
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

    // Load configuration first to check verbosity setting
    let config = Config::load(&cli.config)?;

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
        println!("   Config: {}", cli.config.display());
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
