use anyhow::Result;
use llm_perf::cli::Command;
use llm_perf::{Cli, Config};
use log::{debug, info, warn};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

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
        } => llm_perf::kl_divergence::run_kl_divergence(
            baseline,
            candidate,
            format,
            output.as_deref(),
        ),
        ref cmd @ Command::MmluPro { .. } => run_mmlu_pro_mode(cmd),
    }
}

/// Set up non-blocking logging with tracing-subscriber and tracing-appender.
/// Returns the WorkerGuard which must be held alive for the duration of the program.
fn setup_logging(config: &Config) -> Result<WorkerGuard> {
    // Build env filter from configured level and per-module overrides
    let mut filter = EnvFilter::new(config.log.level.as_str());
    for directive in &config.log.filter {
        filter = filter.add_directive(directive.parse()?);
    }

    // Set up non-blocking writer to file or stderr
    let (non_blocking, guard) = if let Some(ref log_file) = config.output.trace_log {
        let file = std::fs::File::create(log_file)?;
        tracing_appender::non_blocking(file)
    } else {
        tracing_appender::non_blocking(std::io::stderr())
    };

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_writer(non_blocking))
        .init();

    Ok(guard)
}

fn run_bench_mode(config_path: &std::path::Path) -> Result<()> {
    // Load configuration first to check verbosity setting
    let config = Config::load(&config_path.to_path_buf())?;

    // Set up non-blocking logging
    let _guard = setup_logging(&config)?;

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
            llm_perf::admin::start_server(addr).await;
        });
    }

    debug!("Initializing benchmark runner");
    let runner = llm_perf::BenchmarkRunner::new(config).await?;
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

    // Set up non-blocking logging
    let _guard = setup_logging(&config)?;

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

fn run_mmlu_pro_mode(cmd: &Command) -> Result<()> {
    use llm_perf::mmlu_pro::{config, dataset, evaluate, report};
    use std::path::PathBuf;
    use std::time::Instant;

    let Command::MmluPro {
        config: ref config_path,
        ref url,
        ref api_key,
        ref model,
        timeout,
        ref category,
        subset,
        concurrent_requests,
        num_shots,
        verbosity,
        log_prompt,
        ref comment,
    } = *cmd
    else {
        unreachable!()
    };

    let mut config = config::Config::load(config_path)?;

    // Apply CLI overrides
    if let Some(url) = url {
        config.endpoint.base_url = url.clone();
    }
    if let Some(api_key) = api_key {
        config.endpoint.api_key = Some(api_key.clone());
    }
    if let Some(model) = model {
        config.endpoint.model = Some(model.clone());
    }
    if let Some(timeout) = timeout {
        config.endpoint.timeout = timeout;
    }
    if let Some(category) = category {
        config.load.categories = vec![category.clone()];
    }
    if let Some(subset) = subset {
        config.load.subset = subset;
    }
    if let Some(concurrent_requests) = concurrent_requests {
        config.load.concurrent_requests = concurrent_requests;
    }
    if let Some(num_shots) = num_shots {
        config.inference.num_shots = num_shots;
    }
    if let Some(verbosity) = verbosity {
        config.log.verbosity = verbosity;
    }
    if log_prompt {
        config.log.log_prompt = true;
    }
    if let Some(comment) = comment {
        config.comment = comment.clone();
    }

    // Create output directory
    let output_dir = PathBuf::from("eval_results");
    std::fs::create_dir_all(&output_dir)?;

    // Build tokio runtime
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let start = Instant::now();

    // Run evaluation
    let result = runtime.block_on(async {
        // Detect model if not specified
        let model = if let Some(model) = config.endpoint.model.clone() {
            model
        } else {
            eprintln!("Model not specified, querying server...");
            llm_perf::client::detect_model(
                &config.endpoint.base_url,
                config.endpoint.api_key.as_deref(),
                std::time::Duration::from_secs(config.endpoint.timeout),
            )
            .await?
        };

        // Print startup info
        eprintln!("MMLU-Pro Benchmark");
        eprintln!("  Model: {}", model);
        eprintln!("  URL: {}", config.endpoint.base_url);
        eprintln!("  Concurrent Requests: {}", config.load.concurrent_requests);
        eprintln!("  Subset: {}", config.load.subset);
        eprintln!("  Shots: {}", config.inference.num_shots);
        eprintln!("  Max Tokens: {}", config.inference.max_tokens);
        eprintln!("  Temperature: {}", config.inference.temperature);
        eprintln!("  Presence Penalty: {}", config.inference.presence_penalty);
        eprintln!(
            "  Frequency Penalty: {}",
            config.inference.frequency_penalty
        );
        eprintln!();

        eprintln!("Loading MMLU-Pro dataset...");
        let (test_data, val_data) = dataset::load_mmlu_pro(config.load.subset).await?;

        eprintln!(
            "Dataset loaded: {} categories, {} total test questions",
            test_data.len(),
            test_data.values().map(|v| v.len()).sum::<usize>()
        );

        let eval_result =
            evaluate::run_evaluation(&config, &model, &test_data, &val_data, &output_dir).await?;
        Ok::<_, anyhow::Error>((model, eval_result))
    })?;

    let elapsed = start.elapsed();
    let (model, result) = result;

    // Generate report
    report::generate_report(
        &config,
        &model,
        &result.category_stats,
        &result.token_stats,
        elapsed,
        &output_dir,
    );

    Ok(())
}

async fn run_logprobs_collection(
    config: Config,
    lp_config: llm_perf::config::LogprobsConfig,
) -> Result<()> {
    use llm_perf::benchmark::Prompt;
    use llm_perf::logprobs::{LogprobRecord, LogprobWriter};
    use tokio::io::AsyncBufReadExt;

    // Wait for server readiness if configured
    if config.endpoint.health_check_timeout > 0 {
        llm_perf::client::check_server_ready(
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
        llm_perf::client::detect_model(
            &config.endpoint.base_url,
            config.endpoint.api_key.as_deref(),
            std::time::Duration::from_secs(config.endpoint.timeout),
        )
        .await?
    };

    // Create client with pool_size=1 (sequential)
    let client = llm_perf::OpenAIClient::new(llm_perf::ClientConfig {
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
