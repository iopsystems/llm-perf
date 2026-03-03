use anyhow::Result;
use log::{debug, info, warn};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::Deserialize;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::Semaphore;
use tokio::time::{sleep, timeout};

use crate::client::{ClientError, OpenAIClient};
use crate::config::Config;
use crate::distribution::RequestDistribution;
use crate::metrics::{ErrorType, Metrics, RequestStatus};
use crate::report::ReportBuilder;
use crate::tokenizer::Tokenizer;

/// A prompt to be sent to the LLM server.
///
/// Prompts are loaded from JSONL files where each line contains a JSON object
/// with a "prompt" field and an optional "max_tokens" field.
#[derive(Debug, Clone, Deserialize)]
pub struct Prompt {
    /// The text prompt to send to the LLM
    pub prompt: String,
    /// Maximum number of tokens to generate in the response (optional)
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

/// Core benchmarking engine for testing OpenAI-compatible LLM servers.
///
/// The BenchmarkRunner orchestrates the entire benchmarking process including:
/// - Model detection and validation
/// - Prompt loading and shuffling
/// - Concurrent or QPS-based request execution
/// - Metrics collection (TTFT, ITL, throughput, errors)
/// - Report generation in multiple formats
///
/// # Examples
///
/// ```no_run
/// use llm_bench::{BenchmarkRunner, Config};
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let config = Config::load(&"config.toml".into())?;
///     let runner = BenchmarkRunner::new(config).await?;
///     runner.run().await?;
///     Ok(())
/// }
/// ```
pub struct BenchmarkRunner {
    client: Arc<OpenAIClient>,
    config: Config,
    prompts: Vec<Prompt>,
    tokenizer: Arc<Tokenizer>,
}

impl BenchmarkRunner {
    /// Creates a new BenchmarkRunner with the given configuration.
    ///
    /// This performs several initialization steps:
    /// - Auto-detects the model from the server if not specified
    /// - Creates an HTTP client with connection pooling and retry logic
    /// - Loads prompts from the input file
    /// - Applies sampling and shuffling if configured
    ///
    /// # Arguments
    ///
    /// * `config` - The benchmark configuration loaded from a TOML file
    ///
    /// # Returns
    ///
    /// Returns a configured BenchmarkRunner ready to execute tests, or an error if:
    /// - The server is unreachable
    /// - Model detection fails
    /// - Prompt file cannot be read or parsed
    /// - Tokenizer initialization fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use llm_bench::{BenchmarkRunner, Config};
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = Config::load(&"config.toml".into())?;
    /// let runner = BenchmarkRunner::new(config).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new(mut config: Config) -> Result<Self> {
        // Initialize metrics
        Metrics::init();

        // Wait for server to be ready if timeout is set (> 0)
        // This is optional and useful when starting servers that need time to load models
        if config.endpoint.health_check_timeout > 0 {
            crate::client::check_server_ready(
                &config.endpoint.base_url,
                config.endpoint.api_key.as_deref(),
                Duration::from_secs(config.endpoint.health_check_timeout),
                Duration::from_secs(config.endpoint.health_check_interval),
            )
            .await?;
        }

        // Detect model from server if not provided
        let model = if let Some(model) = config.endpoint.model.clone() {
            model
        } else {
            info!("Model not specified, querying server for available models");
            let detected = crate::client::detect_model(
                &config.endpoint.base_url,
                config.endpoint.api_key.as_deref(),
                Duration::from_secs(config.endpoint.timeout),
            )
            .await?;
            // Store the detected model in config
            config.endpoint.model = Some(detected.clone());
            detected
        };

        // Create OpenAI client with retry configuration and connection pool size
        let client = OpenAIClient::new(crate::client::ClientConfig {
            base_url: config.endpoint.base_url.clone(),
            api_key: config.endpoint.api_key.clone(),
            model: model.clone(),
            timeout: Duration::from_secs(config.endpoint.timeout),
            max_retries: config.endpoint.max_retries,
            retry_initial_delay_ms: config.endpoint.retry_initial_delay_ms,
            retry_max_delay_ms: config.endpoint.retry_max_delay_ms,
            pool_size: config.load.concurrent_requests, // Pool size matches concurrency
        })?;

        // Create tokenizer
        let tokenizer = Tokenizer::new(&model)?;

        // Load prompts
        let prompts = Self::load_prompts(&config.input.file).await?;
        let mut prompts: Vec<Prompt> = if let Some(sample_size) = config.input.sample_size {
            prompts.into_iter().take(sample_size).collect()
        } else {
            prompts
        };

        // Shuffle prompts if requested
        if config.input.shuffle {
            let mut rng = thread_rng();
            prompts.shuffle(&mut rng);
            info!("Shuffled {} prompts", prompts.len());
        }

        debug!("Loaded {} prompts", prompts.len());

        Ok(Self {
            client: Arc::new(client),
            config,
            prompts,
            tokenizer: Arc::new(tokenizer),
        })
    }

    async fn load_prompts(path: &Path) -> Result<Vec<Prompt>> {
        let file = File::open(path).await?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        let mut prompts = Vec::new();

        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<Prompt>(&line) {
                Ok(prompt) => prompts.push(prompt),
                Err(e) => warn!("Failed to parse prompt line: {}", e),
            }
        }

        Ok(prompts)
    }

    /// Executes the benchmark test and generates a performance report.
    ///
    /// This method orchestrates the entire benchmark execution:
    /// - Spawns background tasks for periodic stats and metrics capture (if enabled)
    /// - Chooses between concurrent or QPS-based execution based on configuration
    /// - Handles warmup phase (excluded from metrics)
    /// - Tracks all metrics (TTFT, ITL, throughput, errors)
    /// - Generates and outputs the final report
    ///
    /// The test will run until either:
    /// - The configured number of requests is sent (`total_requests`), or
    /// - The configured duration elapses (`duration_seconds`)
    ///
    /// # Returns
    ///
    /// Returns Ok(()) on successful completion, or an error if:
    /// - Server becomes unreachable during the test
    /// - Report generation fails
    /// - File I/O errors occur (for JSON output)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use llm_bench::{BenchmarkRunner, Config};
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = Config::load(&"config.toml".into())?;
    /// let runner = BenchmarkRunner::new(config).await?;
    ///
    /// // Run the benchmark and generate report
    /// runner.run().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn run(&self) -> Result<()> {
        let report_builder = ReportBuilder::new().with_config(self.config.clone());
        let start_instant = Instant::now();

        debug!("Starting benchmark run");

        // Set running flag
        crate::metrics::RUNNING.store(true, std::sync::atomic::Ordering::Relaxed);

        // Create notification for warmup completion
        let warmup_complete = Arc::new(tokio::sync::Notify::new());

        // Spawn periodic stats output task (unless in quiet mode or JSON to stdout)
        let json_to_stdout = matches!(self.config.output.format, crate::config::OutputFormat::Json)
            && self.config.output.file.is_none();
        let stats_handle = if !self.config.output.quiet && !json_to_stdout {
            let config = self.config.clone();
            let warmup_notify = Arc::clone(&warmup_complete);
            Some(tokio::spawn(async move {
                crate::stats::periodic_stats(config, warmup_notify).await;
            }))
        } else {
            None
        };

        // Spawn snapshot task if metrics are configured
        let snapshot_handle = if self.config.metrics.is_some() {
            let config = self.config.clone();
            Some(tokio::spawn(async move {
                if let Err(e) = crate::snapshot::capture_snapshots(config).await {
                    log::error!("Snapshot capture error: {}", e);
                }
            }))
        } else {
            None
        };

        // Run benchmark (without generating report)
        // If qps is set, use fixed QPS mode; otherwise use concurrent mode
        let test_duration = if self.config.load.qps.is_some() {
            self.run_qps_mode_internal(start_instant, warmup_complete)
                .await?
        } else {
            self.run_concurrent_mode_internal(start_instant, warmup_complete)
                .await?
        };

        // Stop metrics capture
        crate::metrics::RUNNING.store(false, std::sync::atomic::Ordering::Relaxed);

        // Wait for background tasks to complete
        if let Some(handle) = stats_handle {
            let _ = handle.await;
        }
        if let Some(handle) = snapshot_handle {
            let _ = handle.await;
        }

        // Add a small delay to ensure all output is flushed
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Generate report after all background tasks are done (use actual test duration)
        let report_builder = report_builder.with_duration(test_duration);
        self.generate_report(report_builder).await
    }

    async fn run_concurrent_mode_internal(
        &self,
        start_instant: Instant,
        warmup_complete: Arc<tokio::sync::Notify>,
    ) -> Result<Duration> {
        info!(
            "Running in concurrent mode with {} workers",
            self.config.load.concurrent_requests
        );

        // Create semaphore for concurrency control
        let semaphore = Arc::new(Semaphore::new(self.config.load.concurrent_requests));

        // Determine test duration or request count
        let (total_requests, duration_limit) = if let Some(total) = self.config.load.total_requests
        {
            (Some(total), None)
        } else if let Some(duration_secs) = self.config.load.duration_seconds {
            (None, Some(Duration::from_secs(duration_secs)))
        } else {
            anyhow::bail!("Either total_requests or duration_seconds must be specified");
        };

        // Calculate warmup parameters
        let warmup_count = self.config.load.warmup_requests.unwrap_or(0);
        let warmup_duration = self.config.load.warmup_duration.map(Duration::from_secs);

        // Log warmup info
        if warmup_count > 0 {
            info!("Starting warmup phase: {} requests", warmup_count);
        } else if let Some(duration) = warmup_duration {
            info!("Starting warmup phase: {} seconds", duration.as_secs());
        } else {
            // No warmup - signal immediately
            warmup_complete.notify_one();
        }

        let completed = Arc::new(AtomicUsize::new(0));
        let should_stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        let prompt_index = Arc::new(AtomicUsize::new(0));
        let warmup_completed = Arc::new(AtomicUsize::new(0));

        // Track when main test starts (initialize now if no warmup, otherwise set after warmup)
        let mut test_start = Instant::now();

        if let Some(total) = total_requests {
            // Fixed request count mode

            // Run warmup requests first
            for _ in 0..warmup_count {
                let idx = prompt_index.fetch_add(1, Ordering::Relaxed);
                let prompt = self.prompts[idx % self.prompts.len()].clone();
                let client = Arc::clone(&self.client);
                let tokenizer = Arc::clone(&self.tokenizer);
                let semaphore = Arc::clone(&semaphore);
                let warmup_completed = Arc::clone(&warmup_completed);

                let handle = tokio::spawn(async move {
                    let _permit = semaphore
                        .acquire()
                        .await
                        .expect("semaphore should never be closed");
                    let result = Self::execute_request(client, tokenizer, prompt, idx, true).await;
                    warmup_completed.fetch_add(1, Ordering::Relaxed);
                    result
                });
                handles.push(handle);
            }

            // Wait for warmup to complete before starting main test
            if warmup_count > 0 {
                for handle in handles.drain(..) {
                    let _ = handle.await?;
                }
                info!("Warmup complete, starting main test");
                // Signal stats task to start intervals
                warmup_complete.notify_one();
                // Reset test_start after warmup
                test_start = Instant::now();
            }

            // Run main test requests
            for _ in 0..total {
                let idx = prompt_index.fetch_add(1, Ordering::Relaxed);
                let prompt = self.prompts[idx % self.prompts.len()].clone();
                let client = Arc::clone(&self.client);
                let tokenizer = Arc::clone(&self.tokenizer);
                let semaphore = Arc::clone(&semaphore);
                let completed = Arc::clone(&completed);

                let handle = tokio::spawn(async move {
                    let _permit = semaphore
                        .acquire()
                        .await
                        .expect("semaphore should never be closed");
                    let result = Self::execute_request(client, tokenizer, prompt, idx, false).await;
                    completed.fetch_add(1, Ordering::Relaxed);
                    result
                });
                handles.push(handle);
            }
        } else if let Some(duration) = duration_limit {
            // Duration-based mode

            // Handle warmup phase first if configured
            if let Some(warmup_dur) = warmup_duration {
                info!("Starting warmup phase for {} seconds", warmup_dur.as_secs());
                let warmup_deadline = Instant::now() + warmup_dur;

                // Spawn warmup workers
                let mut warmup_handles = Vec::new();
                for _worker_id in 0..self.config.load.concurrent_requests {
                    let client = Arc::clone(&self.client);
                    let tokenizer = Arc::clone(&self.tokenizer);
                    let semaphore = Arc::clone(&semaphore);
                    let warmup_completed = Arc::clone(&warmup_completed);
                    let prompt_index = Arc::clone(&prompt_index);
                    let prompts = self.prompts.clone();

                    let handle = tokio::spawn(async move {
                        while Instant::now() < warmup_deadline {
                            let _permit = match semaphore.try_acquire() {
                                Ok(permit) => permit,
                                Err(_) => break,
                            };

                            let idx = prompt_index.fetch_add(1, Ordering::Relaxed);
                            let prompt = prompts[idx % prompts.len()].clone();

                            let _ = Self::execute_request(
                                client.clone(),
                                tokenizer.clone(),
                                prompt,
                                idx,
                                true,
                            )
                            .await;
                            warmup_completed.fetch_add(1, Ordering::Relaxed);
                        }
                        Ok::<(), anyhow::Error>(())
                    });
                    warmup_handles.push(handle);
                }

                // Wait for warmup to complete
                sleep(warmup_dur).await;
                for handle in warmup_handles {
                    let _ = handle.await?;
                }

                info!("Warmup complete, starting main test");
                // Signal stats task to start intervals
                warmup_complete.notify_one();
                // Reset test_start after warmup
                test_start = Instant::now();
            }

            // Main test phase
            let deadline = test_start + duration;
            info!("Running main test for {} seconds", duration.as_secs());

            // Spawn worker tasks
            for _worker_id in 0..self.config.load.concurrent_requests {
                let client = Arc::clone(&self.client);
                let tokenizer = Arc::clone(&self.tokenizer);
                let semaphore = Arc::clone(&semaphore);
                let completed = Arc::clone(&completed);
                let should_stop = Arc::clone(&should_stop);
                let prompt_index = Arc::clone(&prompt_index);
                let prompts = self.prompts.clone();

                let handle = tokio::spawn(async move {
                    while !should_stop.load(Ordering::Relaxed) {
                        // Check deadline before acquiring permit
                        if Instant::now() >= deadline {
                            should_stop.store(true, Ordering::Relaxed);
                            break;
                        }

                        let _permit = semaphore
                            .acquire()
                            .await
                            .expect("semaphore should never be closed");

                        // Check deadline again after acquiring permit
                        if Instant::now() >= deadline {
                            should_stop.store(true, Ordering::Relaxed);
                            break;
                        }

                        let idx = prompt_index.fetch_add(1, Ordering::Relaxed);
                        let prompt = prompts[idx % prompts.len()].clone();

                        // Calculate remaining time until deadline
                        let remaining = deadline.saturating_duration_since(Instant::now());
                        if remaining.is_zero() {
                            break;
                        }

                        // Execute request with timeout based on remaining time
                        let request_future = Self::execute_request(
                            client.clone(),
                            tokenizer.clone(),
                            prompt,
                            idx,
                            false,
                        );
                        match tokio::time::timeout(remaining, request_future).await {
                            Ok(_) => {
                                // Request completed normally
                                completed.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(_) => {
                                // Request timed out due to test ending - don't count as failure
                                debug!("Request {} cancelled due to test duration limit", idx);
                            }
                        }
                    }
                    Ok::<(), anyhow::Error>(())
                });
                handles.push(handle);
            }

            // Wait for duration to complete
            sleep(duration).await;
            should_stop.store(true, Ordering::Relaxed);
        }

        // Wait for all tasks to complete with a grace period
        let grace_period = Duration::from_secs(60);
        for handle in handles {
            match tokio::time::timeout(grace_period, handle).await {
                Ok(result) => {
                    let _ = result?;
                }
                Err(_) => {
                    debug!("Worker task did not complete within grace period");
                }
            }
        }

        // Calculate actual test duration (excluding warmup)
        let test_duration = test_start.elapsed();
        let total_duration = start_instant.elapsed();

        info!(
            "Benchmark completed in {:.1}s total ({:.1}s test)",
            total_duration.as_secs_f64(),
            test_duration.as_secs_f64()
        );

        // Log warmup summary if applicable
        if warmup_count > 0 || warmup_duration.is_some() {
            let warmup_total = warmup_completed.load(Ordering::Relaxed);
            info!("Warmup: {} requests (excluded from metrics)", warmup_total);
            info!("Main test: {} requests", completed.load(Ordering::Relaxed));
        }

        Ok(test_duration)
    }

    async fn run_qps_mode_internal(
        &self,
        start_instant: Instant,
        warmup_complete: Arc<tokio::sync::Notify>,
    ) -> Result<Duration> {
        let qps = self
            .config
            .load
            .qps
            .expect("QPS must be specified for fixed_qps mode");

        // Create request distribution
        let distribution = RequestDistribution::new(&self.config.load.arrival_distribution, qps);

        info!(
            "Running in fixed QPS mode: {} requests/second, {} distribution, max {} in-flight",
            qps,
            distribution.distribution_name(),
            self.config.load.concurrent_requests
        );

        // Calculate warmup parameters
        let warmup_count = self.config.load.warmup_requests.unwrap_or(0);
        let warmup_duration = self.config.load.warmup_duration.map(Duration::from_secs);

        // Log warmup info
        if warmup_count > 0 {
            info!("Starting warmup phase: {} requests", warmup_count);
        } else if let Some(duration) = warmup_duration {
            info!("Starting warmup phase: {} seconds", duration.as_secs());
        } else {
            // No warmup - signal immediately
            warmup_complete.notify_one();
        }

        // Determine test duration or request count
        let (total_requests, duration_limit) = if let Some(total) = self.config.load.total_requests
        {
            (Some(total), None)
        } else if let Some(duration_secs) = self.config.load.duration_seconds {
            (None, Some(Duration::from_secs(duration_secs)))
        } else {
            anyhow::bail!("Either total_requests or duration_seconds must be specified");
        };

        let completed = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        let prompt_index = Arc::new(AtomicUsize::new(0));
        let warmup_completed = Arc::new(AtomicUsize::new(0));

        // Maximum concurrent requests in QPS mode (configurable via concurrent_requests)
        let max_concurrent = self.config.load.concurrent_requests;
        let semaphore = Arc::new(Semaphore::new(max_concurrent));

        // Track when main test starts (initialize now if no warmup, otherwise set after warmup)
        let mut test_start = Instant::now();

        // Run warmup phase if configured
        if warmup_count > 0 {
            for _ in 0..warmup_count {
                tokio::time::sleep(distribution.next_delay()).await;

                let idx = prompt_index.fetch_add(1, Ordering::Relaxed);
                let prompt = self.prompts[idx % self.prompts.len()].clone();
                let client = Arc::clone(&self.client);
                let tokenizer = Arc::clone(&self.tokenizer);
                let semaphore = Arc::clone(&semaphore);
                let warmup_completed = Arc::clone(&warmup_completed);

                let handle = tokio::spawn(async move {
                    let _permit = semaphore
                        .acquire()
                        .await
                        .expect("semaphore should never be closed");
                    let result = Self::execute_request(client, tokenizer, prompt, idx, true).await;
                    warmup_completed.fetch_add(1, Ordering::Relaxed);
                    result
                });
                handles.push(handle);
            }

            // Wait for warmup requests to complete
            for handle in handles.drain(..) {
                let _ = handle.await?;
            }

            info!("Warmup complete, starting main test");
            // Signal stats task to start intervals
            warmup_complete.notify_one();
            // Reset test_start after warmup
            test_start = Instant::now();
        } else if let Some(warmup_dur) = warmup_duration {
            let warmup_deadline = Instant::now() + warmup_dur;

            while Instant::now() < warmup_deadline {
                tokio::time::sleep(distribution.next_delay()).await;

                let idx = prompt_index.fetch_add(1, Ordering::Relaxed);
                let prompt = self.prompts[idx % self.prompts.len()].clone();
                let client = Arc::clone(&self.client);
                let tokenizer = Arc::clone(&self.tokenizer);
                let semaphore = Arc::clone(&semaphore);
                let warmup_completed = Arc::clone(&warmup_completed);

                let handle = tokio::spawn(async move {
                    let _permit = semaphore
                        .acquire()
                        .await
                        .expect("semaphore should never be closed");
                    let result = Self::execute_request(client, tokenizer, prompt, idx, true).await;
                    warmup_completed.fetch_add(1, Ordering::Relaxed);
                    result
                });
                handles.push(handle);
            }

            // Wait for warmup requests to complete
            for handle in handles.drain(..) {
                let _ = handle.await?;
            }

            info!("Warmup complete, starting main test");
            // Signal stats task to start intervals
            warmup_complete.notify_one();
            // Reset test_start after warmup
            test_start = Instant::now();
        }

        // Calculate deadline for main test
        let deadline = duration_limit.map(|d| test_start + d);

        // Main test loop
        loop {
            // Check termination conditions
            if let Some(total) = total_requests
                && prompt_index.load(Ordering::Relaxed) >= total
            {
                break;
            }
            if let Some(deadline) = deadline
                && Instant::now() >= deadline
            {
                break;
            }

            // Wait for rate limit (distribution-based)
            tokio::time::sleep(distribution.next_delay()).await;

            // Calculate remaining time if deadline exists
            let remaining = deadline.map(|d| d.saturating_duration_since(Instant::now()));
            if let Some(remaining) = remaining
                && remaining.is_zero()
            {
                break;
            }

            // Spawn request
            let idx = prompt_index.fetch_add(1, Ordering::Relaxed);
            let prompt = self.prompts[idx % self.prompts.len()].clone();
            let client = Arc::clone(&self.client);
            let tokenizer = Arc::clone(&self.tokenizer);
            let semaphore = Arc::clone(&semaphore);
            let completed = Arc::clone(&completed);
            let request_timeout = remaining;

            let handle = tokio::spawn(async move {
                let _permit = semaphore
                    .acquire()
                    .await
                    .expect("semaphore should never be closed");
                if let Some(timeout_duration) = request_timeout {
                    // Execute with timeout based on remaining time
                    let request_future =
                        Self::execute_request(client, tokenizer, prompt, idx, false);
                    match timeout(timeout_duration, request_future).await {
                        Ok(result) => {
                            // Request completed normally
                            completed.fetch_add(1, Ordering::Relaxed);
                            result
                        }
                        Err(_) => {
                            // Request timed out due to test ending - don't count as failure
                            debug!("Request {} cancelled due to test duration limit", idx);
                            Ok(())
                        }
                    }
                } else {
                    // No deadline, execute normally
                    let result = Self::execute_request(client, tokenizer, prompt, idx, false).await;
                    completed.fetch_add(1, Ordering::Relaxed);
                    result
                }
            });
            handles.push(handle);
        }

        // Wait for all requests to complete with a grace period
        let grace_period = Duration::from_secs(60);
        for handle in handles {
            match timeout(grace_period, handle).await {
                Ok(result) => {
                    let _ = result?;
                }
                Err(_) => {
                    debug!("Request did not complete within grace period");
                }
            }
        }

        // Calculate actual test duration (excluding warmup)
        let test_duration = test_start.elapsed();
        let total_duration = start_instant.elapsed();

        info!(
            "Benchmark completed in {:.1}s total ({:.1}s test)",
            total_duration.as_secs_f64(),
            test_duration.as_secs_f64()
        );

        // Log warmup summary if applicable
        if warmup_count > 0 || warmup_duration.is_some() {
            let warmup_total = warmup_completed.load(Ordering::Relaxed);
            info!("Warmup: {} requests (excluded from metrics)", warmup_total);
        }
        info!("Main test: {} requests", completed.load(Ordering::Relaxed));

        Ok(test_duration)
    }

    async fn generate_report(&self, report_builder: ReportBuilder) -> Result<()> {
        match &self.config.output.format {
            crate::config::OutputFormat::Console => {
                report_builder.print_console_report()?;
            }
            crate::config::OutputFormat::Json => {
                let report = report_builder.build()?;
                let json = serde_json::to_string_pretty(&report)?;

                if let Some(file_path) = &self.config.output.file {
                    // Writing to file - show brief summary to console
                    tokio::fs::write(file_path, json).await?;
                    if !self.config.output.quiet {
                        self.print_brief_summary(&report)?;
                    }
                } else {
                    // Writing JSON to stdout - this is for piping
                    println!("{}", json);
                }
            }
        }
        Ok(())
    }

    async fn execute_request(
        client: Arc<OpenAIClient>,
        tokenizer: Arc<Tokenizer>,
        prompt: Prompt,
        index: usize,
        is_warmup: bool,
    ) -> Result<()> {
        debug!("Executing request {} (warmup: {})", index, is_warmup);

        let request_start = Instant::now();

        // Only record metrics if not in warmup phase
        if !is_warmup {
            Metrics::record_request_sent();
        }

        // Add per-request cache-busting to ensure every request is unique
        let cache_bust_prompt = format!("[req-{}] {}", index, prompt.prompt);
        let request = client.create_request(&cache_bust_prompt, prompt.max_tokens, None, None);

        match client.chat_completion_stream(request).await {
            Ok(mut stream) => {
                // Consume the stream to measure TTFT and total time
                let mut total_content = String::new();

                while let Some(chunk) = stream.next_chunk().await? {
                    for choice in chunk.choices {
                        if let Some(content) = choice.delta.content {
                            total_content.push_str(&content);
                        }
                    }
                }

                // Use server-reported token counts when available (accurate for any model),
                // fall back to tiktoken estimation (may be inaccurate for non-OpenAI models)
                let (input_tokens, output_tokens) = if let Some(usage) = stream.server_usage() {
                    (usage.prompt_tokens as u64, usage.completion_tokens as u64)
                } else {
                    (
                        tokenizer.count_tokens(&prompt.prompt) as u64,
                        tokenizer.count_tokens(&total_content) as u64,
                    )
                };

                // Only record metrics if not in warmup phase
                if !is_warmup {
                    // Record metrics
                    if let Some(ttft) = stream.time_to_first_token() {
                        Metrics::record_ttft_with_context(ttft, input_tokens);
                    }

                    // Record inter-token latencies with context awareness
                    for itl in stream.inter_token_latencies() {
                        Metrics::record_itl_with_context(*itl, input_tokens);
                    }

                    let total_duration = request_start.elapsed();
                    Metrics::record_latency(total_duration);
                    Metrics::record_tokens(input_tokens, output_tokens);

                    // Calculate and record TPOT (Time per Output Token, excluding first token)
                    // TPOT = (total_duration - TTFT) / (num_output_tokens - 1)
                    if let Some(ttft) = stream.time_to_first_token()
                        && output_tokens > 1
                    {
                        let generation_duration = total_duration.saturating_sub(ttft);
                        let tpot = generation_duration.as_nanos() as u64 / (output_tokens - 1);
                        Metrics::record_tpot(Duration::from_nanos(tpot));
                    }

                    Metrics::record_request_complete(RequestStatus::Success);
                }

                // Log detailed metrics for analysis (to trace log if enabled)
                if let Some(ttft) = stream.time_to_first_token() {
                    debug!(
                        "Request completed - id: {}, warmup: {}, input_tokens: {}, ttft_ms: {:.1}, total_ms: {:.1}",
                        index,
                        is_warmup,
                        input_tokens,
                        ttft.as_secs_f64() * 1000.0,
                        request_start.elapsed().as_secs_f64() * 1000.0
                    );
                }

                debug!(
                    "Request {} completed successfully (warmup: {})",
                    index, is_warmup
                );
                Ok(())
            }
            Err(e) => {
                debug!("Request {} failed: {}", index, e);
                if !is_warmup {
                    // Categorize the error
                    let error_type = if let Some(client_error) = e.downcast_ref::<ClientError>() {
                        match client_error {
                            ClientError::Connection(_) => ErrorType::Connection,
                            ClientError::Http4xx { status, .. } => ErrorType::Http4xx(*status),
                            ClientError::Http5xx { status, .. } => ErrorType::Http5xx(*status),
                            ClientError::Parse(_) => ErrorType::Parse,
                            ClientError::Timeout(_) => ErrorType::Timeout,
                            ClientError::Other(_) => ErrorType::Other,
                        }
                    } else if e.to_string().contains("timeout") {
                        ErrorType::Timeout
                    } else if e.to_string().contains("connection") {
                        ErrorType::Connection
                    } else {
                        ErrorType::Other
                    };

                    Metrics::record_request_complete(RequestStatus::Failed(error_type));
                }
                Err(e)
            }
        }
    }

    fn print_brief_summary(&self, report: &crate::report::BenchmarkReport) -> Result<()> {
        use chrono::Utc;

        let now = Utc::now();
        let timestamp = now.to_rfc3339_opts(chrono::SecondsFormat::Millis, false);

        println!();
        println!("{}", timestamp);
        println!("{} -----", timestamp);
        println!("{} Benchmark Complete", timestamp);
        println!(
            "{} Duration: {:.1}s",
            timestamp,
            report.duration.as_secs_f64()
        );
        println!(
            "{} Requests: Sent: {}",
            timestamp, report.summary.total_requests
        );
        println!(
            "{} Responses: Received: {} Ok: {} Err: {} Success: {:.2}%",
            timestamp,
            report.summary.successful_requests + report.summary.failed_requests,
            report.summary.successful_requests,
            report.summary.failed_requests,
            report.summary.success_rate * 100.0
        );

        // Error breakdown if any
        let total_errors = report.errors.timeout_errors
            + report.errors.connection_errors
            + report.errors.http_4xx_errors
            + report.errors.http_5xx_errors
            + report.errors.other_errors;
        if total_errors > 0 {
            println!(
                "{} Errors: Connection: {} 4xx: {} 5xx: {} Timeout: {} Other: {}",
                timestamp,
                report.errors.connection_errors,
                report.errors.http_4xx_errors,
                report.errors.http_5xx_errors,
                report.errors.timeout_errors,
                report.errors.other_errors
            );
        }

        println!(
            "{} Tokens: Input: {} Output: {} Total: {}",
            timestamp,
            report.throughput.total_input_tokens,
            report.throughput.total_output_tokens,
            report.throughput.total_input_tokens + report.throughput.total_output_tokens
        );

        println!(
            "{} Throughput: Requests/s: {:.2} Input tokens/s: {:.2} Output tokens/s: {:.2}",
            timestamp,
            report.throughput.requests_per_second,
            report.throughput.input_tokens_per_second,
            report.throughput.output_tokens_per_second
        );

        if report.latency.ttft_p50_ms > 0.0 {
            println!(
                "{} TTFT (ms): mean: {:.1} p50: {:.0} p90: {:.0} p95: {:.0} p99: {:.0}",
                timestamp,
                report.latency.ttft_mean_ms,
                report.latency.ttft_p50_ms,
                report.latency.ttft_p90_ms,
                report.latency.ttft_p95_ms,
                report.latency.ttft_p99_ms
            );
        }

        if report.latency.tpot_p50_ms > 0.0 {
            println!(
                "{} TPOT (ms): mean: {:.1} p50: {:.0} p90: {:.0} p95: {:.0} p99: {:.0}",
                timestamp,
                report.latency.tpot_mean_ms,
                report.latency.tpot_p50_ms,
                report.latency.tpot_p90_ms,
                report.latency.tpot_p95_ms,
                report.latency.tpot_p99_ms
            );
        }

        if report.latency.itl_p50_ms > 0.0 {
            println!(
                "{} ITL (ms): mean: {:.1} p50: {:.0} p90: {:.0} p95: {:.0} p99: {:.0}",
                timestamp,
                report.latency.itl_mean_ms,
                report.latency.itl_p50_ms,
                report.latency.itl_p90_ms,
                report.latency.itl_p95_ms,
                report.latency.itl_p99_ms
            );
        }

        println!(
            "{} Request Latency (ms): mean: {:.1} p50: {:.0} p90: {:.0} p95: {:.0} p99: {:.0}",
            timestamp,
            report.latency.request_mean_ms,
            report.latency.request_p50_ms,
            report.latency.request_p90_ms,
            report.latency.request_p95_ms,
            report.latency.request_p99_ms
        );

        if let Some(file_path) = &self.config.output.file {
            println!("{} Report written to {}", timestamp, file_path.display());
        }

        Ok(())
    }
}
