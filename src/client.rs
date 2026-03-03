use anyhow::Result;
use rand::Rng;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ClientError {
    #[error("Connection error: {0}")]
    Connection(String),

    #[error("HTTP {status} client error: {message}")]
    Http4xx { status: u16, message: String },

    #[error("HTTP {status} server error: {message}")]
    Http5xx { status: u16, message: String },

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Timeout after {0:?}")]
    Timeout(Duration),

    #[error("Other error: {0}")]
    Other(String),
}

#[derive(Debug, Clone)]
pub struct OpenAIClient {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    model: String,
    max_retries: u32,
    retry_initial_delay_ms: u64,
    retry_max_delay_ms: u64,
}

// Request types for OpenAI Chat Completions API
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

// Response types
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// Streaming response types
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
    /// Server-reported token usage (present in final chunk when stream_options.include_usage is set)
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub logprobs: Option<ChoiceLogprobs>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Top log probability for a single token alternative
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f64,
}

/// Log probability information for a single generated token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f64,
    #[serde(default)]
    pub top_logprobs: Vec<TopLogprob>,
}

/// Log probability information for a choice in a streaming response
#[derive(Debug, Clone, Deserialize)]
pub struct ChoiceLogprobs {
    pub content: Option<Vec<TokenLogprob>>,
}

// Models list response
#[derive(Debug, Clone, Deserialize)]
pub struct ModelsResponse {
    pub data: Vec<Model>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Model {
    pub id: String,
    pub object: String,
    #[serde(default)]
    pub owned_by: String,
}

/// Configuration for creating an OpenAI client.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Base URL of the OpenAI-compatible API endpoint
    pub base_url: String,
    /// Optional API key for authentication
    pub api_key: Option<String>,
    /// Model name to use for requests
    pub model: String,
    /// Request timeout duration
    pub timeout: Duration,
    /// Maximum number of retry attempts for transient failures
    pub max_retries: u32,
    /// Initial delay in milliseconds for exponential backoff
    pub retry_initial_delay_ms: u64,
    /// Maximum delay in milliseconds for exponential backoff
    pub retry_max_delay_ms: u64,
    /// Connection pool size (should match concurrency for optimal performance)
    pub pool_size: usize,
}

impl OpenAIClient {
    /// Creates a new OpenAI-compatible HTTP client with retry logic and connection pooling.
    ///
    /// # Arguments
    ///
    /// * `config` - Client configuration including endpoint, model, timeouts, and retry settings
    ///
    /// # Returns
    ///
    /// Returns a configured client ready to make API requests, or an error if client creation fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use llm_bench::client::{OpenAIClient, ClientConfig};
    /// use std::time::Duration;
    ///
    /// let config = ClientConfig {
    ///     base_url: "http://localhost:8080/v1".to_string(),
    ///     api_key: None,
    ///     model: "llama-3.1-8b".to_string(),
    ///     timeout: Duration::from_secs(60),
    ///     max_retries: 3,
    ///     retry_initial_delay_ms: 100,
    ///     retry_max_delay_ms: 10000,
    ///     pool_size: 10,
    /// };
    ///
    /// let client = OpenAIClient::new(config).unwrap();
    /// ```
    pub fn new(config: ClientConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(config.timeout)
            .pool_max_idle_per_host(config.pool_size) // Match concurrency for optimal connection reuse
            .pool_idle_timeout(Duration::from_secs(300)) // Keep connections alive for 5 minutes
            .tcp_keepalive(Duration::from_secs(60)) // TCP keep-alive every 60 seconds
            .http2_keep_alive_interval(Duration::from_secs(30)) // HTTP/2 keep-alive
            .http2_keep_alive_timeout(Duration::from_secs(20))
            .http2_keep_alive_while_idle(true) // Send keep-alive even when idle
            .build()?;

        Ok(Self {
            client,
            base_url: config.base_url,
            api_key: config.api_key,
            model: config.model,
            max_retries: config.max_retries,
            retry_initial_delay_ms: config.retry_initial_delay_ms,
            retry_max_delay_ms: config.retry_max_delay_ms,
        })
    }

    pub async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        let url = format!("{}/chat/completions", self.base_url);

        let mut req = self.client.post(&url).json(&request);

        if let Some(api_key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = req.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await?;
            anyhow::bail!("API request failed with status {}: {}", status, text);
        }

        let completion: ChatCompletionResponse = response.json().await?;
        Ok(completion)
    }

    pub fn create_request(
        &self,
        prompt: &str,
        max_tokens: Option<u32>,
        logprobs: Option<bool>,
        top_logprobs: Option<u8>,
    ) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: self.model.clone(),
            messages: vec![Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            max_tokens,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: Some(false),
            stream_options: None,
            logprobs,
            top_logprobs,
        }
    }

    /// Execute a streaming request with retry logic
    pub async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<StreamResponse> {
        let mut attempt = 0;

        loop {
            match self.chat_completion_stream_internal(request.clone()).await {
                Ok(stream) => {
                    if attempt > 0 {
                        log::debug!("Request succeeded after {} retries", attempt);
                    }
                    return Ok(stream);
                }
                Err(e) => {
                    // Check if we should retry
                    if attempt < self.max_retries && Self::is_retriable_error(&e) {
                        // Record retry in metrics
                        crate::metrics::Metrics::record_retry();

                        let delay = self.calculate_backoff_delay(attempt);
                        log::debug!(
                            "Request failed (attempt {}/{}): {}. Retrying in {:?}",
                            attempt + 1,
                            self.max_retries + 1,
                            e,
                            delay
                        );

                        tokio::time::sleep(delay).await;
                        attempt += 1;
                    } else {
                        // No more retries or non-retriable error
                        if attempt > 0 {
                            log::debug!("Request failed after {} retries: {}", attempt, e);
                        }
                        return Err(e);
                    }
                }
            }
        }
    }

    /// Internal implementation of streaming request (without retry logic)
    async fn chat_completion_stream_internal(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<StreamResponse> {
        let mut request = request;
        request.stream = Some(true);
        request.stream_options = Some(StreamOptions {
            include_usage: true,
        });

        let url = format!("{}/chat/completions", self.base_url);

        let mut req = self
            .client
            .post(&url)
            .json(&request)
            .header("Connection", "keep-alive"); // Ensure HTTP/1.1 keep-alive

        if let Some(api_key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {}", api_key));
        }

        let start_time = Instant::now();

        // Send request and handle connection errors
        let response = match req.send().await {
            Ok(resp) => resp,
            Err(e) => {
                if e.is_connect() {
                    return Err(ClientError::Connection(e.to_string()).into());
                } else if e.is_timeout() {
                    return Err(ClientError::Timeout(Duration::from_secs(60)).into());
                } else if e.is_request() {
                    // Check if this is a connection-related request error
                    let err_msg = e.to_string();
                    if err_msg.contains("connection closed")
                        || err_msg.contains("connection reset")
                        || err_msg.contains("broken pipe")
                        || err_msg.contains("connection refused")
                    {
                        return Err(ClientError::Connection(format!("Request error: {}", e)).into());
                    } else {
                        return Err(ClientError::Other(format!("Request error: {}", e)).into());
                    }
                } else {
                    return Err(ClientError::Other(e.to_string()).into());
                }
            }
        };

        // Handle HTTP errors
        if !response.status().is_success() {
            let status = response.status();
            let status_code = status.as_u16();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unable to read response body".to_string());

            if (400..500).contains(&status_code) {
                return Err(ClientError::Http4xx {
                    status: status_code,
                    message: text,
                }
                .into());
            } else if (500..600).contains(&status_code) {
                return Err(ClientError::Http5xx {
                    status: status_code,
                    message: text,
                }
                .into());
            } else {
                return Err(ClientError::Other(format!("HTTP {}: {}", status_code, text)).into());
            }
        }

        Ok(StreamResponse {
            response,
            start_time,
            first_token_time: None,
            last_token_time: None,
            total_tokens: 0,
            inter_token_latencies: Vec::new(),
            pending_chunks: std::collections::VecDeque::new(),
            partial_line: String::new(),
            done: false,
            server_usage: None,
            collected_logprobs: Vec::new(),
        })
    }

    /// Determine if an error should be retried
    fn is_retriable_error(error: &anyhow::Error) -> bool {
        if let Some(client_error) = error.downcast_ref::<ClientError>() {
            match client_error {
                ClientError::Connection(_) => true,   // Network issues
                ClientError::Timeout(_) => true,      // Timeout
                ClientError::Http5xx { .. } => true,  // Server errors
                ClientError::Http4xx { .. } => false, // Client errors (don't retry)
                ClientError::Parse(_) => false,       // Parse errors (don't retry)
                ClientError::Other(_) => false,       // Unknown errors (don't retry)
            }
        } else {
            // For non-ClientError types, check the error message
            let err_str = error.to_string().to_lowercase();
            err_str.contains("timeout") || err_str.contains("connection")
        }
    }

    /// Calculate exponential backoff delay with jitter
    fn calculate_backoff_delay(&self, attempt: u32) -> Duration {
        let base_delay_ms = self.retry_initial_delay_ms * 2_u64.pow(attempt);
        let max_delay_ms = self.retry_max_delay_ms;

        // Cap at max delay
        let capped_delay_ms = base_delay_ms.min(max_delay_ms);

        // Add jitter: random value between 50% and 100% of the capped delay
        let mut rng = rand::thread_rng();
        let jitter_factor = rng.gen_range(0.5..=1.0);
        let jittered_delay_ms = (capped_delay_ms as f64 * jitter_factor) as u64;

        Duration::from_millis(jittered_delay_ms)
    }
}

pub struct StreamResponse {
    response: reqwest::Response,
    start_time: Instant,
    first_token_time: Option<Duration>,
    last_token_time: Option<Instant>,
    total_tokens: u32,
    inter_token_latencies: Vec<Duration>,
    /// Buffer for parsed chunks when a single HTTP chunk contains multiple SSE events
    pending_chunks: std::collections::VecDeque<ChatCompletionChunk>,
    /// Buffer for incomplete SSE lines split across HTTP chunks
    partial_line: String,
    /// Set to true when we encounter the [DONE] marker
    done: bool,
    /// Server-reported token usage from the final streaming chunk
    server_usage: Option<Usage>,
    /// Accumulated logprobs from streaming chunks
    collected_logprobs: Vec<TokenLogprob>,
}

impl StreamResponse {
    pub async fn next_chunk(&mut self) -> Result<Option<ChatCompletionChunk>> {
        loop {
            // Return buffered chunks first
            if let Some(chunk) = self.pending_chunks.pop_front() {
                self.record_chunk_metrics(&chunk);
                return Ok(Some(chunk));
            }

            // If we've seen [DONE], no more data
            if self.done {
                return Ok(None);
            }

            let bytes = self.response.chunk().await?;

            // If no more data from server, stream is done
            let Some(data) = bytes else {
                return Ok(None);
            };

            // Prepend any partial line from the previous HTTP chunk
            let text = if self.partial_line.is_empty() {
                String::from_utf8_lossy(&data).into_owned()
            } else {
                let mut combined = std::mem::take(&mut self.partial_line);
                combined.push_str(&String::from_utf8_lossy(&data));
                combined
            };

            // Check if the text ends with a newline; if not, the last line is partial
            let ends_with_newline = text.ends_with('\n') || text.ends_with('\r');
            let lines: Vec<&str> = text.lines().collect();

            for (i, line) in lines.iter().enumerate() {
                // If this is the last line and the chunk didn't end with a newline,
                // it's a partial line split across HTTP chunks
                if i == lines.len() - 1 && !ends_with_newline {
                    self.partial_line = line.to_string();
                    continue;
                }

                if let Some(json_str) = line.strip_prefix("data: ") {
                    if json_str == "[DONE]" {
                        self.done = true;
                        break;
                    }

                    if let Ok(chunk) = serde_json::from_str::<ChatCompletionChunk>(json_str) {
                        self.pending_chunks.push_back(chunk);
                    }
                }
            }

            // Return the first buffered chunk if any were parsed
            if let Some(chunk) = self.pending_chunks.pop_front() {
                self.record_chunk_metrics(&chunk);
                return Ok(Some(chunk));
            }

            // If we hit [DONE] with no pending chunks, we're done
            if self.done {
                return Ok(None);
            }

            // No parseable data in this HTTP chunk, loop and read the next one
        }
    }

    fn record_chunk_metrics(&mut self, chunk: &ChatCompletionChunk) {
        // Capture server-reported usage from the final chunk
        if let Some(usage) = &chunk.usage {
            self.server_usage = Some(usage.clone());
        }

        // Accumulate logprobs from chunk
        for choice in &chunk.choices {
            if let Some(ref lp) = choice.logprobs
                && let Some(ref content) = lp.content
            {
                self.collected_logprobs.extend(content.iter().cloned());
            }
        }

        let has_content = chunk.choices.iter().any(|c| c.delta.content.is_some());

        if has_content {
            let now = Instant::now();

            // Record time to first token
            if self.first_token_time.is_none() {
                self.first_token_time = Some(self.start_time.elapsed());
            } else if let Some(last_time) = self.last_token_time {
                // Record inter-token latency
                let itl = now.duration_since(last_time);
                self.inter_token_latencies.push(itl);
            }

            self.last_token_time = Some(now);

            // Count tokens (approximate - just counting content chunks)
            for choice in &chunk.choices {
                if choice.delta.content.is_some() {
                    self.total_tokens += 1;
                }
            }
        }
    }

    pub fn time_to_first_token(&self) -> Option<Duration> {
        self.first_token_time
    }

    pub fn total_duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub fn inter_token_latencies(&self) -> &[Duration] {
        &self.inter_token_latencies
    }

    /// Server-reported token usage, if the server supports stream_options.include_usage
    pub fn server_usage(&self) -> Option<&Usage> {
        self.server_usage.as_ref()
    }

    /// Collected logprobs from the streaming response
    pub fn logprobs(&self) -> &[TokenLogprob] {
        &self.collected_logprobs
    }
}

/// Wait for server to become ready by polling /v1/models endpoint
///
/// This function polls the /v1/models endpoint until it returns a successful response
/// or the timeout is exceeded. This is useful when starting a server and llm-bench
/// simultaneously, allowing llm-bench to wait for the server to be ready.
///
/// Using /v1/models is better than a dedicated health endpoint because:
/// - All OpenAI-compatible backends must support it
/// - Success means the server is actually ready to handle requests, not just "alive"
/// - Works with vLLM, TGI, llama.cpp, Ollama, etc.
///
/// # Arguments
///
/// * `base_url` - The base URL of the server (e.g., "http://localhost:8080/v1")
/// * `api_key` - Optional API key for authentication
/// * `total_timeout` - Maximum time to wait for server to be ready
/// * `retry_interval` - Time to wait between retry attempts
///
/// # Returns
///
/// Returns Ok(()) if server becomes ready, or an error if timeout is exceeded
pub async fn check_server_ready(
    base_url: &str,
    api_key: Option<&str>,
    total_timeout: Duration,
    retry_interval: Duration,
) -> Result<()> {
    let start_time = Instant::now();
    let mut attempt = 0;

    log::info!("Waiting for server to be ready at {}...", base_url);

    loop {
        attempt += 1;

        log::debug!(
            "Server readiness check attempt {}: GET {}/models",
            attempt,
            base_url
        );

        // Try to list models with a short timeout per request
        match tokio::time::timeout(
            Duration::from_secs(10),
            list_models(base_url, api_key, Duration::from_secs(10)),
        )
        .await
        {
            Ok(Ok(models)) => {
                log::info!(
                    "Server is ready ({} model{} available after {:.1}s)",
                    models.len(),
                    if models.len() == 1 { "" } else { "s" },
                    start_time.elapsed().as_secs_f64()
                );
                return Ok(());
            }
            Ok(Err(e)) => {
                log::debug!("Models endpoint returned error: {}", e);
            }
            Err(_) => {
                log::debug!("Models endpoint request timed out");
            }
        }

        // Check if we've exceeded the timeout
        if start_time.elapsed() >= total_timeout {
            anyhow::bail!(
                "Server readiness timeout after {:.1}s. Server at {} did not become ready.",
                total_timeout.as_secs_f64(),
                base_url
            );
        }

        // Wait before next attempt
        let elapsed = start_time.elapsed();
        let remaining = total_timeout.saturating_sub(elapsed);

        if remaining.is_zero() {
            anyhow::bail!(
                "Server readiness timeout after {:.1}s. Server at {} did not become ready.",
                total_timeout.as_secs_f64(),
                base_url
            );
        }

        // Log progress every 30 seconds
        if attempt % 6 == 0 {
            log::info!(
                "Still waiting for server (elapsed: {:.0}s, timeout: {:.0}s)...",
                elapsed.as_secs_f64(),
                total_timeout.as_secs_f64()
            );
        }

        tokio::time::sleep(retry_interval.min(remaining)).await;
    }
}

// Helper function to list available models
pub async fn list_models(
    base_url: &str,
    api_key: Option<&str>,
    timeout: Duration,
) -> Result<Vec<Model>> {
    let client = Client::builder().timeout(timeout).build()?;

    let url = format!("{}/models", base_url);
    let mut req = client.get(&url);

    if let Some(key) = api_key {
        req = req.header("Authorization", format!("Bearer {}", key));
    }

    let response = req
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to query models endpoint: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unable to read response".to_string());
        anyhow::bail!("Models endpoint returned {}: {}", status, text);
    }

    let models_response: ModelsResponse = response
        .json()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to parse models response: {}", e))?;

    Ok(models_response.data)
}

// Helper function to detect model from server
pub async fn detect_model(
    base_url: &str,
    api_key: Option<&str>,
    timeout: Duration,
) -> Result<String> {
    let models = list_models(base_url, api_key, timeout).await?;

    if models.is_empty() {
        anyhow::bail!("No models available from server at {}/models", base_url);
    }

    // Return the first model (raw name for API requests)
    let raw_model = models[0].id.clone();
    let normalized_model = normalize_model_name(&raw_model);

    if models.len() > 1 {
        log::info!("Found {} models, using: {}", models.len(), normalized_model);
        log::debug!(
            "Available models: {:?}",
            models.iter().map(|m| &m.id).collect::<Vec<_>>()
        );
    } else if raw_model != normalized_model {
        log::info!(
            "Detected model: {} (server reports as: {})",
            normalized_model,
            raw_model
        );
    } else {
        log::info!("Detected model: {}", raw_model);
    }

    // Return raw model name for API requests (not normalized)
    Ok(raw_model)
}

/// Normalize model names, especially for llama.cpp which returns full file paths
///
/// For GGUF files and file paths:
/// - Extracts filename from path
/// - Converts to lowercase
/// - Preserves dots in version numbers (e.g., v0.3, 2.5)
/// - Preserves underscores in quantization formats (e.g., q5_k_m)
/// - Replaces other dots/underscores with hyphens
///
/// For API model names:
/// - Converts to lowercase
/// - Only normalizes underscores to hyphens
/// - Preserves dots (e.g., gpt-3.5-turbo stays as-is)
///
/// Examples:
/// - `/mnt/llm-models/GGUF/Qwen/Qwen3-4B/Qwen3-4B.F16.gguf` -> `qwen3-4b-f16`
/// - `Mistral-7B-Instruct-v0.3-Q5_K_M.gguf` -> `mistral-7b-instruct-v0.3-q5_k_m`
/// - `llama-3.1-8b-instruct-q4_k_m.gguf` -> `llama-3.1-8b-instruct-q4_k_m`
/// - `gpt-3.5-turbo` -> `gpt-3.5-turbo` (API name preserved)
fn normalize_model_name(model: &str) -> String {
    let is_file_path = model.contains('/') || model.contains('\\');
    let is_gguf = model.ends_with(".gguf");

    // If it looks like a file path, extract just the filename
    // Handle both Unix (/) and Windows (\) path separators
    let name = if is_file_path {
        // Try forward slash first, then backslash
        let from_forward = model.rsplit('/').next();
        let from_backward = model.rsplit('\\').next();

        // Use whichever gives us the shortest result (more specific)
        match (from_forward, from_backward) {
            (Some(f), Some(b)) => {
                if f.len() <= b.len() {
                    f
                } else {
                    b
                }
            }
            (Some(f), None) => f,
            (None, Some(b)) => b,
            (None, None) => model,
        }
    } else {
        model
    };

    // Remove .gguf extension if present
    let name = name.strip_suffix(".gguf").unwrap_or(name);

    // Convert to lowercase for consistency
    let name = name.to_lowercase();

    if is_file_path || is_gguf {
        // Smart normalization for GGUF files
        // Preserve dots in version patterns (e.g., v0.3, 2.5)
        // Preserve underscores in quantization patterns (e.g., q5_k_m, f16)
        let chars: Vec<char> = name.chars().collect();
        let len = chars.len();
        let mut result = String::with_capacity(len);

        for i in 0..len {
            let ch = chars[i];

            match ch {
                '.' => {
                    // Preserve dots in version patterns (surrounded by digits or after 'v')
                    let prev_is_digit_or_v =
                        i > 0 && (chars[i - 1].is_ascii_digit() || chars[i - 1] == 'v');
                    let next_is_digit = i + 1 < len && chars[i + 1].is_ascii_digit();

                    if prev_is_digit_or_v && next_is_digit {
                        result.push('.');
                    } else {
                        result.push('-');
                    }
                }
                '_' => {
                    // Preserve underscores in quantization patterns (between alphanumerics)
                    let prev_is_alnum = i > 0 && chars[i - 1].is_ascii_alphanumeric();
                    let next_is_alnum = i + 1 < len && chars[i + 1].is_ascii_alphanumeric();

                    if prev_is_alnum && next_is_alnum {
                        result.push('_');
                    } else {
                        result.push('-');
                    }
                }
                _ => result.push(ch),
            }
        }

        result
    } else {
        // For API model names, only normalize underscores
        name.replace('_', "-")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_model_name() {
        // llama.cpp full path - F16 has no underscores, dot is not a version
        assert_eq!(
            normalize_model_name("/mnt/llm-models/GGUF/Qwen/Qwen3-4B/Qwen3-4B.F16.gguf"),
            "qwen3-4b-f16"
        );

        // Windows path with version number and quantization format
        assert_eq!(
            normalize_model_name("C:\\Models\\llama-3.1-8b-q4_k_m.gguf"),
            "llama-3.1-8b-q4_k_m"
        );

        // GGUF with version number and quantization format
        assert_eq!(
            normalize_model_name("Mistral-7B-Instruct-v0.3-Q5_K_M.gguf"),
            "mistral-7b-instruct-v0.3-q5_k_m"
        );

        // Regular model name (OpenAI style) - preserve dots
        assert_eq!(normalize_model_name("gpt-3.5-turbo"), "gpt-3.5-turbo");

        // Model with underscores (non-GGUF) - normalize underscores only
        assert_eq!(
            normalize_model_name("llama_3_1_8b_instruct"),
            "llama-3-1-8b-instruct"
        );

        // API model name with dots - preserve dots
        assert_eq!(
            normalize_model_name("Qwen2.5-7B-Instruct"),
            "qwen2.5-7b-instruct"
        );

        // API model with mixed case and underscores
        assert_eq!(
            normalize_model_name("Claude_3_5_Sonnet"),
            "claude-3-5-sonnet"
        );
    }
}
