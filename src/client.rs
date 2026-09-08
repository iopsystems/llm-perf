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

    #[error("type: {error_type}, message: {message}")]
    StreamError { error_type: String, message: String },

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
    timeout: Duration,
    /// Per-read idle timeout for streaming responses (None = disabled). Detects a
    /// stalled-but-open stream that the total request timeout may not catch under
    /// HTTP/2 keep-alive.
    stream_idle_timeout: Option<Duration>,
    /// Whether to retry requests that timed out (off by default; retrying re-fires
    /// a possibly-still-running, non-idempotent generation).
    retry_on_timeout: bool,
    chat_template_kwargs: Option<serde_json::Value>,
    ignore_eos: Option<bool>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<serde_json::Value>,
    /// Suppress EOS so generation runs to `max_tokens` (llama.cpp / vLLM).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore_eos: Option<bool>,
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

#[derive(Debug, Clone, Deserialize, Default)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: u32,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct CompletionTokensDetails {
    /// Reasoning tokens, reported by reasoning models (OpenAI o-series, vLLM, etc.).
    #[serde(default)]
    pub reasoning_tokens: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
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
struct StreamError {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
}

// Error response in streaming mode
#[derive(Debug, Clone, Deserialize)]
struct StreamErrorResponse {
    error: StreamError,
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
    /// Reasoning/thinking content from reasoning models (e.g., Qwen3, DeepSeek-R1)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
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
    /// Per-read idle timeout for streaming responses (None = disabled).
    pub stream_idle_timeout: Option<Duration>,
    /// Whether to retry requests that timed out (default false).
    pub retry_on_timeout: bool,
    /// Additional kwargs forwarded to the model's chat template for every request.
    pub chat_template_kwargs: Option<serde_json::Value>,
    /// Suppress EOS so generation runs to `max_tokens` (llama.cpp / vLLM).
    pub ignore_eos: Option<bool>,
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
    /// use llm_perf::client::{OpenAIClient, ClientConfig};
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
    ///     stream_idle_timeout: None,
    ///     retry_on_timeout: false,
    ///     chat_template_kwargs: None,
    ///     ignore_eos: None,
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
            timeout: config.timeout,
            stream_idle_timeout: config.stream_idle_timeout,
            retry_on_timeout: config.retry_on_timeout,
            chat_template_kwargs: config.chat_template_kwargs,
            ignore_eos: config.ignore_eos,
        })
    }

    pub async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        let mut attempt = 0;
        // Overall budget for the whole logical request: enough for every attempt
        // to use its full per-attempt timeout, but no more — this bounds total
        // wall time (trimming trailing backoff) without defeating a legitimate
        // retry after an attempt that consumed most of one timeout (e.g. a
        // timed-out attempt when retry_on_timeout is enabled).
        let deadline = Instant::now() + self.timeout.saturating_mul(self.max_retries + 1);

        loop {
            match self.chat_completion_internal(request.clone()).await {
                Ok(resp) => {
                    if attempt > 0 {
                        log::debug!("Request succeeded after {} retries", attempt);
                    }
                    return Ok(resp);
                }
                Err(e) => {
                    if attempt < self.max_retries && self.is_retriable_error(&e) {
                        let delay = self.calculate_backoff_delay(attempt);
                        if Instant::now() + delay >= deadline {
                            log::debug!("Retry budget exhausted; returning error: {}", e);
                            return Err(e);
                        }
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
                        if attempt > 0 {
                            log::debug!("Request failed after {} retries: {}", attempt, e);
                        }
                        return Err(e);
                    }
                }
            }
        }
    }

    /// Internal implementation of non-streaming request (without retry logic)
    async fn chat_completion_internal(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        let url = format!("{}/chat/completions", self.base_url);

        let mut req = self.client.post(&url).json(&request);

        if let Some(api_key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = match req.send().await {
            Ok(resp) => resp,
            Err(e) => {
                if e.is_connect() {
                    return Err(ClientError::Connection(e.to_string()).into());
                } else if e.is_timeout() {
                    return Err(ClientError::Timeout(self.timeout).into());
                } else if e.is_request() {
                    if is_connection_cause(&e) {
                        return Err(ClientError::Connection(format!("Request error: {}", e)).into());
                    } else {
                        return Err(ClientError::Other(format!("Request error: {}", e)).into());
                    }
                } else {
                    return Err(ClientError::Other(e.to_string()).into());
                }
            }
        };

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
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
        let message = Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        };
        self.create_messages_request(&[message], max_tokens, logprobs, top_logprobs)
    }

    /// Count tokens for `text` using the server's own tokenizer via `/tokenize`
    /// (vLLM/TGI-style, served at the root, not under `/v1`). Returns the count,
    /// or an error if the server doesn't support it or responds unexpectedly — the
    /// caller falls back to estimation.
    pub async fn tokenize(&self, text: &str) -> Result<usize> {
        // /tokenize is served at the server root. Strip any trailing slash(es)
        // first, then an optional `/v1` suffix (handles `…/v1`, `…/v1/`, and `…`).
        let root = self.base_url.trim_end_matches('/');
        let root = root.strip_suffix("/v1").unwrap_or(root);
        let url = format!("{root}/tokenize");
        // `content` is llama.cpp's field; `prompt` is vLLM/TGI's. Send both so a
        // single request works against either — llama.cpp ignores a body without
        // `content` and returns an empty token array with HTTP 200, which would
        // otherwise calibrate to a ratio of zero and yield empty prompts.
        let mut req = self.client.post(&url).json(&serde_json::json!({
            "model": self.model,
            "content": text,
            "prompt": text,
        }));
        if let Some(api_key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {}", api_key));
        }
        let resp = req.send().await?;
        if !resp.status().is_success() {
            anyhow::bail!("/tokenize returned HTTP {}", resp.status());
        }
        let v: serde_json::Value = resp.json().await?;
        if let Some(count) = v.get("count").and_then(|c| c.as_u64()) {
            return Ok(count as usize);
        }
        if let Some(tokens) = v.get("tokens").and_then(|t| t.as_array()) {
            return Ok(tokens.len());
        }
        anyhow::bail!("/tokenize response missing `count`/`tokens`")
    }

    pub fn create_messages_request(
        &self,
        messages: &[Message],
        max_tokens: Option<u32>,
        logprobs: Option<bool>,
        top_logprobs: Option<u8>,
    ) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: self.model.clone(),
            messages: messages.to_vec(),
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
            chat_template_kwargs: self.chat_template_kwargs.clone(),
            ignore_eos: self.ignore_eos,
        }
    }

    /// Execute a streaming request with retry logic
    pub async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<StreamResponse> {
        let mut attempt = 0;
        // Overall budget: enough for every attempt's full per-attempt timeout, but
        // no more — bounds total wall time without defeating a legitimate retry.
        let deadline = Instant::now() + self.timeout.saturating_mul(self.max_retries + 1);

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
                    if attempt < self.max_retries && self.is_retriable_error(&e) {
                        let delay = self.calculate_backoff_delay(attempt);
                        if Instant::now() + delay >= deadline {
                            log::debug!("Retry budget exhausted; returning error: {}", e);
                            return Err(e);
                        }
                        // Record retry in metrics
                        crate::metrics::Metrics::record_retry();

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
                    return Err(ClientError::Timeout(self.timeout).into());
                } else if e.is_request() {
                    if is_connection_cause(&e) {
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
            idle_timeout: self.stream_idle_timeout,
            first_reasoning_token_time: None,
            first_content_token_time: None,
            last_reasoning_token_time: None,
            last_content_token_time: None,
            reasoning_inter_token_latencies: Vec::new(),
            content_inter_token_latencies: Vec::new(),
            reasoning_tokens: 0,
            content_tokens: 0,
            pending_chunks: std::collections::VecDeque::new(),
            line_buffer: SseLineBuffer::default(),
            done: false,
            server_usage: None,
            collected_logprobs: Vec::new(),
        })
    }

    /// Determine if an error should be retried, honoring the configured timeout policy.
    fn is_retriable_error(&self, error: &anyhow::Error) -> bool {
        classify_retriable(error, self.retry_on_timeout)
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

/// Whether a `reqwest` request error was caused by the connection dying rather
/// than by the request itself.
///
/// `reqwest`'s own `Display` is generic ("error sending request for url (...)"),
/// so matching on it alone misses every real cause -- those live in the source
/// chain ("connection closed before message completed"). A pooled keep-alive
/// connection that the server has already closed surfaces exactly this way, and
/// is safe to retry: the request never reached the server, so no generation was
/// started and no work is duplicated.
fn is_connection_cause(e: &reqwest::Error) -> bool {
    let mut chain = e.to_string();
    let mut src: Option<&(dyn std::error::Error + 'static)> = std::error::Error::source(e);
    while let Some(inner) = src {
        chain.push_str("; ");
        chain.push_str(&inner.to_string());
        src = inner.source();
    }
    let chain = chain.to_lowercase();
    [
        "connection closed",
        "connection reset",
        "connection refused",
        "connection aborted",
        "broken pipe",
        "unexpected eof",
        "channel closed",
    ]
    .iter()
    .any(|needle| chain.contains(needle))
}

/// Classify whether an error should be retried.
///
/// Connection errors (pre-flight; the request demonstrably never reached the
/// server) and 5xx server errors are always retriable. Timeouts are retried only
/// when `retry_on_timeout` is set, since a timed-out chat completion may still be
/// running server-side and retrying re-fires an expensive, non-idempotent
/// generation. 4xx / parse / stream / other errors are never retried.
fn classify_retriable(error: &anyhow::Error, retry_on_timeout: bool) -> bool {
    if let Some(client_error) = error.downcast_ref::<ClientError>() {
        match client_error {
            ClientError::Connection(_) => true,
            ClientError::Timeout(_) => retry_on_timeout,
            ClientError::Http5xx { .. } => true,
            ClientError::Http4xx { .. } => false,
            ClientError::Parse(_) => false,
            ClientError::StreamError { .. } => false,
            ClientError::Other(_) => false,
        }
    } else {
        // For non-ClientError types, fall back to message inspection. Connection
        // hints are always retriable; timeout hints only when opted in.
        let err_str = error.to_string().to_lowercase();
        err_str.contains("connection") || (retry_on_timeout && err_str.contains("timeout"))
    }
}

/// Accumulates raw stream bytes and yields complete lines (newline-terminated),
/// preserving any incomplete trailing bytes — crucially including a multi-byte
/// UTF-8 character split across reads — until the rest arrives. Decoding each raw
/// HTTP chunk independently (the previous approach) corrupts such characters into
/// replacement chars, which silently mangles content and can break JSON parsing.
///
/// The buffer is intentionally uncapped: SSE delimits every event with a newline,
/// so a single buffered line is one event's bytes (a JSON chunk, well under a few
/// KB in practice). A server that streamed unbounded bytes with no newline could
/// grow this without limit, but that requires a severely broken server and matches
/// the prior implementation's behavior.
#[derive(Default)]
struct SseLineBuffer {
    buf: Vec<u8>,
}

impl SseLineBuffer {
    /// Append `data` and return all now-complete lines (with the trailing `\n`
    /// and any `\r` removed). Incomplete trailing bytes stay buffered.
    fn push(&mut self, data: &[u8]) -> Vec<Vec<u8>> {
        self.buf.extend_from_slice(data);
        let mut lines = Vec::new();
        let Some(last_nl) = self.buf.iter().rposition(|&b| b == b'\n') else {
            return lines; // no complete line yet
        };
        // Split off the bytes after the last newline; they remain incomplete.
        let tail = self.buf.split_off(last_nl + 1);
        let complete = std::mem::replace(&mut self.buf, tail);
        for raw in complete.split(|&b| b == b'\n') {
            // `complete` ends in '\n', so the final split element is empty; blank
            // SSE separator lines are also empty — skip both.
            if raw.is_empty() {
                continue;
            }
            let line = if raw.last() == Some(&b'\r') {
                &raw[..raw.len() - 1]
            } else {
                raw
            };
            lines.push(line.to_vec());
        }
        lines
    }
}

/// A classified SSE line.
enum SseEvent {
    Chunk(Box<ChatCompletionChunk>),
    Done,
    StreamError {
        error_type: String,
        message: String,
    },
    /// Non-data line, comment/keep-alive, or empty payload — ignore.
    Ignore,
    /// A `data:` line that could not be decoded or parsed — must be surfaced, not
    /// silently dropped, since it represents lost tokens/usage.
    Malformed,
}

/// Classify a single complete SSE line (no trailing newline).
fn parse_sse_line(line: &[u8]) -> SseEvent {
    // Only `data:` lines carry payload; comments (':' prefix) and event/id lines
    // are keep-alives or metadata we ignore.
    let Some(payload) = line.strip_prefix(b"data:") else {
        return SseEvent::Ignore;
    };
    // The SSE spec allows an optional single space after `data:`.
    let payload = payload.strip_prefix(b" ").unwrap_or(payload);

    let json_str = match std::str::from_utf8(payload) {
        Ok(s) => s.trim(),
        Err(_) => return SseEvent::Malformed,
    };
    if json_str.is_empty() {
        return SseEvent::Ignore; // keep-alive `data:` with no payload
    }
    if json_str == "[DONE]" {
        return SseEvent::Done;
    }
    if let Ok(error_resp) = serde_json::from_str::<StreamErrorResponse>(json_str) {
        return SseEvent::StreamError {
            error_type: error_resp.error.error_type,
            message: error_resp.error.message,
        };
    }
    match serde_json::from_str::<ChatCompletionChunk>(json_str) {
        Ok(chunk) => SseEvent::Chunk(Box::new(chunk)),
        Err(_) => SseEvent::Malformed,
    }
}

pub struct StreamResponse {
    response: reqwest::Response,
    start_time: Instant,
    /// Per-read idle timeout (None = disabled).
    idle_timeout: Option<Duration>,
    // Phase-specific TTFT tracking
    first_reasoning_token_time: Option<Duration>,
    first_content_token_time: Option<Duration>,
    // Phase-specific last-token tracking for ITL
    last_reasoning_token_time: Option<Instant>,
    last_content_token_time: Option<Instant>,
    // Phase-specific ITL
    reasoning_inter_token_latencies: Vec<Duration>,
    content_inter_token_latencies: Vec<Duration>,
    // Token counts per phase
    reasoning_tokens: u32,
    content_tokens: u32,
    /// Buffer for parsed chunks when a single HTTP chunk contains multiple SSE events
    pending_chunks: std::collections::VecDeque<ChatCompletionChunk>,
    /// Byte-level buffer for SSE lines split across HTTP chunks (preserves
    /// multi-byte UTF-8 characters that straddle a read boundary).
    line_buffer: SseLineBuffer,
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

            // Read the next chunk, optionally bounded by a per-read idle timeout so
            // a stalled-but-open stream (which HTTP/2 keep-alive can hide from the
            // total request timeout) is detected instead of hanging the worker.
            let bytes = match self.idle_timeout {
                Some(idle) => match tokio::time::timeout(idle, self.response.chunk()).await {
                    Ok(result) => result?,
                    Err(_) => return Err(ClientError::Timeout(idle).into()),
                },
                None => self.response.chunk().await?,
            };

            // If no more data from server, stream is done
            let Some(data) = bytes else {
                return Ok(None);
            };

            // Append the raw bytes and process every now-complete line. Byte-level
            // buffering keeps multi-byte UTF-8 characters intact across HTTP read
            // boundaries instead of lossily decoding each read independently.
            for line in self.line_buffer.push(&data) {
                match parse_sse_line(&line) {
                    SseEvent::Chunk(chunk) => self.pending_chunks.push_back(*chunk),
                    SseEvent::Done => {
                        // Stop at [DONE]; any chunks parsed before it this read are
                        // already buffered and drained by the caller loop. Lines
                        // after [DONE] in the same read are discarded (per spec, and
                        // matching the prior behavior).
                        self.done = true;
                        break;
                    }
                    SseEvent::StreamError {
                        error_type,
                        message,
                    } => {
                        return Err(ClientError::StreamError {
                            error_type,
                            message,
                        }
                        .into());
                    }
                    SseEvent::Malformed => {
                        // Don't silently drop: surface lost tokens/usage.
                        log::warn!("Skipping unparseable SSE data line ({} bytes)", line.len());
                        crate::metrics::Metrics::record_malformed_chunk();
                    }
                    SseEvent::Ignore => {}
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

        let now = Instant::now();

        // Track reasoning tokens
        let has_reasoning = chunk
            .choices
            .iter()
            .any(|c| c.delta.reasoning_content.is_some());
        if has_reasoning {
            if self.first_reasoning_token_time.is_none() {
                self.first_reasoning_token_time = Some(self.start_time.elapsed());
            } else if let Some(last) = self.last_reasoning_token_time {
                self.reasoning_inter_token_latencies
                    .push(now.duration_since(last));
            }
            self.last_reasoning_token_time = Some(now);
            for choice in &chunk.choices {
                if choice.delta.reasoning_content.is_some() {
                    self.reasoning_tokens += 1;
                }
            }
        }

        // Track content tokens
        let has_content = chunk.choices.iter().any(|c| c.delta.content.is_some());
        if has_content {
            if self.first_content_token_time.is_none() {
                self.first_content_token_time = Some(self.start_time.elapsed());
            } else if let Some(last) = self.last_content_token_time {
                self.content_inter_token_latencies
                    .push(now.duration_since(last));
            }
            self.last_content_token_time = Some(now);
            for choice in &chunk.choices {
                if choice.delta.content.is_some() {
                    self.content_tokens += 1;
                }
            }
        }
    }

    /// First token of any kind (reasoning or content) — prefill latency.
    pub fn time_to_first_token(&self) -> Option<Duration> {
        match (
            self.first_reasoning_token_time,
            self.first_content_token_time,
        ) {
            (Some(r), Some(c)) => Some(r.min(c)),
            (Some(r), None) => Some(r),
            (None, Some(c)) => Some(c),
            (None, None) => None,
        }
    }

    pub fn time_to_first_reasoning_token(&self) -> Option<Duration> {
        self.first_reasoning_token_time
    }

    pub fn time_to_first_content_token(&self) -> Option<Duration> {
        self.first_content_token_time
    }

    /// Time spent in reasoning phase (first reasoning token to first content token).
    pub fn think_duration(&self) -> Option<Duration> {
        match (
            self.first_reasoning_token_time,
            self.first_content_token_time,
        ) {
            (Some(r), Some(c)) if c > r => Some(c - r),
            _ => None,
        }
    }

    pub fn total_duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub fn reasoning_inter_token_latencies(&self) -> &[Duration] {
        &self.reasoning_inter_token_latencies
    }

    pub fn content_inter_token_latencies(&self) -> &[Duration] {
        &self.content_inter_token_latencies
    }

    pub fn reasoning_tokens(&self) -> u32 {
        self.reasoning_tokens
    }

    pub fn content_tokens(&self) -> u32 {
        self.content_tokens
    }

    /// Whether this stream contained reasoning tokens.
    pub fn has_reasoning(&self) -> bool {
        self.first_reasoning_token_time.is_some()
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
/// or the timeout is exceeded. This is useful when starting a server and llm-perf
/// simultaneously, allowing llm-perf to wait for the server to be ready.
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
    fn usage_with_cached_tokens_parses() {
        let json = r#"{
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 80}
        }"#;
        let usage: Usage = serde_json::from_str(json).unwrap();
        assert_eq!(
            usage.prompt_tokens_details.as_ref().unwrap().cached_tokens,
            80
        );
    }

    #[test]
    fn usage_without_cached_tokens_parses() {
        let json = r#"{
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }"#;
        let usage: Usage = serde_json::from_str(json).unwrap();
        assert!(usage.prompt_tokens_details.is_none());
    }

    #[test]
    fn usage_with_reasoning_tokens_parses() {
        let json = r#"{
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "completion_tokens_details": {"reasoning_tokens": 20}
        }"#;
        let usage: Usage = serde_json::from_str(json).unwrap();
        assert_eq!(
            usage
                .completion_tokens_details
                .as_ref()
                .unwrap()
                .reasoning_tokens,
            20
        );
    }

    #[test]
    fn usage_without_completion_details_is_none() {
        let json = r#"{"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}"#;
        let usage: Usage = serde_json::from_str(json).unwrap();
        assert!(usage.completion_tokens_details.is_none());
    }

    #[test]
    fn cached_tokens_absent_in_details_defaults_to_zero() {
        let json = r#"{
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {}
        }"#;
        let usage: Usage = serde_json::from_str(json).unwrap();
        assert_eq!(
            usage.prompt_tokens_details.as_ref().unwrap().cached_tokens,
            0
        );
    }

    #[test]
    fn client_retains_configured_timeout_for_timeout_errors() {
        // Timeout errors must report the configured timeout, not a hard-coded 60s.
        let config = ClientConfig {
            base_url: "http://localhost:1".to_string(),
            api_key: None,
            model: "test".to_string(),
            timeout: Duration::from_secs(7),
            max_retries: 0,
            retry_initial_delay_ms: 1,
            retry_max_delay_ms: 1,
            pool_size: 1,
            stream_idle_timeout: None,
            retry_on_timeout: false,
            chat_template_kwargs: None,
            ignore_eos: None,
        };
        let client = OpenAIClient::new(config).unwrap();
        assert_eq!(client.timeout, Duration::from_secs(7));
    }

    #[test]
    fn ignore_eos_is_serialized_only_when_set() {
        // llama.cpp gates on the key's presence, so it must appear on the wire
        // when configured -- and must stay absent otherwise, so servers that
        // reject unknown fields are unaffected when the option is not used.
        let mk = |ignore_eos| {
            let config = ClientConfig {
                base_url: "http://localhost:1".to_string(),
                api_key: None,
                model: "test".to_string(),
                timeout: Duration::from_secs(1),
                max_retries: 0,
                retry_initial_delay_ms: 1,
                retry_max_delay_ms: 1,
                pool_size: 1,
                stream_idle_timeout: None,
                retry_on_timeout: false,
                chat_template_kwargs: None,
                ignore_eos,
            };
            let client = OpenAIClient::new(config).unwrap();
            let req = client.create_request("hi", Some(8), None, None);
            serde_json::to_string(&req).unwrap()
        };

        let body = mk(Some(true));
        assert!(
            body.contains("\"ignore_eos\":true"),
            "ignore_eos missing from wire body: {body}"
        );
        assert!(!mk(None).contains("ignore_eos"));
    }

    #[test]
    fn sse_buffer_preserves_multibyte_char_split_across_reads() {
        // "🚀" is 4 bytes; split it across two reads. Byte-level buffering must
        // reassemble it intact rather than lossily decoding each read (which would
        // corrupt it to replacement chars).
        let rocket = "🚀".as_bytes(); // [0xF0,0x9F,0x9A,0x80]
        let mut buf = SseLineBuffer::default();

        let mut chunk1 = b"data: x".to_vec();
        chunk1.extend_from_slice(&rocket[..2]); // first half of the emoji, no newline
        assert!(buf.push(&chunk1).is_empty(), "no complete line yet");

        let mut chunk2 = rocket[2..].to_vec(); // second half
        chunk2.push(b'\n');
        let lines = buf.push(&chunk2);
        assert_eq!(lines.len(), 1);
        assert_eq!(std::str::from_utf8(&lines[0]).unwrap(), "data: x🚀");
    }

    #[test]
    fn sse_buffer_splits_multiple_lines_and_strips_crlf() {
        let mut buf = SseLineBuffer::default();
        let lines = buf.push(b"data: a\r\ndata: b\n");
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], b"data: a"); // trailing \r stripped
        assert_eq!(lines[1], b"data: b");
    }

    #[test]
    fn sse_buffer_holds_incomplete_trailing_line() {
        let mut buf = SseLineBuffer::default();
        assert!(buf.push(b"data: hel").is_empty());
        let lines = buf.push(b"lo\n");
        assert_eq!(lines, vec![b"data: hello".to_vec()]);
    }

    #[test]
    fn parse_sse_line_classifies_events() {
        assert!(matches!(parse_sse_line(b"data: [DONE]"), SseEvent::Done));
        assert!(matches!(parse_sse_line(b": keep-alive"), SseEvent::Ignore));
        assert!(matches!(parse_sse_line(b"data: "), SseEvent::Ignore));
        assert!(matches!(
            parse_sse_line(b"data: {not valid json"),
            SseEvent::Malformed
        ));
        // A complete data line that is not valid UTF-8 is malformed, not silently dropped.
        assert!(matches!(
            parse_sse_line(b"data: \xF0\x9F"),
            SseEvent::Malformed
        ));
        let valid = br#"data: {"id":"1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}"#;
        assert!(matches!(parse_sse_line(valid), SseEvent::Chunk(_)));
    }

    #[test]
    fn timeout_retry_is_opt_in() {
        let timeout_err: anyhow::Error = ClientError::Timeout(Duration::from_secs(1)).into();
        // Retrying a timeout re-fires a possibly-still-running generation, so it's
        // off by default and only enabled when explicitly opted in.
        assert!(!classify_retriable(&timeout_err, false));
        assert!(classify_retriable(&timeout_err, true));
    }

    #[test]
    fn connection_and_5xx_retry_regardless_of_timeout_flag() {
        let conn: anyhow::Error = ClientError::Connection("refused".into()).into();
        let s5: anyhow::Error = ClientError::Http5xx {
            status: 503,
            message: "busy".into(),
        }
        .into();
        let s4: anyhow::Error = ClientError::Http4xx {
            status: 400,
            message: "bad".into(),
        }
        .into();
        assert!(classify_retriable(&conn, false));
        assert!(classify_retriable(&s5, false));
        assert!(!classify_retriable(&s4, false));
    }

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
