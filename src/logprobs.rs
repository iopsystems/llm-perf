use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::sync::mpsc;

use crate::client::{CompletionRequest, CompletionResponse, OpenAIClient, TokenLogprob};
use crate::config::LogprobsMode;

/// A record of logprobs for a single prompt/response pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogprobRecord {
    /// Index of the prompt in the input file
    pub prompt_index: usize,
    /// The original prompt text
    pub prompt: String,
    /// Token-level logprobs from the response
    pub tokens: Vec<TokenLogprob>,
    /// How the prompt was sent. Records written before the field existed
    /// were all chat captures, so a missing field reads as `chat`.
    #[serde(default)]
    pub mode: LogprobsMode,
}

/// The `/completions` request for one prompt in raw mode: the prompt text
/// with no chat template, and `logprobs` set to the number of top tokens.
/// `max_tokens` defaults to 1, so the capture is the token that follows the
/// prompt. Sampling is greedy: the top-N values do not depend on it, and a
/// prompt whose next token llama-server returns no logprobs for (see
/// `parse_completion_logprobs`) is then the same prompt on repeated runs of
/// the same model file, server build and hardware. Captures of different
/// quantizations can skip different prompts.
pub fn raw_request(
    model: &str,
    prompt: &str,
    max_tokens: Option<u32>,
    top_logprobs: u8,
) -> CompletionRequest {
    CompletionRequest {
        model: model.to_string(),
        prompt: prompt.to_string(),
        max_tokens: Some(max_tokens.unwrap_or(1)),
        temperature: Some(0.0),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        // `OpenAIClient::completion` reads one JSON body.
        stream: Some(false),
        logprobs: Some(top_logprobs),
    }
}

/// The result of sending one prompt.
#[derive(Debug)]
pub enum Capture {
    /// The server returned logprobs for at least one token.
    Record(LogprobRecord),
    /// The server answered without logprobs.
    Empty,
    /// The request failed after the client's retries.
    Failed(anyhow::Error),
}

/// Send prompt `prompt_index` in `mode` and collect its token logprobs.
/// `max_tokens` is the resolved `[endpoint]`/prompt limit. `Err` is an error
/// the `logprobs` subcommand stops on: in chat mode, any error reading the
/// stream after the response started; in raw mode, a `/completions` logprobs
/// shape this module does not read, which would repeat for every prompt.
pub async fn capture_prompt(
    client: &OpenAIClient,
    mode: LogprobsMode,
    prompt_index: usize,
    prompt: &str,
    max_tokens: Option<u32>,
    top_logprobs: u8,
) -> Result<Capture> {
    let tokens = match mode {
        LogprobsMode::Chat => {
            let request = client.create_request(prompt, max_tokens, Some(true), Some(top_logprobs));
            match client.chat_completion_stream(request).await {
                Ok(mut stream) => {
                    while stream.next_chunk().await?.is_some() {}
                    stream.logprobs().to_vec()
                }
                Err(e) => return Ok(Capture::Failed(e)),
            }
        }
        LogprobsMode::Raw => {
            let request = raw_request(client.model(), prompt, max_tokens, top_logprobs);
            match client.completion(request).await {
                Ok(resp) => parse_completion_logprobs(&resp)
                    .map_err(|e| e.context(format!("prompt {prompt_index}")))?,
                Err(e) => return Ok(Capture::Failed(e)),
            }
        }
    };
    if tokens.is_empty() {
        return Ok(Capture::Empty);
    }
    Ok(Capture::Record(LogprobRecord {
        prompt_index,
        prompt: prompt.to_string(),
        tokens,
        mode,
    }))
}

/// Token logprobs from a non-streamed `/completions` response.
///
/// Reads llama-server's shape, `choices[0].logprobs.content`: one entry per
/// generated token with a `top_logprobs` array, the same entries
/// `/chat/completions` returns. Returns an empty list when the server sent no
/// logprobs (`null`). llama-server sends `null` for an empty prompt, and when
/// the generated token ends in an incomplete UTF-8 character, as a byte-level
/// token inside an emoji can: it stores no logprobs for such a token. Any other
/// shape is an error, including the legacy OpenAI shape (`tokens`,
/// `token_logprobs`, and `top_logprobs` as a list of maps).
pub fn parse_completion_logprobs(resp: &CompletionResponse) -> Result<Vec<TokenLogprob>> {
    let Some(choice) = resp.choices.first() else {
        return Ok(Vec::new());
    };
    let logprobs = match &choice.logprobs {
        None | Some(serde_json::Value::Null) => return Ok(Vec::new()),
        Some(v) => v,
    };
    let Some(content) = logprobs.get("content") else {
        let keys: Vec<&str> = logprobs
            .as_object()
            .map(|o| o.keys().map(String::as_str).collect())
            .unwrap_or_default();
        anyhow::bail!(
            "/completions returned logprobs without a `content` array (keys: {keys:?}); \
             raw mode reads llama-server's shape and not the legacy OpenAI shape"
        );
    };
    let tokens: Option<Vec<TokenLogprob>> = serde_json::from_value(content.clone())
        .map_err(|e| anyhow::anyhow!("/completions logprobs.content: {e}"))?;
    Ok(tokens.unwrap_or_default())
}

/// Asynchronous writer that receives LogprobRecords via a channel and writes them as JSONL
pub struct LogprobWriter {
    rx: mpsc::Receiver<LogprobRecord>,
    output_path: std::path::PathBuf,
}

impl LogprobWriter {
    /// Create a new LogprobWriter and its sender channel
    pub fn new(
        output_path: std::path::PathBuf,
        buffer_size: usize,
    ) -> (mpsc::Sender<LogprobRecord>, Self) {
        let (tx, rx) = mpsc::channel(buffer_size);
        (tx, Self { rx, output_path })
    }

    /// Run the writer loop, consuming records from the channel and writing to disk
    pub async fn run(mut self) -> Result<()> {
        use tokio::io::AsyncWriteExt;

        let file = tokio::fs::File::create(&self.output_path).await?;
        let mut writer = tokio::io::BufWriter::new(file);

        while let Some(record) = self.rx.recv().await {
            let json = serde_json::to_string(&record)?;
            writer.write_all(json.as_bytes()).await?;
            writer.write_all(b"\n").await?;
        }

        writer.flush().await?;
        log::info!("Logprobs written to {}", self.output_path.display());
        Ok(())
    }
}

/// Load logprob records from a JSONL file
pub fn load_logprob_file(path: &Path) -> Result<Vec<LogprobRecord>> {
    let contents = std::fs::read_to_string(path)?;
    let mut records = Vec::new();

    for (line_num, line) in contents.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<LogprobRecord>(line) {
            Ok(record) => records.push(record),
            Err(e) => {
                log::warn!(
                    "Failed to parse logprob record at line {}: {}",
                    line_num + 1,
                    e
                );
            }
        }
    }

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{ClientConfig, DEFAULT_POOL_IDLE_TIMEOUT, OpenAIClient};

    const FIXTURE: &str =
        include_str!("../tests/fixtures/llama-server/qwen3.5-0.8b-q2_k/completions-logprobs.jsonl");

    /// The line of the fixture captured for `case`.
    fn captured(case: &str) -> serde_json::Value {
        FIXTURE
            .lines()
            .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
            .find(|v| v["case"] == case)
            .unwrap_or_else(|| panic!("no fixture case {case}"))
    }

    fn captured_response(case: &str) -> CompletionResponse {
        serde_json::from_value(captured(case)["response"].clone()).unwrap()
    }

    #[test]
    fn parses_captured_top_n() {
        let tokens = parse_completion_logprobs(&captured_response("legacy_logprobs_int")).unwrap();
        assert_eq!(tokens.len(), 1);
        let top: Vec<(&str, f64)> = tokens[0]
            .top_logprobs
            .iter()
            .map(|t| (t.token.as_str(), t.logprob))
            .collect();
        assert_eq!(
            top,
            [
                (" located", -2.1098644733428955),
                (" a", -2.431964159011841),
                (" the", -3.2865307331085205),
                (" rich", -3.3819515705108643),
                (" ", -3.5938351154327393),
            ]
        );
        // The sampled token is not always the most likely one.
        assert_eq!(tokens[0].token, " in");
    }

    #[test]
    fn parses_captured_top_20_and_multibyte_tokens() {
        let tokens = parse_completion_logprobs(&captured_response("top20")).unwrap();
        assert_eq!(tokens[0].top_logprobs.len(), 20);
        let tokens = parse_completion_logprobs(&captured_response("cjk")).unwrap();
        assert_eq!(tokens[0].top_logprobs[3].token, "東京");
    }

    #[test]
    fn captured_null_logprobs_is_empty() {
        for case in ["empty_prompt", "no_logprobs"] {
            let tokens = parse_completion_logprobs(&captured_response(case)).unwrap();
            assert!(tokens.is_empty(), "{case}");
        }
    }

    #[test]
    fn logprobs_true_returns_one_alternative() {
        // Why raw mode sends an integer: llama-server reads `logprobs` as
        // `n_probs`, so `true` is 1, and `top_logprobs` is ignored here.
        for case in ["logprobs_bool_true", "logprobs_bool_true_top_logprobs"] {
            let tokens = parse_completion_logprobs(&captured_response(case)).unwrap();
            assert_eq!(tokens[0].top_logprobs.len(), 1, "{case}");
        }
    }

    #[test]
    fn legacy_openai_shape_is_refused() {
        let resp: CompletionResponse = serde_json::from_value(serde_json::json!({
            "choices": [{"text": " Paris", "index": 0, "finish_reason": "length",
                "logprobs": {"tokens": [" Paris"], "token_logprobs": [-0.1],
                    "top_logprobs": [{" Paris": -0.1, " a": -2.5}], "text_offset": [24]}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
        }))
        .unwrap();
        let err = parse_completion_logprobs(&resp).unwrap_err().to_string();
        assert!(err.contains("without a `content` array"), "{err}");
        assert!(err.contains("token_logprobs"), "{err}");
    }

    #[test]
    fn raw_request_body() {
        let body = serde_json::to_value(raw_request("m", "The capital of France is", Some(1), 20))
            .unwrap();
        assert_eq!(
            body,
            serde_json::json!({
                "model": "m",
                "prompt": "The capital of France is",
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": false,
                "logprobs": 20,
            })
        );
    }

    #[test]
    fn raw_request_max_tokens_defaults_to_one() {
        // Without a limit llama-server generates until EOS or the context ends.
        assert_eq!(raw_request("m", "p", None, 5).max_tokens, Some(1));
        assert_eq!(raw_request("m", "p", Some(4), 5).max_tokens, Some(4));
    }

    #[test]
    fn greedy_and_sampled_top_n_agree() {
        // Why raw mode can sample greedily: llama-server's top-N are the
        // pre-sampling probabilities.
        let top = |case| {
            parse_completion_logprobs(&captured_response(case)).unwrap()[0]
                .top_logprobs
                .iter()
                .map(|t| (t.token.clone(), t.logprob))
                .collect::<Vec<_>>()
        };
        assert_eq!(top("greedy"), top("legacy_logprobs_int"));
    }

    #[test]
    fn captured_partial_utf8_token_has_no_logprobs() {
        let tokens = parse_completion_logprobs(&captured_response("sampled_partial_utf8")).unwrap();
        assert!(tokens.is_empty());
        // A fragment among the alternatives is returned with an empty token.
        let tokens = parse_completion_logprobs(&captured_response("top_partial_utf8")).unwrap();
        assert_eq!(tokens[0].top_logprobs[1].token, "");
    }

    #[test]
    fn record_without_mode_reads_as_chat() {
        let line = r#"{"prompt_index":0,"prompt":"p","tokens":[{"token":"X","logprob":-2.9,"top_logprobs":[{"token":"Y","logprob":-2.2}]}]}"#;
        let r: LogprobRecord = serde_json::from_str(line).unwrap();
        assert_eq!(r.mode, LogprobsMode::Chat);
        let r = LogprobRecord {
            mode: LogprobsMode::Raw,
            ..r
        };
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["mode"], "raw");
    }

    fn mock_client(server: &mockito::ServerGuard) -> OpenAIClient {
        OpenAIClient::new(ClientConfig {
            base_url: format!("{}/v1", server.url()),
            api_key: None,
            model: "local".to_string(),
            timeout: std::time::Duration::from_secs(5),
            max_retries: 0,
            connection_retries: 0,
            retry_initial_delay_ms: 1,
            retry_max_delay_ms: 1,
            pool_size: 1,
            stream_idle_timeout: None,
            retry_on_timeout: false,
            chat_template_kwargs: None,
            ignore_eos: None,
            pool_idle_timeout: DEFAULT_POOL_IDLE_TIMEOUT,
        })
        .unwrap()
    }

    /// Mocks for both endpoints; the one `mode` must not use expects no call.
    async fn mock_endpoints(
        server: &mut mockito::ServerGuard,
        mode: LogprobsMode,
        status: usize,
        body: &str,
    ) -> (mockito::Mock, mockito::Mock) {
        let (raw_hits, chat_hits) = match mode {
            LogprobsMode::Raw => (1, 0),
            LogprobsMode::Chat => (0, 1),
        };
        let raw = server
            .mock("POST", "/v1/completions")
            .match_body(mockito::Matcher::Json(serde_json::json!({
                "model": "local",
                "prompt": "The capital of France is",
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": false,
                "logprobs": 5,
            })))
            .with_status(status)
            .with_body(body)
            .expect(raw_hits)
            .create_async()
            .await;
        let chat = server
            .mock("POST", "/v1/chat/completions")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!({
                "messages": [{"role": "user", "content": "The capital of France is"}],
                "logprobs": true,
                "top_logprobs": 5,
            })))
            .with_status(status)
            .with_body(body)
            .expect(chat_hits)
            .create_async()
            .await;
        (raw, chat)
    }

    async fn capture(mode: LogprobsMode, status: usize, body: &str) -> Result<Capture> {
        let mut server = mockito::Server::new_async().await;
        let (raw, chat) = mock_endpoints(&mut server, mode, status, body).await;
        let client = mock_client(&server);
        let out = capture_prompt(&client, mode, 7, "The capital of France is", None, 5).await;
        raw.assert_async().await;
        chat.assert_async().await;
        out
    }

    #[tokio::test]
    async fn raw_capture_posts_to_completions_and_records_raw() {
        let body = captured("legacy_logprobs_int")["response"].to_string();
        let Capture::Record(r) = capture(LogprobsMode::Raw, 200, &body).await.unwrap() else {
            panic!("expected a record");
        };
        assert_eq!(r.mode, LogprobsMode::Raw);
        assert_eq!(r.prompt_index, 7);
        assert_eq!(r.tokens[0].top_logprobs.len(), 5);
    }

    #[tokio::test]
    async fn chat_capture_posts_to_chat_completions_and_records_chat() {
        // One streamed chunk carrying the captured chat logprobs.
        let lp = captured("chat_same_prompt")["response"]["choices"][0]["logprobs"].clone();
        let chunk = serde_json::json!({
            "id": "x", "object": "chat.completion.chunk", "created": 1, "model": "local",
            "choices": [{"index": 0, "delta": {"content": "using"}, "finish_reason": null,
                "logprobs": lp}],
        });
        let body = format!("data: {chunk}\n\ndata: [DONE]\n\n");
        let Capture::Record(r) = capture(LogprobsMode::Chat, 200, &body).await.unwrap() else {
            panic!("expected a record");
        };
        assert_eq!(r.mode, LogprobsMode::Chat);
        assert_eq!(r.tokens[0].top_logprobs[0].token, "<think>");
    }

    #[tokio::test]
    async fn raw_capture_without_logprobs_is_empty() {
        let body = captured("sampled_partial_utf8")["response"].to_string();
        let out = capture(LogprobsMode::Raw, 200, &body).await.unwrap();
        assert!(matches!(out, Capture::Empty), "{out:?}");
    }

    #[tokio::test]
    async fn raw_capture_request_error_is_failed_not_fatal() {
        let out = capture(LogprobsMode::Raw, 400, r#"{"error":"bad"}"#)
            .await
            .unwrap();
        assert!(matches!(out, Capture::Failed(_)), "{out:?}");
    }

    #[tokio::test]
    async fn raw_capture_unknown_shape_is_fatal() {
        let body = serde_json::json!({
            "choices": [{"text": " Paris", "index": 0, "finish_reason": "length",
                "logprobs": {"tokens": [" Paris"], "token_logprobs": [-0.1],
                    "top_logprobs": [{" Paris": -0.1}], "text_offset": [24]}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
        });
        let err = capture(LogprobsMode::Raw, 200, &body.to_string())
            .await
            .unwrap_err();
        assert!(format!("{err:#}").contains("prompt 7"), "{err:#}");
    }
}
