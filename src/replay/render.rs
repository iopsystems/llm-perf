//! Rendering and tokenization of chat messages, as the server does them.

use crate::client::{Message, OpenAIClient, TokenPiece};
use crate::config::ReplayServer;
use anyhow::Result;
use std::future::Future;
use std::sync::Arc;

/// The server operations replay needs to size prompts and find shared
/// prefixes in tokens.
pub trait Renderer: Send + Sync {
    /// Tokens of the prompt the server builds from `messages`, including the
    /// generation prompt.
    fn render(&self, messages: &[Message]) -> impl Future<Output = Result<Vec<u32>>> + Send;
    /// Tokens of `text` with no special tokens added. llama-server leaves
    /// special-token text in `text` as plain text; vLLM turns it into the
    /// special token.
    fn tokenize(&self, text: &str) -> impl Future<Output = Result<Vec<u32>>> + Send;
    /// Text of `tokens`.
    fn detokenize(&self, tokens: &[u32]) -> impl Future<Output = Result<String>> + Send;
    /// Tokens of `text` with each token's text.
    fn tokenize_pieces(&self, text: &str) -> impl Future<Output = Result<Vec<TokenPiece>>> + Send;
}

/// `Renderer` backed by llama-server's `/apply-template`, `/tokenize` and
/// `/detokenize`. `chat_template_kwargs` must match the generation requests.
pub struct LlamaServerRenderer {
    client: Arc<OpenAIClient>,
    chat_template_kwargs: Option<serde_json::Value>,
}

impl LlamaServerRenderer {
    pub fn new(client: Arc<OpenAIClient>, chat_template_kwargs: Option<serde_json::Value>) -> Self {
        Self {
            client,
            chat_template_kwargs,
        }
    }
}

fn ids(pieces: Vec<TokenPiece>) -> Vec<u32> {
    pieces.into_iter().map(|p| p.id).collect()
}

impl Renderer for LlamaServerRenderer {
    async fn render(&self, messages: &[Message]) -> Result<Vec<u32>> {
        let prompt = self
            .client
            .apply_template(messages, self.chat_template_kwargs.as_ref())
            .await?;
        Ok(ids(self
            .client
            .tokenize_pieces(&prompt, true, true, false)
            .await?))
    }

    async fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        Ok(ids(self
            .client
            .tokenize_pieces(text, false, false, false)
            .await?))
    }

    async fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        self.client.detokenize(tokens).await
    }

    async fn tokenize_pieces(&self, text: &str) -> Result<Vec<TokenPiece>> {
        self.client.tokenize_pieces(text, false, false, true).await
    }
}

/// `Renderer` backed by vLLM's `/tokenize` (with chat messages for `render`)
/// and `/detokenize`. `chat_template_kwargs` must match the generation
/// requests.
pub struct VllmRenderer {
    client: Arc<OpenAIClient>,
    chat_template_kwargs: Option<serde_json::Value>,
}

impl VllmRenderer {
    pub fn new(client: Arc<OpenAIClient>, chat_template_kwargs: Option<serde_json::Value>) -> Self {
        Self {
            client,
            chat_template_kwargs,
        }
    }
}

impl Renderer for VllmRenderer {
    async fn render(&self, messages: &[Message]) -> Result<Vec<u32>> {
        self.client
            .vllm_tokenize_messages(messages, self.chat_template_kwargs.as_ref())
            .await
    }

    async fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        Ok(ids(self.client.vllm_tokenize_text(text, false).await?))
    }

    async fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        self.client.vllm_detokenize(tokens).await
    }

    async fn tokenize_pieces(&self, text: &str) -> Result<Vec<TokenPiece>> {
        self.client.vllm_tokenize_text(text, true).await
    }
}

/// The renderer for the server type in `[replay] server`.
pub enum ServerRenderer {
    LlamaServer(LlamaServerRenderer),
    Vllm(VllmRenderer),
}

impl ServerRenderer {
    pub fn new(
        server: ReplayServer,
        client: Arc<OpenAIClient>,
        chat_template_kwargs: Option<serde_json::Value>,
    ) -> Self {
        match server {
            ReplayServer::LlamaServer => {
                Self::LlamaServer(LlamaServerRenderer::new(client, chat_template_kwargs))
            }
            ReplayServer::Vllm => Self::Vllm(VllmRenderer::new(client, chat_template_kwargs)),
        }
    }
}

impl Renderer for ServerRenderer {
    async fn render(&self, messages: &[Message]) -> Result<Vec<u32>> {
        match self {
            Self::LlamaServer(r) => r.render(messages).await,
            Self::Vllm(r) => r.render(messages).await,
        }
    }

    async fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            Self::LlamaServer(r) => r.tokenize(text).await,
            Self::Vllm(r) => r.tokenize(text).await,
        }
    }

    async fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        match self {
            Self::LlamaServer(r) => r.detokenize(tokens).await,
            Self::Vllm(r) => r.detokenize(tokens).await,
        }
    }

    async fn tokenize_pieces(&self, text: &str) -> Result<Vec<TokenPiece>> {
        match self {
            Self::LlamaServer(r) => r.tokenize_pieces(text).await,
            Self::Vllm(r) => r.tokenize_pieces(text).await,
        }
    }
}

/// A word-level tokenizer and chat template for tests. Each space-prefixed
/// word is one token; the template trims message content and wraps each
/// message in role and end tokens, as the Llama 3.1 template does.
#[cfg(test)]
pub mod fake {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// Tokenizes on spaces, a leading space joining the word after it (as
    /// BPE tokenizers do), and renders each message as `<role>`, its content
    /// and `<end>`. The default trims content when rendering;
    /// `untrimmed()` renders it as is, as some chat templates do.
    #[derive(Default)]
    pub struct FakeRenderer {
        vocab: Mutex<(HashMap<String, u32>, Vec<String>)>,
        untrimmed: bool,
    }

    impl FakeRenderer {
        pub fn untrimmed() -> Self {
            Self {
                untrimmed: true,
                ..Self::default()
            }
        }

        fn intern(&self, piece: &str) -> u32 {
            let mut v = self.vocab.lock().unwrap();
            if let Some(id) = v.0.get(piece) {
                return *id;
            }
            let id = v.1.len() as u32;
            v.1.push(piece.to_string());
            v.0.insert(piece.to_string(), id);
            id
        }

        fn pieces(text: &str) -> Vec<String> {
            let mut out: Vec<String> = Vec::new();
            for (i, part) in text.split(' ').enumerate() {
                if part.is_empty() {
                    if i > 0 {
                        out.push(" ".to_string());
                    }
                    continue;
                }
                out.push(if i == 0 {
                    part.to_string()
                } else {
                    format!(" {part}")
                });
            }
            out
        }

        pub fn tokens(&self, text: &str) -> Vec<u32> {
            Self::pieces(text).iter().map(|p| self.intern(p)).collect()
        }
    }

    impl Renderer for FakeRenderer {
        async fn render(&self, messages: &[Message]) -> Result<Vec<u32>> {
            let mut out = vec![self.intern("<bos>")];
            for m in messages {
                out.push(self.intern(&format!("<{}>", m.role)));
                let content = if self.untrimmed {
                    m.content.as_str()
                } else {
                    m.content.trim()
                };
                out.extend(self.tokens(content));
                out.push(self.intern("<end>"));
            }
            out.push(self.intern("<assistant>"));
            Ok(out)
        }

        async fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
            Ok(self.tokens(text))
        }

        async fn detokenize(&self, tokens: &[u32]) -> Result<String> {
            let v = self.vocab.lock().unwrap();
            Ok(tokens.iter().map(|t| v.1[*t as usize].as_str()).collect())
        }

        async fn tokenize_pieces(&self, text: &str) -> Result<Vec<TokenPiece>> {
            Ok(Self::pieces(text)
                .into_iter()
                .map(|p| TokenPiece {
                    id: self.intern(&p),
                    piece: Some(p),
                })
                .collect())
        }
    }
}
