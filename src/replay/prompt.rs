//! Building each call's messages from the previous call's.
//!
//! Call `n` starts from call `n-1`'s messages followed by its reply as an
//! assistant message. That list is cut so its rendered tokens repeat `reuse`
//! tokens of call `n-1`'s rendered prompt followed by the tokenized reply,
//! then filler brings the rendered prompt to `prompt` tokens. Positions are
//! found in rendered tokens, because chat templates add tokens around each
//! message and trim message content.

use super::filler::FillerPool;
use super::render::Renderer;
use crate::client::Message;
use anyhow::Result;
use rand::rngs::StdRng;

/// How far past the end of the last located content to look for a message's
/// content in a rendered prompt; covers role headers and template preambles.
const LOCATE_WINDOW: usize = 256;

/// A message, the tokens of its trimmed content, and where that content
/// starts in the rendered prompt it was last sent in (`None` if not found).
#[derive(Debug, Clone)]
struct Segment {
    msg: Message,
    tokens: Vec<u32>,
    start: Option<usize>,
}

/// One call's messages and how they relate to the previous call.
#[derive(Debug, Clone)]
pub struct BuiltCall {
    pub messages: Vec<Message>,
    /// Tokens of the rendered prompt.
    pub prompt_tokens: usize,
    /// Tokens of the requested reuse beyond the previous rendered prompt plus
    /// reply.
    pub shortfall: usize,
    /// The rendered prompt is longer than requested, because the kept part
    /// plus the smallest addition already exceeds it.
    pub overshoot: bool,
}

/// The state a session carries from one call to the next.
#[derive(Debug, Default)]
pub struct SessionPrompt {
    segments: Vec<Segment>,
    rendered: Vec<u32>,
    /// The last reply as a history message, with its trimmed content tokens.
    reply: Option<Segment>,
    /// Tokens of the last reply as generated (untrimmed): the reference for
    /// reuse and for cuts inside the reply.
    reply_tokens: Vec<u32>,
}

impl SessionPrompt {
    /// Tokens of the prompt the last `build` rendered.
    pub fn rendered(&self) -> &[u32] {
        &self.rendered
    }

    /// Build the next call. `system` is kept whole at the start of every call.
    pub async fn build<R: Renderer>(
        &mut self,
        renderer: &R,
        system: Option<&Message>,
        reuse: usize,
        prompt: usize,
        pool: &FillerPool,
        rng: &mut StdRng,
    ) -> Result<BuiltCall> {
        let (mut kept, shortfall) = self.cut(renderer, reuse).await?;

        if let Some(sys) = system
            && kept
                .first()
                .is_none_or(|s| s.msg.role != "system" || s.msg.content != sys.content)
        {
            if kept.first().is_some_and(|s| s.msg.role == "system") {
                kept.remove(0);
            }
            kept.insert(
                0,
                Segment {
                    msg: sys.clone(),
                    tokens: renderer.tokenize(sys.content.trim()).await?,
                    start: None,
                },
            );
        }

        // Filler goes on the end of a trailing user message, or in a new one.
        let extend = kept.last().is_some_and(|s| s.msg.role == "user");
        let min_words = if extend { 0 } else { 1 };
        let base_content = if extend {
            kept.last().unwrap().msg.content.clone()
        } else {
            String::new()
        };
        let mut messages: Vec<Message> = kept.iter().map(|s| s.msg.clone()).collect();
        if !extend {
            messages.push(Message {
                role: "user".to_string(),
                content: String::new(),
            });
        }
        let base_len = renderer.render(&messages).await?.len();

        // Draw 64 words more than the estimate, so one correction can add up
        // to 64 tokens.
        let mut n = prompt.saturating_sub(base_len).max(min_words);
        let words: Vec<String> = (0..n + 64).map(|_| pool.text(rng, 1)).collect();
        let fill = |messages: &mut Vec<Message>, n: usize| {
            messages.last_mut().unwrap().content = format!("{base_content}{}", words[..n].concat());
        };
        fill(&mut messages, n);
        let mut rendered = renderer.render(&messages).await?;
        let err = rendered.len() as i64 - prompt as i64;
        if err != 0 {
            let adjusted = (n as i64 - err).clamp(min_words as i64, words.len() as i64) as usize;
            if adjusted != n {
                n = adjusted;
                fill(&mut messages, n);
                rendered = renderer.render(&messages).await?;
            }
        }

        // Content tokens: reused for whole kept messages that were located in
        // the previous prompt, re-tokenized otherwise.
        let mut segments = Vec::with_capacity(messages.len());
        for (i, msg) in messages.iter().enumerate() {
            let tokens = match kept.get(i) {
                Some(s) if s.msg.content == msg.content && s.start.is_some() => s.tokens.clone(),
                _ => renderer.tokenize(msg.content.trim()).await?,
            };
            segments.push(Segment {
                msg: msg.clone(),
                tokens,
                start: None,
            });
        }
        locate(&rendered, &mut segments);

        let built = BuiltCall {
            messages,
            prompt_tokens: rendered.len(),
            shortfall,
            overshoot: rendered.len() > prompt,
        };
        self.segments = segments;
        self.rendered = rendered;
        self.reply = None;
        self.reply_tokens.clear();
        Ok(built)
    }

    /// Record the server's reply to the last call built.
    pub async fn record_reply<R: Renderer>(&mut self, renderer: &R, content: String) -> Result<()> {
        self.reply_tokens = renderer.tokenize(&content).await?;
        let tokens = renderer.tokenize(content.trim()).await?;
        self.reply = Some(Segment {
            msg: Message {
                role: "assistant".to_string(),
                content,
            },
            tokens,
            start: Some(self.rendered.len()),
        });
        Ok(())
    }

    /// The previous call's messages and reply, cut to `reuse` tokens of the
    /// reference (previous rendered prompt followed by the tokenized reply).
    async fn cut<R: Renderer>(&self, renderer: &R, reuse: usize) -> Result<(Vec<Segment>, usize)> {
        let mut list = self.segments.clone();
        let reference_len =
            self.rendered.len() + self.reply.as_ref().map_or(0, |_| self.reply_tokens.len());
        let reply_index = self.reply.as_ref().map(|reply| {
            list.push(reply.clone());
            list.len() - 1
        });
        let shortfall = reuse.saturating_sub(reference_len);
        let reuse = reuse.min(reference_len);

        let Some(j) = list
            .iter()
            .rposition(|s| s.start.is_some_and(|start| start <= reuse))
        else {
            return Ok((Vec::new(), shortfall));
        };
        let k = reuse - list[j].start.unwrap();
        // Positions inside the reply count generated tokens.
        let tokens = if reply_index == Some(j) {
            &self.reply_tokens
        } else {
            &list[j].tokens
        };
        let mut kept: Vec<Segment> = list[..j].to_vec();
        if k >= tokens.len() {
            kept.push(list[j].clone());
        } else if k > 0 {
            let content = renderer.detokenize(&tokens[..k]).await?;
            kept.push(Segment {
                msg: Message {
                    role: list[j].msg.role.clone(),
                    content,
                },
                tokens: tokens[..k].to_vec(),
                start: None,
            });
        }
        Ok((kept, shortfall))
    }
}

/// Find each segment's content in `rendered`, in order, searching up to
/// `LOCATE_WINDOW` tokens past the previous segment's content.
fn locate(rendered: &[u32], segments: &mut [Segment]) {
    let mut cursor = 0;
    for s in segments.iter_mut() {
        let n = s.tokens.len();
        let last = (cursor + LOCATE_WINDOW).min(rendered.len().saturating_sub(n));
        s.start = (cursor..=last).find(|&p| rendered.get(p..p + n) == Some(&s.tokens[..]));
        // After a miss, assume the content sat at the cursor, so one miss
        // does not move the window away from every later message.
        cursor = s.start.unwrap_or(cursor) + n;
    }
}

#[cfg(test)]
mod tests {
    use super::super::filler::{FillerPool, call_rng};
    use super::super::render::fake::FakeRenderer;
    use super::*;

    /// The previous call's rendered prompt followed by its tokenized reply.
    fn reference_of(s: &SessionPrompt) -> Vec<u32> {
        [s.rendered.clone(), s.reply_tokens.clone()].concat()
    }

    /// Tokens the rendered prompt of `b` shares with `reference`.
    async fn shared(r: &FakeRenderer, reference: &[u32], b: &BuiltCall) -> usize {
        let rendered = r.render(&b.messages).await.unwrap();
        rendered
            .iter()
            .zip(reference)
            .take_while(|(x, y)| x == y)
            .count()
    }

    fn pool() -> FillerPool {
        let words: Vec<String> = (0..500).map(|i| format!("w{i}")).collect();
        let refs: Vec<&str> = words.iter().map(String::as_str).collect();
        FillerPool::from_words(&refs)
    }

    async fn first(r: &FakeRenderer, s: &mut SessionPrompt, prompt: usize) -> BuiltCall {
        s.build(r, None, 0, prompt, &pool(), &mut call_rng(1, "s", 0))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn first_call_reaches_the_target_size() {
        let r = FakeRenderer::default();
        let mut s = SessionPrompt::default();
        let b = first(&r, &mut s, 50).await;
        assert_eq!(b.prompt_tokens, 50);
        assert!(!b.overshoot);
        // [bos, <user>, 46 words, <end>, <assistant>]
        assert_eq!(s.segments[0].start, Some(2));
        assert_eq!(s.segments[0].tokens.len(), 46);
    }

    #[tokio::test]
    async fn cut_inside_the_reply_keeps_that_many_reply_tokens() {
        let r = FakeRenderer::default();
        let mut s = SessionPrompt::default();
        first(&r, &mut s, 14).await; // bos <user> 10 words <end> <assistant>
        s.record_reply(&r, "r1 r2 r3".to_string()).await.unwrap();
        let reference = reference_of(&s);
        let b = s
            .build(&r, None, 15, 25, &pool(), &mut call_rng(1, "s", 1))
            .await
            .unwrap();
        assert_eq!(shared(&r, &reference, &b).await, 15);
        assert_eq!(b.prompt_tokens, 25);
        let roles: Vec<&str> = b.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["user", "assistant", "user"]);
        assert_eq!(b.messages[1].content, "r1");
    }

    #[tokio::test]
    async fn cut_inside_a_user_message_extends_it_with_filler() {
        let r = FakeRenderer::default();
        let mut s = SessionPrompt::default();
        first(&r, &mut s, 14).await;
        s.record_reply(&r, "r1 r2 r3".to_string()).await.unwrap();
        let reference = reference_of(&s);
        let b = s
            .build(&r, None, 7, 20, &pool(), &mut call_rng(1, "s", 1))
            .await
            .unwrap();
        assert_eq!(b.messages.len(), 1);
        assert_eq!(shared(&r, &reference, &b).await, 7);
        assert_eq!(b.prompt_tokens, 20);
    }

    #[tokio::test]
    async fn reuse_past_the_reference_is_a_shortfall() {
        let r = FakeRenderer::default();
        let mut s = SessionPrompt::default();
        first(&r, &mut s, 14).await;
        s.record_reply(&r, "r1 r2 r3".to_string()).await.unwrap();
        let reference = reference_of(&s);
        // Reference is 14 + 3 = 17 tokens.
        let b = s
            .build(&r, None, 20, 30, &pool(), &mut call_rng(1, "s", 1))
            .await
            .unwrap();
        assert_eq!(b.shortfall, 3);
        assert_eq!(shared(&r, &reference, &b).await, 17);
    }

    #[tokio::test]
    async fn no_new_tokens_after_a_reply_overshoots_by_one_message() {
        let r = FakeRenderer::default();
        let mut s = SessionPrompt::default();
        first(&r, &mut s, 14).await;
        s.record_reply(&r, "r1 r2 r3".to_string()).await.unwrap();
        let reference = reference_of(&s);
        // Keep everything (17) and ask for 17: the reply adds <end>, a new user
        // message adds <user> word <end>, and the generation prompt adds
        // <assistant>.
        let b = s
            .build(&r, None, 17, 17, &pool(), &mut call_rng(1, "s", 1))
            .await
            .unwrap();
        assert!(b.overshoot);
        assert_eq!(shared(&r, &reference, &b).await, 17);
        assert_eq!(b.prompt_tokens, 22);
    }

    #[tokio::test]
    async fn reply_with_surrounding_whitespace_is_still_located() {
        // A reply ending in a space is rendered trimmed in history. The next
        // call must still find it, and the call after must still find the
        // messages that follow it.
        let r = FakeRenderer::default();
        let mut s = SessionPrompt::default();
        first(&r, &mut s, 14).await;
        let reply: String = (0..400).map(|i| format!(" r{i}")).collect::<String>() + " ";
        s.record_reply(&r, reply.trim_start().to_string())
            .await
            .unwrap();
        let reference = reference_of(&s);
        // Keep the whole previous prompt and reply: 14 + 400 words + the
        // trailing space token. History renders the reply trimmed, so the
        // space is not shared.
        let b = s
            .build(&r, None, 415, 430, &pool(), &mut call_rng(1, "s", 1))
            .await
            .unwrap();
        assert_eq!(shared(&r, &reference, &b).await, 414);
        assert!(s.segments.iter().all(|seg| seg.start.is_some()));
        s.record_reply(&r, "x1 x2".to_string()).await.unwrap();
        let reference = reference_of(&s);
        // 425 lands inside the third message (user filler after the reply).
        let b = s
            .build(&r, None, 425, 440, &pool(), &mut call_rng(1, "s", 2))
            .await
            .unwrap();
        assert_eq!(shared(&r, &reference, &b).await, 425);
        let roles: Vec<&str> = b.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["user", "assistant", "user"]);
    }

    #[tokio::test]
    async fn system_prompt_is_kept_whole() {
        let r = FakeRenderer::default();
        let sys = Message {
            role: "system".to_string(),
            content: "s1 s2 s3 s4".to_string(),
        };
        let mut s = SessionPrompt::default();
        let b = s
            .build(&r, Some(&sys), 0, 30, &pool(), &mut call_rng(1, "s", 0))
            .await
            .unwrap();
        assert_eq!(b.messages[0].content, "s1 s2 s3 s4");
        assert_eq!(b.prompt_tokens, 30);
        s.record_reply(&r, "r1".to_string()).await.unwrap();
        let reference = reference_of(&s);
        // A reuse inside the system prompt still keeps it whole.
        let b = s
            .build(&r, Some(&sys), 3, 30, &pool(), &mut call_rng(1, "s", 1))
            .await
            .unwrap();
        assert_eq!(b.messages[0].content, "s1 s2 s3 s4");
        assert!(shared(&r, &reference, &b).await >= 6);
    }
}
