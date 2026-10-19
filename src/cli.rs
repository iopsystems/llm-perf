use clap::{Parser, Subcommand};
use std::path::PathBuf;

use crate::mmlu_pro::config::PromptMode;

#[derive(Parser, Debug)]
#[command(name = "llm-perf")]
#[command(author, version, about = "Benchmark OpenAI-compatible LLM servers", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Run a benchmark against an LLM server
    Bench {
        /// Path to the TOML configuration file
        config: PathBuf,
    },
    /// Collect token-level log probabilities sequentially (one request at a time)
    Logprobs {
        /// Path to the TOML configuration file
        config: PathBuf,
    },
    /// Compare token probability distributions between two logprob captures
    KlDivergence {
        /// Path to baseline logprobs JSONL file
        baseline: PathBuf,
        /// Path to candidate logprobs JSONL file
        candidate: PathBuf,
        /// Output format: "console" or "json"
        #[arg(long, default_value = "console")]
        format: String,
        /// Output file path (writes to stdout if omitted)
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Run the MMLU-Pro accuracy benchmark
    MmluPro {
        /// Path to the TOML configuration file
        config: PathBuf,
        /// Server URL (overrides config endpoint.base_url)
        #[arg(short = 'u', long)]
        url: Option<String>,
        /// API key (overrides config endpoint.api_key)
        #[arg(short = 'a', long = "api")]
        api_key: Option<String>,
        /// Model name (overrides config; auto-detected if omitted)
        #[arg(short, long)]
        model: Option<String>,
        /// Request timeout in seconds (overrides config)
        #[arg(long)]
        timeout: Option<u64>,
        /// Single category to test (overrides config)
        #[arg(long)]
        category: Option<String>,
        /// Fraction of items to keep per category, 0.0-1.0 (overrides config)
        #[arg(long)]
        subset: Option<f64>,
        /// Number of concurrent requests (overrides config)
        #[arg(short = 'p', long)]
        concurrent_requests: Option<usize>,
        /// Number of few-shot examples, 0 for zero-shot (overrides config)
        #[arg(long)]
        num_shots: Option<usize>,
        /// Prompt mode (overrides config inference.mode)
        #[arg(long, value_enum)]
        mode: Option<PromptMode>,
        /// Context length available to one request, in tokens (overrides config
        /// inference.max_context_tokens)
        ///
        /// For llama-server this is `default_generation_settings.n_ctx` from
        /// `GET /props`. This can be less than `-c`. With `-np` greater than 1
        /// and without `--kv-unified`, it is `-c` divided by `-np`, rounded up
        /// to a multiple of 256. `--kv-unified-per-slot` and the model's
        /// training context also cap it.
        ///
        /// Drops shots per question until the prompt plus max_tokens fits;
        /// needs llama-server's /tokenize (and /apply-template in chat mode).
        #[arg(long)]
        max_context_tokens: Option<u32>,
        /// Verbosity level 0-2 (overrides config)
        #[arg(short, long)]
        verbosity: Option<u8>,
        /// Log exact prompts in result files
        #[arg(long)]
        log_prompt: bool,
        /// Comment to include in the report
        #[arg(long)]
        comment: Option<String>,
    },
    /// Generate prompts from config and save to JSONL file
    GeneratePrompts {
        /// Path to the TOML configuration file
        config: PathBuf,
        /// Path to output JSONL file
        output: PathBuf,
    },
    /// Convert a coding-agent session trace (Azure public dataset,
    /// GitHubCopilotCodingAgentDataset2026) into a per-session replay trace
    ConvertTrace {
        /// Input files or directories: per-day .tar.gz archives, .jsonl.gz or
        /// .jsonl shards. Directories are searched recursively.
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        /// Output JSONL file (writes to stdout if omitted)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Keep only sessions in which every call's model label (e.g. "Model E")
        /// is one of these; repeatable
        #[arg(long = "model")]
        models: Vec<String>,
        /// Drop sessions with any call whose prompt + completion tokens exceed this
        #[arg(long)]
        max_context: Option<u64>,
        /// Idle gap in seconds at or above which a call whose cached tokens
        /// fall below the --evicted-reuse-ratio estimate gets the estimate as
        /// its reuse. 0 applies this to every call after the first
        #[arg(long, default_value_t = 300)]
        cache_ttl_secs: u64,
        /// Fraction of min(previous prompt, prompt) used as the estimated
        /// reuse. Must be in [0, 1]; 0 disables the estimate
        #[arg(long, default_value_t = 0.98)]
        evicted_reuse_ratio: f64,
        /// Keep only sessions starting at or after this RFC 3339 time
        /// (e.g. 2026-06-03T00:00:00Z)
        #[arg(long)]
        from: Option<chrono::DateTime<chrono::Utc>>,
        /// Keep only sessions starting before this RFC 3339 time
        /// (e.g. 2026-06-04T00:00:00Z)
        #[arg(long)]
        to: Option<chrono::DateTime<chrono::Utc>>,
        /// Worker threads (defaults to the number of CPUs)
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Match a replay log against the vLLM KV event log recorded with it
    /// (replay.kv_events): which prompt blocks the server held when each call
    /// was sent and at its first token
    KvResidency {
        /// Per-call replay log (replay.log)
        replay_log: PathBuf,
        /// KV event log (replay.kv_events.log)
        events_log: PathBuf,
        /// Block size the replay log's block_fps were computed at
        /// (replay.kv_events.block_size)
        #[arg(long)]
        block_size: usize,
        /// Output format: "console" or "json"
        #[arg(long, default_value = "console")]
        format: String,
        /// Output file path (writes to stdout if omitted)
        #[arg(long)]
        output: Option<PathBuf>,
        /// Write one JSON line per analysed call to this file
        #[arg(long)]
        calls_out: Option<PathBuf>,
    },
}

impl Cli {
    pub fn parse_args() -> Self {
        // Preprocess args for backward compatibility:
        // If first arg isn't a known subcommand and looks like a config file, inject "bench"
        let args: Vec<String> = std::env::args().collect();

        if args.len() >= 2 {
            let first_arg = &args[1];
            // If the first arg is not a known subcommand and not a flag, treat it as bench config
            if !matches!(
                first_arg.as_str(),
                "bench"
                    | "logprobs"
                    | "kl-divergence"
                    | "mmlu-pro"
                    | "generate-prompts"
                    | "convert-trace"
                    | "kv-residency"
                    | "help"
                    | "--help"
                    | "-h"
                    | "--version"
                    | "-V"
            ) {
                let mut new_args = vec![args[0].clone(), "bench".to_string()];
                new_args.extend_from_slice(&args[1..]);
                return Cli::parse_from(new_args);
            }
        }

        Cli::parse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mmlu_mode(args: &[&str]) -> Result<Option<PromptMode>, clap::Error> {
        let mut argv = vec!["llm-perf", "mmlu-pro", "config.toml"];
        argv.extend_from_slice(args);
        match Cli::try_parse_from(argv)?.command {
            Command::MmluPro { mode, .. } => Ok(mode),
            other => panic!("parsed as {other:?}"),
        }
    }

    #[test]
    fn mmlu_pro_mode_flag_is_optional() {
        assert_eq!(mmlu_mode(&[]).unwrap(), None);
    }

    #[test]
    fn mmlu_pro_mode_flag_parses_both_modes() {
        assert_eq!(
            mmlu_mode(&["--mode", "completion"]).unwrap(),
            Some(PromptMode::Completion)
        );
        assert_eq!(
            mmlu_mode(&["--mode", "chat"]).unwrap(),
            Some(PromptMode::Chat)
        );
    }

    #[test]
    fn mmlu_pro_max_context_tokens_flag() {
        let parse = |args: &[&str]| {
            let mut argv = vec!["llm-perf", "mmlu-pro", "config.toml"];
            argv.extend_from_slice(args);
            match Cli::try_parse_from(argv).unwrap().command {
                Command::MmluPro {
                    max_context_tokens, ..
                } => max_context_tokens,
                other => panic!("parsed as {other:?}"),
            }
        };
        assert_eq!(parse(&[]), None);
        assert_eq!(parse(&["--max-context-tokens", "2048"]), Some(2048));
    }

    #[test]
    fn mmlu_pro_mode_flag_rejects_unknown_values() {
        assert!(mmlu_mode(&["--mode", "raw"]).is_err());
    }
}
