use clap::{Parser, Subcommand};
use std::path::PathBuf;

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
