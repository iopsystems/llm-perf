use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "llm-bench")]
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
