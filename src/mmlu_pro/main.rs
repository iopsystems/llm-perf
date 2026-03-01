use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

mod config;
mod dataset;
mod evaluate;
mod extract;
mod prompt;
mod report;

#[derive(Parser, Debug)]
#[command(name = "mmlu-pro")]
#[command(about = "MMLU-Pro benchmark for OpenAI-compatible LLM servers")]
struct Cli {
	/// Path to the TOML configuration file
	config: PathBuf,

	/// Server URL (overrides config)
	#[arg(short = 'u', long)]
	url: Option<String>,

	/// API key (overrides config)
	#[arg(short = 'a', long = "api")]
	api_key: Option<String>,

	/// Model name (overrides config)
	#[arg(short, long)]
	model: Option<String>,

	/// Request timeout in seconds (overrides config)
	#[arg(long)]
	timeout: Option<f64>,

	/// Single category to test (overrides config)
	#[arg(long)]
	category: Option<String>,

	/// Fraction of items to keep per category, 0.0-1.0 (overrides config)
	#[arg(long)]
	subset: Option<f64>,

	/// Number of parallel requests (overrides config)
	#[arg(short, long)]
	parallel: Option<usize>,

	/// Verbosity level 0-2 (overrides config)
	#[arg(short, long)]
	verbosity: Option<u8>,

	/// Log exact prompts in result files
	#[arg(long)]
	log_prompt: bool,

	/// Comment to include in the report
	#[arg(long)]
	comment: Option<String>,
}

fn main() -> Result<()> {
	let cli = Cli::parse();

	// Load config
	let mut config = config::Config::load(&cli.config)?;

	// Apply CLI overrides
	if let Some(url) = cli.url {
		config.server.url = url;
	}
	if let Some(api_key) = cli.api_key {
		config.server.api_key = api_key;
	}
	if let Some(model) = cli.model {
		config.server.model = model;
	}
	if let Some(timeout) = cli.timeout {
		config.server.timeout = timeout;
	}
	if let Some(category) = cli.category {
		config.test.categories = vec![category];
	}
	if let Some(subset) = cli.subset {
		config.test.subset = subset;
	}
	if let Some(parallel) = cli.parallel {
		config.test.parallel = parallel;
	}
	if let Some(verbosity) = cli.verbosity {
		config.log.verbosity = verbosity;
	}
	if cli.log_prompt {
		config.log.log_prompt = true;
	}
	if let Some(comment) = cli.comment {
		config.comment = comment;
	}

	// Print startup info
	eprintln!("MMLU-Pro Benchmark");
	eprintln!("  Model: {}", config.server.model);
	eprintln!("  URL: {}", config.server.url);
	eprintln!("  Parallel: {}", config.test.parallel);
	eprintln!("  Subset: {}", config.test.subset);
	eprintln!("  Max Tokens: {}", config.inference.max_tokens);
	eprintln!();

	// Create output directory
	let model_dir_name = regex::Regex::new(r"\W")
		.unwrap()
		.replace_all(&config.server.model, "-")
		.to_string();
	let output_dir = PathBuf::from("eval_results").join(&model_dir_name);
	std::fs::create_dir_all(&output_dir)?;

	// Build tokio runtime
	let runtime = tokio::runtime::Builder::new_multi_thread()
		.enable_all()
		.build()?;

	let start = Instant::now();

	// Run evaluation
	let result = runtime.block_on(async {
		eprintln!("Loading MMLU-Pro dataset...");
		let (test_data, val_data) = dataset::load_mmlu_pro(config.test.subset).await?;

		eprintln!(
			"Dataset loaded: {} categories, {} total test questions",
			test_data.len(),
			test_data.values().map(|v| v.len()).sum::<usize>()
		);

		evaluate::run_evaluation(&config, &test_data, &val_data, &output_dir).await
	})?;

	let elapsed = start.elapsed();

	// Generate report
	report::generate_report(
		&config,
		&result.category_stats,
		&result.token_stats,
		elapsed,
		&output_dir,
	);

	Ok(())
}
