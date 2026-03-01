use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
	#[serde(default)]
	pub comment: String,
	pub server: ServerConfig,
	pub inference: InferenceConfig,
	pub test: TestConfig,
	#[serde(default)]
	pub log: LogConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
	pub url: String,
	#[serde(default = "default_api_key")]
	pub api_key: String,
	pub model: String,
	#[serde(default = "default_timeout")]
	pub timeout: f64,
}

fn default_api_key() -> String {
	"api key".to_string()
}

fn default_timeout() -> f64 {
	600.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct InferenceConfig {
	#[serde(default)]
	pub temperature: f32,
	#[serde(default = "default_top_p")]
	pub top_p: f32,
	#[serde(default = "default_max_tokens")]
	pub max_tokens: u32,
	#[serde(default = "default_system_prompt")]
	pub system_prompt: String,
}

fn default_top_p() -> f32 {
	1.0
}

fn default_max_tokens() -> u32 {
	4096
}

fn default_system_prompt() -> String {
	"The following are multiple choice questions (with answers) about {subject}. Think step by \
	 step and then finish your answer with \"the answer is (X)\" where X is the correct letter \
	 choice."
		.to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct TestConfig {
	#[serde(default = "default_categories")]
	pub categories: Vec<String>,
	#[serde(default = "default_subset")]
	pub subset: f64,
	#[serde(default = "default_parallel")]
	pub parallel: usize,
}

fn default_categories() -> Vec<String> {
	vec!["all".to_string()]
}

fn default_subset() -> f64 {
	1.0
}

fn default_parallel() -> usize {
	1
}

#[derive(Debug, Clone, Deserialize)]
pub struct LogConfig {
	#[serde(default)]
	pub verbosity: u8,
	#[serde(default)]
	pub log_prompt: bool,
}

impl Default for LogConfig {
	fn default() -> Self {
		Self {
			verbosity: 0,
			log_prompt: false,
		}
	}
}

impl Config {
	pub fn load(path: &Path) -> Result<Self> {
		let content = std::fs::read_to_string(path)?;
		let config: Config = toml::from_str(&content)?;
		Ok(config)
	}
}
