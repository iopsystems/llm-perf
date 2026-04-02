use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub comment: String,
    pub endpoint: EndpointConfig,
    pub inference: InferenceConfig,
    pub load: LoadConfig,
    #[serde(default)]
    pub log: LogConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EndpointConfig {
    pub base_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default = "default_timeout")]
    pub timeout: u64,
}

fn default_timeout() -> u64 {
    600
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
pub struct LoadConfig {
    #[serde(default = "default_categories")]
    pub categories: Vec<String>,
    #[serde(default = "default_subset")]
    pub subset: f64,
    #[serde(default = "default_concurrent_requests")]
    pub concurrent_requests: usize,
}

fn default_categories() -> Vec<String> {
    vec!["all".to_string()]
}

fn default_subset() -> f64 {
    1.0
}

fn default_concurrent_requests() -> usize {
    1
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct LogConfig {
    #[serde(default)]
    pub verbosity: u8,
    #[serde(default)]
    pub log_prompt: bool,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
}
