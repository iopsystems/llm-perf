//! Reproduce the wire body for the apc-mitm config to verify chat_template_kwargs flow.
use llm_perf::client::{ClientConfig, Message, OpenAIClient};
use llm_perf::config::Config;
use std::path::PathBuf;
use std::time::Duration;

#[test]
fn wire_body_matches_expected() {
    let toml_content = r#"
[endpoint]
base_url = "http://127.0.0.1:8001/v1"
timeout = 300
max_tokens = 1
chat_template_kwargs = {enable_thinking = false}

[input]
file = "synthetic"
sample_size = 10000

[input.synthetic]
prompt_tokens = 1000
common_prefix_sample_ratio = 1.0
common_prefix_tokens = 1000

[load]
arrival_distribution = "poisson"
concurrent_requests = 4
duration_seconds = 15
warmup_duration = 0

[metrics]
interval = "1s"
output = "llmperf.parquet"

[output]
file = "results.json"
format = "json"
"#;
    let path = PathBuf::from("/tmp/test_cfg_wire.toml");
    std::fs::write(&path, toml_content).unwrap();
    let cfg = Config::load(&path).unwrap();

    eprintln!(
        "config.endpoint.chat_template_kwargs = {:?}",
        cfg.endpoint.chat_template_kwargs
    );

    let client = OpenAIClient::new(ClientConfig {
        base_url: cfg.endpoint.base_url.clone(),
        api_key: cfg.endpoint.api_key.clone(),
        model: "Qwen/Qwen3-1.7B".to_string(),
        timeout: Duration::from_secs(cfg.endpoint.timeout),
        max_retries: cfg.endpoint.max_retries,
        retry_initial_delay_ms: cfg.endpoint.retry_initial_delay_ms,
        retry_max_delay_ms: cfg.endpoint.retry_max_delay_ms,
        pool_size: 4,
        stream_idle_timeout: None,
        retry_on_timeout: false,
        chat_template_kwargs: cfg.endpoint.chat_template_kwargs.clone(),
        ignore_eos: cfg.endpoint.ignore_eos,
    })
    .unwrap();

    let messages = vec![Message {
        role: "user".to_string(),
        content: "hello world".to_string(),
    }];
    let request = client.create_messages_request(&messages, Some(1), None, None);
    let body = serde_json::to_string(&request).unwrap();
    eprintln!("WIRE BODY: {}", body);
    assert!(
        body.contains("chat_template_kwargs"),
        "chat_template_kwargs missing from wire body: {}",
        body
    );
    assert!(
        body.contains("enable_thinking"),
        "enable_thinking missing from wire body: {}",
        body
    );
}
