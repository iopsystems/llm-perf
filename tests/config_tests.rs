use llm_perf::config::{Config, ReplayServer};
use std::path::PathBuf;

fn base_toml() -> &'static str {
    r#"
[endpoint]
base_url = "http://localhost:8080/v1"

[load]
total_requests = 10
concurrent_requests = 1

[input]
file = "examples/prompts/openorca-10000.jsonl"

[output]
format = "console"
"#
}

#[test]
fn system_prompt_inline_parses() {
    let toml = format!(
        "{}\n[input.system_prompt]\ncontent = \"You are helpful\"\n",
        base_toml()
    );
    let path = PathBuf::from("/tmp/sp_inline.toml");
    std::fs::write(&path, &toml).unwrap();
    let cfg = Config::load(&path).unwrap();
    let sp = cfg.input.unwrap().system_prompt.unwrap();
    assert_eq!(sp.content.as_deref(), Some("You are helpful"));
    assert!(sp.file.is_none());
    assert!(sp.tokens.is_none());
}

#[test]
fn system_prompt_tokens_parses() {
    let toml = format!("{}\n[input.system_prompt]\ntokens = 512\n", base_toml());
    let path = PathBuf::from("/tmp/sp_tokens.toml");
    std::fs::write(&path, &toml).unwrap();
    let cfg = Config::load(&path).unwrap();
    let sp = cfg.input.unwrap().system_prompt.unwrap();
    assert_eq!(sp.tokens, Some(512));
}

#[test]
fn system_prompt_multiple_sources_is_error() {
    let toml = format!(
        "{}\n[input.system_prompt]\ncontent = \"hi\"\ntokens = 64\n",
        base_toml()
    );
    let path = PathBuf::from("/tmp/sp_multi.toml");
    std::fs::write(&path, &toml).unwrap();
    assert!(Config::load(&path).is_err());
}

#[test]
fn shared_prefix_parses() {
    let toml = format!(
        "{}\n[input.shared_prefix]\ntokens = 1024\nmiss_rate = 0.1\n",
        base_toml()
    );
    let path = PathBuf::from("/tmp/pfx.toml");
    std::fs::write(&path, &toml).unwrap();
    let cfg = Config::load(&path).unwrap();
    let pfx = cfg.input.unwrap().shared_prefix.unwrap();
    assert_eq!(pfx.tokens, Some(1024));
    assert!((pfx.miss_rate - 0.1).abs() < 1e-9);
}

#[test]
fn shared_prefix_miss_rate_defaults_to_zero() {
    let toml = format!("{}\n[input.shared_prefix]\ntokens = 512\n", base_toml());
    let path = PathBuf::from("/tmp/pfx_default.toml");
    std::fs::write(&path, &toml).unwrap();
    let cfg = Config::load(&path).unwrap();
    assert_eq!(cfg.input.unwrap().shared_prefix.unwrap().miss_rate, 0.0);
}

#[test]
fn shared_prefix_miss_rate_out_of_range_is_error() {
    let toml = format!(
        "{}\n[input.shared_prefix]\ntokens = 512\nmiss_rate = 1.5\n",
        base_toml()
    );
    let path = PathBuf::from("/tmp/pfx_bad.toml");
    std::fs::write(&path, &toml).unwrap();
    assert!(Config::load(&path).is_err());
}

#[test]
fn unknown_field_in_input_is_error() {
    let toml = base_toml().replace("[input]\nfile", "[input]\nunknown_field = true\nfile");
    let path = PathBuf::from("/tmp/unknown.toml");
    std::fs::write(&path, &toml).unwrap();
    assert!(Config::load(&path).is_err());
}

#[test]
fn unknown_field_at_top_level_is_error() {
    let toml = format!("{}\n[bogus_section]\nfoo = 1\n", base_toml());
    let path = PathBuf::from("/tmp/bogus_section.toml");
    std::fs::write(&path, &toml).unwrap();
    assert!(Config::load(&path).is_err());
}

#[test]
fn cache_busting_field_is_error() {
    let toml = base_toml().replace("[input]\nfile", "[input]\ncache_busting = false\nfile");
    let path = PathBuf::from("/tmp/cb_field.toml");
    std::fs::write(&path, &toml).unwrap();
    assert!(Config::load(&path).is_err());
}

#[test]
fn system_prompt_empty_table_is_error() {
    let toml = format!("{}\n[input.system_prompt]\n", base_toml());
    let path = PathBuf::from("/tmp/sp_empty.toml");
    std::fs::write(&path, &toml).unwrap();
    assert!(Config::load(&path).is_err());
}

fn replay_toml(extra: &str) -> String {
    format!(
        r#"
[endpoint]
base_url = "http://localhost:8080/v1"

[output]
format = "console"

[replay]
trace = "trace.jsonl"
sample = 0.01
speedup = 24.0
{extra}
"#
    )
}

#[test]
fn replay_parses_without_input_or_load() {
    let cfg = Config::from_toml(&replay_toml("")).unwrap();
    let replay = cfg.replay.unwrap();
    assert!(cfg.input.is_none());
    assert_eq!(replay.sample, 0.01);
    assert_eq!(replay.speedup, 24.0);
    assert_eq!(replay.seed, 1);
    assert_eq!(replay.system_prompt_tokens, 0);
    assert_eq!(replay.server, ReplayServer::LlamaServer);
}

#[test]
fn replay_server_selects_vllm_and_rejects_unknown() {
    let cfg = Config::from_toml(&replay_toml("server = \"vllm\"")).unwrap();
    assert_eq!(cfg.replay.unwrap().server, ReplayServer::Vllm);
    let cfg = Config::from_toml(&replay_toml("server = \"llama-server\"")).unwrap();
    assert_eq!(cfg.replay.unwrap().server, ReplayServer::LlamaServer);
    assert!(Config::from_toml(&replay_toml("server = \"sglang\"")).is_err());
}

#[test]
fn replay_kv_events_needs_vllm_a_call_log_and_its_own_file() {
    let kv = |log: &str, block_size: u32| {
        format!(
            "\n[replay.kv_events]\nendpoint = \"tcp://127.0.0.1:5557\"\nblock_size = {block_size}\nlog = \"{log}\"\n"
        )
    };
    let ok = replay_toml(&format!(
        "server = \"vllm\"\nlog = \"r.jsonl\"{}",
        kv("kv.jsonl", 528)
    ));
    let cfg = Config::from_toml(&ok).unwrap();
    let k = cfg.replay.unwrap().kv_events.unwrap();
    assert_eq!((k.block_size, k.topic.as_str()), (528, ""));
    for (extra, needle) in [
        (
            format!("log = \"r.jsonl\"{}", kv("kv.jsonl", 528)),
            "requires replay.server",
        ),
        (
            format!("server = \"vllm\"{}", kv("kv.jsonl", 528)),
            "requires replay.log",
        ),
        (
            format!("server = \"vllm\"\nlog = \"r.jsonl\"{}", kv("kv.jsonl", 0)),
            "block_size must be greater than 0",
        ),
        (
            format!("server = \"vllm\"\nlog = \"r.jsonl\"{}", kv("r.jsonl", 528)),
            "must differ from replay.log",
        ),
        (
            format!(
                "server = \"vllm\"\nlog = \"r.jsonl\"{}",
                kv("./r.jsonl", 528)
            ),
            "must differ from replay.log",
        ),
    ] {
        let err = Config::from_toml(&replay_toml(&extra))
            .unwrap_err()
            .to_string();
        assert!(err.contains(needle), "{extra}: {err}");
    }
}

#[test]
fn replay_accepts_duration_and_warmup_from_load() {
    let toml = format!(
        "{}\n[load]\nduration_seconds = 600\nwarmup_duration = 60\n",
        replay_toml("")
    );
    let cfg = Config::from_toml(&toml).unwrap();
    assert_eq!(cfg.load.duration_seconds, Some(600));
}

#[test]
fn replay_rejects_other_load_keys_even_at_their_default() {
    // concurrent_requests = 10 is the serde default; it is still rejected.
    let toml = format!("{}\n[load]\nconcurrent_requests = 10\n", replay_toml(""));
    let err = Config::from_toml(&toml).unwrap_err().to_string();
    assert!(err.contains("load.concurrent_requests"), "{err}");
}

#[test]
fn replay_rejects_input_section() {
    let toml = format!("{}\n[input]\nfile = \"synthetic\"\n", replay_toml(""));
    let err = Config::from_toml(&toml).unwrap_err().to_string();
    assert!(
        err.contains("[input] cannot be used with [replay]"),
        "{err}"
    );
}

#[test]
fn replay_rejects_retries_max_tokens_and_bad_ranges() {
    for (patch, needle) in [
        ("max_retries = 1", "max_retries"),
        ("max_tokens = 64", "max_tokens"),
        ("ignore_eos = false", "ignore_eos"),
    ] {
        let toml = replay_toml("").replace(
            "base_url = \"http://localhost:8080/v1\"",
            &format!("base_url = \"http://localhost:8080/v1\"\n{patch}"),
        );
        let err = Config::from_toml(&toml).unwrap_err().to_string();
        assert!(err.contains(needle), "{patch}: {err}");
    }
    for (value, needle) in [
        ("sample = 0.0", "replay.sample"),
        ("sample = 1.5", "replay.sample"),
    ] {
        let toml = replay_toml("").replace("sample = 0.01", value);
        let err = Config::from_toml(&toml).unwrap_err().to_string();
        assert!(err.contains(needle), "{value}: {err}");
    }
    for value in ["speedup = 0.0", "speedup = inf"] {
        let toml = replay_toml("").replace("speedup = 24.0", value);
        let err = Config::from_toml(&toml).unwrap_err().to_string();
        assert!(err.contains("replay.speedup"), "{value}: {err}");
    }
    let toml = replay_toml("").replace("sample = 0.01", "sample = nan");
    let err = Config::from_toml(&toml).unwrap_err().to_string();
    assert!(err.contains("replay.sample"), "{err}");
}

#[test]
fn replay_rejects_settings_it_would_ignore() {
    for (patch, needle) in [
        ("tokenizer = \"gpt2\"", "endpoint.tokenizer"),
        ("retry_on_timeout = true", "retry_on_timeout"),
    ] {
        let toml = replay_toml("").replace(
            "base_url = \"http://localhost:8080/v1\"",
            &format!("base_url = \"http://localhost:8080/v1\"\n{patch}"),
        );
        let err = Config::from_toml(&toml).unwrap_err().to_string();
        assert!(err.contains(needle), "{patch}: {err}");
    }
    for (load, needle) in [
        (
            "duration_seconds = 0",
            "duration_seconds must be greater than 0",
        ),
        (
            "duration_seconds = 60\nwarmup_duration = 60",
            "warmup_duration (60) must be less than",
        ),
    ] {
        let toml = format!("{}\n[load]\n{load}\n", replay_toml(""));
        let err = Config::from_toml(&toml).unwrap_err().to_string();
        assert!(err.contains(needle), "{load}: {err}");
    }
}

#[test]
fn replay_rejects_other_mode_sections() {
    for (section, body) in [
        ("[saturation]", "[saturation.slo.ttft]\np99_ms = 100.0"),
        ("[metrics]", "[metrics]\noutput = \"m.parquet\""),
        ("[logprobs]", "[logprobs]\noutput = \"l.jsonl\""),
        ("[conversation]", "[conversation]\nturn_delay_ms = 10"),
    ] {
        let toml = format!("{}\n{body}\n", replay_toml(""));
        let err = Config::from_toml(&toml).unwrap_err().to_string();
        assert!(
            err.contains(&format!("{section} cannot be used with [replay]")),
            "{section}: {err}"
        );
    }
}

#[test]
fn bench_without_input_or_replay_is_rejected() {
    let toml = r#"
[endpoint]
base_url = "http://localhost:8080/v1"

[load]
total_requests = 10

[output]
format = "console"
"#;
    let err = Config::from_toml(toml).unwrap_err().to_string();
    assert!(err.contains("[input] section is required"), "{err}");
}

#[test]
fn pool_idle_timeout_defaults_to_two_seconds_and_can_be_set() {
    let cfg = Config::from_toml(base_toml()).unwrap();
    assert_eq!(cfg.endpoint.pool_idle_timeout_ms, 2000);
    let toml = base_toml().replace(
        "base_url = \"http://localhost:8080/v1\"",
        "base_url = \"http://localhost:8080/v1\"\npool_idle_timeout_ms = 60000",
    );
    let cfg = Config::from_toml(&toml).unwrap();
    assert_eq!(cfg.endpoint.pool_idle_timeout_ms, 60000);
}
