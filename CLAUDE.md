# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`llm-perf` is a benchmarking tool for OpenAI-compatible LLM servers. It's designed to measure performance characteristics of local LLM inference servers like llama-server, vLLM, TGI, etc. The project uses Rust edition 2024 with async/await via Tokio.

Also includes an `mmlu-pro` subcommand for MMLU-Pro accuracy evaluation.

## Common Development Commands

### Build the project
```bash
cargo build
```

### Run the project
```bash
cargo run -- config.example.toml
```

### Build in release mode (recommended for benchmarking)
```bash
cargo build --release
./target/release/llm-perf config.toml
```

### Run tests
```bash
cargo test
```

### Check code without building
```bash
cargo check
```

### Format code
```bash
cargo fmt
```

### Run linter
```bash
cargo clippy
```

## Project Structure

- `Cargo.toml` - Project manifest with dependencies (one binary, llm-perf)
- `src/main.rs` - Entry point, CLI dispatch, tokio runtime and logging setup
- `src/lib.rs` - Library root that exports public modules
- `src/cli.rs` - Command-line interface using clap (bench, logprobs, kl-divergence, mmlu-pro, generate-prompts, convert-trace, kv-residency subcommands)
- `src/config.rs` - Configuration structures with TOML parsing
- `src/benchmark.rs` - Core benchmarking engine (concurrent and QPS modes)
- `src/client.rs` - OpenAI-compatible HTTP client with SSE streaming, retries, model detection
- `src/metrics.rs` - Metric declarations using metriken (counters, gauges, histograms)
- `src/report.rs` - Report generation (console and JSON output)
- `src/stats.rs` - Periodic windowed stats output during benchmark runs
- `src/snapshot.rs` - Metrics snapshot to parquet pipeline
- `src/recording.rs` - The run's `.dendro` archive (llm-perf and the server under test), recorded in process with metriken-recorder
- `src/admin.rs` - HTTP admin server for live metrics (Prometheus/JSON, msgpack `/metrics/binary`, dendro replication `/metrics/stream`)
- `src/distribution.rs` - Request arrival distributions (uniform, poisson)
- `src/tokenizer.rs` - Token counting via tiktoken
- `src/logprobs.rs` - Log probability collection and JSONL writer
- `src/kl_divergence.rs` - KL divergence computation between logprob captures
- `src/trace.rs` - `convert-trace`: coding-agent session dataset to per-session replay trace
- `src/replay/` - Trace replay (`[replay]` config): session sampling, scaled schedule, filler, prompt construction against llama-server or vLLM, server probing, runner, vLLM KV event capture, and the `kv-residency` subcommand
- `src/mmlu_pro/` - MMLU-Pro accuracy evaluation (`mmlu-pro` subcommand)
- `examples/config.example.toml` - Comprehensive configuration reference

## Key Architecture Decisions

1. **Configuration-driven**: Benchmark settings come from the TOML config file; `mmlu-pro` accepts CLI overrides, and the utility subcommands (`kl-divergence`, `convert-trace`) take CLI arguments
2. **Async runtime**: Uses Tokio with configurable worker threads
3. **OpenAI-compatible**: Focuses on `/v1/chat/completions` endpoint
4. **Logging**: Uses ringlog with async ring buffer

## Configuration

The tool expects a TOML configuration file with the following structure:
- `endpoint`: Server URL, model name, timeout, retry settings, health check
- `load`: Concurrency settings, QPS, request count or duration, warmup
- `input`: Path to JSONL file with prompts, shuffle, sample size
- `output`: Format (json/console) and optional file path
- `runtime`: Worker thread count for Tokio
- `log`: Log level and per-module filters
- `metrics`: Optional parquet snapshot capture
- `admin`: Optional live metrics HTTP server
- `logprobs`: Optional log probability collection
