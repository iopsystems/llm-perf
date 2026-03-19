# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`llm-perf` is a benchmarking tool for OpenAI-compatible LLM servers. It's designed to measure performance characteristics of local LLM inference servers like llama-server, vLLM, TGI, etc. The project uses Rust edition 2024 with async/await via Tokio.

Also includes `mmlu-pro`, a separate binary for MMLU-Pro accuracy evaluation.

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

- `Cargo.toml` - Project manifest with dependencies (two binaries: llm-perf, mmlu-pro)
- `src/main.rs` - Entry point, CLI dispatch, tokio runtime and logging setup
- `src/lib.rs` - Library root that exports public modules
- `src/cli.rs` - Command-line interface using clap (bench, logprobs, kl-divergence subcommands)
- `src/config.rs` - Configuration structures with TOML parsing
- `src/benchmark.rs` - Core benchmarking engine (concurrent and QPS modes)
- `src/client.rs` - OpenAI-compatible HTTP client with SSE streaming, retries, model detection
- `src/metrics.rs` - Metric declarations using metriken (counters, gauges, histograms)
- `src/report.rs` - Report generation (console and JSON output)
- `src/stats.rs` - Periodic windowed stats output during benchmark runs
- `src/snapshot.rs` - Metrics snapshot to parquet pipeline
- `src/admin.rs` - HTTP admin server for live metrics (Prometheus/JSON)
- `src/distribution.rs` - Request arrival distributions (uniform, poisson)
- `src/tokenizer.rs` - Token counting via tiktoken
- `src/logprobs.rs` - Log probability collection and JSONL writer
- `src/kl_divergence.rs` - KL divergence computation between logprob captures
- `src/mmlu_pro/` - MMLU-Pro accuracy evaluation binary
- `examples/config.example.toml` - Comprehensive configuration reference

## Key Architecture Decisions

1. **Configuration-driven**: All settings come from TOML config file, no CLI overrides
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
