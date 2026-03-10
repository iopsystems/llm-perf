# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`llm-perf` is a benchmarking tool for OpenAI-compatible LLM servers. It's designed to measure performance characteristics of local LLM inference servers like llama-server, vLLM, TGI, etc. The project uses Rust edition 2024 with async/await via Tokio.

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

- `Cargo.toml` - Project manifest with dependencies
- `src/main.rs` - Entry point, sets up tokio runtime and logging
- `src/lib.rs` - Library root that exports public modules
- `src/cli.rs` - Command-line interface using clap
- `src/config.rs` - Configuration structures with TOML parsing
- `config.example.toml` - Example configuration file
- `prompts.example.jsonl` - Example input prompts
- `MVP_PLAN.md` - MVP implementation plan
- `DESIGN.md` - Full design document

## Key Architecture Decisions

1. **Configuration-driven**: All settings come from TOML config file, no CLI overrides
2. **Async runtime**: Uses Tokio with configurable worker threads
3. **OpenAI-compatible**: Focuses on `/v1/chat/completions` endpoint
4. **Structured logging**: Uses tracing for structured logs

## Configuration

The tool expects a TOML configuration file with the following structure:
- `endpoint`: Server URL, model name, timeout
- `load`: Concurrency settings, request count or duration
- `input`: Path to JSONL file with prompts
- `output`: Format (json/csv/console) and optional file path
- `runtime`: Worker thread count for Tokio