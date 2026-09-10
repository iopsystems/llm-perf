# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.1.18] - 2026-09-10

### Bug Fixes

- Report TTFT when a response carries no textual delta (#170) — a reasoning model can spend its whole token budget on the `<think>` open tag, which the server's parser consumes without emitting a delta, leaving the TTFT histogram empty and the report showing `0.0 ms`. TTFT now falls back to the first stream event, and a warning says when that substitution happened.
- Divide throughput by the window the counted work occupied (#169) — `duration_secs` was nominal wall clock while `output_tokens` counted only completed requests, quantizing throughput onto `n x max_tokens / window`.
- Derive teardown grace from the work, and say when it truncates (#168) — a fixed 60s grace bounded the whole test under a closed-loop worker pool, silently dropping requests from `total_requests`.

### Infrastructure

- Drop Debian 11 (bullseye) from the release matrix — EOL 2026-08-31
- Update GitHub Actions to their Node 24 majors — checkout v7, upload-artifact v7, download-artifact v7, action-gh-release v3
- Fix the automated dev-version bump corrupting `Cargo.lock` — it rewrote the first `[[package]]` entry instead of `llm-perf`; repair the current lock

## [0.1.17] - 2026-09-08

### Features

- Capture server-side metrics by shelling out to `rezolus record` (#159)
- Configurable saturation tuning knobs; deprecate `stop_after_failures` (#151)
- Flag chunk-granular ITL when the server batches tokens (#150)
- Real per-model tokenizer for prompt sizing (#149)
- QPS overload cap — shed instead of unbounded queueing (#148)
- Saturation bisect + drain search, marginal-gain plateau, transition flags (#145)
- Configurable stream idle timeout; opt-in, deadline-aware retries (#143)
- New workload model — shared prefix, cache-hit targeting, outcome metrics (#130)
- Synthetic multi-turn conversations (#112)
- Common prefix support for testing vLLM prefix caching (#83)
- Synthetic data generation support (#79)
- Allow `max_tokens` to be specified in the config file (#77)
- Add `ignore_eos` option (#161)
- Add `nothink` option (#120)

### Bug Fixes

- llama.cpp `/tokenize` response shape; retry on connection errors (#161)
- QPS schedule drift, handle reaping, stats window drift, sampling bias (#146)
- Byte-level SSE buffering; surface malformed chunks (#144)
- Prefer server token counts; flag non-OpenAI tokenizer as estimate (#142)
- Measure QPS latency from arrival, add schedule-slip metric (#141)
- Correctness fixes for metrics, extraction, timeouts, resume (#140)
- Warmup mutual-exclusion validation, `turn_prompt_tokens` ceiling, doc fixes (#126)
- Correct system prompt injection, cache busting, and single-turn system prompts (#125)
- Cache busting flag rename (#120)
- Include reasoning tokens in multi-turn conversation history (#115)
- Use a unique prefix when `common_prefix_tokens` is 0 (#85)
- Balance `requests_inflight` via RAII guard (#80)
- Cancel in-flight warmup requests at deadline expiry (#78)

### Changes

- Concurrent fixed-count mode reimplemented as a worker pool (#147)
- Remove `calc/` — extracted to iopsystems/llm-calc (#158)

### Security

- Clear cargo audit advisories (#162)
- Bump quinn-proto 0.11.14 -> 0.11.15 (RUSTSEC-2026-0185) (#155)


## [0.1.16] - 2026-05-01

### Features

- Add system prompt from file support (#73)

### Performance

- Pre-allocate output buffer in admin human_metrics endpoint (#72)
- Pre-allocate capacity for streaming response strings (#71)
- Eliminate `messages.clone()` in multi-turn conversation loop (#70)
- Eliminate workload and system prompt cloning (#69)

### Bug Fixes

- Simplify rpmbuild cleanup command (#67)

### Changes

- Reduce unnecessary allocations

## [0.1.14] - 2026-04-30

### Infrastructure

- Add cross-compile.mk and rustc-architecture.mk for Debian cross-compilation support

## [0.1.13] - 2026-04-30

### Bug Fixes

- Fix Debian packaging for native package format (remove invalid Debian revision)

### Infrastructure

- Add gen-changelog.sh to auto-generate changelog from Cargo.toml
- Update package.sh to generate changelog before dpkg-source build

## [0.1.12] - 2026-04-30

### Features

- Add RPM packaging support for Amazon Linux 2023 (#54)
- Add system prompt override and new datasets (#53)

### Infrastructure

- Update CI workflow with Docker-based deb/rpm builds
- Add RPM spec template for proper header generation
- Implement Debian packaging with Docker-based build scripts

## [0.1.11] - 2026-04-28

### Features

- Add saturation search to find max compliant concurrency (#42)
- Add reasoning model support with phase-aware metrics (#33)
- Split TTFT into separate metrics and add phase-aware token counting (#34)
- Add multi-turn conversation support and dataset auto-download (#36)
- Add configurable shots, penalties, and OpenCompass-style eval for MMLU-Pro (#38, #39)
- Add overall progress tracking across categories for MMLU-Pro (#44)

### Bug Fixes

- Measure throughput with successful requests only (#43)
- Add error tracking, retries, and progress improvements to MMLU-Pro (#40)
- Upgrade metriken-exposition to 0.13 for PromQL-compatible snapshots (#32)
- Fix TTFT and ITL always reported as 0.0 in JSON and console output.
  Aggregate context-aware histogram buckets to produce overall percentiles.
- Use server-reported token counts via `stream_options.include_usage` instead of
  re-tokenizing with tiktoken's `cl100k_base`, which produced inaccurate counts
  for non-OpenAI models (Llama, Qwen, Mistral, etc.). Falls back to tiktoken
  when the server doesn't support it.
- Add TTFT and ITL to console and brief summary output.
- Fix SSE streaming parser to handle multiple events batched in a single HTTP
  chunk. Previously only the first event was processed and the rest were silently
  dropped, causing lost response content and underreported token/s — especially
  at low concurrency where servers like llama.cpp may batch multiple SSE events
  into one TCP segment.
- Handle partial SSE lines split across HTTP chunk boundaries.

### Changes

- Rename project from llm-bench to llm-perf (#27)
- Convert mmlu-pro from separate binary to subcommand (#29)
- Refactor request status tracking to distinguish errors, timeouts, and cancellations (#41)
- Align mmlu-pro config with benchmark conventions (#37)
- Replace ringlog with tracing-appender for non-blocking logging (#35)
- Add canonical formatter for unique parquet column names (#30)
- Upgrade reqwest from 0.11 to 0.12, removing duplicate dependency and unmaintained rustls-pemfile
- Update quinn-proto to 0.11.14 to fix RUSTSEC-2026-0037
- Housekeeping cleanup — docs, dead code, and bug fixes (#28)
- Add `logprobs` subcommand for sequential token-level log probability collection (one request at a time, no concurrent load) to avoid GPU batching non-determinism
- Add `kl-divergence` subcommand to compare token probability distributions between two logprob JSONL captures
- Add logprobs streaming support to OpenAI client (TokenLogprob, TopLogprob, ChoiceLogprobs types)
- Add CLI subcommand architecture with backward-compatible config file argument
- Replace OpenSSL with rustls for TLS, eliminating the native OpenSSL/native-tls dependency chain
- Use prebuilt cargo-audit binary in CI
- Gitignore generated prompt files
- Add PR skill and update release skill for fork workflow
- Add CI workflow with fmt, clippy, doc, audit, test, and test-release jobs
- Track Cargo.lock for reproducible builds
- Update dependencies to resolve security advisories (bytes, slab, time)
- Fix doc examples referencing renamed `Config::from_file` method
- Apply rustfmt formatting fixes
- Check server readiness on launch before starting benchmark
- Remove redundant overall TTFT/ITL histograms in favor of context-aware variants
- Fix collapsible if statements for clippy compliance
- Add release and tag-release GitHub Actions workflows
- Add /release skill for creating release PRs
- Add mmlu-pro binary to deb and rpm packaging
- Fixed success rate calculation in benchmark reports. Previously, success rate was calculated as `successful_requests / total_sent_requests`, which included in-flight requests that hadn't completed yet in duration-based tests. Now correctly calculated as `successful_requests / completed_requests`, providing an accurate success rate for completed requests only.

### Infrastructure

- Add CI workflow with fmt, clippy, doc, audit, test, and test-release jobs
- Track Cargo.lock for reproducible builds
- Update dependencies to resolve security advisories (bytes, slab, time)
- Fix doc examples referencing renamed `Config::from_file` method
- Apply rustfmt formatting fixes
- Check server readiness on launch before starting benchmark
- Remove redundant overall TTFT/ITL histograms in favor of context-aware variants
- Fix collapsible if statements for clippy compliance
- Add release and tag-release GitHub Actions workflows
- Add /release skill for creating release PRs
- Add mmlu-pro binary to deb and rpm packaging
- Fix tag-release workflow to match squash-merge commit message format

## [0.1.10] - 2026-04-15

### Features

- Add saturation search to find max compliant concurrency (#42)
- Add reasoning model support with phase-aware metrics (#33)
- Split TTFT into separate metrics and add phase-aware token counting (#34)
- Add multi-turn conversation support and dataset auto-download (#36)
- Add configurable shots, penalties, and OpenCompass-style eval for MMLU-Pro (#38, #39)
- Add overall progress tracking across categories for MMLU-Pro (#44)

### Bug Fixes

- Measure throughput with successful requests only (#43)
- Add error tracking, retries, and progress improvements to MMLU-Pro (#40)
- Upgrade metriken-exposition to 0.13 for PromQL-compatible snapshots (#32)

### Changes

- Rename project from llm-bench to llm-perf (#27)
- Convert mmlu-pro from separate binary to subcommand (#29)
- Refactor request status tracking to distinguish errors, timeouts, and cancellations (#41)
- Align mmlu-pro config with benchmark conventions (#37)
- Replace ringlog with tracing-appender for non-blocking logging (#35)
- Add canonical formatter for unique parquet column names (#30)
- Upgrade reqwest from 0.11 to 0.12, removing duplicate dependency and unmaintained rustls-pemfile
- Update quinn-proto to 0.11.14 to fix RUSTSEC-2026-0037
- Housekeeping cleanup — docs, dead code, and bug fixes (#28)

## [0.1.9] - 2026-03-03

### Features

- Add `logprobs` subcommand for sequential token-level log probability collection (one request at a time, no concurrent load) to avoid GPU batching non-determinism
- Add `kl-divergence` subcommand to compare token probability distributions between two logprob JSONL captures
- Add logprobs streaming support to OpenAI client (TokenLogprob, TopLogprob, ChoiceLogprobs types)
- Add CLI subcommand architecture with backward-compatible config file argument

## [0.1.8] - 2026-03-01

### Bug Fixes

- Add mmlu-pro binary to deb and rpm packaging

## [0.1.7] - 2026-03-01

### Changes

- Replace OpenSSL with rustls for TLS, eliminating the native OpenSSL/native-tls dependency chain

## [0.1.6] - 2026-03-01

### Features

- Add MMLU-Pro benchmark test support

### Infrastructure

- Use prebuilt cargo-audit binary in CI
- Gitignore generated prompt files
- Add PR skill and update release skill for fork workflow

## [0.1.5] - 2026-02-27

### Bug Fixes

- Fix TTFT and ITL always reported as 0.0 in JSON and console output.
  Aggregate context-aware histogram buckets to produce overall percentiles.
- Use server-reported token counts via `stream_options.include_usage` instead of
  re-tokenizing with tiktoken's `cl100k_base`, which produced inaccurate counts
  for non-OpenAI models (Llama, Qwen, Mistral, etc.). Falls back to tiktoken
  when the server doesn't support it.
- Add TTFT and ITL to console and brief summary output.

## [0.1.4] - 2026-02-27

### Bug Fixes

- Fix SSE streaming parser to handle multiple events batched in a single HTTP
  chunk. Previously only the first event was processed and the rest were silently
  dropped, causing lost response content and underreported token/s — especially
  at low concurrency where servers like llama.cpp may batch multiple SSE events
  into one TCP segment.
- Handle partial SSE lines split across HTTP chunk boundaries.

### Infrastructure

- Fix tag-release workflow to match squash-merge commit message format

## [0.1.3] - 2026-02-25

### Infrastructure

- Add CI workflow with fmt, clippy, doc, audit, test, and test-release jobs
- Track Cargo.lock for reproducible builds
- Update dependencies to resolve security advisories (bytes, slab, time)

### Bug Fixes

- Fix doc examples referencing renamed `Config::from_file` method
- Apply rustfmt formatting fixes

## [0.1.2] - 2026-02-25

### Improvements

- Check server readiness on launch before starting benchmark
- Remove redundant overall TTFT/ITL histograms in favor of context-aware variants
- Fix collapsible if statements for clippy compliance

### Infrastructure

- Add release and tag-release GitHub Actions workflows
- Add /release skill for creating release PRs

## [0.1.1] - 2025-10-14

### Bug Fixes

- Fixed success rate calculation in benchmark reports. Previously, success rate was calculated as `successful_requests / total_sent_requests`, which included in-flight requests that hadn't completed yet in duration-based tests. Now correctly calculated as `successful_requests / completed_requests`, providing an accurate success rate for completed requests only.

## [0.1.0] - 2025-01-15

### Initial Release

A high-performance benchmarking tool for OpenAI-compatible LLM inference servers. Designed to measure detailed performance characteristics of local LLM servers like llama-server, vLLM, TGI, and other OpenAI API-compatible endpoints.

#### Core Features

- **OpenAI API Compatibility**: Works with any server implementing `/v1/chat/completions` endpoint
- **Streaming Support**: Measures Time-To-First-Token (TTFT) via SSE streaming
- **Async/Concurrent Testing**: Configurable concurrent request handling with Tokio runtime
- **Token Counting**: Built-in tokenizer support for accurate token metrics
- **Automatic Retries**: Exponential backoff with jitter for transient failures

#### Load Patterns

- **Concurrent Mode**: Fixed number of concurrent workers
- **Fixed QPS Mode**: Maintain precise queries per second rate
- **Arrival Distributions**:
  - Uniform: Fixed intervals between requests (deterministic)
  - Poisson: Variable intervals following exponential distribution (realistic traffic)
- **Duration-Based Testing**: Run tests for specified time period
- **Request Count Mode**: Run fixed number of requests
- **Warmup Period**: Optional warmup phase to exclude cold start effects

#### Performance Metrics

- **Time to First Token (TTFT)**: Critical for streaming response UX
  - P50, P90, P95, P99 percentiles
  - Context-aware buckets (small, medium, large, xlarge, xxlarge)
- **Inter-Token Latency (ITL)**: Time between consecutive tokens
  - P50, P90, P95, P99 percentiles
  - Context-aware analysis by input size
- **End-to-End Request Latency**: Total request completion time
  - P50, P90, P95, P99 percentiles
- **Throughput Metrics**: Requests/s, input tokens/s, output tokens/s
- **Error Analysis**: Categorized tracking of connection, HTTP 4xx/5xx, timeout, and parse errors

#### Input Management

- **JSONL Input Format**: Simple prompt format with optional max_tokens
- **Prompt Cycling**: Automatically cycles through prompts for longer tests
- **Sample Size Control**: Limit prompts for quick tests
- **Shuffle Support**: Randomize prompt order for realistic patterns
- **Cache Busting**: Automatic per-request unique IDs to prevent response caching

#### Output Options

- **Console Output**: Clean, formatted results with detailed metrics
- **JSON Export**: Complete structured results with metadata for automation
- **Metrics Capture**: Periodic snapshots in Parquet format for time-series analysis

#### Observability

- **Efficient Logging**: Via `ringlog` with asynchronous ring buffer
- **Configurable Log Levels**: error, warn, info, debug, trace
- **Log File Support**: Automatic rotation at 10MB
- **Progress Indicators**: Real-time progress bars
- **Periodic Stats**: Optional runtime statistics display
- **Admin Metrics Server**: HTTP endpoint exposing metrics in Prometheus/JSON formats

#### Model Support

- **Auto-Detection**: Automatically detect model from server
- **Model Name Normalization**: Intelligent cleanup of paths and formats
- **Multi-Server Support**: Works with llama.cpp, vLLM, TGI, and other OpenAI-compatible servers

#### Configuration

- **TOML-Based**: Simple, readable configuration files
- **Comprehensive Examples**: Multiple scenario examples for different use cases
- **Real-World Datasets**: OpenOrca dataset with 10,000 diverse prompts included

#### Architecture

- **Rust 2024 Edition**: Modern Rust with latest language features
- **High Performance**: Efficient async I/O with Tokio
- **Connection Pooling**: Optimized HTTP client configuration
- **Production Ready**: Comprehensive error handling, proper resource cleanup
- **Well Documented**: Rustdoc comments on public API
