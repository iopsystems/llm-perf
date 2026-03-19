# llm-perf

A high-performance benchmarking tool for OpenAI-compatible LLM inference servers. Designed to measure detailed performance characteristics of local LLM servers like llama-server, vLLM, TGI, and other OpenAI API-compatible endpoints.

## Features

### Core Capabilities

- **OpenAI API Compatibility**: Works with any server implementing the `/v1/chat/completions` endpoint
- **Streaming Support**: Measures Time-To-First-Token (TTFT) via SSE streaming
- **Async/Concurrent Testing**: Configurable concurrent request handling with Tokio runtime
- **Token Counting**: Server-reported token counts with tiktoken fallback
- **Automatic Retries**: Exponential backoff with jitter for transient failures (timeouts, connection errors, 5xx errors)
- **Flexible Load Patterns**:
  - **Concurrent Mode**: Fixed number of concurrent workers
  - **Fixed QPS Mode**: Maintain precise queries per second rate
  - **Arrival Distributions**:
    - **Uniform**: Fixed intervals between requests (deterministic)
    - **Poisson**: Variable intervals following exponential distribution (realistic traffic)
  - **Duration-Based Testing**: Run tests for a specified time period
  - **Request Count Mode**: Run a fixed number of requests
  - **Warmup Period**: Optional warmup phase to exclude cold start effects from metrics

### Subcommands

- **`bench`** (default) - Run load benchmarks against an LLM server
- **`logprobs`** - Collect token-level log probabilities sequentially (one request at a time to avoid GPU batching effects on distributions)
- **`kl-divergence`** - Compare token probability distributions between two logprob captures (e.g., baseline FP16 vs quantized model)

### Performance Metrics

#### Error Analysis
- **Categorized Error Tracking**: Detailed breakdown of failures
  - Connection errors (network issues)
  - HTTP 4xx errors (client errors with status codes)
  - HTTP 5xx errors (server errors with status codes)
  - Timeout errors (request exceeded timeout)
  - Parse errors (invalid response format)

#### Latency Measurements
- **Time to First Token (TTFT)**: Critical for streaming response UX
  - Mean, P50, P90, P95, P99 percentiles
  - Context-aware buckets (small, medium, large, xlarge, xxlarge)
- **Time per Output Token (TPOT)**: Average generation speed excluding first token
  - Mean, P50, P90, P95, P99 percentiles
- **Inter-Token Latency (ITL)**: Time between consecutive tokens
  - Mean, P50, P90, P95, P99 percentiles
  - Context-aware analysis (automatically categorized by input size)
- **End-to-End Request Latency**: Total request completion time
  - Mean, P50, P90, P95, P99 percentiles

#### Throughput Metrics
- Requests per second
- Input tokens per second
- Output tokens per second
- Total token counts

#### Context-Aware Analysis
Automatically categorizes TTFT and ITL by input context size:
- **Small** (0-200 tokens): Simple Q&A
- **Medium** (200-500 tokens): Short conversations
- **Large** (500-2K tokens): Technical/code help
- **XLarge** (2K-8K tokens): Document analysis
- **XXLarge** (8K+ tokens): Full context utilization

### Output Options

#### Console Output
- Timestamped summary with detailed metrics
- Periodic windowed stats during benchmark runs
- Detailed percentile breakdowns

#### JSON Export
- Complete structured results with metadata
- Test configuration included for reproducibility
- Timestamps and version tracking
- Error breakdown by type
- All metrics and percentiles
- Context-aware TTFT and ITL metrics
- Machine-readable format for analysis and automation

### Observability

- **Efficient Logging**: Via `ringlog` with asynchronous ring buffer
- **Configurable Log Levels**: error, warn, info, debug, trace (configure via `[log]` section)
- **Per-Module Filters**: Fine-grained log control (e.g., `filter = ["hyper=warn"]`)
- **Log File Support**: Automatic rotation at 10MB with backup file preservation
- **Periodic Stats**: Windowed metrics output during benchmark runs
- **Quiet Mode**: Suppress periodic stats for CI/CD or JSON piping
- **Admin Metrics Server**: Optional HTTP endpoint exposing Prometheus/JSON metrics during runs

### Metrics Capture

- **Periodic Snapshots**: Capture all metrics at regular intervals (e.g., secondly resolution)
- **Parquet Output**: Columnar format optimized for analytics and data processing
- **Configurable Intervals**: From milliseconds to minutes
- **Batch Control**: Configurable batch sizes for efficient parquet file generation

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/iopsystems/llm-perf.git
cd llm-perf

# Build in release mode (recommended for benchmarking)
cargo build --release

# Binary will be at ./target/release/llm-perf
```

### Prerequisites

- Rust 1.85+ (uses 2024 edition features)
- An OpenAI-compatible LLM server running locally or remotely

## Usage

### Quick Start

```bash
# Copy the comprehensive example configuration
cp examples/config.example.toml my-config.toml

# Edit configuration for your needs
vim my-config.toml

# Run benchmark
./llm-perf my-config.toml
```

### Subcommands

```bash
# Run a benchmark (default subcommand — config path alone works too)
llm-perf bench my-config.toml

# Collect logprobs sequentially
llm-perf logprobs my-config.toml

# Compare two logprob captures
llm-perf kl-divergence baseline.jsonl candidate.jsonl
llm-perf kl-divergence baseline.jsonl candidate.jsonl --format json --output report.json
```

### Configuration

**Use `examples/config.example.toml` as your starting point.** It contains all available options with detailed comments explaining each field.

Example minimal configuration:

```toml
[endpoint]
base_url = "http://localhost:8080/v1"  # Your LLM server URL
model = "llama-3.1-8b"                  # Model name (omit to auto-detect)
timeout = 60                             # Request timeout in seconds
# api_key = "optional-api-key"          # If authentication required
# max_retries = 3                       # Retry failed requests (0 = no retries)

[load]
concurrent_requests = 10                 # Parallel workers (also max in-flight for QPS mode)
total_requests = 100                     # Total requests to send
# qps = 5.0                              # Set to enable fixed QPS mode
# duration_seconds = 60                  # OR: Run for N seconds
# warmup_requests = 10                   # Warmup requests (excluded from metrics)
# warmup_duration = 5                    # OR: Warmup duration in seconds

[input]
file = "prompts.jsonl"                  # Input prompts file
shuffle = false                          # Randomize prompt order
# sample_size = 50                      # Use only first N prompts

[output]
format = "console"                       # "console" or "json"
# file = "results.json"                 # For json: write to file (omit for stdout)
# quiet = false                         # Suppress periodic stats

[runtime]
worker_threads = 8                       # Tokio worker threads (default: CPU count)

[log]
level = "info"

# Optional: Enable metrics capture for detailed time-series analysis
# [metrics]
# output = "metrics.parquet"
# interval = "1s"
# batch_size = 10000
```

### Input Format

Create a JSONL file with prompts:

```jsonl
{"prompt": "What is the capital of France?"}
{"prompt": "Explain quantum computing", "max_tokens": 500}
{"prompt": "Write a Python hello world program"}
```

### Configuration Reference

**Primary Reference**: `examples/config.example.toml` contains:
- All available configuration options
- Detailed comments for each field
- Common configuration patterns
- Examples for different use cases

### Available Scenarios

The `examples/scenarios/` directory contains example configurations for distinct workload patterns. **Always refer to `examples/config.example.toml` for the most current options.**

Each scenario demonstrates a distinct load pattern:

- `basic.toml` - Simple concurrent load test with fixed request count
- `duration-test.toml` - Time-based concurrent load (soak testing)
- `fixed-qps.toml` - Fixed QPS with uniform arrival distribution (capacity testing)
- `poisson-arrival.toml` - Fixed QPS with Poisson arrival distribution (realistic traffic)
- `qps-duration.toml` - Fixed QPS with time-based limit

### Available Datasets

The `examples/prompts/` directory contains prompt datasets:

- **openorca-10000.jsonl** - 10,000 unique instruction-following prompts from the OpenOrca dataset (9.3MB)
  - Realistic token distribution: 57% short (< 200 tokens), 27% medium (200-500), 12% long (500-1000), 4% very long (> 1000)
  - Per-request cache-busting automatically applied by the benchmark tool

**Quick testing:** Use the `sample_size` parameter to limit prompts:
```toml
[input]
file = "examples/prompts/openorca-10000.jsonl"
sample_size = 100    # Use only first 100 prompts for quick tests
```

**Generating custom datasets:**

```bash
python3 scripts/prepare_dataset.py
python3 scripts/prepare_dataset.py --samples 50000
```

### Common Configuration Patterns

**1. Quick Concurrent Test**
```toml
[load]
concurrent_requests = 10
total_requests = 100
```

**2. Fixed QPS (Capacity Testing)**
```toml
[load]
qps = 10.0
arrival_distribution = "uniform"
total_requests = 1000
```

**3. Realistic Traffic Pattern**
```toml
[load]
qps = 10.0
arrival_distribution = "poisson"
duration_seconds = 60
```

**4. Soak Test**
```toml
[load]
concurrent_requests = 10
duration_seconds = 300
warmup_duration = 10
```

## Output Formats

### Console Output (Default)

Timestamped summary with detailed metrics and percentile breakdowns:

```
2026-03-10T12:00:00.000-07:00
2026-03-10T12:00:00.000-07:00 -----
2026-03-10T12:00:00.000-07:00 Benchmark Complete
2026-03-10T12:00:00.000-07:00 Duration: 45.2s
2026-03-10T12:00:00.000-07:00 Requests: Sent: 100 Retries: 0
2026-03-10T12:00:00.000-07:00 Responses: Received: 100 Ok: 98 Err: 2 Success: 98.00%
2026-03-10T12:00:00.000-07:00 Tokens: Input: 20492 Output: 82456 Total: 102948
2026-03-10T12:00:00.000-07:00 Throughput: Requests/s: 2.21 Input tokens/s: 453.2 Output tokens/s: 1823.5
2026-03-10T12:00:00.000-07:00 TTFT (ms): mean: 145.2 p50: 125 p90: 342 p95: 422 p99: 623
2026-03-10T12:00:00.000-07:00 TPOT (ms): mean: 28.3 p50: 24 p90: 45 p95: 52 p99: 73
2026-03-10T12:00:00.000-07:00 ITL (ms): mean: 25.1 p50: 23 p90: 45 p95: 52 p99: 79
2026-03-10T12:00:00.000-07:00 Request Latency (ms): mean: 1823.4 p50: 1500 p90: 3421 p95: 4124 p99: 5234
```

### JSON Output

Structured output for automation and analysis:

```toml
[output]
format = "json"
file = "results.json"  # Write to file (also shows brief console summary)
```

Or pipe JSON to stdout for processing:

```bash
llm-perf config.toml | jq '.throughput.requests_per_second'
llm-perf config.toml | jq '.errors'
```

**JSON Report Structure:**
- **Metadata**: Timestamp, duration, version
- **Configuration**: Complete test parameters for reproducibility
- **Summary**: Request counts, success rate, retries
- **Throughput**: Requests/sec, tokens/sec metrics
- **Latency**: TTFT, TPOT, ITL, and end-to-end percentiles
- **Errors**: Breakdown by error type
- **Context Metrics**: TTFT and ITL percentiles by input token count

## Additional Tools

### MMLU-Pro Evaluation

The `mmlu-pro` binary runs MMLU-Pro accuracy evaluations against LLM servers:

```bash
./target/release/mmlu-pro config.toml
```

This is a separate binary for measuring model quality (accuracy) rather than performance (latency/throughput). It downloads the MMLU-Pro dataset from HuggingFace, runs chain-of-thought evaluation, and produces per-category accuracy scores with resume support.

## Development

### Building and Testing

```bash
cargo check      # Check code without building
cargo test       # Run tests
cargo fmt        # Format code
cargo clippy     # Run linter
cargo build --release  # Build optimized for benchmarking
```

### Project Structure

```
llm-perf/
├── src/
│   ├── main.rs           # Entry point, CLI dispatch, runtime setup
│   ├── lib.rs            # Library root and public exports
│   ├── cli.rs            # Subcommand definitions (bench, logprobs, kl-divergence)
│   ├── benchmark.rs      # Core benchmarking engine (concurrent and QPS modes)
│   ├── client.rs         # OpenAI API client with SSE streaming and retries
│   ├── config.rs         # Configuration structures
│   ├── metrics.rs        # Metric declarations (metriken counters/histograms)
│   ├── report.rs         # Report generation (console and JSON)
│   ├── stats.rs          # Periodic windowed stats during runs
│   ├── snapshot.rs       # Metrics snapshot to parquet pipeline
│   ├── admin.rs          # HTTP admin server (Prometheus/JSON metrics)
│   ├── distribution.rs   # Request arrival distributions (uniform, poisson)
│   ├── tokenizer.rs      # Token counting via tiktoken
│   ├── logprobs.rs       # Log probability collection and JSONL writer
│   ├── kl_divergence.rs  # KL divergence computation
│   └── mmlu_pro/         # MMLU-Pro evaluation binary
├── examples/
│   ├── config.example.toml  # Comprehensive configuration reference
│   ├── scenarios/           # Workload pattern examples
│   └── prompts/             # Prompt datasets
├── Cargo.toml           # Rust dependencies
└── CHANGELOG.md         # Version history
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
