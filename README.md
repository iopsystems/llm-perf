# llm-perf

A high-performance benchmarking tool for OpenAI-compatible LLM inference servers. Designed to measure detailed performance characteristics of local LLM servers like llama-server, vLLM, TGI, and other OpenAI API-compatible endpoints.

## Features

### Core Capabilities

- **OpenAI API Compatibility**: Works with any server implementing the `/v1/chat/completions` endpoint
- **Streaming Support**: Measures Time-To-First-Token (TTFT) via SSE streaming
- **Async/Concurrent Testing**: Configurable concurrent request handling with Tokio runtime
- **Token Counting**: Built-in tokenizer support for accurate token metrics
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

### Performance Metrics

#### Error Analysis
- **Categorized Error Tracking**: Detailed breakdown of failures
  - Connection errors (network issues)
  - HTTP 4xx errors (client errors with status codes)
  - HTTP 5xx errors (server errors with status codes)  
  - Timeout errors (request exceeded timeout)
  - Parse errors (invalid response format)
  - Detailed error messages for debugging

#### Latency Measurements
- **Time to First Token (TTFT)**: Critical for streaming response UX
  - P50, P90, P95, P99 percentiles
  - Context-aware buckets (small, medium, large, xlarge, xxlarge)
- **Inter-Token Latency (ITL)**: Time between consecutive tokens
  - P50, P90, P95, P99 percentiles
  - Measures streaming smoothness
  - Context-aware analysis (automatically categorized by input size)
- **End-to-End Request Latency**: Total request completion time
  - P50, P90, P95, P99 percentiles

#### Throughput Metrics
- Requests per second
- Input tokens per second
- Output tokens per second
- Total token counts

#### Context-Aware Analysis
Automatically categorizes TTFT by input context size:
- **Small** (0-200 tokens): Simple Q&A
- **Medium** (200-500 tokens): Short conversations  
- **Large** (500-2K tokens): Technical/code help
- **XLarge** (2K-8K tokens): Document analysis
- **XXLarge** (8K+ tokens): Full context utilization

### Input Management

- **JSONL Input Format**: Simple prompt format with optional max_tokens
- **Prompt Cycling**: Automatically cycles through prompts for longer tests
- **Sample Size Control**: Limit prompts for quick tests
- **Shuffle Support**: Randomize prompt order for more realistic patterns

### Output Options

#### Console Output
- Clean, formatted results with progress bar
- Summary statistics
- Detailed percentile breakdowns
- Context-aware latency analysis

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
- **Configurable Log Levels**: 
  - Supports: error, warn, info, debug, trace
  - Default: info level
  - Configure via `[log]` section in config
- **Log File Support**: 
  - Automatic rotation at 10MB
  - Backup file preservation
- **Progress Indicators**: Real-time progress bars
- **Quiet Mode**: Minimal output for CI/CD

### Metrics Capture

- **Periodic Snapshots**: Capture all metrics at regular intervals (e.g., secondly resolution)
- **Parquet Output**: Columnar format optimized for analytics and data processing
- **Configurable Intervals**: From milliseconds to minutes
- **Batch Control**: Configurable batch sizes for efficient parquet file generation

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-perf.git
cd llm-perf

# Build in release mode (recommended for benchmarking)
cargo build --release

# Binary will be at ./target/release/llm-perf
```

### Prerequisites

- Rust 1.75+ (uses 2024 edition features)
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

### Configuration

**Use `config.example.toml` as your starting point.** It contains all available options with detailed comments explaining each field.

Example minimal configuration:

```toml
[endpoint]
base_url = "http://localhost:8080/v1"  # Your LLM server URL
model = "llama-3.1-8b"                  # Model name
timeout = 60                             # Request timeout in seconds
# api_key = "optional-api-key"          # If authentication required
# max_retries = 3                       # Retry failed requests (0 = no retries)
# retry_initial_delay_ms = 100          # Initial retry delay
# retry_max_delay_ms = 10000            # Max retry delay (10s)

[load]
load_pattern = "concurrent"              # "concurrent" or "fixed_qps"
concurrent_requests = 10                 # For concurrent mode: parallel workers
# qps = 5.0                              # For fixed_qps mode: requests per second
total_requests = 100                     # Total requests to send
# duration_seconds = 60                  # OR: Run for N seconds
# rate_limit = 50                        # Max requests per second (optional)
# warmup_requests = 10                   # Number of warmup requests (excluded from metrics)
# warmup_duration = 5                    # OR: Warmup duration in seconds

[input]
file = "prompts.jsonl"                  # Input prompts file
shuffle = false                          # Randomize prompt order
sample_size = 50                        # Use only first N prompts

[output]
format = "console"                       # "console" or "json"
file = "results.json"                   # Optional for json: write to file (omit for stdout)
quiet = false                           # Suppress periodic stats
trace_log = "debug.log"                 # Optional: Debug trace file

[runtime]
worker_threads = 8                       # Tokio worker threads (default: CPU count)

[log]
# Controls the log level: "error", "warn", "info", "debug", "trace"
level = "info"

# Optional: Enable metrics capture for detailed time-series analysis
# [metrics]
# output = "metrics.parquet"      # Output file path (parquet format)
# interval = "1s"                  # Snapshot interval
# batch_size = 10000               # Batch size for parquet conversion
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

The `examples/scenarios/` directory contains example configurations for distinct workload patterns. These are provided for reference but may be outdated. **Always refer to `examples/config.example.toml` for the most current options.**

**Note on Features vs Scenarios:**
- **Metrics capture** (`[metrics]` section) - Optional feature available for any scenario
- **Model auto-detection** (omit `model` field) - Optional feature available for any scenario
- All scenario files include commented examples of these features

Each scenario demonstrates a distinct load pattern:

- `basic.toml` - Simple concurrent load test with fixed request count
- `duration-test.toml` - Time-based concurrent load (soak testing)
- `fixed-qps.toml` - Fixed QPS with uniform arrival distribution (capacity testing)
- `poisson-arrival.toml` - Fixed QPS with Poisson arrival distribution (realistic traffic)
- `qps-duration.toml` - Fixed QPS with time-based limit

### Available Datasets

The `examples/prompts/` directory contains various prompt datasets. Use the `[input]` section to select a dataset for any scenario:

```toml
[input]
file = "examples/prompts/openorca-10000.jsonl"
shuffle = true
```

**Available datasets:**

- **openorca-10000.jsonl** - 10,000 unique instruction-following prompts from the OpenOrca dataset (9.3MB)
  - Real-world instruction data from FLAN and T0 collections
  - Ensures diverse prompts even for long-duration, high-throughput tests
  - Realistic token distribution: 57% short (< 200 tokens), 27% medium (200-500), 12% long (500-1000), 4% very long (> 1000)
  - Average max_tokens: 266
  - Diverse mix of Q&A, reasoning, code generation, and explanations
  - Per-request cache-busting automatically applied by the benchmark tool

**Quick testing:** Use the `sample_size` parameter to limit prompts:
```toml
[input]
file = "examples/prompts/openorca-10000.jsonl"
sample_size = 100    # Use only first 100 prompts for quick tests
```

**Generating custom datasets:**

Use `scripts/prepare_dataset.py` to create datasets with different sizes:

```bash
# Generate default 10k dataset
python3 scripts/prepare_dataset.py

# Or specify custom size
python3 scripts/prepare_dataset.py --samples 50000

# See scripts/README.md for details
```

Any dataset can be used with any workload scenario by changing the `file` path.

**Using external datasets:**

For production benchmarking, consider using:
- **ShareGPT**: Real ChatGPT conversations (available on HuggingFace)
- **OpenOrca**: Instruction-following dataset
- **Alpaca**: 52K instruction-response pairs

These can be converted to our JSONL format with appropriate `max_tokens` limits.

### Testing Methodologies

To conduct systematic performance analysis, you can modify any scenario to vary parameters:

**Concurrency scaling:** Run the same scenario multiple times with different `concurrent_requests` values (e.g., 1, 5, 10, 20, 50) to understand how concurrency affects throughput and latency.

**Context length analysis:** Filter or sort the OpenOrca dataset by input token ranges and run sequentially (`concurrent_requests = 1`, `shuffle = false`) to isolate context length effects. The tool's context-aware metrics automatically categorize performance by input size.

**Stress testing:** Use higher `concurrent_requests` values to push server limits and identify breaking points.

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

Human-readable summary with detailed metrics, percentile breakdowns, and context-aware analysis. Use `format = "console"` (default) in config.

### JSON Output

Structured output for automation and analysis:

```toml
[output]
format = "json"
file = "results.json"  # Write to file (also shows brief console summary)
```

Or pipe JSON to stdout for processing:

```toml
[output]
format = "json"
# No file specified - outputs to stdout for piping
```

```bash
# Example: Extract request rate
llm-perf config.toml | jq '.throughput.requests_per_second'

# Example: Filter errors
llm-perf config.toml | jq '.errors'
```

**JSON Report Structure:**
- **Metadata**: Timestamp, duration, version
- **Configuration**: Complete test parameters for reproducibility
- **Summary**: Request counts and success rates
- **Throughput**: Requests/sec, tokens/sec metrics
- **Latency**: TTFT, ITL, and end-to-end percentiles
- **Errors**: Breakdown by error type
- **Context Metrics**: Performance by input token count

## Example Console Output

```
LLM Benchmark Tool
   Config: config.toml
   Target: http://localhost:8080/v1
   Requests: 100

[========================================] 100/100 (00:45)

╔══════════════════════════════════════════╗
║          BENCHMARK RESULTS               ║
╚══════════════════════════════════════════╝

Duration: 45.23s

Summary
├─ Total Requests: 100
├─ Successful: 98 (98.0%)
└─ Failed: 2

Throughput
├─ Requests/sec: 2.21
├─ Input tokens/sec: 453.2
├─ Output tokens/sec: 1823.5
└─ Total tokens: 20492 in, 82456 out

Latency (ms)
├─ TTFT p50: 125.3ms
├─ TTFT p90: 342.1ms
├─ TTFT p95: 421.5ms
├─ TTFT p99: 623.2ms

├─ ITL p50: 23.4ms
├─ ITL p90: 45.2ms
├─ ITL p95: 52.3ms
├─ ITL p99: 78.9ms

├─ Total p50: 1823.4ms
├─ Total p90: 3421.2ms
├─ Total p95: 4123.5ms
└─ Total p99: 5234.1ms

TTFT by Context Length
├─ Small (0-200) tokens:
   p50: 89.2ms
   p90: 123.4ms
   p95: 145.2ms
   p99: 189.3ms
├─ Medium (200-500) tokens:
   p50: 125.3ms
   p90: 234.5ms
   p95: 298.7ms
   p99: 412.3ms
└─ Large (500-2K) tokens:
   p50: 234.5ms
   p90: 456.7ms
   p95: 523.4ms
   p99: 712.3ms

ITL by Context Length
├─ Small (0-200) tokens:
   p50: 18.3ms
   p90: 28.4ms
   p95: 32.1ms
   p99: 45.2ms
├─ Medium (200-500) tokens:
   p50: 22.4ms
   p90: 38.5ms
   p95: 44.3ms
   p99: 62.1ms
└─ Large (500-2K) tokens:
   p50: 31.2ms
   p90: 52.3ms
   p95: 61.4ms
   p99: 89.3ms
```

## Development

### Building and Testing

```bash
# Check code without building
cargo check

# Run tests
cargo test

# Format code
cargo fmt

# Run linter
cargo clippy

# Build for development
cargo build

# Build optimized for benchmarking
cargo build --release
```

### Project Structure

```
llm-perf/
├── src/
│   ├── main.rs           # Entry point and runtime setup
│   ├── benchmark.rs      # Core benchmarking logic
│   ├── client.rs         # OpenAI API client implementation
│   ├── config.rs         # Configuration structures
│   ├── metrics.rs        # Metrics collection
│   ├── report.rs         # Report generation
│   ├── tokenizer.rs      # Token counting
│   └── cli.rs            # Command-line interface
├── examples/
│   ├── config.example.toml  # Comprehensive configuration reference
│   ├── scenarios/           # Workload pattern examples
│   └── prompts/             # Prompt datasets
├── Cargo.toml           # Rust dependencies
└── CLAUDE.md            # AI assistant instructions
```

## Future Enhancements

llm-perf is production-ready with all core benchmarking features implemented. Potential future enhancements:

- **Worker Ramp-up Control**: Gradual worker spawning to avoid thundering herd on very large scale tests
- **Custom Headers**: Additional HTTP header support if authentication schemes require it
- **WebSocket Streaming**: Alternative to SSE if servers support it

Post-processing capabilities (comparison mode, cost estimation, result analysis) are intentionally handled by external tools and data pipelines using the JSON/Parquet output formats.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

### Development Guidelines

1. Follow Rust best practices and idioms
2. Maintain comprehensive error handling
3. Add tests for new features
4. Update documentation for API changes
5. Run `cargo fmt` and `cargo clippy` before submitting

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

## Acknowledgments

- Built with Rust and Tokio for high-performance async I/O
- Uses the OpenAI API specification for compatibility
- Metrics powered by metriken for efficient collection