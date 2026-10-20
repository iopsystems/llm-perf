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
- **`convert-trace`** - Convert the coding-agent session dataset from [Azure/AzurePublicDataset](https://github.com/Azure/AzurePublicDataset/blob/master/GitHubCopilotCodingAgentDataset2026.md) into a per-session replay trace
- **`kv-residency`** - Match a trace replay's per-call log against the vLLM KV cache events recorded with it (see [KV cache events](#kv-cache-events-vllm))

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
- **Admin Metrics Server**: Optional HTTP server exposing metrics during runs as Prometheus text, JSON, a msgpack snapshot, and a replication stream that `rezolus record` writes into a `.dendro` (see [Admin server](#admin-server))

### Metrics Capture

- **Periodic Snapshots**: Capture all metrics at regular intervals (e.g., secondly resolution)
- **One archive per run**: `[metrics] output` ending in `.dendro` records llm-perf and the server under test into one dendro archive, in process (see [Recording a run](#recording-a-run))
- **Parquet Output**: any other `output` is a parquet of llm-perf's metrics, written at exit
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

# Convert one day of the coding-agent dataset, keeping sessions that fit 128K
# context and start on that day
gh release download ghcp-coding-agent-2026 -R Azure/AzurePublicDataset -p 'date.2026-06-03.tar.gz'
llm-perf convert-trace date.2026-06-03.tar.gz --max-context 131072 \
    --from 2026-06-03T00:00:00Z --to 2026-06-04T00:00:00Z -o trace.jsonl
```

`convert-trace` writes one JSON object per session, sorted by start. A session
carries `session_id`, `start` (RFC 3339), `start_ms` (milliseconds from the
earliest `start` in the output), `models` (distinct labels) and `calls`. Each
call carries:

- `turn`: index of the user turn in the source record
- `initiator`: the source's `initiator_type` (`user` or `agent`), when present
- `model`: the call's model label, so a replayer can see where a session
  switches model
- `prompt`, `completion`, `cached`: token counts from the source
- `gap_ms`: idle time from the end of the previous call to the start of this
  one
- `reuse`: how many leading tokens of this prompt repeat the previous call's
  prompt followed by its completion
- `reuse_inferred`: present and `true` when `reuse` is an estimate
- `duration_ms`: the source's latency for the call

`reuse` is derived from `cached` and capped at the previous prompt plus
completion; the stats line on stderr reports how many calls were capped. The
source's `timestamp` is a call's end time; start is computed as
`timestamp - duration_ms`. After an idle gap of at least `--cache-ttl-secs`
(default 300), a call whose `cached` is below `--evicted-reuse-ratio` (default
0.98) times the smaller of the two prompts gets that estimate as its `reuse`
and is marked `reuse_inferred`. A ratio of 0 disables the estimate; a TTL of 0
applies it to every call after the first.

Calls whose `tokens.prompt` is missing or 0 are dropped, and their duration
becomes part of the next call's `gap_ms`. `--model` keeps a session only if
every kept call's model label is listed; a call with no label fails the
filter. `--max-context` drops a session if any call's prompt plus completion
exceeds the limit. `--from` is inclusive and `--to` exclusive, both compared
with the start of the session's first kept call. A `session_id` that appears
more than once across the inputs is an error that names both files; this
catches passing a tarball together with its extracted directory.

Without `--from`/`--to`, one partition's session starts can span several
months (59 to 127 days for 2026-06-02 to 2026-06-04). All kept sessions are
held in memory before sorting. Peak resident memory was about 0.72 GB for one
day of input and 1.4 GB for three days.

### Trace replay

A `[replay]` section in a bench config replays the sessions in a
`convert-trace` file against llama-server or vLLM, instead of drawing requests
from `[input]`. `server` selects the server type: `"llama-server"` (the
default) or `"vllm"`. See `examples/scenarios/trace-replay.toml`.

```bash
llm-perf bench examples/scenarios/trace-replay.toml
```

Each session starts at its traced start time divided by `speedup`, and each
call is sent at its scaled time on the source timeline, never before the
previous call of the session has finished. Each prompt repeats the previous
call's messages and the server's actual reply, cut to the traced `reuse` in
tokens, and adds filler words to reach the traced `prompt` size. Each call asks
for the traced `completion` length with `ignore_eos`. Sizes and cut points are
measured with the server's own chat template and tokenizer:

| | llama-server | vLLM |
|---|---|---|
| Render chat messages | `/apply-template`, then `/tokenize` | `/tokenize` with `messages` |
| Tokenize, detokenize | `/tokenize`, `/detokenize` | `/tokenize` with `prompt`, `/detokenize` |
| Per-request context | `/props` `n_ctx` | `/v1/models` `max_model_len` |
| Build | `/props` `build_info` | `/version` |

vLLM must run with `--enable-prefix-caching` and
`--enable-prompt-tokens-details`; without the second, responses do not report
cached tokens. Replay does not check that prefix caching is on; without it
every call reports 0 cached tokens.

Reuse is measured from what the server reports for each call: `prompt_tokens`
and `cached_tokens`. Replay does not predict how many tokens should be cached.
To see how much reuse a run lost to load, compare its cached/prompt ratio
against a low-load run of the same sessions, on the calls both runs sent (join
the logs on `session_id` and `call`). An overloaded run sends fewer of each
session's later, longer calls when `duration_seconds` ends the run or failed
calls end their sessions. Its summary then covers different calls from the
low-load run's. `calls_measured` is the number of calls the cached/prompt
percentiles cover.

The per-call log (`replay.log`) records, among other fields:

- `prompt_tokens`, `cached_tokens`: as the server reported them
- `rendered_tokens`: the prompt length replay rendered with the server's
  tokenize endpoint, or null when building the prompt failed; it equals
  `prompt_tokens` unless the server built a different prompt from the one
  replay rendered
- `reuse`: the traced reuse the call was built with
- `lag_ms`: how far the call was sent behind its scaled source time
- `overshoot`: the prompt is longer than traced, because the call adds fewer
  tokens than a new message needs
- `shortfall`: requested reuse beyond the previous rendered prompt plus the
  tokenized reply; a token or two when re-tokenizing the reply gives fewer
  tokens than were generated, more when the server stopped early

The summary counts calls whose `prompt_tokens` differ from `rendered_tokens`
(`calls_render_mismatch`) and records the largest difference
(`render_mismatch_max_tokens`); both include warmup. Replay logs the first
mismatched call as a warning. A mismatch means the server built a different
prompt from the one replay rendered. Replay cuts in its own rendered tokens and
sends text. The list below describes a server prompt longer than replay's; a
shorter one changes the lengths the other way.

- a constant difference at the start (a start token) makes every prompt, and
  the prefix it shares with the previous call, that many tokens longer in the
  server's tokens; the cut falls at the same text;
- a constant difference at the end (a generation-prompt suffix) makes the
  prompt that many tokens longer. When the cut falls inside the previous
  reply, the server can share fewer tokens with the previous call than the
  traced reuse, because the suffix sat between that prompt and the reply;
- a difference in how earlier messages render can change the prefix the
  server shares with the previous call by more than the length difference
  shows.

The per-call log shows whether the difference is the same on every call; it
does not show whether it is at the start or the end.

A mismatch does not fail the run. Each time another 100 calls have reported
`prompt_tokens`, warmup included, the run fails if the median of
`|prompt_tokens - target_prompt| / target_prompt` over all of them exceeds
0.01.

The summary also reports prompt, cached and prefill-computed token totals, and
percentiles of `1000 * cached_tokens / prompt_tokens` over successful
non-warmup calls whose traced `reuse` is above 0.

Replay requires `endpoint.max_retries = 0` and no `endpoint.max_tokens`, and
accepts only `duration_seconds` and `warmup_duration` from `[load]`. Every
selected call must fit the server's per-request context. The run fails if a
response reports no `usage.prompt_tokens_details`, since reuse cannot be
measured without the server's cached token count. The summary reports the
server type, build and per-request context.
Filler is fixed by `seed`, the session id and the call index, so a second run
against a server that still holds the first run's prompts gets cache hits on
first calls; restart the server or change `seed` between runs.

#### KV cache events (vLLM)

With `[replay.kv_events]`, replay subscribes to the KV cache events vLLM
publishes and writes them to a log, so cache misses can be traced to the
blocks the server removed. Start vLLM with

```bash
--kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557"}'
```

vLLM binds the endpoint when it contains `*` or `::`, or is `ipc://` or
`inproc://`, and otherwise connects to it. Replay connects, so give vLLM a
bind endpoint such as `tcp://*:5557`. With a connect endpoint nothing binds,
and the run fails when replay's connect times out (30 s). Only one vLLM
publisher is read: with data parallelism, each rank publishes on its own
port. Start vLLM fresh, or reset its prefix cache, before a captured run;
blocks stored before the capture have no fingerprint. Then set

```toml
[replay.kv_events]
endpoint = "tcp://127.0.0.1:5557"
block_size = 528          # the server's prefix-cache block size in tokens
log = "kv-events.jsonl"
```

`block_size` must be the block size vLLM reports in its stored events. The
summary counts stored events with another size (`block_size_mismatch`),
which also counts events for blocks that are not full yet. Each line of the
event log is one `stored`, `removed` or `cleared` event with the time replay
received it (`t_ms`, on the per-call log's clock), the batch sequence number,
the KV cache group, the block hashes, the group's spec kind and the storage
medium where vLLM sends them, and, for stored blocks, a fingerprint of the
prefix each block ends (`fps`). A fingerprint is `null` where the event alone
does not determine it; `kv-residency` fills these from other events with the
same hash. Each line of the per-call log gains `block_fps`, the fingerprint of
each full block of the rendered prompt. Events are published after the
engine step in which the blocks changed, so `t_ms` can trail the change by up
to one step, and `kv-residency` can then count a removal made just before a
call's send as made while the call was queued.

The summary reports batches received, the first sequence number received
(`first_seq`; vLLM numbers batches from 0 when it starts), sequence numbers
skipped after it (`batches_missed`), batches whose sequence number was not
above the previous one's (`seq_resets`), batches that could not be decoded,
and receive errors. A sequence reset means the server restarted and holds no
blocks, so the log gets a `cleared` event before that batch. vLLM's PUB socket drops batches for a subscriber whose queue
has reached its high-water mark (`hwm`, 100000 by default), and a batch whose
send fails leaves a gap. The analysis assumes vLLM's default
`kv_cache_report_mode`, `incremental`, in which a stored event means a block
was added to the cache.

`llm-perf kv-residency replay.jsonl kv-events.jsonl --block-size 528` replays
the event log against the calls; `--block-size` is the `block_size` the run
used. For each successful non-warmup call it finds how many leading prompt
blocks the server held when the call was sent, and how many of those it
could have reused. Only events for GPU blocks count. A Mamba group (spec kind
`mamba`, or, in events without a spec kind, one whose stored events skip
blocks, as in "align" prefix caching) counts a prefix of `k` blocks as
reusable when it holds the state at the end of the prefix's last block. A
prompt of exactly `k` blocks can reuse at most `k - 1`, since vLLM recomputes
the last token. Event logs with a sliding-window or other group kind are
refused. Calls without a first-token time are not analysed.
vLLM stores a call's prompt blocks when it schedules the call, so at the
call's first token only blocks held without a break since the send count.

It checks that every call's blocks appear in a stored event, counts stored
blocks with no fingerprint and calls whose reported cached blocks exceed what
was reusable at send, and classifies calls with traced reuse and no cached
tokens by whether and when their prefix was stored and removed: never
stored, attention blocks held but the state at the end of no held prefix
stored by every Mamba group, removed before
the call was sent (attention blocks, or only the Mamba state), removed
between the send and the first token, or held without a break from the send
to the first token. `--calls-out` writes one line per call.

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

### Admin server

`[admin] enabled = true` starts an HTTP server on `listen` (default
`127.0.0.1:9090`) for the length of a `bench` run:

| Path | Body |
|------|------|
| `/metrics` | Prometheus text; histograms as p50/p90/p95/p99/p99.9 gauges |
| `/metrics.json`, `/vars.json` | JSON, histograms as `<name>/p<permille>` keys |
| `/vars` | the same, one `name: value` line each |
| `/metrics/binary` | one msgpack snapshot of every metric (`application/msgpack`) |
| `/metrics/stream` | dendro replication frames (`application/vnd.metriken.replication.v1+dendro`) |

`/metrics/stream` takes `interval` (default `1s`) and `layout` (`wide`, the
default, or `long`). It sends a handshake naming `source=llm-perf`, with the
version and the [KPI template](#kpi-template) in its metadata, and then one
frame per interval, each carrying every metric recorded so far, histograms with their buckets,
until the client disconnects. A zero or malformed interval and an unknown
layout are refused with 400 and the reason; an interval below 10 ms is served
at 10 ms, which `x-metriken-frame-interval` names. The
response headers `x-metriken-frame-interval`, `x-metriken-update-floor` and
`x-metriken-layout` state what is served.

`rezolus record` given the admin server's base URL probes `/metrics/binary`,
recognizes llm-perf as a metriken source, and records it into a `.dendro`.
rezolus at or after 6.0.0-alpha.45 records the stream; until 6.0.0 is
released, that is a build of rezolus main at or after ab24b072. rezolus 5.23
to 5.26 scrape `/metrics/binary` into the `.dendro` instead (measured with
5.26.0). 6.0 alphas before alpha.45 refuse the stream's content type, and
rezolus before 5.23.0 writes parquet to a path ending in `.dendro`.

```bash
rezolus record --endpoint http://127.0.0.1:9090,source=llm-perf -o run.dendro
```


At the default `[log] level = "info"`, llm-perf hides `warp::server` events,
so the `server connection error: hyper::Error(IncompleteMessage)` that warp
logs when `rezolus record` closes `/metrics/stream` at the end of a run does
not appear. The same filter hides the admin server's other connection errors
and its accept errors, such as `accept error` when llm-perf runs out of file
descriptors. `level = "debug"` shows them, as does
`filter = ["warp::server=error"]`. A failure to bind the admin address still
shows: warp panics with `failed to bind to address`.

### Recording a run

With `[metrics] output` ending in `.dendro`, llm-perf records the run into
that archive itself, through metriken-recorder: its own metrics, and the
server under test when `server_metrics_url` is set. Both are on one timeline,
each as its own recording, and the archive is readable while the run goes. No
rezolus binary is needed.

```toml
[metrics]
output = "run.dendro"
interval = "1s"
server_metrics_url = "http://localhost:8080/metrics"
```

llm-perf's histograms are recorded with their buckets. `server_metrics_url` is
either a Prometheus `/metrics` URL, scraped each interval, or the base URL of
a producer that serves `/metrics/stream` (a rezolus agent, cachecannon,
another llm-perf), which is streamed; a base URL without a stream is scraped
at `/metrics`. A server that is not reachable or answers with an error at the
start is detected again each interval. If it is never recorded, the console
summary says so in a warning with the last reason, and the archive holds
llm-perf alone.

llm-perf's recording is labeled `source=llm-perf, role=loadgen`. The server's
is labeled `role=service`, and its `source` is `server_source` when set,
otherwise the name its stream's handshake gives (`rezolus`, `llm-perf`, ...)
or, for a scraped server, one made from its host and port.

With a parquet `output`, llm-perf still writes the parquet at exit, and with
`server_metrics_url` set it also records the archive at
`server_metrics_output`, by default `<output-stem>.server.dendro`. That
archive holds llm-perf too. With `server_metrics_url` set,
`server_metrics_output` must end in `.dendro` (exactly: not `.DENDRO`), and
must not be set when `output` is the archive. The archive must not exist yet:
llm-perf refuses to start rather than overwrite one. `rezolus_bin` is ignored
and logs a warning.

A run that completes ends the recording with one more pass of llm-perf and
a scraped server, and waits up to one interval plus the scrape timeout for a
streamed server's last frame. A run that fails ends it without the wait.

SIGINT (Ctrl-C) or SIGTERM during a recorded run finalizes the archive with
a last pass of llm-perf and a scraped server (no wait for a streamed one),
prints the outcome to stderr, and then llm-perf dies by that signal, so a
shell loop stops and the exit status is 130 or 143; no report or parquet is
written. A second signal while it finalizes kills it at once and leaves the
archive unfinalized. A signal after the run, while llm-perf finalizes the
archive (its last pass, or the wait for a streamed server's last frame,
which the signal ends), lets it finish; the report is written, and then
llm-perf dies by the signal. A signal after that, while the report is
written, kills llm-perf at once with no report. A signal while the
server under test is detected, before the archive exists, kills llm-perf at
once. A signal the parent set to be ignored (a background job in a script)
stays ignored.

To pick llm-perf's own recording out of the archive, select `role=loadgen`
(for example `rezolus mcp query run.dendro ... --recording role=loadgen`):
when the server under test is another llm-perf, `source=llm-perf` matches
both.

### KPI template

llm-perf's KPI template is `templates/llm-perf.json`, a Rezolus
`ServiceExtension`: the throughput, latency and error charts the Rezolus
viewer shows as llm-perf's section. It travels in every recording llm-perf
makes or serves, under the `service_queries` metadata key: in the
`/metrics/stream` handshake, which `rezolus record` keeps in the source's
metadata; in llm-perf's own source in the run's `.dendro`; and in the file
metadata of the parquet written at exit. The viewer renders the template a
`.dendro` recording carries, so a change to it ships with llm-perf. Rezolus's
built-in copy is used only for recordings that carry none (made by an older
llm-perf, or scraped from `/metrics`). Every query selects
`source="llm-perf"`, and a parquet's series carry no `source` label (the
source is in the file metadata only), so the parquet's KPI charts are empty
until rezolus attaches a single-source parquet's source to its series as a
label. Tests in `src/template.rs` check that each query names metrics only
through selectors, each naming `source="llm-perf"`, that those are metrics
llm-perf registers, with label values its metrics carry, and that every
`unit_system` is one the viewer knows.

llm-perf attaches no template to the server under test's recording. A
streamed server sends its own. A scraped one (vLLM, SGLang) is named after
its host and port unless `[metrics] server_source` names it; the viewer's
built-in vLLM and SGLang templates select `source="vllm"` and
`source="sglang"`, so set `server_source = "vllm"` (or `"sglang"`) for them
to apply. `server_source` needs `server_metrics_url`, and an empty value or
`llm-perf` (in any case) is refused at startup:

```toml
[metrics]
output = "run.dendro"
server_metrics_url = "http://localhost:8000/metrics"
server_source = "vllm"
```

When `rezolus record` records the server instead, give it the same name:
`rezolus record --endpoint http://host:8000/metrics,source=vllm -o
server.dendro`; a recording whose source is `vllm` gets the viewer's built-in
vLLM template.

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

Results go to `eval_results/`, and a rerun resumes from the questions already answered there. A run also writes `eval_results/run_config.json` with these settings: `model`, `mode`, `num_shots`, `max_context_tokens`, `max_tokens`, `temperature`, `top_p`, `frequency_penalty`, `presence_penalty` and `system_prompt`. `model` is the `--model` value, or else the first id the server lists at `/v1/models`. llama-server reports the `--alias`, the `-hf` name or the GGUF file name, so changing how the model is loaded changes it. A rerun with `--model <saved value>` resumes. If `eval_results/` holds saved results and a rerun's settings differ from that file, the rerun stops with an error listing each difference; move or delete `eval_results/` to start fresh. If there are no saved results, the file is overwritten. Results written by an older llm-perf have no `run_config.json`; they are resumed with a warning that they could not be checked.

By default the few-shot prompt is sent as chat turns to `/chat/completions`. For base models without a chat template, set `[inference] mode = "completion"` (or pass `--mode completion`) to send a single plain-text prompt to `/completions` in the layout of TIGER-Lab's script for evaluating local models (`evaluate_from_local.py`).

For models with a short context window, set `[inference] max_context_tokens` (or `--max-context-tokens`) to the context length available to one request. For llama-server that is `default_generation_settings.n_ctx` from `GET /props`. This can be less than `-c`. With `-np` greater than 1 and without `--kv-unified`, it is `-c` divided by `-np`, rounded up to a multiple of 256. `--kv-unified-per-slot` and the model's training context also cap it. Each question then drops few-shot examples, last first, until the prompt plus `max_tokens` fits, as TIGER-Lab's script does. Prompt tokens are counted by llama-server's `POST /tokenize` with `add_special: true`, so the count includes BOS. In chat mode the messages are first rendered with `POST /apply-template`, so the count includes the chat template. Other servers are not supported for this option, and the run stops at startup if either endpoint fails. Questions that do not fit even at 0 shots are not sent. They are reported as skipped and counted as wrong, and a resumed run tries them again. The reports record how many questions received a response at each shot count.

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
│   ├── recording.rs      # The run's .dendro archive, recorded in process
│   ├── admin.rs          # HTTP admin server (Prometheus/JSON/msgpack metrics, /metrics/stream)
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
