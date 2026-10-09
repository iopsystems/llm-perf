# Recording llm-perf and the server under test into one `.dendro`

**Status:** OPEN. Plan opened 2026-10-08, nothing built.

Related entries opened the same day: rezolus
`docs/journal/2026-10-08-6-0-release-readiness.md`, metriken
`docs/journal/2026-10-08-one-recording-stack.md` (one recording, exposition
and viewing stack for every metriken producer), systemslab
`docs/journal/2026-10-08-dendro-artifacts.md`, cachecannon
`docs/journal/2026-10-08-dendro-recording.md`.

## Today

- `metriken` 0.8.0 and `metriken-exposition` 0.14.0 from a git dependency
  (rev `98db358` in `Cargo.lock`).
- `[metrics] output`: `src/snapshot.rs` appends msgpack snapshots to a
  temporary file and converts them to parquet at exit through
  `MsgpackToParquet`.
- The optional admin server's `/metrics` (`src/admin.rs`) renders histograms
  as percentile gauges (`PrometheusOptions::with_percentiles`).
- `src/server_metrics.rs` runs `rezolus record <url> <output> --interval <i>`
  to record the server under test's Prometheus `/metrics` into a second
  parquet, `<output-stem>.server.parquet` by default (`src/config.rs`). Its
  comment explains why it avoids `--endpoint url,source=...`: the annotated
  form collided with the `[OUTPUT]` positional.
- Downstream, rezolus ships an `llm-perf.json` KPI template and recognizes
  `source=llm-perf`; the client and server recordings are separate files,
  joined with `rezolus recording combine`.

## Plan

1. **One recorder for both sides.** Change `record_args` to
   `rezolus record --endpoint <server>/metrics,source=<service> --endpoint <llm-perf admin>/metrics,source=llm-perf -o <run>.dendro --interval <i>`.
   `-o` removes the positional collision noted in `src/server_metrics.rs`.
   The run then leaves one archive holding the server and llm-perf on one
   timeline, readable while the run goes and after a crash. This needs rezolus
   6.0's `record`. v5.25.1's `record` help text says a Prometheus endpoint
   records into a `.dendro` too; that is not run.
2. **Histograms as buckets on `/metrics`.** Expose histograms as Prometheus
   `_bucket` series rather than percentile gauges, so a scrape keeps a
   distribution that rezolus's Prometheus conversion turns back into a
   histogram, approximately. The percentile gauges are kept if anything reads
   them. `src/admin.rs` builds its output with
   `PrometheusOptions::with_percentiles`; whether metriken-exposition 0.14 can
   emit `_bucket` series without a metriken change is not checked.
3. **Leave the git dependency.** Move to published metriken 0.11 and
   metriken-exposition 0.21. `metriken-core` declares `links`, so a build holds
   one metriken-core version: llm-perf's metriken 0.8 is on core 0.1, and the
   stream route and recording its own registry (both through
   `metriken-exposition`) need metriken 0.11, on core 0.3.
4. **Serve `/metrics/stream`** through `metriken-exposition`'s stream route
   for a registry (metriken entry, piece 1), and pass llm-perf's base URL in
   `record_args` rather than its `/metrics` path, so the recorder can find the
   stream. Once rezolus detects a source by its stream (rezolus entry,
   "Endpoint detection"), the recorder records llm-perf as it records a
   Rezolus agent: rows stamped by llm-perf at read time and native histograms,
   with no Prometheus conversion. Step 2's buckets then only matter to other
   Prometheus consumers.
5. **Record in process with `metriken-recorder`** (metriken entry, piece 2):
   llm-perf records the server under test and its own registry into one
   archive itself, in place of running `rezolus record`, and no longer needs a
   rezolus binary. This replaces step 1's launcher.
6. **The template moves here.** llm-perf's dashboard template
   (`llm-perf.json`, today in rezolus's `crates/dashboard/templates/`) is
   owned by llm-perf and written into the archive, in the stream's handshake
   and in its own source's metadata. The viewer loads it from the archive
   (metriken entry, piece 3).
7. **Retire the parquet-at-exit path** once step 5 is in use, keeping
   `[metrics] output` as an option for one release.

Step 1 needs no metriken change and can ship first; step 2 may need one (see
above). Step 4 waits for the metriken entry's path step 3 (the stream route),
step 5 for path step 4 (`metriken-recorder`), and step 6 for path step 5
(templates).

## GO criteria

- One llm-perf run leaves one `.dendro` with the server and llm-perf as two
  sources, and `rezolus view <run>.dendro` shows the llm-perf KPI dashboard for
  the llm-perf recording without `recording combine`.
- After step 4, TTFT and inter-token latency in that archive are histograms,
  with the same percentiles as the parquet path for the same run within the
  histogram's bucket error.
