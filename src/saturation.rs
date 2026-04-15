//! Saturation search state management.
//!
//! Finds the maximum concurrency an LLM server can handle while maintaining
//! SLO compliance on TTFT, ITL, and/or TPOT latency percentiles.

use crate::config::{SaturationConfig, SloPercentiles};
use crate::metrics;

use metriken::AtomicHistogram;
use metriken::histogram::Histogram;
use serde::Serialize;
use std::time::Instant;
use tokio::sync::Semaphore;

use std::sync::Arc;

/// Collected percentiles for a single step (used internally to reduce arg count).
struct StepPercentiles {
    ttft_p50: f64,
    ttft_p99: f64,
    ttft_p999: f64,
    itl_p50: f64,
    itl_p99: f64,
    itl_p999: f64,
    tpot_p50: f64,
    tpot_p99: f64,
    tpot_p999: f64,
}

/// State machine for concurrency-based saturation search.
pub struct SaturationSearchState {
    config: SaturationConfig,
    semaphore: Arc<Semaphore>,
    sample_window: std::time::Duration,

    current_concurrency: usize,
    last_good_concurrency: Option<usize>,
    last_good_throughput: Option<f64>,
    consecutive_failures: u32,
    step_start: Instant,

    // Histogram snapshots at step start (for delta computation)
    step_ttft_snapshot: Option<Histogram>,
    step_itl_snapshot: Option<Histogram>,
    step_tpot_snapshot: Option<Histogram>,

    // Counter snapshots at step start
    step_output_tokens: u64,
    step_requests_success: u64,

    results: Vec<SaturationStep>,
    completed: bool,
    header_printed: bool,
}

/// Result of a single concurrency step.
#[derive(Debug, Clone, Serialize)]
pub struct SaturationStep {
    pub concurrency: usize,
    pub duration_secs: f64,
    pub requests_completed: u64,
    pub output_tokens_per_sec: f64,
    pub ttft_p50_ms: f64,
    pub ttft_p99_ms: f64,
    pub ttft_p999_ms: f64,
    pub itl_p50_ms: f64,
    pub itl_p99_ms: f64,
    pub itl_p999_ms: f64,
    pub tpot_p50_ms: f64,
    pub tpot_p99_ms: f64,
    pub tpot_p999_ms: f64,
    pub slo_passed: bool,
    pub fail_reason: String,
}

/// Final saturation search results.
#[derive(Debug, Clone, Serialize)]
pub struct SaturationResults {
    pub max_compliant_concurrency: Option<usize>,
    pub steps: Vec<SaturationStep>,
}

impl SaturationSearchState {
    pub fn new(config: SaturationConfig, semaphore: Arc<Semaphore>) -> Self {
        let sample_window =
            humantime::parse_duration(&config.sample_window).expect("validated in Config");

        Self {
            config,
            semaphore,
            sample_window,
            current_concurrency: 0, // set in initialize()
            last_good_concurrency: None,
            last_good_throughput: None,
            consecutive_failures: 0,
            step_start: Instant::now(),
            step_ttft_snapshot: None,
            step_itl_snapshot: None,
            step_tpot_snapshot: None,
            step_output_tokens: 0,
            step_requests_success: 0,
            results: Vec::new(),
            completed: false,
            header_printed: false,
        }
    }

    /// Take initial snapshots and record the starting state.
    /// Must be called after warmup completes and before the control loop begins.
    pub fn initialize(&mut self) {
        self.current_concurrency = self.config.start_concurrency;
        self.step_start = Instant::now();
        self.step_ttft_snapshot = merge_histograms(&metrics::ALL_TTFT);
        self.step_itl_snapshot = merge_histograms(&metrics::ALL_ITL);
        self.step_tpot_snapshot = merge_histograms(&metrics::ALL_TPOT);
        self.step_output_tokens = output_tokens_total();
        self.step_requests_success = metrics::REQUESTS_SUCCESS.value();
    }

    /// Check if the sample window has elapsed and advance to the next step.
    ///
    /// Returns `true` if a step was completed.
    pub fn check_and_advance(&mut self) -> bool {
        if self.completed {
            return false;
        }

        if self.step_start.elapsed() < self.sample_window {
            return false;
        }

        if !self.header_printed {
            print_header();
            self.header_printed = true;
        }

        let elapsed_secs = self.step_start.elapsed().as_secs_f64();

        // Snapshot current state
        let current_ttft = merge_histograms(&metrics::ALL_TTFT);
        let current_itl = merge_histograms(&metrics::ALL_ITL);
        let current_tpot = merge_histograms(&metrics::ALL_TPOT);
        let current_output_tokens = output_tokens_total();
        let current_requests_success = metrics::REQUESTS_SUCCESS.value();

        // Compute deltas
        let delta_output_tokens = current_output_tokens.saturating_sub(self.step_output_tokens);
        let delta_requests = current_requests_success.saturating_sub(self.step_requests_success);
        let output_tokens_per_sec = delta_output_tokens as f64 / elapsed_secs;

        let delta_ttft = compute_delta(&current_ttft, &self.step_ttft_snapshot);
        let delta_itl = compute_delta(&current_itl, &self.step_itl_snapshot);
        let delta_tpot = compute_delta(&current_tpot, &self.step_tpot_snapshot);

        // Extract percentiles
        let (ttft_p50, ttft_p99, ttft_p999) = extract_percentiles_ms(&delta_ttft);
        let (itl_p50, itl_p99, itl_p999) = extract_percentiles_ms(&delta_itl);
        let (tpot_p50, tpot_p99, tpot_p999) = extract_percentiles_ms(&delta_tpot);

        let percentiles = StepPercentiles {
            ttft_p50,
            ttft_p99,
            ttft_p999,
            itl_p50,
            itl_p99,
            itl_p999,
            tpot_p50,
            tpot_p99,
            tpot_p999,
        };

        // Check SLO
        let latency_reason = self.slo_fail_reason(&percentiles);

        // Check throughput ratio (skip on first step — no baseline yet)
        let throughput_ok = if let Some(last_throughput) = self.last_good_throughput {
            let last_concurrency = self.last_good_concurrency.unwrap_or(1) as f64;
            let expected = last_throughput * (self.current_concurrency as f64 / last_concurrency);
            let ratio = output_tokens_per_sec / expected;
            ratio >= self.config.min_throughput_ratio
        } else {
            true
        };

        let slo_passed = throughput_ok && latency_reason.is_none();

        let fail_reason = if slo_passed {
            String::new()
        } else if !throughput_ok {
            let last_throughput = self.last_good_throughput.unwrap_or(0.0);
            let last_concurrency = self.last_good_concurrency.unwrap_or(1) as f64;
            let expected = last_throughput * (self.current_concurrency as f64 / last_concurrency);
            let ratio = output_tokens_per_sec / expected;
            format!(
                "Throughput: {:.0}% of expected (need {:.0}%)",
                ratio * 100.0,
                self.config.min_throughput_ratio * 100.0
            )
        } else {
            latency_reason.unwrap_or_default()
        };

        let step = SaturationStep {
            concurrency: self.current_concurrency,
            duration_secs: elapsed_secs,
            requests_completed: delta_requests,
            output_tokens_per_sec,
            ttft_p50_ms: ttft_p50,
            ttft_p99_ms: ttft_p99,
            ttft_p999_ms: ttft_p999,
            itl_p50_ms: itl_p50,
            itl_p99_ms: itl_p99,
            itl_p999_ms: itl_p999,
            tpot_p50_ms: tpot_p50,
            tpot_p99_ms: tpot_p99,
            tpot_p999_ms: tpot_p999,
            slo_passed,
            fail_reason,
        };

        print_step(self.results.len() + 1, &step);
        self.results.push(step);

        // Update state
        if slo_passed {
            self.last_good_concurrency = Some(self.current_concurrency);
            self.last_good_throughput = Some(output_tokens_per_sec);
            self.consecutive_failures = 0;
        } else {
            self.consecutive_failures += 1;
        }

        // Check termination
        if self.consecutive_failures >= self.config.stop_after_failures {
            self.completed = true;
            print_summary(&self.results());
            return true;
        }

        // Advance concurrency
        let next = ((self.current_concurrency as f64 * self.config.step_multiplier).ceil()
            as usize)
            .max(self.current_concurrency + 1); // guarantee progress

        if next > self.config.max_concurrency {
            self.completed = true;
            print_summary(&self.results());
            return true;
        }

        // Grow the semaphore
        let delta_permits = next - self.current_concurrency;
        self.semaphore.add_permits(delta_permits);
        self.current_concurrency = next;

        // Reset step tracking
        self.step_start = Instant::now();
        self.step_ttft_snapshot = current_ttft;
        self.step_itl_snapshot = current_itl;
        self.step_tpot_snapshot = current_tpot;
        self.step_output_tokens = current_output_tokens;
        self.step_requests_success = current_requests_success;

        true
    }

    /// Check all configured SLO thresholds, returning the first violation found.
    fn slo_fail_reason(&self, percentiles: &StepPercentiles) -> Option<String> {
        if let Some(ref slo) = self.config.slo.ttft
            && let Some(reason) = check_percentile_slo(
                "TTFT",
                slo,
                percentiles.ttft_p50,
                percentiles.ttft_p99,
                percentiles.ttft_p999,
            )
        {
            return Some(reason);
        }
        if let Some(ref slo) = self.config.slo.itl
            && let Some(reason) = check_percentile_slo(
                "ITL",
                slo,
                percentiles.itl_p50,
                percentiles.itl_p99,
                percentiles.itl_p999,
            )
        {
            return Some(reason);
        }
        if let Some(ref slo) = self.config.slo.tpot
            && let Some(reason) = check_percentile_slo(
                "TPOT",
                slo,
                percentiles.tpot_p50,
                percentiles.tpot_p99,
                percentiles.tpot_p999,
            )
        {
            return Some(reason);
        }
        None
    }

    pub fn is_completed(&self) -> bool {
        self.completed
    }

    pub fn current_concurrency(&self) -> usize {
        self.current_concurrency
    }

    pub fn results(&self) -> SaturationResults {
        SaturationResults {
            max_compliant_concurrency: self.last_good_concurrency,
            steps: self.results.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn check_percentile_slo(
    metric_name: &str,
    slo: &SloPercentiles,
    p50: f64,
    p99: f64,
    p999: f64,
) -> Option<String> {
    if let Some(threshold) = slo.p50_ms
        && p50 > threshold
    {
        return Some(format!(
            "{} p50 {:.0}ms > {:.0}ms SLO",
            metric_name, p50, threshold
        ));
    }
    if let Some(threshold) = slo.p99_ms
        && p99 > threshold
    {
        return Some(format!(
            "{} p99 {:.0}ms > {:.0}ms SLO",
            metric_name, p99, threshold
        ));
    }
    if let Some(threshold) = slo.p999_ms
        && p999 > threshold
    {
        return Some(format!(
            "{} p999 {:.0}ms > {:.0}ms SLO",
            metric_name, p999, threshold
        ));
    }
    None
}

/// Merge an array of context-bucketed AtomicHistograms into a single Histogram.
fn merge_histograms(histograms: &[&AtomicHistogram]) -> Option<Histogram> {
    let mut merged: Option<Histogram> = None;
    for h in histograms {
        if let Some(loaded) = h.load() {
            merged = Some(match merged {
                Some(existing) => existing.checked_add(&loaded).unwrap_or(existing),
                None => loaded,
            });
        }
    }
    merged
}

/// Compute a delta histogram (current - previous snapshot).
fn compute_delta(current: &Option<Histogram>, previous: &Option<Histogram>) -> Option<Histogram> {
    match (current, previous) {
        (Some(cur), Some(prev)) => cur.wrapping_sub(prev).ok(),
        (Some(cur), None) => Some(cur.clone()),
        _ => None,
    }
}

/// Extract (p50, p99, p999) from a histogram in milliseconds.
fn extract_percentiles_ms(histogram: &Option<Histogram>) -> (f64, f64, f64) {
    let Some(hist) = histogram else {
        return (0.0, 0.0, 0.0);
    };

    let mut p50 = 0.0;
    let mut p99 = 0.0;
    let mut p999 = 0.0;

    if let Ok(Some(percentiles)) = hist.percentiles(&[50.0, 99.0, 99.9]) {
        for (pct, bucket) in percentiles.iter() {
            let value_ms = bucket.end() as f64 / 1_000_000.0;
            match pct.round() as u32 {
                50 => p50 = value_ms,
                99 => p99 = value_ms,
                // 99.9 rounds to 100
                _ => p999 = value_ms,
            }
        }
    }

    (p50, p99, p999)
}

/// Total output tokens (reasoning + content).
fn output_tokens_total() -> u64 {
    metrics::TOKENS_OUTPUT_REASONING.value() + metrics::TOKENS_OUTPUT_CONTENT.value()
}

// ---------------------------------------------------------------------------
// Console output
// ---------------------------------------------------------------------------

fn print_header() {
    println!();
    println!(
        "{:>6} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | Result",
        "Step", "Concurrency", "Tokens/s", "TTFT p99", "ITL p99", "TPOT p99"
    );
    println!("{}", "-".repeat(90));
}

fn print_step(step_num: usize, step: &SaturationStep) {
    let result = if step.slo_passed {
        "PASS".to_string()
    } else {
        format!("FAIL ({})", step.fail_reason)
    };

    println!(
        "{:>6} | {:>12} | {:>10.1} | {:>9.0}ms | {:>9.0}ms | {:>9.0}ms | {}",
        step_num,
        step.concurrency,
        step.output_tokens_per_sec,
        step.ttft_p99_ms,
        step.itl_p99_ms,
        step.tpot_p99_ms,
        result,
    );
}

fn print_summary(results: &SaturationResults) {
    println!("{}", "-".repeat(90));
    println!();
    if let Some(max_c) = results.max_compliant_concurrency {
        let best_step = results
            .steps
            .iter()
            .filter(|s| s.slo_passed)
            .max_by(|a, b| {
                a.output_tokens_per_sec
                    .partial_cmp(&b.output_tokens_per_sec)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        println!("Saturation Search Complete");
        println!("  Max compliant concurrency: {}", max_c);
        if let Some(step) = best_step {
            println!(
                "  Peak throughput: {:.1} tokens/s @ concurrency {}",
                step.output_tokens_per_sec, step.concurrency
            );
        }
    } else {
        println!("Saturation Search Complete");
        println!("  No compliant concurrency found — SLO failed at start_concurrency");
    }
    println!();
}
