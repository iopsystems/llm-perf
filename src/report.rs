use anyhow::Result;
use chrono::{DateTime, Utc};
use metriken::histogram::Histogram;
use serde::Serialize;
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkReport {
    // Metadata
    pub timestamp: DateTime<Utc>,
    pub duration: Duration,
    pub version: String,

    // Test configuration
    pub configuration: TestConfiguration,

    // Results
    pub summary: Summary,
    pub throughput: ThroughputStats,
    pub latency: LatencyStats,
    pub errors: ErrorBreakdown,

    // Context-aware metrics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_latency: Option<ContextLatencyStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_itl: Option<ContextITLStats>,

    // Multi-turn conversation metrics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<ConversationStats>,

    // Saturation search results
    #[serde(skip_serializing_if = "Option::is_none")]
    pub saturation: Option<crate::saturation::SaturationResults>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TestConfiguration {
    pub endpoint: String,
    pub model: String,
    pub load_pattern: String,
    pub concurrent_requests: Option<usize>,
    pub qps: Option<f64>,
    pub total_requests: Option<usize>,
    pub duration_seconds: Option<u64>,
    pub warmup_requests: Option<usize>,
    pub warmup_duration: Option<u64>,
    pub prompt_file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    pub sample_size: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorBreakdown {
    pub errors_timeout: u64,
    pub errors_connection: u64,
    pub errors_http_4xx: u64,
    pub errors_http_5xx: u64,
    pub errors_other: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub requests_total: u64,
    pub requests_successful: u64,
    pub requests_failed: u64,
    pub requests_error: u64,
    pub requests_timeout: u64,
    pub requests_canceled: u64,
    pub success_rate: f64,
    pub retries: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ThroughputStats {
    pub requests_per_second: f64,
    pub input_tokens_per_second: f64,
    pub output_tokens_per_second: f64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LatencyStats {
    pub ttft_mean_ms: f64,
    pub ttft_p50_ms: f64,
    pub ttft_p90_ms: f64,
    pub ttft_p95_ms: f64,
    pub ttft_p99_ms: f64,
    pub tpot_mean_ms: f64,
    pub tpot_p50_ms: f64,
    pub tpot_p90_ms: f64,
    pub tpot_p95_ms: f64,
    pub tpot_p99_ms: f64,
    pub itl_mean_ms: f64,
    pub itl_p50_ms: f64,
    pub itl_p90_ms: f64,
    pub itl_p95_ms: f64,
    pub itl_p99_ms: f64,
    pub request_mean_ms: f64,
    pub request_p50_ms: f64,
    pub request_p90_ms: f64,
    pub request_p95_ms: f64,
    pub request_p99_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContextLatencyStats {
    pub small: LatencyPercentiles,
    pub medium: LatencyPercentiles,
    pub large: LatencyPercentiles,
    pub xlarge: LatencyPercentiles,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xxlarge: Option<LatencyPercentiles>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LatencyPercentiles {
    pub ttft_p50_ms: f64,
    pub ttft_p90_ms: f64,
    pub ttft_p95_ms: f64,
    pub ttft_p99_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContextITLStats {
    pub small: ITLPercentiles,
    pub medium: ITLPercentiles,
    pub large: ITLPercentiles,
    pub xlarge: ITLPercentiles,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xxlarge: Option<ITLPercentiles>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ITLPercentiles {
    pub itl_p50_ms: f64,
    pub itl_p90_ms: f64,
    pub itl_p95_ms: f64,
    pub itl_p99_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConversationStats {
    pub conversations_sent: u64,
    pub conversations_success: u64,
    pub conversations_failed: u64,
    pub total_turns: u64,
    pub avg_turns_per_conversation: f64,
    pub conversation_latency_mean_ms: f64,
    pub conversation_latency_p50_ms: f64,
    pub conversation_latency_p90_ms: f64,
    pub conversation_latency_p95_ms: f64,
    pub conversation_latency_p99_ms: f64,
}

pub struct ReportBuilder {
    start_time: SystemTime,
    config: Option<crate::config::Config>,
    duration: Option<Duration>,
    saturation_results: Option<crate::saturation::SaturationResults>,
}

impl Default for ReportBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ReportBuilder {
    pub fn new() -> Self {
        Self {
            start_time: SystemTime::now(),
            config: None,
            duration: None,
            saturation_results: None,
        }
    }

    pub fn with_config(mut self, config: crate::config::Config) -> Self {
        self.config = Some(config);
        self
    }

    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    pub fn with_saturation_results(
        mut self,
        results: crate::saturation::SaturationResults,
    ) -> Self {
        self.saturation_results = Some(results);
        self
    }

    pub fn build(&self) -> Result<BenchmarkReport> {
        // Use provided duration if available, otherwise calculate from elapsed time
        let duration = if let Some(d) = self.duration {
            d
        } else {
            let end_time = SystemTime::now();
            end_time.duration_since(self.start_time)?
        };

        use crate::metrics::{
            ERRORS_CONNECTION, ERRORS_HTTP_4XX, ERRORS_HTTP_5XX, ERRORS_OTHER, REQUESTS_CANCELED,
            REQUESTS_ERROR, REQUESTS_RETRIED, REQUESTS_SENT, REQUESTS_SUCCESS, REQUESTS_TIMEOUT,
            TOKENS_INPUT, TOKENS_OUTPUT_CONTENT, TOKENS_OUTPUT_REASONING,
        };

        let requests_sent = REQUESTS_SENT.value();
        let requests_success = REQUESTS_SUCCESS.value();
        let requests_error = REQUESTS_ERROR.value();
        let requests_timeout = REQUESTS_TIMEOUT.value();
        let requests_canceled = REQUESTS_CANCELED.value();

        let input_tokens = TOKENS_INPUT.value();
        let output_tokens = TOKENS_OUTPUT_REASONING.value() + TOKENS_OUTPUT_CONTENT.value();

        let duration_secs = duration.as_secs_f64();

        // Build test configuration from config
        let configuration = if let Some(ref config) = self.config {
            TestConfiguration {
                endpoint: config.endpoint.base_url.clone(),
                model: config
                    .endpoint
                    .model
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                load_pattern: if config.load.qps.is_some() {
                    "FixedQps".to_string()
                } else {
                    "Concurrent".to_string()
                },
                concurrent_requests: Some(config.load.concurrent_requests),
                qps: config.load.qps,
                total_requests: config.load.total_requests,
                duration_seconds: config.load.duration_seconds,
                warmup_requests: config.load.warmup_requests,
                warmup_duration: config.load.warmup_duration,
                prompt_file: config.input.file.display().to_string(),
                seed: config.input.seed,
                sample_size: config.input.sample_size,
            }
        } else {
            // Default configuration if not provided
            TestConfiguration {
                endpoint: "unknown".to_string(),
                model: "unknown".to_string(),
                load_pattern: "unknown".to_string(),
                concurrent_requests: None,
                qps: None,
                total_requests: None,
                duration_seconds: None,
                warmup_requests: None,
                warmup_duration: None,
                prompt_file: "unknown".to_string(),
                seed: None,
                sample_size: None,
            }
        };

        let retries = REQUESTS_RETRIED.value();

        // Completed = received a complete response (success or server error)
        let requests_completed = requests_success + requests_error;
        // Failed = error + timeout (but not canceled)
        let requests_failed = requests_error + requests_timeout;

        let summary = Summary {
            requests_total: requests_sent,
            requests_successful: requests_success,
            requests_failed,
            requests_error,
            requests_timeout,
            requests_canceled,
            success_rate: if requests_completed > 0 {
                requests_success as f64 / requests_completed as f64
            } else {
                0.0
            },
            retries,
        };

        let throughput = ThroughputStats {
            requests_per_second: requests_sent as f64 / duration_secs,
            input_tokens_per_second: input_tokens as f64 / duration_secs,
            output_tokens_per_second: output_tokens as f64 / duration_secs,
            total_input_tokens: input_tokens,
            total_output_tokens: output_tokens,
        };

        // Build error breakdown
        let errors = ErrorBreakdown {
            errors_timeout: requests_timeout,
            errors_connection: ERRORS_CONNECTION.value(),
            errors_http_4xx: ERRORS_HTTP_4XX.value(),
            errors_http_5xx: ERRORS_HTTP_5XX.value(),
            errors_other: ERRORS_OTHER.value(),
        };

        // Get latency percentiles
        let latency = self.build_latency_stats()?;
        let context_latency = self.build_context_latency_stats();
        let context_itl = self.build_context_itl_stats();
        let conversation = self.build_conversation_stats();

        // Convert SystemTime to DateTime<Utc>
        let timestamp: DateTime<Utc> = self.start_time.into();

        Ok(BenchmarkReport {
            timestamp,
            duration,
            version: env!("CARGO_PKG_VERSION").to_string(),
            configuration,
            summary,
            throughput,
            latency,
            errors,
            context_latency,
            context_itl,
            conversation,
            saturation: self.saturation_results.clone(),
        })
    }

    /// Merge all context-aware histograms into a single combined histogram.
    fn merge_context_histograms(histograms: &[&metriken::AtomicHistogram]) -> Option<Histogram> {
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

    /// Extract percentiles from a histogram, returning (mean, p50, p90, p95, p99) in milliseconds.
    fn extract_percentiles_ms(histogram: &Histogram) -> (f64, f64, f64, f64, f64) {
        let mean = {
            let mut sum = 0u64;
            let mut count = 0u64;
            for bucket in histogram.iter() {
                let bucket_count = bucket.count();
                if bucket_count > 0 {
                    sum += bucket.end() * bucket_count;
                    count += bucket_count;
                }
            }
            if count > 0 {
                (sum as f64 / count as f64) / 1_000_000.0
            } else {
                0.0
            }
        };

        let mut p50 = 0.0;
        let mut p90 = 0.0;
        let mut p95 = 0.0;
        let mut p99 = 0.0;

        if let Ok(Some(percentiles)) = histogram.percentiles(&[50.0, 90.0, 95.0, 99.0]) {
            for (percentile, bucket) in percentiles.iter() {
                let value_ms = bucket.end() as f64 / 1_000_000.0;
                match percentile.round() as u32 {
                    50 => p50 = value_ms,
                    90 => p90 = value_ms,
                    95 => p95 = value_ms,
                    99 => p99 = value_ms,
                    _ => {}
                }
            }
        }

        (mean, p50, p90, p95, p99)
    }

    fn build_latency_stats(&self) -> Result<LatencyStats> {
        use crate::metrics::{ALL_ITL, ALL_TPOT, ALL_TTFT, REQUEST_LATENCY};

        // Aggregate TTFT (prefill latency) across context buckets
        let (ttft_mean, ttft_p50, ttft_p90, ttft_p95, ttft_p99) =
            if let Some(ttft) = Self::merge_context_histograms(&ALL_TTFT) {
                Self::extract_percentiles_ms(&ttft)
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0)
            };

        // Aggregate ITL across all phases and context buckets
        let (itl_mean, itl_p50, itl_p90, itl_p95, itl_p99) =
            if let Some(itl) = Self::merge_context_histograms(&ALL_ITL) {
                Self::extract_percentiles_ms(&itl)
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0)
            };

        // Aggregate TPOT across all phases
        let (tpot_mean, tpot_p50, tpot_p90, tpot_p95, tpot_p99) =
            if let Some(tpot) = Self::merge_context_histograms(&ALL_TPOT) {
                Self::extract_percentiles_ms(&tpot)
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0)
            };

        // Extract request latency percentiles
        let (request_mean, request_p50, request_p90, request_p95, request_p99) =
            if let Some(request) = REQUEST_LATENCY.load() {
                Self::extract_percentiles_ms(&request)
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0)
            };

        Ok(LatencyStats {
            ttft_mean_ms: ttft_mean,
            ttft_p50_ms: ttft_p50,
            ttft_p90_ms: ttft_p90,
            ttft_p95_ms: ttft_p95,
            ttft_p99_ms: ttft_p99,
            tpot_mean_ms: tpot_mean,
            tpot_p50_ms: tpot_p50,
            tpot_p90_ms: tpot_p90,
            tpot_p95_ms: tpot_p95,
            tpot_p99_ms: tpot_p99,
            itl_mean_ms: itl_mean,
            itl_p50_ms: itl_p50,
            itl_p90_ms: itl_p90,
            itl_p95_ms: itl_p95,
            itl_p99_ms: itl_p99,
            request_mean_ms: request_mean,
            request_p50_ms: request_p50,
            request_p90_ms: request_p90,
            request_p95_ms: request_p95,
            request_p99_ms: request_p99,
        })
    }

    fn build_context_latency_stats(&self) -> Option<ContextLatencyStats> {
        use crate::metrics::{TTFT_LARGE, TTFT_MEDIUM, TTFT_SMALL, TTFT_XLARGE, TTFT_XXLARGE};

        let extract_percentiles =
            |histogram: &metriken::AtomicHistogram| -> Option<LatencyPercentiles> {
                if let Some(loaded) = histogram.load()
                    && let Ok(Some(percentiles)) = loaded.percentiles(&[50.0, 90.0, 95.0, 99.0])
                    && !percentiles.is_empty()
                {
                    let mut p50 = 0.0;
                    let mut p90 = 0.0;
                    let mut p95 = 0.0;
                    let mut p99 = 0.0;
                    for (percentile, bucket) in percentiles.iter() {
                        let value_ms = bucket.end() as f64 / 1_000_000.0;
                        match percentile.round() as u32 {
                            50 => p50 = value_ms,
                            90 => p90 = value_ms,
                            95 => p95 = value_ms,
                            99 => p99 = value_ms,
                            _ => {}
                        }
                    }
                    return Some(LatencyPercentiles {
                        ttft_p50_ms: p50,
                        ttft_p90_ms: p90,
                        ttft_p95_ms: p95,
                        ttft_p99_ms: p99,
                    });
                }
                None
            };

        let has_data = TTFT_SMALL.load().is_some()
            || TTFT_MEDIUM.load().is_some()
            || TTFT_LARGE.load().is_some()
            || TTFT_XLARGE.load().is_some()
            || TTFT_XXLARGE.load().is_some();

        if !has_data {
            return None;
        }

        let small = extract_percentiles(&TTFT_SMALL)?;
        let medium = extract_percentiles(&TTFT_MEDIUM)?;
        let large = extract_percentiles(&TTFT_LARGE)?;
        let xlarge = extract_percentiles(&TTFT_XLARGE)?;
        let xxlarge = extract_percentiles(&TTFT_XXLARGE);

        Some(ContextLatencyStats {
            small,
            medium,
            large,
            xlarge,
            xxlarge,
        })
    }

    fn build_context_itl_stats(&self) -> Option<ContextITLStats> {
        use crate::metrics::{
            ITL_CONTENT_LARGE, ITL_CONTENT_MEDIUM, ITL_CONTENT_SMALL, ITL_CONTENT_XLARGE,
            ITL_CONTENT_XXLARGE, ITL_REASONING_LARGE, ITL_REASONING_MEDIUM, ITL_REASONING_SMALL,
            ITL_REASONING_XLARGE, ITL_REASONING_XXLARGE,
        };

        let extract_merged = |histograms: &[&metriken::AtomicHistogram]| -> Option<ITLPercentiles> {
            let merged = Self::merge_context_histograms(histograms)?;
            if let Ok(Some(percentiles)) = merged.percentiles(&[50.0, 90.0, 95.0, 99.0])
                && !percentiles.is_empty()
            {
                let mut p50 = 0.0;
                let mut p90 = 0.0;
                let mut p95 = 0.0;
                let mut p99 = 0.0;
                for (percentile, bucket) in percentiles.iter() {
                    let value_ms = bucket.end() as f64 / 1_000_000.0;
                    match percentile.round() as u32 {
                        50 => p50 = value_ms,
                        90 => p90 = value_ms,
                        95 => p95 = value_ms,
                        99 => p99 = value_ms,
                        _ => {}
                    }
                }
                return Some(ITLPercentiles {
                    itl_p50_ms: p50,
                    itl_p90_ms: p90,
                    itl_p95_ms: p95,
                    itl_p99_ms: p99,
                });
            }
            None
        };

        // Merge both phases per context size (ITL merging IS meaningful — overall decode speed)
        let small = extract_merged(&[&ITL_REASONING_SMALL, &ITL_CONTENT_SMALL])?;
        let medium = extract_merged(&[&ITL_REASONING_MEDIUM, &ITL_CONTENT_MEDIUM])?;
        let large = extract_merged(&[&ITL_REASONING_LARGE, &ITL_CONTENT_LARGE])?;
        let xlarge = extract_merged(&[&ITL_REASONING_XLARGE, &ITL_CONTENT_XLARGE])?;
        let xxlarge = extract_merged(&[&ITL_REASONING_XXLARGE, &ITL_CONTENT_XXLARGE]);

        Some(ContextITLStats {
            small,
            medium,
            large,
            xlarge,
            xxlarge,
        })
    }

    fn build_conversation_stats(&self) -> Option<ConversationStats> {
        use crate::metrics::{
            CONVERSATION_LATENCY, CONVERSATIONS_FAILED, CONVERSATIONS_SENT, CONVERSATIONS_SUCCESS,
            TURNS_TOTAL,
        };

        let sent = CONVERSATIONS_SENT.value();
        if sent == 0 {
            return None;
        }

        let success = CONVERSATIONS_SUCCESS.value();
        let failed = CONVERSATIONS_FAILED.value();
        let turns = TURNS_TOTAL.value();
        let completed = success + failed;
        let avg_turns = if completed > 0 {
            turns as f64 / completed as f64
        } else {
            0.0
        };

        let (mean, p50, p90, p95, p99) = if let Some(hist) = CONVERSATION_LATENCY.load() {
            Self::extract_percentiles_ms(&hist)
        } else {
            (0.0, 0.0, 0.0, 0.0, 0.0)
        };

        Some(ConversationStats {
            conversations_sent: sent,
            conversations_success: success,
            conversations_failed: failed,
            total_turns: turns,
            avg_turns_per_conversation: avg_turns,
            conversation_latency_mean_ms: mean,
            conversation_latency_p50_ms: p50,
            conversation_latency_p90_ms: p90,
            conversation_latency_p95_ms: p95,
            conversation_latency_p99_ms: p99,
        })
    }

    pub fn print_console_report(&self) -> Result<()> {
        let report = self.build()?;

        use chrono::Utc;
        let now = Utc::now();
        let timestamp = now.to_rfc3339_opts(chrono::SecondsFormat::Millis, false);

        println!();
        println!("{}", timestamp);
        println!("{} -----", timestamp);
        println!("{} Benchmark Complete", timestamp);
        println!(
            "{} Duration: {:.1}s",
            timestamp,
            report.duration.as_secs_f64()
        );
        println!(
            "{} Requests: Sent: {} Retries: {}",
            timestamp, report.summary.requests_total, report.summary.retries
        );
        println!(
            "{} Responses: Ok: {} Err: {} Timeout: {} Success: {:.2}%",
            timestamp,
            report.summary.requests_successful,
            report.summary.requests_error,
            report.summary.requests_timeout,
            report.summary.success_rate * 100.0
        );
        if report.summary.requests_canceled > 0 {
            println!(
                "{} Canceled: {}",
                timestamp, report.summary.requests_canceled
            );
        }

        // Error breakdown if any
        let total_errors = report.errors.errors_connection
            + report.errors.errors_http_4xx
            + report.errors.errors_http_5xx
            + report.errors.errors_other;
        if total_errors > 0 {
            println!(
                "{} Errors: Connection: {} 4xx: {} 5xx: {} Other: {}",
                timestamp,
                report.errors.errors_connection,
                report.errors.errors_http_4xx,
                report.errors.errors_http_5xx,
                report.errors.errors_other
            );
        }

        println!(
            "{} Tokens: Input: {} Output: {} Total: {}",
            timestamp,
            report.throughput.total_input_tokens,
            report.throughput.total_output_tokens,
            report.throughput.total_input_tokens + report.throughput.total_output_tokens
        );

        println!(
            "{} Throughput: Requests/s: {:.2} Input tokens/s: {:.2} Output tokens/s: {:.2}",
            timestamp,
            report.throughput.requests_per_second,
            report.throughput.input_tokens_per_second,
            report.throughput.output_tokens_per_second
        );

        if report.latency.ttft_p50_ms > 0.0 {
            println!(
                "{} TTFT (ms): mean: {:.1} p50: {:.0} p90: {:.0} p95: {:.0} p99: {:.0}",
                timestamp,
                report.latency.ttft_mean_ms,
                report.latency.ttft_p50_ms,
                report.latency.ttft_p90_ms,
                report.latency.ttft_p95_ms,
                report.latency.ttft_p99_ms
            );
        }

        if report.latency.tpot_p50_ms > 0.0 {
            println!(
                "{} TPOT (ms): mean: {:.1} p50: {:.0} p90: {:.0} p95: {:.0} p99: {:.0}",
                timestamp,
                report.latency.tpot_mean_ms,
                report.latency.tpot_p50_ms,
                report.latency.tpot_p90_ms,
                report.latency.tpot_p95_ms,
                report.latency.tpot_p99_ms
            );
        }

        if report.latency.itl_p50_ms > 0.0 {
            println!(
                "{} ITL (ms): mean: {:.1} p50: {:.0} p90: {:.0} p95: {:.0} p99: {:.0}",
                timestamp,
                report.latency.itl_mean_ms,
                report.latency.itl_p50_ms,
                report.latency.itl_p90_ms,
                report.latency.itl_p95_ms,
                report.latency.itl_p99_ms
            );
        }

        println!(
            "{} Request Latency (ms): mean: {:.1} p50: {:.0} p90: {:.0} p95: {:.0} p99: {:.0}",
            timestamp,
            report.latency.request_mean_ms,
            report.latency.request_p50_ms,
            report.latency.request_p90_ms,
            report.latency.request_p95_ms,
            report.latency.request_p99_ms
        );

        // Multi-turn conversation stats
        if let Some(ref conv) = report.conversation {
            println!(
                "{} Conversations: Sent: {} Ok: {} Err: {} Turns: {} Avg turns: {:.1}",
                timestamp,
                conv.conversations_sent,
                conv.conversations_success,
                conv.conversations_failed,
                conv.total_turns,
                conv.avg_turns_per_conversation
            );
            println!(
                "{} Conversation Latency (ms): mean: {:.1} p50: {:.0} p90: {:.0} p95: {:.0} p99: {:.0}",
                timestamp,
                conv.conversation_latency_mean_ms,
                conv.conversation_latency_p50_ms,
                conv.conversation_latency_p90_ms,
                conv.conversation_latency_p95_ms,
                conv.conversation_latency_p99_ms
            );
        }

        println!("\n");

        Ok(())
    }
}
