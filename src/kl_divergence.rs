use anyhow::Result;
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;

use crate::logprobs::{LogprobRecord, load_logprob_file};

/// Small epsilon for smoothing to avoid log(0)
const EPSILON: f64 = 1e-10;

/// Per-position KL divergence value
#[derive(Debug, Clone, Serialize)]
pub struct PositionKl {
    pub position: usize,
    pub kl_divergence: f64,
}

/// Per-prompt KL divergence breakdown
#[derive(Debug, Clone, Serialize)]
pub struct PromptKl {
    pub prompt_index: usize,
    pub mean_kl: f64,
    pub max_kl: f64,
    pub num_positions: usize,
}

/// Aggregate KL divergence statistics
#[derive(Debug, Clone, Serialize)]
pub struct KlReport {
    pub num_prompts_compared: usize,
    pub num_positions_compared: usize,
    pub num_prompts_skipped: usize,
    pub aggregate: AggregateStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_prompt: Option<Vec<PromptKl>>,
}

/// Aggregate statistics across all token positions
#[derive(Debug, Clone, Serialize)]
pub struct AggregateStats {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
}

/// Run the KL divergence comparison subcommand
pub fn run_kl_divergence(
    baseline_path: &Path,
    candidate_path: &Path,
    format: &str,
    output: Option<&Path>,
) -> Result<()> {
    let baseline = load_logprob_file(baseline_path)?;
    let candidate = load_logprob_file(candidate_path)?;

    if baseline.is_empty() {
        anyhow::bail!("Baseline file is empty: {}", baseline_path.display());
    }
    if candidate.is_empty() {
        anyhow::bail!("Candidate file is empty: {}", candidate_path.display());
    }

    let include_per_prompt = format == "json";
    let report = compute_kl_report(&baseline, &candidate, include_per_prompt)?;

    match format {
        "json" => {
            let json = serde_json::to_string_pretty(&report)?;
            if let Some(path) = output {
                std::fs::write(path, &json)?;
                println!("KL divergence report written to {}", path.display());
            } else {
                println!("{}", json);
            }
        }
        _ => {
            print_console_report(&report, baseline_path, candidate_path);
            if let Some(path) = output {
                let json = serde_json::to_string_pretty(&report)?;
                std::fs::write(path, &json)?;
                println!("\nDetailed report written to {}", path.display());
            }
        }
    }

    Ok(())
}

/// Compute KL divergence report comparing baseline and candidate logprob records
fn compute_kl_report(
    baseline: &[LogprobRecord],
    candidate: &[LogprobRecord],
    include_per_prompt: bool,
) -> Result<KlReport> {
    // Index candidate records by prompt_index for matching
    let candidate_map: HashMap<usize, &LogprobRecord> =
        candidate.iter().map(|r| (r.prompt_index, r)).collect();

    let mut all_kl_values: Vec<f64> = Vec::new();
    let mut per_prompt_results: Vec<PromptKl> = Vec::new();
    let mut skipped = 0;

    for base_record in baseline {
        let Some(cand_record) = candidate_map.get(&base_record.prompt_index) else {
            skipped += 1;
            log::warn!(
                "No matching candidate for prompt_index {}; skipping",
                base_record.prompt_index
            );
            continue;
        };

        let num_positions = base_record.tokens.len().min(cand_record.tokens.len());
        if num_positions == 0 {
            skipped += 1;
            continue;
        }

        if base_record.tokens.len() != cand_record.tokens.len() {
            log::debug!(
                "prompt_index {}: baseline has {} tokens, candidate has {}; comparing first {}",
                base_record.prompt_index,
                base_record.tokens.len(),
                cand_record.tokens.len(),
                num_positions
            );
        }

        let mut prompt_kl_values: Vec<f64> = Vec::new();

        for pos in 0..num_positions {
            let base_token = &base_record.tokens[pos];
            let cand_token = &cand_record.tokens[pos];

            let kl = compute_position_kl(&base_token.top_logprobs, &cand_token.top_logprobs);

            prompt_kl_values.push(kl);
            all_kl_values.push(kl);
        }

        if include_per_prompt && !prompt_kl_values.is_empty() {
            let mean_kl = prompt_kl_values.iter().sum::<f64>() / prompt_kl_values.len() as f64;
            let max_kl = prompt_kl_values
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);

            per_prompt_results.push(PromptKl {
                prompt_index: base_record.prompt_index,
                mean_kl,
                max_kl,
                num_positions: prompt_kl_values.len(),
            });
        }
    }

    if all_kl_values.is_empty() {
        anyhow::bail!("No matching prompt pairs found between baseline and candidate");
    }

    let aggregate = compute_aggregate_stats(&mut all_kl_values);

    Ok(KlReport {
        num_prompts_compared: baseline.len() - skipped,
        num_positions_compared: all_kl_values.len(),
        num_prompts_skipped: skipped,
        aggregate,
        per_prompt: if include_per_prompt {
            Some(per_prompt_results)
        } else {
            None
        },
    })
}

/// Compute KL(P || Q) for a single token position using top_logprobs
///
/// Converts logprobs to probability distributions, adds a remainder bucket
/// for tokens outside the top-N, and uses epsilon smoothing for tokens
/// present in P but missing from Q.
fn compute_position_kl(
    p_top: &[crate::client::TopLogprob],
    q_top: &[crate::client::TopLogprob],
) -> f64 {
    if p_top.is_empty() {
        return 0.0;
    }

    // Build probability distributions from logprobs
    let p_dist: HashMap<&str, f64> = p_top
        .iter()
        .map(|t| (t.token.as_str(), t.logprob.exp()))
        .collect();

    let q_dist: HashMap<&str, f64> = q_top
        .iter()
        .map(|t| (t.token.as_str(), t.logprob.exp()))
        .collect();

    // Compute remainder probabilities for tokens outside top-N
    let p_sum: f64 = p_dist.values().sum();
    let q_sum: f64 = q_dist.values().sum();
    let p_remainder = (1.0 - p_sum).max(0.0);
    let q_remainder = (1.0 - q_sum).max(0.0);

    // KL(P || Q) = Σ P(t) × ln(P(t) / Q(t))
    let mut kl = 0.0;

    for (token, &p_prob) in &p_dist {
        if p_prob <= 0.0 {
            continue;
        }
        let q_prob = q_dist.get(token).copied().unwrap_or(EPSILON);
        let q_prob = q_prob.max(EPSILON);
        kl += p_prob * (p_prob / q_prob).ln();
    }

    // Add remainder bucket contribution if P has significant remainder mass
    if p_remainder > EPSILON {
        let q_rem = q_remainder.max(EPSILON);
        kl += p_remainder * (p_remainder / q_rem).ln();
    }

    kl.max(0.0) // KL divergence is non-negative
}

/// Compute aggregate statistics from a vector of values
fn compute_aggregate_stats(values: &mut [f64]) -> AggregateStats {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    let median = percentile(values, 50.0);
    let p95 = percentile(values, 95.0);
    let p99 = percentile(values, 99.0);
    let max = values.last().copied().unwrap_or(0.0);

    AggregateStats {
        mean,
        median,
        std_dev,
        p95,
        p99,
        max,
    }
}

/// Compute percentile from a sorted slice
fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (pct / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Print a human-readable KL divergence report to console
fn print_console_report(report: &KlReport, baseline_path: &Path, candidate_path: &Path) {
    println!("KL Divergence Report");
    println!("====================");
    println!();
    println!("  Baseline:  {}", baseline_path.display());
    println!("  Candidate: {}", candidate_path.display());
    println!();
    println!("  Prompts compared: {}", report.num_prompts_compared);
    println!("  Prompts skipped:  {}", report.num_prompts_skipped);
    println!("  Token positions:  {}", report.num_positions_compared);
    println!();
    println!("  Aggregate KL Divergence (nats):");
    println!("    Mean:    {:.6}", report.aggregate.mean);
    println!("    Median:  {:.6}", report.aggregate.median);
    println!("    Std Dev: {:.6}", report.aggregate.std_dev);
    println!("    P95:     {:.6}", report.aggregate.p95);
    println!("    P99:     {:.6}", report.aggregate.p99);
    println!("    Max:     {:.6}", report.aggregate.max);
    println!();
    println!("  Interpretation:");

    let mean = report.aggregate.mean;
    if mean < 0.01 {
        println!("    KL < 0.01: Distributions are nearly identical.");
        println!("    Quantization has minimal impact on output quality.");
    } else if mean < 0.1 {
        println!("    0.01 <= KL < 0.1: Small divergence.");
        println!("    Minor differences in token probabilities; outputs likely similar.");
    } else if mean < 0.5 {
        println!("    0.1 <= KL < 0.5: Moderate divergence.");
        println!("    Noticeable differences in token distributions; may affect output quality.");
    } else {
        println!("    KL >= 0.5: Large divergence.");
        println!("    Significant differences in output distributions; quality may be degraded.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{TokenLogprob, TopLogprob};

    fn make_top_logprob(token: &str, logprob: f64) -> TopLogprob {
        TopLogprob {
            token: token.to_string(),
            logprob,
        }
    }

    #[test]
    fn test_identical_distributions() {
        let top = vec![
            make_top_logprob("hello", -0.1),
            make_top_logprob("world", -2.0),
            make_top_logprob("foo", -3.0),
        ];
        let kl = compute_position_kl(&top, &top);
        assert!(
            kl < 1e-6,
            "KL of identical distributions should be ~0, got {}",
            kl
        );
    }

    #[test]
    fn test_different_distributions() {
        let p = vec![
            make_top_logprob("hello", -0.1), // ~0.905
            make_top_logprob("world", -3.0), // ~0.050
        ];
        let q = vec![
            make_top_logprob("hello", -1.0), // ~0.368
            make_top_logprob("world", -1.5), // ~0.223
        ];
        let kl = compute_position_kl(&p, &q);
        assert!(kl > 0.0, "KL of different distributions should be positive");
    }

    #[test]
    fn test_missing_token_in_q() {
        let p = vec![
            make_top_logprob("hello", -0.5),
            make_top_logprob("world", -2.0),
        ];
        let q = vec![
            make_top_logprob("hello", -0.5),
            make_top_logprob("other", -2.0), // "world" missing
        ];
        let kl = compute_position_kl(&p, &q);
        assert!(kl > 0.0, "Missing token should increase KL");
    }

    #[test]
    fn test_empty_p() {
        let kl = compute_position_kl(&[], &[make_top_logprob("a", -1.0)]);
        assert_eq!(kl, 0.0);
    }

    #[test]
    fn test_compute_report() {
        let baseline = vec![LogprobRecord {
            prompt_index: 0,
            prompt: "test".to_string(),
            tokens: vec![TokenLogprob {
                token: "hello".to_string(),
                logprob: -0.5,
                top_logprobs: vec![
                    make_top_logprob("hello", -0.5),
                    make_top_logprob("hi", -2.0),
                ],
            }],
        }];
        let candidate = vec![LogprobRecord {
            prompt_index: 0,
            prompt: "test".to_string(),
            tokens: vec![TokenLogprob {
                token: "hello".to_string(),
                logprob: -0.5,
                top_logprobs: vec![
                    make_top_logprob("hello", -0.5),
                    make_top_logprob("hi", -2.0),
                ],
            }],
        }];

        let report = compute_kl_report(&baseline, &candidate, true).unwrap();
        assert_eq!(report.num_prompts_compared, 1);
        assert_eq!(report.num_prompts_skipped, 0);
        assert!(report.aggregate.mean < 1e-6);
    }
}
