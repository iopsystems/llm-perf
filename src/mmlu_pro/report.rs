use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

use super::config::Config;
use super::evaluate::{CategoryStats, TokenStats};

/// Canonical MMLU-Pro category order, matching the parser in llm-performance.
/// Categories not in this list are appended alphabetically at the end.
const CANONICAL_CATEGORY_ORDER: &[&str] = &[
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "philosophy",
    "physics",
    "psychology",
    "other",
];

/// Order categories to match the canonical MMLU-Pro order expected by llm-performance.
fn order_categories(stats: &HashMap<String, CategoryStats>) -> Vec<String> {
    let mut ordered = Vec::new();

    // Add categories in canonical order if they exist in stats
    for &cat in CANONICAL_CATEGORY_ORDER {
        if stats.contains_key(cat) {
            ordered.push(cat.to_string());
        }
    }

    // Append any extra categories not in the canonical list, sorted alphabetically
    let mut extra: Vec<String> = stats
        .keys()
        .filter(|k| !CANONICAL_CATEGORY_ORDER.contains(&k.as_str()))
        .cloned()
        .collect();
    extra.sort();
    ordered.extend(extra);

    ordered
}

/// Generate and save the final report.
pub fn generate_report(
    config: &Config,
    stats: &HashMap<String, CategoryStats>,
    token_stats: &TokenStats,
    elapsed: Duration,
    output_dir: &Path,
) {
    let report_path = output_dir.join("report.txt");
    let mut lines: Vec<String> = Vec::new();

    // Timestamp
    lines.push(format!("{}", chrono::Local::now()));

    // Config (without api_key)
    lines.push(format!("Model: {}", config.server.model));
    lines.push(format!("URL: {}", config.server.url));
    lines.push(format!("Temperature: {}", config.inference.temperature));
    lines.push(format!("Top P: {}", config.inference.top_p));
    lines.push(format!("Max Tokens: {}", config.inference.max_tokens));
    lines.push(format!("Parallel: {}", config.test.parallel));
    lines.push(format!("Subset: {}", config.test.subset));
    if !config.comment.is_empty() {
        lines.push(format!("Comment: {}", config.comment));
    }
    lines.push(String::new());

    // Per-category results
    let mut total_corr = 0u32;
    let mut total_wrong = 0u32;
    let mut total_extraction_failures = 0u32;
    let categories = order_categories(stats);

    for category in &categories {
        if let Some(s) = stats.get(category) {
            let total = s.correct + s.wrong;
            let acc = if total > 0 {
                s.correct as f64 / total as f64 * 100.0
            } else {
                0.0
            };
            lines.push(format!(
                "{}: {}/{} ({:.2}%), {} extraction failures",
                category, s.correct, total, acc, s.extraction_failures
            ));
            total_corr += s.correct;
            total_wrong += s.wrong;
            total_extraction_failures += s.extraction_failures;
        }
    }

    lines.push(String::new());

    // Total
    let total = total_corr + total_wrong;
    let acc = if total > 0 {
        total_corr as f64 / total as f64 * 100.0
    } else {
        0.0
    };
    lines.push(format!("Total: {}/{}, {:.2}%", total_corr, total, acc));
    lines.push(format!(
        "Extraction failures: {} (counted as wrong)",
        total_extraction_failures
    ));
    lines.push(String::new());

    // Markdown table
    let mut header_names = vec!["overall".to_string()];
    header_names.extend(categories.iter().cloned());
    let header = format!("| {} |", header_names.join(" | "));
    let separator = format!(
        "| {} |",
        header_names
            .iter()
            .map(|name| "-".repeat(name.len()))
            .collect::<Vec<_>>()
            .join(" | ")
    );

    let mut scores = Vec::new();
    // Overall score first
    scores.push(format!("{:.2}", acc));
    for category in &categories {
        if let Some(s) = stats.get(category) {
            let cat_total = s.correct + s.wrong;
            let cat_acc = if cat_total > 0 {
                s.correct as f64 / cat_total as f64 * 100.0
            } else {
                0.0
            };
            scores.push(format!("{:.2}", cat_acc));
        }
    }
    let score_row = format!("| {} |", scores.join(" | "));

    lines.push("Markdown Table:".to_string());
    lines.push(header);
    lines.push(separator);
    lines.push(score_row);
    lines.push(String::new());

    // Token usage
    if !token_stats.prompt_tokens.is_empty() {
        let duration_secs = elapsed.as_secs_f64();

        let ptoks: Vec<f64> = token_stats
            .prompt_tokens
            .iter()
            .map(|&t| t as f64)
            .collect();
        let ctoks: Vec<f64> = token_stats
            .completion_tokens
            .iter()
            .map(|&t| t as f64)
            .collect();

        let p_min = ptoks.iter().cloned().fold(f64::INFINITY, f64::min);
        let p_max = ptoks.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let p_sum: f64 = ptoks.iter().sum();
        let p_avg = p_sum / ptoks.len() as f64;

        let c_min = ctoks.iter().cloned().fold(f64::INFINITY, f64::min);
        let c_max = ctoks.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let c_sum: f64 = ctoks.iter().sum();
        let c_avg = c_sum / ctoks.len() as f64;

        lines.push("Token Usage:".to_string());
        lines.push(format!(
            "Prompt tokens: min {:.0}, average {:.0}, max {:.0}, total {:.0}, tk/s {:.2}",
            p_min,
            p_avg,
            p_max,
            p_sum,
            p_sum / duration_secs
        ));
        lines.push(format!(
            "Completion tokens: min {:.0}, average {:.0}, max {:.0}, total {:.0}, tk/s {:.2}",
            c_min,
            c_avg,
            c_max,
            c_sum,
            c_sum / duration_secs
        ));
        lines.push(String::new());
    }

    // Elapsed time
    lines.push(format!(
        "Finished the benchmark in {}.",
        format_duration(elapsed)
    ));

    // Print to stdout
    for line in &lines {
        println!("{}", line);
    }

    // Write to file
    if let Ok(mut file) = std::fs::File::create(&report_path) {
        for line in &lines {
            let _ = writeln!(file, "{}", line);
        }
        eprintln!("Report saved to: {}", report_path.display());
    }
}

fn format_duration(d: Duration) -> String {
    let total_secs = d.as_secs();
    let days = total_secs / 86400;
    let hours = (total_secs % 86400) / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;

    let mut parts = Vec::new();
    if days > 0 {
        parts.push(format!("{} days", days));
    }
    if hours > 0 {
        parts.push(format!("{} hours", hours));
    }
    if minutes > 0 {
        parts.push(format!("{} minutes", minutes));
    }
    if seconds > 0 || parts.is_empty() {
        parts.push(format!("{} seconds", seconds));
    }
    parts.join(" ")
}
