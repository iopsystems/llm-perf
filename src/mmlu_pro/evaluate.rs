use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;
use tokio::sync::Semaphore;

use crate::client::{
    ChatCompletionRequest, ClientConfig, ClientError, CompletionRequest, OpenAIClient, Usage,
};

use super::config::{Config, PromptMode};
use super::dataset::Question;
use super::extract::extract_answer;
use super::fit::{ShotFit, fit_shots};
use super::prompt::{build_completion_prompt, build_messages};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionResult {
    pub question_id: i64,
    pub question: String,
    pub category: String,
    pub options: Vec<String>,
    pub answer: String,
    pub answer_index: i64,
    pub response: String,
    pub pred: Option<String>,
    /// The exact prompt sent, when `log_prompt` is set. In chat mode this is
    /// one entry per chat message; in completion mode it is a single entry with
    /// role `"prompt"` holding the whole prompt string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<Vec<PromptMessage>>,
    /// Number of few-shot examples in the prompt that was sent. `None` for a
    /// skipped question and in result files written before this field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shots_used: Option<usize>,
    /// The prompt did not fit `max_context_tokens` even with 0 shots, so no
    /// request was sent. Counted in the accuracy denominator.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub skipped: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Default)]
pub struct CategoryStats {
    pub correct: u32,
    pub wrong: u32,
    pub extraction_failures: u32,
    pub errors: u32,
    /// Questions not sent because they did not fit `max_context_tokens`.
    pub skipped: u32,
    /// Number of questions that received a response, by shot count.
    pub shots_used: BTreeMap<usize, u32>,
}

impl CategoryStats {
    /// Accuracy denominator: every question attempted, including request
    /// errors and questions skipped for length.
    pub fn total(&self) -> u32 {
        self.correct + self.wrong + self.errors + self.skipped
    }

    /// Count a saved, answered result (used when resuming from a result
    /// file). Skipped results are dropped before this and re-evaluated.
    fn record_saved(&mut self, r: &QuestionResult) {
        match &r.pred {
            Some(pred) if pred == &r.answer => self.correct += 1,
            Some(_) => self.wrong += 1,
            None => {
                self.wrong += 1;
                self.extraction_failures += 1;
            }
        }
        if let Some(k) = r.shots_used {
            *self.shots_used.entry(k).or_default() += 1;
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TokenStats {
    pub prompt_tokens: Vec<u32>,
    pub completion_tokens: Vec<u32>,
}

pub struct EvaluationResult {
    pub category_stats: HashMap<String, CategoryStats>,
    pub token_stats: TokenStats,
}

/// Shared atomic counters for lock-free progress reporting.
struct ProgressCounters {
    completed: AtomicU32,
    correct: AtomicU32,
    wrong: AtomicU32,
    extraction_failures: AtomicU32,
    errors: AtomicU32,
    skipped: AtomicU32,
    prompt_tokens: AtomicU32,
    completion_tokens: AtomicU32,
    total: u32,
}

/// Load existing results from a category result file for resume support.
fn load_existing_results(path: &Path) -> Vec<QuestionResult> {
    if !path.exists() {
        return Vec::new();
    }
    match std::fs::read_to_string(path) {
        Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

/// Atomically write `bytes` to `path` by writing a sibling temp file and
/// renaming it into place. A crash or error mid-write leaves the existing
/// file untouched, rather than truncating it (which would make resume discard
/// all prior results for the category).
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Save results to a category result file.
fn save_results(results: &[QuestionResult], path: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(results)?;
    write_atomic(path, json.as_bytes())
}

/// Save category summary to a JSON file.
fn save_summary(stats: &HashMap<String, CategoryStats>, path: &Path) -> Result<()> {
    let mut summary: HashMap<String, serde_json::Value> = HashMap::new();
    let mut total_corr = 0u32;
    let mut total_wrong = 0u32;
    let mut total_errors = 0u32;
    let mut total_skipped = 0u32;

    for (category, s) in stats {
        let total = s.total();
        let acc = if total > 0 {
            s.correct as f64 / total as f64
        } else {
            0.0
        };
        summary.insert(
            category.clone(),
            serde_json::json!({
                "corr": s.correct,
                "wrong": s.wrong,
                "errors": s.errors,
                "skipped": s.skipped,
                "extraction_failures": s.extraction_failures,
                "acc": acc,
            }),
        );
        total_corr += s.correct;
        total_wrong += s.wrong;
        total_errors += s.errors;
        total_skipped += s.skipped;
    }

    let total = total_corr + total_wrong + total_errors + total_skipped;
    let acc = if total > 0 {
        total_corr as f64 / total as f64
    } else {
        0.0
    };
    summary.insert(
        "total".to_string(),
        serde_json::json!({
            "corr": total_corr,
            "wrong": total_wrong,
            "errors": total_errors,
            "skipped": total_skipped,
            "acc": acc,
        }),
    );

    let json = serde_json::to_string_pretty(&summary)?;
    write_atomic(path, json.as_bytes())
}

fn format_eta(secs: u64) -> String {
    if secs >= 3600 {
        format!("{}h{}m{}s", secs / 3600, (secs % 3600) / 60, secs % 60)
    } else {
        format!("{}m{}s", secs / 60, secs % 60)
    }
}

fn print_status(
    category: &str,
    counters: &ProgressCounters,
    start: Instant,
    overall: &ProgressCounters,
    overall_start: Instant,
) {
    let completed = counters.completed.load(Ordering::Relaxed);
    let correct = counters.correct.load(Ordering::Relaxed);
    let wrong = counters.wrong.load(Ordering::Relaxed);
    let failures = counters.extraction_failures.load(Ordering::Relaxed);
    let errors = counters.errors.load(Ordering::Relaxed);
    let skipped = counters.skipped.load(Ordering::Relaxed);
    let prompt_tokens = overall.prompt_tokens.load(Ordering::Relaxed);
    let completion_tokens = overall.completion_tokens.load(Ordering::Relaxed);
    let total = correct + wrong + errors + skipped;
    let acc = if total > 0 {
        correct as f64 / total as f64 * 100.0
    } else {
        0.0
    };
    let elapsed_secs = start.elapsed().as_secs_f64();
    let elapsed = elapsed_secs as u64;

    let mut msg = format!(
        "  {}: {}/{} completed, {}/{} correct ({:.2}%)",
        category, completed, counters.total, correct, total, acc
    );
    if failures > 0 {
        msg.push_str(&format!(", {} failed extractions", failures));
    }
    if errors > 0 {
        msg.push_str(&format!(", {} errors", errors));
    }
    if skipped > 0 {
        msg.push_str(&format!(", {} skipped (too long)", skipped));
    }

    // Token throughput (use overall elapsed for accurate rates)
    let overall_elapsed = overall_start.elapsed().as_secs_f64();
    if overall_elapsed > 0.0 && completion_tokens > 0 {
        let prompt_tps = prompt_tokens as f64 / overall_elapsed;
        let completion_tps = completion_tokens as f64 / overall_elapsed;
        msg.push_str(&format!(
            ", {:.0} prompt tk/s, {:.0} completion tk/s",
            prompt_tps, completion_tps
        ));
    }

    // Overall ETA
    let overall_completed = overall.completed.load(Ordering::Relaxed);
    if overall_completed > 0 && overall_completed < overall.total {
        let overall_remaining = overall.total - overall_completed;
        let secs_per_item = overall_elapsed / overall_completed as f64;
        let eta_secs = (overall_remaining as f64 * secs_per_item) as u64;
        msg.push_str(&format!(", ETA {}", format_eta(eta_secs)));
    }

    msg.push_str(&format!(", {} elapsed", format_eta(elapsed)));
    eprintln!("{}", msg);
}

/// Append `result` to the category's result file and rewrite its summary.
async fn persist(
    results: &tokio::sync::Mutex<Vec<QuestionResult>>,
    stats: &tokio::sync::Mutex<CategoryStats>,
    result: QuestionResult,
    category: &str,
    result_path: &Path,
    summary_path: &Path,
) {
    {
        let mut res = results.lock().await;
        res.push(result);

        // Deduplicate by question_id
        let mut seen = std::collections::HashSet::new();
        res.retain(|r| seen.insert(r.question_id));

        let _ = save_results(&res, result_path);
    }
    let s = stats.lock().await;
    let mut summary_stats: HashMap<String, CategoryStats> = HashMap::new();
    summary_stats.insert(category.to_string(), s.clone());
    let _ = save_summary(&summary_stats, summary_path);
}

/// The value sent as `chat_template_kwargs` on chat requests. `/apply-template`
/// gets the same value so the prompt it renders for counting is the prompt the
/// generation request produces.
const CHAT_TEMPLATE_KWARGS: Option<&serde_json::Value> = None;

/// Exact prompt length in tokens, as the server will see it, for the prompt
/// with the given `shots`.
///
/// Completion mode counts the prompt string. Chat mode first renders the
/// messages with the server's chat template (`/apply-template`) and counts
/// that. Both counts include BOS (see `OpenAIClient::count_prompt_tokens`).
async fn count_prompt_tokens(
    client: &OpenAIClient,
    mode: PromptMode,
    system_prompt: &str,
    shots: &[Question],
    question: &Question,
) -> Result<usize> {
    let text = match mode {
        PromptMode::Chat => {
            let messages =
                build_messages(system_prompt, shots, &question.question, &question.options);
            client
                .apply_template(&messages, CHAT_TEMPLATE_KWARGS)
                .await?
        }
        PromptMode::Completion => {
            build_completion_prompt(system_prompt, shots, &question.question, &question.options)
        }
    };
    client.count_prompt_tokens(&text).await
}

/// Fail early when `max_context_tokens` is set but cannot be applied: the
/// generation budget alone fills the window, llama-server's `POST /tokenize`
/// does not answer with tokens, or (chat mode) its `POST /apply-template`
/// does not answer with a prompt.
async fn check_token_counting_available(
    client: &OpenAIClient,
    mode: PromptMode,
    max_context_tokens: u32,
    max_tokens: u32,
) -> Result<()> {
    if max_tokens >= max_context_tokens {
        anyhow::bail!(
            "max_tokens ({max_tokens}) must be less than max_context_tokens \
             ({max_context_tokens}); otherwise no prompt fits"
        );
    }
    match client.count_prompt_tokens("The answer is (A).").await {
        Ok(n) if n > 0 => {}
        Ok(_) => anyhow::bail!(
            "max_context_tokens is set, but POST /tokenize returned no tokens for a \
             non-empty string. Counting needs llama-server's /tokenize \
             ({{\"content\": ..., \"add_special\": true}} -> {{\"tokens\": [...]}})."
        ),
        Err(e) => anyhow::bail!(
            "max_context_tokens is set, which counts prompt tokens with llama-server's \
             POST /tokenize at the server root (not under /v1), but that request \
             failed: {e}. Unset max_context_tokens for servers other than llama-server."
        ),
    }
    if mode == PromptMode::Chat {
        let probe = [crate::client::Message {
            role: "user".to_string(),
            content: "The answer is (A).".to_string(),
        }];
        if let Err(e) = client.apply_template(&probe, CHAT_TEMPLATE_KWARGS).await {
            anyhow::bail!(
                "max_context_tokens is set in chat mode, which renders the chat template \
                 with llama-server's POST /apply-template at the server root to count \
                 prompt tokens, but that request failed: {e}. Unset max_context_tokens \
                 or use a server that provides /apply-template."
            );
        }
    }
    Ok(())
}

/// Settings that change what a question's result means. Saved as
/// `run_config.json` in the output directory; resuming into a directory whose
/// saved settings differ is refused, because the old and new results would be
/// mixed in one score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct RunConfig {
    model: String,
    mode: String,
    num_shots: usize,
    max_context_tokens: Option<u32>,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    system_prompt: String,
}

impl RunConfig {
    fn from_config(config: &Config, model: &str) -> Self {
        let inf = &config.inference;
        Self {
            model: model.to_string(),
            mode: inf.mode.as_str().to_string(),
            num_shots: inf.num_shots,
            max_context_tokens: inf.max_context_tokens,
            max_tokens: inf.max_tokens,
            temperature: inf.temperature,
            top_p: inf.top_p,
            frequency_penalty: inf.frequency_penalty,
            presence_penalty: inf.presence_penalty,
            system_prompt: inf.system_prompt.clone(),
        }
    }

    /// Every field except `system_prompt`, as (name, displayed value).
    fn displayed_fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("model", self.model.clone()),
            ("mode", self.mode.clone()),
            ("num_shots", self.num_shots.to_string()),
            (
                "max_context_tokens",
                self.max_context_tokens
                    .map_or_else(|| "unset".to_string(), |c| c.to_string()),
            ),
            ("max_tokens", self.max_tokens.to_string()),
            ("temperature", self.temperature.to_string()),
            ("top_p", self.top_p.to_string()),
            ("frequency_penalty", self.frequency_penalty.to_string()),
            ("presence_penalty", self.presence_penalty.to_string()),
        ]
    }

    /// One line per field that differs between `self` (saved) and `current`.
    fn differing_fields(&self, current: &RunConfig) -> Vec<String> {
        let mut out: Vec<String> = self
            .displayed_fields()
            .into_iter()
            .zip(current.displayed_fields())
            .filter(|((_, saved), (_, now))| saved != now)
            .map(|((name, saved), (_, now))| {
                if name == "model" {
                    format!(
                        "model: saved {saved}, now {now} (pass --model {saved} to resume \
                         against the saved model name)"
                    )
                } else {
                    format!("{name}: saved {saved}, now {now}")
                }
            })
            .collect();
        if self.system_prompt != current.system_prompt {
            out.push("system_prompt: text differs".to_string());
        }
        out
    }
}

const RUN_CONFIG_FILE: &str = "run_config.json";

/// Check `output_dir/run_config.json` against this run's settings, then write
/// it.
///
/// - Saved results and a `run_config.json` that differs: refuse to resume.
/// - Saved results and no `run_config.json` (written by an older llm-perf):
///   they cannot be checked; warn and proceed.
/// - No saved results: write this run's settings, replacing any
///   `run_config.json` left by a run that failed before saving a result.
fn check_and_write_run_config(config: &Config, model: &str, output_dir: &Path) -> Result<()> {
    let path = output_dir.join(RUN_CONFIG_FILE);
    let current = RunConfig::from_config(config, model);
    // Skipped rows do not count: resume evaluates them again, so they do not
    // tie the directory to the settings that skipped them.
    let has_results = std::fs::read_dir(output_dir)
        .map(|entries| {
            entries.flatten().any(|e| {
                e.file_name().to_string_lossy().ends_with("_result.json")
                    && load_existing_results(&e.path()).iter().any(|r| !r.skipped)
            })
        })
        .unwrap_or(false);
    if has_results && path.exists() {
        let saved: RunConfig =
            serde_json::from_str(&std::fs::read_to_string(&path)?).map_err(|e| {
                anyhow::anyhow!(
                    "failed to parse {}: {e}. Use a fresh output directory (move or \
                     delete {}).",
                    path.display(),
                    output_dir.display()
                )
            })?;
        let diffs = saved.differing_fields(&current);
        if !diffs.is_empty() {
            anyhow::bail!(
                "{} holds saved results from a run with different settings, and \
                 resuming would mix them into one score:\n  {}\nUse a fresh output \
                 directory (move or delete {}).",
                output_dir.display(),
                diffs.join("\n  "),
                output_dir.display()
            );
        }
        return Ok(());
    }
    if has_results {
        eprintln!(
            "Warning: {} has results but no {RUN_CONFIG_FILE} (written by an older \
             llm-perf), so they cannot be checked against this run's settings. \
             Resuming anyway.",
            output_dir.display()
        );
    }
    write_atomic(&path, serde_json::to_string_pretty(&current)?.as_bytes())
}

/// Run evaluation across all specified categories.
pub async fn run_evaluation(
    config: &Config,
    model: &str,
    test_data: &HashMap<String, Vec<Question>>,
    val_data: &HashMap<String, Vec<Question>>,
    output_dir: &Path,
) -> Result<EvaluationResult> {
    let client_config = ClientConfig {
        base_url: config.endpoint.base_url.clone(),
        api_key: config.endpoint.api_key.clone(),
        model: model.to_string(),
        timeout: std::time::Duration::from_secs(config.endpoint.timeout),
        max_retries: 3,
        retry_initial_delay_ms: 1000,
        retry_max_delay_ms: 30000,
        pool_size: config.load.concurrent_requests,
        // Non-streaming offline eval: no streaming idle timeout, and retrying
        // timeouts is desirable here (no coordinated-omission concern), matching
        // the prior retry-all-transient behavior.
        stream_idle_timeout: None,
        retry_on_timeout: true,
        chat_template_kwargs: None,
        extra_headers: None,
        ignore_eos: None,
        pool_idle_timeout: crate::client::DEFAULT_POOL_IDLE_TIMEOUT,
    };

    let client = Arc::new(OpenAIClient::new(client_config)?);
    let semaphore = Arc::new(Semaphore::new(config.load.concurrent_requests));

    let mut all_stats: HashMap<String, CategoryStats> = HashMap::new();
    let mut all_token_stats = TokenStats::default();

    // Determine which categories to evaluate
    let categories: Vec<String> = if config.load.categories.contains(&"all".to_string()) {
        let mut cats: Vec<String> = test_data.keys().cloned().collect();
        cats.sort();
        cats
    } else {
        config.load.categories.clone()
    };

    let system_prompt_template = &config.inference.system_prompt;
    let max_context_tokens = config.inference.max_context_tokens;

    if let Some(ctx) = max_context_tokens {
        check_token_counting_available(
            &client,
            config.inference.mode,
            ctx,
            config.inference.max_tokens,
        )
        .await?;
    }
    check_and_write_run_config(config, model, output_dir)?;

    // The first skip of a run is always printed; later ones at verbosity >= 1.
    let skip_reported = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Compute overall question count for ETA across all categories
    let overall_total: u32 = categories
        .iter()
        .filter_map(|cat| test_data.get(cat))
        .map(|qs| qs.len() as u32)
        .sum();

    let overall_start = Instant::now();
    let overall_counters = Arc::new(ProgressCounters {
        completed: AtomicU32::new(0),
        correct: AtomicU32::new(0),
        wrong: AtomicU32::new(0),
        extraction_failures: AtomicU32::new(0),
        errors: AtomicU32::new(0),
        skipped: AtomicU32::new(0),
        prompt_tokens: AtomicU32::new(0),
        completion_tokens: AtomicU32::new(0),
        total: overall_total,
    });

    for category in &categories {
        let test_questions = match test_data.get(category) {
            Some(q) => q,
            None => {
                eprintln!(
                    "Warning: category '{}' not found in test data, skipping.",
                    category
                );
                continue;
            }
        };

        let cot_examples: Vec<Question> = val_data
            .get(category)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .take(config.inference.num_shots)
            .collect();

        let system_prompt = system_prompt_template.replace("{subject}", category);

        let result_path = output_dir.join(format!("{}_result.json", category));
        let summary_path = output_dir.join(format!("{}_summary.json", category));

        // Load existing results for resume
        // Skipped results are dropped so those questions are evaluated again.
        let existing_results: Vec<QuestionResult> = load_existing_results(&result_path)
            .into_iter()
            .filter(|r| !r.skipped)
            .collect();
        let existing_ids: std::collections::HashSet<i64> =
            existing_results.iter().map(|r| r.question_id).collect();

        // Count stats from existing results
        let mut cat_stats = CategoryStats::default();
        for r in &existing_results {
            cat_stats.record_saved(r);
        }

        // Filter to only new questions
        let new_questions: Vec<&Question> = test_questions
            .iter()
            .filter(|q| !existing_ids.contains(&q.question_id))
            .collect();

        let total = test_questions.len();
        let already_done = existing_ids.len();

        // Account for already-completed questions in overall counters
        overall_counters
            .completed
            .fetch_add(already_done as u32, Ordering::Relaxed);
        overall_counters
            .correct
            .fetch_add(cat_stats.correct, Ordering::Relaxed);
        overall_counters
            .wrong
            .fetch_add(cat_stats.wrong, Ordering::Relaxed);
        overall_counters
            .extraction_failures
            .fetch_add(cat_stats.extraction_failures, Ordering::Relaxed);

        if new_questions.is_empty() {
            eprintln!(
                "{}: all {}/{} questions already completed, skipping.",
                category, already_done, total
            );
            all_stats.insert(category.clone(), cat_stats);
            continue;
        }

        eprintln!(
            "{}: {}/{} already done, {} remaining.",
            category,
            already_done,
            total,
            new_questions.len()
        );

        let cat_start = Instant::now();

        // Atomic counters for lock-free progress reporting
        let counters = Arc::new(ProgressCounters {
            completed: AtomicU32::new(0),
            correct: AtomicU32::new(cat_stats.correct),
            wrong: AtomicU32::new(cat_stats.wrong),
            extraction_failures: AtomicU32::new(cat_stats.extraction_failures),
            errors: AtomicU32::new(cat_stats.errors),
            skipped: AtomicU32::new(cat_stats.skipped),
            prompt_tokens: AtomicU32::new(0),
            completion_tokens: AtomicU32::new(0),
            total: total as u32,
        });

        // Spawn periodic status printer (every 60s)
        let status_counters = Arc::clone(&counters);
        let status_overall = Arc::clone(&overall_counters);
        let status_category = category.clone();
        let status_handle = tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                print_status(
                    &status_category,
                    &status_counters,
                    cat_start,
                    &status_overall,
                    overall_start,
                );
            }
        });

        // Shared state for collecting results
        let results = Arc::new(tokio::sync::Mutex::new(existing_results));
        let stats = Arc::new(tokio::sync::Mutex::new(cat_stats));
        let token_stats = Arc::new(tokio::sync::Mutex::new(TokenStats::default()));

        let mut handles = Vec::new();

        for question in new_questions {
            let client: Arc<OpenAIClient> = Arc::clone(&client);
            let semaphore = Arc::clone(&semaphore);
            let results = Arc::clone(&results);
            let stats = Arc::clone(&stats);
            let token_stats = Arc::clone(&token_stats);
            let counters = Arc::clone(&counters);
            let overall_counters = Arc::clone(&overall_counters);
            let result_path = result_path.clone();
            let summary_path = summary_path.clone();
            let system_prompt = system_prompt.clone();
            let cot_examples = cot_examples.clone();
            let question = question.clone();
            let log_prompt = config.log.log_prompt;
            let temperature = config.inference.temperature;
            let top_p = config.inference.top_p;
            let max_tokens = config.inference.max_tokens;
            let frequency_penalty = config.inference.frequency_penalty;
            let presence_penalty = config.inference.presence_penalty;
            let model = model.to_string();
            let verbosity = config.log.verbosity;
            let mode = config.inference.mode;
            let category = category.clone();
            let skip_reported = Arc::clone(&skip_reported);

            let handle = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();

                // Choose how many shots to send. Without max_context_tokens this
                // is always every available shot (num_shots, or fewer if the
                // validation split has fewer for this category).
                let shots = match max_context_tokens {
                    None => cot_examples.len(),
                    Some(ctx) => {
                        let fit = fit_shots(cot_examples.len(), max_tokens, ctx, |k| {
                            count_prompt_tokens(
                                &client,
                                mode,
                                &system_prompt,
                                &cot_examples[..k],
                                &question,
                            )
                        })
                        .await;
                        match fit {
                            Ok(ShotFit::Fits {
                                shots,
                                prompt_tokens,
                            }) => {
                                if verbosity >= 2 {
                                    eprintln!(
                                        "Q{}: {} shots, {} prompt tokens",
                                        question.question_id, shots, prompt_tokens
                                    );
                                }
                                shots
                            }
                            Ok(ShotFit::TooLong { zero_shot_tokens }) => {
                                let first = !skip_reported.swap(true, Ordering::Relaxed);
                                if first || verbosity >= 1 {
                                    eprintln!(
                                        "Skipping question {}: {} prompt tokens at 0 shots + \
                                         {} max_tokens exceeds max_context_tokens {}{}",
                                        question.question_id,
                                        zero_shot_tokens,
                                        max_tokens,
                                        ctx,
                                        if first && verbosity == 0 {
                                            " (further skips are counted in the report; \
                                             -v 1 prints each)"
                                        } else {
                                            ""
                                        }
                                    );
                                }
                                stats.lock().await.skipped += 1;
                                counters.skipped.fetch_add(1, Ordering::Relaxed);
                                overall_counters.skipped.fetch_add(1, Ordering::Relaxed);
                                let result = QuestionResult {
                                    question_id: question.question_id,
                                    question: question.question.clone(),
                                    category: question.category.clone(),
                                    options: question.options.clone(),
                                    answer: question.answer.clone(),
                                    answer_index: question.answer_index,
                                    response: String::new(),
                                    pred: None,
                                    prompt: None,
                                    shots_used: None,
                                    skipped: true,
                                };
                                persist(
                                    &results,
                                    &stats,
                                    result,
                                    &category,
                                    &result_path,
                                    &summary_path,
                                )
                                .await;
                                counters.completed.fetch_add(1, Ordering::Relaxed);
                                overall_counters.completed.fetch_add(1, Ordering::Relaxed);
                                return;
                            }
                            Err(e) => {
                                eprintln!(
                                    "Error for question {} (token count): {}",
                                    question.question_id, e
                                );
                                stats.lock().await.errors += 1;
                                counters.errors.fetch_add(1, Ordering::Relaxed);
                                counters.completed.fetch_add(1, Ordering::Relaxed);
                                overall_counters.errors.fetch_add(1, Ordering::Relaxed);
                                overall_counters.completed.fetch_add(1, Ordering::Relaxed);
                                return;
                            }
                        }
                    }
                };
                let cot_examples = &cot_examples[..shots];

                let stop = Some(vec!["Question:".to_string()]);

                // Send the request in the configured mode. Both arms yield the
                // generated text, the server's token usage, and the prompt as it
                // would be logged.
                let (outcome, prompt_messages) = match mode {
                    PromptMode::Chat => {
                        let messages = build_messages(
                            &system_prompt,
                            cot_examples,
                            &question.question,
                            &question.options,
                        );
                        let request = ChatCompletionRequest {
                            model,
                            messages: messages.clone(),
                            max_tokens: Some(max_tokens),
                            temperature: Some(temperature),
                            top_p: Some(top_p),
                            frequency_penalty: Some(frequency_penalty),
                            presence_penalty: Some(presence_penalty),
                            stop,
                            stream: Some(false),
                            stream_options: None,
                            logprobs: None,
                            top_logprobs: None,
                            chat_template_kwargs: CHAT_TEMPLATE_KWARGS.cloned(),
                            ignore_eos: None,
                        };
                        let outcome = client.chat_completion(request).await.map(|r| {
                            let text = r
                                .choices
                                .first()
                                .map(|c| c.message.content.clone())
                                .unwrap_or_default();
                            (text, r.usage)
                        });
                        let logged = messages
                            .into_iter()
                            .map(|m| PromptMessage {
                                role: m.role,
                                content: m.content,
                            })
                            .collect::<Vec<_>>();
                        (outcome, logged)
                    }
                    PromptMode::Completion => {
                        let prompt = build_completion_prompt(
                            &system_prompt,
                            cot_examples,
                            &question.question,
                            &question.options,
                        );
                        let request = CompletionRequest {
                            model,
                            prompt: prompt.clone(),
                            max_tokens: Some(max_tokens),
                            temperature: Some(temperature),
                            top_p: Some(top_p),
                            frequency_penalty: Some(frequency_penalty),
                            presence_penalty: Some(presence_penalty),
                            stop,
                            stream: Some(false),
                        };
                        let outcome = client.completion(request).await.map(|r| {
                            let text = r
                                .choices
                                .first()
                                .map(|c| c.text.clone())
                                .unwrap_or_default();
                            (text, r.usage)
                        });
                        let logged = vec![PromptMessage {
                            role: "prompt".to_string(),
                            content: prompt,
                        }];
                        (outcome, logged)
                    }
                };

                let (response_text, usage): (String, Usage) = match outcome {
                    Ok(out) => out,
                    Err(e) => {
                        let error_kind = match e.downcast_ref::<ClientError>() {
                            Some(ClientError::Connection(_)) => "connection",
                            Some(ClientError::Timeout(_)) => "timeout",
                            Some(ClientError::Http4xx { status, .. }) => {
                                // Leak a short label; there are only a few distinct status codes
                                Box::leak(format!("http {status}").into_boxed_str())
                            }
                            Some(ClientError::Http5xx { status, .. }) => {
                                Box::leak(format!("http {status}").into_boxed_str())
                            }
                            Some(ClientError::Parse(_)) => "parse",
                            Some(ClientError::StreamError { .. }) => "stream",
                            Some(ClientError::Other(_)) | None => "unknown",
                        };
                        eprintln!(
                            "Error for question {} ({}): {}",
                            question.question_id, error_kind, e
                        );
                        {
                            let mut s = stats.lock().await;
                            s.errors += 1;
                        }
                        counters.errors.fetch_add(1, Ordering::Relaxed);
                        counters.completed.fetch_add(1, Ordering::Relaxed);
                        overall_counters.errors.fetch_add(1, Ordering::Relaxed);
                        overall_counters.completed.fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                };

                // Track token usage
                {
                    let mut ts = token_stats.lock().await;
                    ts.prompt_tokens.push(usage.prompt_tokens);
                    ts.completion_tokens.push(usage.completion_tokens);
                }
                overall_counters
                    .prompt_tokens
                    .fetch_add(usage.prompt_tokens, Ordering::Relaxed);
                overall_counters
                    .completion_tokens
                    .fetch_add(usage.completion_tokens, Ordering::Relaxed);

                let response_text = response_text.trim().to_string();

                let pred = extract_answer(&response_text);
                let pred_str = pred.map(|c| c.to_string());

                if verbosity >= 2 {
                    eprintln!(
                        "Q{}: pred={:?} answer={} | {}",
                        question.question_id,
                        pred_str,
                        question.answer,
                        &response_text[..response_text.len().min(100)]
                    );
                }

                let prompt_log = log_prompt.then_some(prompt_messages);

                let result = QuestionResult {
                    question_id: question.question_id,
                    question: question.question.clone(),
                    category: question.category.clone(),
                    options: question.options.clone(),
                    answer: question.answer.clone(),
                    answer_index: question.answer_index,
                    response: response_text,
                    pred: pred_str.clone(),
                    prompt: prompt_log,
                    shots_used: Some(shots),
                    skipped: false,
                };

                // Update stats
                {
                    let mut s = stats.lock().await;
                    *s.shots_used.entry(shots).or_default() += 1;
                    match &pred_str {
                        Some(p) if p == &question.answer => {
                            s.correct += 1;
                            counters.correct.fetch_add(1, Ordering::Relaxed);
                            overall_counters.correct.fetch_add(1, Ordering::Relaxed);
                        }
                        Some(_) => {
                            s.wrong += 1;
                            counters.wrong.fetch_add(1, Ordering::Relaxed);
                            overall_counters.wrong.fetch_add(1, Ordering::Relaxed);
                        }
                        None => {
                            s.wrong += 1;
                            s.extraction_failures += 1;
                            counters.wrong.fetch_add(1, Ordering::Relaxed);
                            counters.extraction_failures.fetch_add(1, Ordering::Relaxed);
                            overall_counters.wrong.fetch_add(1, Ordering::Relaxed);
                            overall_counters
                                .extraction_failures
                                .fetch_add(1, Ordering::Relaxed);
                            if verbosity >= 2 {
                                // Show the tail of the response where the answer should be
                                let tail = if result.response.len() > 300 {
                                    format!(
                                        "...{}",
                                        &result.response[result.response.len() - 300..]
                                    )
                                } else {
                                    result.response.clone()
                                };
                                eprintln!(
                                    "Extraction failed for Q{}: «{}»",
                                    question.question_id, tail
                                );
                            }
                        }
                    }
                }

                persist(
                    &results,
                    &stats,
                    result,
                    &category,
                    &result_path,
                    &summary_path,
                )
                .await;

                counters.completed.fetch_add(1, Ordering::Relaxed);
                overall_counters.completed.fetch_add(1, Ordering::Relaxed);
            });

            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            let _ = handle.await;
        }

        // Stop the status printer
        status_handle.abort();

        // Print final status for this category
        print_status(
            category,
            &counters,
            cat_start,
            &overall_counters,
            overall_start,
        );

        // Collect final stats
        let final_stats = stats.lock().await.clone();
        let final_token_stats = token_stats.lock().await.clone();

        // Final save
        {
            let final_results = results.lock().await;
            save_results(&final_results, &result_path)?;
        }

        // Save final summary
        {
            let mut summary_stats: HashMap<String, CategoryStats> = HashMap::new();
            summary_stats.insert(category.clone(), final_stats.clone());
            save_summary(&summary_stats, &summary_path)?;
        }

        all_token_stats
            .prompt_tokens
            .extend(&final_token_stats.prompt_tokens);
        all_token_stats
            .completion_tokens
            .extend(&final_token_stats.completion_tokens);
        all_stats.insert(category.clone(), final_stats);
    }

    Ok(EvaluationResult {
        category_stats: all_stats,
        token_stats: all_token_stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_result(id: i64) -> QuestionResult {
        QuestionResult {
            question_id: id,
            question: "q".to_string(),
            category: "cat".to_string(),
            options: vec!["A".to_string(), "B".to_string()],
            answer: "A".to_string(),
            answer_index: 0,
            response: "the answer is (A)".to_string(),
            pred: Some("A".to_string()),
            prompt: None,
            shots_used: Some(5),
            skipped: false,
        }
    }

    #[test]
    fn resume_counts_saved_results_and_shots_used() {
        let mut stats = CategoryStats::default();
        let mut wrong = sample_result(2);
        wrong.pred = Some("B".to_string());
        wrong.shots_used = Some(3);
        let mut legacy = sample_result(3);
        legacy.shots_used = None;

        for r in [sample_result(1), wrong, legacy] {
            stats.record_saved(&r);
        }
        assert_eq!(stats.correct, 2);
        assert_eq!(stats.wrong, 1);
        assert_eq!(stats.extraction_failures, 0);
        assert_eq!(stats.total(), 3);
        assert_eq!(stats.shots_used, BTreeMap::from([(3, 1), (5, 1)]));
    }

    #[test]
    fn result_without_new_fields_still_loads() {
        let json = r#"[{"question_id": 1, "question": "q", "category": "c",
            "options": ["a"], "answer": "A", "answer_index": 0,
            "response": "the answer is (A)", "pred": "A"}]"#;
        let r: Vec<QuestionResult> = serde_json::from_str(json).unwrap();
        assert_eq!(r[0].shots_used, None);
        assert!(!r[0].skipped);
    }

    #[test]
    fn run_config_reports_each_differing_field() {
        let toml = "[endpoint]\nbase_url = \"x\"\n[inference]\n[load]\n";
        let config: Config = toml::from_str(toml).unwrap();
        let saved = RunConfig::from_config(&config, "model-a");
        assert!(saved.differing_fields(&saved.clone()).is_empty());

        let mut changed = config.clone();
        changed.inference.mode = PromptMode::Completion;
        changed.inference.max_context_tokens = Some(2048);
        changed.inference.temperature = 0.5;
        changed.inference.system_prompt = "other".to_string();
        let now = RunConfig::from_config(&changed, "model-b");
        assert_eq!(
            saved.differing_fields(&now),
            vec![
                "model: saved model-a, now model-b (pass --model model-a to resume \
                 against the saved model name)",
                "mode: saved chat, now completion",
                "max_context_tokens: saved unset, now 2048",
                "temperature: saved 0, now 0.5",
                "system_prompt: text differs",
            ]
        );
    }

    #[test]
    fn save_results_roundtrips_through_load() {
        let dir = std::env::temp_dir().join(format!("mmlu_rt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cat_result.json");

        let results = vec![sample_result(1), sample_result(2)];
        save_results(&results, &path).unwrap();
        let loaded = load_existing_results(&path);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].question_id, 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn write_atomic_preserves_existing_file_when_write_fails() {
        // A crash/failure mid-write must not destroy already-saved results.
        let dir = std::env::temp_dir().join(format!("mmlu_atomic_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("results.json");
        std::fs::write(&dest, b"GOOD").unwrap();

        // Block the temp path by occupying it with a directory, so the temp
        // write fails before any rename can touch the destination.
        let tmp = dest.with_extension("tmp");
        std::fs::create_dir_all(&tmp).unwrap();

        let result = write_atomic(&dest, b"NEW");
        assert!(
            result.is_err(),
            "write should fail when temp path is blocked"
        );
        assert_eq!(
            std::fs::read(&dest).unwrap(),
            b"GOOD",
            "destination must retain its prior contents on a failed write"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Tests that drive `run_evaluation` against a mock llama-server.
    mod run {
        use super::*;
        use mockito::{Matcher, Server, ServerGuard};
        use serde_json::json;

        const CATEGORY: &str = "math";
        const MAX_TOKENS: u32 = 100;
        const SYSTEM_PROMPT: &str = "Questions about {subject}.";

        fn question(id: i64, text: &str, cot: &str) -> Question {
            Question {
                question_id: id,
                question: text.to_string(),
                options: vec!["yes".to_string(), "no".to_string()],
                answer: "A".to_string(),
                answer_index: 0,
                cot_content: cot.to_string(),
                category: CATEGORY.to_string(),
            }
        }

        type Data = HashMap<String, Vec<Question>>;

        fn dataset() -> (Data, Data) {
            let test = vec![
                question(100, "Test one?", ""),
                question(101, "Test two?", ""),
            ];
            let val = (0..5)
                .map(|i| {
                    question(
                        i,
                        &format!("Shot {i}?"),
                        &format!("A: Let's think step by step. Shot {i}. The answer is (A)."),
                    )
                })
                .collect();
            (
                HashMap::from([(CATEGORY.to_string(), test)]),
                HashMap::from([(CATEGORY.to_string(), val)]),
            )
        }

        fn config(server: &ServerGuard, mode: &str, max_context_tokens: Option<u32>) -> Config {
            let ctx = max_context_tokens
                .map(|c| format!("max_context_tokens = {c}"))
                .unwrap_or_default();
            let toml = format!(
                "[endpoint]\nbase_url = \"{}/v1\"\ntimeout = 5\n\
                 [inference]\nmode = \"{mode}\"\nnum_shots = 5\nmax_tokens = {MAX_TOKENS}\n\
                 system_prompt = \"{SYSTEM_PROMPT}\"\n{ctx}\n\
                 [load]\nconcurrent_requests = 1\n",
                server.url()
            );
            toml::from_str(&toml).unwrap()
        }

        fn header() -> String {
            SYSTEM_PROMPT.replace("{subject}", CATEGORY)
        }

        fn shots(k: usize) -> Vec<Question> {
            dataset().1[CATEGORY][..k].to_vec()
        }

        fn completion_prompt(k: usize, q: &Question) -> String {
            build_completion_prompt(&header(), &shots(k), &q.question, &q.options)
        }

        /// Fake `/tokenize`: `1 + per_block * n` tokens, where n is the number
        /// of "Question:" blocks in the content (shots + 1). The 1 stands for
        /// BOS. Requires `add_special: true`.
        async fn mock_tokenize(server: &mut ServerGuard, per_block: usize) -> mockito::Mock {
            server
                .mock("POST", "/tokenize")
                .match_body(Matcher::PartialJson(json!({"add_special": true})))
                .with_body_from_request(move |req| {
                    let body: serde_json::Value =
                        serde_json::from_slice(req.body().unwrap()).unwrap();
                    let content = body["content"].as_str().unwrap();
                    let n = 1 + per_block * content.matches("Question:").count();
                    json!({ "tokens": vec![0; n] }).to_string().into_bytes()
                })
                .expect_at_least(1)
                .create_async()
                .await
        }

        fn completion_body() -> String {
            json!({
                "choices": [{"text": " The answer is (A).", "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            })
            .to_string()
        }

        fn chat_body() -> String {
            json!({
                "id": "x", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "The answer is (A)."}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            })
            .to_string()
        }

        async fn mock_completion_for(
            server: &mut ServerGuard,
            prompt: &str,
            hits: usize,
        ) -> mockito::Mock {
            server
                .mock("POST", "/v1/completions")
                .match_body(Matcher::PartialJson(
                    json!({"prompt": prompt, "stop": ["Question:"], "max_tokens": MAX_TOKENS}),
                ))
                .with_body(completion_body())
                .expect(hits)
                .create_async()
                .await
        }

        async fn mock_any(server: &mut ServerGuard, path: &str, hits: usize) -> mockito::Mock {
            let body = if path.contains("chat") {
                chat_body()
            } else {
                completion_body()
            };
            server
                .mock("POST", path)
                .with_body(body)
                .expect(hits)
                .create_async()
                .await
        }

        fn saved(dir: &Path) -> Vec<QuestionResult> {
            load_existing_results(&dir.join(format!("{CATEGORY}_result.json")))
        }

        async fn run(config: &Config, dir: &Path) -> Result<EvaluationResult> {
            let (test, val) = dataset();
            run_evaluation(config, "m", &test, &val, dir).await
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn completion_mode_sends_one_reference_prompt_per_question() {
            let mut server = Server::new_async().await;
            let dir = tempfile::tempdir().unwrap();
            let (test, _) = dataset();
            let m0 = mock_completion_for(&mut server, &completion_prompt(5, &test[CATEGORY][0]), 1)
                .await;
            let m1 = mock_completion_for(&mut server, &completion_prompt(5, &test[CATEGORY][1]), 1)
                .await;
            let chat = mock_any(&mut server, "/v1/chat/completions", 0).await;

            let result = run(&config(&server, "completion", None), dir.path())
                .await
                .unwrap();

            m0.assert_async().await;
            m1.assert_async().await;
            chat.assert_async().await;
            assert_eq!(result.category_stats[CATEGORY].correct, 2);
            assert!(saved(dir.path()).iter().all(|r| r.shots_used == Some(5)));
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn max_context_tokens_reduces_shots_to_fit() {
            // 1 + 400 * (k + 1) + 100 <= 1500 first holds at k = 2.
            let mut server = Server::new_async().await;
            let dir = tempfile::tempdir().unwrap();
            let (test, _) = dataset();
            let tok = mock_tokenize(&mut server, 400).await;
            let m0 = mock_completion_for(&mut server, &completion_prompt(2, &test[CATEGORY][0]), 1)
                .await;
            let m1 = mock_completion_for(&mut server, &completion_prompt(2, &test[CATEGORY][1]), 1)
                .await;

            let result = run(&config(&server, "completion", Some(1500)), dir.path())
                .await
                .unwrap();

            tok.assert_async().await;
            m0.assert_async().await;
            m1.assert_async().await;
            let results = saved(dir.path());
            assert_eq!(results.len(), 2);
            assert!(
                results
                    .iter()
                    .all(|r| r.shots_used == Some(2) && !r.skipped)
            );
            assert_eq!(
                result.category_stats[CATEGORY].shots_used,
                BTreeMap::from([(2, 2)])
            );
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn chat_mode_counts_the_rendered_template() {
            // `/apply-template` adds one 'Question:' block, standing in for
            // template overhead. With it, 1 + 400 * (k + 2) + 100 <= 1500
            // gives k = 1; without it, 1 + 400 * (k + 1) + 100 <= 1500 gives
            // k = 2.
            let mut server = Server::new_async().await;
            let dir = tempfile::tempdir().unwrap();
            let (test, _) = dataset();
            let tok = mock_tokenize(&mut server, 400).await;
            let tmpl = server
                .mock("POST", "/apply-template")
                .with_body_from_request(|req| {
                    let body: serde_json::Value =
                        serde_json::from_slice(req.body().unwrap()).unwrap();
                    let joined = body["messages"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|m| m["content"].as_str().unwrap().to_string())
                        .collect::<Vec<_>>()
                        .join("\n");
                    json!({ "prompt": format!("Question: (template)\n{joined}") })
                        .to_string()
                        .into_bytes()
                })
                .expect_at_least(1)
                .create_async()
                .await;
            let mut chats = Vec::new();
            for q in &test[CATEGORY] {
                let messages = build_messages(&header(), &shots(1), &q.question, &q.options);
                chats.push(
                    server
                        .mock("POST", "/v1/chat/completions")
                        .match_body(Matcher::PartialJson(json!({ "messages": messages })))
                        .with_body(chat_body())
                        .expect(1)
                        .create_async()
                        .await,
                );
            }

            run(&config(&server, "chat", Some(1500)), dir.path())
                .await
                .unwrap();

            tok.assert_async().await;
            tmpl.assert_async().await;
            for m in &chats {
                m.assert_async().await;
            }
            assert!(saved(dir.path()).iter().all(|r| r.shots_used == Some(1)));
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn too_long_at_zero_shots_is_skipped_then_retried_on_resume() {
            // 1 + 400 + 100 > 300: even 0 shots does not fit.
            let mut server = Server::new_async().await;
            let dir = tempfile::tempdir().unwrap();
            let config_a = config(&server, "completion", Some(300));
            let _tok = mock_tokenize(&mut server, 400).await;
            let gen_calls = mock_any(&mut server, "/v1/completions", 0).await;

            let result = run(&config_a, dir.path()).await.unwrap();

            gen_calls.assert_async().await;
            assert_eq!(result.category_stats[CATEGORY].skipped, 2);
            assert_eq!(result.category_stats[CATEGORY].total(), 2);
            let results = saved(dir.path());
            assert_eq!(results.len(), 2);
            assert!(results.iter().all(|r| r.skipped && r.shots_used.is_none()));

            // Same settings, but a server whose prompts are shorter: the skipped
            // questions are evaluated again rather than kept from the file.
            let mut server_b = Server::new_async().await;
            let config_b = config(&server_b, "completion", Some(300));
            let _tok_b = mock_tokenize(&mut server_b, 10).await;
            let gen_b = mock_any(&mut server_b, "/v1/completions", 2).await;

            let result = run(&config_b, dir.path()).await.unwrap();

            gen_b.assert_async().await;
            assert_eq!(result.category_stats[CATEGORY].skipped, 0);
            assert_eq!(result.category_stats[CATEGORY].correct, 2);
            let results = saved(dir.path());
            assert_eq!(results.len(), 2);
            assert!(
                results
                    .iter()
                    .all(|r| !r.skipped && r.shots_used == Some(5))
            );
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn resume_with_a_changed_mode_is_refused() {
            let mut server = Server::new_async().await;
            let dir = tempfile::tempdir().unwrap();
            let gen_calls = mock_any(&mut server, "/v1/completions", 2).await;
            run(&config(&server, "completion", None), dir.path())
                .await
                .unwrap();
            gen_calls.assert_async().await;
            assert!(dir.path().join(RUN_CONFIG_FILE).exists());

            let chat = mock_any(&mut server, "/v1/chat/completions", 0).await;
            let err = run(&config(&server, "chat", None), dir.path())
                .await
                .err()
                .expect("resume with a different mode must fail");
            let msg = err.to_string();
            assert!(msg.contains("mode: saved completion, now chat"), "{msg}");
            assert!(msg.contains("fresh output directory"), "{msg}");
            chat.assert_async().await;
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn run_config_without_results_is_replaced() {
            // A run that stopped before saving any result must not lock the
            // directory to its settings.
            let mut server = Server::new_async().await;
            let dir = tempfile::tempdir().unwrap();
            let chat = config(&server, "chat", None);
            check_and_write_run_config(&chat, "m", dir.path()).unwrap();
            let gen_calls = mock_any(&mut server, "/v1/completions", 2).await;

            let completion = config(&server, "completion", None);
            run(&completion, dir.path()).await.unwrap();

            gen_calls.assert_async().await;
            let saved: RunConfig = serde_json::from_str(
                &std::fs::read_to_string(dir.path().join(RUN_CONFIG_FILE)).unwrap(),
            )
            .unwrap();
            assert_eq!(saved, RunConfig::from_config(&completion, "m"));
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn all_skipped_results_do_not_lock_settings() {
            // Every question is skipped at max_context_tokens 300; a rerun with
            // a larger window is a different setting but must be accepted,
            // because the only saved rows are skipped ones.
            let mut server = Server::new_async().await;
            let dir = tempfile::tempdir().unwrap();
            let _tok = mock_tokenize(&mut server, 400).await;
            let none = mock_any(&mut server, "/v1/completions", 0).await;
            run(&config(&server, "completion", Some(300)), dir.path())
                .await
                .unwrap();
            none.assert_async().await;
            assert!(saved(dir.path()).iter().all(|r| r.skipped));

            // 1 + 400 * (k + 1) + 100 <= 1500 gives k = 2.
            let gen_calls = mock_any(&mut server, "/v1/completions", 2).await;
            let result = run(&config(&server, "completion", Some(1500)), dir.path())
                .await
                .unwrap();
            gen_calls.assert_async().await;
            assert_eq!(result.category_stats[CATEGORY].skipped, 0);
            assert!(saved(dir.path()).iter().all(|r| r.shots_used == Some(2)));
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn results_without_run_config_are_resumed() {
            let mut server = Server::new_async().await;
            let dir = tempfile::tempdir().unwrap();
            let mut old = sample_result(100);
            old.category = CATEGORY.to_string();
            old.shots_used = None;
            save_results(&[old], &dir.path().join(format!("{CATEGORY}_result.json"))).unwrap();
            let gen_calls = mock_any(&mut server, "/v1/completions", 1).await;

            let result = run(&config(&server, "completion", None), dir.path())
                .await
                .unwrap();

            gen_calls.assert_async().await;
            assert_eq!(result.category_stats[CATEGORY].correct, 2);
            assert!(dir.path().join(RUN_CONFIG_FILE).exists());
        }
    }
}
