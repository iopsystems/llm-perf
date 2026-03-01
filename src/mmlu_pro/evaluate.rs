use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Semaphore;

use llm_bench::client::{ChatCompletionRequest, ClientConfig, OpenAIClient};

use super::config::Config;
use super::dataset::Question;
use super::extract::extract_answer;
use super::prompt::build_messages;

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
	#[serde(skip_serializing_if = "Option::is_none")]
	pub prompt: Option<Vec<PromptMessage>>,
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

/// Save results to a category result file.
fn save_results(results: &[QuestionResult], path: &Path) -> Result<()> {
	let json = serde_json::to_string_pretty(results)?;
	std::fs::write(path, json)?;
	Ok(())
}

/// Save category summary to a JSON file.
fn save_summary(stats: &HashMap<String, CategoryStats>, path: &Path) -> Result<()> {
	let mut summary: HashMap<String, serde_json::Value> = HashMap::new();
	let mut total_corr = 0u32;
	let mut total_wrong = 0u32;

	for (category, s) in stats {
		let total = s.correct + s.wrong;
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
				"extraction_failures": s.extraction_failures,
				"acc": acc,
			}),
		);
		total_corr += s.correct;
		total_wrong += s.wrong;
	}

	let total = total_corr + total_wrong;
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
			"acc": acc,
		}),
	);

	let json = serde_json::to_string_pretty(&summary)?;
	std::fs::write(path, json)?;
	Ok(())
}

/// Run evaluation across all specified categories.
pub async fn run_evaluation(
	config: &Config,
	test_data: &HashMap<String, Vec<Question>>,
	val_data: &HashMap<String, Vec<Question>>,
	output_dir: &Path,
) -> Result<EvaluationResult> {
	let client_config = ClientConfig {
		base_url: config.server.url.clone(),
		api_key: Some(config.server.api_key.clone()),
		model: config.server.model.clone(),
		timeout: std::time::Duration::from_secs_f64(config.server.timeout),
		max_retries: 3,
		retry_initial_delay_ms: 1000,
		retry_max_delay_ms: 30000,
		pool_size: config.test.parallel,
	};

	let client = Arc::new(OpenAIClient::new(client_config)?);
	let semaphore = Arc::new(Semaphore::new(config.test.parallel));

	let mut all_stats: HashMap<String, CategoryStats> = HashMap::new();
	let mut all_token_stats = TokenStats::default();

	// Determine which categories to evaluate
	let categories: Vec<String> = if config.test.categories.contains(&"all".to_string()) {
		let mut cats: Vec<String> = test_data.keys().cloned().collect();
		cats.sort();
		cats
	} else {
		config.test.categories.clone()
	};

	let system_prompt_template = &config.inference.system_prompt;

	for category in &categories {
		let test_questions = match test_data.get(category) {
			Some(q) => q,
			None => {
				eprintln!("Warning: category '{}' not found in test data, skipping.", category);
				continue;
			}
		};

		let cot_examples: Vec<Question> = val_data.get(category).cloned().unwrap_or_default();

		let system_prompt = system_prompt_template.replace("{subject}", category);

		let result_path = output_dir.join(format!("{}_result.json", category));
		let summary_path = output_dir.join(format!("{}_summary.json", category));

		// Load existing results for resume
		let existing_results = load_existing_results(&result_path);
		let existing_ids: std::collections::HashSet<i64> =
			existing_results.iter().map(|r| r.question_id).collect();

		// Count stats from existing results
		let mut cat_stats = CategoryStats::default();
		for r in &existing_results {
			match &r.pred {
				Some(pred) if pred == &r.answer => cat_stats.correct += 1,
				Some(_) => cat_stats.wrong += 1,
				None => {
					cat_stats.wrong += 1;
					cat_stats.extraction_failures += 1;
				}
			}
		}

		// Filter to only new questions
		let new_questions: Vec<&Question> = test_questions
			.iter()
			.filter(|q| !existing_ids.contains(&q.question_id))
			.collect();

		let total = test_questions.len();
		let already_done = existing_ids.len();

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

		let pb = ProgressBar::new(new_questions.len() as u64);
		pb.set_style(
			ProgressStyle::default_bar()
				.template(&format!(
					"{{spinner:.green}} {} [{{bar:40.cyan/blue}}] {{pos}}/{{len}} [{{elapsed_precise}}] {{msg}}",
					category
				))
				.unwrap()
				.progress_chars("=>-"),
		);

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
			let pb = pb.clone();
			let result_path = result_path.clone();
			let summary_path = summary_path.clone();
			let system_prompt = system_prompt.clone();
			let cot_examples = cot_examples.clone();
			let question = question.clone();
			let log_prompt = config.log.log_prompt;
			let temperature = config.inference.temperature;
			let top_p = config.inference.top_p;
			let max_tokens = config.inference.max_tokens;
			let model = config.server.model.clone();
			let verbosity = config.log.verbosity;

			let handle = tokio::spawn(async move {
				let _permit = semaphore.acquire().await.unwrap();

				let messages =
					build_messages(&system_prompt, &cot_examples, &question.question, &question.options);

				let request = ChatCompletionRequest {
					model,
					messages: messages.clone(),
					max_tokens: Some(max_tokens),
					temperature: Some(temperature),
					top_p: Some(top_p),
					frequency_penalty: Some(0.0),
					presence_penalty: Some(0.0),
					stop: Some(vec!["Question:".to_string()]),
					stream: Some(false),
					stream_options: None,
				};

				let response = match client.chat_completion(request).await {
					Ok(resp) => resp,
					Err(e) => {
						eprintln!("Error for question {}: {}", question.question_id, e);
						pb.inc(1);
						return;
					}
				};

				// Track token usage
				{
					let mut ts = token_stats.lock().await;
					ts.prompt_tokens.push(response.usage.prompt_tokens);
					ts.completion_tokens.push(response.usage.completion_tokens);
				}

				let response_text = response
					.choices
					.first()
					.map(|c| c.message.content.trim().to_string())
					.unwrap_or_default();

				let pred = extract_answer(&response_text);
				let pred_str = pred.map(|c| c.to_string());

				if verbosity >= 2 {
					eprintln!(
						"\nQ{}: pred={:?} answer={} | {}",
						question.question_id,
						pred_str,
						question.answer,
						&response_text[..response_text.len().min(100)]
					);
				}

				let prompt_log = if log_prompt {
					Some(
						messages
							.iter()
							.map(|m| PromptMessage {
								role: m.role.clone(),
								content: m.content.clone(),
							})
							.collect(),
					)
				} else {
					None
				};

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
				};

				// Update stats
				{
					let mut s = stats.lock().await;
					match &pred_str {
						Some(p) if p == &question.answer => s.correct += 1,
						Some(_) => s.wrong += 1,
						None => {
							s.wrong += 1;
							s.extraction_failures += 1;
							if verbosity >= 1 {
								eprintln!(
									"Extraction failed for question {}: {}",
									question.question_id,
									&result.response[..result.response.len().min(200)]
								);
							}
						}
					}
				}

				// Save result
				{
					let mut res = results.lock().await;
					res.push(result);

					// Deduplicate by question_id
					let mut seen = std::collections::HashSet::new();
					res.retain(|r| seen.insert(r.question_id));

					let _ = save_results(&res, &result_path);
				}

				// Save summary
				{
					let s = stats.lock().await;
					let mut summary_stats: HashMap<String, CategoryStats> = HashMap::new();
					summary_stats.insert(question.category.clone(), s.clone());
					let _ = save_summary(&summary_stats, &summary_path);
				}

				// Update progress bar
				{
					let s = stats.lock().await;
					let total = s.correct + s.wrong;
					let acc = if total > 0 {
						s.correct as f64 / total as f64 * 100.0
					} else {
						0.0
					};
					pb.set_message(format!(
						"Correct: {} Wrong: {} Accuracy: {:.2}%",
						s.correct, s.wrong, acc
					));
				}
				pb.inc(1);
			});

			handles.push(handle);
		}

		// Wait for all tasks to complete
		for handle in handles {
			let _ = handle.await;
		}

		pb.finish();

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

		let total = final_stats.correct + final_stats.wrong;
		let acc = if total > 0 {
			final_stats.correct as f64 / total as f64 * 100.0
		} else {
			0.0
		};
		eprintln!(
			"{}: {}/{} correct ({:.2}%), {} extraction failures",
			category, final_stats.correct, total, acc, final_stats.extraction_failures
		);

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
