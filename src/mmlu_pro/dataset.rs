use anyhow::{Context, Result};
use arrow::array::{Array, AsArray, ListArray, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Question {
    pub question_id: i64,
    pub question: String,
    pub options: Vec<String>,
    pub answer: String,
    pub answer_index: i64,
    pub cot_content: String,
    pub category: String,
}

/// Download MMLU-Pro dataset from HuggingFace and return paths to test and validation parquet files.
pub async fn download_dataset() -> Result<(PathBuf, PathBuf)> {
    let api = hf_hub::api::tokio::Api::new()?;
    let repo = api.dataset("TIGER-Lab/MMLU-Pro".to_string());

    eprintln!("Downloading MMLU-Pro dataset from HuggingFace...");

    let test_path = repo
        .get("data/test-00000-of-00001.parquet")
        .await
        .context("Failed to download test split")?;

    let val_path = repo
        .get("data/validation-00000-of-00001.parquet")
        .await
        .context("Failed to download validation split")?;

    eprintln!("Dataset cached locally.");

    Ok((test_path, val_path))
}

/// Load questions from a parquet file.
fn load_parquet(path: &PathBuf) -> Result<Vec<Question>> {
    let file = std::fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut questions = Vec::new();

    for batch in reader {
        let batch = batch?;

        let question_id_col = batch
            .column_by_name("question_id")
            .context("missing question_id column")?;
        let question_col = batch
            .column_by_name("question")
            .context("missing question column")?;
        let options_col = batch
            .column_by_name("options")
            .context("missing options column")?;
        let answer_col = batch
            .column_by_name("answer")
            .context("missing answer column")?;
        let answer_index_col = batch
            .column_by_name("answer_index")
            .context("missing answer_index column")?;
        let cot_content_col = batch
            .column_by_name("cot_content")
            .context("missing cot_content column")?;
        let category_col = batch
            .column_by_name("category")
            .context("missing category column")?;

        let question_ids = question_id_col.as_primitive::<arrow::datatypes::Int64Type>();
        let question_strings = question_col
            .as_any()
            .downcast_ref::<StringArray>()
            .context("question column is not string")?;
        let options_list = options_col
            .as_any()
            .downcast_ref::<ListArray>()
            .context("options column is not list")?;
        let answers = answer_col
            .as_any()
            .downcast_ref::<StringArray>()
            .context("answer column is not string")?;
        let answer_indices = answer_index_col.as_primitive::<arrow::datatypes::Int64Type>();
        let cot_contents = cot_content_col
            .as_any()
            .downcast_ref::<StringArray>()
            .context("cot_content column is not string")?;
        let categories = category_col
            .as_any()
            .downcast_ref::<StringArray>()
            .context("category column is not string")?;

        for i in 0..batch.num_rows() {
            let options_array = options_list.value(i);
            let options_strings = options_array
                .as_any()
                .downcast_ref::<StringArray>()
                .context("options values are not strings")?;

            let mut options = Vec::new();
            for j in 0..options_strings.len() {
                let opt = options_strings.value(j);
                if opt != "N/A" {
                    options.push(opt.to_string());
                }
            }

            questions.push(Question {
                question_id: question_ids.value(i),
                question: question_strings.value(i).to_string(),
                options,
                answer: answers.value(i).to_string(),
                answer_index: answer_indices.value(i),
                cot_content: cot_contents.value(i).to_string(),
                category: categories.value(i).to_string(),
            });
        }
    }

    Ok(questions)
}

/// Group questions by category.
fn group_by_category(questions: Vec<Question>) -> HashMap<String, Vec<Question>> {
    let mut groups: HashMap<String, Vec<Question>> = HashMap::new();
    for q in questions {
        groups.entry(q.category.clone()).or_default().push(q);
    }
    groups
}

/// Apply subset sampling: take the first `subset` fraction of items per category.
fn apply_subset(
    groups: HashMap<String, Vec<Question>>,
    subset: f64,
) -> HashMap<String, Vec<Question>> {
    let mut result = HashMap::new();
    for (category, items) in groups {
        let subset_size = ((items.len() as f64 * subset).round() as usize).max(1);
        result.insert(category, items.into_iter().take(subset_size).collect());
    }
    result
}

/// Load and preprocess the MMLU-Pro dataset.
/// Returns (test_data_by_category, validation_data_by_category).
pub async fn load_mmlu_pro(
    subset: f64,
) -> Result<(
    HashMap<String, Vec<Question>>,
    HashMap<String, Vec<Question>>,
)> {
    let (test_path, val_path) = download_dataset().await?;

    let test_questions = load_parquet(&test_path)?;
    let val_questions = load_parquet(&val_path)?;

    let test_grouped = group_by_category(test_questions);
    let val_grouped = group_by_category(val_questions);

    let test_data = apply_subset(test_grouped, subset);

    Ok((test_data, val_grouped))
}
