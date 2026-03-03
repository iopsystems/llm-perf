use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::sync::mpsc;

use crate::client::TokenLogprob;

/// A record of logprobs for a single prompt/response pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogprobRecord {
    /// Index of the prompt in the input file
    pub prompt_index: usize,
    /// The original prompt text
    pub prompt: String,
    /// Token-level logprobs from the response
    pub tokens: Vec<TokenLogprob>,
}

/// Asynchronous writer that receives LogprobRecords via a channel and writes them as JSONL
pub struct LogprobWriter {
    rx: mpsc::Receiver<LogprobRecord>,
    output_path: std::path::PathBuf,
}

impl LogprobWriter {
    /// Create a new LogprobWriter and its sender channel
    pub fn new(
        output_path: std::path::PathBuf,
        buffer_size: usize,
    ) -> (mpsc::Sender<LogprobRecord>, Self) {
        let (tx, rx) = mpsc::channel(buffer_size);
        (tx, Self { rx, output_path })
    }

    /// Run the writer loop, consuming records from the channel and writing to disk
    pub async fn run(mut self) -> Result<()> {
        use tokio::io::AsyncWriteExt;

        let file = tokio::fs::File::create(&self.output_path).await?;
        let mut writer = tokio::io::BufWriter::new(file);

        while let Some(record) = self.rx.recv().await {
            let json = serde_json::to_string(&record)?;
            writer.write_all(json.as_bytes()).await?;
            writer.write_all(b"\n").await?;
        }

        writer.flush().await?;
        log::info!("Logprobs written to {}", self.output_path.display());
        Ok(())
    }
}

/// Load logprob records from a JSONL file
pub fn load_logprob_file(path: &Path) -> Result<Vec<LogprobRecord>> {
    let contents = std::fs::read_to_string(path)?;
    let mut records = Vec::new();

    for (line_num, line) in contents.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<LogprobRecord>(line) {
            Ok(record) => records.push(record),
            Err(e) => {
                log::warn!(
                    "Failed to parse logprob record at line {}: {}",
                    line_num + 1,
                    e
                );
            }
        }
    }

    Ok(records)
}
