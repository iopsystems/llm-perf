use anyhow::Result;
use chrono::{Timelike, Utc};
use metriken_exposition::{
    MsgpackToParquet, ParquetHistogramType, ParquetOptions, Snapshot, SnapshotterBuilder,
};
use std::sync::atomic::Ordering;
use std::time::Duration;
use tempfile::NamedTempFile;
use tokio::fs::File;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::time::{Instant, interval_at, timeout};

use crate::config::Config;
use crate::metrics::RUNNING;

/// Captures metric snapshots at regular intervals and writes them to a parquet file
pub async fn capture_snapshots(config: Config) -> Result<()> {
    let metrics_config = match config.metrics.as_ref() {
        Some(cfg) => cfg,
        None => return Ok(()), // No metrics configured
    };

    // Parse interval
    let interval_duration = humantime::parse_duration(&metrics_config.interval)?;

    // Create a temporary file for msgpack data (will be converted to parquet)
    let temp_file = NamedTempFile::new()?;
    let file = File::from_std(temp_file.reopen()?);
    let mut writer = BufWriter::new(file);

    // Get an aligned start time (aligned to the second)
    let start = Instant::now() - Duration::from_nanos(Utc::now().nanosecond() as u64)
        + Duration::from_secs(1);

    // Create interval timer
    let mut interval = interval_at(start, interval_duration);

    // Build snapshotter with metadata
    let snapshotter = SnapshotterBuilder::new()
        .metadata("source".to_string(), env!("CARGO_PKG_NAME").to_string())
        .metadata("version".to_string(), env!("CARGO_PKG_VERSION").to_string())
        .metadata(
            "sampling_interval_ms".to_string(),
            interval_duration.as_millis().to_string(),
        )
        .build();

    log::info!(
        "Starting metrics capture to {:?} with interval {:?}",
        metrics_config.output,
        interval_duration
    );

    // Main snapshot loop
    while RUNNING.load(Ordering::Relaxed) {
        // Use timeout to check RUNNING flag periodically
        if timeout(Duration::from_secs(1), interval.tick())
            .await
            .is_err()
        {
            continue;
        }

        let snapshot = snapshotter.snapshot();

        // Serialize to msgpack using the Snapshot::to_msgpack API
        let buf = Snapshot::to_msgpack(&snapshot).expect("failed to serialize snapshot");

        // Write to temporary file
        if let Err(e) = writer.write_all(&buf).await {
            log::error!("Error writing metrics snapshot: {}", e);
            break;
        }
    }

    // Flush writer
    writer.flush().await?;
    drop(writer); // Close the writer

    // Convert msgpack to parquet
    log::info!("Converting msgpack to parquet...");

    // Set up parquet options with standard histogram type
    let mut opts = ParquetOptions::new().histogram_type(ParquetHistogramType::Standard);

    if let Some(batch_size) = metrics_config.batch_size {
        opts = opts.max_batch_size(batch_size);
    }

    // Convert msgpack to parquet
    MsgpackToParquet::with_options(opts)
        .convert_file_path(temp_file.path(), &metrics_config.output)?;

    log::info!(
        "Successfully wrote metrics snapshots to {:?} (parquet format)",
        metrics_config.output
    );

    Ok(())
}
