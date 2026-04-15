use std::convert::Infallible;
use std::net::SocketAddr;
use warp::Filter;

/// Start the HTTP admin server for metrics
pub async fn start_server(addr: SocketAddr) {
    info!("Starting metrics server on {}", addr);

    let routes = metrics_endpoint()
        .or(metrics_json_endpoint())
        .or(vars_endpoint())
        .or(vars_json_endpoint());

    warp::serve(routes).run(addr).await;
}

/// GET /metrics - Prometheus/OpenMetrics format
fn metrics_endpoint() -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone
{
    warp::path!("metrics")
        .and(warp::get())
        .and_then(prometheus_metrics)
}

/// GET /metrics.json - JSON format (Finagle compatible)
fn metrics_json_endpoint()
-> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("metrics.json")
        .and(warp::get())
        .and_then(json_metrics)
}

/// GET /vars - Human readable format
fn vars_endpoint() -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("vars").and(warp::get()).and_then(human_metrics)
}

/// GET /vars.json - JSON format (alias)
fn vars_json_endpoint()
-> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("vars.json")
        .and(warp::get())
        .and_then(json_metrics)
}

async fn prometheus_metrics() -> Result<impl warp::Reply, Infallible> {
    let options = metriken_exposition::PrometheusOptions::default()
        .with_percentiles(vec![0.5, 0.9, 0.95, 0.99, 0.999]);
    let content = metriken_exposition::prometheus_text(&options);
    Ok(warp::reply::with_header(
        content,
        "content-type",
        "text/plain; version=0.0.4; charset=utf-8",
    ))
}

async fn json_metrics() -> Result<impl warp::Reply, Infallible> {
    use histogram::SampleQuantiles;
    use metriken::Value;
    use serde_json::json;

    let mut metrics = serde_json::Map::new();
    let quantile_values = [0.5, 0.9, 0.95, 0.99, 0.999];

    for metric in &metriken::metrics() {
        let name = metric.name();

        match metric.value() {
            Some(Value::Counter(value)) => {
                metrics.insert(name.to_string(), json!(value));
            }
            Some(Value::Gauge(value)) => {
                metrics.insert(name.to_string(), json!(value));
            }
            Some(Value::Histogram(histogram)) => {
                if let Some(loaded) = histogram.load()
                    && let Ok(Some(result)) = loaded.quantiles(&quantile_values)
                {
                    for (quantile, bucket) in result.entries() {
                        let key = format!("{}/p{}", name, (quantile.as_f64() * 1000.0) as u32);
                        metrics.insert(key, json!(bucket.end()));
                    }
                }
            }
            Some(Value::HistogramGroup(group)) => {
                let snapshot = group.metadata_snapshot();
                for (idx, meta) in &snapshot {
                    if let Some(loaded) = group.load_histogram(*idx)
                        && let Ok(Some(result)) = loaded.quantiles(&quantile_values)
                    {
                        let suffix = format_metadata_suffix(meta);
                        for (quantile, bucket) in result.entries() {
                            let key = format!(
                                "{}{}/p{}",
                                name,
                                suffix,
                                (quantile.as_f64() * 1000.0) as u32
                            );
                            metrics.insert(key, json!(bucket.end()));
                        }
                    }
                }
            }
            _ => continue,
        }
    }

    Ok(warp::reply::json(&metrics))
}

async fn human_metrics() -> Result<impl warp::Reply, Infallible> {
    use histogram::SampleQuantiles;
    use metriken::Value;

    let mut lines = Vec::new();
    let quantile_values = [0.5, 0.9, 0.95, 0.99, 0.999];

    for metric in &metriken::metrics() {
        let name = metric.name();

        match metric.value() {
            Some(Value::Counter(value)) => {
                lines.push(format!("{}: {}", name, value));
            }
            Some(Value::Gauge(value)) => {
                lines.push(format!("{}: {}", name, value));
            }
            Some(Value::Histogram(histogram)) => {
                if let Some(loaded) = histogram.load()
                    && let Ok(Some(result)) = loaded.quantiles(&quantile_values)
                {
                    for (quantile, bucket) in result.entries() {
                        lines.push(format!(
                            "{}/p{}: {}",
                            name,
                            (quantile.as_f64() * 1000.0) as u32,
                            bucket.end()
                        ));
                    }
                }
            }
            Some(Value::HistogramGroup(group)) => {
                let snapshot = group.metadata_snapshot();
                for (idx, meta) in &snapshot {
                    if let Some(loaded) = group.load_histogram(*idx)
                        && let Ok(Some(result)) = loaded.quantiles(&quantile_values)
                    {
                        let suffix = format_metadata_suffix(meta);
                        for (quantile, bucket) in result.entries() {
                            lines.push(format!(
                                "{}{}/p{}: {}",
                                name,
                                suffix,
                                (quantile.as_f64() * 1000.0) as u32,
                                bucket.end()
                            ));
                        }
                    }
                }
            }
            _ => continue,
        }
    }

    lines.sort();
    let content = lines.join("\n") + "\n";
    Ok(warp::reply::with_header(
        content,
        "content-type",
        "text/plain; charset=utf-8",
    ))
}

/// Format metadata as a path suffix (e.g., "/reasoning/small").
fn format_metadata_suffix(meta: &std::collections::HashMap<String, String>) -> String {
    if meta.is_empty() {
        return String::new();
    }
    let mut parts: Vec<&str> = meta.values().map(|v| v.as_str()).collect();
    parts.sort();
    format!("/{}", parts.join("/"))
}

use log::info;
