// Each test binary uses part of this module.
#![allow(dead_code)]

//! A streaming chat-completions server for whole-run tests, and a runner for a
//! config written to a temporary file.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// What the server saw for one chat-completions request, in arrival order.
#[derive(Debug, Clone)]
pub struct Seen {
    /// When the server finished reading the request.
    pub at: Instant,
    /// The contents of the request's user messages.
    pub user: Vec<String>,
    /// The contents of the request's assistant messages.
    pub assistant: Vec<String>,
    /// Whether every chunk and `[DONE]` was written; false if a write failed
    /// because the client dropped the connection.
    pub completed: bool,
}

/// Starts a server that streams each chat-completions response as SSE chunks
/// 100 ms apart, each carrying the content `r<index> `, where index is the
/// request's arrival order. The first `slow` requests get `slow_chunks` chunks,
/// the rest 2. Other paths get a 404. Returns the base URL (`.../v1`) and the
/// per-request record.
pub async fn streaming_server(slow: usize, slow_chunks: usize) -> (String, Arc<Mutex<Vec<Seen>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let record = Arc::clone(&seen);
    tokio::spawn(async move {
        loop {
            let Ok((sock, _)) = listener.accept().await else {
                return;
            };
            let record = Arc::clone(&record);
            tokio::spawn(async move { serve(sock, record, slow, slow_chunks).await });
        }
    });
    (format!("http://{addr}/v1"), seen)
}

async fn serve(
    mut sock: TcpStream,
    record: Arc<Mutex<Vec<Seen>>>,
    slow: usize,
    slow_chunks: usize,
) {
    let Some((path, body)) = read_request(&mut sock).await else {
        return;
    };
    if !path.ends_with("/chat/completions") {
        let _ = sock
            .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .await;
        return;
    }
    let messages = serde_json::from_slice::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v["messages"].as_array().cloned())
        .unwrap_or_default();
    let contents = |role: &str| -> Vec<String> {
        messages
            .iter()
            .filter(|m| m["role"] == role)
            .filter_map(|m| m["content"].as_str().map(str::to_string))
            .collect()
    };
    let index = {
        let mut seen = record.lock().unwrap();
        seen.push(Seen {
            at: Instant::now(),
            user: contents("user"),
            assistant: contents("assistant"),
            completed: false,
        });
        seen.len() - 1
    };
    let chunks = if index < slow { slow_chunks } else { 2 };
    let mut writes = vec![
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
            .to_string(),
    ];
    for i in 0..chunks {
        let finish = if i + 1 == chunks { "\"stop\"" } else { "null" };
        writes.push(format!(
            "data: {{\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"m\",\
             \"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"r{index} \"}},\"finish_reason\":{finish}}}]}}\n\n"
        ));
    }
    writes.push("data: [DONE]\n\n".to_string());
    for (i, w) in writes.iter().enumerate() {
        if i > 1 {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        if sock.write_all(w.as_bytes()).await.is_err() || sock.flush().await.is_err() {
            return;
        }
    }
    record.lock().unwrap()[index].completed = true;
    let _ = sock.shutdown().await;
}

/// Reads one request; returns its path and body.
async fn read_request(sock: &mut TcpStream) -> Option<(String, Vec<u8>)> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
        let n = sock.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let path = head.split_whitespace().nth(1)?.to_string();
    let length = head
        .lines()
        .find_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.eq_ignore_ascii_case("content-length")
                .then(|| v.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    while buf.len() < header_end + length {
        let n = sock.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    Some((path, buf[header_end..header_end + length].to_vec()))
}

/// Runs a benchmark against `base_url` over the workloads in `input_jsonl`,
/// with `load` as the body of the config's `[load]` table and `extra`
/// appended to the config.
pub async fn run(base_url: &str, input_jsonl: &str, load: &str, extra: &str) {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.jsonl");
    std::fs::write(&input, input_jsonl).unwrap();
    let config: PathBuf = dir.path().join("config.toml");
    std::fs::write(
        &config,
        format!(
            r#"
[endpoint]
base_url = "{base_url}"
model = "m"
timeout = 30
max_tokens = 8
max_retries = 0
health_check_timeout = 0

[input]
file = "{}"

[load]
{load}

[output]
format = "json"
quiet = true

{extra}
"#,
            input.display()
        ),
    )
    .unwrap();
    let config = llm_perf::Config::load(&config).unwrap();
    let runner = llm_perf::BenchmarkRunner::new(config).await.unwrap();
    runner.run().await.unwrap();
}
