//! A warmup conversation sleeping between turns at the warmup deadline does not
//! hold up the main test. Its own test binary: the runner's metrics are
//! process-global.

mod common;

#[tokio::test(flavor = "multi_thread")]
async fn a_warmup_think_delay_ends_at_the_deadline() {
    // Each request streams for about 0.2 s, so the warmup conversation's first
    // turn ends well before the 1 s deadline and its 5 s think delay crosses it.
    let (url, seen) = common::streaming_server(0, 0).await;
    let conversation = r#"{"conversations": [{"from": "human", "value": "one"}, {"from": "gpt", "value": "a"}, {"from": "human", "value": "two"}, {"from": "gpt", "value": "b"}, {"from": "human", "value": "three"}]}"#;
    common::run(
        &url,
        &format!("{conversation}\n"),
        "concurrent_requests = 1\nduration_seconds = 1\nwarmup_duration = 1",
        "[conversation]\nturn_delay_ms = 5000\nturn_delay_max_ms = 60000",
    )
    .await;
    let seen = seen.lock().unwrap();
    // First turns carry no assistant message: the warmup's, then the main test's.
    let first_turns: Vec<_> = seen.iter().filter(|s| s.assistant.is_empty()).collect();
    assert!(first_turns.len() >= 2, "{seen:?}");
    let gap = first_turns[1].at - first_turns[0].at;
    assert!(
        gap.as_secs_f64() < 2.5,
        "the main test started {gap:?} after the warmup's first request, past the 1 s deadline"
    );
}
