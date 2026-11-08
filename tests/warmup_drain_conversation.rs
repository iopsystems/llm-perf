//! A warmup conversation starts no turn after the warmup deadline. Its own test
//! binary: the runner's metrics are process-global.

mod common;

#[tokio::test(flavor = "multi_thread")]
async fn a_warmup_conversation_starts_no_turn_after_the_deadline() {
    // The first turn of each of the 2 warmup conversations streams for about
    // 1.5 s, past the 1 s warmup deadline. A request carrying an assistant
    // message "r0 ..." or "r1 ..." is a warmup conversation's second turn.
    let (url, seen) = common::streaming_server(2, 15).await;
    let conversation = r#"{"conversations": [{"from": "human", "value": "one"}, {"from": "gpt", "value": "a"}, {"from": "human", "value": "two"}, {"from": "gpt", "value": "b"}, {"from": "human", "value": "three"}]}"#;
    common::run(
        &url,
        &format!("{conversation}\n"),
        "concurrent_requests = 2\nduration_seconds = 1\nwarmup_duration = 1",
        "",
    )
    .await;
    let seen = seen.lock().unwrap();
    assert!(
        seen[..2].iter().all(|s| s.completed),
        "a warmup turn in flight at the deadline was dropped: {seen:?}"
    );
    let second_turns: Vec<&common::Seen> = seen
        .iter()
        .filter(|s| {
            s.assistant
                .iter()
                .any(|a| a.starts_with("r0 ") || a.starts_with("r1 "))
        })
        .collect();
    assert!(
        second_turns.is_empty(),
        "warmup conversations sent turns after the deadline: {second_turns:?}"
    );
    assert!(seen.len() > 2, "the main test sent no requests: {seen:?}");
}
