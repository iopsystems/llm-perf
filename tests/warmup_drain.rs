//! A request in flight at the end of a duration warmup runs to completion
//! instead of being aborted. Its own test binary: the runner's metrics are
//! process-global.

mod common;

#[tokio::test(flavor = "multi_thread")]
async fn requests_in_flight_at_the_warmup_deadline_finish() {
    // Each of the 4 warmup requests streams for about 1.5 s, past the 1 s
    // warmup deadline, so the first 4 requests are exactly the warmup's.
    let (url, seen) = common::streaming_server(4, 15).await;
    common::run(
        &url,
        "{\"prompt\": \"hello\"}\n",
        "concurrent_requests = 4\nduration_seconds = 1\nwarmup_duration = 1",
        "",
    )
    .await;
    let seen = seen.lock().unwrap();
    assert!(seen.len() > 4, "the main test sent no requests: {seen:?}");
    let dropped: Vec<usize> = (0..4).filter(|&i| !seen[i].completed).collect();
    assert!(
        dropped.is_empty(),
        "warmup requests {dropped:?} were dropped mid-stream"
    );
}
