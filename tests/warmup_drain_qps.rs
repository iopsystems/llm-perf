//! In QPS mode, a warmup arrival still waiting for a slot at the warmup
//! deadline is not sent. Its own test binary: the runner's metrics are
//! process-global.

mod common;

#[tokio::test(flavor = "multi_thread")]
async fn a_queued_qps_warmup_arrival_is_not_sent_after_the_deadline() {
    // 20 arrivals a second against 2 slots. The first 2 requests stream for
    // about 1.5 s, past the 1 s warmup deadline, so the warmup's other
    // arrivals are still queued at the deadline. Warmup and main test take
    // prompt indexes from one counter, in arrival order (the arrival that ends
    // the warmup loop takes one and is not sent).
    let (url, seen) = common::streaming_server(2, 15).await;
    let prompts: String = (0..1000)
        .map(|i| format!("{{\"prompt\": \"p{i}\"}}\n"))
        .collect();
    common::run(
        &url,
        &prompts,
        "concurrent_requests = 2\nqps = 20.0\nduration_seconds = 1\nwarmup_duration = 1",
        "",
    )
    .await;
    let seen = seen.lock().unwrap();
    assert!(
        seen[..2].iter().all(|s| s.completed),
        "a warmup request in flight at the deadline was dropped: {seen:?}"
    );
    let mut sent: Vec<usize> = seen
        .iter()
        .filter_map(|s| s.user.first()?.strip_prefix('p')?.parse().ok())
        .collect();
    sent.sort_unstable();
    assert_eq!(sent[..2], [0, 1], "{sent:?}");
    // Uniform arrivals 50 ms apart put about 20 in the 1 s warmup, so the main
    // test's first prompt is near p20; had the queued warmup arrivals been
    // sent, the third prompt would be p2.
    assert!(
        sent[2] >= 10,
        "warmup arrivals queued at the deadline were sent: {sent:?}"
    );
}
