use std::time::{Duration, Instant};

use betanet_mixnode::rate::TokenBucket;

#[tokio::test]
async fn test_token_bucket_waits_without_spinning() {
    // 5 tokens per second => 200ms to refill one token
    let bucket = TokenBucket::new(1, 5.0);

    // Consume initial token immediately
    bucket.consume(1).await.unwrap();

    // Measure time for next token which requires refill
    let denied_before = bucket
        .stats()
        .requests_denied
        .load(std::sync::atomic::Ordering::Relaxed);
    let start = Instant::now();
    bucket.consume(1).await.unwrap();
    let elapsed = start.elapsed();
    let denied_after = bucket
        .stats()
        .requests_denied
        .load(std::sync::atomic::Ordering::Relaxed);
    let denied_diff = denied_after - denied_before;

    // Expect roughly 200ms wait and only a single denied request indicating no busy loop
    assert!(elapsed >= Duration::from_millis(180));
    assert!(denied_diff <= 2, "too many denied attempts: {}", denied_diff);
}
