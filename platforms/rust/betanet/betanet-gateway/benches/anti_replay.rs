// Criterion benchmarks for anti-replay protection system
// Measures sequence validation, sliding window operations, and RocksDB persistence

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::runtime::Runtime;

use betanet_gateway::anti_replay::AntiReplayManager;
use betanet_gateway::config::{AntiReplayConfig, GatewayConfig};
use betanet_gateway::metrics::MetricsCollector;

/// Create test anti-replay manager for benchmarks
async fn create_bench_manager() -> (Arc<AntiReplayManager>, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("bench_replay.db");

    let config = AntiReplayConfig {
        db_path,
        window_size: 1024,
        cleanup_ttl: Duration::from_secs(3600),
        cleanup_interval: Duration::from_secs(300),
        sync_interval: Duration::from_secs(60),
        max_sequence_age: Duration::from_secs(300),
    };

    let gateway_config = Arc::new(GatewayConfig::default());
    let metrics = Arc::new(MetricsCollector::new(gateway_config).unwrap());

    let manager = Arc::new(AntiReplayManager::new(config, metrics).await.unwrap());
    (manager, temp_dir)
}

/// Benchmark sequence validation for in-order packets
fn bench_sequential_validation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (manager, _temp_dir) = rt.block_on(create_bench_manager());

    let peer_id = "sequential_peer";
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    c.bench_function("sequential_validation", |b| {
        let mut sequence = 1u64;

        b.to_async(&rt).iter(|| async {
            let result = manager.validate_sequence(
                peer_id,
                black_box(sequence),
                timestamp,
                true,
            ).await;
            sequence += 1;
            black_box(result);
        });
    });
}

/// Benchmark sequence validation for out-of-order packets
fn bench_random_validation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (manager, _temp_dir) = rt.block_on(create_bench_manager());

    let peer_id = "random_peer";
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    // Pre-populate with some sequences to create realistic window state
    rt.block_on(async {
        for i in (1..100).step_by(2) {  // Only odd numbers, leaving gaps
            manager.validate_sequence(peer_id, i, timestamp, true).await;
        }
    });

    c.bench_function("random_validation", |b| {
        let sequences = [2, 50, 4, 98, 6, 24, 8, 76, 10, 42]; // Fill gaps randomly
        let mut seq_index = 0;

        b.to_async(&rt).iter(|| async {
            let sequence = sequences[seq_index % sequences.len()];
            seq_index += 1;

            let result = manager.validate_sequence(
                peer_id,
                black_box(sequence),
                timestamp,
                true,
            ).await;
            black_box(result);
        });
    });
}

/// Benchmark sliding window operations
fn bench_window_sliding(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (manager, _temp_dir) = rt.block_on(create_bench_manager());

    let peer_id = "sliding_peer";
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    // Initialize with base sequence
    rt.block_on(async {
        manager.validate_sequence(peer_id, 1000, timestamp, true).await;
    });

    c.bench_function("window_sliding", |b| {
        let mut sequence = 1100u64; // Start beyond window to force sliding

        b.to_async(&rt).iter(|| async {
            sequence += 100; // Large jumps to trigger sliding
            let result = manager.validate_sequence(
                peer_id,
                black_box(sequence),
                timestamp,
                true,
            ).await;
            black_box(result);
        });
    });
}

/// Benchmark replay detection
fn bench_replay_detection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (manager, _temp_dir) = rt.block_on(create_bench_manager());

    let peer_id = "replay_peer";
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    // Pre-populate window with sequences
    let valid_sequences: Vec<u64> = (1..=100).collect();
    rt.block_on(async {
        for seq in &valid_sequences {
            manager.validate_sequence(peer_id, *seq, timestamp, true).await;
        }
    });

    c.bench_function("replay_detection", |b| {
        let mut seq_index = 0;

        b.to_async(&rt).iter(|| async {
            let sequence = valid_sequences[seq_index % valid_sequences.len()];
            seq_index += 1;

            // This should be detected as a replay
            let result = manager.validate_sequence(
                peer_id,
                black_box(sequence),
                timestamp,
                false, // Don't update window
            ).await;
            black_box(result);
        });
    });
}

/// Benchmark multi-peer validation
fn bench_multi_peer_validation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (manager, _temp_dir) = rt.block_on(create_bench_manager());

    let peer_count = 100;
    let peers: Vec<String> = (0..peer_count).map(|i| format!("peer_{}", i)).collect();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    // Initialize each peer with some sequences
    rt.block_on(async {
        for peer in &peers {
            for seq in 1..=10 {
                manager.validate_sequence(peer, seq, timestamp, true).await;
            }
        }
    });

    let mut group = c.benchmark_group("multi_peer_validation");

    for peer_count in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(peer_count),
            &peer_count,
            |b, &peer_count| {
                let mut peer_index = 0;
                let mut sequence = 11u64;

                b.to_async(&rt).iter(|| async {
                    let peer_id = &peers[peer_index % peer_count];
                    peer_index += 1;
                    sequence += 1;

                    let result = manager.validate_sequence(
                        peer_id,
                        black_box(sequence),
                        timestamp,
                        true,
                    ).await;
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark RocksDB persistence operations
fn bench_database_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (manager, _temp_dir) = rt.block_on(create_bench_manager());

    let peer_id = "db_peer";
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    c.bench_function("database_write_operations", |b| {
        let mut sequence = 1u64;

        b.to_async(&rt).iter(|| async {
            sequence += 1;

            // This will trigger database write operations
            let result = manager.validate_sequence(
                &format!("{}_{}", peer_id, sequence), // Use different peer IDs to force writes
                black_box(sequence),
                timestamp,
                true,
            ).await;
            black_box(result);
        });
    });
}

/// Benchmark validation under high load
fn bench_high_load_validation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (manager, _temp_dir) = rt.block_on(create_bench_manager());

    let peers: Vec<String> = (0..50).map(|i| format!("load_peer_{}", i)).collect();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    // Create realistic load with mixed validation patterns
    c.bench_function("high_load_mixed_validation", |b| {
        let mut peer_index = 0;
        let mut sequence = 1u64;
        let patterns = [1, 1, 2, 1, 3, 1, 1, 1]; // Mostly sequential with some gaps
        let mut pattern_index = 0;

        b.to_async(&rt).iter(|| async {
            let peer_id = &peers[peer_index % peers.len()];
            peer_index += 1;

            // Apply different sequence patterns
            sequence += patterns[pattern_index % patterns.len()];
            pattern_index += 1;

            let result = manager.validate_sequence(
                peer_id,
                black_box(sequence),
                timestamp,
                true,
            ).await;
            black_box(result);
        });
    });
}

/// Benchmark validation with expired sequences
fn bench_expired_validation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (manager, _temp_dir) = rt.block_on(create_bench_manager());

    let peer_id = "expired_peer";
    let current_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    // Use old timestamps to test expiration
    let old_timestamps = [
        current_time - 600_000_000_000, // 10 minutes old
        current_time - 300_000_000_000, // 5 minutes old
        current_time - 120_000_000_000, // 2 minutes old
        current_time - 60_000_000_000,  // 1 minute old
    ];

    c.bench_function("expired_sequence_validation", |b| {
        let mut sequence = 1u64;
        let mut timestamp_index = 0;

        b.to_async(&rt).iter(|| async {
            let timestamp = old_timestamps[timestamp_index % old_timestamps.len()];
            timestamp_index += 1;
            sequence += 1;

            let result = manager.validate_sequence(
                peer_id,
                black_box(sequence),
                timestamp,
                true,
            ).await;
            black_box(result);
        });
    });
}

/// Benchmark throughput with target of 500k validations per minute
fn bench_throughput_target(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (manager, _temp_dir) = rt.block_on(create_bench_manager());

    let peer_id = "throughput_peer";
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    // Target: 500k validations per minute = ~8,333 per second
    c.bench_function("throughput_target_500k_per_min", |b| {
        let mut sequence = 1u64;

        b.iter_batched(
            || {
                sequence += 1;
                sequence
            },
            |seq| {
                rt.block_on(async {
                    let result = manager.validate_sequence(
                        peer_id,
                        black_box(seq),
                        timestamp,
                        true,
                    ).await;
                    black_box(result);
                })
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    benches,
    bench_sequential_validation,
    bench_random_validation,
    bench_window_sliding,
    bench_replay_detection,
    bench_multi_peer_validation,
    bench_database_operations,
    bench_high_load_validation,
    bench_expired_validation,
    bench_throughput_target
);

criterion_main!(benches);
