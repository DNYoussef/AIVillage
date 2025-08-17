// Criterion benchmarks for SCION packet encapsulation and AEAD performance
// Measures encryption/decryption throughput and anti-replay validation speed

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::runtime::Runtime;

// Import the modules we need to benchmark
use betanet_gateway::aead::{AeadManager, FrameType};
use betanet_gateway::anti_replay::AntiReplayManager;
use betanet_gateway::integrated_protection::IntegratedProtectionManager;
use betanet_gateway::config::{AeadConfig, AntiReplayConfig, GatewayConfig};
use betanet_gateway::metrics::MetricsCollector;

/// Create test configuration for benchmarks
fn create_bench_configs() -> (AeadConfig, AntiReplayConfig, Arc<MetricsCollector>, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("bench.db");

    let aead_config = AeadConfig {
        max_bytes_per_key: 1024 * 1024 * 1024, // 1GB for benchmarks
        max_time_per_key: Duration::from_secs(3600),
    };

    let anti_replay_config = AntiReplayConfig {
        db_path,
        window_size: 1024, // Larger window for benchmarks
        cleanup_ttl: Duration::from_secs(3600),
        cleanup_interval: Duration::from_secs(300),
        sync_interval: Duration::from_secs(60),
        max_sequence_age: Duration::from_secs(300),
    };

    let gateway_config = Arc::new(GatewayConfig::default());
    let metrics = Arc::new(MetricsCollector::new(gateway_config).unwrap());

    (aead_config, anti_replay_config, metrics, temp_dir)
}

/// Benchmark AEAD encryption performance
fn bench_aead_encrypt(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (aead_config, _, metrics, _temp_dir) = create_bench_configs();
    let master_key = [42u8; 32];

    let aead = Arc::new(AeadManager::new(aead_config, metrics, master_key));

    // Test different payload sizes
    let sizes = vec![64, 512, 1024, 4096, 16384, 65536]; // 64B to 64KB

    let mut group = c.benchmark_group("aead_encrypt");

    for size in sizes {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                let payload = vec![0u8; size];
                let aad = b"benchmark_aad";
                let peer_id = "bench_peer";

                b.to_async(&rt).iter(|| async {
                    let _encrypted = aead.encrypt_frame(
                        peer_id,
                        FrameType::ScionData,
                        black_box(&payload),
                        black_box(aad),
                    ).await.unwrap();
                });
            },
        );
    }
    group.finish();
}

/// Benchmark AEAD decryption performance
fn bench_aead_decrypt(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (aead_config, _, metrics, _temp_dir) = create_bench_configs();
    let master_key = [42u8; 32];

    let aead = Arc::new(AeadManager::new(aead_config, metrics, master_key));

    // Pre-encrypt frames for different sizes
    let sizes = vec![64, 512, 1024, 4096, 16384, 65536];
    let peer_id = "bench_peer";

    let mut encrypted_frames = Vec::new();

    rt.block_on(async {
        for size in &sizes {
            let payload = vec![0u8; *size];
            let aad = b"benchmark_aad";

            let frame = aead.encrypt_frame(
                peer_id,
                FrameType::ScionData,
                &payload,
                aad,
            ).await.unwrap();

            encrypted_frames.push((*size, frame));
        }
    });

    let mut group = c.benchmark_group("aead_decrypt");

    for (size, frame) in encrypted_frames {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &frame,
            |b, frame| {
                b.to_async(&rt).iter(|| async {
                    let _decrypted = aead.decrypt_frame(
                        peer_id,
                        black_box(frame),
                    ).await.unwrap();
                });
            },
        );
    }
    group.finish();
}

/// Benchmark anti-replay validation performance
fn bench_anti_replay_validation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (_, anti_replay_config, metrics, _temp_dir) = create_bench_configs();

    let anti_replay = rt.block_on(async {
        Arc::new(AntiReplayManager::new(anti_replay_config, metrics).await.unwrap())
    });

    let peer_id = "bench_peer";
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    // Warm up the manager with some sequences
    rt.block_on(async {
        for i in 1..=100 {
            anti_replay.validate_sequence(peer_id, i, timestamp, true).await;
        }
    });

    c.bench_function("anti_replay_validation", |b| {
        let mut sequence = 101u64;

        b.to_async(&rt).iter(|| async {
            sequence += 1;
            let _result = anti_replay.validate_sequence(
                peer_id,
                black_box(sequence),
                timestamp,
                true,
            ).await;
        });
    });
}

/// Benchmark RocksDB operations for anti-replay persistence
fn bench_rocksdb_ops(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (_, anti_replay_config, metrics, _temp_dir) = create_bench_configs();

    let anti_replay = rt.block_on(async {
        Arc::new(AntiReplayManager::new(anti_replay_config, metrics).await.unwrap())
    });

    // Test with multiple peers to stress the database
    let peers: Vec<String> = (0..10).map(|i| format!("peer_{}", i)).collect();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    c.bench_function("rocksdb_multi_peer_validation", |b| {
        let mut sequence = 1u64;
        let mut peer_index = 0;

        b.to_async(&rt).iter(|| async {
            let peer_id = &peers[peer_index % peers.len()];
            peer_index += 1;
            sequence += 1;

            let _result = anti_replay.validate_sequence(
                peer_id,
                black_box(sequence),
                timestamp,
                true,
            ).await;
        });
    });
}

/// Benchmark integrated protection (AEAD + anti-replay)
fn bench_integrated_protection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (aead_config, anti_replay_config, metrics, _temp_dir) = create_bench_configs();
    let master_key = [42u8; 32];

    let integrated = rt.block_on(async {
        Arc::new(IntegratedProtectionManager::new(
            aead_config,
            anti_replay_config,
            metrics,
            master_key,
        ).await.unwrap())
    });

    // Test different payload sizes
    let sizes = vec![512, 1024, 4096, 16384]; // Focus on realistic SCION packet sizes
    let peer_id = "bench_peer";

    let mut group = c.benchmark_group("integrated_protection");

    for size in sizes {
        group.throughput(Throughput::Bytes(size as u64));

        // Benchmark protect operation
        group.bench_with_input(
            BenchmarkId::new("protect", size),
            &size,
            |b, &size| {
                let payload = vec![0u8; size];

                b.to_async(&rt).iter(|| async {
                    let _protected = integrated.protect_packet(
                        peer_id,
                        black_box(&payload),
                        FrameType::ScionData,
                    ).await.unwrap();
                });
            },
        );

        // Benchmark unprotect operation
        let protected_frame = rt.block_on(async {
            let payload = vec![0u8; size];
            integrated.protect_packet(peer_id, &payload, FrameType::ScionData).await.unwrap()
        });

        group.bench_with_input(
            BenchmarkId::new("unprotect", size),
            &protected_frame,
            |b, frame| {
                b.to_async(&rt).iter(|| async {
                    let _unprotected = integrated.unprotect_packet(
                        peer_id,
                        black_box(frame),
                    ).await.unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark sustained throughput (simulating real workload)
fn bench_sustained_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (aead_config, anti_replay_config, metrics, _temp_dir) = create_bench_configs();
    let master_key = [42u8; 32];

    let integrated = rt.block_on(async {
        Arc::new(IntegratedProtectionManager::new(
            aead_config,
            anti_replay_config,
            metrics,
            master_key,
        ).await.unwrap())
    });

    // Simulate realistic SCION packet processing
    let packet_size = 1500; // Typical MTU size
    let payload = vec![0u8; packet_size];
    let peer_id = "throughput_peer";

    c.bench_function("sustained_throughput_1500b", |b| {
        b.to_async(&rt).iter(|| async {
            // Protect packet
            let protected = integrated.protect_packet(
                peer_id,
                black_box(&payload),
                FrameType::ScionData,
            ).await.unwrap();

            // Unprotect packet
            let _unprotected = integrated.unprotect_packet(
                peer_id,
                black_box(&protected),
            ).await.unwrap();
        });
    });
}

/// Benchmark memory usage and allocations
fn bench_memory_efficiency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (aead_config, anti_replay_config, metrics, _temp_dir) = create_bench_configs();
    let master_key = [42u8; 32];

    let integrated = rt.block_on(async {
        Arc::new(IntegratedProtectionManager::new(
            aead_config,
            anti_replay_config,
            metrics,
            master_key,
        ).await.unwrap())
    });

    let payload = vec![0u8; 4096]; // 4KB payload
    let peers: Vec<String> = (0..100).map(|i| format!("peer_{}", i)).collect();

    c.bench_function("multi_peer_memory_efficiency", |b| {
        let mut peer_index = 0;

        b.to_async(&rt).iter(|| async {
            let peer_id = &peers[peer_index % peers.len()];
            peer_index += 1;

            let protected = integrated.protect_packet(
                peer_id,
                black_box(&payload),
                FrameType::ScionData,
            ).await.unwrap();

            let _unprotected = integrated.unprotect_packet(
                peer_id,
                black_box(&protected),
            ).await.unwrap();
        });
    });
}

criterion_group!(
    benches,
    bench_aead_encrypt,
    bench_aead_decrypt,
    bench_anti_replay_validation,
    bench_rocksdb_ops,
    bench_integrated_protection,
    bench_sustained_throughput,
    bench_memory_efficiency
);

criterion_main!(benches);
