//! Mixnode performance benchmarks
//!
//! Target: 25,000 packets/second sustained throughput

use std::time::{Duration, Instant};

use betanet_mixnode::pipeline::{PacketPipeline, PipelinePacket, PipelineBenchmark};

/// Custom benchmark for mixnode-specific performance testing
async fn custom_mixnode_benchmark() {
    println!("🚀 Custom Mixnode Performance Test");
    println!("==================================");

    // Test 1: Quick throughput check
    println!("\n📊 Test 1: Quick Throughput Check (5 seconds)");
    let mut benchmark = PipelineBenchmark::new(4, 5000);
    let results = benchmark.run_throughput_test(5).await.unwrap();
    results.print_results();

    let meets_target = results.meets_target(25000.0);
    println!("🎯 Meets 25k pkt/s target: {}", if meets_target { "✅ YES" } else { "❌ NO" });

    // Test 2: Longer sustained test
    println!("\n📊 Test 2: Sustained Throughput (15 seconds)");
    let mut benchmark = PipelineBenchmark::new(4, 10000);
    let results = benchmark.run_throughput_test(15).await.unwrap();
    results.print_results();

    let meets_target = results.meets_target(25000.0);
    println!("🎯 Meets 25k pkt/s target: {}", if meets_target { "✅ YES" } else { "❌ NO" });

    // Test 3: Worker scaling test
    println!("\n📊 Test 3: Worker Scaling Analysis");
    for workers in [1, 2, 4, 6, 8] {
        let mut benchmark = PipelineBenchmark::new(workers, 2000);
        let results = benchmark.run_throughput_test(3).await.unwrap();
        println!("  {} workers: {:.0} pkt/s", workers, results.throughput_pps);
    }

    // Test 4: Memory efficiency test
    println!("\n📊 Test 4: Memory Pool Efficiency");
    let pool = betanet_mixnode::pipeline::MemoryPool::new(1000, 2048);

    let start = Instant::now();
    for _ in 0..10000 {
        let buf = pool.get_buffer(1024);
        pool.return_buffer(buf);
    }
    let elapsed = start.elapsed();

    let (allocated, reused) = pool.stats();
    let hit_rate = reused as f64 / (allocated + reused) as f64 * 100.0;

    println!("  10k allocations in {:.2}ms", elapsed.as_millis());
    println!("  Pool hit rate: {:.1}%", hit_rate);
    println!("  Allocated: {}, Reused: {}", allocated, reused);

    println!("\n🏁 Performance Testing Complete!");
}

/// Run the benchmark
#[tokio::main]
async fn main() {
    custom_mixnode_benchmark().await;
}
