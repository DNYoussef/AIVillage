//! Simple performance test for mixnode pipeline
//! Target: 25,000 packets/second sustained throughput

use std::time::{Duration, Instant};

use bytes::Bytes;

use betanet_mixnode::pipeline::{PacketPipeline, PipelinePacket, PipelineBenchmark};

#[tokio::test]
async fn test_pipeline_performance_target() {
    println!("🚀 Mixnode Pipeline Performance Test");
    println!("===================================");
    println!("Target: 25,000 packets/second");

    // Test 1: Quick throughput check (5 seconds)
    println!("\n📊 Test 1: Quick Throughput Check (5 seconds)");
    let mut benchmark = PipelineBenchmark::new(4, 5000);
    let results = benchmark.run_throughput_test(5).await.unwrap();
    results.print_results();

    let meets_target = results.meets_target(25000.0);
    println!("🎯 Meets 25k pkt/s target: {}", if meets_target { "✅ YES" } else { "❌ NO" });

    // Test 2: Memory pool efficiency
    println!("\n📊 Test 2: Memory Pool Efficiency");
    let pool = betanet_mixnode::pipeline::MemoryPool::new(1000, 2048);

    let start = Instant::now();
    for _ in 0..10000 {
        let buf = pool.get_buffer(1024);
        pool.return_buffer(buf);
    }
    let elapsed = start.elapsed();

    let (allocated, reused) = pool.stats();
    let hit_rate = if allocated + reused > 0 {
        reused as f64 / (allocated + reused) as f64 * 100.0
    } else {
        0.0
    };

    println!("  10k allocations in {:.2}ms", elapsed.as_millis());
    println!("  Pool hit rate: {:.1}%", hit_rate);
    println!("  Allocated: {}, Reused: {}", allocated, reused);

    // Test 3: Basic pipeline throughput
    println!("\n📊 Test 3: Basic Pipeline Throughput");
    let mut pipeline = PacketPipeline::new(4);
    pipeline.start().await.unwrap();

    let start = Instant::now();
    let test_data = Bytes::from(vec![0u8; 1200]); // Typical packet size
    let num_packets = 10000;

    // Submit packets
    for _ in 0..num_packets {
        let packet = PipelinePacket::new(test_data.clone());
        if pipeline.submit_packet(packet).await.is_err() {
            break; // Pipeline full
        }
    }

    // Wait for processing
    tokio::time::sleep(Duration::from_millis(200)).await;

    let elapsed = start.elapsed();
    let processed = pipeline.get_processed_packets(num_packets);

    pipeline.stop().await.unwrap();

    let throughput = processed.len() as f64 / elapsed.as_secs_f64();
    println!("  Submitted: {} packets", num_packets);
    println!("  Processed: {} packets", processed.len());
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.0} pkt/s", throughput);

    // Test 4: Worker scaling analysis
    println!("\n📊 Test 4: Worker Scaling Analysis");
    for workers in [1, 2, 4, 6, 8] {
        let mut pipeline = PacketPipeline::new(workers);
        pipeline.start().await.unwrap();

        let start = Instant::now();
        let test_packets = 5000;

        for _ in 0..test_packets {
            let packet = PipelinePacket::new(test_data.clone());
            if pipeline.submit_packet(packet).await.is_err() {
                break;
            }
        }

        tokio::time::sleep(Duration::from_millis(100)).await;
        let elapsed = start.elapsed();
        let processed = pipeline.get_processed_packets(test_packets);
        pipeline.stop().await.unwrap();

        let throughput = processed.len() as f64 / elapsed.as_secs_f64();
        println!("  {} workers: {:.0} pkt/s", workers, throughput);
    }

    println!("\n🏁 Performance Testing Complete!");

    // Basic assertion - we should be able to process some packets
    assert!(results.packets_processed > 0, "Should process some packets");
    assert!(results.throughput_pps > 1000.0, "Should achieve at least 1k pkt/s");
}

#[tokio::test]
async fn test_pipeline_basic_functionality() {
    let mut pipeline = PacketPipeline::new(2);
    pipeline.start().await.unwrap();

    // Submit test packets
    let test_data = Bytes::from(vec![0u8; 1000]);
    for i in 0..10 {
        let packet = PipelinePacket::with_priority(test_data.clone(), i % 3);
        pipeline.submit_packet(packet).await.unwrap();
    }

    // Wait for processing
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Check results
    let processed = pipeline.get_processed_packets(100);
    let stats = pipeline.stats();

    println!("Processed {} packets", processed.len());
    println!("Pipeline processed: {}", stats.packets_processed.load(std::sync::atomic::Ordering::Relaxed));

    pipeline.stop().await.unwrap();

    assert!(processed.len() > 0, "Should process some packets");
}
