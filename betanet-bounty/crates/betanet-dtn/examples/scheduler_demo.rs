//! DTN Scheduler Demonstration
//!
//! This example demonstrates the Lyapunov scheduler in action, showing how it makes
//! transmission decisions based on queue stability and energy constraints.

use betanet_dtn::sched::{
    LyapunovScheduler, LyapunovConfig,
    SyntheticContactGenerator, PerformanceTestFramework
};

fn main() {
    println!("🔬 DTN Lyapunov Scheduler Demonstration");
    println!("=========================================");
    println!();

    // Demonstrate basic scheduler functionality
    basic_scheduler_demo();

    println!();

    // Demonstrate synthetic contact generation
    synthetic_contact_demo();

    println!();

    // Run quick performance comparison
    quick_performance_demo();

    println!();
    println!("✅ Scheduler demonstration complete!");
    println!("📊 Run 'cargo test --release sched::performance_tests' for comprehensive benchmarks");
}

fn basic_scheduler_demo() {
    println!("🧪 Basic Scheduler Demo");
    println!("{}", "-".repeat(30));

    // Create Lyapunov scheduler with different V parameters
    let configs = vec![
        ("Stability-focused (V=0.1)", LyapunovConfig { v_parameter: 0.1, ..Default::default() }),
        ("Balanced (V=1.0)", LyapunovConfig { v_parameter: 1.0, ..Default::default() }),
        ("Energy-focused (V=10.0)", LyapunovConfig { v_parameter: 10.0, ..Default::default() }),
    ];

    for (name, config) in configs {
        println!("📋 {}: V = {:.1}", name, config.v_parameter);

        let scheduler_result = LyapunovScheduler::new(config);
        match scheduler_result {
            Ok(scheduler) => {
                println!("   ✅ Created scheduler with V parameter = {:.1}", scheduler.get_config().v_parameter);
                println!("   📊 Max queue length: {}", scheduler.get_config().max_queue_length);
                println!("   ⚡ Energy weight: {:.2}", scheduler.get_config().energy_cost_weight);
            }
            Err(e) => {
                println!("   ❌ Failed to create scheduler: {}", e);
            }
        }
        println!();
    }
}

fn synthetic_contact_demo() {
    println!("🌐 Synthetic Contact Generation Demo");
    println!("{}", "-".repeat(35));

    let mut generator = SyntheticContactGenerator::new(42, 4, 1000);

    // Generate different topologies
    let topologies = vec![
        ("Linear", "Sequential node connections"),
        ("Star", "Hub-and-spoke pattern"),
        ("Mesh", "Dense interconnected network"),
    ];

    for (name, description) in topologies {
        println!("🔗 {} Topology: {}", name, description);

        let contacts = match name {
            "Linear" => generator.linear_topology(60, 120),
            "Star" => generator.star_topology(60, 120),
            "Mesh" => generator.mesh_topology(60, 120, 0.6),
            _ => Vec::new(),
        };

        println!("   📊 Generated {} contacts", contacts.len());

        if !contacts.is_empty() {
            let avg_energy = contacts.iter().map(|c| c.energy_cost).sum::<f64>() / contacts.len() as f64;
            let max_energy = contacts.iter().map(|c| c.energy_cost).fold(0.0, f64::max);
            println!("   ⚡ Avg energy cost: {:.2}, Max: {:.2}", avg_energy, max_energy);
        }
        println!();
    }

    // Generate test bundles
    let bundles = generator.generate_test_bundles(10, 3600);
    println!("📦 Generated {} test bundles", bundles.len());

    if !bundles.is_empty() {
        let avg_size = bundles.iter().map(|b| b.size()).sum::<usize>() / bundles.len();
        let avg_lifetime = bundles.iter().map(|b| b.primary.lifetime).sum::<u64>() / bundles.len() as u64;
        println!("   📏 Avg bundle size: {} bytes", avg_size);
        println!("   ⏰ Avg lifetime: {} ms", avg_lifetime);
    }
}

fn quick_performance_demo() {
    println!("⚡ Quick Performance Comparison");
    println!("{}", "-".repeat(35));

    // Create a small-scale performance test
    let mut framework = PerformanceTestFramework::new(123, 3, 60); // 1 minute test
    framework.set_bundle_arrival_rate(0.5); // 1 bundle per 2 seconds

    let config = LyapunovConfig {
        v_parameter: 1.0,
        max_queue_length: 20,
        energy_cost_weight: 0.3,
        privacy_penalty_weight: 0.1,
        observation_window: 60,
        min_utility_threshold: 0.05,
    };

    println!("🧮 Running quick comparison test...");
    println!("   Duration: 1 minute");
    println!("   Bundle rate: 0.5/second");
    println!("   Topology: Linear (3 nodes)");
    println!();

    let (lyap_results, fifo_results) = framework.run_comparison_test(
        config,
        betanet_dtn::sched::TopologyType::Linear {
            contact_duration: 20,
            interval: 40
        },
        "Quick Demo Test",
    );

    // Print simplified results
    println!("🏆 Results Summary:");
    println!("   Lyapunov - Delivery: {:.1}%, Energy/Bundle: {:.2}",
             lyap_results.delivery_rate * 100.0, lyap_results.energy_per_bundle);
    println!("   FIFO     - Delivery: {:.1}%, Energy/Bundle: {:.2}",
             fifo_results.delivery_rate * 100.0, fifo_results.energy_per_bundle);

    if lyap_results.delivery_rate > fifo_results.delivery_rate {
        println!("   ✅ Lyapunov shows {:.1}% better delivery rate!",
                 (lyap_results.delivery_rate - fifo_results.delivery_rate) * 100.0);
    }

    if lyap_results.energy_per_bundle < fifo_results.energy_per_bundle {
        println!("   ⚡ Lyapunov is {:.1}% more energy efficient!",
                 (1.0 - lyap_results.energy_per_bundle / fifo_results.energy_per_bundle) * 100.0);
    }
}
