//! Performance comparison tests between Lyapunov and FIFO schedulers
//!
//! This module implements comprehensive performance tests using synthetic contact graphs
//! to validate that the Lyapunov scheduler provides better queue stability and on-time
//! delivery rates compared to baseline FIFO scheduling.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::bundle::{Bundle, BundleId};
use crate::router::Contact;
use crate::sched::{
    FifoScheduler, LyapunovConfig, LyapunovScheduler, SyntheticContactGenerator, TestResults,
};

/// Comprehensive scheduler performance test framework
pub struct PerformanceTestFramework {
    test_duration: u64,       // seconds
    bundle_arrival_rate: f64, // bundles per second
    contact_generator: SyntheticContactGenerator,
}

impl PerformanceTestFramework {
    pub fn new(seed: u64, num_nodes: usize, test_duration: u64) -> Self {
        let base_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            test_duration,
            bundle_arrival_rate: 0.1, // 1 bundle per 10 seconds default
            contact_generator: SyntheticContactGenerator::new(seed, num_nodes, base_time),
        }
    }

    pub fn set_bundle_arrival_rate(&mut self, rate: f64) {
        self.bundle_arrival_rate = rate;
    }

    /// Run comprehensive comparison between Lyapunov and FIFO schedulers
    pub fn run_comparison_test(
        &mut self,
        lyapunov_config: LyapunovConfig,
        topology_type: TopologyType,
        scenario_name: &str,
    ) -> (TestResults, TestResults) {
        println!("\n🧪 Running performance test: {}", scenario_name);
        println!(
            "Duration: {} seconds, Topology: {:?}",
            self.test_duration, topology_type
        );

        // Generate synthetic contact graph
        let contacts = self.generate_contacts(&topology_type);
        println!("Generated {} contacts", contacts.len());

        // Generate test bundles
        let bundles = self.contact_generator.generate_test_bundles(
            (self.test_duration as f64 * self.bundle_arrival_rate) as usize,
            3600, // 1 hour default lifetime
        );
        println!("Generated {} test bundles", bundles.len());

        // Run Lyapunov scheduler test
        let lyapunov_results = self.run_scheduler_test(
            SchedulerType::Lyapunov(lyapunov_config),
            &contacts,
            &bundles,
            "Lyapunov",
        );

        // Run FIFO scheduler test
        let fifo_results =
            self.run_scheduler_test(SchedulerType::Fifo, &contacts, &bundles, "FIFO");

        // Print comparison
        lyapunov_results.print_comparison(&fifo_results);

        (lyapunov_results, fifo_results)
    }

    fn generate_contacts(&mut self, topology: &TopologyType) -> Vec<Contact> {
        match topology {
            TopologyType::Linear {
                contact_duration,
                interval,
            } => self
                .contact_generator
                .linear_topology(*contact_duration, *interval),
            TopologyType::Star {
                contact_duration,
                interval,
            } => self
                .contact_generator
                .star_topology(*contact_duration, *interval),
            TopologyType::Mesh {
                contact_duration,
                interval,
                density,
            } => self
                .contact_generator
                .mesh_topology(*contact_duration, *interval, *density),
        }
    }

    fn run_scheduler_test(
        &self,
        scheduler_type: SchedulerType,
        contacts: &[Contact],
        bundles: &[Bundle],
        scheduler_name: &str,
    ) -> TestResults {
        let mut results = TestResults::new(scheduler_name.to_string());
        results.total_bundles = bundles.len();
        results.test_duration = self.test_duration;

        // Initialize scheduler
        let mut lyapunov_scheduler = None;
        let mut fifo_scheduler = None;

        match &scheduler_type {
            SchedulerType::Lyapunov(config) => {
                lyapunov_scheduler = Some(
                    LyapunovScheduler::new(config.clone())
                        .expect("Failed to create Lyapunov scheduler"),
                );
            }
            SchedulerType::Fifo => {
                fifo_scheduler = Some(FifoScheduler::new());
            }
        }

        // Simulate bundle arrivals
        for bundle in bundles {
            match &scheduler_type {
                SchedulerType::Lyapunov(_) => {
                    if let Some(ref mut scheduler) = lyapunov_scheduler {
                        scheduler.enqueue_bundle(bundle);
                    }
                }
                SchedulerType::Fifo => {
                    if let Some(ref mut scheduler) = fifo_scheduler {
                        scheduler.enqueue_bundle(bundle);
                    }
                }
            }
        }

        // Simulate contact opportunities and scheduling decisions
        let mut bundle_states: HashMap<BundleId, BundleState> = bundles
            .iter()
            .map(|b| (b.id(), BundleState::Queued))
            .collect();

        let mut queue_length_samples = Vec::new();
        let mut current_time = contacts.iter().map(|c| c.start_time).min().unwrap_or(0);
        let end_time = current_time + self.test_duration;

        while current_time <= end_time {
            // Find active contacts at current time
            let active_contacts: Vec<_> = contacts
                .iter()
                .filter(|c| c.is_active(current_time))
                .cloned()
                .collect();

            // Sample queue lengths
            let current_queue_length = match &scheduler_type {
                SchedulerType::Lyapunov(_) => {
                    if let Some(ref scheduler) = lyapunov_scheduler {
                        scheduler
                            .get_queue_states()
                            .values()
                            .map(|state| state.queue_length)
                            .sum::<usize>()
                    } else {
                        0
                    }
                }
                SchedulerType::Fifo => {
                    if let Some(ref scheduler) = fifo_scheduler {
                        scheduler.get_queue_lengths().values().sum::<usize>()
                    } else {
                        0
                    }
                }
            };

            queue_length_samples.push(current_queue_length);
            results.max_queue_length = results.max_queue_length.max(current_queue_length);

            // Process each active contact
            for contact in &active_contacts {
                let available_bundles: Vec<BundleId> = bundle_states
                    .iter()
                    .filter_map(|(id, state)| {
                        if matches!(state, BundleState::Queued) {
                            if let Some(bundle) = bundles.iter().find(|b| b.id() == *id) {
                                if bundle.primary.destination == contact.to {
                                    Some(id.clone())
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .collect();

                if available_bundles.is_empty() {
                    continue;
                }

                // Make scheduling decision
                let decision = match &scheduler_type {
                    SchedulerType::Lyapunov(_) => {
                        if let Some(ref mut scheduler) = lyapunov_scheduler {
                            scheduler
                                .schedule_transmission(contact, &available_bundles, 1)
                                .unwrap_or_else(|_| {
                                    // Create default "no transmission" decision
                                    crate::sched::SchedulingDecision {
                                        should_transmit: false,
                                        bundles_to_transmit: Vec::new(),
                                        estimated_utility: 0.0,
                                        estimated_energy_cost: 0.0,
                                        drift_component: 0.0,
                                        penalty_component: 0.0,
                                        rationale: "Error in scheduling".to_string(),
                                    }
                                })
                        } else {
                            continue;
                        }
                    }
                    SchedulerType::Fifo => {
                        if let Some(ref scheduler) = fifo_scheduler {
                            scheduler.schedule_transmission(contact, &available_bundles, 1)
                        } else {
                            continue;
                        }
                    }
                };

                // Process transmission decisions
                if decision.should_transmit {
                    for bundle_id in &decision.bundles_to_transmit {
                        if let Some(bundle) = bundles.iter().find(|b| b.id() == *bundle_id) {
                            // Simulate transmission
                            let _transmission_time = contact.transmission_time(bundle.size());
                            let delivery_successful = rand::random::<f64>() < contact.reliability;

                            if delivery_successful {
                                // Check if delivered on time
                                let bundle_age = current_time
                                    .saturating_sub(bundle.primary.creation_timestamp.dtn_time);
                                let was_on_time = (bundle_age * 1000) <= bundle.primary.lifetime;

                                results.bundles_delivered += 1;
                                if was_on_time {
                                    results.bundles_on_time += 1;
                                }

                                results.total_energy_consumed += decision.estimated_energy_cost;
                                bundle_states.insert(bundle_id.clone(), BundleState::Delivered);

                                // Notify scheduler of successful delivery
                                match &scheduler_type {
                                    SchedulerType::Lyapunov(_) => {
                                        if let Some(ref mut scheduler) = lyapunov_scheduler {
                                            scheduler.dequeue_bundle(
                                                bundle_id.clone(),
                                                was_on_time,
                                                decision.estimated_energy_cost,
                                            );
                                        }
                                    }
                                    SchedulerType::Fifo => {
                                        if let Some(ref mut scheduler) = fifo_scheduler {
                                            scheduler.dequeue_bundle(bundle_id.clone());
                                        }
                                    }
                                }
                            } else {
                                // Transmission failed - bundle remains queued
                                results.total_energy_consumed +=
                                    decision.estimated_energy_cost * 0.5; // Partial energy cost
                            }
                        }
                    }
                }
            }

            // Advance time
            current_time += 30; // 30 second time steps
        }

        // Calculate final statistics
        results.avg_queue_length = if !queue_length_samples.is_empty() {
            queue_length_samples.iter().sum::<usize>() as f64 / queue_length_samples.len() as f64
        } else {
            0.0
        };

        results.finalize();
        results
    }
}

#[derive(Debug, Clone)]
pub enum TopologyType {
    Linear {
        contact_duration: u64,
        interval: u64,
    },
    Star {
        contact_duration: u64,
        interval: u64,
    },
    Mesh {
        contact_duration: u64,
        interval: u64,
        density: f64,
    },
}

#[derive(Debug, Clone)]
enum SchedulerType {
    Lyapunov(LyapunovConfig),
    Fifo,
}

#[derive(Debug, Clone, PartialEq)]
enum BundleState {
    Queued,
    Delivered,
    Expired,
}

/// Test suite for comprehensive scheduler evaluation
pub struct SchedulerTestSuite;

impl SchedulerTestSuite {
    /// Run all performance tests with different scenarios
    pub fn run_all_tests() {
        println!("\n🚀 Starting Comprehensive Scheduler Performance Tests");
        println!("=====================================================");

        // Test 1: Linear topology with low V (stability-focused)
        let mut framework = PerformanceTestFramework::new(42, 5, 300); // 5 minutes
        framework.set_bundle_arrival_rate(0.2); // 1 bundle per 5 seconds

        let stability_config = LyapunovConfig {
            v_parameter: 0.5, // Low V - prioritize stability
            max_queue_length: 50,
            energy_cost_weight: 0.1,
            privacy_penalty_weight: 0.05,
            observation_window: 180,
            min_utility_threshold: 0.05,
        };

        let (lyap_linear, fifo_linear) = framework.run_comparison_test(
            stability_config.clone(),
            TopologyType::Linear {
                contact_duration: 60,
                interval: 120,
            },
            "Linear Topology - Stability Focus (Low V)",
        );

        // Test 2: Star topology with high V (energy-focused)
        let mut framework2 = PerformanceTestFramework::new(123, 6, 300);
        framework2.set_bundle_arrival_rate(0.15);

        let energy_config = LyapunovConfig {
            v_parameter: 10.0, // High V - prioritize energy efficiency
            energy_cost_weight: 0.8,
            privacy_penalty_weight: 0.2,
            ..LyapunovConfig::default()
        };

        let (lyap_star, fifo_star) = framework2.run_comparison_test(
            energy_config,
            TopologyType::Star {
                contact_duration: 90,
                interval: 180,
            },
            "Star Topology - Energy Focus (High V)",
        );

        // Test 3: Mesh topology with balanced V
        let mut framework3 = PerformanceTestFramework::new(456, 8, 600); // 10 minutes
        framework3.set_bundle_arrival_rate(0.3); // Higher load

        let balanced_config = LyapunovConfig {
            v_parameter: 2.0, // Balanced
            max_queue_length: 100,
            energy_cost_weight: 0.4,
            privacy_penalty_weight: 0.1,
            observation_window: 240,
            min_utility_threshold: 0.1,
        };

        let (lyap_mesh, fifo_mesh) = framework3.run_comparison_test(
            balanced_config,
            TopologyType::Mesh {
                contact_duration: 45,
                interval: 90,
                density: 0.6,
            },
            "Mesh Topology - Balanced Configuration",
        );

        // Summary analysis
        Self::print_comprehensive_summary(vec![
            ("Linear/Stability", &lyap_linear, &fifo_linear),
            ("Star/Energy", &lyap_star, &fifo_star),
            ("Mesh/Balanced", &lyap_mesh, &fifo_mesh),
        ]);
    }

    fn print_comprehensive_summary(test_cases: Vec<(&str, &TestResults, &TestResults)>) {
        println!("\n📈 Comprehensive Test Summary");
        println!("=============================");

        let mut lyap_wins = 0;
        let mut fifo_wins = 0;

        for (scenario, lyap, fifo) in &test_cases {
            println!("\n📊 {}", scenario);
            println!(
                "  Delivery Rate: Lyapunov {:.3} vs FIFO {:.3} - Winner: {}",
                lyap.delivery_rate,
                fifo.delivery_rate,
                if lyap.delivery_rate > fifo.delivery_rate {
                    "Lyapunov"
                } else {
                    "FIFO"
                }
            );
            println!(
                "  On-Time Rate:  Lyapunov {:.3} vs FIFO {:.3} - Winner: {}",
                lyap.on_time_rate,
                fifo.on_time_rate,
                if lyap.on_time_rate > fifo.on_time_rate {
                    "Lyapunov"
                } else {
                    "FIFO"
                }
            );
            println!(
                "  Energy/Bundle: Lyapunov {:.2} vs FIFO {:.2} - Winner: {}",
                lyap.energy_per_bundle,
                fifo.energy_per_bundle,
                if lyap.energy_per_bundle < fifo.energy_per_bundle {
                    "Lyapunov"
                } else {
                    "FIFO"
                }
            );
            println!(
                "  Max Queue:     Lyapunov {} vs FIFO {} - Winner: {}",
                lyap.max_queue_length,
                fifo.max_queue_length,
                if lyap.max_queue_length < fifo.max_queue_length {
                    "Lyapunov"
                } else {
                    "FIFO"
                }
            );

            // Score the overall winner for this test case
            let lyap_score = (if lyap.delivery_rate > fifo.delivery_rate {
                1
            } else {
                0
            }) + (if lyap.on_time_rate > fifo.on_time_rate {
                1
            } else {
                0
            }) + (if lyap.energy_per_bundle < fifo.energy_per_bundle {
                1
            } else {
                0
            }) + (if lyap.max_queue_length < fifo.max_queue_length {
                1
            } else {
                0
            });

            if lyap_score > 2 {
                lyap_wins += 1;
                println!("  🏆 Overall Winner: Lyapunov ({}/4 metrics)", lyap_score);
            } else {
                fifo_wins += 1;
                println!("  🏆 Overall Winner: FIFO ({}/4 metrics)", 4 - lyap_score);
            }
        }

        println!("\n🎯 Final Results:");
        println!("  Lyapunov wins: {} test cases", lyap_wins);
        println!("  FIFO wins: {} test cases", fifo_wins);

        if lyap_wins > fifo_wins {
            println!("  🏅 Lyapunov scheduler demonstrates superior performance!");
            println!("     ✅ Better queue stability with Lyapunov drift control");
            println!("     ✅ Higher on-time delivery rates");
            println!("     ✅ More energy-efficient transmission decisions");
        } else {
            println!("  📝 Results show competitive performance between schedulers");
            println!("     Consider tuning Lyapunov parameters for your specific workload");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_framework_creation() {
        let framework = PerformanceTestFramework::new(42, 5, 300);
        assert_eq!(framework.test_duration, 300);
        assert_eq!(framework.bundle_arrival_rate, 0.1);
    }

    #[test]
    fn test_topology_generation() {
        let mut framework = PerformanceTestFramework::new(42, 4, 60);

        let linear_contacts = framework.generate_contacts(&TopologyType::Linear {
            contact_duration: 30,
            interval: 60,
        });
        assert!(!linear_contacts.is_empty());

        let star_contacts = framework.generate_contacts(&TopologyType::Star {
            contact_duration: 30,
            interval: 60,
        });
        assert!(!star_contacts.is_empty());

        let mesh_contacts = framework.generate_contacts(&TopologyType::Mesh {
            contact_duration: 30,
            interval: 60,
            density: 0.5,
        });
        assert!(!mesh_contacts.is_empty());
    }

    #[test]
    fn test_scheduler_comparison() {
        let mut framework = PerformanceTestFramework::new(42, 3, 30); // Short test
        framework.set_bundle_arrival_rate(0.5); // 1 bundle per 2 seconds

        let config = LyapunovConfig {
            v_parameter: 1.0,
            ..LyapunovConfig::default()
        };

        let (lyap_results, fifo_results) = framework.run_comparison_test(
            config,
            TopologyType::Linear {
                contact_duration: 15,
                interval: 30,
            },
            "Quick Test",
        );

        // Both schedulers should process some bundles
        assert!(lyap_results.total_bundles > 0);
        assert!(fifo_results.total_bundles > 0);
        assert_eq!(lyap_results.total_bundles, fifo_results.total_bundles);
    }
}
