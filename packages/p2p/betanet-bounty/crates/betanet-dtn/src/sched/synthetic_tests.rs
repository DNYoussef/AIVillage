//! Synthetic contact graph tests for Lyapunov scheduler validation
//!
//! This module provides comprehensive testing of the Lyapunov scheduler against
//! synthetic contact graphs to validate queue stability, on-time delivery rates,
//! and energy efficiency compared to baseline FIFO scheduling.

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::bundle::{Bundle, BundleId, EndpointId};
use crate::router::Contact;
use crate::sched::SchedulingDecision;

/// Synthetic contact graph generator for testing
pub struct SyntheticContactGenerator {
    rng: ChaCha8Rng,
    nodes: Vec<EndpointId>,
    base_time: u64,
}

impl SyntheticContactGenerator {
    /// Create a new generator with a fixed seed for reproducible tests
    pub fn new(seed: u64, num_nodes: usize, base_time: u64) -> Self {
        let mut nodes = Vec::new();
        for i in 0..num_nodes {
            nodes.push(EndpointId::node(&format!("node{}", i)));
        }

        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            nodes,
            base_time,
        }
    }

    /// Generate a linear topology: node0 -> node1 -> node2 -> ... -> nodeN
    pub fn linear_topology(&mut self, contact_duration: u64, interval: u64) -> Vec<Contact> {
        let mut contacts = Vec::new();
        let mut time = self.base_time;

        for _ in 0..10 {
            // Generate 10 rounds of contacts
            for i in 0..(self.nodes.len() - 1) {
                let contact = Contact {
                    from: self.nodes[i].clone(),
                    to: self.nodes[i + 1].clone(),
                    start_time: time,
                    end_time: time + contact_duration,
                    data_rate: 1_000_000, // 1 Mbps
                    latency: Duration::from_millis(50),
                    reliability: 0.95,
                    energy_cost: 1.0 + self.rng.gen::<f64>() * 0.5, // 1.0-1.5 cost
                };
                contacts.push(contact);
            }
            time += interval;
        }

        contacts
    }

    /// Generate a star topology: all nodes connect to central hub (node0)
    pub fn star_topology(&mut self, contact_duration: u64, interval: u64) -> Vec<Contact> {
        let mut contacts = Vec::new();
        let mut time = self.base_time;
        let hub = self.nodes[0].clone();

        for _ in 0..10 {
            // Generate 10 rounds of contacts
            for i in 1..self.nodes.len() {
                // Bidirectional connections to hub
                let to_hub = Contact {
                    from: self.nodes[i].clone(),
                    to: hub.clone(),
                    start_time: time,
                    end_time: time + contact_duration,
                    data_rate: 2_000_000, // 2 Mbps to hub
                    latency: Duration::from_millis(25),
                    reliability: 0.98,
                    energy_cost: 0.8 + self.rng.gen::<f64>() * 0.4, // 0.8-1.2 cost
                };

                let from_hub = Contact {
                    from: hub.clone(),
                    to: self.nodes[i].clone(),
                    start_time: time + contact_duration / 4, // Slight offset
                    end_time: time + contact_duration + contact_duration / 4,
                    data_rate: 2_000_000, // 2 Mbps from hub
                    latency: Duration::from_millis(25),
                    reliability: 0.98,
                    energy_cost: 0.8 + self.rng.gen::<f64>() * 0.4,
                };

                contacts.push(to_hub);
                contacts.push(from_hub);
            }
            time += interval;
        }

        contacts
    }

    /// Generate a mesh topology with intermittent connectivity
    pub fn mesh_topology(
        &mut self,
        contact_duration: u64,
        interval: u64,
        density: f64,
    ) -> Vec<Contact> {
        let mut contacts = Vec::new();
        let mut time = self.base_time;

        for _ in 0..10 {
            // Generate 10 rounds of contacts
            for i in 0..self.nodes.len() {
                for j in (i + 1)..self.nodes.len() {
                    // Only create contact with given probability (density)
                    if self.rng.gen::<f64>() < density {
                        let high_energy = self.rng.gen::<f64>() < 0.3; // 30% chance of high energy cost
                        let energy_cost = if high_energy {
                            3.0 + self.rng.gen::<f64>() * 2.0 // 3.0-5.0 for expensive links
                        } else {
                            0.5 + self.rng.gen::<f64>() * 1.0 // 0.5-1.5 for cheap links
                        };

                        let contact = Contact {
                            from: self.nodes[i].clone(),
                            to: self.nodes[j].clone(),
                            start_time: time,
                            end_time: time + contact_duration,
                            data_rate: if high_energy { 500_000 } else { 1_500_000 }, // Lower rate for expensive links
                            latency: Duration::from_millis(if high_energy { 100 } else { 30 }),
                            reliability: if high_energy { 0.85 } else { 0.95 },
                            energy_cost,
                        };
                        contacts.push(contact);
                    }
                }
            }
            time += interval;
        }

        contacts
    }

    /// Generate bundles with various priorities and lifetimes
    pub fn generate_test_bundles(&mut self, count: usize, base_lifetime: u64) -> Vec<Bundle> {
        let mut bundles = Vec::new();

        for _i in 0..count {
            let src_idx = self.rng.gen_range(0..self.nodes.len());
            let mut dest_idx = self.rng.gen_range(0..self.nodes.len());
            while dest_idx == src_idx {
                dest_idx = self.rng.gen_range(0..self.nodes.len());
            }

            let priority_multiplier = match self.rng.gen_range(0..3) {
                0 => 0.5, // Low priority - longer lifetime
                1 => 1.0, // Normal priority
                2 => 2.0, // High priority - shorter lifetime but urgent
                _ => 1.0,
            };

            let lifetime = (base_lifetime as f64 * priority_multiplier) as u64;
            let payload_size = self.rng.gen_range(100..10000); // 100 bytes to 10KB
            let payload = Bytes::from(vec![0u8; payload_size]);

            let bundle = Bundle::new(
                self.nodes[dest_idx].clone(),
                self.nodes[src_idx].clone(),
                payload,
                lifetime,
            );

            bundles.push(bundle);
        }

        bundles
    }
}

/// FIFO (First-In-First-Out) scheduler for baseline comparison
pub struct FifoScheduler {
    bundle_queues: HashMap<EndpointId, Vec<BundleId>>,
    bundle_metadata: HashMap<BundleId, BundleMetadata>,
}

#[derive(Debug, Clone)]
struct BundleMetadata {
    id: BundleId,
    destination: EndpointId,
    arrival_time: u64,
    size: usize,
}

impl Default for FifoScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl FifoScheduler {
    pub fn new() -> Self {
        Self {
            bundle_queues: HashMap::new(),
            bundle_metadata: HashMap::new(),
        }
    }

    pub fn enqueue_bundle(&mut self, bundle: &Bundle) {
        let bundle_id = bundle.id();
        let destination = bundle.primary.destination.clone();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let metadata = BundleMetadata {
            id: bundle_id.clone(),
            destination: destination.clone(),
            arrival_time: now,
            size: bundle.size(),
        };

        self.bundle_metadata.insert(bundle_id.clone(), metadata);

        let queue = self.bundle_queues.entry(destination).or_default();
        queue.push(bundle_id);
    }

    pub fn dequeue_bundle(&mut self, bundle_id: BundleId) {
        self.bundle_metadata.remove(&bundle_id);

        // Remove from all queues (inefficient but simple for testing)
        for queue in self.bundle_queues.values_mut() {
            queue.retain(|id| *id != bundle_id);
        }
    }

    pub fn schedule_transmission(
        &self,
        contact: &Contact,
        available_bundles: &[BundleId],
        max_transmissions: usize,
    ) -> SchedulingDecision {
        // FIFO: always transmit oldest bundles first, regardless of queue state or energy
        let destination = &contact.to;

        let mut candidate_bundles = Vec::new();
        if let Some(queue) = self.bundle_queues.get(destination) {
            for bundle_id in queue {
                if available_bundles.contains(bundle_id) {
                    if let Some(metadata) = self.bundle_metadata.get(bundle_id) {
                        candidate_bundles.push((bundle_id.clone(), metadata.arrival_time));
                    }
                }
            }
        }

        // Sort by arrival time (FIFO order)
        candidate_bundles.sort_by(|a, b| a.1.cmp(&b.1));

        let bundles_to_transmit: Vec<BundleId> = candidate_bundles
            .into_iter()
            .take(max_transmissions)
            .map(|(id, _)| id)
            .collect();

        let should_transmit = !bundles_to_transmit.is_empty();

        SchedulingDecision {
            should_transmit,
            bundles_to_transmit,
            estimated_utility: if should_transmit { 1.0 } else { 0.0 },
            estimated_energy_cost: contact.energy_cost,
            drift_component: 0.0,   // FIFO doesn't consider drift
            penalty_component: 0.0, // FIFO doesn't consider penalty
            rationale: "FIFO: Always transmit oldest bundles first".to_string(),
        }
    }

    pub fn get_queue_lengths(&self) -> HashMap<EndpointId, usize> {
        self.bundle_queues
            .iter()
            .map(|(dest, queue)| (dest.clone(), queue.len()))
            .collect()
    }
}

/// Comprehensive test results for scheduler comparison
#[derive(Debug, Clone)]
pub struct TestResults {
    pub scheduler_name: String,
    pub total_bundles: usize,
    pub bundles_delivered: usize,
    pub bundles_on_time: usize,
    pub total_energy_consumed: f64,
    pub max_queue_length: usize,
    pub avg_queue_length: f64,
    pub delivery_rate: f64,
    pub on_time_rate: f64,
    pub energy_per_bundle: f64,
    pub test_duration: u64,
}

impl TestResults {
    pub fn new(scheduler_name: String) -> Self {
        Self {
            scheduler_name,
            total_bundles: 0,
            bundles_delivered: 0,
            bundles_on_time: 0,
            total_energy_consumed: 0.0,
            max_queue_length: 0,
            avg_queue_length: 0.0,
            delivery_rate: 0.0,
            on_time_rate: 0.0,
            energy_per_bundle: 0.0,
            test_duration: 0,
        }
    }

    pub fn finalize(&mut self) {
        self.delivery_rate = if self.total_bundles > 0 {
            self.bundles_delivered as f64 / self.total_bundles as f64
        } else {
            0.0
        };

        self.on_time_rate = if self.bundles_delivered > 0 {
            self.bundles_on_time as f64 / self.bundles_delivered as f64
        } else {
            0.0
        };

        self.energy_per_bundle = if self.bundles_delivered > 0 {
            self.total_energy_consumed / self.bundles_delivered as f64
        } else {
            0.0
        };
    }

    pub fn print_comparison(&self, other: &TestResults) {
        println!("\n📊 Scheduler Performance Comparison");
        println!("====================================");
        println!("Test Duration: {} seconds", self.test_duration);
        println!();

        println!(
            "{:<25} {:<15} {:<15}",
            "Metric", &self.scheduler_name, &other.scheduler_name
        );
        println!("{}", "-".repeat(55));

        println!(
            "{:<25} {:<15} {:<15}",
            "Total Bundles", self.total_bundles, other.total_bundles
        );
        println!(
            "{:<25} {:<15} {:<15}",
            "Delivered", self.bundles_delivered, other.bundles_delivered
        );
        println!(
            "{:<25} {:<15.3} {:<15.3}",
            "Delivery Rate", self.delivery_rate, other.delivery_rate
        );
        println!(
            "{:<25} {:<15.3} {:<15.3}",
            "On-Time Rate", self.on_time_rate, other.on_time_rate
        );
        println!(
            "{:<25} {:<15.2} {:<15.2}",
            "Energy/Bundle", self.energy_per_bundle, other.energy_per_bundle
        );
        println!(
            "{:<25} {:<15} {:<15}",
            "Max Queue Length", self.max_queue_length, other.max_queue_length
        );
        println!(
            "{:<25} {:<15.2} {:<15.2}",
            "Avg Queue Length", self.avg_queue_length, other.avg_queue_length
        );

        println!();
        println!("🎯 Performance Summary:");

        if self.delivery_rate > other.delivery_rate {
            println!(
                "  ✅ {} has {:.1}% higher delivery rate",
                self.scheduler_name,
                (self.delivery_rate - other.delivery_rate) * 100.0
            );
        } else if other.delivery_rate > self.delivery_rate {
            println!(
                "  ✅ {} has {:.1}% higher delivery rate",
                other.scheduler_name,
                (other.delivery_rate - self.delivery_rate) * 100.0
            );
        }

        if self.on_time_rate > other.on_time_rate {
            println!(
                "  ⏰ {} has {:.1}% higher on-time rate",
                self.scheduler_name,
                (self.on_time_rate - other.on_time_rate) * 100.0
            );
        } else if other.on_time_rate > self.on_time_rate {
            println!(
                "  ⏰ {} has {:.1}% higher on-time rate",
                other.scheduler_name,
                (other.on_time_rate - self.on_time_rate) * 100.0
            );
        }

        if self.energy_per_bundle < other.energy_per_bundle {
            println!(
                "  ⚡ {} is {:.1}% more energy efficient",
                self.scheduler_name,
                (1.0 - self.energy_per_bundle / other.energy_per_bundle) * 100.0
            );
        } else if other.energy_per_bundle < self.energy_per_bundle {
            println!(
                "  ⚡ {} is {:.1}% more energy efficient",
                other.scheduler_name,
                (1.0 - other.energy_per_bundle / self.energy_per_bundle) * 100.0
            );
        }

        match self.max_queue_length.cmp(&other.max_queue_length) {
            std::cmp::Ordering::Less => {
                println!(
                    "  📊 {} has better queue stability (max: {} vs {})",
                    self.scheduler_name, self.max_queue_length, other.max_queue_length
                );
            }
            std::cmp::Ordering::Greater => {
                println!(
                    "  📊 {} has better queue stability (max: {} vs {})",
                    other.scheduler_name, other.max_queue_length, self.max_queue_length
                );
            }
            std::cmp::Ordering::Equal => {
                // Same queue stability
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_contact_generator() {
        let mut generator = SyntheticContactGenerator::new(42, 5, 1000);

        // Test linear topology
        let linear_contacts = generator.linear_topology(60, 120);
        assert!(!linear_contacts.is_empty());
        assert_eq!(linear_contacts[0].from, EndpointId::node("node0"));
        assert_eq!(linear_contacts[0].to, EndpointId::node("node1"));

        // Test star topology
        let star_contacts = generator.star_topology(60, 120);
        assert!(!star_contacts.is_empty());
        // Should have bidirectional connections to hub (node0)
        assert!(star_contacts
            .iter()
            .any(|c| c.to == EndpointId::node("node0")));
        assert!(star_contacts
            .iter()
            .any(|c| c.from == EndpointId::node("node0")));

        // Test mesh topology
        let mesh_contacts = generator.mesh_topology(60, 120, 0.5);
        assert!(!mesh_contacts.is_empty());

        // Test bundle generation
        let bundles = generator.generate_test_bundles(10, 3600);
        assert_eq!(bundles.len(), 10);
        assert!(bundles.iter().all(|b| b.primary.lifetime > 0));
    }

    #[test]
    fn test_fifo_scheduler() {
        let mut fifo = FifoScheduler::new();

        // Create test bundle
        let bundle = Bundle::new(
            EndpointId::node("dest"),
            EndpointId::node("src"),
            Bytes::from("test data"),
            60000,
        );

        fifo.enqueue_bundle(&bundle);

        let queue_lengths = fifo.get_queue_lengths();
        assert_eq!(queue_lengths[&EndpointId::node("dest")], 1);

        let contact = Contact::new(
            EndpointId::node("local"),
            EndpointId::node("dest"),
            1000,
            1060,
            1_000_000,
        );

        let decision = fifo.schedule_transmission(&contact, &[bundle.id()], 1);
        assert!(decision.should_transmit);
        assert_eq!(decision.bundles_to_transmit.len(), 1);
        assert_eq!(
            decision.rationale,
            "FIFO: Always transmit oldest bundles first"
        );
    }

    #[test]
    fn test_results_calculation() {
        let mut results = TestResults::new("Test".to_string());
        results.total_bundles = 100;
        results.bundles_delivered = 80;
        results.bundles_on_time = 60;
        results.total_energy_consumed = 400.0;
        results.test_duration = 3600;

        results.finalize();

        assert_eq!(results.delivery_rate, 0.8);
        assert_eq!(results.on_time_rate, 0.75);
        assert_eq!(results.energy_per_bundle, 5.0);
    }
}
