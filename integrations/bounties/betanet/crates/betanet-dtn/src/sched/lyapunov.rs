//! Lyapunov-based DTN queue scheduler for stability and energy optimization
//!
//! Implements the Lyapunov optimization framework to stabilize DTN queues while maximizing
//! on-time delivery under energy constraints. The scheduler minimizes:
//!
//!   ΔL + V·(−U)
//!
//! Where:
//! - L = Σ Q_i^2 (sum of squared queue lengths for queue stability)
//! - U = utility (on-time delivery rate, throughput, etc.)
//! - V = knob parameter trading off lateness vs energy efficiency
//!
//! The scheduler chooses which bundles to transmit per contact opportunity to minimize
//! this drift-plus-penalty expression, incorporating link energy costs and privacy penalties.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info, trace, warn};

use crate::bundle::{Bundle, BundleId, EndpointId};
use crate::router::Contact;

/// Configuration parameters for the Lyapunov scheduler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LyapunovConfig {
    /// V parameter: Controls trade-off between queue stability (low V) and utility optimization (high V)
    /// - Low V (0.1-1.0): Prioritize queue stability, transmit aggressively
    /// - High V (10.0-100.0): Prioritize energy efficiency, transmit selectively
    pub v_parameter: f64,

    /// Maximum queue length before triggering aggressive transmission
    pub max_queue_length: usize,

    /// Weight for energy cost in utility function (relative to delivery utility)
    pub energy_cost_weight: f64,

    /// Weight for privacy penalty in utility function
    pub privacy_penalty_weight: f64,

    /// Time window for computing delivery rates and queue statistics (seconds)
    pub observation_window: u64,

    /// Minimum utility threshold for transmission decisions
    pub min_utility_threshold: f64,
}

impl Default for LyapunovConfig {
    fn default() -> Self {
        Self {
            v_parameter: 1.0,            // Balanced stability vs efficiency
            max_queue_length: 1000,      // Reasonable queue limit
            energy_cost_weight: 0.3,     // Moderate energy awareness
            privacy_penalty_weight: 0.2, // Some privacy consideration
            observation_window: 300,     // 5 minutes
            min_utility_threshold: 0.1,  // Prevent negative utility transmissions
        }
    }
}

/// Current state of a destination queue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueState {
    /// Current number of bundles in queue
    pub queue_length: usize,

    /// Queue length at last decision point
    pub prev_queue_length: usize,

    /// Number of bundles delivered on-time in observation window
    pub delivered_on_time: u64,

    /// Total number of bundles processed in observation window
    pub total_processed: u64,

    /// Average queue length over observation window
    pub avg_queue_length: f64,

    /// Total energy consumed for this destination in observation window
    pub energy_consumed: f64,

    /// Last update timestamp
    pub last_update: u64,
}

impl Default for QueueState {
    fn default() -> Self {
        Self {
            queue_length: 0,
            prev_queue_length: 0,
            delivered_on_time: 0,
            total_processed: 0,
            avg_queue_length: 0.0,
            energy_consumed: 0.0,
            last_update: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

/// Scheduling decision for a specific transmission opportunity
#[derive(Debug, Clone)]
pub struct SchedulingDecision {
    /// Whether to transmit on this contact
    pub should_transmit: bool,

    /// Bundle IDs to transmit, in priority order
    pub bundles_to_transmit: Vec<BundleId>,

    /// Estimated utility gain from this transmission
    pub estimated_utility: f64,

    /// Estimated energy cost
    pub estimated_energy_cost: f64,

    /// Lyapunov drift component (queue stability)
    pub drift_component: f64,

    /// Penalty component (negative utility)
    pub penalty_component: f64,

    /// Reasoning for decision (for debugging/monitoring)
    pub rationale: String,
}

/// Errors that can occur during scheduling
#[derive(Error, Debug)]
pub enum SchedulingError {
    #[error("Invalid configuration: {message}")]
    InvalidConfig { message: String },

    #[error("Queue state not found for destination: {destination}")]
    QueueStateNotFound { destination: EndpointId },

    #[error("Bundle not found: {bundle_id}")]
    BundleNotFound { bundle_id: BundleId },

    #[error("Contact validation failed: {message}")]
    ContactValidation { message: String },
}

/// Lyapunov-based queue scheduler
pub struct LyapunovScheduler {
    /// Configuration parameters
    config: LyapunovConfig,

    /// Per-destination queue states
    queue_states: HashMap<EndpointId, QueueState>,

    /// Pending bundles organized by destination
    pending_bundles: HashMap<EndpointId, Vec<BundleId>>,

    /// Bundle metadata cache for quick access
    bundle_cache: HashMap<BundleId, BundleMetadata>,

    /// Historical statistics for utility computation
    delivery_history: HashMap<EndpointId, Vec<DeliveryRecord>>,
}

/// Cached bundle metadata for scheduling decisions
#[derive(Debug, Clone)]
struct BundleMetadata {
    pub id: BundleId,
    pub destination: EndpointId,
    pub size: usize,
    pub creation_time: u64,
    pub lifetime_ms: u64,
    pub priority: u8,
    pub arrival_time: u64, // When bundle entered queue
}

/// Record of a bundle delivery for utility computation
#[derive(Debug, Clone)]
struct DeliveryRecord {
    pub timestamp: u64,
    pub was_on_time: bool,
    pub energy_cost: f64,
    pub queue_length_at_delivery: usize,
}

impl LyapunovScheduler {
    /// Create a new Lyapunov scheduler
    pub fn new(config: LyapunovConfig) -> Result<Self, SchedulingError> {
        // Validate configuration
        if config.v_parameter <= 0.0 {
            return Err(SchedulingError::InvalidConfig {
                message: "V parameter must be positive".to_string(),
            });
        }

        if config.max_queue_length == 0 {
            return Err(SchedulingError::InvalidConfig {
                message: "Max queue length must be positive".to_string(),
            });
        }

        info!("Creating Lyapunov scheduler with V={}", config.v_parameter);

        Ok(Self {
            config,
            queue_states: HashMap::new(),
            pending_bundles: HashMap::new(),
            bundle_cache: HashMap::new(),
            delivery_history: HashMap::new(),
        })
    }

    /// Add a bundle to the scheduling queue
    pub fn enqueue_bundle(&mut self, bundle: &Bundle) {
        let bundle_id = bundle.id();
        let destination = &bundle.primary.destination;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Cache bundle metadata
        let metadata = BundleMetadata {
            id: bundle_id.clone(),
            destination: destination.clone(),
            size: bundle.size(),
            creation_time: bundle.primary.creation_timestamp.dtn_time,
            lifetime_ms: bundle.primary.lifetime,
            priority: 1, // Implementation required: Extract from bundle flags
            arrival_time: now,
        };

        self.bundle_cache.insert(bundle_id.clone(), metadata);

        // Add to pending bundles for destination
        let queue = self.pending_bundles.entry(destination.clone()).or_default();
        queue.push(bundle_id);

        // Update queue state
        let state = self.queue_states.entry(destination.clone()).or_default();
        state.prev_queue_length = state.queue_length;
        state.queue_length = queue.len();
        state.last_update = now;

        // Update average queue length
        state.avg_queue_length = 0.9 * state.avg_queue_length + 0.1 * state.queue_length as f64;

        debug!(
            "Enqueued bundle {} for {}, queue length now {}",
            bundle.id(),
            destination,
            state.queue_length
        );

        // Warn if queue is getting large
        if state.queue_length > self.config.max_queue_length {
            warn!(
                "Queue for {} exceeded max length: {} > {}",
                destination, state.queue_length, self.config.max_queue_length
            );
        }
    }

    /// Remove a bundle from the scheduling queue (e.g., after successful delivery)
    pub fn dequeue_bundle(
        &mut self,
        bundle_id: BundleId,
        was_delivered_on_time: bool,
        energy_cost: f64,
    ) {
        if let Some(metadata) = self.bundle_cache.remove(&bundle_id) {
            let destination = &metadata.destination;
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            // Remove from pending bundles
            if let Some(queue) = self.pending_bundles.get_mut(destination) {
                queue.retain(|id| *id != bundle_id);

                // Update queue state
                if let Some(state) = self.queue_states.get_mut(destination) {
                    state.prev_queue_length = state.queue_length;
                    state.queue_length = queue.len();
                    state.last_update = now;

                    // Update delivery statistics
                    state.total_processed += 1;
                    if was_delivered_on_time {
                        state.delivered_on_time += 1;
                    }
                    state.energy_consumed += energy_cost;

                    // Update average queue length
                    state.avg_queue_length =
                        0.9 * state.avg_queue_length + 0.1 * state.queue_length as f64;

                    // Record delivery for utility computation
                    let history = self
                        .delivery_history
                        .entry(destination.clone())
                        .or_default();
                    history.push(DeliveryRecord {
                        timestamp: now,
                        was_on_time: was_delivered_on_time,
                        energy_cost,
                        queue_length_at_delivery: state.queue_length,
                    });

                    // Keep only recent history
                    let cutoff = now.saturating_sub(self.config.observation_window);
                    history.retain(|record| record.timestamp >= cutoff);
                }
            }

            debug!(
                "Dequeued bundle {} from {}, on_time: {}, energy: {:.3}",
                bundle_id, destination, was_delivered_on_time, energy_cost
            );
        }
    }

    /// Make a scheduling decision for a given contact opportunity
    pub fn schedule_transmission(
        &mut self,
        contact: &Contact,
        available_bundles: &[BundleId],
        max_transmissions: usize,
    ) -> Result<SchedulingDecision, SchedulingError> {
        let destination = &contact.to;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        trace!("Scheduling transmission for contact to {}", destination);

        // Get current queue state
        let queue_state = self
            .queue_states
            .get(destination)
            .cloned()
            .unwrap_or_default();

        // Filter bundles that are actually for this destination and available
        let mut candidate_bundles: Vec<_> = available_bundles
            .iter()
            .filter_map(|bundle_id| {
                self.bundle_cache
                    .get(bundle_id)
                    .map(|metadata| (bundle_id.clone(), metadata))
            })
            .filter(|(_, metadata)| metadata.destination == *destination)
            .collect();

        // Sort by priority (higher priority first) and age (older first)
        candidate_bundles.sort_by(|a, b| {
            b.1.priority
                .cmp(&a.1.priority)
                .then_with(|| a.1.arrival_time.cmp(&b.1.arrival_time))
        });

        // Compute Lyapunov drift component: Q_i * (arrivals - departures)
        // For simplicity, assume arrivals = 0 during contact window
        let queue_length = queue_state.queue_length as f64;
        let drift_component = queue_length * (-(max_transmissions as f64));

        // Compute utility of transmitting
        let base_utility = self.compute_transmission_utility(
            destination,
            &candidate_bundles,
            contact,
            max_transmissions,
            now,
        );

        // Account for energy costs
        let energy_penalty = self.config.energy_cost_weight * contact.energy_cost;

        // Account for privacy penalties (simplified - could be more sophisticated)
        let privacy_penalty = self.config.privacy_penalty_weight * 0.1; // Placeholder

        let total_utility = base_utility - energy_penalty - privacy_penalty;

        // Lyapunov decision: transmit if drift + V * penalty < 0
        // Equivalently: transmit if V * utility > queue_drift_magnitude
        let penalty_component = -self.config.v_parameter * total_utility;
        let lyapunov_expression = drift_component + penalty_component;

        let should_transmit =
            lyapunov_expression < 0.0 && total_utility > self.config.min_utility_threshold;

        let bundles_to_transmit = if should_transmit {
            candidate_bundles
                .iter()
                .take(max_transmissions)
                .map(|(bundle_id, _)| bundle_id.clone())
                .collect()
        } else {
            Vec::new()
        };

        let rationale = format!(
            "Queue={:.1}, Drift={:.2}, Utility={:.3}, Energy={:.3}, Privacy={:.3}, Lyapunov={:.3}",
            queue_length,
            drift_component,
            base_utility,
            energy_penalty,
            privacy_penalty,
            lyapunov_expression
        );

        debug!(
            "Scheduling decision for {}: {} ({})",
            destination,
            if should_transmit { "TRANSMIT" } else { "DEFER" },
            rationale
        );

        Ok(SchedulingDecision {
            should_transmit,
            bundles_to_transmit,
            estimated_utility: total_utility,
            estimated_energy_cost: contact.energy_cost,
            drift_component,
            penalty_component,
            rationale,
        })
    }

    /// Compute the utility of transmitting bundles to a destination
    fn compute_transmission_utility(
        &self,
        _destination: &EndpointId,
        candidate_bundles: &[(BundleId, &BundleMetadata)],
        contact: &Contact,
        max_transmissions: usize,
        current_time: u64,
    ) -> f64 {
        if candidate_bundles.is_empty() {
            return 0.0;
        }

        let bundles_to_consider = candidate_bundles.iter().take(max_transmissions);
        let mut total_utility = 0.0;

        for (_, metadata) in bundles_to_consider {
            // Compute on-time delivery probability
            let bundle_age = current_time.saturating_sub(metadata.creation_time);
            let remaining_lifetime = metadata.lifetime_ms.saturating_sub(bundle_age * 1000) / 1000;
            let transmission_time = contact.transmission_time(metadata.size).as_secs();

            // Utility increases for bundles that will be delivered on-time
            let on_time_delivery_utility = if transmission_time <= remaining_lifetime {
                1.0 // Full utility for on-time delivery
            } else {
                // Reduced utility for late delivery, but still positive
                0.3 * (remaining_lifetime as f64 / transmission_time as f64).max(0.1)
            };

            // Priority boost
            let priority_multiplier = 1.0 + (metadata.priority as f64 - 1.0) * 0.2;

            // Age boost (older bundles get priority)
            let age_hours = bundle_age as f64 / 3600.0;
            let age_multiplier = 1.0 + (age_hours / 24.0).min(2.0); // Up to 3x boost for 2+ day old bundles

            let bundle_utility = on_time_delivery_utility * priority_multiplier * age_multiplier;
            total_utility += bundle_utility;
        }

        // Apply delivery reliability
        total_utility *= contact.delivery_probability();

        // Normalize by number of transmissions for fair comparison
        total_utility / max_transmissions.max(1) as f64
    }

    /// Update configuration parameters (e.g., adjusting V parameter)
    pub fn update_config(&mut self, new_config: LyapunovConfig) -> Result<(), SchedulingError> {
        if new_config.v_parameter <= 0.0 {
            return Err(SchedulingError::InvalidConfig {
                message: "V parameter must be positive".to_string(),
            });
        }

        info!(
            "Updating Lyapunov config: V {} -> {}",
            self.config.v_parameter, new_config.v_parameter
        );

        self.config = new_config;
        Ok(())
    }

    /// Get current queue states for monitoring
    pub fn get_queue_states(&self) -> &HashMap<EndpointId, QueueState> {
        &self.queue_states
    }

    /// Get current configuration
    pub fn get_config(&self) -> &LyapunovConfig {
        &self.config
    }

    /// Clean up expired data and optimize memory usage
    pub fn cleanup(&mut self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let cutoff = now.saturating_sub(self.config.observation_window);

        // Clean up delivery history
        for history in self.delivery_history.values_mut() {
            history.retain(|record| record.timestamp >= cutoff);
        }

        // Clean up empty queues
        self.pending_bundles
            .retain(|_, bundles| !bundles.is_empty());
        self.queue_states
            .retain(|destination, _| self.pending_bundles.contains_key(destination));

        debug!(
            "Cleaned up scheduler data, {} active destinations",
            self.queue_states.len()
        );
    }

    /// Get comprehensive statistics for monitoring and debugging
    pub fn get_statistics(&self) -> LyapunovStatistics {
        let total_queued = self
            .queue_states
            .values()
            .map(|state| state.queue_length)
            .sum::<usize>();

        let total_delivered = self
            .queue_states
            .values()
            .map(|state| state.total_processed)
            .sum::<u64>();

        let total_on_time = self
            .queue_states
            .values()
            .map(|state| state.delivered_on_time)
            .sum::<u64>();

        let avg_queue_length = if !self.queue_states.is_empty() {
            self.queue_states
                .values()
                .map(|state| state.avg_queue_length)
                .sum::<f64>()
                / self.queue_states.len() as f64
        } else {
            0.0
        };

        let on_time_delivery_rate = if total_delivered > 0 {
            total_on_time as f64 / total_delivered as f64
        } else {
            0.0
        };

        LyapunovStatistics {
            active_destinations: self.queue_states.len(),
            total_queued_bundles: total_queued,
            total_delivered_bundles: total_delivered,
            on_time_delivery_rate,
            average_queue_length: avg_queue_length,
            v_parameter: self.config.v_parameter,
        }
    }
}

/// Statistics for monitoring scheduler performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LyapunovStatistics {
    pub active_destinations: usize,
    pub total_queued_bundles: usize,
    pub total_delivered_bundles: u64,
    pub on_time_delivery_rate: f64,
    pub average_queue_length: f64,
    pub v_parameter: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::Bundle;
    use std::time::Duration;

    fn create_test_bundle(destination: &str, lifetime_ms: u64) -> Bundle {
        let dest = EndpointId::node(destination);
        let source = EndpointId::node("source");
        let payload = bytes::Bytes::from(b"test data".to_vec());
        Bundle::new(dest, source, payload, lifetime_ms)
    }

    fn create_test_contact(to: &str, energy_cost: f64) -> Contact {
        Contact {
            from: EndpointId::node("local"),
            to: EndpointId::node(to),
            start_time: 1000,
            end_time: 2000,
            data_rate: 1_000_000, // 1 Mbps
            latency: Duration::from_millis(100),
            reliability: 0.95,
            energy_cost,
        }
    }

    #[test]
    fn test_scheduler_creation() {
        let config = LyapunovConfig::default();
        let scheduler = LyapunovScheduler::new(config).unwrap();

        assert_eq!(scheduler.queue_states.len(), 0);
        assert_eq!(scheduler.get_config().v_parameter, 1.0);
    }

    #[test]
    fn test_invalid_config() {
        let mut config = LyapunovConfig::default();
        config.v_parameter = -1.0;

        let result = LyapunovScheduler::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_bundle_enqueue_dequeue() {
        let config = LyapunovConfig::default();
        let mut scheduler = LyapunovScheduler::new(config).unwrap();

        let bundle = create_test_bundle("dest1", 60000);
        let bundle_id = bundle.id();

        // Enqueue bundle
        scheduler.enqueue_bundle(&bundle);

        let dest = EndpointId::node("dest1");
        let state = scheduler.queue_states.get(&dest).unwrap();
        assert_eq!(state.queue_length, 1);

        // Dequeue bundle
        scheduler.dequeue_bundle(bundle_id, true, 1.0);

        let state = scheduler.queue_states.get(&dest).unwrap();
        assert_eq!(state.queue_length, 0);
        assert_eq!(state.delivered_on_time, 1);
        assert_eq!(state.total_processed, 1);
    }

    #[test]
    fn test_scheduling_decision_high_queue() {
        let config = LyapunovConfig {
            v_parameter: 1.0,
            max_queue_length: 100,
            energy_cost_weight: 0.3,
            privacy_penalty_weight: 0.1,
            observation_window: 300,
            min_utility_threshold: 0.1,
        };
        let mut scheduler = LyapunovScheduler::new(config).unwrap();

        // Create multiple bundles to build up queue
        let mut bundle_ids = Vec::new();
        for _i in 0..10 {
            let bundle = create_test_bundle("dest1", 60000);
            let bundle_id = bundle.id();
            bundle_ids.push(bundle_id);
            scheduler.enqueue_bundle(&bundle);
        }

        let contact = create_test_contact("dest1", 1.0);
        let decision = scheduler
            .schedule_transmission(&contact, &bundle_ids, 3)
            .unwrap();

        // Should transmit when queue is building up
        assert!(decision.should_transmit);
        assert_eq!(decision.bundles_to_transmit.len(), 3);
        assert!(decision.drift_component < 0.0); // Negative drift (reducing queue)
    }

    #[test]
    fn test_scheduling_decision_high_energy_cost() {
        let config = LyapunovConfig {
            v_parameter: 10.0, // High V - prioritize energy efficiency
            energy_cost_weight: 1.0,
            ..LyapunovConfig::default()
        };
        let mut scheduler = LyapunovScheduler::new(config).unwrap();

        let bundle = create_test_bundle("dest1", 60000);
        let bundle_id = bundle.id();
        scheduler.enqueue_bundle(&bundle);

        let high_energy_contact = create_test_contact("dest1", 10.0); // Very expensive
        let decision = scheduler
            .schedule_transmission(&high_energy_contact, &[bundle_id], 1)
            .unwrap();

        // Should defer transmission when energy cost is high and V is large
        assert!(!decision.should_transmit);
    }

    #[test]
    fn test_utility_computation_deadline() {
        let config = LyapunovConfig::default();
        let mut scheduler = LyapunovScheduler::new(config).unwrap();

        // Create bundle with short lifetime
        let short_lifetime_bundle = create_test_bundle("dest1", 1000); // 1 second
        let bundle_id = short_lifetime_bundle.id();
        scheduler.enqueue_bundle(&short_lifetime_bundle);

        let contact = create_test_contact("dest1", 1.0);
        let decision = scheduler
            .schedule_transmission(&contact, &[bundle_id], 1)
            .unwrap();

        // Should still transmit even if deadline is tight (reduced utility but positive)
        assert!(decision.estimated_utility > 0.0);
    }

    #[test]
    fn test_cleanup() {
        let config = LyapunovConfig {
            observation_window: 0, // Zero window means immediate cleanup
            ..LyapunovConfig::default()
        };
        let mut scheduler = LyapunovScheduler::new(config).unwrap();

        // Add a delivery record with a very old timestamp
        let dest = EndpointId::node("dest1");
        let old_record = DeliveryRecord {
            timestamp: 1000, // Very old timestamp
            was_on_time: true,
            energy_cost: 1.0,
            queue_length_at_delivery: 0,
        };

        scheduler
            .delivery_history
            .entry(dest.clone())
            .or_insert_with(Vec::new)
            .push(old_record);

        // Verify history was created
        assert!(scheduler.delivery_history.contains_key(&dest));
        assert!(!scheduler.delivery_history.get(&dest).unwrap().is_empty());

        // Call cleanup - should remove the old record
        scheduler.cleanup();

        // Delivery history should be cleaned up
        if let Some(history) = scheduler.delivery_history.get(&dest) {
            assert!(
                history.is_empty(),
                "History should be empty after cleanup, but has {} items",
                history.len()
            );
        }
    }

    #[test]
    fn test_statistics() {
        let config = LyapunovConfig::default();
        let mut scheduler = LyapunovScheduler::new(config).unwrap();

        // Add some bundles and process them
        for i in 0..5 {
            let bundle = create_test_bundle(&format!("dest{}", i), 60000);
            let bundle_id = bundle.id();
            scheduler.enqueue_bundle(&bundle);

            if i < 3 {
                scheduler.dequeue_bundle(bundle_id, true, 1.0);
            }
        }

        let stats = scheduler.get_statistics();
        assert_eq!(stats.active_destinations, 5);
        assert_eq!(stats.total_queued_bundles, 2); // 2 still in queue
        assert_eq!(stats.total_delivered_bundles, 3);
        assert_eq!(stats.on_time_delivery_rate, 1.0); // All delivered bundles were on time
    }
}
