//! TTL-based probabilistic rebroadcast for BitChat mesh networking
//!
//! Implements intelligent message rebroadcasting with TTL management and collision avoidance.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Rebroadcast configuration parameters
#[derive(Debug, Clone)]
pub struct RebroadcastConfig {
    /// Base rebroadcast probability (0.0 to 1.0)
    pub base_probability: f64,
    /// Maximum TTL value for new messages
    pub max_ttl: u8,
    /// Minimum delay before rebroadcasting (milliseconds)
    pub min_delay_ms: u64,
    /// Maximum delay before rebroadcasting (milliseconds)
    pub max_delay_ms: u64,
    /// Enable adaptive probability based on network density
    pub adaptive_probability: bool,
    /// Duplicate detection window (seconds)
    pub duplicate_window_secs: u64,
}

impl Default for RebroadcastConfig {
    fn default() -> Self {
        Self {
            base_probability: 0.7,
            max_ttl: 7,
            min_delay_ms: 50,
            max_delay_ms: 200,
            adaptive_probability: true,
            duplicate_window_secs: 300, // 5 minutes
        }
    }
}

/// Message rebroadcast decision
#[derive(Debug, Clone, PartialEq)]
pub enum RebroadcastDecision {
    /// Rebroadcast the message after specified delay
    Rebroadcast { delay_ms: u64 },
    /// Drop the message (already seen or TTL expired)
    Drop { reason: String },
    /// Forward immediately (high priority)
    ForwardImmediate,
}

/// Rebroadcast engine for managing message propagation
pub struct RebroadcastEngine {
    config: RebroadcastConfig,
    seen_messages: HashMap<String, Instant>,
    network_density: f64,
}

impl RebroadcastEngine {
    pub fn new(config: RebroadcastConfig) -> Self {
        Self {
            config,
            seen_messages: HashMap::new(),
            network_density: 1.0, // Default medium density
        }
    }

    /// Evaluate whether to rebroadcast a message
    pub fn should_rebroadcast(
        &mut self,
        message_id: &str,
        current_ttl: u8,
        hop_count: u8,
        is_urgent: bool,
    ) -> RebroadcastDecision {
        // Clean up old entries
        self.cleanup_old_entries();

        // Check if message was already seen
        if let Some(seen_time) = self.seen_messages.get(message_id) {
            if seen_time.elapsed().as_secs() < self.config.duplicate_window_secs {
                return RebroadcastDecision::Drop {
                    reason: "Duplicate message detected".to_string(),
                };
            }
        }

        // Record this message as seen
        self.seen_messages.insert(message_id.to_string(), Instant::now());

        // Check TTL expiry
        if current_ttl == 0 {
            return RebroadcastDecision::Drop {
                reason: "TTL expired".to_string(),
            };
        }

        // Immediate forward for urgent messages
        if is_urgent && hop_count < 2 {
            return RebroadcastDecision::ForwardImmediate;
        }

        // Calculate rebroadcast probability
        let probability = self.calculate_rebroadcast_probability(current_ttl, hop_count);
        
        // Make probabilistic decision (simplified random for now)
        if (hop_count as f64 * 0.3) <= probability {
            let delay_ms = self.calculate_rebroadcast_delay(hop_count);
            RebroadcastDecision::Rebroadcast { delay_ms }
        } else {
            RebroadcastDecision::Drop {
                reason: format!("Probabilistic drop (p={:.2})", probability),
            }
        }
    }

    /// Calculate rebroadcast probability based on network conditions
    fn calculate_rebroadcast_probability(&self, ttl: u8, hop_count: u8) -> f64 {
        let mut probability = self.config.base_probability;

        // Reduce probability for messages that have traveled far
        if hop_count > 0 {
            probability *= (1.0 - (hop_count as f64 * 0.1)).max(0.1);
        }

        // Reduce probability for low TTL messages
        if ttl <= 2 {
            probability *= 0.5;
        }

        // Apply adaptive adjustment based on network density
        if self.config.adaptive_probability {
            // Higher density = lower probability to reduce collisions
            probability *= (2.0 - self.network_density).max(0.2).min(1.0);
        }

        probability.max(0.0).min(1.0)
    }

    /// Calculate random delay to avoid broadcast collisions
    fn calculate_rebroadcast_delay(&self, hop_count: u8) -> u64 {
        let base_delay = self.config.min_delay_ms;
        let max_additional = self.config.max_delay_ms - self.config.min_delay_ms;
        
        // Add hop-based delay to spread out rebroadcasts
        let hop_delay = (hop_count as u64) * 25; // 25ms per hop
        
        // Simplified random delay
        let random_delay = (hop_count as u64 * 10) % max_additional;
        
        base_delay + hop_delay + random_delay
    }

    /// Update network density estimation
    pub fn update_network_density(&mut self, active_peers: usize, total_discovered: usize) {
        if total_discovered > 0 {
            self.network_density = (active_peers as f64) / (total_discovered as f64);
            self.network_density = self.network_density.max(0.1).min(2.0);
        }
    }

    /// Clean up old message entries
    fn cleanup_old_entries(&mut self) {
        let cutoff_time = Duration::from_secs(self.config.duplicate_window_secs);
        let now = Instant::now();
        
        self.seen_messages.retain(|_, seen_time| {
            now.duration_since(*seen_time) < cutoff_time
        });
    }

    /// Create TTL for new outbound message
    pub fn create_new_message_ttl(&self) -> u8 {
        self.config.max_ttl
    }

    /// Decrement TTL for forwarded message
    pub fn decrement_ttl(&self, current_ttl: u8) -> u8 {
        current_ttl.saturating_sub(1)
    }
}
