//! Label-setting router with Pareto frontier optimization
//!
//! Extends the existing DTN ContactGraphRouter with proper multi-criteria optimization
//! using semiring algebra and Pareto frontier management.

use crate::semiring::{Cost, WeightVector};
use betanet_dtn::{ContactGraphRouter, EndpointId};
use betanet_dtn::router::Contact;
use std::collections::HashMap;
use std::cmp::{Ordering, PartialEq, PartialOrd};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tracing::debug;

/// Maximum number of labels to keep in Pareto frontier per node
const DEFAULT_FRONTIER_CAP: usize = 8;

/// Errors that can occur during routing
#[derive(Error, Debug)]
pub enum RoutingError {
    #[error("No path found to destination {destination}")]
    NoPathFound { destination: EndpointId },
    #[error("DTN routing error: {0}")]
    DtnError(#[from] betanet_dtn::router::RoutingError),
    #[error("Invalid routing parameters: {message}")]
    InvalidParameters { message: String },
}

/// A labeled path in the routing graph
#[derive(Debug, Clone, PartialEq)]
pub struct LabeledPath {
    /// Destination endpoint
    pub destination: EndpointId,
    /// Total cost from source to this node
    pub cost: Cost,
    /// Path through contacts (for reconstruction)
    pub contacts: Vec<Contact>,
    /// Last hop arrival time
    pub arrival_time: u64,
    /// Privacy budget consumed
    pub privacy_consumed: f64,
}

impl LabeledPath {
    /// Create a new labeled path
    pub fn new(destination: EndpointId, cost: Cost, arrival_time: u64) -> Self {
        Self {
            destination,
            cost,
            contacts: Vec::new(),
            arrival_time,
            privacy_consumed: cost.privacy_eps.0,
        }
    }

    /// Extend path with a new contact
    pub fn extend(&self, contact: Contact, new_cost: Cost) -> Self {
        let mut contacts = self.contacts.clone();
        contacts.push(contact.clone());

        Self {
            destination: contact.to.clone(),
            cost: new_cost,
            contacts,
            arrival_time: contact.end_time,
            privacy_consumed: new_cost.privacy_eps.0,
        }
    }

    /// Get the number of hops in this path
    pub fn hop_count(&self) -> usize {
        self.contacts.len()
    }

    /// Check if path meets timing constraints
    pub fn meets_deadline(&self, deadline: Option<u64>) -> bool {
        deadline.map_or(true, |d| self.arrival_time <= d)
    }

    /// Check if path respects privacy budget
    pub fn respects_privacy_cap(&self, privacy_cap: Option<f64>) -> bool {
        privacy_cap.map_or(true, |cap| self.privacy_consumed <= cap)
    }
}

impl PartialOrd for LabeledPath {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Order by cost for priority queue using scalarized cost
        let weights = crate::semiring::WeightVector::balanced();
        let self_score = self.cost.scalarize(&weights);
        let other_score = other.cost.scalarize(&weights);
        self_score.partial_cmp(&other_score)
    }
}

/// Pareto frontier manager for efficient non-dominated set maintenance
#[derive(Debug, Clone)]
pub struct ParetoFrontier {
    /// Non-dominated paths
    paths: Vec<LabeledPath>,
    /// Maximum frontier size
    cap: usize,
}

impl ParetoFrontier {
    /// Create new frontier with capacity
    pub fn new(cap: usize) -> Self {
        Self {
            paths: Vec::with_capacity(cap),
            cap,
        }
    }

    /// Try to insert a new path, maintaining Pareto optimality
    /// Returns true if the path was added or if it updated an existing dominated path
    pub fn try_insert(&mut self, path: LabeledPath) -> bool {
        // Check if new path is dominated by any existing path
        if self.paths.iter().any(|p| p.cost.dominates(&path.cost)) {
            return false; // Dominated, don't insert
        }

        // Remove all paths dominated by the new path
        self.paths.retain(|p| !path.cost.dominates(&p.cost));

        // Add the new path
        self.paths.push(path);

        // If over capacity, remove the worst path by scalarization
        if self.paths.len() > self.cap {
            self.trim_to_capacity();
        }

        true
    }

    /// Remove worst paths when over capacity using balanced scalarization
    fn trim_to_capacity(&mut self) {
        if self.paths.len() <= self.cap {
            return;
        }

        let weights = WeightVector::balanced();

        // Sort by scalarized cost (higher is worse)
        self.paths.sort_by(|a, b| {
            b.cost.scalarize(&weights)
                .partial_cmp(&a.cost.scalarize(&weights))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep only the best paths
        self.paths.truncate(self.cap);
    }

    /// Get all paths in the frontier
    pub fn paths(&self) -> &[LabeledPath] {
        &self.paths
    }

    /// Select best path using given weights
    pub fn select_best(&self, weights: &WeightVector) -> Option<&LabeledPath> {
        self.paths
            .iter()
            .min_by(|a, b| {
                a.cost.scalarize(weights)
                    .partial_cmp(&b.cost.scalarize(weights))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Check if frontier is empty
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    /// Get frontier size
    pub fn len(&self) -> usize {
        self.paths.len()
    }
}

/// Multi-criteria label-setting router building on DTN ContactGraphRouter
pub struct SemiringRouter {
    /// Underlying DTN router for contact management
    dtn_router: ContactGraphRouter,
    /// Local node ID (cached for easy access)
    local_node: EndpointId,
    /// Frontier capacity (K parameter)
    frontier_cap: usize,
    /// Cache of computed frontiers
    frontier_cache: HashMap<EndpointId, ParetoFrontier>,
    /// Cache TTL
    cache_ttl: Duration,
    /// Last cache cleanup time
    last_cleanup: u64,
}

impl SemiringRouter {
    /// Create new semiring router wrapping DTN router
    pub fn new(dtn_router: ContactGraphRouter, local_node: EndpointId, frontier_cap: Option<usize>) -> Self {
        Self {
            dtn_router,
            local_node,
            frontier_cap: frontier_cap.unwrap_or(DEFAULT_FRONTIER_CAP),
            frontier_cache: HashMap::new(),
            cache_ttl: Duration::from_secs(300), // 5 minutes
            last_cleanup: Self::current_time(),
        }
    }

    /// Find Pareto-optimal paths to destination using label-setting algorithm
    pub fn find_pareto_paths(
        &mut self,
        destination: &EndpointId,
        current_time: u64,
        deadline: Option<u64>,
        privacy_cap: Option<f64>,
    ) -> Result<ParetoFrontier, RoutingError> {
        // Check cache first
        if let Some(cached) = self.frontier_cache.get(destination) {
            if !cached.is_empty() && self.is_cache_valid(current_time) {
                debug!("Using cached Pareto frontier for {}", destination);
                return Ok(cached.clone());
            }
        }

        debug!("Computing Pareto frontier for {} with label-setting", destination);

        // Get contact plan from DTN router
        let contacts = self.get_active_and_future_contacts(current_time);

        if contacts.is_empty() {
            return Err(RoutingError::NoPathFound {
                destination: destination.clone(),
            });
        }

        // Label-setting algorithm with Pareto frontier
        let frontier = self.label_setting_search(
            destination,
            contacts,
            current_time,
            deadline,
            privacy_cap,
        )?;

        // Cache result
        self.frontier_cache.insert(destination.clone(), frontier.clone());

        // Cleanup old cache entries periodically
        if self.should_cleanup_cache(current_time) {
            self.cleanup_cache(current_time);
        }

        Ok(frontier)
    }

    /// Select best path from Pareto frontier using QoS requirements
    pub fn select_path(
        &mut self,
        destination: &EndpointId,
        qos: QosRequirements,
        deadline: Option<u64>,
        privacy_cap: Option<f64>,
    ) -> Result<Option<LabeledPath>, RoutingError> {
        let current_time = Self::current_time();
        let frontier = self.find_pareto_paths(destination, current_time, deadline, privacy_cap)?;

        if frontier.is_empty() {
            return Ok(None);
        }

        // Filter paths that meet constraints
        let valid_paths: Vec<&LabeledPath> = frontier
            .paths()
            .iter()
            .filter(|path| {
                path.meets_deadline(deadline) && path.respects_privacy_cap(privacy_cap)
            })
            .collect();

        if valid_paths.is_empty() {
            debug!("No paths meet deadline/privacy constraints");
            return Ok(None);
        }

        // Select best path using QoS-derived weights
        let weights = qos.to_weight_vector();
        let best_path = valid_paths
            .into_iter()
            .min_by(|a, b| {
                a.cost.scalarize(&weights)
                    .partial_cmp(&b.cost.scalarize(&weights))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        debug!(
            "Selected path to {} with cost {} using weights {:?}",
            destination,
            best_path.as_ref().map(|p| p.cost.to_string()).unwrap_or_else(|| "None".to_string()),
            weights
        );

        Ok(best_path.cloned())
    }

    /// Core label-setting algorithm with Pareto frontier management
    fn label_setting_search(
        &self,
        destination: &EndpointId,
        contacts: Vec<&Contact>,
        start_time: u64,
        deadline: Option<u64>,
        privacy_cap: Option<f64>,
    ) -> Result<ParetoFrontier, RoutingError> {
        // Use a simple Vec instead of BinaryHeap to avoid Ord trait issues
        let mut queue: Vec<LabeledPath> = Vec::new();

        // Pareto frontiers for each node
        let mut frontiers: HashMap<EndpointId, ParetoFrontier> = HashMap::new();

        // Get local node from cached value
        let local_node = &self.local_node;

        // Initialize with local node
        let initial_cost = Cost::one(); // Identity for start
        let initial_path = LabeledPath::new(local_node.clone(), initial_cost, start_time);

        queue.push(initial_path);

        // Track nodes we've found optimal paths to
        let processed: HashMap<EndpointId, bool> = HashMap::new();

        while let Some(current_path) = self.pop_best_from_queue(&mut queue) {
            let current_node = &current_path.destination;
            // Skip if we've already processed this node optimally
            if *processed.get(&current_node).unwrap_or(&false) {
                continue;
            }

            // Get or create frontier for current node
            let frontier = frontiers
                .entry(current_node.clone())
                .or_insert_with(|| ParetoFrontier::new(self.frontier_cap));

            // Try to insert current path into frontier
            if !frontier.try_insert(current_path.clone()) {
                continue; // Path was dominated, skip expansion
            }

            // If this is the destination, we don't need to expand further for this path
            if *current_node == *destination {
                continue;
            }

            // Check timing constraints for expansion
            if let Some(dl) = deadline {
                if current_path.arrival_time >= dl {
                    continue; // Too late to reach deadline
                }
            }

            // Check privacy budget
            if let Some(cap) = privacy_cap {
                if current_path.privacy_consumed >= cap {
                    continue; // Privacy budget exhausted
                }
            }

            // Expand to neighboring nodes via contacts
            for contact in &contacts {
                if contact.from != *current_node || !contact.is_active(current_path.arrival_time) {
                    continue;
                }

                // Check if contact timing is feasible
                let earliest_departure = current_path.arrival_time.max(contact.start_time);
                if earliest_departure > contact.end_time {
                    continue; // Contact window missed
                }

                // Calculate edge cost (extends existing DTN Contact metrics)
                let edge_cost = self.calculate_edge_cost(contact, &current_path, privacy_cap);

                // Compose costs using semiring multiplication
                let new_cost = current_path.cost * edge_cost;

                // Check if new cost is still feasible
                if !new_cost.is_finite() {
                    continue;
                }

                // Create extended path
                let new_path = current_path.extend((*contact).clone(), new_cost);

                // Add to queue for further exploration
                queue.push(new_path);
            }
        }

        // Return frontier for destination
        frontiers
            .remove(destination)
            .ok_or_else(|| RoutingError::NoPathFound {
                destination: destination.clone(),
            })
    }

    /// Calculate edge cost from DTN contact, building on existing metrics
    fn calculate_edge_cost(
        &self,
        contact: &Contact,
        _current_path: &LabeledPath,
        privacy_cap: Option<f64>,
    ) -> Cost {
        // Base privacy cost - varies by transport type and mixnode usage
        let base_privacy_eps = self.estimate_privacy_cost(contact, privacy_cap);

        // Convert DTN contact to our cost structure
        // This bridges the existing Contact metrics with our semiring
        Cost::from_dtn_contact(contact, 1024, base_privacy_eps) // Assume 1KB bundle
    }

    /// Estimate privacy epsilon cost for a contact
    fn estimate_privacy_cost(
        &self,
        contact: &Contact,
        privacy_cap: Option<f64>,
    ) -> f64 {
        // Privacy cost depends on transport type and configuration
        // This would integrate with the privacy composer from existing code

        if privacy_cap.is_none() {
            return 0.0; // No privacy requirements
        }

        // Estimate based on contact characteristics
        let base_eps = 0.1; // Base privacy cost per hop

        // Higher energy cost contacts likely use more private routes
        let privacy_multiplier = if contact.energy_cost > 2.0 {
            0.5 // Mixnode routing - lower epsilon consumption
        } else {
            1.0 // Direct routing - higher epsilon consumption
        };

        base_eps * privacy_multiplier
    }

    /// Get active and future contacts from DTN router
    fn get_active_and_future_contacts(&self, _current_time: u64) -> Vec<&Contact> {
        // This would interface with the DTN router's contact plan
        // For now, return empty vec - would be populated from dtn_router.contact_plan
        Vec::new()
    }

    /// Check if cache is still valid
    fn is_cache_valid(&self, current_time: u64) -> bool {
        current_time - self.last_cleanup < self.cache_ttl.as_secs()
    }

    /// Check if we should cleanup cache
    fn should_cleanup_cache(&self, current_time: u64) -> bool {
        current_time - self.last_cleanup > self.cache_ttl.as_secs()
    }

    /// Clean up old cache entries
    fn cleanup_cache(&mut self, current_time: u64) {
        self.frontier_cache.clear(); // Simple cleanup - could be more sophisticated
        self.last_cleanup = current_time;
    }

    /// Get current timestamp
    fn current_time() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Get underlying DTN router reference
    pub fn dtn_router(&self) -> &ContactGraphRouter {
        &self.dtn_router
    }

    /// Get mutable DTN router reference
    pub fn dtn_router_mut(&mut self) -> &mut ContactGraphRouter {
        &mut self.dtn_router
    }

    /// Pop the best (lowest scalarized cost) path from the queue
    fn pop_best_from_queue(&self, queue: &mut Vec<LabeledPath>) -> Option<LabeledPath> {
        if queue.is_empty() {
            return None;
        }

        let weights = WeightVector::balanced();
        let mut best_idx = 0;
        let mut best_score = queue[0].cost.scalarize(&weights);

        for (i, path) in queue.iter().enumerate().skip(1) {
            let score = path.cost.scalarize(&weights);
            if score < best_score {
                best_score = score;
                best_idx = i;
            }
        }

        Some(queue.swap_remove(best_idx))
    }
}

/// QoS requirements for path selection
#[derive(Debug, Clone)]
pub struct QosRequirements {
    /// Priority for low latency (0.0 - 1.0)
    pub latency_priority: f64,
    /// Priority for energy efficiency (0.0 - 1.0)
    pub energy_priority: f64,
    /// Priority for reliability (0.0 - 1.0)
    pub reliability_priority: f64,
    /// Priority for privacy (0.0 - 1.0)
    pub privacy_priority: f64,
}

impl QosRequirements {
    /// Create balanced QoS requirements
    pub fn balanced() -> Self {
        Self {
            latency_priority: 0.25,
            energy_priority: 0.25,
            reliability_priority: 0.25,
            privacy_priority: 0.25,
        }
    }

    /// Create real-time QoS (prioritize latency)
    pub fn real_time() -> Self {
        Self {
            latency_priority: 0.6,
            energy_priority: 0.1,
            reliability_priority: 0.2,
            privacy_priority: 0.1,
        }
    }

    /// Create energy-efficient QoS
    pub fn energy_efficient() -> Self {
        Self {
            latency_priority: 0.1,
            energy_priority: 0.6,
            reliability_priority: 0.2,
            privacy_priority: 0.1,
        }
    }

    /// Create privacy-first QoS
    pub fn privacy_first() -> Self {
        Self {
            latency_priority: 0.1,
            energy_priority: 0.1,
            reliability_priority: 0.2,
            privacy_priority: 0.6,
        }
    }

    /// Convert to weight vector for scalarization
    pub fn to_weight_vector(&self) -> WeightVector {
        WeightVector::new(
            self.latency_priority,
            self.energy_priority,
            self.reliability_priority,
            self.privacy_priority,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use betanet_dtn::EndpointId;

    #[test]
    fn test_pareto_frontier_basic() {
        let mut frontier = ParetoFrontier::new(3);

        let dest = EndpointId::node("test");
        let path1 = LabeledPath::new(dest.clone(), Cost::new(100.0, 5.0, 0.9, 0.1), 1000);
        let path2 = LabeledPath::new(dest.clone(), Cost::new(50.0, 10.0, 0.8, 0.2), 1000);
        let path3 = LabeledPath::new(dest.clone(), Cost::new(200.0, 2.0, 0.95, 0.05), 1000);

        assert!(frontier.try_insert(path1));
        assert!(frontier.try_insert(path2));
        assert!(frontier.try_insert(path3));

        assert_eq!(frontier.len(), 3);
    }

    #[test]
    fn test_pareto_frontier_dominance() {
        let mut frontier = ParetoFrontier::new(5);

        let dest = EndpointId::node("test");
        let better = LabeledPath::new(dest.clone(), Cost::new(50.0, 5.0, 0.9, 0.1), 1000);
        let worse = LabeledPath::new(dest.clone(), Cost::new(100.0, 10.0, 0.8, 0.2), 1000);

        assert!(frontier.try_insert(better));
        assert!(!frontier.try_insert(worse)); // Should be dominated

        assert_eq!(frontier.len(), 1);
    }

    #[test]
    fn test_qos_weight_conversion() {
        let qos = QosRequirements::real_time();
        let weights = qos.to_weight_vector();

        assert!(weights.latency > 0.5); // Should prioritize latency
        assert!(weights.energy < 0.2);  // Should de-prioritize energy
    }

    #[test]
    fn test_labeled_path_extension() {
        let dest1 = EndpointId::node("node1");
        let dest2 = EndpointId::node("node2");

        let initial_path = LabeledPath::new(dest1.clone(), Cost::new(50.0, 2.0, 0.95, 0.05), 1000);

        let contact = Contact::new(dest1, dest2.clone(), 1000, 2000, 1_000_000);
        let new_cost = Cost::new(100.0, 5.0, 0.90, 0.10);

        let extended = initial_path.extend(contact, new_cost);

        assert_eq!(extended.destination, dest2);
        assert_eq!(extended.cost, new_cost);
        assert_eq!(extended.hop_count(), 1);
        assert_eq!(extended.arrival_time, 2000);
    }
}
