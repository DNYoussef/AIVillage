//! Contact Graph Routing for DTN
//!
//! Implements contact graph routing with support for delay budgets, energy constraints,
//! and custody probability optimization.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::bundle::{Bundle, EndpointId};

/// Contact information between nodes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Contact {
    pub from: EndpointId,
    pub to: EndpointId,
    pub start_time: u64, // Unix timestamp
    pub end_time: u64,   // Unix timestamp
    pub data_rate: u64,  // bits per second
    pub latency: Duration,
    pub reliability: f64, // 0.0 to 1.0
    pub energy_cost: f64, // Relative energy cost
}

impl Contact {
    pub fn new(
        from: EndpointId,
        to: EndpointId,
        start_time: u64,
        end_time: u64,
        data_rate: u64,
    ) -> Self {
        Self {
            from,
            to,
            start_time,
            end_time,
            data_rate,
            latency: Duration::from_millis(10), // Default 10ms latency
            reliability: 0.95,                  // Default 95% reliability
            energy_cost: 1.0,                   // Default energy cost
        }
    }

    pub fn is_active(&self, time: u64) -> bool {
        time >= self.start_time && time <= self.end_time
    }

    pub fn duration(&self) -> Duration {
        Duration::from_secs(self.end_time.saturating_sub(self.start_time))
    }

    pub fn transmission_time(&self, bytes: usize) -> Duration {
        let bits = bytes * 8;
        let seconds = bits as f64 / self.data_rate as f64;
        Duration::from_secs_f64(seconds)
    }

    pub fn delivery_probability(&self) -> f64 {
        self.reliability
    }
}

/// Contact plan containing all known contacts
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContactPlan {
    contacts: Vec<Contact>,
    last_updated: u64,
}

impl ContactPlan {
    pub fn new() -> Self {
        Self {
            contacts: Vec::new(),
            last_updated: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    pub fn add_contact(&mut self, contact: Contact) {
        self.contacts.push(contact);
        self.last_updated = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }

    pub fn get_active_contacts(&self, time: u64) -> Vec<&Contact> {
        self.contacts.iter().filter(|c| c.is_active(time)).collect()
    }

    pub fn get_future_contacts(&self, from: &EndpointId, time: u64) -> Vec<&Contact> {
        self.contacts
            .iter()
            .filter(|c| c.from == *from && c.start_time > time)
            .collect()
    }

    pub fn remove_expired_contacts(&mut self, time: u64) {
        self.contacts.retain(|c| c.end_time > time);
        self.last_updated = time;
    }
}

/// Routing policies for path selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingPolicy {
    pub max_delay: Duration,
    pub max_energy_cost: f64,
    pub min_delivery_probability: f64,
    pub prefer_direct_paths: bool,
    pub max_hops: u32,
    pub custody_threshold: f64, // Probability threshold for requesting custody
}

impl Default for RoutingPolicy {
    fn default() -> Self {
        Self {
            max_delay: Duration::from_secs(24 * 60 * 60), // 24 hours
            max_energy_cost: 10.0,
            min_delivery_probability: 0.8,
            prefer_direct_paths: true,
            max_hops: 10,
            custody_threshold: 0.9,
        }
    }
}

/// Route information
#[derive(Debug, Clone, PartialEq)]
pub struct Route {
    pub hops: Vec<EndpointId>,
    pub contacts: Vec<Contact>,
    pub total_delay: Duration,
    pub total_energy_cost: f64,
    pub delivery_probability: f64,
    pub earliest_departure: u64,
    pub latest_arrival: u64,
}

impl Default for Route {
    fn default() -> Self {
        Self::new()
    }
}

impl Route {
    pub fn new() -> Self {
        Self {
            hops: Vec::new(),
            contacts: Vec::new(),
            total_delay: Duration::ZERO,
            total_energy_cost: 0.0,
            delivery_probability: 1.0,
            earliest_departure: 0,
            latest_arrival: 0,
        }
    }

    pub fn add_hop(&mut self, contact: Contact) {
        if self.hops.is_empty() {
            self.hops.push(contact.from.clone());
        }
        self.hops.push(contact.to.clone());

        // Update metrics
        self.total_delay += contact.latency + contact.transmission_time(1024); // Estimate with 1KB
        self.total_energy_cost += contact.energy_cost;
        self.delivery_probability *= contact.delivery_probability();

        if self.earliest_departure == 0 {
            self.earliest_departure = contact.start_time;
        }
        self.latest_arrival = contact.end_time;

        self.contacts.push(contact);
    }

    pub fn hop_count(&self) -> usize {
        self.contacts.len()
    }

    pub fn is_valid_for_policy(&self, policy: &RoutingPolicy) -> bool {
        self.total_delay <= policy.max_delay
            && self.total_energy_cost <= policy.max_energy_cost
            && self.delivery_probability >= policy.min_delivery_probability
            && self.hop_count() as u32 <= policy.max_hops
    }
}

/// Contact Graph Router implementing Dijkstra-based routing
pub struct ContactGraphRouter {
    contact_plan: ContactPlan,
    local_node: EndpointId,
    policy: RoutingPolicy,
    route_cache: HashMap<(EndpointId, u64), Vec<Route>>, // (destination, time) -> routes
    cache_timeout: Duration,
}

impl ContactGraphRouter {
    pub fn new(local_node: EndpointId, policy: RoutingPolicy) -> Self {
        Self {
            contact_plan: ContactPlan::new(),
            local_node,
            policy,
            route_cache: HashMap::new(),
            cache_timeout: Duration::from_secs(5 * 60), // 5 minutes
        }
    }

    pub fn update_contact_plan(&mut self, plan: ContactPlan) {
        self.contact_plan = plan;
        self.route_cache.clear(); // Invalidate cache
    }

    pub fn add_contact(&mut self, contact: Contact) {
        self.contact_plan.add_contact(contact);
        self.route_cache.clear(); // Invalidate cache
    }

    pub fn set_policy(&mut self, policy: RoutingPolicy) {
        self.policy = policy;
        self.route_cache.clear(); // Invalidate cache
    }

    /// Find routes to destination using contact graph routing
    pub fn find_routes(
        &mut self,
        destination: &EndpointId,
        bundle_size: usize,
        current_time: u64,
    ) -> Result<Vec<Route>, RoutingError> {
        // Check cache first
        let cache_key = (destination.clone(), current_time / 60); // Cache per minute
        if let Some(cached_routes) = self.route_cache.get(&cache_key) {
            return Ok(cached_routes.clone());
        }

        let routes = self.compute_routes(destination, bundle_size, current_time)?;

        // Cache the results
        self.route_cache.insert(cache_key, routes.clone());

        Ok(routes)
    }

    /// Find the best route to destination
    pub fn find_best_route(
        &mut self,
        destination: &EndpointId,
        bundle_size: usize,
        current_time: u64,
    ) -> Result<Option<Route>, RoutingError> {
        let routes = self.find_routes(destination, bundle_size, current_time)?;

        if routes.is_empty() {
            return Ok(None);
        }

        // Score routes based on policy preferences
        let best_route = routes
            .into_iter()
            .filter(|r| r.is_valid_for_policy(&self.policy))
            .min_by(|a, b| {
                let score_a = self.score_route(a);
                let score_b = self.score_route(b);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        Ok(best_route)
    }

    /// Check if bundle should request custody transfer
    pub fn should_request_custody(&self, route: &Route) -> bool {
        route.delivery_probability < self.policy.custody_threshold
    }

    /// Get next hop for a bundle
    pub fn get_next_hop(
        &mut self,
        bundle: &Bundle,
        current_time: u64,
    ) -> Result<Option<EndpointId>, RoutingError> {
        let bundle_size = bundle.size();

        if let Some(route) =
            self.find_best_route(&bundle.primary.destination, bundle_size, current_time)?
        {
            // Find the next hop from current node
            for (i, hop) in route.hops.iter().enumerate() {
                if *hop == self.local_node && i + 1 < route.hops.len() {
                    return Ok(Some(route.hops[i + 1].clone()));
                }
            }
        }

        Ok(None)
    }

    /// Cleanup expired contacts and cache entries
    pub fn cleanup(&mut self, current_time: u64) {
        self.contact_plan.remove_expired_contacts(current_time);

        // Remove expired cache entries
        let cache_cutoff = current_time.saturating_sub(self.cache_timeout.as_secs());
        self.route_cache
            .retain(|(_, time), _| *time >= cache_cutoff / 60);
    }

    /// Get active contacts at the given time
    pub fn get_active_contacts(&self, current_time: u64) -> Vec<Contact> {
        self.contact_plan
            .get_active_contacts(current_time)
            .into_iter()
            .cloned()
            .collect()
    }

    // Private methods

    fn compute_routes(
        &self,
        destination: &EndpointId,
        bundle_size: usize,
        current_time: u64,
    ) -> Result<Vec<Route>, RoutingError> {
        let mut routes = Vec::new();

        // Direct routes first if preferred
        if self.policy.prefer_direct_paths {
            if let Some(direct_route) =
                self.find_direct_route(destination, bundle_size, current_time)?
            {
                routes.push(direct_route);
            }
        }

        // Multi-hop routes using modified Dijkstra
        let multi_hop_routes =
            self.find_multi_hop_routes(destination, bundle_size, current_time)?;
        routes.extend(multi_hop_routes);

        // Sort by score
        routes.sort_by(|a, b| {
            let score_a = self.score_route(a);
            let score_b = self.score_route(b);
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return top routes
        routes.truncate(5); // Limit to 5 best routes

        Ok(routes)
    }

    fn find_direct_route(
        &self,
        destination: &EndpointId,
        bundle_size: usize,
        current_time: u64,
    ) -> Result<Option<Route>, RoutingError> {
        // Look for direct contacts from local node to destination
        let contacts = self.contact_plan.get_active_contacts(current_time);

        for contact in contacts {
            if contact.from == self.local_node && contact.to == *destination {
                let transmission_time = contact.transmission_time(bundle_size);
                if contact.start_time + transmission_time.as_secs() <= contact.end_time {
                    let mut route = Route::new();
                    route.add_hop(contact.clone());
                    return Ok(Some(route));
                }
            }
        }

        // Look for future direct contacts
        let future_contacts = self
            .contact_plan
            .get_future_contacts(&self.local_node, current_time);

        for contact in future_contacts {
            if contact.to == *destination {
                let transmission_time = contact.transmission_time(bundle_size);
                if contact.start_time + transmission_time.as_secs() <= contact.end_time {
                    let mut route = Route::new();
                    route.add_hop(contact.clone());
                    return Ok(Some(route));
                }
            }
        }

        Ok(None)
    }

    fn find_multi_hop_routes(
        &self,
        destination: &EndpointId,
        _bundle_size: usize,
        current_time: u64,
    ) -> Result<Vec<Route>, RoutingError> {
        let mut routes = Vec::new();

        // Simple BFS for multi-hop routing (could be enhanced with Dijkstra)
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();

        // Start from local node
        queue.push_back((self.local_node.clone(), Route::new(), current_time));

        while let Some((current_node, current_route, current_time)) = queue.pop_front() {
            if current_route.hop_count() >= self.policy.max_hops as usize {
                continue;
            }

            // Find contacts from current node
            let available_contacts: Vec<_> = self
                .contact_plan
                .contacts
                .iter()
                .filter(|c| c.from == current_node && c.is_active(current_time))
                .collect();

            for contact in available_contacts {
                if visited.contains(&(contact.to.clone(), contact.start_time)) {
                    continue;
                }

                let mut new_route = current_route.clone();
                new_route.add_hop(contact.clone());

                // Check if we reached destination
                if contact.to == *destination {
                    if new_route.is_valid_for_policy(&self.policy) {
                        routes.push(new_route);
                    }
                    continue;
                }

                // Check if route is still viable
                if new_route.is_valid_for_policy(&self.policy) {
                    visited.insert((contact.to.clone(), contact.start_time));
                    queue.push_back((contact.to.clone(), new_route, contact.end_time));
                }
            }
        }

        Ok(routes)
    }

    fn score_route(&self, route: &Route) -> f64 {
        // Multi-objective scoring function
        let delay_score = route.total_delay.as_secs_f64() / self.policy.max_delay.as_secs_f64();
        let energy_score = route.total_energy_cost / self.policy.max_energy_cost;
        let reliability_score = 1.0 - route.delivery_probability;
        let hop_score = route.hop_count() as f64 / self.policy.max_hops as f64;

        // Weighted combination (these weights could be configurable)
        0.4 * delay_score + 0.3 * energy_score + 0.2 * reliability_score + 0.1 * hop_score
    }
}

#[derive(Debug, Error)]
pub enum RoutingError {
    #[error("No route to destination: {0}")]
    NoRoute(EndpointId),

    #[error("Route calculation failed: {0}")]
    CalculationFailed(String),

    #[error("Invalid contact plan")]
    InvalidContactPlan,

    #[error("Policy constraints too restrictive")]
    PolicyTooRestrictive,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contact_creation() {
        let node1 = EndpointId::node("node1");
        let node2 = EndpointId::node("node2");

        let contact = Contact::new(node1, node2, 1000, 2000, 1_000_000);

        assert_eq!(contact.duration(), Duration::from_secs(1000));
        assert!(contact.is_active(1500));
        assert!(!contact.is_active(500));
        assert!(!contact.is_active(2500));
    }

    #[test]
    fn test_contact_plan() {
        let mut plan = ContactPlan::new();

        let node1 = EndpointId::node("node1");
        let node2 = EndpointId::node("node2");
        let contact = Contact::new(node1.clone(), node2, 1000, 2000, 1_000_000);

        plan.add_contact(contact);

        let active = plan.get_active_contacts(1500);
        assert_eq!(active.len(), 1);

        let active = plan.get_active_contacts(500);
        assert_eq!(active.len(), 0);
    }

    #[test]
    fn test_route_metrics() {
        let mut route = Route::new();

        let node1 = EndpointId::node("node1");
        let node2 = EndpointId::node("node2");
        let node3 = EndpointId::node("node3");

        let contact1 = Contact::new(node1, node2.clone(), 1000, 2000, 1_000_000);
        let contact2 = Contact::new(node2, node3, 1500, 2500, 1_000_000);

        route.add_hop(contact1);
        route.add_hop(contact2);

        assert_eq!(route.hop_count(), 2);
        assert_eq!(route.hops.len(), 3); // Including source
        assert!(route.total_delay > Duration::ZERO);
        assert!(route.delivery_probability < 1.0);
    }

    #[test]
    fn test_routing_policy() {
        let policy = RoutingPolicy::default();

        let mut route = Route::new();
        let node1 = EndpointId::node("node1");
        let node2 = EndpointId::node("node2");
        let contact = Contact::new(node1, node2, 1000, 2000, 1_000_000);

        route.add_hop(contact);

        assert!(route.is_valid_for_policy(&policy));

        // Test with restrictive policy
        let restrictive_policy = RoutingPolicy {
            max_delay: Duration::from_millis(1),
            ..policy
        };

        assert!(!route.is_valid_for_policy(&restrictive_policy));
    }

    #[test]
    fn test_contact_graph_router() {
        let local_node = EndpointId::node("node1");
        let destination = EndpointId::node("node3");
        let mut router = ContactGraphRouter::new(local_node.clone(), RoutingPolicy::default());

        // Add contacts for multi-hop route
        let node2 = EndpointId::node("node2");
        let contact1 = Contact::new(local_node, node2.clone(), 1000, 2000, 1_000_000);
        let contact2 = Contact::new(node2, destination.clone(), 1500, 2500, 1_000_000);

        router.add_contact(contact1);
        router.add_contact(contact2);

        let routes = router.find_routes(&destination, 1024, 1000).unwrap();
        assert!(!routes.is_empty());

        let best_route = router.find_best_route(&destination, 1024, 1000).unwrap();
        assert!(best_route.is_some());
    }
}
