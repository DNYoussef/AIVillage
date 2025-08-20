//! Privacy Accounting with Epsilon Budget Management
//!
//! Implements additive privacy accounting for differential privacy:
//! - Edge-wise epsilon estimation
//! - Route composition and accumulation
//! - Privacy budget enforcement
//! - Leak parameter tracking

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::SystemTime;
use thiserror::Error;

/// Privacy accounting errors
#[derive(Debug, Error)]
pub enum PrivacyError {
    #[error("Privacy budget exceeded: requested {requested}, available {available}")]
    BudgetExceeded { requested: f64, available: f64 },

    #[error("Invalid epsilon value: {0} (must be positive and finite)")]
    InvalidEpsilon(f64),

    #[error("Route composition failed: {0}")]
    RouteComposition(String),

    #[error("Edge not found: {0}")]
    EdgeNotFound(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, PrivacyError>;

/// Edge identifier in network routing
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EdgeId {
    pub from_node: String,
    pub to_node: String,
    pub edge_type: EdgeType,
}

impl EdgeId {
    pub fn new(from: impl Into<String>, to: impl Into<String>, edge_type: EdgeType) -> Self {
        Self {
            from_node: from.into(),
            to_node: to.into(),
            edge_type,
        }
    }
}

impl std::fmt::Display for EdgeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}->{}({})",
            self.from_node,
            self.to_node,
            self.edge_type.as_str()
        )
    }
}

/// Types of network edges with different privacy characteristics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    DirectConnection, // Direct peer-to-peer connection
    RelayConnection,  // Through relay/proxy
    MixnetConnection, // Through mixnet layer
    TorConnection,    // Through Tor-like onion routing
    CdnConnection,    // Content delivery network
    VpnConnection,    // VPN tunnel
}

impl EdgeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DirectConnection => "direct",
            Self::RelayConnection => "relay",
            Self::MixnetConnection => "mixnet",
            Self::TorConnection => "tor",
            Self::CdnConnection => "cdn",
            Self::VpnConnection => "vpn",
        }
    }

    pub fn base_epsilon(&self) -> f64 {
        match self {
            Self::DirectConnection => 0.5, // High privacy leak
            Self::RelayConnection => 0.3,  // Medium privacy leak
            Self::MixnetConnection => 0.1, // Low privacy leak
            Self::TorConnection => 0.05,   // Very low privacy leak
            Self::CdnConnection => 0.4,    // Medium-high privacy leak
            Self::VpnConnection => 0.2,    // Medium-low privacy leak
        }
    }
}

/// Privacy parameters for an edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgePrivacy {
    pub edge_id: EdgeId,
    pub base_epsilon: f64,
    pub traffic_analysis_factor: f64,
    pub timing_correlation_factor: f64,
    pub metadata_leak_factor: f64,
    pub last_updated: SystemTime,
    pub confidence_level: f64, // 0.0 to 1.0
}

impl EdgePrivacy {
    pub fn new(edge_id: EdgeId) -> Self {
        let base_epsilon = edge_id.edge_type.base_epsilon();

        Self {
            edge_id,
            base_epsilon,
            traffic_analysis_factor: 1.0,
            timing_correlation_factor: 1.0,
            metadata_leak_factor: 1.0,
            last_updated: SystemTime::now(),
            confidence_level: 0.8, // Default confidence
        }
    }

    /// Estimate epsilon for this edge based on current conditions
    pub fn estimate_epsilon(&self, traffic_volume: u64, connection_duration: f64) -> f64 {
        // Base epsilon from edge type
        let mut epsilon = self.base_epsilon;

        // Traffic analysis factor: more traffic = more identifiable patterns
        let traffic_factor = 1.0 + (traffic_volume as f64).ln() / 1000.0;
        epsilon *= traffic_factor * self.traffic_analysis_factor;

        // Timing correlation: longer connections = more timing attacks
        let timing_factor = 1.0 + connection_duration / 3600.0; // per hour
        epsilon *= timing_factor * self.timing_correlation_factor;

        // Metadata leaks
        epsilon *= self.metadata_leak_factor;

        // Clamp to reasonable bounds
        epsilon.max(0.001).min(10.0)
    }

    /// Update privacy parameters based on observed conditions
    pub fn update_parameters(
        &mut self,
        traffic_analysis: f64,
        timing_correlation: f64,
        metadata_leak: f64,
        confidence: f64,
    ) {
        // Apply exponential smoothing for stability
        let alpha = 0.3; // Smoothing factor

        self.traffic_analysis_factor =
            alpha * traffic_analysis + (1.0 - alpha) * self.traffic_analysis_factor;
        self.timing_correlation_factor =
            alpha * timing_correlation + (1.0 - alpha) * self.timing_correlation_factor;
        self.metadata_leak_factor =
            alpha * metadata_leak + (1.0 - alpha) * self.metadata_leak_factor;

        self.confidence_level = confidence.clamp(0.0, 1.0);
        self.last_updated = SystemTime::now();
    }
}

/// Route privacy composition
#[derive(Debug, Clone)]
pub struct RoutePrivacy {
    pub route_id: String,
    pub edges: Vec<EdgeId>,
    pub total_epsilon: f64,
    pub composition_method: CompositionMethod,
    pub created_at: SystemTime,
    pub metadata: RouteMetadata,
}

#[derive(Debug, Clone)]
pub struct RouteMetadata {
    pub expected_latency: f64,
    pub bandwidth_estimate: f64,
    pub reliability_score: f64,
    pub anonymity_set_size: usize,
}

/// Methods for composing privacy parameters across route edges
#[derive(Debug, Clone, Copy)]
pub enum CompositionMethod {
    Additive,   // Sum of epsilons
    Sequential, // Product composition
    Advanced,   // RDP (Rényi Differential Privacy) composition
}

impl RoutePrivacy {
    pub fn new(route_id: String, edges: Vec<EdgeId>) -> Self {
        Self {
            route_id,
            edges,
            total_epsilon: 0.0,
            composition_method: CompositionMethod::Additive,
            created_at: SystemTime::now(),
            metadata: RouteMetadata {
                expected_latency: 0.0,
                bandwidth_estimate: 0.0,
                reliability_score: 1.0,
                anonymity_set_size: 0,
            },
        }
    }

    /// Compose privacy parameters across route
    pub fn compose_privacy(
        &mut self,
        edge_privacy: &HashMap<EdgeId, EdgePrivacy>,
        traffic_volume: u64,
        connection_duration: f64,
    ) -> Result<()> {
        match self.composition_method {
            CompositionMethod::Additive => {
                self.compose_additive(edge_privacy, traffic_volume, connection_duration)
            }
            CompositionMethod::Sequential => {
                self.compose_sequential(edge_privacy, traffic_volume, connection_duration)
            }
            CompositionMethod::Advanced => {
                self.compose_advanced(edge_privacy, traffic_volume, connection_duration)
            }
        }
    }

    fn compose_additive(
        &mut self,
        edge_privacy: &HashMap<EdgeId, EdgePrivacy>,
        traffic_volume: u64,
        connection_duration: f64,
    ) -> Result<()> {
        // Simple additive composition: ε_total = Σ ε_i
        let mut total_epsilon = 0.0;

        for edge_id in &self.edges {
            let edge_priv = edge_privacy
                .get(edge_id)
                .ok_or_else(|| PrivacyError::EdgeNotFound(edge_id.to_string()))?;

            let edge_epsilon = edge_priv.estimate_epsilon(traffic_volume, connection_duration);
            total_epsilon += edge_epsilon;
        }

        self.total_epsilon = total_epsilon;
        Ok(())
    }

    fn compose_sequential(
        &mut self,
        edge_privacy: &HashMap<EdgeId, EdgePrivacy>,
        traffic_volume: u64,
        connection_duration: f64,
    ) -> Result<()> {
        // Sequential composition with correlation factors
        let mut composed_epsilon = 0.0;

        for (i, edge_id) in self.edges.iter().enumerate() {
            let edge_priv = edge_privacy
                .get(edge_id)
                .ok_or_else(|| PrivacyError::EdgeNotFound(edge_id.to_string()))?;

            let edge_epsilon = edge_priv.estimate_epsilon(traffic_volume, connection_duration);

            // Apply composition formula: stronger composition for correlated events
            let correlation_factor = if i == 0 { 1.0 } else { 1.2 }; // Subsequent edges add more
            composed_epsilon += edge_epsilon * correlation_factor;
        }

        self.total_epsilon = composed_epsilon;
        Ok(())
    }

    fn compose_advanced(
        &mut self,
        edge_privacy: &HashMap<EdgeId, EdgePrivacy>,
        traffic_volume: u64,
        connection_duration: f64,
    ) -> Result<()> {
        // Advanced composition using RDP approximation
        let alpha = 2.0; // Rényi parameter
        let mut total_alpha_epsilon = 0.0;

        for edge_id in &self.edges {
            let edge_priv = edge_privacy
                .get(edge_id)
                .ok_or_else(|| PrivacyError::EdgeNotFound(edge_id.to_string()))?;

            let edge_epsilon = edge_priv.estimate_epsilon(traffic_volume, connection_duration);

            // RDP composition: sum of α-epsilons
            total_alpha_epsilon += edge_epsilon.powi(2) / (2.0 * alpha); // Simplified RDP
        }

        // Convert back to (ε, δ)-DP
        let delta = 1e-5; // Small delta
        self.total_epsilon = total_alpha_epsilon + (1.0_f64 / delta).ln() / (alpha - 1.0);

        Ok(())
    }
}

/// Privacy budget manager
pub struct PrivacyBudgetManager {
    epsilon_max: f64,
    epsilon_used: f64,
    edge_privacy: HashMap<EdgeId, EdgePrivacy>,
    active_routes: HashMap<String, RoutePrivacy>,
    budget_history: VecDeque<BudgetTransaction>,
    max_history: usize,
    policy: PrivacyPolicy,
}

/// Budget transaction record
#[derive(Debug, Clone)]
pub struct BudgetTransaction {
    pub timestamp: SystemTime,
    pub route_id: String,
    pub epsilon_requested: f64,
    pub epsilon_granted: f64,
    pub action: BudgetAction,
}

#[derive(Debug, Clone)]
pub enum BudgetAction {
    Allocated,
    Released,
    Denied,
}

/// Privacy policy configuration
#[derive(Debug, Clone)]
pub struct PrivacyPolicy {
    pub epsilon_max_per_hour: f64,
    pub epsilon_max_per_route: f64,
    pub epsilon_max_total: f64,
    pub allow_budget_overdraft: bool,
    pub reset_period_hours: f64,
}

impl Default for PrivacyPolicy {
    fn default() -> Self {
        Self {
            epsilon_max_per_hour: 1.0,
            epsilon_max_per_route: 0.5,
            epsilon_max_total: 10.0,
            allow_budget_overdraft: false,
            reset_period_hours: 24.0,
        }
    }
}

impl PrivacyBudgetManager {
    pub fn new(epsilon_max: f64) -> Result<Self> {
        if epsilon_max <= 0.0 || !epsilon_max.is_finite() {
            return Err(PrivacyError::InvalidEpsilon(epsilon_max));
        }

        Ok(Self {
            epsilon_max,
            epsilon_used: 0.0,
            edge_privacy: HashMap::new(),
            active_routes: HashMap::new(),
            budget_history: VecDeque::new(),
            max_history: 1000,
            policy: PrivacyPolicy::default(),
        })
    }

    pub fn with_policy(epsilon_max: f64, policy: PrivacyPolicy) -> Result<Self> {
        let mut manager = Self::new(epsilon_max)?;
        manager.policy = policy;
        Ok(manager)
    }

    /// Register edge with privacy parameters
    pub fn register_edge(&mut self, edge_id: EdgeId, edge_privacy: EdgePrivacy) {
        self.edge_privacy.insert(edge_id, edge_privacy);
    }

    /// Request privacy budget for a route
    pub fn request_budget(
        &mut self,
        route_id: String,
        edges: Vec<EdgeId>,
        traffic_volume: u64,
        connection_duration: f64,
    ) -> Result<f64> {
        // Create route privacy composition
        let mut route_privacy = RoutePrivacy::new(route_id.clone(), edges);
        route_privacy.compose_privacy(&self.edge_privacy, traffic_volume, connection_duration)?;

        let requested_epsilon = route_privacy.total_epsilon;

        // Check policy constraints
        if requested_epsilon > self.policy.epsilon_max_per_route {
            self.record_transaction(route_id, requested_epsilon, 0.0, BudgetAction::Denied);
            return Err(PrivacyError::BudgetExceeded {
                requested: requested_epsilon,
                available: self.policy.epsilon_max_per_route,
            });
        }

        // Check total budget
        let available_budget = self.epsilon_max - self.epsilon_used;
        if requested_epsilon > available_budget && !self.policy.allow_budget_overdraft {
            self.record_transaction(route_id, requested_epsilon, 0.0, BudgetAction::Denied);
            return Err(PrivacyError::BudgetExceeded {
                requested: requested_epsilon,
                available: available_budget,
            });
        }

        // Allocate budget
        self.epsilon_used += requested_epsilon;
        self.active_routes.insert(route_id.clone(), route_privacy);
        self.record_transaction(
            route_id,
            requested_epsilon,
            requested_epsilon,
            BudgetAction::Allocated,
        );

        Ok(requested_epsilon)
    }

    /// Release privacy budget when route completes
    pub fn release_budget(&mut self, route_id: &str) -> Result<f64> {
        let route_privacy = self.active_routes.remove(route_id).ok_or_else(|| {
            PrivacyError::RouteComposition(format!("Route {} not found", route_id))
        })?;

        let released_epsilon = route_privacy.total_epsilon;
        self.epsilon_used -= released_epsilon;
        self.record_transaction(
            route_id.to_string(),
            released_epsilon,
            released_epsilon,
            BudgetAction::Released,
        );

        Ok(released_epsilon)
    }

    /// Get current budget status
    pub fn budget_status(&self) -> BudgetStatus {
        BudgetStatus {
            epsilon_max: self.epsilon_max,
            epsilon_used: self.epsilon_used,
            epsilon_available: self.epsilon_max - self.epsilon_used,
            utilization: self.epsilon_used / self.epsilon_max,
            active_routes: self.active_routes.len(),
            total_transactions: self.budget_history.len(),
        }
    }

    /// Update edge privacy parameters based on observations
    pub fn update_edge_privacy(
        &mut self,
        edge_id: &EdgeId,
        traffic_analysis: f64,
        timing_correlation: f64,
        metadata_leak: f64,
        confidence: f64,
    ) -> Result<()> {
        let edge_privacy = self
            .edge_privacy
            .get_mut(edge_id)
            .ok_or_else(|| PrivacyError::EdgeNotFound(edge_id.to_string()))?;

        edge_privacy.update_parameters(
            traffic_analysis,
            timing_correlation,
            metadata_leak,
            confidence,
        );
        Ok(())
    }

    /// Reset budget based on policy
    pub fn reset_budget(&mut self) {
        self.epsilon_used = 0.0;
        self.active_routes.clear();

        // Keep transaction history for analysis
        self.record_transaction("system".to_string(), 0.0, 0.0, BudgetAction::Released);
    }

    fn record_transaction(
        &mut self,
        route_id: String,
        epsilon_requested: f64,
        epsilon_granted: f64,
        action: BudgetAction,
    ) {
        let transaction = BudgetTransaction {
            timestamp: SystemTime::now(),
            route_id,
            epsilon_requested,
            epsilon_granted,
            action,
        };

        self.budget_history.push_back(transaction);

        // Limit history size
        while self.budget_history.len() > self.max_history {
            self.budget_history.pop_front();
        }
    }

    /// Get budget utilization over time
    pub fn utilization_history(&self, hours: f64) -> Vec<(SystemTime, f64)> {
        let cutoff = SystemTime::now() - std::time::Duration::from_secs_f64(hours * 3600.0);

        self.budget_history
            .iter()
            .filter(|tx| tx.timestamp >= cutoff)
            .map(|tx| (tx.timestamp, tx.epsilon_granted))
            .collect()
    }

    /// Check if route is within policy limits
    pub fn check_policy_compliance(&self, epsilon: f64) -> Result<()> {
        if epsilon > self.policy.epsilon_max_per_route {
            return Err(PrivacyError::BudgetExceeded {
                requested: epsilon,
                available: self.policy.epsilon_max_per_route,
            });
        }

        let available = self.epsilon_max - self.epsilon_used;
        if epsilon > available && !self.policy.allow_budget_overdraft {
            return Err(PrivacyError::BudgetExceeded {
                requested: epsilon,
                available,
            });
        }

        Ok(())
    }
}

/// Budget status information
#[derive(Debug, Clone)]
pub struct BudgetStatus {
    pub epsilon_max: f64,
    pub epsilon_used: f64,
    pub epsilon_available: f64,
    pub utilization: f64,
    pub active_routes: usize,
    pub total_transactions: usize,
}

impl BudgetStatus {
    pub fn is_low(&self) -> bool {
        self.utilization > 0.8
    }

    pub fn is_critical(&self) -> bool {
        self.utilization > 0.95
    }
}

/// Utility functions for privacy analysis
pub mod analysis {
    use super::*;

    /// Calculate optimal epsilon allocation for multiple routes
    pub fn optimize_epsilon_allocation(
        routes: &[RoutePrivacy],
        total_budget: f64,
    ) -> HashMap<String, f64> {
        let mut allocations = HashMap::new();

        if routes.is_empty() {
            return allocations;
        }

        // Simple proportional allocation based on route epsilon requirements
        let total_requested: f64 = routes.iter().map(|r| r.total_epsilon).sum();

        if total_requested <= total_budget {
            // Allocate exactly what each route needs
            for route in routes {
                allocations.insert(route.route_id.clone(), route.total_epsilon);
            }
        } else {
            // Proportional scaling
            let scale_factor = total_budget / total_requested;
            for route in routes {
                allocations.insert(route.route_id.clone(), route.total_epsilon * scale_factor);
            }
        }

        allocations
    }

    /// Estimate privacy leakage for different routing strategies
    pub fn compare_routing_strategies(
        strategies: &[Vec<EdgeId>],
        edge_privacy: &HashMap<EdgeId, EdgePrivacy>,
    ) -> Vec<(usize, f64)> {
        strategies
            .iter()
            .enumerate()
            .map(|(i, edges)| {
                let mut route = RoutePrivacy::new(format!("strategy_{}", i), edges.clone());
                route
                    .compose_privacy(edge_privacy, 1000, 300.0)
                    .unwrap_or(());
                (i, route.total_epsilon)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_privacy_creation() {
        let edge_id = EdgeId::new("A", "B", EdgeType::DirectConnection);
        let edge_privacy = EdgePrivacy::new(edge_id.clone());

        assert_eq!(edge_privacy.edge_id, edge_id);
        assert_eq!(
            edge_privacy.base_epsilon,
            EdgeType::DirectConnection.base_epsilon()
        );
    }

    #[test]
    fn test_epsilon_estimation() {
        let edge_id = EdgeId::new("A", "B", EdgeType::MixnetConnection);
        let edge_privacy = EdgePrivacy::new(edge_id);

        let epsilon = edge_privacy.estimate_epsilon(1000, 300.0);
        assert!(epsilon > 0.0);
        assert!(epsilon < 10.0);
    }

    #[test]
    fn test_budget_manager_creation() {
        let manager = PrivacyBudgetManager::new(5.0).unwrap();
        let status = manager.budget_status();

        assert_eq!(status.epsilon_max, 5.0);
        assert_eq!(status.epsilon_used, 0.0);
        assert_eq!(status.utilization, 0.0);
    }

    #[test]
    fn test_budget_allocation() {
        let mut manager = PrivacyBudgetManager::new(5.0).unwrap();

        // Increase the per-route limit to accommodate the test scenario
        manager.policy.epsilon_max_per_route = 1.0;

        // Register edges
        let edge1 = EdgeId::new("A", "B", EdgeType::DirectConnection);
        let edge2 = EdgeId::new("B", "C", EdgeType::MixnetConnection);

        manager.register_edge(edge1.clone(), EdgePrivacy::new(edge1.clone()));
        manager.register_edge(edge2.clone(), EdgePrivacy::new(edge2.clone()));

        // Request budget (values should now work with increased policy limit)
        let result = manager.request_budget("route1".to_string(), vec![edge1, edge2], 10, 10.0);
        assert!(result.is_ok());

        let status = manager.budget_status();
        assert!(status.epsilon_used > 0.0);
        assert_eq!(status.active_routes, 1);
    }

    #[test]
    fn test_budget_exceeded() {
        let mut manager = PrivacyBudgetManager::new(0.1).unwrap(); // Very small budget

        let edge = EdgeId::new("A", "B", EdgeType::DirectConnection);
        manager.register_edge(edge.clone(), EdgePrivacy::new(edge.clone()));

        let result = manager.request_budget("route1".to_string(), vec![edge], 10000, 3600.0);
        assert!(matches!(result, Err(PrivacyError::BudgetExceeded { .. })));
    }

    #[test]
    fn test_route_composition() {
        let edges = vec![
            EdgeId::new("A", "B", EdgeType::DirectConnection),
            EdgeId::new("B", "C", EdgeType::MixnetConnection),
        ];

        let mut route = RoutePrivacy::new("test_route".to_string(), edges.clone());

        let mut edge_privacy = HashMap::new();
        for edge in &edges {
            edge_privacy.insert(edge.clone(), EdgePrivacy::new(edge.clone()));
        }

        let result = route.compose_privacy(&edge_privacy, 1000, 300.0);
        assert!(result.is_ok());
        assert!(route.total_epsilon > 0.0);
    }

    #[test]
    fn test_budget_release() {
        let mut manager = PrivacyBudgetManager::new(5.0).unwrap();

        let edge = EdgeId::new("A", "B", EdgeType::MixnetConnection);
        manager.register_edge(edge.clone(), EdgePrivacy::new(edge.clone()));

        // Allocate budget
        let _allocated = manager
            .request_budget("route1".to_string(), vec![edge], 1000, 300.0)
            .unwrap();
        let status_before = manager.budget_status();

        // Release budget
        let released = manager.release_budget("route1").unwrap();
        let status_after = manager.budget_status();

        assert!(released > 0.0);
        assert_eq!(
            status_after.epsilon_used,
            status_before.epsilon_used - released
        );
        assert_eq!(status_after.active_routes, 0);
    }
}
