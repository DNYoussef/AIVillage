//! Semiring-based Navigator for AIVillage Multi-Criteria Routing
//!
//! This crate provides a mathematically sound approach to multi-criteria path selection
//! that builds on and extends the existing AIVillage navigation infrastructure.
//!
//! # Features
//!
//! - **Semiring algebra** for sound composition of {latency, energy, reliability, privacy} costs
//! - **Pareto frontier management** with configurable capacity (K=8 by default)
//! - **Label-setting algorithm** for multi-criteria optimization
//! - **Integration with existing DTN router** and contact graph routing
//! - **CLA-aware path selection** with BitChat/Betanet/QUIC support
//! - **Privacy budget tracking** with epsilon-differential privacy accounting
//! - **QoS-driven scalarization** for tie-breaking in Pareto frontiers
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
//! │   Application   │────▶│    Navigator     │────▶│  CLA Selection  │
//! │                 │     │                  │     │                 │
//! └─────────────────┘     └──────────────────┘     └─────────────────┘
//!                                  │
//!                         ┌────────▼────────┐
//!                         │ Semiring Router │
//!                         │                 │
//!                         └────────┬────────┘
//!                                  │
//!                         ┌────────▼────────┐
//!                         │  DTN Contact    │
//!                         │  Graph Router   │
//!                         └─────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust
//! use navigator::{Navigator, QosRequirements};
//! use betanet_dtn::{ContactGraphRouter, EndpointId, RoutingPolicy};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create DTN router
//! let local_node = EndpointId::node("local");
//! let dtn_router = ContactGraphRouter::new(local_node.clone(), RoutingPolicy::default());
//!
//! // Create Navigator
//! let navigator = Navigator::new(dtn_router, local_node);
//!
//! // Select path with QoS requirements
//! let destination = EndpointId::node("destination");
//! let qos = QosRequirements::real_time(); // Prioritize latency
//! let deadline = Some(1000); // 1000ms deadline
//! let privacy_cap = Some(0.5); // Privacy budget
//!
//! let selection = navigator.select_path(&destination, qos, deadline, privacy_cap).await?;
//! println!("Selected CLA: {} with custody: {}", selection.cla_name, selection.custody);
//! # Ok(())
//! # }
//! ```
//!
//! # Integration with Existing Code
//!
//! This Navigator builds on the existing AIVillage infrastructure:
//!
//! - **Extends DTN ContactGraphRouter**: Uses existing contact plans and routing policies
//! - **Integrates with CLAs**: BitChat (BLE mesh), Betanet HTX (TCP/TLS), QUIC datagrams
//! - **Respects privacy infrastructure**: Integrates with existing mixnode and privacy routing
//! - **Maintains compatibility**: Can be used alongside existing Python Navigator agents
//!
//! # Mathematical Foundation
//!
//! The Navigator uses a semiring structure over Cost vectors:
//!
//! - **Semiring elements**: `Cost = (latency_ms, mAh, inv_reliability, privacy_eps)`
//! - **Addition (⊕)**: Pareto-optimal merge keeping non-dominated costs
//! - **Multiplication (⊗)**: Path composition for multi-hop routes
//! - **Zero element**: `(∞, ∞, 1.0, ∞)` - worst possible cost
//! - **One element**: `(0, 0, 0, 0)` - identity for composition
//!
//! This ensures mathematically sound optimization while respecting the partial order
//! defined by cost dominance relationships.

#![warn(clippy::all)]
#![allow(missing_docs)]
#![allow(dead_code)]

// Core modules
pub mod api;
pub mod route;
pub mod semiring;

// Re-export main types for convenience
pub use api::{ClaSelection, Navigator, NavigatorError, NavigatorStats, PrivacyMode};
pub use route::{LabeledPath, ParetoFrontier, QosRequirements, RoutingError, SemiringRouter};
pub use semiring::{Cost, WeightVector};

// Re-export key DTN types for integration
pub use betanet_dtn::router::Contact;
pub use betanet_dtn::{Bundle, ContactPlan, EndpointId, RoutingPolicy};

/// Current version of the Navigator crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default Pareto frontier capacity
pub const DEFAULT_FRONTIER_CAPACITY: usize = 8;

/// Default privacy budget for epsilon-differential privacy
pub const DEFAULT_PRIVACY_BUDGET: f64 = 1.0;

#[cfg(test)]
mod integration_tests {
    use super::*;
    use betanet_dtn::{ContactGraphRouter, EndpointId, RoutingPolicy};
    use route::QosRequirements;

    #[tokio::test]
    async fn test_end_to_end_path_selection() {
        // Create a simple network topology for testing
        let local_node = EndpointId::node("local");
        let destination = EndpointId::node("destination");

        // Create DTN router with test policy
        let policy = RoutingPolicy::default();
        let dtn_router = ContactGraphRouter::new(local_node.clone(), policy);

        // Create Navigator
        let navigator = Navigator::new(dtn_router, local_node.clone());

        // Test path selection with different QoS requirements
        let qos_real_time = QosRequirements::real_time();
        let qos_energy = QosRequirements::energy_efficient();
        let qos_privacy = QosRequirements::privacy_first();

        // These should not fail even with empty contact plan (will return NoPathFound)
        let result_rt = navigator
            .select_path(&destination, qos_real_time, None, None)
            .await;
        let result_energy = navigator
            .select_path(&destination, qos_energy, None, None)
            .await;
        let result_privacy = navigator
            .select_path(&destination, qos_privacy, None, None)
            .await;

        // All should fail with NoPathFound since we have no contacts
        assert!(matches!(
            result_rt,
            Err(NavigatorError::Routing(RoutingError::NoPathFound { .. }))
        ));
        assert!(matches!(
            result_energy,
            Err(NavigatorError::Routing(RoutingError::NoPathFound { .. }))
        ));
        assert!(matches!(
            result_privacy,
            Err(NavigatorError::Routing(RoutingError::NoPathFound { .. }))
        ));
    }

    #[tokio::test]
    async fn test_qos_requirements() {
        let qos = QosRequirements::balanced();
        let weights = qos.to_weight_vector();

        // All weights should be equal for balanced QoS
        assert!((weights.latency - 0.25).abs() < 1e-6);
        assert!((weights.energy - 0.25).abs() < 1e-6);
        assert!((weights.reliability - 0.25).abs() < 1e-6);
        assert!((weights.privacy - 0.25).abs() < 1e-6);

        // Real-time QoS should prioritize latency
        let qos_rt = QosRequirements::real_time();
        let weights_rt = qos_rt.to_weight_vector();
        assert!(weights_rt.latency > 0.5);
        assert!(weights_rt.energy < 0.2);
    }

    #[test]
    fn test_cost_semiring_properties() {
        use semiring::Cost;

        // Test basic semiring properties with realistic values
        let cost1 = Cost::new(100.0, 5.0, 0.9, 0.1); // Good path
        let cost2 = Cost::new(200.0, 3.0, 0.95, 0.05); // Different tradeoffs
        let _cost3 = Cost::new(50.0, 10.0, 0.8, 0.2); // Fast but expensive

        // Test composition (multiplication)
        let composed = cost1 * cost2;
        assert_eq!(composed.latency_ms.0, 300.0); // Latencies add
        assert_eq!(composed.mah.0, 8.0); // Energy adds

        // Test merge (addition) - should keep better cost
        let better = Cost::new(50.0, 3.0, 0.95, 0.05);
        let worse = Cost::new(200.0, 10.0, 0.8, 0.2);
        let merged = better + worse;
        assert_eq!(merged, better); // Should keep the dominating cost

        // Test zero and one elements
        let zero = Cost::zero();
        let one = Cost::one();

        assert!(zero.latency_ms.is_infinite());
        assert_eq!(one.latency_ms.0, 0.0);

        // Composition with one should be identity
        let identity_test = cost1 * one;
        assert!((identity_test.latency_ms - cost1.latency_ms).abs() < 1e-10);
    }

    #[test]
    fn test_pareto_frontier_capacity() {
        use route::ParetoFrontier;
        use semiring::Cost;

        let mut frontier = ParetoFrontier::new(3); // Capacity 3
        let dest = EndpointId::node("test");

        // Add multiple incomparable paths
        let paths = vec![
            (100.0, 5.0, 0.9, 0.1),   // Balanced
            (50.0, 10.0, 0.8, 0.2),   // Fast but expensive
            (200.0, 2.0, 0.95, 0.05), // Slow but efficient
            (150.0, 7.0, 0.85, 0.15), // Another point
            (75.0, 8.0, 0.9, 0.1),    // Another tradeoff
        ];

        for (lat, energy, rel, priv_eps) in paths {
            let cost = Cost::new(lat, energy, rel, priv_eps);
            let path = route::LabeledPath::new(dest.clone(), cost, 1000);
            frontier.try_insert(path);
        }

        // Should not exceed capacity
        assert!(frontier.len() <= 3);
        assert!(!frontier.is_empty());

        // All remaining paths should be non-dominated
        let paths = frontier.paths();
        for i in 0..paths.len() {
            for j in 0..paths.len() {
                if i != j {
                    assert!(!paths[i].cost.dominates(&paths[j].cost));
                }
            }
        }
    }
}
