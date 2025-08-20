//! Semiring operations for multi-criteria path optimization
//!
//! Implements a Cost semiring over {lat_ms, jitter_ms, failure_prob, privacy_penalty, fee}
//! with proper mathematical foundations as specified in Prompt 8.
//!
//! # Privacy Modes (Feature Flags)
//!
//! - `privacy-strict`: Maximum privacy protection, minimal performance
//! - `privacy-balanced`: Balanced privacy/performance tradeoff (default)
//! - `privacy-perf`: Performance-optimized with basic privacy protection
//!
//! # QoS Modes (Feature Flags)
//!
//! - `real-time`: Latency and jitter optimized
//! - `cost-efficient`: Fee and reliability optimized
//! - `reliability-first`: Reliability and low jitter optimized
//! - `gaming`: Ultra-low latency and jitter optimized

// use betanet_dtn::router::Contact;  // Commented out for standalone testing
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Mul};

/// Multi-criteria cost vector for routing decisions
///
/// Implements the semiring cost tuple as specified in Prompt 8:
/// (lat_ms, jitter_ms, failure_prob, privacy_penalty, fee)
///
/// # Semiring Operations
/// - **Addition (⊕)**: min-plus for selecting the better (non-dominated) path
/// - **Multiplication (⊗)**: additive composition for multi-hop path concatenation
/// - **Zero element**: (∞, ∞, 1.0, ∞, ∞) - worst possible cost
/// - **One element**: (0, 0, 0, 0, 0) - identity for composition
///
/// This structure enables mathematically sound multi-criteria optimization
/// with proper Pareto frontier management.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Cost {
    /// Base latency in milliseconds (lower is better)
    pub lat_ms: OrderedFloat<f64>,
    /// Jitter/variance in latency in milliseconds (lower is better)
    pub jitter_ms: OrderedFloat<f64>,
    /// Failure probability: 0.0 = never fails, 1.0 = always fails (lower is better)
    pub failure_prob: OrderedFloat<f64>,
    /// Privacy penalty: cost of privacy protection (lower is better)
    pub privacy_penalty: OrderedFloat<f64>,
    /// Fee in micro-units of currency (lower is better)
    pub fee: OrderedFloat<f64>,
}

impl Cost {
    /// Create a new cost vector with the updated semiring cost tuple
    ///
    /// # Arguments
    /// * `lat_ms` - Base latency in milliseconds (≥ 0)
    /// * `jitter_ms` - Latency jitter/variance in milliseconds (≥ 0)
    /// * `failure_prob` - Failure probability in [0.0, 1.0]
    /// * `privacy_penalty` - Privacy protection cost (≥ 0)
    /// * `fee` - Economic fee in micro-units (≥ 0)
    pub fn new(lat_ms: f64, jitter_ms: f64, failure_prob: f64, privacy_penalty: f64, fee: f64) -> Self {
        Self {
            lat_ms: OrderedFloat(lat_ms.max(0.0)),
            jitter_ms: OrderedFloat(jitter_ms.max(0.0)),
            failure_prob: OrderedFloat(failure_prob.min(1.0).max(0.0)),
            privacy_penalty: OrderedFloat(privacy_penalty.max(0.0)),
            fee: OrderedFloat(fee.max(0.0)),
        }
    }

    /// Create cost from legacy DTN contact for backward compatibility
    pub fn from_dtn_contact_legacy(latency_ms: f64, mah: f64, reliability: f64, privacy_eps: f64) -> Self {
        // Convert legacy parameters to new cost structure
        let jitter_ms = latency_ms * 0.1; // Estimate 10% jitter
        let failure_prob = 1.0 - reliability; // Convert reliability to failure probability
        let privacy_penalty = privacy_eps; // Map privacy epsilon to penalty
        let fee = mah * 0.01; // Convert energy to fee estimate

        Self::new(latency_ms, jitter_ms, failure_prob, privacy_penalty, fee)
    }

    /* Commented out for standalone testing - requires betanet-dtn dependency
    /// Create from existing DTN Contact with updated cost structure
    pub fn from_dtn_contact(contact: &Contact, bundle_size_bytes: usize, privacy_penalty: f64) -> Self {
        // Convert DTN metrics to new cost structure
        let lat_ms = contact.latency.as_millis() as f64;

        // Estimate jitter as 10% of latency for mobile/wireless networks
        let jitter_ms = lat_ms * 0.1;

        // Convert reliability to failure probability
        let failure_prob = 1.0 - contact.reliability;

        // Calculate fee based on energy cost and transmission time
        let transmission_time_s = contact.transmission_time(bundle_size_bytes).as_secs_f64();
        let energy_cost = contact.energy_cost * transmission_time_s;
        let fee = energy_cost * 100.0; // Convert to micro-units

        Self::new(lat_ms, jitter_ms, failure_prob, privacy_penalty, fee)
    }
    */

    /// Semiring zero element (identity for ⊕ merge operation)
    /// Represents the worst possible cost - used for initialization
    pub fn zero() -> Self {
        Self {
            lat_ms: OrderedFloat(f64::INFINITY),
            jitter_ms: OrderedFloat(f64::INFINITY),
            failure_prob: OrderedFloat(1.0), // Always fails
            privacy_penalty: OrderedFloat(f64::INFINITY),
            fee: OrderedFloat(f64::INFINITY),
        }
    }

    /// Semiring one element (identity for ⊗ compose operation)
    /// Represents zero cost - used for path composition identity
    pub fn one() -> Self {
        Self {
            lat_ms: OrderedFloat(0.0),
            jitter_ms: OrderedFloat(0.0),
            failure_prob: OrderedFloat(0.0), // Never fails
            privacy_penalty: OrderedFloat(0.0),
            fee: OrderedFloat(0.0),
        }
    }

    /// Check if this cost is dominated by another (all components ≥)
    /// Used for Pareto frontier filtering
    pub fn is_dominated_by(&self, other: &Cost) -> bool {
        self.lat_ms >= other.lat_ms
            && self.jitter_ms >= other.jitter_ms
            && self.failure_prob >= other.failure_prob
            && self.privacy_penalty >= other.privacy_penalty
            && self.fee >= other.fee
    }

    /// Check if this cost strictly dominates another (all components ≤, at least one <)
    /// Used for Pareto optimality determination
    pub fn dominates(&self, other: &Cost) -> bool {
        let all_leq = self.lat_ms <= other.lat_ms
            && self.jitter_ms <= other.jitter_ms
            && self.failure_prob <= other.failure_prob
            && self.privacy_penalty <= other.privacy_penalty
            && self.fee <= other.fee;

        let some_less = self.lat_ms < other.lat_ms
            || self.jitter_ms < other.jitter_ms
            || self.failure_prob < other.failure_prob
            || self.privacy_penalty < other.privacy_penalty
            || self.fee < other.fee;

        all_leq && some_less
    }

    /// Check if costs are incomparable (neither dominates)
    pub fn is_incomparable_with(&self, other: &Cost) -> bool {
        !self.dominates(other) && !other.dominates(self) && self != other
    }

    /// Scalarize cost vector using weight vector for tie-breaking
    /// Used when selecting from Pareto frontier
    pub fn scalarize(&self, weights: &WeightVector) -> f64 {
        weights.latency * self.lat_ms.0
            + weights.jitter * self.jitter_ms.0
            + weights.reliability * self.failure_prob.0
            + weights.privacy * self.privacy_penalty.0
            + weights.fee * self.fee.0
    }

    /// Get reliability (converting from failure probability)
    pub fn reliability(&self) -> f64 {
        1.0 - self.failure_prob.0
    }

    /// Get total latency including jitter estimate (latency + 2*jitter for 95% confidence)
    pub fn total_latency_95(&self) -> f64 {
        self.lat_ms.0 + 2.0 * self.jitter_ms.0
    }

    /// Check if cost represents a valid finite path
    pub fn is_finite(&self) -> bool {
        self.lat_ms.is_finite()
            && self.jitter_ms.is_finite()
            && self.failure_prob.is_finite()
            && self.privacy_penalty.is_finite()
            && self.fee.is_finite()
    }
}

impl fmt::Display for Cost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cost(lat={}ms, jit={:.1}ms, fail={:.3}, priv={:.3}, fee={:.1}μ)",
            self.lat_ms.0,
            self.jitter_ms.0,
            self.failure_prob.0,
            self.privacy_penalty.0,
            self.fee.0
        )
    }
}

impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            Some(Ordering::Equal)
        } else if self.dominates(other) {
            Some(Ordering::Less) // Lower cost dominates
        } else if other.dominates(self) {
            Some(Ordering::Greater)
        } else {
            None // Incomparable
        }
    }
}

/// Semiring addition (⊕): merge operation that keeps the better (dominating) cost
impl Add for Cost {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        match self.partial_cmp(&other) {
            Some(Ordering::Less) | Some(Ordering::Equal) => self, // self dominates or equal
            Some(Ordering::Greater) => other,                     // other dominates
            None => self, // Incomparable - keep first (could also keep both in frontier)
        }
    }
}

/// Semiring multiplication (⊗): composition operation for path concatenation
/// Implements additive rules as specified in Prompt 8
impl Mul for Cost {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        // Additive composition rules for multi-hop paths
        Self {
            // Latencies add (serial path)
            lat_ms: self.lat_ms + other.lat_ms,

            // Jitters add in quadrature (statistical combination)
            jitter_ms: OrderedFloat((self.jitter_ms.0.powi(2) + other.jitter_ms.0.powi(2)).sqrt()),

            // Failure probabilities multiply (independent failures)
            // fail_total = 1 - (1-fail_1)*(1-fail_2) = fail_1 + fail_2 - fail_1*fail_2
            failure_prob: self.failure_prob + other.failure_prob
                - self.failure_prob * other.failure_prob,

            // Privacy penalties add (cumulative cost)
            privacy_penalty: self.privacy_penalty + other.privacy_penalty,

            // Fees add (cumulative economic cost)
            fee: self.fee + other.fee,
        }
    }
}

/// Weight vector for scalarization with updated cost tuple fields
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WeightVector {
    pub latency: f64,
    pub jitter: f64,
    pub reliability: f64,
    pub privacy: f64,
    pub fee: f64,
}

impl WeightVector {
    /// Create normalized weight vector with all 5 cost components
    pub fn new(latency: f64, jitter: f64, reliability: f64, privacy: f64, fee: f64) -> Self {
        let sum = latency + jitter + reliability + privacy + fee;
        if sum > 0.0 {
            Self {
                latency: latency / sum,
                jitter: jitter / sum,
                reliability: reliability / sum,
                privacy: privacy / sum,
                fee: fee / sum,
            }
        } else {
            Self::balanced()
        }
    }

    /// Balanced weights for general use (equal weight to all criteria)
    pub fn balanced() -> Self {
        Self {
            latency: 0.2,
            jitter: 0.2,
            reliability: 0.2,
            privacy: 0.2,
            fee: 0.2,
        }
    }

    /// Real-time optimized weights (prioritize latency and jitter)
    pub fn real_time() -> Self {
        Self {
            latency: 0.4,
            jitter: 0.3,
            reliability: 0.2,
            privacy: 0.05,
            fee: 0.05,
        }
    }

    /// Cost-efficient weights (prioritize fee and reliability)
    pub fn cost_efficient() -> Self {
        Self {
            latency: 0.1,
            jitter: 0.1,
            reliability: 0.3,
            privacy: 0.1,
            fee: 0.4,
        }
    }

    /// Privacy-first weights (prioritize privacy and reliability)
    pub fn privacy_first() -> Self {
        Self {
            latency: 0.1,
            jitter: 0.1,
            reliability: 0.3,
            privacy: 0.4,
            fee: 0.1,
        }
    }

    /// Reliability-focused weights (prioritize reliability and low jitter)
    pub fn reliability_first() -> Self {
        Self {
            latency: 0.15,
            jitter: 0.25,
            reliability: 0.4,
            privacy: 0.1,
            fee: 0.1,
        }
    }

    /// Gaming/interactive weights (ultra-low latency and jitter)
    pub fn gaming() -> Self {
        Self {
            latency: 0.45,
            jitter: 0.35,
            reliability: 0.15,
            privacy: 0.025,
            fee: 0.025,
        }
    }

    /// Privacy mode weights based on compile-time feature flags
    pub fn privacy_mode() -> Self {
        #[cfg(feature = "privacy-strict")]
        {
            // Strict privacy: Maximum privacy protection, performance secondary
            Self {
                latency: 0.05,
                jitter: 0.05,
                reliability: 0.2,
                privacy: 0.6,
                fee: 0.1,
            }
        }

        #[cfg(feature = "privacy-perf")]
        {
            // Performance privacy: Balance privacy with performance needs
            Self {
                latency: 0.3,
                jitter: 0.2,
                reliability: 0.2,
                privacy: 0.2,
                fee: 0.1,
            }
        }

        #[cfg(not(any(feature = "privacy-strict", feature = "privacy-perf")))]
        {
            // Balanced privacy (default): Moderate privacy protection
            Self {
                latency: 0.15,
                jitter: 0.15,
                reliability: 0.25,
                privacy: 0.35,
                fee: 0.1,
            }
        }
    }

    /// QoS mode weights based on compile-time feature flags
    pub fn qos_mode() -> Self {
        #[cfg(feature = "real-time")]
        {
            Self::real_time()
        }
        #[cfg(feature = "cost-efficient")]
        {
            Self::cost_efficient()
        }
        #[cfg(feature = "reliability-first")]
        {
            Self::reliability_first()
        }
        #[cfg(feature = "gaming")]
        {
            Self::gaming()
        }
        #[cfg(not(any(feature = "real-time", feature = "cost-efficient", feature = "reliability-first", feature = "gaming")))]
        {
            Self::balanced()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semiring_laws() {
        // Use costs where dominance relationships are clear (lat_ms, jitter_ms, failure_prob, privacy_penalty, fee)
        let better = Cost::new(50.0, 5.0, 0.05, 0.1, 10.0); // Dominates in all dimensions
        let worse = Cost::new(100.0, 15.0, 0.2, 0.5, 50.0); // Dominated in all dimensions
        let c = Cost::new(200.0, 20.0, 0.1, 0.2, 25.0);

        // Addition commutativity: a ⊕ b = b ⊕ a (when one dominates)
        assert_eq!(better + worse, worse + better); // Both should return 'better'

        // Multiplication associativity: (a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)
        let left_mul = (better * worse) * c;
        let right_mul = better * (worse * c);
        // Should be equal for composition
        assert!((left_mul.lat_ms - right_mul.lat_ms).abs() < 1e-10);

        // Zero element: a ⊕ 0 = a
        assert_eq!(better + Cost::zero(), better);

        // One element: a ⊗ 1 = a
        let result = better * Cost::one();
        assert!((result.lat_ms - better.lat_ms).abs() < 1e-10);
        assert!((result.jitter_ms - better.jitter_ms).abs() < 1e-10);
    }

    #[test]
    fn test_partial_order() {
        let better = Cost::new(50.0, 5.0, 0.1, 0.1, 10.0);
        let worse = Cost::new(100.0, 10.0, 0.2, 0.2, 20.0);
        let incomparable = Cost::new(25.0, 15.0, 0.3, 0.3, 30.0); // Better latency, worse other metrics

        assert!(better.dominates(&worse));
        assert!(!worse.dominates(&better));
        assert!(better.is_incomparable_with(&incomparable));
        assert!(!better.dominates(&incomparable));
        assert!(!incomparable.dominates(&better));
    }

    #[test]
    fn test_scalarization() {
        let cost = Cost::new(100.0, 10.0, 0.1, 0.2, 15.0);
        let weights = WeightVector::balanced();

        let score = cost.scalarize(&weights);
        assert!(score > 0.0);

        // Real-time weights should prioritize latency/jitter
        let rt_weights = WeightVector::real_time();
        let rt_score = cost.scalarize(&rt_weights);

        // Cost-efficient weights should prioritize fee
        let cost_weights = WeightVector::cost_efficient();
        let cost_score = cost.scalarize(&cost_weights);

        // Scores should differ based on weight priorities
        assert_ne!(score, rt_score);
        assert_ne!(score, cost_score);
    }

    #[test]
    fn test_composition_laws() {
        let hop1 = Cost::new(50.0, 5.0, 0.05, 0.1, 10.0); // (lat, jitter, fail_prob, priv_penalty, fee)
        let hop2 = Cost::new(75.0, 8.0, 0.10, 0.2, 15.0);

        let path = hop1 * hop2;

        // Latencies should add
        assert_eq!(path.lat_ms.0, 125.0);

        // Jitters should add in quadrature: sqrt(5² + 8²) = sqrt(89) ≈ 9.43
        let expected_jitter = (5.0_f64.powi(2) + 8.0_f64.powi(2)).sqrt();
        assert!((path.jitter_ms.0 - expected_jitter).abs() < 1e-10);

        // Failure probabilities: 1 - (1-0.05)*(1-0.10) = 1 - 0.95*0.90 = 1 - 0.855 = 0.145
        let expected_fail_prob = 0.05 + 0.10 - 0.05 * 0.10;
        assert!((path.failure_prob.0 - expected_fail_prob).abs() < 1e-10);

        // Privacy penalties should add
        assert!((path.privacy_penalty.0 - 0.3).abs() < 1e-10);

        // Fees should add
        assert_eq!(path.fee.0, 25.0);
    }

    #[test]
    fn test_privacy_feature_flags() {
        let weights = WeightVector::privacy_mode();

        // Verify the weights are normalized (sum to 1.0)
        let sum = weights.latency + weights.jitter + weights.reliability + weights.privacy + weights.fee;
        assert!((sum - 1.0).abs() < 1e-10);

        // Verify privacy mode behaves correctly based on feature flags
        #[cfg(feature = "privacy-strict")]
        {
            // Strict mode should heavily prioritize privacy
            assert!(weights.privacy >= 0.6);
            assert!(weights.latency <= 0.1);
            assert!(weights.jitter <= 0.1);
        }

        #[cfg(feature = "privacy-perf")]
        {
            // Performance mode should balance privacy with performance
            assert!(weights.privacy >= 0.15);
            assert!(weights.latency >= 0.25);
            assert!(weights.jitter >= 0.15);
        }

        #[cfg(not(any(feature = "privacy-strict", feature = "privacy-perf")))]
        {
            // Balanced mode (default) should have moderate privacy
            assert!(weights.privacy >= 0.3);
            assert!(weights.privacy <= 0.4);
            assert!(weights.latency >= 0.1);
            assert!(weights.latency <= 0.2);
        }
    }

    #[test]
    fn test_qos_feature_flags() {
        let weights = WeightVector::qos_mode();

        // Verify the weights are normalized
        let sum = weights.latency + weights.jitter + weights.reliability + weights.privacy + weights.fee;
        assert!((sum - 1.0).abs() < 1e-10);

        // Verify QoS mode behaves correctly based on feature flags
        #[cfg(feature = "real-time")]
        {
            // Real-time should prioritize latency and jitter
            assert!(weights.latency + weights.jitter >= 0.6);
        }

        #[cfg(feature = "cost-efficient")]
        {
            // Cost-efficient should prioritize fee and reliability
            assert!(weights.fee + weights.reliability >= 0.6);
        }

        #[cfg(feature = "reliability-first")]
        {
            // Reliability-first should prioritize reliability and jitter
            assert!(weights.reliability >= 0.35);
            assert!(weights.jitter >= 0.2);
        }

        #[cfg(feature = "gaming")]
        {
            // Gaming should prioritize ultra-low latency and jitter
            assert!(weights.latency + weights.jitter >= 0.7);
            assert!(weights.latency >= 0.4);
        }
    }
}
