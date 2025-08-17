//! Semiring operations for multi-criteria path optimization
//!
//! Implements a Cost semiring over {latency_ms, mAh, inv_reliability, privacy_eps}
//! with proper mathematical foundations extending the existing DTN Contact scoring.

use betanet_dtn::router::Contact;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Mul};

/// Multi-criteria cost vector for routing decisions
///
/// Builds on existing DTN Contact metrics but formalizes them into a proper semiring
/// structure for Pareto frontier optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Cost {
    /// Latency in milliseconds (lower is better)
    pub latency_ms: OrderedFloat<f64>,
    /// Energy consumption in milliampere-hours (lower is better)
    pub mah: OrderedFloat<f64>,
    /// Inverse reliability: 1.0 - reliability (lower is better)
    pub inv_reliability: OrderedFloat<f64>,
    /// Privacy epsilon: privacy budget consumed (lower is better)
    pub privacy_eps: OrderedFloat<f64>,
}

impl Cost {
    /// Create a new cost vector
    pub fn new(latency_ms: f64, mah: f64, reliability: f64, privacy_eps: f64) -> Self {
        Self {
            latency_ms: OrderedFloat(latency_ms.max(0.0)),
            mah: OrderedFloat(mah.max(0.0)),
            inv_reliability: OrderedFloat(1.0 - reliability.min(1.0).max(0.0)),
            privacy_eps: OrderedFloat(privacy_eps.max(0.0)),
        }
    }

    /// Create from existing DTN Contact
    pub fn from_dtn_contact(contact: &Contact, bundle_size_bytes: usize, privacy_eps: f64) -> Self {
        // Convert DTN metrics to our cost structure
        let latency_ms = contact.latency.as_millis() as f64;

        // Energy cost estimation: base cost + transmission cost
        let transmission_time_s = contact.transmission_time(bundle_size_bytes).as_secs_f64();
        let base_energy_mah = 1.0; // Base radio energy
        let transmission_energy_mah = contact.energy_cost * transmission_time_s * 10.0; // Scaled
        let total_mah = base_energy_mah + transmission_energy_mah;

        Self::new(latency_ms, total_mah, contact.reliability, privacy_eps)
    }

    /// Semiring zero element (identity for ⊕ merge operation)
    pub fn zero() -> Self {
        Self {
            latency_ms: OrderedFloat(f64::INFINITY),
            mah: OrderedFloat(f64::INFINITY),
            inv_reliability: OrderedFloat(1.0), // Worst reliability
            privacy_eps: OrderedFloat(f64::INFINITY),
        }
    }

    /// Semiring one element (identity for ⊗ compose operation)
    pub fn one() -> Self {
        Self {
            latency_ms: OrderedFloat(0.0),
            mah: OrderedFloat(0.0),
            inv_reliability: OrderedFloat(0.0), // Perfect reliability
            privacy_eps: OrderedFloat(0.0),
        }
    }

    /// Check if this cost is dominated by another (all components ≥)
    pub fn is_dominated_by(&self, other: &Cost) -> bool {
        self.latency_ms >= other.latency_ms
            && self.mah >= other.mah
            && self.inv_reliability >= other.inv_reliability
            && self.privacy_eps >= other.privacy_eps
    }

    /// Check if this cost strictly dominates another (all components ≤, at least one <)
    pub fn dominates(&self, other: &Cost) -> bool {
        let all_leq = self.latency_ms <= other.latency_ms
            && self.mah <= other.mah
            && self.inv_reliability <= other.inv_reliability
            && self.privacy_eps <= other.privacy_eps;

        let some_less = self.latency_ms < other.latency_ms
            || self.mah < other.mah
            || self.inv_reliability < other.inv_reliability
            || self.privacy_eps < other.privacy_eps;

        all_leq && some_less
    }

    /// Check if costs are incomparable (neither dominates)
    pub fn is_incomparable_with(&self, other: &Cost) -> bool {
        !self.dominates(other) && !other.dominates(self) && self != other
    }

    /// Scalarize cost vector using weight vector for tie-breaking
    /// Used when selecting from Pareto frontier
    pub fn scalarize(&self, weights: &WeightVector) -> f64 {
        weights.latency * self.latency_ms.0
            + weights.energy * self.mah.0
            + weights.reliability * self.inv_reliability.0
            + weights.privacy * self.privacy_eps.0
    }

    /// Get reliability (converting from inverse)
    pub fn reliability(&self) -> f64 {
        1.0 - self.inv_reliability.0
    }

    /// Check if cost represents a valid finite path
    pub fn is_finite(&self) -> bool {
        self.latency_ms.is_finite()
            && self.mah.is_finite()
            && self.inv_reliability.is_finite()
            && self.privacy_eps.is_finite()
    }
}

impl fmt::Display for Cost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cost({}ms, {:.1}mAh, rel={:.3}, eps={:.3})",
            self.latency_ms.0,
            self.mah.0,
            self.reliability(),
            self.privacy_eps.0
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
impl Mul for Cost {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        // For path concatenation, we add costs and multiply reliabilities
        Self {
            latency_ms: self.latency_ms + other.latency_ms,
            mah: self.mah + other.mah,
            // Reliability multiplication: rel_total = rel_1 * rel_2
            // So inv_rel_total = 1 - (1-inv_rel_1)*(1-inv_rel_2) = inv_rel_1 + inv_rel_2 - inv_rel_1*inv_rel_2
            inv_reliability: self.inv_reliability + other.inv_reliability
                - self.inv_reliability * other.inv_reliability,
            privacy_eps: self.privacy_eps + other.privacy_eps,
        }
    }
}

/// Weight vector for scalarization
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WeightVector {
    pub latency: f64,
    pub energy: f64,
    pub reliability: f64,
    pub privacy: f64,
}

impl WeightVector {
    /// Create normalized weight vector
    pub fn new(latency: f64, energy: f64, reliability: f64, privacy: f64) -> Self {
        let sum = latency + energy + reliability + privacy;
        if sum > 0.0 {
            Self {
                latency: latency / sum,
                energy: energy / sum,
                reliability: reliability / sum,
                privacy: privacy / sum,
            }
        } else {
            Self::balanced()
        }
    }

    /// Balanced weights for general use
    pub fn balanced() -> Self {
        Self {
            latency: 0.25,
            energy: 0.25,
            reliability: 0.25,
            privacy: 0.25,
        }
    }

    /// Real-time optimized weights (prioritize latency)
    pub fn real_time() -> Self {
        Self {
            latency: 0.6,
            energy: 0.1,
            reliability: 0.2,
            privacy: 0.1,
        }
    }

    /// Energy-efficient weights (prioritize battery)
    pub fn energy_efficient() -> Self {
        Self {
            latency: 0.1,
            energy: 0.6,
            reliability: 0.2,
            privacy: 0.1,
        }
    }

    /// Privacy-first weights
    pub fn privacy_first() -> Self {
        Self {
            latency: 0.1,
            energy: 0.1,
            reliability: 0.2,
            privacy: 0.6,
        }
    }

    /// Reliability-focused weights
    pub fn reliability_first() -> Self {
        Self {
            latency: 0.15,
            energy: 0.15,
            reliability: 0.6,
            privacy: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semiring_laws() {
        // Use costs where dominance relationships are clear
        let better = Cost::new(50.0, 3.0, 0.95, 0.05); // Dominates in all dimensions
        let worse = Cost::new(100.0, 10.0, 0.8, 0.2); // Dominated in all dimensions
        let c = Cost::new(200.0, 2.0, 0.95, 0.05);

        // Addition commutativity: a ⊕ b = b ⊕ a (when one dominates)
        assert_eq!(better + worse, worse + better); // Both should return 'better'

        // Multiplication associativity: (a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)
        let left_mul = (better * worse) * c;
        let right_mul = better * (worse * c);
        // Should be equal for composition
        assert!((left_mul.latency_ms - right_mul.latency_ms).abs() < 1e-10);

        // Zero element: a ⊕ 0 = a
        assert_eq!(better + Cost::zero(), better);

        // One element: a ⊗ 1 = a
        let result = better * Cost::one();
        assert!((result.latency_ms - better.latency_ms).abs() < 1e-10);
        assert!((result.mah - better.mah).abs() < 1e-10);
    }

    #[test]
    fn test_partial_order() {
        let better = Cost::new(50.0, 5.0, 0.9, 0.1);
        let worse = Cost::new(100.0, 10.0, 0.8, 0.2);
        let incomparable = Cost::new(25.0, 15.0, 0.7, 0.3); // Better latency, worse energy/reliability

        assert!(better.dominates(&worse));
        assert!(!worse.dominates(&better));
        assert!(better.is_incomparable_with(&incomparable));
        assert!(!better.dominates(&incomparable));
        assert!(!incomparable.dominates(&better));
    }

    #[test]
    fn test_scalarization() {
        let cost = Cost::new(100.0, 5.0, 0.9, 0.1);
        let weights = WeightVector::balanced();

        let score = cost.scalarize(&weights);
        assert!(score > 0.0);

        // Real-time weights should prioritize latency
        let rt_weights = WeightVector::real_time();
        let rt_score = cost.scalarize(&rt_weights);

        // Energy weights should prioritize energy
        let energy_weights = WeightVector::energy_efficient();
        let energy_score = cost.scalarize(&energy_weights);

        // Scores should differ based on weight priorities
        assert_ne!(score, rt_score);
        assert_ne!(score, energy_score);
    }

    #[test]
    fn test_composition_laws() {
        let hop1 = Cost::new(50.0, 2.0, 0.95, 0.05);
        let hop2 = Cost::new(75.0, 3.0, 0.90, 0.10);

        let path = hop1 * hop2;

        // Latency and energy should add
        assert_eq!(path.latency_ms.0, 125.0);
        assert_eq!(path.mah.0, 5.0);

        // Reliability should multiply: 0.95 * 0.90 = 0.855
        let expected_reliability = 0.95 * 0.90;
        assert!((path.reliability() - expected_reliability).abs() < 1e-10);

        // Privacy epsilon should add (with small tolerance for floating point)
        assert!((path.privacy_eps.0 - 0.15).abs() < 1e-10);
    }
}
