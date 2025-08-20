//! Python bindings for the Navigator semiring and routing functionality
//!
//! This module provides Python integration for the semiring-based cost model
//! and path selection algorithms, enabling Python orchestrator calls as
//! specified in Prompt 8.

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

#[cfg(feature = "python-bindings")]
use crate::semiring::{Cost, WeightVector};

/// Python wrapper for the Cost semiring
#[cfg(feature = "python-bindings")]
#[pyclass(name = "Cost")]
#[derive(Clone)]
pub struct PyCost {
    #[pyo3(get, set)]
    pub lat_ms: f64,
    #[pyo3(get, set)]
    pub jitter_ms: f64,
    #[pyo3(get, set)]
    pub failure_prob: f64,
    #[pyo3(get, set)]
    pub privacy_penalty: f64,
    #[pyo3(get, set)]
    pub fee: f64,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyCost {
    #[new]
    fn new(lat_ms: f64, jitter_ms: f64, failure_prob: f64, privacy_penalty: f64, fee: f64) -> Self {
        Self {
            lat_ms,
            jitter_ms,
            failure_prob,
            privacy_penalty,
            fee,
        }
    }

    // Internal conversion method - not exposed to Python

    /// Check if this cost dominates another cost (all components ≤)
    fn dominates(&self, other: &PyCost) -> bool {
        self.to_rust_cost().dominates(&other.to_rust_cost())
    }

    /// Check if this cost is dominated by another cost (all components ≥)
    fn is_dominated_by(&self, other: &PyCost) -> bool {
        self.to_rust_cost().is_dominated_by(&other.to_rust_cost())
    }

    /// Check if costs are incomparable (neither dominates)
    fn is_incomparable_with(&self, other: &PyCost) -> bool {
        self.to_rust_cost().is_incomparable_with(&other.to_rust_cost())
    }

    /// Compose two costs (path concatenation using ⊗ operator)
    fn compose(&self, other: &PyCost) -> PyCost {
        let result = self.to_rust_cost() * other.to_rust_cost();
        PyCost::from_rust_cost(result)
    }

    /// Merge two costs (select better using ⊕ operator)
    fn merge(&self, other: &PyCost) -> PyCost {
        let result = self.to_rust_cost() + other.to_rust_cost();
        PyCost::from_rust_cost(result)
    }

    /// Scalarize using weight vector for tie-breaking
    fn scalarize(&self, weights: &PyWeightVector) -> f64 {
        self.to_rust_cost().scalarize(&weights.to_rust_weights())
    }

    /// Get reliability (1.0 - failure_prob)
    fn reliability(&self) -> f64 {
        self.to_rust_cost().reliability()
    }

    /// Get total latency with 95% confidence interval
    fn total_latency_95(&self) -> f64 {
        self.to_rust_cost().total_latency_95()
    }

    /// Check if cost is finite (not infinite)
    fn is_finite(&self) -> bool {
        self.to_rust_cost().is_finite()
    }

    /// String representation
    fn __str__(&self) -> String {
        format!(
            "Cost(lat={}ms, jit={:.1}ms, fail={:.3}, priv={:.3}, fee={:.1}μ)",
            self.lat_ms, self.jitter_ms, self.failure_prob, self.privacy_penalty, self.fee
        )
    }

    /// Representation
    fn __repr__(&self) -> String {
        self.__str__()
    }

    /// Zero element (worst possible cost)
    #[staticmethod]
    fn zero() -> PyCost {
        let cost = Cost::zero();
        PyCost {
            lat_ms: cost.lat_ms.into_inner(),
            jitter_ms: cost.jitter_ms.into_inner(),
            failure_prob: cost.failure_prob.into_inner(),
            privacy_penalty: cost.privacy_penalty.into_inner(),
            fee: cost.fee.into_inner(),
        }
    }

    /// One element (identity for composition)
    #[staticmethod]
    fn one() -> PyCost {
        let cost = Cost::one();
        PyCost {
            lat_ms: cost.lat_ms.into_inner(),
            jitter_ms: cost.jitter_ms.into_inner(),
            failure_prob: cost.failure_prob.into_inner(),
            privacy_penalty: cost.privacy_penalty.into_inner(),
            fee: cost.fee.into_inner(),
        }
    }
}

#[cfg(feature = "python-bindings")]
impl PyCost {
    /// Internal helper to create from Rust Cost
    fn from_rust_cost(cost: Cost) -> Self {
        Self {
            lat_ms: cost.lat_ms.into_inner(),
            jitter_ms: cost.jitter_ms.into_inner(),
            failure_prob: cost.failure_prob.into_inner(),
            privacy_penalty: cost.privacy_penalty.into_inner(),
            fee: cost.fee.into_inner(),
        }
    }

    /// Internal helper to convert to Rust Cost
    fn to_rust_cost(&self) -> Cost {
        Cost::new(
            self.lat_ms,
            self.jitter_ms,
            self.failure_prob,
            self.privacy_penalty,
            self.fee,
        )
    }
}

/// Python wrapper for WeightVector
#[cfg(feature = "python-bindings")]
#[pyclass(name = "WeightVector")]
#[derive(Clone)]
pub struct PyWeightVector {
    #[pyo3(get, set)]
    pub latency: f64,
    #[pyo3(get, set)]
    pub jitter: f64,
    #[pyo3(get, set)]
    pub reliability: f64,
    #[pyo3(get, set)]
    pub privacy: f64,
    #[pyo3(get, set)]
    pub fee: f64,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyWeightVector {
    #[new]
    fn new(latency: f64, jitter: f64, reliability: f64, privacy: f64, fee: f64) -> Self {
        let rust_weights = WeightVector::new(latency, jitter, reliability, privacy, fee);
        Self {
            latency: rust_weights.latency,
            jitter: rust_weights.jitter,
            reliability: rust_weights.reliability,
            privacy: rust_weights.privacy,
            fee: rust_weights.fee,
        }
    }

    // Internal conversion method - not exposed to Python

    /// Balanced weights (equal weight to all criteria)
    #[staticmethod]
    fn balanced() -> PyWeightVector {
        let weights = WeightVector::balanced();
        PyWeightVector {
            latency: weights.latency,
            jitter: weights.jitter,
            reliability: weights.reliability,
            privacy: weights.privacy,
            fee: weights.fee,
        }
    }

    /// Real-time optimized weights
    #[staticmethod]
    fn real_time() -> PyWeightVector {
        let weights = WeightVector::real_time();
        PyWeightVector {
            latency: weights.latency,
            jitter: weights.jitter,
            reliability: weights.reliability,
            privacy: weights.privacy,
            fee: weights.fee,
        }
    }

    /// Cost-efficient weights
    #[staticmethod]
    fn cost_efficient() -> PyWeightVector {
        let weights = WeightVector::cost_efficient();
        PyWeightVector {
            latency: weights.latency,
            jitter: weights.jitter,
            reliability: weights.reliability,
            privacy: weights.privacy,
            fee: weights.fee,
        }
    }

    /// Privacy-first weights
    #[staticmethod]
    fn privacy_first() -> PyWeightVector {
        let weights = WeightVector::privacy_first();
        PyWeightVector {
            latency: weights.latency,
            jitter: weights.jitter,
            reliability: weights.reliability,
            privacy: weights.privacy,
            fee: weights.fee,
        }
    }

    /// Reliability-focused weights
    #[staticmethod]
    fn reliability_first() -> PyWeightVector {
        let weights = WeightVector::reliability_first();
        PyWeightVector {
            latency: weights.latency,
            jitter: weights.jitter,
            reliability: weights.reliability,
            privacy: weights.privacy,
            fee: weights.fee,
        }
    }

    /// Gaming/interactive weights
    #[staticmethod]
    fn gaming() -> PyWeightVector {
        let weights = WeightVector::gaming();
        PyWeightVector {
            latency: weights.latency,
            jitter: weights.jitter,
            reliability: weights.reliability,
            privacy: weights.privacy,
            fee: weights.fee,
        }
    }

    /// Privacy mode weights (based on compile-time features)
    #[staticmethod]
    fn privacy_mode() -> PyWeightVector {
        let weights = WeightVector::privacy_mode();
        PyWeightVector {
            latency: weights.latency,
            jitter: weights.jitter,
            reliability: weights.reliability,
            privacy: weights.privacy,
            fee: weights.fee,
        }
    }

    /// QoS mode weights (based on compile-time features)
    #[staticmethod]
    fn qos_mode() -> PyWeightVector {
        let weights = WeightVector::qos_mode();
        PyWeightVector {
            latency: weights.latency,
            jitter: weights.jitter,
            reliability: weights.reliability,
            privacy: weights.privacy,
            fee: weights.fee,
        }
    }

    fn __str__(&self) -> String {
        format!(
            "WeightVector(lat={:.3}, jit={:.3}, rel={:.3}, priv={:.3}, fee={:.3})",
            self.latency, self.jitter, self.reliability, self.privacy, self.fee
        )
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}

#[cfg(feature = "python-bindings")]
impl PyWeightVector {
    /// Internal helper to convert to Rust WeightVector
    fn to_rust_weights(&self) -> WeightVector {
        WeightVector {
            latency: self.latency,
            jitter: self.jitter,
            reliability: self.reliability,
            privacy: self.privacy,
            fee: self.fee,
        }
    }
}

/// Route computation result for Python integration
#[cfg(feature = "python-bindings")]
#[pyclass(name = "RouteResult")]
pub struct PyRouteResult {
    #[pyo3(get)]
    pub best_cost: PyCost,
    #[pyo3(get)]
    pub num_hops: usize,
    #[pyo3(get)]
    pub selected_path: String,
    #[pyo3(get)]
    pub decision_log: Vec<String>,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyRouteResult {
    #[new]
    fn new(best_cost: PyCost, num_hops: usize, selected_path: String, decision_log: Vec<String>) -> Self {
        Self {
            best_cost,
            num_hops,
            selected_path,
            decision_log,
        }
    }

    fn __str__(&self) -> String {
        format!(
            "RouteResult(cost={}, hops={}, path={})",
            self.best_cost.__str__(),
            self.num_hops,
            self.selected_path
        )
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}

/// Core routing computation function for Python orchestrator
#[cfg(feature = "python-bindings")]
#[pyfunction]
fn compute_best_route(
    candidate_costs: Vec<PyCost>,
    weights: &PyWeightVector,
    max_candidates: Option<usize>,
) -> PyResult<PyRouteResult> {
    use std::cmp::Ordering;

    if candidate_costs.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "No candidate costs provided"
        ));
    }

    let max_candidates = max_candidates.unwrap_or(8); // Default Pareto frontier capacity
    let rust_weights = weights.to_rust_weights();
    let mut decision_log = Vec::new();

    decision_log.push(format!(
        "Starting route computation with {} candidates, max_candidates={}",
        candidate_costs.len(), max_candidates
    ));
    decision_log.push(format!("Using weights: {}", weights.__str__()));

    // Convert to Rust costs
    let rust_costs: Vec<_> = candidate_costs
        .iter()
        .enumerate()
        .map(|(i, py_cost)| (i, py_cost.to_rust_cost()))
        .collect();

    decision_log.push(format!("Converted {} costs to internal representation", rust_costs.len()));

    // Filter dominated costs (Pareto frontier)
    let mut pareto_frontier = Vec::new();

    for (idx, cost) in rust_costs.iter() {
        let mut is_dominated = false;

        // Check if this cost is dominated by any existing frontier cost
        for (_, frontier_cost) in &pareto_frontier {
            if cost.is_dominated_by(frontier_cost) {
                is_dominated = true;
                break;
            }
        }

        if !is_dominated {
            // Remove any frontier costs that are dominated by this new cost
            pareto_frontier.retain(|(_, frontier_cost)| !frontier_cost.is_dominated_by(cost));

            // Add this cost to the frontier
            pareto_frontier.push((*idx, *cost));

            decision_log.push(format!(
                "Added cost {} to Pareto frontier: {}",
                idx,
                PyCost::from_rust_cost(*cost).__str__()
            ));
        } else {
            decision_log.push(format!("Cost {} dominated, excluded from frontier", idx));
        }
    }

    // Limit frontier size if needed
    if pareto_frontier.len() > max_candidates {
        // Sort by scalarized cost and keep the best ones
        pareto_frontier.sort_by(|(_, a), (_, b)| {
            let score_a = a.scalarize(&rust_weights);
            let score_b = b.scalarize(&rust_weights);
            score_a.partial_cmp(&score_b).unwrap_or(Ordering::Equal)
        });

        pareto_frontier.truncate(max_candidates);
        decision_log.push(format!("Truncated frontier to {} candidates", max_candidates));
    }

    if pareto_frontier.is_empty() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "No valid paths found after Pareto filtering"
        ));
    }

    // Select best cost using scalarization
    let (best_idx, best_cost) = pareto_frontier
        .iter()
        .min_by(|(_, a), (_, b)| {
            let score_a = a.scalarize(&rust_weights);
            let score_b = b.scalarize(&rust_weights);
            score_a.partial_cmp(&score_b).unwrap_or(Ordering::Equal)
        })
        .copied()
        .unwrap();

    decision_log.push(format!(
        "Selected best cost {} with scalarized score {:.6}",
        best_idx,
        best_cost.scalarize(&rust_weights)
    ));

    let result = PyRouteResult::new(
        PyCost::from_rust_cost(best_cost),
        1, // Single hop for now (would be computed from path length)
        format!("path_{}", best_idx),
        decision_log,
    );

    Ok(result)
}

/// Pareto frontier computation for multiple costs
#[cfg(feature = "python-bindings")]
#[pyfunction]
fn compute_pareto_frontier(
    costs: Vec<PyCost>,
    max_size: Option<usize>,
) -> PyResult<Vec<PyCost>> {
    let max_size = max_size.unwrap_or(8);
    let mut frontier = Vec::new();

    for py_cost in costs {
        let rust_cost = py_cost.to_rust_cost();

        // Check if this cost is dominated by any existing frontier cost
        let is_dominated = frontier.iter().any(|frontier_cost: &Cost| {
            rust_cost.is_dominated_by(frontier_cost)
        });

        if !is_dominated {
            // Remove any frontier costs that are dominated by this new cost
            frontier.retain(|frontier_cost| !frontier_cost.is_dominated_by(&rust_cost));

            // Add this cost to the frontier
            frontier.push(rust_cost);
        }
    }

    // Limit frontier size if needed
    if frontier.len() > max_size {
        // For simplicity, keep first max_size costs
        // In practice, would use scalarization for selection
        frontier.truncate(max_size);
    }

    // Convert back to Python costs
    let py_frontier: Vec<PyCost> = frontier
        .iter()
        .map(|cost| PyCost::from_rust_cost(*cost))
        .collect();

    Ok(py_frontier)
}

/// Python module definition
#[cfg(feature = "python-bindings")]
#[pymodule]
fn navigator(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyCost>()?;
    m.add_class::<PyWeightVector>()?;
    m.add_class::<PyRouteResult>()?;
    m.add_function(wrap_pyfunction!(compute_best_route, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pareto_frontier, m)?)?;

    // Add version and constants
    m.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    m.add("DEFAULT_FRONTIER_CAPACITY", crate::DEFAULT_FRONTIER_CAPACITY)?;

    Ok(())
}

#[cfg(test)]
#[cfg(feature = "python-bindings")]
mod tests {
    use super::*;

    #[test]
    fn test_python_cost_creation() {
        let cost = PyCost::new(100.0, 10.0, 0.1, 0.2, 15.0);
        assert_eq!(cost.lat_ms, 100.0);
        assert_eq!(cost.jitter_ms, 10.0);
        assert_eq!(cost.failure_prob, 0.1);
        assert_eq!(cost.privacy_penalty, 0.2);
        assert_eq!(cost.fee, 15.0);
    }

    #[test]
    fn test_python_cost_operations() {
        let cost1 = PyCost::new(50.0, 5.0, 0.05, 0.1, 10.0);
        let cost2 = PyCost::new(100.0, 15.0, 0.2, 0.3, 25.0);

        // Test dominance
        assert!(cost1.dominates(&cost2));
        assert!(!cost2.dominates(&cost1));
        assert!(cost2.is_dominated_by(&cost1));

        // Test composition (path concatenation)
        let composed = cost1.compose(&cost2);
        assert_eq!(composed.lat_ms, 150.0); // Latencies add
        assert_eq!(composed.fee, 35.0); // Fees add

        // Test merge (Pareto selection)
        let merged = cost1.merge(&cost2);
        assert_eq!(merged.lat_ms, cost1.lat_ms); // Should keep the dominating cost
    }

    #[test]
    fn test_python_weight_vector() {
        let weights = PyWeightVector::new(0.3, 0.2, 0.2, 0.2, 0.1);

        // Weights should be normalized
        let sum = weights.latency + weights.jitter + weights.reliability + weights.privacy + weights.fee;
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_python_predefined_weights() {
        let balanced = PyWeightVector::balanced();
        assert!((balanced.latency - 0.2).abs() < 1e-6);
        assert!((balanced.jitter - 0.2).abs() < 1e-6);

        let real_time = PyWeightVector::real_time();
        assert!(real_time.latency > 0.3); // Should prioritize latency
        assert!(real_time.jitter > 0.2); // Should prioritize jitter

        let cost_efficient = PyWeightVector::cost_efficient();
        assert!(cost_efficient.fee > 0.3); // Should prioritize cost
    }

    #[test]
    fn test_python_route_computation() {
        let costs = vec![
            PyCost::new(100.0, 10.0, 0.1, 0.2, 15.0),  // Balanced option
            PyCost::new(50.0, 20.0, 0.2, 0.3, 25.0),   // Fast but expensive
            PyCost::new(200.0, 5.0, 0.05, 0.1, 8.0),   // Slow but cheap
        ];

        let weights = PyWeightVector::balanced();

        let result = compute_best_route(costs, &weights, Some(3)).unwrap();

        // Should have selected a valid cost
        assert!(result.best_cost.is_finite());
        assert_eq!(result.num_hops, 1);
        assert!(result.selected_path.starts_with("path_"));
        assert!(!result.decision_log.is_empty());
    }

    #[test]
    fn test_python_pareto_frontier() {
        let costs = vec![
            PyCost::new(100.0, 10.0, 0.1, 0.2, 15.0),  // Balanced
            PyCost::new(50.0, 5.0, 0.05, 0.1, 10.0),   // Dominates all
            PyCost::new(200.0, 20.0, 0.3, 0.4, 30.0),  // Dominated
            PyCost::new(75.0, 15.0, 0.15, 0.25, 20.0), // Incomparable
        ];

        let frontier = compute_pareto_frontier(costs, Some(3)).unwrap();

        // Should exclude dominated costs
        assert!(frontier.len() <= 3);
        // The dominating cost should be in the frontier
        assert!(frontier.len() >= 1);

        // All costs in frontier should be non-dominated
        for i in 0..frontier.len() {
            for j in 0..frontier.len() {
                if i != j {
                    assert!(!frontier[i].dominates(&frontier[j]));
                }
            }
        }
    }

    #[test]
    fn test_python_cost_scalarization() {
        let cost = PyCost::new(100.0, 10.0, 0.1, 0.2, 15.0);
        let weights = PyWeightVector::balanced();

        let score = cost.scalarize(&weights);
        assert!(score > 0.0);

        // Different weights should produce different scores
        let rt_weights = PyWeightVector::real_time();
        let rt_score = cost.scalarize(&rt_weights);
        assert_ne!(score, rt_score);
    }

    #[test]
    fn test_python_error_handling() {
        let empty_costs = vec![];
        let weights = PyWeightVector::balanced();

        let result = compute_best_route(empty_costs, &weights, Some(3));
        assert!(result.is_err()); // Should error on empty input
    }
}

// Empty export for conditional compilation
