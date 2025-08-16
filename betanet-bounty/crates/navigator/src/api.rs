//! Navigator API for intelligent path selection
//!
//! Integrates semiring router with existing DTN and CLA components to provide
//! intelligent path selection based on QoS requirements and constraints.

use crate::route::{QosRequirements, RoutingError, SemiringRouter};
use crate::semiring::Cost;
use betanet_dtn::{ContactGraphRouter, ConvergenceLayer, EndpointId};
use betanet_dtn::router::Contact;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Navigator errors
#[derive(Error, Debug)]
pub enum NavigatorError {
    #[error("Routing error: {0}")]
    Routing(#[from] RoutingError),
    #[error("No suitable CLA found for path requirements")]
    NoSuitableCla,
    #[error("CLA integration error: {message}")]
    ClaIntegration { message: String },
    #[error("Privacy budget exceeded: consumed {consumed}, limit {limit}")]
    PrivacyBudgetExceeded { consumed: f64, limit: f64 },
}

/// CLA selection result
#[derive(Debug, Clone)]
pub struct ClaSelection {
    /// Selected CLA name
    pub cla_name: String,
    /// Whether to request custody transfer
    pub custody: bool,
    /// CLA-specific parameters
    pub params: HashMap<String, String>,
    /// Estimated path cost
    pub estimated_cost: Cost,
    /// Selection rationale
    pub rationale: String,
}

/// Path selection statistics
#[derive(Debug, Clone, Default)]
pub struct NavigatorStats {
    /// Total path selections performed
    pub total_selections: u64,
    /// Selections by CLA type
    pub selections_by_cla: HashMap<String, u64>,
    /// Average selection time in milliseconds
    pub avg_selection_time_ms: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Privacy budget utilization
    pub avg_privacy_utilization: f64,
}

/// Main Navigator for intelligent path selection
pub struct Navigator {
    /// Semiring router for multi-criteria optimization
    router: Arc<RwLock<SemiringRouter>>,
    /// Available convergence layer adapters
    clas: Arc<RwLock<HashMap<String, Box<dyn ConvergenceLayer>>>>,
    /// CLA capabilities and characteristics
    cla_characteristics: HashMap<String, ClaCharacteristics>,
    /// Navigator statistics
    stats: Arc<RwLock<NavigatorStats>>,
    /// Privacy composer for epsilon accounting
    privacy_composer: PrivacyComposer,
}

/// Characteristics of a Convergence Layer Adapter
#[derive(Debug, Clone)]
pub struct ClaCharacteristics {
    /// Transport type (e.g., "bitchat", "betanet-htx", "quic")
    pub transport_type: String,
    /// Typical latency in milliseconds
    pub typical_latency_ms: f64,
    /// Energy cost factor (relative)
    pub energy_cost_factor: f64,
    /// Reliability score (0.0 - 1.0)
    pub reliability: f64,
    /// Privacy features available
    pub privacy_features: Vec<String>,
    /// Maximum transmission unit
    pub mtu: usize,
    /// Whether CLA supports custody transfer
    pub supports_custody: bool,
    /// Whether CLA is suitable for real-time traffic
    pub real_time_capable: bool,
}

/// Privacy budget tracking and epsilon accounting
#[derive(Debug)]
pub struct PrivacyComposer {
    /// Current privacy budget consumed
    budget_consumed: f64,
    /// Privacy budget limit
    budget_limit: f64,
    /// Privacy mode configuration
    mode: PrivacyMode,
}

/// Privacy protection modes
#[derive(Debug, Clone, Copy)]
pub enum PrivacyMode {
    /// No privacy protection
    None,
    /// Basic privacy (minimal epsilon consumption)
    Basic,
    /// Strong privacy (moderate epsilon consumption)
    Strong,
    /// Maximum privacy (strict epsilon limits)
    Maximum,
}

impl Navigator {
    /// Create new Navigator with DTN router and CLA registry
    pub fn new(dtn_router: ContactGraphRouter, local_node: EndpointId) -> Self {
        let semiring_router = SemiringRouter::new(dtn_router, local_node, Some(8));

        let mut cla_characteristics = HashMap::new();

        // Register BitChat CLA characteristics
        cla_characteristics.insert(
            "bitchat".to_string(),
            ClaCharacteristics {
                transport_type: "bluetooth-mesh".to_string(),
                typical_latency_ms: 200.0,
                energy_cost_factor: 0.3, // Energy efficient
                reliability: 0.85,
                privacy_features: vec!["mesh-routing".to_string(), "store-forward".to_string()],
                mtu: 100, // BLE MTU
                supports_custody: true,
                real_time_capable: false,
            },
        );

        // Register Betanet HTX CLA characteristics
        cla_characteristics.insert(
            "betanet-htx".to_string(),
            ClaCharacteristics {
                transport_type: "tcp-tls".to_string(),
                typical_latency_ms: 100.0,
                energy_cost_factor: 1.0, // Moderate energy usage
                reliability: 0.95,
                privacy_features: vec!["mixnode-routing".to_string(), "onion-routing".to_string()],
                mtu: 1024, // TCP MTU
                supports_custody: true,
                real_time_capable: true,
            },
        );

        // Register QUIC CLA characteristics
        cla_characteristics.insert(
            "betanet-quic".to_string(),
            ClaCharacteristics {
                transport_type: "quic-datagram".to_string(),
                typical_latency_ms: 50.0,
                energy_cost_factor: 0.8, // Efficient protocol
                reliability: 0.90,
                privacy_features: vec!["path-diversity".to_string()],
                mtu: 1200, // QUIC datagram MTU
                supports_custody: false, // Datagram mode
                real_time_capable: true,
            },
        );

        Self {
            router: Arc::new(RwLock::new(semiring_router)),
            clas: Arc::new(RwLock::new(HashMap::new())),
            cla_characteristics,
            stats: Arc::new(RwLock::new(NavigatorStats::default())),
            privacy_composer: PrivacyComposer::new(1.0, PrivacyMode::Basic),
        }
    }

    /// Register a convergence layer adapter
    pub async fn register_cla(&self, name: String, cla: Box<dyn ConvergenceLayer>) {
        let mut clas = self.clas.write().await;
        clas.insert(name, cla);
    }

    /// Main path selection API integrating all components
    pub async fn select_path(
        &self,
        destination: &EndpointId,
        qos: QosRequirements,
        deadline: Option<u64>,
        privacy_cap: Option<f64>,
    ) -> Result<ClaSelection, NavigatorError> {
        let start_time = std::time::Instant::now();

        debug!(
            "Selecting path to {} with QoS {:?}, deadline {:?}, privacy_cap {:?}",
            destination, qos, deadline, privacy_cap
        );

        // Check privacy budget if cap is specified
        if let Some(cap) = privacy_cap {
            self.privacy_composer.check_budget(cap)?;
        }

        // Find optimal path using semiring router
        let mut router = self.router.write().await;
        let selected_path = router.select_path(destination, qos.clone(), deadline, privacy_cap)?;

        let path = match selected_path {
            Some(p) => p,
            None => {
                warn!("No path found to destination {}", destination);
                return Err(NavigatorError::Routing(RoutingError::NoPathFound {
                    destination: destination.clone(),
                }));
            }
        };

        // Select appropriate CLA based on path characteristics
        let cla_selection = self.select_cla_for_path(&path, &qos, deadline, privacy_cap).await?;

        // Update statistics
        let selection_time = start_time.elapsed().as_millis() as f64;
        self.update_stats(&cla_selection.cla_name, selection_time, privacy_cap).await;

        info!(
            "Selected {} for {} with cost {} ({}ms)",
            cla_selection.cla_name,
            destination,
            cla_selection.estimated_cost,
            selection_time
        );

        Ok(cla_selection)
    }

    /// Select appropriate CLA based on path characteristics and requirements
    async fn select_cla_for_path(
        &self,
        path: &crate::route::LabeledPath,
        qos: &QosRequirements,
        deadline: Option<u64>,
        privacy_cap: Option<f64>,
    ) -> Result<ClaSelection, NavigatorError> {
        let clas = self.clas.read().await;

        if clas.is_empty() {
            return Err(NavigatorError::NoSuitableCla);
        }

        let mut best_cla: Option<(String, f64)> = None;
        let weights = qos.to_weight_vector();

        // Evaluate each available CLA
        for (cla_name, _cla) in clas.iter() {
            if let Some(characteristics) = self.cla_characteristics.get(cla_name) {
                let cla_cost = self.estimate_cla_cost(characteristics, path, privacy_cap);
                let score = cla_cost.scalarize(&weights);

                // Check constraints
                if let Some(dl) = deadline {
                    let estimated_completion = path.arrival_time + (characteristics.typical_latency_ms as u64);
                    if estimated_completion > dl {
                        continue; // CLA too slow for deadline
                    }
                }

                if best_cla.is_none() || score < best_cla.as_ref().unwrap().1 {
                    best_cla = Some((cla_name.clone(), score));
                }
            }
        }

        let (selected_cla, _score) = best_cla.ok_or(NavigatorError::NoSuitableCla)?;
        let characteristics = self.cla_characteristics.get(&selected_cla).unwrap();

        // Determine custody transfer requirement
        let needs_custody = self.should_request_custody(path, characteristics, qos);

        // Generate CLA-specific parameters
        let params = self.generate_cla_params(&selected_cla, path, qos, privacy_cap);

        // Create rationale
        let rationale = self.generate_selection_rationale(&selected_cla, path, qos);

        Ok(ClaSelection {
            cla_name: selected_cla,
            custody: needs_custody,
            params,
            estimated_cost: self.estimate_cla_cost(characteristics, path, privacy_cap),
            rationale,
        })
    }

    /// Estimate CLA cost based on characteristics and path
    fn estimate_cla_cost(
        &self,
        characteristics: &ClaCharacteristics,
        path: &crate::route::LabeledPath,
        privacy_cap: Option<f64>,
    ) -> Cost {
        // Base cost from CLA characteristics
        let latency_ms = characteristics.typical_latency_ms;
        let energy_mah = characteristics.energy_cost_factor * 5.0; // Base energy per hop
        let reliability = characteristics.reliability;

        // Privacy cost based on features and requirements
        let privacy_eps = if privacy_cap.is_some() {
            if characteristics.privacy_features.contains(&"mixnode-routing".to_string()) {
                0.1 // Mixnode routing provides good privacy
            } else if characteristics.privacy_features.contains(&"mesh-routing".to_string()) {
                0.2 // Mesh routing provides some privacy
            } else {
                0.5 // Direct routing, higher epsilon consumption
            }
        } else {
            0.0 // No privacy requirements
        };

        // Combine with path cost using semiring multiplication
        let cla_cost = Cost::new(latency_ms, energy_mah, reliability, privacy_eps);
        path.cost * cla_cost
    }

    /// Determine if custody transfer should be requested
    fn should_request_custody(
        &self,
        path: &crate::route::LabeledPath,
        characteristics: &ClaCharacteristics,
        qos: &QosRequirements,
    ) -> bool {
        // Request custody for:
        // 1. Long paths (>3 hops)
        // 2. Low reliability CLAs
        // 3. High reliability QoS requirements

        if !characteristics.supports_custody {
            return false;
        }

        path.hop_count() > 3
            || characteristics.reliability < 0.9
            || qos.reliability_priority > 0.5
    }

    /// Generate CLA-specific parameters
    fn generate_cla_params(
        &self,
        cla_name: &str,
        _path: &crate::route::LabeledPath,
        qos: &QosRequirements,
        privacy_cap: Option<f64>,
    ) -> HashMap<String, String> {
        let mut params = HashMap::new();

        match cla_name {
            "bitchat" => {
                params.insert("mesh_ttl".to_string(), "7".to_string());
                params.insert("store_forward".to_string(), "true".to_string());
                if qos.energy_priority > 0.5 {
                    params.insert("power_mode".to_string(), "low".to_string());
                }
            }
            "betanet-htx" => {
                if privacy_cap.is_some() {
                    params.insert("use_mixnodes".to_string(), "true".to_string());
                    params.insert("mixnode_hops".to_string(), "2".to_string());
                }
                if qos.latency_priority > 0.5 {
                    params.insert("tcp_nodelay".to_string(), "true".to_string());
                }
            }
            "betanet-quic" => {
                params.insert("mode".to_string(), "datagram".to_string());
                if qos.latency_priority > 0.5 {
                    params.insert("priority".to_string(), "high".to_string());
                }
            }
            _ => {}
        }

        params
    }

    /// Generate human-readable selection rationale
    fn generate_selection_rationale(
        &self,
        cla_name: &str,
        path: &crate::route::LabeledPath,
        qos: &QosRequirements,
    ) -> String {
        let primary_factor = if qos.latency_priority > 0.4 {
            "low latency"
        } else if qos.energy_priority > 0.4 {
            "energy efficiency"
        } else if qos.reliability_priority > 0.4 {
            "reliability"
        } else if qos.privacy_priority > 0.4 {
            "privacy"
        } else {
            "balanced performance"
        };

        format!(
            "Selected {} for {} with {} hops, optimizing for {}",
            cla_name,
            path.destination,
            path.hop_count(),
            primary_factor
        )
    }

    /// Update navigator statistics
    async fn update_stats(
        &self,
        cla_name: &str,
        selection_time_ms: f64,
        privacy_cap: Option<f64>,
    ) {
        let mut stats = self.stats.write().await;

        stats.total_selections += 1;
        *stats.selections_by_cla.entry(cla_name.to_string()).or_insert(0) += 1;

        // Update running average of selection time
        let alpha = 0.1; // EWMA smoothing factor
        stats.avg_selection_time_ms = alpha * selection_time_ms + (1.0 - alpha) * stats.avg_selection_time_ms;

        // Update privacy utilization if applicable
        if let Some(cap) = privacy_cap {
            let utilization = self.privacy_composer.budget_consumed / cap;
            stats.avg_privacy_utilization = alpha * utilization + (1.0 - alpha) * stats.avg_privacy_utilization;
        }
    }

    /// Get navigator statistics
    pub async fn get_stats(&self) -> NavigatorStats {
        self.stats.read().await.clone()
    }

    /// Update DTN contact plan
    pub async fn update_contact_plan(&self, contacts: Vec<Contact>) {
        let mut router = self.router.write().await;
        let dtn_router = router.dtn_router_mut();

        // Clear existing contacts and add new ones
        // This would integrate with the DTN router's contact plan management
        for contact in contacts {
            dtn_router.add_contact(contact);
        }
    }

    /// Get DTN router reference for advanced operations
    pub async fn dtn_router(&self) -> tokio::sync::RwLockReadGuard<'_, SemiringRouter> {
        self.router.read().await
    }
}

impl PrivacyComposer {
    /// Create new privacy composer
    pub fn new(budget_limit: f64, mode: PrivacyMode) -> Self {
        Self {
            budget_consumed: 0.0,
            budget_limit,
            mode,
        }
    }

    /// Check if privacy budget allows for additional consumption
    pub fn check_budget(&self, requested: f64) -> Result<(), NavigatorError> {
        if self.budget_consumed + requested > self.budget_limit {
            return Err(NavigatorError::PrivacyBudgetExceeded {
                consumed: self.budget_consumed + requested,
                limit: self.budget_limit,
            });
        }
        Ok(())
    }

    /// Consume privacy budget
    pub fn consume_budget(&mut self, amount: f64) {
        self.budget_consumed += amount;
    }

    /// Reset privacy budget
    pub fn reset_budget(&mut self) {
        self.budget_consumed = 0.0;
    }

    /// Get remaining privacy budget
    pub fn remaining_budget(&self) -> f64 {
        self.budget_limit - self.budget_consumed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use betanet_dtn::{EndpointId, RoutingPolicy};

    #[tokio::test]
    async fn test_navigator_creation() {
        let local_node = EndpointId::node("local");
        let dtn_router = ContactGraphRouter::new(local_node.clone(), RoutingPolicy::default());
        let navigator = Navigator::new(dtn_router, local_node);

        let stats = navigator.get_stats().await;
        assert_eq!(stats.total_selections, 0);
    }

    #[test]
    fn test_privacy_composer() {
        let mut composer = PrivacyComposer::new(1.0, PrivacyMode::Basic);

        assert!(composer.check_budget(0.5).is_ok());
        composer.consume_budget(0.3);
        assert!(composer.check_budget(0.5).is_ok());
        composer.consume_budget(0.5);
        assert!(composer.check_budget(0.5).is_err()); // Would exceed budget
    }

    #[test]
    fn test_cla_characteristics() {
        let local_node = EndpointId::node("local");
        let dtn_router = ContactGraphRouter::new(local_node.clone(), RoutingPolicy::default());
        let navigator = Navigator::new(dtn_router, local_node);

        // Should have registered BitChat characteristics
        assert!(navigator.cla_characteristics.contains_key("bitchat"));
        let bitchat = &navigator.cla_characteristics["bitchat"];
        assert_eq!(bitchat.transport_type, "bluetooth-mesh");
        assert!(bitchat.supports_custody);

        // Should have registered Betanet characteristics
        assert!(navigator.cla_characteristics.contains_key("betanet-htx"));
        let betanet = &navigator.cla_characteristics["betanet-htx"];
        assert_eq!(betanet.transport_type, "tcp-tls");
        assert!(betanet.real_time_capable);
    }
}
