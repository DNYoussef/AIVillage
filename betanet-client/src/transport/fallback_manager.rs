//! Transport fallback manager for QUIC→TCP automatic failover
//!
//! Provides intelligent fallback between transport types based on network
//! conditions, failure rates, and performance metrics.

use crate::{
    config::FallbackConfig,
    error::{BetanetError, Result, TransportError},
    transport::TransportType,
    BetanetMessage, BetanetPeer,
};

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Fallback manager for transport switching
pub struct FallbackManager {
    /// Configuration
    config: FallbackConfig,
    /// Transport health monitors
    health_monitors: Arc<RwLock<HashMap<TransportType, TransportHealthMonitor>>>,
    /// Current primary transport
    primary_transport: Arc<RwLock<Option<TransportType>>>,
    /// Current fallback transport
    fallback_transport: Arc<RwLock<Option<TransportType>>>,
    /// Transport registry
    transport_registry: Arc<RwLock<HashMap<TransportType, Arc<dyn TransportInterface + Send + Sync>>>>,
    /// Fallback state
    fallback_state: Arc<RwLock<FallbackState>>,
    /// Running state
    is_running: Arc<RwLock<bool>>,
}

/// Transport interface trait
pub trait TransportInterface {
    /// Send message via this transport
    async fn send_message(&self, recipient: String, message: BetanetMessage) -> Result<()>;

    /// Check if transport is available
    async fn is_available(&self) -> bool;

    /// Get transport health metrics
    async fn get_health_metrics(&self) -> TransportHealthMetrics;

    /// Discover peers via this transport
    async fn discover_peers(&self) -> Result<Vec<BetanetPeer>>;
}

/// Transport health monitor
#[derive(Debug, Clone)]
struct TransportHealthMonitor {
    /// Transport type
    transport_type: TransportType,
    /// Success rate (0.0-1.0)
    success_rate: f64,
    /// Average latency
    average_latency: Duration,
    /// Recent failure count
    recent_failures: u32,
    /// Last health check
    last_health_check: Instant,
    /// Health check interval
    health_check_interval: Duration,
    /// Failure threshold for switching
    failure_threshold: u32,
    /// Is currently healthy
    is_healthy: bool,
}

/// Transport health metrics
#[derive(Debug, Clone, Default)]
pub struct TransportHealthMetrics {
    /// Total attempts
    pub total_attempts: u64,
    /// Successful attempts
    pub successful_attempts: u64,
    /// Failed attempts
    pub failed_attempts: u64,
    /// Average latency
    pub average_latency_ms: f64,
    /// Current availability
    pub is_available: bool,
    /// Last error
    pub last_error: Option<String>,
}

/// Fallback state
#[derive(Debug, Clone)]
struct FallbackState {
    /// Currently in fallback mode
    in_fallback: bool,
    /// Fallback start time
    fallback_started_at: Option<Instant>,
    /// Fallback attempt count
    fallback_attempts: u32,
    /// Last primary transport check
    last_primary_check: Instant,
    /// Pending recovery
    pending_recovery: bool,
}

/// Fallback decision
#[derive(Debug, Clone)]
pub struct FallbackDecision {
    /// Selected transport
    pub transport: TransportType,
    /// Reason for selection
    pub reason: String,
    /// Confidence score (0.0-1.0)
    pub confidence: f64,
    /// Expected latency
    pub expected_latency: Duration,
}

impl FallbackManager {
    /// Create new fallback manager
    pub async fn new(config: FallbackConfig) -> Result<Self> {
        info!("Creating fallback manager (enabled: {})", config.enabled);

        Ok(Self {
            config,
            health_monitors: Arc::new(RwLock::new(HashMap::new())),
            primary_transport: Arc::new(RwLock::new(None)),
            fallback_transport: Arc::new(RwLock::new(None)),
            transport_registry: Arc::new(RwLock::new(HashMap::new())),
            fallback_state: Arc::new(RwLock::new(FallbackState {
                in_fallback: false,
                fallback_started_at: None,
                fallback_attempts: 0,
                last_primary_check: Instant::now(),
                pending_recovery: false,
            })),
            is_running: Arc::new(RwLock::new(false)),
        })
    }

    /// Start fallback manager with transport registry
    pub async fn start(
        &self,
        transports: Vec<(TransportType, Arc<dyn TransportInterface + Send + Sync>)>,
    ) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        if *is_running {
            warn!("Fallback manager already running");
            return Ok(());
        }

        info!("Starting fallback manager...");

        // Register transports
        let mut registry = self.transport_registry.write().await;
        let mut monitors = self.health_monitors.write().await;

        for (transport_type, transport_impl) in transports {
            registry.insert(transport_type, transport_impl);

            monitors.insert(transport_type, TransportHealthMonitor {
                transport_type,
                success_rate: 1.0,
                average_latency: Duration::from_millis(100),
                recent_failures: 0,
                last_health_check: Instant::now(),
                health_check_interval: self.config.health_check_interval,
                failure_threshold: self.config.failure_threshold,
                is_healthy: true,
            });
        }

        // Set initial primary transport (prefer QUIC)
        if monitors.contains_key(&TransportType::Quic) {
            *self.primary_transport.write().await = Some(TransportType::Quic);
            *self.fallback_transport.write().await = Some(TransportType::Tcp);
        } else if monitors.contains_key(&TransportType::Tcp) {
            *self.primary_transport.write().await = Some(TransportType::Tcp);
        }

        drop(registry);
        drop(monitors);

        // Start health monitoring
        if self.config.enabled {
            self.start_health_monitoring().await;
            self.start_recovery_monitoring().await;
        }

        *is_running = true;
        info!("Fallback manager started successfully");

        Ok(())
    }

    /// Stop fallback manager
    pub async fn stop(&self) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        if !*is_running {
            warn!("Fallback manager not running");
            return Ok(());
        }

        info!("Stopping fallback manager...");

        // Clear state
        self.transport_registry.write().await.clear();
        self.health_monitors.write().await.clear();
        *self.primary_transport.write().await = None;
        *self.fallback_transport.write().await = None;

        *is_running = false;
        info!("Fallback manager stopped");

        Ok(())
    }

    /// Handle transport failure and attempt fallback
    pub async fn handle_transport_failure(
        &self,
        recipient: String,
        message: BetanetMessage,
        failed_transport: TransportType,
    ) -> Result<()> {
        if !self.config.enabled {
            return Err(BetanetError::Transport(TransportError::Fallback(
                "Fallback disabled".to_string()
            )));
        }

        warn!("Transport failure detected: {:?}", failed_transport);

        // Update health monitor
        self.record_transport_failure(failed_transport).await;

        // Make fallback decision
        let decision = self.make_fallback_decision(failed_transport).await?;

        info!("Fallback decision: using {:?} (reason: {})", decision.transport, decision.reason);

        // Update fallback state
        let mut state = self.fallback_state.write().await;
        if !state.in_fallback {
            state.in_fallback = true;
            state.fallback_started_at = Some(Instant::now());
            state.fallback_attempts = 0;
        }
        state.fallback_attempts += 1;
        drop(state);

        // Attempt delivery via fallback transport
        self.send_via_transport(recipient, message, decision.transport).await
    }

    /// Select best available transport
    pub async fn select_best_available_transport(&self) -> Option<TransportType> {
        let monitors = self.health_monitors.read().await;

        // Find the healthiest available transport
        let mut best_transport = None;
        let mut best_score = 0.0f64;

        for (transport_type, monitor) in monitors.iter() {
            if monitor.is_healthy {
                let score = monitor.success_rate *
                           (1.0 / (monitor.average_latency.as_millis() as f64 + 1.0)) *
                           (1.0 / (monitor.recent_failures as f64 + 1.0));

                if score > best_score {
                    best_score = score;
                    best_transport = Some(*transport_type);
                }
            }
        }

        best_transport
    }

    /// Get current primary transport
    pub async fn get_primary_transport(&self) -> Option<TransportType> {
        *self.primary_transport.read().await
    }

    /// Get current fallback transport
    pub async fn get_fallback_transport(&self) -> Option<TransportType> {
        let state = self.fallback_state.read().await;
        if state.in_fallback {
            *self.fallback_transport.read().await
        } else {
            None
        }
    }

    /// Record transport success
    pub async fn record_transport_success(&self, transport_type: TransportType, latency: Duration) {
        let mut monitors = self.health_monitors.write().await;
        if let Some(monitor) = monitors.get_mut(&transport_type) {
            // Update success rate (exponential moving average)
            monitor.success_rate = monitor.success_rate * 0.9 + 0.1;

            // Update average latency
            let current_ms = monitor.average_latency.as_millis() as f64;
            let new_ms = latency.as_millis() as f64;
            monitor.average_latency = Duration::from_millis(
                (current_ms * 0.9 + new_ms * 0.1) as u64
            );

            // Reset failure count on success
            monitor.recent_failures = 0;
            monitor.is_healthy = true;
            monitor.last_health_check = Instant::now();
        }
    }

    /// Record transport failure
    async fn record_transport_failure(&self, transport_type: TransportType) {
        let mut monitors = self.health_monitors.write().await;
        if let Some(monitor) = monitors.get_mut(&transport_type) {
            // Update success rate
            monitor.success_rate = monitor.success_rate * 0.9;

            // Increment failure count
            monitor.recent_failures += 1;

            // Mark as unhealthy if failures exceed threshold
            if monitor.recent_failures >= monitor.failure_threshold {
                monitor.is_healthy = false;
                warn!("Transport {:?} marked as unhealthy (failures: {})",
                      transport_type, monitor.recent_failures);
            }

            monitor.last_health_check = Instant::now();
        }
    }

    /// Make fallback decision based on current state
    async fn make_fallback_decision(&self, failed_transport: TransportType) -> Result<FallbackDecision> {
        let monitors = self.health_monitors.read().await;
        let state = self.fallback_state.read().await;

        // Check if we've exceeded max fallback attempts
        if state.fallback_attempts >= self.config.max_fallback_attempts {
            return Err(BetanetError::Transport(TransportError::Fallback(
                "Maximum fallback attempts exceeded".to_string()
            )));
        }

        // Find best alternative transport
        let mut best_alternative = None;
        let mut best_score = 0.0f64;

        for (transport_type, monitor) in monitors.iter() {
            if *transport_type != failed_transport && monitor.is_healthy {
                let score = monitor.success_rate * (1.0 / (monitor.recent_failures as f64 + 1.0));
                if score > best_score {
                    best_score = score;
                    best_alternative = Some(*transport_type);
                }
            }
        }

        let transport = best_alternative.ok_or_else(|| {
            BetanetError::Transport(TransportError::Fallback(
                "No healthy alternative transport available".to_string()
            ))
        })?;

        let monitor = monitors.get(&transport).unwrap();

        Ok(FallbackDecision {
            transport,
            reason: format!("Primary transport {:?} failed, switching to {:?}", failed_transport, transport),
            confidence: monitor.success_rate,
            expected_latency: monitor.average_latency,
        })
    }

    /// Send message via specific transport
    async fn send_via_transport(
        &self,
        recipient: String,
        message: BetanetMessage,
        transport_type: TransportType,
    ) -> Result<()> {
        let registry = self.transport_registry.read().await;
        let transport = registry.get(&transport_type).ok_or_else(|| {
            BetanetError::Transport(TransportError::Unavailable(
                format!("Transport {:?} not available", transport_type)
            ))
        })?;

        let start_time = Instant::now();
        let result = transport.send_message(recipient, message).await;
        let latency = start_time.elapsed();

        // Record result
        if result.is_ok() {
            self.record_transport_success(transport_type, latency).await;
        } else {
            self.record_transport_failure(transport_type).await;
        }

        result
    }

    /// Start health monitoring background task
    async fn start_health_monitoring(&self) {
        let health_monitors = self.health_monitors.clone();
        let transport_registry = self.transport_registry.clone();
        let health_check_interval = self.config.health_check_interval;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(health_check_interval);

            loop {
                interval.tick().await;

                let monitors = health_monitors.read().await;
                let registry = transport_registry.read().await;

                for (transport_type, monitor) in monitors.iter() {
                    if let Some(transport) = registry.get(transport_type) {
                        // Perform health check
                        let is_available = transport.is_available().await;
                        let metrics = transport.get_health_metrics().await;

                        // Update monitor based on health check
                        drop(monitors);
                        let mut monitors_write = health_monitors.write().await;
                        if let Some(monitor_mut) = monitors_write.get_mut(transport_type) {
                            monitor_mut.is_healthy = is_available && metrics.is_available;
                            monitor_mut.last_health_check = Instant::now();

                            if !monitor_mut.is_healthy {
                                debug!("Health check failed for transport {:?}", transport_type);
                            }
                        }
                        drop(monitors_write);
                        let monitors = health_monitors.read().await;
                    }
                }
            }
        });
    }

    /// Start recovery monitoring to switch back to primary transport
    async fn start_recovery_monitoring(&self) {
        let health_monitors = self.health_monitors.clone();
        let primary_transport = self.primary_transport.clone();
        let fallback_state = self.fallback_state.clone();
        let fallback_timeout = self.config.fallback_timeout;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(fallback_timeout);

            loop {
                interval.tick().await;

                let state = fallback_state.read().await;
                if !state.in_fallback {
                    continue;
                }

                // Check if primary transport has recovered
                let primary = *primary_transport.read().await;
                if let Some(primary_transport_type) = primary {
                    let monitors = health_monitors.read().await;
                    if let Some(monitor) = monitors.get(&primary_transport_type) {
                        if monitor.is_healthy && monitor.success_rate > 0.8 {
                            // Primary transport appears healthy, attempt recovery
                            drop(state);
                            drop(monitors);

                            let mut state_write = fallback_state.write().await;
                            state_write.in_fallback = false;
                            state_write.fallback_started_at = None;
                            state_write.fallback_attempts = 0;
                            state_write.pending_recovery = false;

                            info!("Recovered to primary transport: {:?}", primary_transport_type);
                        }
                    }
                }
            }
        });
    }
}

impl Default for FallbackState {
    fn default() -> Self {
        Self {
            in_fallback: false,
            fallback_started_at: None,
            fallback_attempts: 0,
            last_primary_check: Instant::now(),
            pending_recovery: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FallbackConfig;

    #[tokio::test]
    async fn test_fallback_manager_creation() {
        let config = FallbackConfig::default();
        let manager = FallbackManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[test]
    fn test_fallback_decision() {
        let decision = FallbackDecision {
            transport: TransportType::Tcp,
            reason: "QUIC failed".to_string(),
            confidence: 0.9,
            expected_latency: Duration::from_millis(100),
        };

        assert_eq!(decision.transport, TransportType::Tcp);
        assert_eq!(decision.confidence, 0.9);
    }

    #[test]
    fn test_transport_health_monitor() {
        let monitor = TransportHealthMonitor {
            transport_type: TransportType::Quic,
            success_rate: 0.95,
            average_latency: Duration::from_millis(50),
            recent_failures: 0,
            last_health_check: Instant::now(),
            health_check_interval: Duration::from_secs(60),
            failure_threshold: 3,
            is_healthy: true,
        };

        assert!(monitor.is_healthy);
        assert_eq!(monitor.success_rate, 0.95);
        assert_eq!(monitor.recent_failures, 0);
    }
}
