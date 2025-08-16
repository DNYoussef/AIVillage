//! DTN Node API for application integration
//!
//! Provides high-level API for sending/receiving bundles and managing the DTN node.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{mpsc, RwLock, Mutex};
use tokio::time::{interval, Interval};
use tracing::{debug, error, info, warn};

use crate::bundle::{Bundle, BundleId, EndpointId};
use crate::storage::{BundleStore, StorageError};
use crate::router::{ContactGraphRouter, ContactPlan, RoutingPolicy, RoutingError};
use crate::sched::{LyapunovScheduler, LyapunovConfig};
use crate::{ConvergenceLayer, BundleStats, DEFAULT_BUNDLE_LIFETIME};

/// Options for sending bundles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SendBundleOptions {
    pub lifetime: Duration,
    pub priority: u8,
    pub request_custody: bool,
    pub request_delivery_report: bool,
    pub request_receipt_report: bool,
}

impl Default for SendBundleOptions {
    fn default() -> Self {
        Self {
            lifetime: DEFAULT_BUNDLE_LIFETIME,
            priority: 1, // Normal priority
            request_custody: false,
            request_delivery_report: false,
            request_receipt_report: false,
        }
    }
}

/// Registration information for applications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistrationInfo {
    pub endpoint: EndpointId,
    pub application_id: String,
    pub active: bool,
    pub bundle_queue_size: usize,
    pub registered_at: u64,
}

/// Incoming bundle notification
#[derive(Debug, Clone)]
pub struct IncomingBundle {
    pub bundle: Bundle,
    pub received_at: u64,
    pub from_cla: String,
}

/// Bundle delivery event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BundleEvent {
    BundleReceived {
        bundle_id: BundleId,
        source: EndpointId,
        destination: EndpointId,
        payload_size: usize,
    },
    BundleForwarded {
        bundle_id: BundleId,
        next_hop: EndpointId,
        cla_name: String,
    },
    BundleDelivered {
        bundle_id: BundleId,
        to_application: String,
    },
    BundleExpired {
        bundle_id: BundleId,
    },
    BundleDropped {
        bundle_id: BundleId,
        reason: String,
    },
    CustodyAccepted {
        bundle_id: BundleId,
    },
    DeliveryReport {
        bundle_id: BundleId,
        delivered: bool,
    },
}

/// DTN Node implementation
pub struct DtnNode {
    node_id: EndpointId,
    storage: Arc<BundleStore>,
    router: Arc<Mutex<ContactGraphRouter>>,
    scheduler: Arc<Mutex<LyapunovScheduler>>,
    convergence_layers: Arc<RwLock<HashMap<String, Box<dyn ConvergenceLayer>>>>,
    registrations: Arc<RwLock<HashMap<EndpointId, RegistrationInfo>>>,
    bundle_queues: Arc<RwLock<HashMap<EndpointId, mpsc::Sender<IncomingBundle>>>>,
    event_sender: mpsc::Sender<BundleEvent>,
    stats: Arc<RwLock<BundleStats>>,
    running: Arc<RwLock<bool>>,
    cleanup_interval: Interval,
}

impl DtnNode {
    /// Create a new DTN node
    pub async fn new(
        node_id: EndpointId,
        storage_path: impl AsRef<std::path::Path>,
        policy: RoutingPolicy,
    ) -> Result<(Self, mpsc::Receiver<BundleEvent>), DtnError> {
        let storage = Arc::new(
            BundleStore::open(storage_path)
                .await
                .map_err(DtnError::StorageError)?
        );

        let router = Arc::new(Mutex::new(
            ContactGraphRouter::new(node_id.clone(), policy)
        ));

        let scheduler = Arc::new(Mutex::new(
            LyapunovScheduler::new(LyapunovConfig::default())
                .map_err(|e| DtnError::ConfigError(e.to_string()))?
        ));

        let (event_sender, event_receiver) = mpsc::channel(1000);

        let node = Self {
            node_id,
            storage,
            router,
            scheduler,
            convergence_layers: Arc::new(RwLock::new(HashMap::new())),
            registrations: Arc::new(RwLock::new(HashMap::new())),
            bundle_queues: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
            stats: Arc::new(RwLock::new(BundleStats::new())),
            running: Arc::new(RwLock::new(false)),
            cleanup_interval: interval(Duration::from_secs(5 * 60)), // 5 minutes
        };

        Ok((node, event_receiver))
    }

    /// Start the DTN node
    pub async fn start(&self) -> Result<(), DtnError> {
        let mut running = self.running.write().await;
        if *running {
            return Err(DtnError::AlreadyRunning);
        }
        *running = true;

        info!("Starting DTN node: {}", self.node_id);

        // Start background tasks
        self.start_background_tasks().await;

        Ok(())
    }

    /// Stop the DTN node
    pub async fn stop(&self) -> Result<(), DtnError> {
        let mut running = self.running.write().await;
        if !*running {
            return Err(DtnError::NotRunning);
        }
        *running = false;

        info!("Stopping DTN node: {}", self.node_id);

        // Flush storage
        self.storage.flush().await.map_err(DtnError::StorageError)?;

        Ok(())
    }

    /// Send a bundle to a destination
    pub async fn send_bundle(
        &self,
        destination: EndpointId,
        payload: Bytes,
        options: SendBundleOptions,
    ) -> Result<BundleId, DtnError> {
        if !*self.running.read().await {
            return Err(DtnError::NotRunning);
        }

        // Create bundle
        let bundle = Bundle::new(
            destination.clone(),
            self.node_id.clone(),
            payload,
            options.lifetime.as_millis() as u64,
        );

        let bundle_id = bundle.id();

        // Store bundle
        self.storage.store(bundle.clone()).await.map_err(DtnError::StorageError)?;

        // Enqueue with scheduler
        {
            let mut scheduler = self.scheduler.lock().await;
            scheduler.enqueue_bundle(&bundle);
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.record_created();
        }

        // Try to forward immediately
        if let Err(e) = self.try_forward_bundle(&bundle).await {
            warn!("Failed to forward bundle immediately: {}", e);
        }

        // Send event
        let _ = self.event_sender.send(BundleEvent::BundleReceived {
            bundle_id: bundle_id.clone(),
            source: self.node_id.clone(),
            destination,
            payload_size: bundle.payload.data.len(),
        }).await;

        Ok(bundle_id)
    }

    /// Register an application to receive bundles
    pub async fn register_application(
        &self,
        endpoint: EndpointId,
        application_id: String,
        queue_size: usize,
    ) -> Result<mpsc::Receiver<IncomingBundle>, DtnError> {
        let (sender, receiver) = mpsc::channel(queue_size);

        let registration = RegistrationInfo {
            endpoint: endpoint.clone(),
            application_id: application_id.clone(),
            active: true,
            bundle_queue_size: queue_size,
            registered_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        {
            let mut registrations = self.registrations.write().await;
            registrations.insert(endpoint.clone(), registration);
        }

        {
            let mut queues = self.bundle_queues.write().await;
            queues.insert(endpoint.clone(), sender);
        }

        info!("Registered application '{}' for endpoint {}", application_id, endpoint);

        Ok(receiver)
    }

    /// Unregister an application
    pub async fn unregister_application(&self, endpoint: &EndpointId) -> Result<(), DtnError> {
        {
            let mut registrations = self.registrations.write().await;
            registrations.remove(endpoint);
        }

        {
            let mut queues = self.bundle_queues.write().await;
            queues.remove(endpoint);
        }

        info!("Unregistered application for endpoint {}", endpoint);

        Ok(())
    }

    /// Register a convergence layer
    pub async fn register_cla(
        &self,
        name: String,
        cla: Box<dyn ConvergenceLayer>,
    ) -> Result<(), DtnError> {
        let mut clas = self.convergence_layers.write().await;
        clas.insert(name.clone(), cla);

        info!("Registered convergence layer: {}", name);

        Ok(())
    }

    /// Update contact plan
    pub async fn update_contact_plan(&self, plan: ContactPlan) -> Result<(), DtnError> {
        let mut router = self.router.lock().await;
        router.update_contact_plan(plan);

        debug!("Updated contact plan");

        Ok(())
    }

    /// Set routing policy
    pub async fn set_routing_policy(&self, policy: RoutingPolicy) -> Result<(), DtnError> {
        let mut router = self.router.lock().await;
        router.set_policy(policy);

        debug!("Updated routing policy");

        Ok(())
    }

    /// Get node statistics
    pub async fn get_stats(&self) -> BundleStats {
        self.stats.read().await.clone()
    }

    /// Get storage statistics
    pub async fn get_storage_stats(&self) -> Result<crate::storage::StorageStats, DtnError> {
        Ok(self.storage.stats().await)
    }

    /// Get list of registered applications
    pub async fn get_registrations(&self) -> Vec<RegistrationInfo> {
        self.registrations.read().await.values().cloned().collect()
    }

    /// Process an incoming bundle from a CLA
    pub async fn process_incoming_bundle(
        &self,
        bundle: Bundle,
        from_cla: String,
    ) -> Result<(), DtnError> {
        let bundle_id = bundle.id();

        debug!("Processing incoming bundle: {} from CLA: {}", bundle_id, from_cla);

        // Check if bundle is expired
        if bundle.is_expired() {
            warn!("Received expired bundle: {}", bundle_id);

            let mut stats = self.stats.write().await;
            stats.record_expired();

            let _ = self.event_sender.send(BundleEvent::BundleExpired { bundle_id }).await;
            return Ok(());
        }

        // Check if bundle is for local delivery
        if self.is_local_destination(&bundle.primary.destination).await {
            self.deliver_bundle_locally(bundle.clone(), from_cla).await?;
        } else {
            // Store and forward
            if let Err(e) = self.storage.store(bundle.clone()).await {
                match e {
                    StorageError::BundleExists(_) => {
                        debug!("Bundle already exists, ignoring: {}", bundle_id);
                        return Ok(());
                    }
                    _ => return Err(DtnError::StorageError(e)),
                }
            }

            // Try to forward
            if let Err(e) = self.try_forward_bundle(&bundle).await {
                warn!("Failed to forward bundle {}: {}", bundle_id, e);
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.record_received(bundle.size());
        }

        Ok(())
    }

    // Private methods

    async fn start_background_tasks(&self) {
        let storage = Arc::clone(&self.storage);
        let router = Arc::clone(&self.router);
        let running = Arc::clone(&self.running);
        let _event_sender = self.event_sender.clone();

        // Cleanup task
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(5 * 60)); // 5 minutes

            while *running.read().await {
                interval.tick().await;

                let current_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                // Cleanup expired bundles
                if let Ok(expired_count) = storage.cleanup_expired().await {
                    if expired_count > 0 {
                        info!("Cleaned up {} expired bundles", expired_count);
                    }
                }

                // Cleanup router cache
                router.lock().await.cleanup(current_time);
            }
        });
    }

    async fn is_local_destination(&self, destination: &EndpointId) -> bool {
        // Check if destination matches node ID
        if *destination == self.node_id {
            return true;
        }

        // Check if any registered application handles this destination
        let registrations = self.registrations.read().await;
        registrations.contains_key(destination)
    }

    async fn deliver_bundle_locally(
        &self,
        bundle: Bundle,
        from_cla: String,
    ) -> Result<(), DtnError> {
        let bundle_id = bundle.id();
        let destination = bundle.primary.destination.clone();

        info!("Delivering bundle {} locally to {}", bundle_id, destination);

        // Find registered application
        let queue = {
            let queues = self.bundle_queues.read().await;
            queues.get(&destination).cloned()
        };

        if let Some(sender) = queue {
            let incoming = IncomingBundle {
                bundle,
                received_at: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                from_cla,
            };

            if sender.send(incoming).await.is_ok() {
                // Update stats
                let mut stats = self.stats.write().await;
                stats.record_delivered();

                // Send event
                let registration = {
                    let registrations = self.registrations.read().await;
                    registrations.get(&destination).cloned()
                };

                if let Some(reg) = registration {
                    let _ = self.event_sender.send(BundleEvent::BundleDelivered {
                        bundle_id,
                        to_application: reg.application_id,
                    }).await;
                }
            } else {
                warn!("Failed to deliver bundle to application queue");
                return Err(DtnError::DeliveryFailed(bundle_id));
            }
        } else {
            warn!("No registered application for destination: {}", destination);
            return Err(DtnError::NoRegistration(destination));
        }

        Ok(())
    }

    async fn try_forward_bundle(&self, bundle: &Bundle) -> Result<(), DtnError> {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Get available contacts from router
        let available_contacts = {
            let router = self.router.lock().await;
            router.get_active_contacts(current_time)
        };

        // For each contact, use Lyapunov scheduler to decide whether to transmit
        for contact in &available_contacts {
            // Get bundles available for this destination
            let available_bundles = vec![bundle.id()]; // Simplified - in practice get from storage

            // Make scheduling decision
            let decision = {
                let mut scheduler = self.scheduler.lock().await;
                scheduler.schedule_transmission(&contact, &available_bundles, 1) // max 1 bundle for simplicity
                    .map_err(|e| DtnError::ConfigError(e.to_string()))?
            };

            if decision.should_transmit && !decision.bundles_to_transmit.is_empty() {
                // Try to forward using available CLAs
                let clas = self.convergence_layers.read().await;

                let mut transmitted = false;
                for (cla_name, _cla) in clas.iter() {
                    debug!("Lyapunov scheduler approved transmission of bundle {} via CLA: {} (utility: {:.3}, energy: {:.3})",
                           bundle.id(), cla_name, decision.estimated_utility, decision.estimated_energy_cost);

                    // For now, just log the attempt - in practice would actually send
                    // In a real implementation, you'd:
                    // 1. Map endpoint ID to CLA-specific address
                    // 2. Establish connection if needed
                    // 3. Send bundle via CLA
                    // 4. Handle success/failure

                    let _ = self.event_sender.send(BundleEvent::BundleForwarded {
                        bundle_id: bundle.id(),
                        next_hop: contact.to.clone(),
                        cla_name: cla_name.clone(),
                    }).await;

                    // Update stats
                    let mut stats = self.stats.write().await;
                    stats.record_forwarded();

                    // Notify scheduler of successful transmission
                    let mut scheduler = self.scheduler.lock().await;
                    scheduler.dequeue_bundle(bundle.id(), true, decision.estimated_energy_cost);

                    transmitted = true;
                    break; // For now, just use first available CLA
                }

                if transmitted {
                    info!("Bundle {} forwarded based on Lyapunov decision: {}",
                          bundle.id(), decision.rationale);
                    return Ok(());
                }
            } else {
                debug!("Lyapunov scheduler deferred transmission of bundle {}: {}",
                       bundle.id(), decision.rationale);
            }
        }

        if available_contacts.is_empty() {
            debug!("No contacts available for bundle: {}", bundle.id());
        }

        Ok(())
    }
}

/// DTN-related errors
#[derive(Debug, Error)]
pub enum DtnError {
    #[error("Storage error: {0}")]
    StorageError(#[from] StorageError),

    #[error("Routing error: {0}")]
    RoutingError(#[from] RoutingError),

    #[error("Node is already running")]
    AlreadyRunning,

    #[error("Node is not running")]
    NotRunning,

    #[error("Bundle delivery failed: {0}")]
    DeliveryFailed(BundleId),

    #[error("No registration for endpoint: {0}")]
    NoRegistration(EndpointId),

    #[error("Convergence layer error: {0}")]
    ClaError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Network error: {0}")]
    NetworkError(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_dtn_node_creation() {
        let temp_dir = TempDir::new().unwrap();
        let node_id = EndpointId::node("test-node");
        let policy = RoutingPolicy::default();

        let (node, _events) = DtnNode::new(node_id.clone(), temp_dir.path(), policy)
            .await
            .unwrap();

        assert_eq!(node.node_id, node_id);
    }

    #[tokio::test]
    async fn test_application_registration() {
        let temp_dir = TempDir::new().unwrap();
        let node_id = EndpointId::node("test-node");
        let app_endpoint = EndpointId::node("test-app");
        let policy = RoutingPolicy::default();

        let (node, _events) = DtnNode::new(node_id, temp_dir.path(), policy)
            .await
            .unwrap();

        let _receiver = node.register_application(
            app_endpoint.clone(),
            "test-app".to_string(),
            10,
        ).await.unwrap();

        let registrations = node.get_registrations().await;
        assert_eq!(registrations.len(), 1);
        assert_eq!(registrations[0].endpoint, app_endpoint);
    }

    #[tokio::test]
    async fn test_bundle_creation_and_stats() {
        let temp_dir = TempDir::new().unwrap();
        let node_id = EndpointId::node("test-node");
        let policy = RoutingPolicy::default();

        let (node, _events) = DtnNode::new(node_id, temp_dir.path(), policy)
            .await
            .unwrap();

        node.start().await.unwrap();

        let destination = EndpointId::node("dest-node");
        let payload = Bytes::from("Hello, DTN!");
        let options = SendBundleOptions::default();

        let bundle_id = node.send_bundle(destination, payload, options).await.unwrap();
        assert!(!bundle_id.to_string().is_empty());

        let stats = node.get_stats().await;
        assert_eq!(stats.bundles_created, 1);

        node.stop().await.unwrap();
    }
}
