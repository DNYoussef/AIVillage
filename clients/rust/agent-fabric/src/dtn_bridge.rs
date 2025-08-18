//! DTN bridge for async bundle delivery via betanet-dtn
//!
//! Provides delay-tolerant messaging for agent communication when real-time
//! connectivity is not available.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Mutex, RwLock};
use tracing::{debug, error, info, warn};

use betanet_dtn::api::{BundleEvent, IncomingBundle, RegistrationInfo};
use betanet_dtn::{
    BundleId, DtnError, DtnNode, EndpointId, RoutingPolicy, SendBundleOptions as DtnSendOptions,
};

use crate::{
    AgentClient, AgentFabricError, AgentId, AgentMessage, AgentResponse, AgentServer,
    DeliveryOptions, MessagePriority, Result,
};

/// Bundle message wrapper for agent messaging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleMessage {
    pub from: AgentId,
    pub to: AgentId,
    pub content: AgentMessage,
    pub options: DeliveryOptions,
}

/// Bundle receipt notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleReceipt {
    pub bundle_id: BundleId,
    pub delivered: bool,
    pub timestamp: u64,
    pub error: Option<String>,
}

/// DTN bridge for agent messaging
pub struct DtnBridge {
    node: Arc<DtnNode>,
    local_endpoint: EndpointId,
    message_handlers: Arc<RwLock<HashMap<String, Box<dyn AgentServer>>>>,
    pending_responses: Arc<RwLock<HashMap<BundleId, tokio::sync::oneshot::Sender<AgentResponse>>>>,
    event_receiver: Arc<Mutex<Option<mpsc::Receiver<BundleEvent>>>>,
    bundle_receiver: Arc<Mutex<Option<mpsc::Receiver<IncomingBundle>>>>,
    stats: Arc<RwLock<BundleStats>>,
}

#[derive(Debug, Clone, Default)]
pub struct BundleStats {
    pub bundles_sent: u64,
    pub bundles_received: u64,
    pub bundles_delivered: u64,
    pub bundles_failed: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
}

impl DtnBridge {
    /// Create new DTN bridge
    pub async fn new(local_endpoint: EndpointId, storage_path: impl AsRef<Path>) -> Result<Self> {
        Self::with_policy(local_endpoint, storage_path, RoutingPolicy::default()).await
    }

    /// Create new DTN bridge with routing policy
    pub async fn with_policy(
        local_endpoint: EndpointId,
        storage_path: impl AsRef<Path>,
        policy: RoutingPolicy,
    ) -> Result<Self> {
        let (node, event_receiver) = DtnNode::new(local_endpoint.clone(), storage_path, policy)
            .await
            .map_err(AgentFabricError::DtnError)?;

        let node = Arc::new(node);

        // Register application for receiving bundles
        let bundle_receiver = node
            .register_application(
                local_endpoint.clone(),
                "agent-fabric".to_string(),
                1000, // Queue size
            )
            .await
            .map_err(AgentFabricError::DtnError)?;

        let bridge = Self {
            node,
            local_endpoint,
            message_handlers: Arc::new(RwLock::new(HashMap::new())),
            pending_responses: Arc::new(RwLock::new(HashMap::new())),
            event_receiver: Arc::new(Mutex::new(Some(event_receiver))),
            bundle_receiver: Arc::new(Mutex::new(Some(bundle_receiver))),
            stats: Arc::new(RwLock::new(BundleStats::default())),
        };

        Ok(bridge)
    }

    /// Start the DTN bridge
    pub async fn start(&self) -> Result<()> {
        // Start DTN node
        self.node
            .start()
            .await
            .map_err(AgentFabricError::DtnError)?;

        // Start background tasks
        self.start_background_tasks().await;

        info!("DTN bridge started for endpoint: {}", self.local_endpoint);
        Ok(())
    }

    /// Stop the DTN bridge
    pub async fn stop(&self) -> Result<()> {
        self.node.stop().await.map_err(AgentFabricError::DtnError)?;
        info!("DTN bridge stopped");
        Ok(())
    }

    /// Send bundle to destination agent
    pub async fn send_bundle(
        &self,
        destination: EndpointId,
        message: BundleMessage,
        options: DeliveryOptions,
    ) -> Result<BundleId> {
        let payload = serde_json::to_vec(&message)
            .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?;

        let dtn_options = DtnSendOptions {
            lifetime: options
                .timeout_ms
                .map(Duration::from_millis)
                .unwrap_or(Duration::from_secs(24 * 60 * 60)), // 24 hours default
            priority: match options.priority {
                MessagePriority::Low => 0,
                MessagePriority::Normal => 1,
                MessagePriority::High => 2,
                MessagePriority::Critical => 2,
            },
            request_custody: false,
            request_delivery_report: options.require_receipt,
            request_receipt_report: options.require_receipt,
        };

        let bundle_id = self
            .node
            .send_bundle(destination.clone(), Bytes::from(payload), dtn_options)
            .await
            .map_err(AgentFabricError::DtnError)?;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.bundles_sent += 1;
            stats.bytes_sent += message.content.payload.len() as u64;
        }

        debug!("Sent bundle {} to {} via DTN", bundle_id, destination);

        Ok(bundle_id)
    }

    /// Register message handler for incoming bundles
    pub async fn register_handler(&self, message_type: String, handler: Box<dyn AgentServer>) {
        let mut handlers = self.message_handlers.write().await;
        handlers.insert(message_type, handler);
    }

    /// Get bundle statistics
    pub async fn get_stats(&self) -> BundleStats {
        self.stats.read().await.clone()
    }

    /// Get DTN node statistics
    pub async fn get_dtn_stats(&self) -> Result<betanet_dtn::BundleStats> {
        Ok(self.node.get_stats().await)
    }

    /// Get local endpoint
    pub fn local_endpoint(&self) -> &EndpointId {
        &self.local_endpoint
    }

    // Private methods

    async fn start_background_tasks(&self) {
        // Start bundle processing task
        let bridge_clone = Arc::new(self.clone());
        tokio::spawn(async move {
            bridge_clone.bundle_processing_loop().await;
        });

        // Start event processing task
        let bridge_clone = Arc::new(self.clone());
        tokio::spawn(async move {
            bridge_clone.event_processing_loop().await;
        });
    }

    async fn bundle_processing_loop(&self) {
        let mut receiver = {
            let mut recv_guard = self.bundle_receiver.lock().await;
            recv_guard.take()
        };

        if let Some(mut receiver) = receiver {
            while let Some(incoming) = receiver.recv().await {
                if let Err(e) = self.process_incoming_bundle(incoming).await {
                    error!("Error processing incoming bundle: {}", e);
                }
            }
        }
    }

    async fn event_processing_loop(&self) {
        let mut receiver = {
            let mut recv_guard = self.event_receiver.lock().await;
            recv_guard.take()
        };

        if let Some(mut receiver) = receiver {
            while let Some(event) = receiver.recv().await {
                self.process_bundle_event(event).await;
            }
        }
    }

    async fn process_incoming_bundle(&self, incoming: IncomingBundle) -> Result<()> {
        let bundle_id = incoming.bundle.id();
        debug!("Processing incoming bundle: {}", bundle_id);

        // Deserialize bundle message
        let bundle_message: BundleMessage = serde_json::from_slice(&incoming.bundle.payload.data.0)
            .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.bundles_received += 1;
            stats.bytes_received += incoming.bundle.payload.data.0.len() as u64;
        }

        // Route to appropriate handler
        let handlers = self.message_handlers.read().await;
        if let Some(handler) = handlers.get(&bundle_message.content.message_type) {
            match handler
                .handle_message(bundle_message.from.clone(), bundle_message.content.clone())
                .await
            {
                Ok(Some(response)) => {
                    // For async bundle delivery, we typically don't send responses back
                    // unless specifically requested. Could implement response bundles here.
                    debug!("Handler processed bundle successfully: {}", bundle_id);

                    let mut stats = self.stats.write().await;
                    stats.bundles_delivered += 1;
                }
                Ok(None) => {
                    debug!("Handler processed bundle (no response): {}", bundle_id);

                    let mut stats = self.stats.write().await;
                    stats.bundles_delivered += 1;
                }
                Err(e) => {
                    warn!("Handler error for bundle {}: {}", bundle_id, e);

                    let mut stats = self.stats.write().await;
                    stats.bundles_failed += 1;
                }
            }
        } else {
            warn!(
                "No handler for message type '{}' in bundle {}",
                bundle_message.content.message_type, bundle_id
            );

            let mut stats = self.stats.write().await;
            stats.bundles_failed += 1;
        }

        Ok(())
    }

    async fn process_bundle_event(&self, event: BundleEvent) {
        match event {
            BundleEvent::BundleReceived {
                bundle_id,
                source,
                destination,
                payload_size,
            } => {
                debug!(
                    "Bundle {} received: {} -> {} ({} bytes)",
                    bundle_id, source, destination, payload_size
                );
            }
            BundleEvent::BundleForwarded {
                bundle_id,
                next_hop,
                cla_name,
            } => {
                debug!(
                    "Bundle {} forwarded to {} via {}",
                    bundle_id, next_hop, cla_name
                );
            }
            BundleEvent::BundleDelivered {
                bundle_id,
                to_application,
            } => {
                info!(
                    "Bundle {} delivered to application {}",
                    bundle_id, to_application
                );
            }
            BundleEvent::BundleExpired { bundle_id } => {
                warn!("Bundle {} expired", bundle_id);

                let mut stats = self.stats.write().await;
                stats.bundles_failed += 1;
            }
            BundleEvent::BundleDropped { bundle_id, reason } => {
                warn!("Bundle {} dropped: {}", bundle_id, reason);

                let mut stats = self.stats.write().await;
                stats.bundles_failed += 1;
            }
            BundleEvent::CustodyAccepted { bundle_id } => {
                debug!("Custody accepted for bundle {}", bundle_id);
            }
            BundleEvent::DeliveryReport {
                bundle_id,
                delivered,
            } => {
                debug!(
                    "Delivery report for bundle {}: {}",
                    bundle_id,
                    if delivered { "delivered" } else { "failed" }
                );
            }
        }
    }
}

// Clone implementation for background tasks
impl Clone for DtnBridge {
    fn clone(&self) -> Self {
        Self {
            node: Arc::clone(&self.node),
            local_endpoint: self.local_endpoint.clone(),
            message_handlers: Arc::clone(&self.message_handlers),
            pending_responses: Arc::clone(&self.pending_responses),
            event_receiver: Arc::clone(&self.event_receiver),
            bundle_receiver: Arc::clone(&self.bundle_receiver),
            stats: Arc::clone(&self.stats),
        }
    }
}

/// DTN-based agent client for async messaging
pub struct DtnClient {
    bridge: Arc<DtnBridge>,
    target_agent: AgentId,
}

impl DtnClient {
    pub fn new(bridge: Arc<DtnBridge>, target_agent: AgentId) -> Self {
        Self {
            bridge,
            target_agent,
        }
    }

    pub async fn send_bundle(
        &self,
        message: AgentMessage,
        options: DeliveryOptions,
    ) -> Result<BundleId> {
        let bundle_message = BundleMessage {
            from: AgentId::new("local", self.bridge.local_endpoint.to_string()),
            to: self.target_agent.clone(),
            content: message,
            options: options.clone(),
        };

        let destination = self.target_agent.to_endpoint();
        self.bridge
            .send_bundle(destination, bundle_message, options)
            .await
    }
}

#[async_trait]
impl AgentClient for DtnClient {
    async fn send_message(&self, message: AgentMessage) -> Result<AgentResponse> {
        // For DTN/bundle delivery, we typically don't get immediate responses
        // This could be enhanced to support request-response patterns over DTN
        let _bundle_id = self
            .send_bundle(message.clone(), DeliveryOptions::default())
            .await?;

        // Return a success response indicating the bundle was sent
        Ok(AgentResponse::success(
            &message.id,
            Bytes::from("Bundle sent via DTN"),
        ))
    }

    async fn send_notification(&self, message: AgentMessage) -> Result<()> {
        let _bundle_id = self
            .send_bundle(message, DeliveryOptions::default())
            .await?;
        Ok(())
    }

    fn target_agent(&self) -> &AgentId {
        &self.target_agent
    }
}

/// DTN server wrapper for handling bundle messages
pub struct DtnServer {
    bridge: Arc<DtnBridge>,
}

impl DtnServer {
    pub fn new(bridge: Arc<DtnBridge>) -> Self {
        Self { bridge }
    }

    pub async fn register_handler(&self, message_type: String, handler: Box<dyn AgentServer>) {
        self.bridge.register_handler(message_type, handler).await;
    }

    pub async fn start(&self) -> Result<()> {
        self.bridge.start().await
    }

    pub async fn stop(&self) -> Result<()> {
        self.bridge.stop().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::TempDir;

    struct TestDtnServer {
        call_count: AtomicUsize,
    }

    impl TestDtnServer {
        fn new() -> Self {
            Self {
                call_count: AtomicUsize::new(0),
            }
        }

        fn call_count(&self) -> usize {
            self.call_count.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl AgentServer for TestDtnServer {
        async fn handle_message(
            &self,
            _from: AgentId,
            message: AgentMessage,
        ) -> Result<Option<AgentResponse>> {
            self.call_count.fetch_add(1, Ordering::Relaxed);

            if message.message_type == "ping" {
                Ok(Some(AgentResponse::success(
                    &message.id,
                    Bytes::from("pong"),
                )))
            } else {
                Ok(Some(AgentResponse::error(
                    &message.id,
                    "Unknown message type",
                )))
            }
        }

        fn supported_message_types(&self) -> Vec<String> {
            vec!["ping".to_string()]
        }
    }

    #[tokio::test]
    async fn test_dtn_bridge_creation() {
        let temp_dir = TempDir::new().unwrap();
        let endpoint = EndpointId::node("test-node");

        let bridge = DtnBridge::new(endpoint.clone(), temp_dir.path())
            .await
            .unwrap();
        assert_eq!(bridge.local_endpoint(), &endpoint);
    }

    #[tokio::test]
    async fn test_bundle_message_serialization() {
        let bundle_msg = BundleMessage {
            from: AgentId::new("sender", "node1"),
            to: AgentId::new("receiver", "node2"),
            content: AgentMessage::new("test", Bytes::from("hello")),
            options: DeliveryOptions::default(),
        };

        let serialized = serde_json::to_vec(&bundle_msg).unwrap();
        let deserialized: BundleMessage = serde_json::from_slice(&serialized).unwrap();

        assert_eq!(bundle_msg.from, deserialized.from);
        assert_eq!(bundle_msg.to, deserialized.to);
        assert_eq!(
            bundle_msg.content.message_type,
            deserialized.content.message_type
        );
    }

    #[tokio::test]
    async fn test_dtn_client_creation() {
        let temp_dir = TempDir::new().unwrap();
        let endpoint = EndpointId::node("test-node");
        let target = AgentId::new("target", "target-node");

        let bridge = Arc::new(DtnBridge::new(endpoint, temp_dir.path()).await.unwrap());
        let client = DtnClient::new(bridge, target.clone());

        assert_eq!(client.target_agent(), &target);
    }

    #[test]
    fn test_bundle_stats() {
        let mut stats = BundleStats::default();
        assert_eq!(stats.bundles_sent, 0);
        assert_eq!(stats.bytes_sent, 0);

        stats.bundles_sent += 1;
        stats.bytes_sent += 1024;

        assert_eq!(stats.bundles_sent, 1);
        assert_eq!(stats.bytes_sent, 1024);
    }
}
