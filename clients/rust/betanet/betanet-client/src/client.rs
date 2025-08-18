//! Main Betanet client implementation

use crate::{
    config::BetanetConfig,
    error::{BetanetError, Result},
    gateway::GatewayManager,
    metrics::MetricsCollector,
    transport::{TransportManager, TransportType},
    BetanetMessage, BetanetPeer, MessagePriority,
};

use anyhow::Context;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Message handler function type
pub type MessageHandler = Arc<dyn Fn(BetanetMessage) -> Result<()> + Send + Sync>;

/// Main Betanet client
pub struct BetanetClient {
    /// Configuration
    config: BetanetConfig,
    /// Transport manager
    transport_manager: Arc<TransportManager>,
    /// Gateway manager
    gateway_manager: Option<Arc<GatewayManager>>,
    /// Metrics collector
    metrics: Arc<MetricsCollector>,
    /// Peer registry
    peers: Arc<RwLock<HashMap<String, BetanetPeer>>>,
    /// Message handlers
    message_handlers: Arc<RwLock<HashMap<String, MessageHandler>>>,
    /// Running state
    is_running: Arc<RwLock<bool>>,
    /// Message sender for internal communication
    message_sender: mpsc::UnboundedSender<InternalMessage>,
    /// Message receiver for internal communication
    message_receiver: Arc<RwLock<Option<mpsc::UnboundedReceiver<InternalMessage>>>>,
}

/// Internal messages for client communication
#[derive(Debug)]
enum InternalMessage {
    /// Send message to peer
    SendMessage {
        recipient: String,
        message: BetanetMessage,
        response_sender: Option<tokio::sync::oneshot::Sender<Result<()>>>,
    },
    /// Receive message from peer
    ReceiveMessage {
        sender: String,
        message: BetanetMessage,
    },
    /// Peer discovered
    PeerDiscovered {
        peer: BetanetPeer,
    },
    /// Peer lost
    PeerLost {
        peer_id: String,
    },
    /// Transport status change
    TransportStatusChange {
        transport_type: TransportType,
        available: bool,
    },
    /// Shutdown signal
    Shutdown,
}

impl BetanetClient {
    /// Create new Betanet client
    pub async fn new(config: BetanetConfig) -> Result<Self> {
        info!("Creating Betanet client with node ID: {}", config.node_id);

        // Validate configuration
        config.validate().context("Invalid configuration")?;

        // Create message channel
        let (message_sender, message_receiver) = mpsc::unbounded_channel();

        // Create transport manager
        let transport_manager = Arc::new(
            TransportManager::new(config.transport.clone(), message_sender.clone()).await?
        );

        // Create gateway manager if enabled
        let gateway_manager = if config.gateway.enabled {
            Some(Arc::new(
                GatewayManager::new(config.gateway.clone()).await?
            ))
        } else {
            None
        };

        // Create metrics collector
        let metrics = Arc::new(MetricsCollector::new(&config.integration.monitoring)?);

        Ok(Self {
            config,
            transport_manager,
            gateway_manager,
            metrics,
            peers: Arc::new(RwLock::new(HashMap::new())),
            message_handlers: Arc::new(RwLock::new(HashMap::new())),
            is_running: Arc::new(RwLock::new(false)),
            message_sender,
            message_receiver: Arc::new(RwLock::new(Some(message_receiver))),
        })
    }

    /// Start the Betanet client
    pub async fn start(&self) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        if *is_running {
            warn!("Betanet client already running");
            return Ok(());
        }

        info!("Starting Betanet client...");

        // Start transport manager
        self.transport_manager.start().await?;

        // Start gateway manager if enabled
        if let Some(gateway) = &self.gateway_manager {
            gateway.start().await?;
        }

        // Start message processing loop
        self.start_message_processing().await?;

        // Start peer discovery
        self.start_peer_discovery().await?;

        // Start metrics collection
        self.metrics.start().await?;

        *is_running = true;
        info!("Betanet client started successfully");

        Ok(())
    }

    /// Stop the Betanet client
    pub async fn stop(&self) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        if !*is_running {
            warn!("Betanet client not running");
            return Ok(());
        }

        info!("Stopping Betanet client...");

        // Send shutdown signal
        if let Err(e) = self.message_sender.send(InternalMessage::Shutdown) {
            warn!("Failed to send shutdown signal: {}", e);
        }

        // Stop components
        self.metrics.stop().await?;

        if let Some(gateway) = &self.gateway_manager {
            gateway.stop().await?;
        }

        self.transport_manager.stop().await?;

        *is_running = false;
        info!("Betanet client stopped");

        Ok(())
    }

    /// Send message to peer
    pub async fn send_message(
        &self,
        recipient: String,
        payload: bytes::Bytes,
        priority: MessagePriority,
    ) -> Result<()> {
        let is_running = self.is_running.read().await;
        if !*is_running {
            return Err(BetanetError::Internal("Client not running".to_string()));
        }

        // Create message
        let mut message = BetanetMessage::new(self.config.node_id.clone(), recipient.clone(), payload);
        message.priority = priority;
        message.compute_content_hash();

        // Record metrics
        self.metrics.increment_counter("messages_sent_total", &[("recipient", &recipient)]);

        // Send via transport manager
        let (response_sender, response_receiver) = tokio::sync::oneshot::channel();

        self.message_sender.send(InternalMessage::SendMessage {
            recipient,
            message,
            response_sender: Some(response_sender),
        })?;

        // Wait for response
        response_receiver.await.map_err(|e| {
            BetanetError::Internal(format!("Failed to receive send response: {}", e))
        })?
    }

    /// Send message via specific transport
    pub async fn send_message_via_transport(
        &self,
        recipient: String,
        payload: bytes::Bytes,
        transport_type: TransportType,
        priority: MessagePriority,
    ) -> Result<()> {
        let is_running = self.is_running.read().await;
        if !*is_running {
            return Err(BetanetError::Internal("Client not running".to_string()));
        }

        // Create message
        let mut message = BetanetMessage::new(self.config.node_id.clone(), recipient.clone(), payload);
        message.priority = priority;
        message.compute_content_hash();

        // Send via specific transport
        self.transport_manager.send_message_via_transport(recipient, message, transport_type).await
    }

    /// Broadcast message to all peers
    pub async fn broadcast_message(
        &self,
        payload: bytes::Bytes,
        priority: MessagePriority,
    ) -> Result<usize> {
        let is_running = self.is_running.read().await;
        if !*is_running {
            return Err(BetanetError::Internal("Client not running".to_string()));
        }

        let peers = self.peers.read().await;
        let peer_count = peers.len();

        // Create message
        let mut message = BetanetMessage::new(self.config.node_id.clone(), "".to_string(), payload);
        message.priority = priority;
        message.compute_content_hash();

        // Send to all peers
        let mut sent_count = 0;
        for peer_id in peers.keys() {
            let mut broadcast_msg = message.clone();
            broadcast_msg.recipient = peer_id.clone();
            broadcast_msg.id = Uuid::new_v4(); // New ID for each recipient

            if let Ok(()) = self.transport_manager.send_message(peer_id.clone(), broadcast_msg).await {
                sent_count += 1;
            }
        }

        self.metrics.increment_counter("messages_broadcast_total", &[]);
        self.metrics.record_histogram("broadcast_peer_count", sent_count as f64, &[]);

        Ok(sent_count)
    }

    /// Register message handler
    pub async fn register_message_handler<F>(&self, message_type: String, handler: F) -> Result<()>
    where
        F: Fn(BetanetMessage) -> Result<()> + Send + Sync + 'static,
    {
        let mut handlers = self.message_handlers.write().await;
        handlers.insert(message_type.clone(), Arc::new(handler));

        debug!("Registered message handler for type: {}", message_type);
        Ok(())
    }

    /// Get peer information
    pub async fn get_peer(&self, peer_id: &str) -> Option<BetanetPeer> {
        let peers = self.peers.read().await;
        peers.get(peer_id).cloned()
    }

    /// Get all peers
    pub async fn get_all_peers(&self) -> Vec<BetanetPeer> {
        let peers = self.peers.read().await;
        peers.values().cloned().collect()
    }

    /// Get client status
    pub async fn get_status(&self) -> ClientStatus {
        let is_running = *self.is_running.read().await;
        let peer_count = self.peers.read().await.len();
        let transport_status = self.transport_manager.get_status().await;

        ClientStatus {
            node_id: self.config.node_id.clone(),
            is_running,
            peer_count,
            transport_status,
            uptime: self.metrics.get_uptime(),
        }
    }

    /// Start message processing loop
    async fn start_message_processing(&self) -> Result<()> {
        let mut receiver_opt = self.message_receiver.write().await;
        let receiver = receiver_opt.take().ok_or_else(|| {
            BetanetError::Internal("Message receiver already taken".to_string())
        })?;

        let transport_manager = self.transport_manager.clone();
        let peers = self.peers.clone();
        let handlers = self.message_handlers.clone();
        let metrics = self.metrics.clone();
        let is_running = self.is_running.clone();

        tokio::spawn(async move {
            let mut receiver = receiver;

            while let Some(msg) = receiver.recv().await {
                match msg {
                    InternalMessage::SendMessage { recipient, message, response_sender } => {
                        let result = transport_manager.send_message(recipient, message).await;

                        if let Some(sender) = response_sender {
                            let _ = sender.send(result);
                        }
                    }

                    InternalMessage::ReceiveMessage { sender, message } => {
                        Self::handle_received_message(message, &handlers, &metrics).await;
                    }

                    InternalMessage::PeerDiscovered { peer } => {
                        let mut peers_map = peers.write().await;
                        peers_map.insert(peer.peer_id.clone(), peer);
                        metrics.increment_counter("peers_discovered_total", &[]);
                    }

                    InternalMessage::PeerLost { peer_id } => {
                        let mut peers_map = peers.write().await;
                        peers_map.remove(&peer_id);
                        metrics.increment_counter("peers_lost_total", &[]);
                    }

                    InternalMessage::TransportStatusChange { transport_type, available } => {
                        debug!("Transport status change: {:?} = {}", transport_type, available);
                        metrics.record_gauge(
                            "transport_availability",
                            if available { 1.0 } else { 0.0 },
                            &[("transport", &format!("{:?}", transport_type))],
                        );
                    }

                    InternalMessage::Shutdown => {
                        info!("Message processing loop shutting down");
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    /// Handle received message
    async fn handle_received_message(
        message: BetanetMessage,
        handlers: &Arc<RwLock<HashMap<String, MessageHandler>>>,
        metrics: &Arc<MetricsCollector>,
    ) {
        // Check if message is expired
        if message.is_expired() {
            debug!("Dropping expired message: {}", message.id);
            metrics.increment_counter("messages_expired_total", &[]);
            return;
        }

        // Update metrics
        metrics.increment_counter("messages_received_total", &[("sender", &message.sender)]);

        // Find appropriate handler
        let handlers_map = handlers.read().await;
        let handler = handlers_map
            .get(&message.content_type)
            .or_else(|| handlers_map.get("default"));

        if let Some(handler) = handler {
            if let Err(e) = handler(message) {
                error!("Message handler error: {}", e);
                metrics.increment_counter("message_handler_errors_total", &[]);
            }
        } else {
            debug!("No handler for message type: {}", message.content_type);
            metrics.increment_counter("messages_unhandled_total", &[]);
        }
    }

    /// Start peer discovery
    async fn start_peer_discovery(&self) -> Result<()> {
        let transport_manager = self.transport_manager.clone();
        let message_sender = self.message_sender.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));

            loop {
                interval.tick().await;

                // Discover peers via transport manager
                if let Ok(discovered_peers) = transport_manager.discover_peers().await {
                    for peer in discovered_peers {
                        let _ = message_sender.send(InternalMessage::PeerDiscovered { peer });
                    }
                }
            }
        });

        Ok(())
    }
}

/// Client status information
#[derive(Debug, Clone, serde::Serialize)]
pub struct ClientStatus {
    /// Node identifier
    pub node_id: String,
    /// Running state
    pub is_running: bool,
    /// Number of connected peers
    pub peer_count: usize,
    /// Transport status
    pub transport_status: crate::transport::TransportStatus,
    /// Client uptime
    pub uptime: std::time::Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BetanetConfig;

    #[tokio::test]
    async fn test_client_creation() {
        let config = BetanetConfig::default();
        let client = BetanetClient::new(config).await;
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_client_start_stop() {
        let config = BetanetConfig::default();
        let client = BetanetClient::new(config).await.unwrap();

        // Start client
        assert!(client.start().await.is_ok());

        let status = client.get_status().await;
        assert!(status.is_running);

        // Stop client
        assert!(client.stop().await.is_ok());

        let status = client.get_status().await;
        assert!(!status.is_running);
    }

    #[tokio::test]
    async fn test_message_handler_registration() {
        let config = BetanetConfig::default();
        let client = BetanetClient::new(config).await.unwrap();

        let result = client.register_message_handler(
            "test_type".to_string(),
            |_msg| Ok(()),
        ).await;

        assert!(result.is_ok());
    }
}
