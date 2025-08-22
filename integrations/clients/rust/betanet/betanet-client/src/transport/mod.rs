//! Transport layer implementation for Betanet

use crate::{
    config::TransportConfig,
    error::{BetanetError, Result, TransportError},
    BetanetMessage, BetanetPeer,
};

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};

pub mod htx;
pub mod quic_transport;
pub mod tcp_transport;
pub mod fallback_manager;
pub mod chrome_fingerprint;

use htx::HtxTransport;
use quic_transport::QuicTransport;
use tcp_transport::TcpTransport;
use fallback_manager::FallbackManager;

/// Transport types available
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransportType {
    /// QUIC transport (primary)
    Quic,
    /// TCP transport (fallback)
    Tcp,
    /// HTTP/1.1 over TCP
    Http1,
    /// HTTP/2 over TCP/TLS
    Http2,
    /// HTTP/3 over QUIC
    Http3,
}

/// Transport status information
#[derive(Debug, Clone, serde::Serialize)]
pub struct TransportStatus {
    /// Available transports
    pub available_transports: Vec<TransportType>,
    /// Primary active transport
    pub primary_transport: Option<TransportType>,
    /// Fallback active transport
    pub fallback_transport: Option<TransportType>,
    /// Connection statistics
    pub connection_stats: HashMap<TransportType, ConnectionStats>,
    /// Health status
    pub health_status: HealthStatus,
}

/// Connection statistics per transport
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConnectionStats {
    /// Total connections established
    pub total_connections: u64,
    /// Active connections
    pub active_connections: u32,
    /// Failed connections
    pub failed_connections: u64,
    /// Average latency
    pub average_latency_ms: f64,
    /// Bytes sent
    pub bytes_sent: u64,
    /// Bytes received
    pub bytes_received: u64,
}

/// Health status
#[derive(Debug, Clone, serde::Serialize)]
pub enum HealthStatus {
    /// All transports healthy
    Healthy,
    /// Primary transport degraded, using fallback
    Degraded,
    /// Limited connectivity
    Limited,
    /// No connectivity
    Unavailable,
}

/// Transport manager coordinates all transport types
pub struct TransportManager {
    /// Configuration
    config: TransportConfig,
    /// HTX transport for covert communication
    htx_transport: Arc<HtxTransport>,
    /// QUIC transport for primary communication
    quic_transport: Arc<QuicTransport>,
    /// TCP transport for fallback
    tcp_transport: Arc<TcpTransport>,
    /// Fallback manager
    fallback_manager: Arc<FallbackManager>,
    /// Message sender for client communication
    message_sender: mpsc::UnboundedSender<crate::client::InternalMessage>,
    /// Running state
    is_running: Arc<RwLock<bool>>,
    /// Transport statistics
    stats: Arc<RwLock<HashMap<TransportType, ConnectionStats>>>,
}

impl TransportManager {
    /// Create new transport manager
    pub async fn new(
        config: TransportConfig,
        message_sender: mpsc::UnboundedSender<crate::client::InternalMessage>,
    ) -> Result<Self> {
        info!("Creating transport manager");

        // Create HTX transport for covert communication
        let htx_transport = Arc::new(
            HtxTransport::new(config.http.clone()).await?
        );

        // Create QUIC transport
        let quic_transport = Arc::new(
            QuicTransport::new(config.quic.clone()).await?
        );

        // Create TCP transport
        let tcp_transport = Arc::new(
            TcpTransport::new(config.tcp.clone()).await?
        );

        // Create fallback manager
        let fallback_manager = Arc::new(
            FallbackManager::new(config.fallback.clone()).await?
        );

        Ok(Self {
            config,
            htx_transport,
            quic_transport,
            tcp_transport,
            fallback_manager,
            message_sender,
            is_running: Arc::new(RwLock::new(false)),
            stats: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Start transport manager
    pub async fn start(&self) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        if *is_running {
            warn!("Transport manager already running");
            return Ok(());
        }

        info!("Starting transport manager...");

        // Initialize statistics
        let mut stats = self.stats.write().await;
        for transport_type in [TransportType::Quic, TransportType::Tcp, TransportType::Http1, TransportType::Http2, TransportType::Http3] {
            stats.insert(transport_type, ConnectionStats::default());
        }
        drop(stats);

        // Start transports
        self.htx_transport.start().await?;
        self.quic_transport.start().await?;
        self.tcp_transport.start().await?;

        // Start fallback manager
        self.fallback_manager.start(
            vec![
                (TransportType::Quic, self.quic_transport.clone()),
                (TransportType::Tcp, self.tcp_transport.clone()),
            ]
        ).await?;

        // Start message processing
        self.start_message_processing().await?;

        *is_running = true;
        info!("Transport manager started successfully");

        Ok(())
    }

    /// Stop transport manager
    pub async fn stop(&self) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        if !*is_running {
            warn!("Transport manager not running");
            return Ok(());
        }

        info!("Stopping transport manager...");

        // Stop components
        self.fallback_manager.stop().await?;
        self.tcp_transport.stop().await?;
        self.quic_transport.stop().await?;
        self.htx_transport.stop().await?;

        *is_running = false;
        info!("Transport manager stopped");

        Ok(())
    }

    /// Send message using optimal transport
    pub async fn send_message(&self, recipient: String, message: BetanetMessage) -> Result<()> {
        let is_running = self.is_running.read().await;
        if !*is_running {
            return Err(BetanetError::Transport(TransportError::Unavailable(
                "Transport manager not running".to_string()
            )));
        }

        // Determine optimal transport based on message characteristics
        let transport_choice = self.select_optimal_transport(&message).await?;

        // Send via selected transport
        self.send_message_via_transport(recipient, message, transport_choice).await
    }

    /// Send message via specific transport
    pub async fn send_message_via_transport(
        &self,
        recipient: String,
        message: BetanetMessage,
        transport_type: TransportType,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        let result = match transport_type {
            TransportType::Quic => {
                self.quic_transport.send_message(recipient, message).await
            }
            TransportType::Tcp => {
                self.tcp_transport.send_message(recipient, message).await
            }
            TransportType::Http1 | TransportType::Http2 | TransportType::Http3 => {
                self.htx_transport.send_message(recipient, message, transport_type).await
            }
        };

        // Update statistics
        let latency = start_time.elapsed().as_millis() as f64;
        self.update_transport_stats(transport_type, result.is_ok(), latency).await;

        // Handle fallback on failure
        if result.is_err() && self.config.fallback.enabled {
            warn!("Primary transport failed, attempting fallback");
            return self.fallback_manager.handle_transport_failure(
                recipient,
                message,
                transport_type,
            ).await;
        }

        result
    }

    /// Discover peers via all available transports
    pub async fn discover_peers(&self) -> Result<Vec<BetanetPeer>> {
        let mut all_peers = Vec::new();

        // Discover via QUIC
        if let Ok(quic_peers) = self.quic_transport.discover_peers().await {
            all_peers.extend(quic_peers);
        }

        // Discover via TCP
        if let Ok(tcp_peers) = self.tcp_transport.discover_peers().await {
            all_peers.extend(tcp_peers);
        }

        // Discover via HTX (if available)
        if let Ok(htx_peers) = self.htx_transport.discover_peers().await {
            all_peers.extend(htx_peers);
        }

        // Deduplicate peers
        let mut unique_peers = HashMap::new();
        for peer in all_peers {
            unique_peers.insert(peer.peer_id.clone(), peer);
        }

        Ok(unique_peers.into_values().collect())
    }

    /// Get transport status
    pub async fn get_status(&self) -> TransportStatus {
        let stats = self.stats.read().await;
        let connection_stats = stats.clone();

        // Determine available transports
        let mut available_transports = Vec::new();
        if self.quic_transport.is_available().await {
            available_transports.push(TransportType::Quic);
        }
        if self.tcp_transport.is_available().await {
            available_transports.push(TransportType::Tcp);
        }
        if self.htx_transport.is_available().await {
            available_transports.extend(vec![
                TransportType::Http1,
                TransportType::Http2,
                TransportType::Http3,
            ]);
        }

        // Determine primary and fallback transports
        let primary_transport = self.fallback_manager.get_primary_transport().await;
        let fallback_transport = self.fallback_manager.get_fallback_transport().await;

        // Determine health status
        let health_status = if available_transports.is_empty() {
            HealthStatus::Unavailable
        } else if primary_transport.is_none() {
            HealthStatus::Limited
        } else if fallback_transport.is_some() && primary_transport != fallback_transport {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        TransportStatus {
            available_transports,
            primary_transport,
            fallback_transport,
            connection_stats,
            health_status,
        }
    }

    /// Select optimal transport for message
    async fn select_optimal_transport(&self, message: &BetanetMessage) -> Result<TransportType> {
        // Check message requirements
        let requires_covert = message.routing_metadata
            .as_ref()
            .and_then(|meta| meta.get("covert_required"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let high_priority = matches!(message.priority, crate::MessagePriority::High | crate::MessagePriority::Emergency);
        let large_payload = message.payload.len() > 32768; // 32KB

        // Transport selection logic
        if requires_covert {
            // Use covert HTX transport
            if self.htx_transport.is_available().await {
                return Ok(TransportType::Http2); // HTTP/2 is most common for covert
            }
        }

        if high_priority && self.quic_transport.is_available().await {
            // Use QUIC for low latency
            return Ok(TransportType::Quic);
        }

        if large_payload && self.quic_transport.is_available().await {
            // Use QUIC for large payloads (multiplexing)
            return Ok(TransportType::Quic);
        }

        // Default fallback selection
        self.fallback_manager.select_best_available_transport().await
            .ok_or_else(|| BetanetError::Transport(TransportError::Unavailable(
                "No transports available".to_string()
            )))
    }

    /// Update transport statistics
    async fn update_transport_stats(&self, transport_type: TransportType, success: bool, latency_ms: f64) {
        let mut stats = self.stats.write().await;
        let stat = stats.entry(transport_type).or_insert_with(ConnectionStats::default);

        if success {
            stat.total_connections += 1;
            // Update average latency (simple moving average)
            stat.average_latency_ms = (stat.average_latency_ms * 0.9) + (latency_ms * 0.1);
        } else {
            stat.failed_connections += 1;
        }
    }

    /// Start message processing for incoming messages
    async fn start_message_processing(&self) -> Result<()> {
        // Each transport will handle its own incoming messages
        // and forward them to the client via message_sender

        let message_sender = self.message_sender.clone();

        // Set up message handlers for each transport
        self.quic_transport.set_message_handler(Box::new(move |sender, message| {
            let sender_clone = message_sender.clone();
            tokio::spawn(async move {
                let _ = sender_clone.send(crate::client::InternalMessage::ReceiveMessage {
                    sender,
                    message,
                });
            });
        })).await?;

        // Similar setup for other transports...

        Ok(())
    }
}

impl Default for ConnectionStats {
    fn default() -> Self {
        Self {
            total_connections: 0,
            active_connections: 0,
            failed_connections: 0,
            average_latency_ms: 0.0,
            bytes_sent: 0,
            bytes_received: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TransportConfig;

    #[tokio::test]
    async fn test_transport_manager_creation() {
        let config = TransportConfig::default();
        let (sender, _receiver) = mpsc::unbounded_channel();

        let manager = TransportManager::new(config, sender).await;
        assert!(manager.is_ok());
    }

    #[test]
    fn test_transport_type_serialization() {
        let transport = TransportType::Quic;
        assert_eq!(format!("{:?}", transport), "Quic");
    }

    #[test]
    fn test_connection_stats_default() {
        let stats = ConnectionStats::default();
        assert_eq!(stats.total_connections, 0);
        assert_eq!(stats.failed_connections, 0);
        assert_eq!(stats.average_latency_ms, 0.0);
    }
}
