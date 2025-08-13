//! Betanet Rust Client
//!
//! A high-performance, covert communication client implementing:
//! - HTX covert transport over TLS/H2/H3
//! - Noise XK inner tunnel for forward secrecy
//! - QUIC→TCP fallback mechanism
//! - Chrome fingerprint calibration for censorship resistance
//! - SCION-ish gateway with signed CBOR control

#![warn(missing_docs)]
#![warn(clippy::all)]

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

pub mod client;
pub mod config;
pub mod error;
pub mod gateway;
pub mod integration;
pub mod metrics;
pub mod transport;

// Re-exports for convenience
pub use client::BetanetClient;
pub use config::{BetanetConfig, TransportConfig, SecurityConfig};
pub use error::BetanetError;

#[cfg(feature = "python-ffi")]
pub mod python_ffi;

/// Core Betanet client result type
pub type BetanetResult<T> = Result<T, BetanetError>;

/// Message priority levels compatible with existing AIVillage system
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    /// Low priority (3)
    Low = 3,
    /// Normal priority (5)
    Normal = 5,
    /// High priority (7)
    High = 7,
    /// Emergency priority (10)
    Emergency = 10,
}

impl From<u8> for MessagePriority {
    fn from(value: u8) -> Self {
        match value {
            1..=3 => MessagePriority::Low,
            4..=6 => MessagePriority::Normal,
            7..=9 => MessagePriority::High,
            _ => MessagePriority::Emergency,
        }
    }
}

impl From<MessagePriority> for u8 {
    fn from(priority: MessagePriority) -> Self {
        priority as u8
    }
}

/// Betanet message format compatible with existing Python implementation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BetanetMessage {
    /// Unique message identifier
    pub id: uuid::Uuid,
    /// Protocol version (htx/1.1, htxquic/1.1)
    pub protocol: String,
    /// Sender peer ID
    pub sender: String,
    /// Recipient peer ID
    pub recipient: String,
    /// Message payload
    pub payload: bytes::Bytes,
    /// Content type
    pub content_type: String,
    /// Content hash for integrity
    pub content_hash: Option<String>,
    /// Chunk information for large messages
    pub chunk_index: u32,
    pub total_chunks: u32,
    /// Privacy routing configuration
    pub mixnode_path: Vec<String>,
    pub encryption_layers: u8,
    /// Message metadata
    pub timestamp: u64,
    pub ttl_seconds: u32,
    pub priority: MessagePriority,
    /// QoS configuration
    pub bandwidth_tier: String,
    pub latency_target_ms: u32,
    pub reliability_level: String,
}

impl BetanetMessage {
    /// Create a new Betanet message
    pub fn new(
        sender: String,
        recipient: String,
        payload: bytes::Bytes,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            protocol: "htx/1.1".to_string(),
            sender,
            recipient,
            payload,
            content_type: "application/octet-stream".to_string(),
            content_hash: None,
            chunk_index: 0,
            total_chunks: 1,
            mixnode_path: Vec::new(),
            encryption_layers: 2,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            ttl_seconds: 300, // 5 minutes
            priority: MessagePriority::Normal,
            bandwidth_tier: "standard".to_string(),
            latency_target_ms: 1000,
            reliability_level: "best_effort".to_string(),
        }
    }

    /// Check if message has expired
    pub fn is_expired(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        now > self.timestamp + u64::from(self.ttl_seconds)
    }

    /// Calculate content hash
    pub fn compute_content_hash(&mut self) {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(&self.payload);
        self.content_hash = Some(hex::encode(hasher.finalize()));
    }
}

/// Peer information compatible with existing system
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BetanetPeer {
    /// Unique peer identifier
    pub peer_id: String,
    /// Network address
    pub multiaddr: String,
    /// Supported protocols
    pub protocols: Vec<String>,
    /// Peer capabilities
    pub capabilities: std::collections::HashSet<String>,
    /// Performance metrics
    pub latency_ms: f64,
    pub bandwidth_mbps: f64,
    pub reliability_score: f64,
    /// Security information
    pub is_mixnode: bool,
    pub trust_score: f64,
    pub encryption_support: Vec<String>,
    /// Status information
    pub last_seen: u64,
    pub connection_count: u32,
    pub geographic_region: Option<String>,
}

impl BetanetPeer {
    /// Check if peer is available
    pub fn is_available(&self, max_age_seconds: u64) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        (now - self.last_seen) < max_age_seconds
    }

    /// Check if peer supports protocol
    pub fn supports_protocol(&self, protocol: &str) -> bool {
        self.protocols.iter().any(|p| p == protocol)
    }
}

/// Initialize the Betanet client with tracing and metrics
pub async fn initialize() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    info!("Betanet Rust client initializing...");

    // Initialize metrics
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new()
        .build_recorder();
    metrics::set_global_recorder(recorder)?;

    info!("Betanet initialization complete");
    Ok(())
}

/// Create a new Betanet client with default configuration
pub async fn create_client() -> Result<Arc<RwLock<BetanetClient>>> {
    let config = BetanetConfig::default();
    let client = BetanetClient::new(config).await?;
    Ok(Arc::new(RwLock::new(client)))
}

/// Create a new Betanet client with custom configuration
pub async fn create_client_with_config(config: BetanetConfig) -> Result<Arc<RwLock<BetanetClient>>> {
    let client = BetanetClient::new(config).await?;
    Ok(Arc::new(RwLock::new(client)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_priority_conversion() {
        assert_eq!(MessagePriority::from(1u8), MessagePriority::Low);
        assert_eq!(MessagePriority::from(5u8), MessagePriority::Normal);
        assert_eq!(MessagePriority::from(8u8), MessagePriority::High);
        assert_eq!(MessagePriority::from(10u8), MessagePriority::Emergency);

        assert_eq!(u8::from(MessagePriority::Low), 3);
        assert_eq!(u8::from(MessagePriority::Normal), 5);
        assert_eq!(u8::from(MessagePriority::High), 7);
        assert_eq!(u8::from(MessagePriority::Emergency), 10);
    }

    #[test]
    fn test_message_creation() {
        let msg = BetanetMessage::new(
            "sender".to_string(),
            "recipient".to_string(),
            bytes::Bytes::from("test payload"),
        );

        assert_eq!(msg.sender, "sender");
        assert_eq!(msg.recipient, "recipient");
        assert_eq!(msg.payload, bytes::Bytes::from("test payload"));
        assert_eq!(msg.protocol, "htx/1.1");
        assert_eq!(msg.priority, MessagePriority::Normal);
        assert!(!msg.is_expired());
    }

    #[test]
    fn test_peer_availability() {
        let peer = BetanetPeer {
            peer_id: "test".to_string(),
            multiaddr: "/ip4/127.0.0.1/tcp/4001".to_string(),
            protocols: vec!["htx/1.1".to_string()],
            capabilities: std::collections::HashSet::new(),
            latency_ms: 100.0,
            bandwidth_mbps: 10.0,
            reliability_score: 0.9,
            is_mixnode: false,
            trust_score: 0.8,
            encryption_support: vec!["aes256".to_string()],
            last_seen: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            connection_count: 0,
            geographic_region: None,
        };

        assert!(peer.is_available(600)); // 10 minutes
        assert!(peer.supports_protocol("htx/1.1"));
        assert!(!peer.supports_protocol("unknown"));
    }
}
