//! Agent Fabric: Unified Agent Messaging API
//!
//! Provides a single API for agent communication using:
//! - RPC streams via betanet-htx for real-time messaging
//! - Bundle delivery via betanet-dtn for offline/async messaging
//! - MLS group cohorts for secure group communication

#![deny(clippy::all)]
#![allow(missing_docs)]

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;
use uuid::Uuid;

// Re-export core types from dependencies
pub use betanet_dtn::{BundleId, DtnError, DtnNode, EndpointId, SendBundleOptions};
pub use betanet_htx::{HtxConfig, HtxError, HtxSession, StreamId};

// Module declarations
pub mod api;
pub mod dtn_bridge;
pub mod groups;
pub mod rpc;

// Re-export from modules
pub use api::{AgentClient, AgentMessage, AgentResponse, AgentServer};
pub use dtn_bridge::{BundleMessage, DtnBridge};
pub use groups::{GroupConfig, GroupMessage, MlsGroup};
pub use rpc::{RpcClient, RpcServer, RpcTransport};

/// Agent identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId {
    pub id: String,
    pub node: String,
}

impl AgentId {
    pub fn new(id: impl Into<String>, node: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            node: node.into(),
        }
    }

    pub fn generate(node: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            node: node.into(),
        }
    }

    pub fn to_endpoint(&self) -> EndpointId {
        EndpointId::node(format!("{}/{}", self.node, self.id))
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.id, self.node)
    }
}

/// Message priority for routing decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MessagePriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl Default for MessagePriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Transport method for agent messaging
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Transport {
    /// Real-time RPC via HTX streams
    Rpc,
    /// Async bundles via DTN
    Bundle,
    /// Automatic selection based on connectivity
    Auto,
}

impl Default for Transport {
    fn default() -> Self {
        Self::Auto
    }
}

/// Message delivery options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryOptions {
    pub transport: Transport,
    pub priority: MessagePriority,
    pub timeout_ms: Option<u64>,
    pub retry_count: u32,
    pub require_receipt: bool,
}

impl Default for DeliveryOptions {
    fn default() -> Self {
        Self {
            transport: Transport::Auto,
            priority: MessagePriority::Normal,
            timeout_ms: Some(30000), // 30 seconds
            retry_count: 3,
            require_receipt: false,
        }
    }
}

/// Agent Fabric main interface
pub struct AgentFabric {
    node_id: AgentId,
    rpc_transport: Arc<RwLock<Option<RpcTransport>>>,
    dtn_bridge: Arc<RwLock<Option<DtnBridge>>>,
    mls_groups: Arc<RwLock<HashMap<String, MlsGroup>>>,
    clients: Arc<RwLock<HashMap<AgentId, Box<dyn AgentClient>>>>,
    servers: Arc<RwLock<HashMap<String, Box<dyn AgentServer>>>>,
}

impl AgentFabric {
    /// Create new Agent Fabric instance
    pub fn new(node_id: AgentId) -> Self {
        Self {
            node_id,
            rpc_transport: Arc::new(RwLock::new(None)),
            dtn_bridge: Arc::new(RwLock::new(None)),
            mls_groups: Arc::new(RwLock::new(HashMap::new())),
            clients: Arc::new(RwLock::new(HashMap::new())),
            servers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Initialize RPC transport
    pub async fn init_rpc(&self, config: HtxConfig) -> Result<()> {
        let transport = RpcTransport::new(config).await?;
        *self.rpc_transport.write().await = Some(transport);
        Ok(())
    }

    /// Initialize DTN bridge
    pub async fn init_dtn(&self, storage_path: impl AsRef<std::path::Path>) -> Result<()> {
        let bridge = DtnBridge::new(self.node_id.to_endpoint(), storage_path).await?;
        *self.dtn_bridge.write().await = Some(bridge);
        Ok(())
    }

    /// Create or join MLS group
    pub async fn join_group(&self, group_id: String, config: GroupConfig) -> Result<()> {
        let group = MlsGroup::new(group_id.clone(), config).await?;
        self.mls_groups.write().await.insert(group_id, group);
        Ok(())
    }

    /// Send message to agent
    pub async fn send_message(
        &self,
        to: AgentId,
        message: AgentMessage,
        options: DeliveryOptions,
    ) -> Result<Option<AgentResponse>> {
        match options.transport {
            Transport::Rpc => self.send_via_rpc(to, message, options).await,
            Transport::Bundle => self.send_via_bundle(to, message, options).await,
            Transport::Auto => {
                // Try RPC first, fallback to bundle if not available
                match self
                    .send_via_rpc(to.clone(), message.clone(), options.clone())
                    .await
                {
                    Ok(response) => Ok(response),
                    Err(AgentFabricError::TransportUnavailable) => {
                        self.send_via_bundle(to, message, options).await
                    }
                    Err(e) => Err(e),
                }
            }
        }
    }

    /// Send message to MLS group
    pub async fn send_to_group(&self, group_id: String, message: GroupMessage) -> Result<()> {
        let groups = self.mls_groups.read().await;
        if let Some(group) = groups.get(&group_id) {
            let _ = group.send_message(message).await?;
            Ok(())
        } else {
            Err(AgentFabricError::GroupNotFound(group_id))
        }
    }

    /// Register agent client
    pub async fn register_client(&self, agent_id: AgentId, client: Box<dyn AgentClient>) {
        self.clients.write().await.insert(agent_id, client);
    }

    /// Register agent server
    pub async fn register_server(&self, service_name: String, server: Box<dyn AgentServer>) {
        self.servers.write().await.insert(service_name, server);
    }

    /// Get node ID
    pub fn node_id(&self) -> &AgentId {
        &self.node_id
    }

    // Private methods

    async fn send_via_rpc(
        &self,
        to: AgentId,
        message: AgentMessage,
        _options: DeliveryOptions,
    ) -> Result<Option<AgentResponse>> {
        let transport = self.rpc_transport.read().await;
        if let Some(ref rpc) = *transport {
            let client = RpcClient::new(Arc::new(rpc.clone()));
            let response = client.call(to, message).await?;
            Ok(Some(response))
        } else {
            Err(AgentFabricError::TransportUnavailable)
        }
    }

    async fn send_via_bundle(
        &self,
        to: AgentId,
        message: AgentMessage,
        options: DeliveryOptions,
    ) -> Result<Option<AgentResponse>> {
        let bridge = self.dtn_bridge.read().await;
        if let Some(ref dtn) = *bridge {
            let bundle_message = BundleMessage {
                from: self.node_id.clone(),
                to: to.clone(),
                content: message,
                options: options.clone(),
            };
            dtn.send_bundle(to.to_endpoint(), bundle_message, options)
                .await?;
            Ok(None) // Bundle delivery is async, no immediate response
        } else {
            Err(AgentFabricError::TransportUnavailable)
        }
    }
}

/// Agent Fabric errors
#[derive(Debug, Error)]
pub enum AgentFabricError {
    #[error("HTX transport error: {0}")]
    HtxError(#[from] HtxError),

    #[error("DTN error: {0}")]
    DtnError(#[from] DtnError),

    #[error("MLS error: {0}")]
    MlsError(String),

    #[error("Transport not available")]
    TransportUnavailable,

    #[error("Agent not found: {0}")]
    AgentNotFound(AgentId),

    #[error("Group not found: {0}")]
    GroupNotFound(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Result type for Agent Fabric operations
pub type Result<T> = std::result::Result<T, AgentFabricError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_id_creation() {
        let agent_id = AgentId::new("twin-001", "node-alpha");
        assert_eq!(agent_id.id, "twin-001");
        assert_eq!(agent_id.node, "node-alpha");
        assert_eq!(agent_id.to_string(), "twin-001@node-alpha");
    }

    #[test]
    fn test_agent_id_generation() {
        let agent_id = AgentId::generate("node-beta");
        assert_eq!(agent_id.node, "node-beta");
        assert!(!agent_id.id.is_empty());
    }

    #[test]
    fn test_endpoint_conversion() {
        let agent_id = AgentId::new("tutor-002", "node-gamma");
        let endpoint = agent_id.to_endpoint();
        assert_eq!(endpoint.scheme, "dtn");
        assert_eq!(endpoint.specific_part, "node-gamma/tutor-002");
    }

    #[test]
    fn test_delivery_options_default() {
        let options = DeliveryOptions::default();
        assert_eq!(options.transport, Transport::Auto);
        assert_eq!(options.priority, MessagePriority::Normal);
        assert_eq!(options.timeout_ms, Some(30000));
        assert_eq!(options.retry_count, 3);
        assert!(!options.require_receipt);
    }
}
