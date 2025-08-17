//! Twin Vault: Conflict-free twin state with receipts for every operation
//!
//! Provides CRDT-based state management with cryptographic receipts and integration
//! with Agent Fabric for secure communication and consent management.

#![deny(clippy::all)]
#![allow(missing_docs)]

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;

// Re-export core types from Agent Fabric
pub use agent_fabric::{AgentFabric, AgentId, AgentMessage, AgentResponse};

// Module declarations
pub mod crdt;
pub mod integration;
pub mod receipts;
pub mod vault;

// Re-export from modules
pub use crdt::{CrdtState, GCounter, LwwMap, SignedOperation};
pub use integration::{message_types, TwinVaultClient, TwinVaultServer};
pub use receipts::{Receipt, ReceiptSigner, ReceiptVerifier};
pub use vault::{EncryptionKey, TwinVault, VaultConfig};

/// Twin identifier for state management
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TwinId {
    pub agent_id: AgentId,
    pub twin_name: String,
}

impl TwinId {
    pub fn new(agent_id: AgentId, twin_name: impl Into<String>) -> Self {
        Self {
            agent_id,
            twin_name: twin_name.into(),
        }
    }

    pub fn generate(agent_id: AgentId) -> Self {
        Self {
            agent_id,
            twin_name: uuid::Uuid::new_v4().to_string(),
        }
    }
}

impl std::fmt::Display for TwinId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.agent_id, self.twin_name)
    }
}

/// Twin state operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TwinOperation {
    /// Read operation
    Read { key: String, timestamp: u64 },
    /// Write operation
    Write {
        key: String,
        value: Bytes,
        timestamp: u64,
    },
    /// Delete operation
    Delete { key: String, timestamp: u64 },
    /// Increment counter
    Increment {
        counter_id: String,
        amount: u64,
        actor_id: String,
        timestamp: u64,
    },
}

impl TwinOperation {
    pub fn timestamp(&self) -> u64 {
        match self {
            TwinOperation::Read { timestamp, .. } => *timestamp,
            TwinOperation::Write { timestamp, .. } => *timestamp,
            TwinOperation::Delete { timestamp, .. } => *timestamp,
            TwinOperation::Increment { timestamp, .. } => *timestamp,
        }
    }

    pub fn key(&self) -> Option<&str> {
        match self {
            TwinOperation::Read { key, .. } => Some(key),
            TwinOperation::Write { key, .. } => Some(key),
            TwinOperation::Delete { key, .. } => Some(key),
            TwinOperation::Increment { .. } => None,
        }
    }
}

/// Twin preferences for consent management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwinPreferences {
    pub allow_read: bool,
    pub allow_write: bool,
    pub allow_sync: bool,
    pub trusted_agents: Vec<AgentId>,
    pub blocked_agents: Vec<AgentId>,
    pub consent_timeout_ms: u64,
}

impl Default for TwinPreferences {
    fn default() -> Self {
        Self {
            allow_read: false, // Secure by default
            allow_write: false,
            allow_sync: false,
            trusted_agents: Vec::new(),
            blocked_agents: Vec::new(),
            consent_timeout_ms: 60000, // 1 minute
        }
    }
}

/// Twin state manager integrating with Agent Fabric
#[derive(Clone)]
pub struct TwinManager {
    agent_fabric: Arc<AgentFabric>,
    vaults: Arc<RwLock<HashMap<TwinId, Arc<TwinVault>>>>,
    preferences: Arc<RwLock<HashMap<TwinId, TwinPreferences>>>,
    receipt_signer: Arc<ReceiptSigner>,
    receipt_verifier: Arc<ReceiptVerifier>,
}

impl TwinManager {
    /// Create new twin manager
    pub async fn new(
        agent_fabric: Arc<AgentFabric>,
        receipt_signer: ReceiptSigner,
        receipt_verifier: ReceiptVerifier,
    ) -> Result<Self> {
        Ok(Self {
            agent_fabric,
            vaults: Arc::new(RwLock::new(HashMap::new())),
            preferences: Arc::new(RwLock::new(HashMap::new())),
            receipt_signer: Arc::new(receipt_signer),
            receipt_verifier: Arc::new(receipt_verifier),
        })
    }

    /// Create or get twin vault
    pub async fn get_twin(&self, twin_id: TwinId, config: VaultConfig) -> Result<Arc<TwinVault>> {
        let mut vaults = self.vaults.write().await;
        if let Some(vault) = vaults.get(&twin_id) {
            Ok(Arc::clone(vault))
        } else {
            let vault = Arc::new(TwinVault::new(twin_id.clone(), config).await?);
            vaults.insert(twin_id, Arc::clone(&vault));
            Ok(vault)
        }
    }

    /// Set twin preferences for consent management
    pub async fn set_preferences(&self, twin_id: TwinId, preferences: TwinPreferences) {
        let mut prefs = self.preferences.write().await;
        prefs.insert(twin_id, preferences);
    }

    /// Get twin preferences
    pub async fn get_preferences(&self, twin_id: &TwinId) -> TwinPreferences {
        let prefs = self.preferences.read().await;
        prefs.get(twin_id).cloned().unwrap_or_default()
    }

    /// Check if operation is allowed based on preferences
    pub async fn check_consent(
        &self,
        twin_id: &TwinId,
        requester: &AgentId,
        operation: &TwinOperation,
    ) -> bool {
        let preferences = self.get_preferences(twin_id).await;

        // Check if requester is blocked
        if preferences.blocked_agents.contains(requester) {
            return false;
        }

        // Check if requester is trusted
        if preferences.trusted_agents.contains(requester) {
            return true;
        }

        // Check operation-specific permissions
        match operation {
            TwinOperation::Read { .. } => preferences.allow_read,
            TwinOperation::Write { .. }
            | TwinOperation::Delete { .. }
            | TwinOperation::Increment { .. } => preferences.allow_write,
        }
    }

    /// Perform twin operation with receipt generation
    pub async fn perform_operation(
        &self,
        twin_id: TwinId,
        operation: TwinOperation,
        requester: AgentId,
    ) -> Result<(Option<Bytes>, Receipt)> {
        // Check consent
        if !self.check_consent(&twin_id, &requester, &operation).await {
            return Err(TwinVaultError::ConsentDenied {
                twin_id,
                requester,
                operation: format!("{:?}", operation),
            });
        }

        // Get vault
        let vault = {
            let vaults = self.vaults.read().await;
            vaults
                .get(&twin_id)
                .cloned()
                .ok_or_else(|| TwinVaultError::TwinNotFound(twin_id.clone()))?
        };

        // Perform operation
        let result = match &operation {
            TwinOperation::Read { key, .. } => vault.get(key).await.map(|v| v.map(|b| b.into())),
            TwinOperation::Write {
                key,
                value,
                timestamp,
            } => {
                vault.set(key.clone(), value.clone(), *timestamp).await?;
                Ok(None)
            }
            TwinOperation::Delete { key, timestamp } => {
                vault.delete(key, *timestamp).await?;
                Ok(None)
            }
            TwinOperation::Increment {
                counter_id,
                amount,
                actor_id,
                timestamp,
            } => {
                vault
                    .increment_counter(counter_id, *amount, actor_id, *timestamp)
                    .await?;
                Ok(None)
            }
        }?;

        // Generate receipt
        let receipt = self
            .receipt_signer
            .sign_operation(&twin_id, &operation, &requester, result.is_some())
            .await?;

        Ok((result, receipt))
    }

    /// Sync twin state with remote agent
    pub async fn sync_with_agent(&self, twin_id: TwinId, remote_agent: AgentId) -> Result<()> {
        // Check sync permission
        let preferences = self.get_preferences(&twin_id).await;
        if !preferences.allow_sync && !preferences.trusted_agents.contains(&remote_agent) {
            return Err(TwinVaultError::ConsentDenied {
                twin_id,
                requester: remote_agent,
                operation: "sync".to_string(),
            });
        }

        // Get vault
        let vault = {
            let vaults = self.vaults.read().await;
            vaults
                .get(&twin_id)
                .cloned()
                .ok_or_else(|| TwinVaultError::TwinNotFound(twin_id.clone()))?
        };

        // Create sync message
        let state = vault.get_state().await?;
        let sync_message = AgentMessage::new(
            "twin_sync",
            serde_json::to_vec(&state)
                .map_err(|e| TwinVaultError::SerializationError(e.to_string()))?
                .into(),
        );

        // Send sync via Agent Fabric
        self.agent_fabric
            .send_message(
                remote_agent,
                sync_message,
                agent_fabric::DeliveryOptions::default(),
            )
            .await
            .map_err(|e| TwinVaultError::AgentFabricError(e.to_string()))?;

        Ok(())
    }
}

/// Twin Vault errors
#[derive(Debug, Error)]
pub enum TwinVaultError {
    #[error("CRDT error: {0}")]
    CrdtError(#[from] crdt::CrdtError),

    #[error("Vault error: {0}")]
    VaultError(#[from] vault::VaultError),

    #[error("Receipt error: {0}")]
    ReceiptError(#[from] receipts::ReceiptError),

    #[error("Twin not found: {0}")]
    TwinNotFound(TwinId),

    #[error("Consent denied for twin {twin_id} by agent {requester} for operation {operation}")]
    ConsentDenied {
        twin_id: TwinId,
        requester: AgentId,
        operation: String,
    },

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Agent Fabric error: {0}")]
    AgentFabricError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for Twin Vault operations
pub type Result<T> = std::result::Result<T, TwinVaultError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twin_id_creation() {
        let agent_id = AgentId::new("agent-001", "node-alpha");
        let twin_id = TwinId::new(agent_id.clone(), "preferences");

        assert_eq!(twin_id.agent_id, agent_id);
        assert_eq!(twin_id.twin_name, "preferences");
        assert_eq!(twin_id.to_string(), "agent-001@node-alpha:preferences");
    }

    #[test]
    fn test_twin_operation_timestamp() {
        let op = TwinOperation::Write {
            key: "test".to_string(),
            value: Bytes::from("value"),
            timestamp: 12345,
        };

        assert_eq!(op.timestamp(), 12345);
        assert_eq!(op.key(), Some("test"));
    }

    #[test]
    fn test_twin_preferences_default() {
        let prefs = TwinPreferences::default();
        assert!(!prefs.allow_read);
        assert!(!prefs.allow_write);
        assert!(!prefs.allow_sync);
        assert_eq!(prefs.consent_timeout_ms, 60000);
    }
}
