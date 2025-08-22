//! Agent Fabric integration for twin vault operations
//!
//! Provides seamless integration between twin vault state management
//! and Agent Fabric communication, including consent handling and
//! distributed synchronization.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{oneshot, RwLock};
use tracing::{debug, error, info, warn};

use agent_fabric::{
    AgentFabric, AgentFabricError, AgentId, AgentMessage, AgentResponse, AgentServer,
    DeliveryOptions, MessagePriority, Transport,
};

use crate::{
    crdt::CrdtState,
    receipts::{Receipt, ReceiptSigner, ReceiptVerifier},
    vault::{TwinVault, VaultConfig},
    TwinId, TwinOperation, TwinPreferences,
};

/// Twin operation request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwinOperationRequest {
    pub twin_id: TwinId,
    pub operation: TwinOperation,
    pub require_receipt: bool,
    pub consent_timeout_ms: Option<u64>,
}

/// Twin operation response message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwinOperationResponse {
    pub success: bool,
    pub result: Option<Bytes>,
    pub receipt: Option<Receipt>,
    pub error: Option<String>,
}

/// Twin sync request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwinSyncRequest {
    pub twin_id: TwinId,
    pub state: CrdtState,
    pub force_merge: bool,
}

/// Twin sync response message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwinSyncResponse {
    pub success: bool,
    pub merged_state: Option<CrdtState>,
    pub error: Option<String>,
}

/// Consent request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRequest {
    pub twin_id: TwinId,
    pub requester: AgentId,
    pub operation: TwinOperation,
    pub timeout_ms: u64,
}

/// Consent response message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentResponse {
    pub twin_id: TwinId,
    pub granted: bool,
    pub reason: Option<String>,
    pub valid_until: Option<u64>,
}

/// Twin preference update message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceUpdateRequest {
    pub twin_id: TwinId,
    pub preferences: TwinPreferences,
}

/// Message types for twin vault operations
pub mod message_types {
    pub const TWIN_OPERATION: &str = "twin_operation";
    pub const TWIN_SYNC: &str = "twin_sync";
    pub const CONSENT_REQUEST: &str = "consent_request";
    pub const CONSENT_RESPONSE: &str = "consent_response";
    pub const PREFERENCE_UPDATE: &str = "preference_update";
    pub const TWIN_STATUS: &str = "twin_status";
}

/// Agent Fabric server for handling twin vault operations
pub struct TwinVaultServer {
    twin_manager: Arc<crate::TwinManager>,
    pending_consents: Arc<RwLock<HashMap<String, oneshot::Sender<ConsentResponse>>>>,
}

impl TwinVaultServer {
    pub fn new(twin_manager: Arc<crate::TwinManager>) -> Self {
        Self {
            twin_manager,
            pending_consents: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn handle_twin_operation(
        &self,
        from: AgentId,
        request: TwinOperationRequest,
    ) -> Result<TwinOperationResponse, crate::TwinVaultError> {
        debug!(
            "Handling twin operation from {}: {:?}",
            from, request.operation
        );

        // Check if consent is required
        let needs_consent = !self
            .twin_manager
            .check_consent(&request.twin_id, &from, &request.operation)
            .await;

        if needs_consent {
            // Request consent
            let consent_granted = self
                .request_consent(
                    &request.twin_id,
                    from.clone(),
                    &request.operation,
                    request.consent_timeout_ms.unwrap_or(30000),
                )
                .await?;

            if !consent_granted {
                return Ok(TwinOperationResponse {
                    success: false,
                    result: None,
                    receipt: None,
                    error: Some("Consent denied".to_string()),
                });
            }
        }

        // Perform operation
        match self
            .twin_manager
            .perform_operation(request.twin_id, request.operation, from)
            .await
        {
            Ok((result, receipt)) => Ok(TwinOperationResponse {
                success: true,
                result,
                receipt: if request.require_receipt {
                    Some(receipt)
                } else {
                    None
                },
                error: None,
            }),
            Err(e) => Ok(TwinOperationResponse {
                success: false,
                result: None,
                receipt: None,
                error: Some(e.to_string()),
            }),
        }
    }

    async fn handle_twin_sync(
        &self,
        from: AgentId,
        request: TwinSyncRequest,
    ) -> Result<TwinSyncResponse, crate::TwinVaultError> {
        debug!("Handling twin sync from {}: {}", from, request.twin_id);

        // Check sync permission
        let preferences = self.twin_manager.get_preferences(&request.twin_id).await;
        if !preferences.allow_sync && !preferences.trusted_agents.contains(&from) {
            return Ok(TwinSyncResponse {
                success: false,
                merged_state: None,
                error: Some("Sync not allowed".to_string()),
            });
        }

        // Get vault
        let vault = match self
            .twin_manager
            .get_twin(request.twin_id.clone(), VaultConfig::default())
            .await
        {
            Ok(vault) => vault,
            Err(e) => {
                return Ok(TwinSyncResponse {
                    success: false,
                    merged_state: None,
                    error: Some(e.to_string()),
                })
            }
        };

        // Merge state
        match vault.merge_state(request.state).await {
            Ok(()) => {
                let merged_state = vault
                    .get_state()
                    .await
                    .map_err(crate::TwinVaultError::VaultError)?;
                Ok(TwinSyncResponse {
                    success: true,
                    merged_state: Some(merged_state),
                    error: None,
                })
            }
            Err(e) => Ok(TwinSyncResponse {
                success: false,
                merged_state: None,
                error: Some(e.to_string()),
            }),
        }
    }

    async fn handle_consent_response(
        &self,
        response: ConsentResponse,
    ) -> Result<(), crate::TwinVaultError> {
        let consent_key = format!(
            "{}:{}",
            response.twin_id,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis()
        );

        let mut pending = self.pending_consents.write().await;
        if let Some(sender) = pending.remove(&consent_key) {
            let _ = sender.send(response);
        }

        Ok(())
    }

    async fn handle_preference_update(
        &self,
        from: AgentId,
        request: PreferenceUpdateRequest,
    ) -> Result<(), crate::TwinVaultError> {
        debug!(
            "Updating preferences for twin {} from {}",
            request.twin_id, from
        );

        // Only allow twin owner to update preferences
        if request.twin_id.agent_id != from {
            warn!(
                "Unauthorized preference update attempt by {} for twin {}",
                from, request.twin_id
            );
            return Err(crate::TwinVaultError::ConsentDenied {
                twin_id: request.twin_id,
                requester: from,
                operation: "preference_update".to_string(),
            });
        }

        self.twin_manager
            .set_preferences(request.twin_id, request.preferences)
            .await;
        Ok(())
    }

    async fn request_consent(
        &self,
        twin_id: &TwinId,
        requester: AgentId,
        operation: &TwinOperation,
        timeout_ms: u64,
    ) -> Result<bool, crate::TwinVaultError> {
        debug!("Requesting consent for twin {} from {}", twin_id, requester);

        // Create consent request
        let consent_request = ConsentRequest {
            twin_id: twin_id.clone(),
            requester: requester.clone(),
            operation: operation.clone(),
            timeout_ms,
        };

        // Send consent request to twin owner
        let consent_message = AgentMessage::new(
            message_types::CONSENT_REQUEST,
            serde_json::to_vec(&consent_request)
                .map_err(|e| crate::TwinVaultError::SerializationError(e.to_string()))?
                .into(),
        );

        let delivery_options = DeliveryOptions {
            transport: Transport::Auto,
            priority: MessagePriority::High,
            timeout_ms: Some(timeout_ms),
            retry_count: 2,
            require_receipt: false,
        };

        match self
            .twin_manager
            .agent_fabric
            .send_message(twin_id.agent_id.clone(), consent_message, delivery_options)
            .await
        {
            Ok(Some(response)) => {
                if response.success {
                    if let Some(ref payload) = response.payload {
                        if let Ok(consent_response) =
                            serde_json::from_slice::<ConsentResponse>(payload)
                        {
                            return Ok(consent_response.granted);
                        }
                    }
                }
                Ok(false)
            }
            Ok(None) => Ok(false), // No response means denied
            Err(e) => {
                warn!("Failed to request consent: {}", e);
                Ok(false)
            }
        }
    }
}

#[async_trait]
impl AgentServer for TwinVaultServer {
    async fn handle_message(
        &self,
        from: AgentId,
        message: AgentMessage,
    ) -> Result<Option<AgentResponse>, AgentFabricError> {
        match message.message_type.as_str() {
            message_types::TWIN_OPERATION => {
                let request: TwinOperationRequest = serde_json::from_slice(&message.payload)
                    .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?;
                let response = self
                    .handle_twin_operation(from, request)
                    .await
                    .map_err(|e| AgentFabricError::NetworkError(e.to_string()))?;

                let response_payload = serde_json::to_vec(&response)
                    .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?;
                Ok(Some(AgentResponse::success(
                    &message.id,
                    Bytes::from(response_payload),
                )))
            }
            message_types::TWIN_SYNC => {
                let request: TwinSyncRequest = serde_json::from_slice(&message.payload)
                    .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?;
                let response = self
                    .handle_twin_sync(from, request)
                    .await
                    .map_err(|e| AgentFabricError::NetworkError(e.to_string()))?;

                let response_payload = serde_json::to_vec(&response)
                    .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?;
                Ok(Some(AgentResponse::success(
                    &message.id,
                    Bytes::from(response_payload),
                )))
            }
            message_types::CONSENT_RESPONSE => {
                let response: ConsentResponse = serde_json::from_slice(&message.payload)
                    .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?;
                self.handle_consent_response(response)
                    .await
                    .map_err(|e| AgentFabricError::NetworkError(e.to_string()))?;
                Ok(Some(AgentResponse::success(&message.id, Bytes::new())))
            }
            message_types::PREFERENCE_UPDATE => {
                let request: PreferenceUpdateRequest = serde_json::from_slice(&message.payload)
                    .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?;
                self.handle_preference_update(from, request)
                    .await
                    .map_err(|e| AgentFabricError::NetworkError(e.to_string()))?;
                Ok(Some(AgentResponse::success(&message.id, Bytes::new())))
            }
            _ => {
                warn!("Unknown message type: {}", message.message_type);
                Ok(Some(AgentResponse::error(
                    &message.id,
                    "Unknown message type",
                )))
            }
        }
    }

    fn supported_message_types(&self) -> Vec<String> {
        vec![
            message_types::TWIN_OPERATION.to_string(),
            message_types::TWIN_SYNC.to_string(),
            message_types::CONSENT_REQUEST.to_string(),
            message_types::CONSENT_RESPONSE.to_string(),
            message_types::PREFERENCE_UPDATE.to_string(),
            message_types::TWIN_STATUS.to_string(),
        ]
    }
}

/// Agent Fabric client for twin vault operations
pub struct TwinVaultClient {
    agent_fabric: Arc<AgentFabric>,
    receipt_verifier: Arc<ReceiptVerifier>,
}

impl TwinVaultClient {
    pub fn new(agent_fabric: Arc<AgentFabric>, receipt_verifier: ReceiptVerifier) -> Self {
        Self {
            agent_fabric,
            receipt_verifier: Arc::new(receipt_verifier),
        }
    }

    /// Perform remote twin operation
    pub async fn perform_operation(
        &self,
        target_agent: AgentId,
        twin_id: TwinId,
        operation: TwinOperation,
        require_receipt: bool,
    ) -> Result<TwinOperationResponse, IntegrationError> {
        let request = TwinOperationRequest {
            twin_id,
            operation,
            require_receipt,
            consent_timeout_ms: Some(30000),
        };

        let message = AgentMessage::new(
            message_types::TWIN_OPERATION,
            serde_json::to_vec(&request)
                .map_err(|e| IntegrationError::SerializationError(e.to_string()))?
                .into(),
        );

        let delivery_options = DeliveryOptions {
            transport: Transport::Auto,
            priority: MessagePriority::Normal,
            timeout_ms: Some(60000),
            retry_count: 3,
            require_receipt: false,
        };

        match self
            .agent_fabric
            .send_message(target_agent.clone(), message, delivery_options)
            .await
        {
            Ok(Some(response)) => {
                if response.success {
                    let twin_response: TwinOperationResponse =
                        if let Some(ref payload) = response.payload {
                            serde_json::from_slice(payload)
                                .map_err(|e| IntegrationError::SerializationError(e.to_string()))?
                        } else {
                            return Err(IntegrationError::SerializationError(
                                "Empty payload".to_string(),
                            ));
                        };

                    // Verify receipt if present
                    if let Some(ref receipt) = twin_response.receipt {
                        let verification = self
                            .receipt_verifier
                            .verify_receipt(receipt)
                            .map_err(|e| IntegrationError::ReceiptError(e.to_string()))?;

                        if !verification.is_trusted_and_valid() {
                            warn!("Invalid receipt from {}: {:?}", target_agent, verification);
                        }
                    }

                    Ok(twin_response)
                } else {
                    Err(IntegrationError::OperationFailed(
                        response
                            .error
                            .unwrap_or_else(|| "Unknown error".to_string()),
                    ))
                }
            }
            Ok(None) => Err(IntegrationError::NoResponse),
            Err(e) => Err(IntegrationError::CommunicationError(e.to_string())),
        }
    }

    /// Sync twin state with remote agent
    pub async fn sync_twin(
        &self,
        target_agent: AgentId,
        twin_id: TwinId,
        local_state: CrdtState,
    ) -> Result<Option<CrdtState>, IntegrationError> {
        let request = TwinSyncRequest {
            twin_id,
            state: local_state,
            force_merge: false,
        };

        let message = AgentMessage::new(
            message_types::TWIN_SYNC,
            serde_json::to_vec(&request)
                .map_err(|e| IntegrationError::SerializationError(e.to_string()))?
                .into(),
        );

        let delivery_options = DeliveryOptions {
            transport: Transport::Auto,
            priority: MessagePriority::Normal,
            timeout_ms: Some(30000),
            retry_count: 2,
            require_receipt: false,
        };

        match self
            .agent_fabric
            .send_message(target_agent, message, delivery_options)
            .await
        {
            Ok(Some(response)) => {
                if response.success {
                    let sync_response: TwinSyncResponse =
                        if let Some(ref payload) = response.payload {
                            serde_json::from_slice(payload)
                                .map_err(|e| IntegrationError::SerializationError(e.to_string()))?
                        } else {
                            return Err(IntegrationError::SerializationError(
                                "Empty payload".to_string(),
                            ));
                        };

                    if sync_response.success {
                        Ok(sync_response.merged_state)
                    } else {
                        Err(IntegrationError::SyncFailed(
                            sync_response
                                .error
                                .unwrap_or_else(|| "Unknown sync error".to_string()),
                        ))
                    }
                } else {
                    Err(IntegrationError::SyncFailed(
                        response
                            .error
                            .unwrap_or_else(|| "Sync request failed".to_string()),
                    ))
                }
            }
            Ok(None) => Ok(None), // No response for sync
            Err(e) => Err(IntegrationError::CommunicationError(e.to_string())),
        }
    }

    /// Update twin preferences
    pub async fn update_preferences(
        &self,
        target_agent: AgentId,
        twin_id: TwinId,
        preferences: TwinPreferences,
    ) -> Result<(), IntegrationError> {
        let request = PreferenceUpdateRequest {
            twin_id,
            preferences,
        };

        let message = AgentMessage::new(
            message_types::PREFERENCE_UPDATE,
            serde_json::to_vec(&request)
                .map_err(|e| IntegrationError::SerializationError(e.to_string()))?
                .into(),
        );

        let delivery_options = DeliveryOptions {
            transport: Transport::Auto,
            priority: MessagePriority::Normal,
            timeout_ms: Some(10000),
            retry_count: 1,
            require_receipt: false,
        };

        match self
            .agent_fabric
            .send_message(target_agent, message, delivery_options)
            .await
        {
            Ok(Some(response)) => {
                if response.success {
                    Ok(())
                } else {
                    Err(IntegrationError::OperationFailed(
                        response
                            .error
                            .unwrap_or_else(|| "Preference update failed".to_string()),
                    ))
                }
            }
            Ok(None) => Ok(()), // Notification, no response expected
            Err(e) => Err(IntegrationError::CommunicationError(e.to_string())),
        }
    }
}

/// Integration-related errors
#[derive(Debug, Error)]
pub enum IntegrationError {
    #[error("Communication error: {0}")]
    CommunicationError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Receipt error: {0}")]
    ReceiptError(String),

    #[error("Operation failed: {0}")]
    OperationFailed(String),

    #[error("Sync failed: {0}")]
    SyncFailed(String),

    #[error("No response received")]
    NoResponse,

    #[error("Consent denied")]
    ConsentDenied,

    #[error("Timeout")]
    Timeout,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_twin_operation_request_serialization() {
        let twin_id = TwinId::new(AgentId::new("test-agent", "test-node"), "test-twin");

        let operation = TwinOperation::Write {
            key: "test_key".to_string(),
            value: Bytes::from("test_value"),
            timestamp: 12345,
        };

        let request = TwinOperationRequest {
            twin_id,
            operation,
            require_receipt: true,
            consent_timeout_ms: Some(30000),
        };

        // Test serialization
        let serialized = serde_json::to_vec(&request).unwrap();
        let deserialized: TwinOperationRequest = serde_json::from_slice(&serialized).unwrap();

        assert_eq!(
            request.twin_id.to_string(),
            deserialized.twin_id.to_string()
        );
        assert_eq!(request.require_receipt, deserialized.require_receipt);
    }

    #[tokio::test]
    async fn test_consent_request_response() {
        let twin_id = TwinId::new(AgentId::new("test-agent", "test-node"), "test-twin");

        let requester = AgentId::new("requester", "requester-node");

        let operation = TwinOperation::Read {
            key: "test_key".to_string(),
            timestamp: 12345,
        };

        let consent_request = ConsentRequest {
            twin_id: twin_id.clone(),
            requester: requester.clone(),
            operation,
            timeout_ms: 30000,
        };

        let consent_response = ConsentResponse {
            twin_id,
            granted: true,
            reason: None,
            valid_until: None,
        };

        // Test serialization
        let req_serialized = serde_json::to_vec(&consent_request).unwrap();
        let req_deserialized: ConsentRequest = serde_json::from_slice(&req_serialized).unwrap();

        let resp_serialized = serde_json::to_vec(&consent_response).unwrap();
        let resp_deserialized: ConsentResponse = serde_json::from_slice(&resp_serialized).unwrap();

        assert_eq!(
            consent_request.twin_id.to_string(),
            req_deserialized.twin_id.to_string()
        );
        assert_eq!(consent_response.granted, resp_deserialized.granted);
    }

    #[test]
    fn test_message_types() {
        // Ensure all message types are unique
        let types = vec![
            message_types::TWIN_OPERATION,
            message_types::TWIN_SYNC,
            message_types::CONSENT_REQUEST,
            message_types::CONSENT_RESPONSE,
            message_types::PREFERENCE_UPDATE,
            message_types::TWIN_STATUS,
        ];

        let mut unique_types = types.clone();
        unique_types.sort();
        unique_types.dedup();

        assert_eq!(
            types.len(),
            unique_types.len(),
            "Message types must be unique"
        );
    }
}
