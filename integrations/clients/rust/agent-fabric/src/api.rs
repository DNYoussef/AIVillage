//! Core API traits for Agent Fabric
//!
//! Defines the fundamental interfaces for agent communication.

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::{AgentFabricError, AgentId, Result};

/// Generic agent message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    pub id: String,
    pub message_type: String,
    pub payload: Bytes,
    pub metadata: std::collections::HashMap<String, String>,
}

impl AgentMessage {
    pub fn new(message_type: impl Into<String>, payload: Bytes) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            message_type: message_type.into(),
            payload,
            metadata: std::collections::HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Generic agent response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    pub request_id: String,
    pub success: bool,
    pub payload: Option<Bytes>,
    pub error: Option<String>,
    pub metadata: std::collections::HashMap<String, String>,
}

impl AgentResponse {
    pub fn success(request_id: impl Into<String>, payload: Bytes) -> Self {
        Self {
            request_id: request_id.into(),
            success: true,
            payload: Some(payload),
            error: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    pub fn error(request_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            success: false,
            payload: None,
            error: Some(error.into()),
            metadata: std::collections::HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Agent client trait for sending messages
#[async_trait]
pub trait AgentClient: Send + Sync {
    /// Send a message and wait for response
    async fn send_message(&self, message: AgentMessage) -> Result<AgentResponse>;

    /// Send a message without waiting for response
    async fn send_notification(&self, message: AgentMessage) -> Result<()>;

    /// Get the target agent ID
    fn target_agent(&self) -> &AgentId;
}

/// Agent server trait for handling incoming messages
#[async_trait]
pub trait AgentServer: Send + Sync {
    /// Handle an incoming message
    async fn handle_message(
        &self,
        from: AgentId,
        message: AgentMessage,
    ) -> Result<Option<AgentResponse>>;

    /// Get supported message types
    fn supported_message_types(&self) -> Vec<String>;

    /// Get server capabilities
    fn capabilities(&self) -> Vec<String> {
        vec![]
    }
}

/// Specific message types for common agent interactions

/// Twin agent message for twin-to-twin communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwinMessage {
    pub twin_id: String,
    pub action: TwinAction,
    pub data: Bytes,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TwinAction {
    Sync,
    Update,
    Query,
    Notify,
}

/// Tutor agent message for training coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutorMessage {
    pub session_id: String,
    pub action: TutorAction,
    pub model_data: Option<Bytes>,
    pub parameters: std::collections::HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TutorAction {
    StartTraining,
    UpdateModel,
    GetMetrics,
    StopTraining,
}

/// Agent message types
pub mod message_types {
    pub const PING: &str = "ping";
    pub const PONG: &str = "pong";
    pub const TWIN_MESSAGE: &str = "twin.message";
    pub const TUTOR_MESSAGE: &str = "tutor.message";
    pub const GROUP_JOIN: &str = "group.join";
    pub const GROUP_LEAVE: &str = "group.leave";
    pub const HEARTBEAT: &str = "heartbeat";
    pub const STATUS_REQUEST: &str = "status.request";
    pub const STATUS_RESPONSE: &str = "status.response";
}

/// Helper functions for common message patterns
impl AgentMessage {
    pub fn ping() -> Self {
        Self::new(message_types::PING, Bytes::from("ping"))
    }

    pub fn pong(request_id: &str) -> Self {
        Self::new(message_types::PONG, Bytes::from("pong")).with_metadata("request_id", request_id)
    }

    pub fn twin_message(twin_msg: TwinMessage) -> Result<Self> {
        let payload = serde_json::to_vec(&twin_msg)
            .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?;
        Ok(Self::new(message_types::TWIN_MESSAGE, Bytes::from(payload)))
    }

    pub fn tutor_message(tutor_msg: TutorMessage) -> Result<Self> {
        let payload = serde_json::to_vec(&tutor_msg)
            .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?;
        Ok(Self::new(
            message_types::TUTOR_MESSAGE,
            Bytes::from(payload),
        ))
    }

    pub fn status_request() -> Self {
        Self::new(message_types::STATUS_REQUEST, Bytes::new())
    }

    pub fn heartbeat(agent_id: &AgentId) -> Self {
        Self::new(message_types::HEARTBEAT, Bytes::from(agent_id.to_string()))
    }
}

impl AgentResponse {
    pub fn pong(request_id: &str) -> Self {
        Self::success(request_id, Bytes::from("pong"))
    }

    pub fn status(
        request_id: &str,
        status: std::collections::HashMap<String, String>,
    ) -> Result<Self> {
        let payload = serde_json::to_vec(&status)
            .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?;
        Ok(Self::success(request_id, Bytes::from(payload)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_message_creation() {
        let msg = AgentMessage::new("test", Bytes::from("hello"));
        assert_eq!(msg.message_type, "test");
        assert_eq!(msg.payload, Bytes::from("hello"));
        assert!(!msg.id.is_empty());
    }

    #[test]
    fn test_agent_message_with_metadata() {
        let msg = AgentMessage::new("test", Bytes::from("hello"))
            .with_metadata("key1", "value1")
            .with_metadata("key2", "value2");

        assert_eq!(msg.metadata.get("key1"), Some(&"value1".to_string()));
        assert_eq!(msg.metadata.get("key2"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_agent_response_success() {
        let resp = AgentResponse::success("req-123", Bytes::from("result"));
        assert_eq!(resp.request_id, "req-123");
        assert!(resp.success);
        assert_eq!(resp.payload, Some(Bytes::from("result")));
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_agent_response_error() {
        let resp = AgentResponse::error("req-456", "Something went wrong");
        assert_eq!(resp.request_id, "req-456");
        assert!(!resp.success);
        assert!(resp.payload.is_none());
        assert_eq!(resp.error, Some("Something went wrong".to_string()));
    }

    #[test]
    fn test_ping_pong_messages() {
        let ping = AgentMessage::ping();
        assert_eq!(ping.message_type, message_types::PING);
        assert_eq!(ping.payload, Bytes::from("ping"));

        let pong = AgentMessage::pong("ping-123");
        assert_eq!(pong.message_type, message_types::PONG);
        assert_eq!(
            pong.metadata.get("request_id"),
            Some(&"ping-123".to_string())
        );
    }

    #[test]
    fn test_twin_message_serialization() {
        let twin_msg = TwinMessage {
            twin_id: "twin-001".to_string(),
            action: TwinAction::Sync,
            data: Bytes::from("sync data"),
        };

        let agent_msg = AgentMessage::twin_message(twin_msg).unwrap();
        assert_eq!(agent_msg.message_type, message_types::TWIN_MESSAGE);
    }

    #[test]
    fn test_tutor_message_serialization() {
        let mut params = std::collections::HashMap::new();
        params.insert("learning_rate".to_string(), 0.001);
        params.insert("batch_size".to_string(), 32.0);

        let tutor_msg = TutorMessage {
            session_id: "session-001".to_string(),
            action: TutorAction::StartTraining,
            model_data: Some(Bytes::from("model weights")),
            parameters: params,
        };

        let agent_msg = AgentMessage::tutor_message(tutor_msg).unwrap();
        assert_eq!(agent_msg.message_type, message_types::TUTOR_MESSAGE);
    }
}
