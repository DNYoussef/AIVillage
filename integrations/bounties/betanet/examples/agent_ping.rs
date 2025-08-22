//! Agent Ping Example: Twin ↔ Tutor Communication
//!
//! Demonstrates the unified agent messaging API with:
//! - Real-time RPC communication when online
//! - Automatic fallback to DTN bundles when offline
//! - Twin and Tutor agent interactions
//! - MLS group coordination

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use tempfile::TempDir;
use tokio::time::{sleep, timeout};
use tracing::{info, warn, error};

use agent_fabric::{
    AgentFabric, AgentId, AgentMessage, AgentResponse, AgentClient, AgentServer,
    HtxConfig, DeliveryOptions, Transport, MessagePriority,
    api::{TwinMessage, TutorMessage, TwinAction, TutorAction, message_types},
    GroupConfig, GroupMessage, GroupMessageType, TrainingAction,
};

/// Twin agent implementation
struct TwinAgent {
    agent_id: AgentId,
    peer_updates: std::sync::atomic::AtomicU64,
}

impl TwinAgent {
    fn new(agent_id: AgentId) -> Self {
        Self {
            agent_id,
            peer_updates: std::sync::atomic::AtomicU64::new(0),
        }
    }

    fn get_update_count(&self) -> u64 {
        self.peer_updates.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[async_trait::async_trait]
impl AgentServer for TwinAgent {
    async fn handle_message(
        &self,
        from: AgentId,
        message: AgentMessage,
    ) -> agent_fabric::Result<Option<AgentResponse>> {
        match message.message_type.as_str() {
            message_types::PING => {
                info!("Twin {} received ping from {}", self.agent_id, from);
                Ok(Some(AgentResponse::pong(&message.id)))
            }
            message_types::TWIN_MESSAGE => {
                let twin_msg: TwinMessage = serde_json::from_slice(&message.payload)
                    .map_err(|e| agent_fabric::AgentFabricError::SerializationError(e.to_string()))?;

                info!("Twin {} received {} from {}", self.agent_id, twin_msg.action, from);

                match twin_msg.action {
                    TwinAction::Sync => {
                        self.peer_updates.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let response_data = format!("Sync completed. Updates: {}", self.get_update_count());
                        Ok(Some(AgentResponse::success(&message.id, Bytes::from(response_data))))
                    }
                    TwinAction::Update => {
                        self.peer_updates.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        Ok(Some(AgentResponse::success(&message.id, Bytes::from("Update received"))))
                    }
                    TwinAction::Query => {
                        let status = format!("Twin status: {} updates received", self.get_update_count());
                        Ok(Some(AgentResponse::success(&message.id, Bytes::from(status))))
                    }
                    TwinAction::Notify => {
                        info!("Twin {} notification: {}", self.agent_id, String::from_utf8_lossy(&twin_msg.data));
                        Ok(None) // No response for notifications
                    }
                }
            }
            _ => {
                warn!("Twin {} received unknown message type: {}", self.agent_id, message.message_type);
                Ok(Some(AgentResponse::error(&message.id, "Unknown message type")))
            }
        }
    }

    fn supported_message_types(&self) -> Vec<String> {
        vec![
            message_types::PING.to_string(),
            message_types::TWIN_MESSAGE.to_string(),
        ]
    }
}

impl std::fmt::Display for TwinAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TwinAction::Sync => write!(f, "Sync"),
            TwinAction::Update => write!(f, "Update"),
            TwinAction::Query => write!(f, "Query"),
            TwinAction::Notify => write!(f, "Notify"),
        }
    }
}

/// Tutor agent implementation
struct TutorAgent {
    agent_id: AgentId,
    training_sessions: std::sync::atomic::AtomicU64,
}

impl TutorAgent {
    fn new(agent_id: AgentId) -> Self {
        Self {
            agent_id,
            training_sessions: std::sync::atomic::AtomicU64::new(0),
        }
    }

    fn get_session_count(&self) -> u64 {
        self.training_sessions.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[async_trait::async_trait]
impl AgentServer for TutorAgent {
    async fn handle_message(
        &self,
        from: AgentId,
        message: AgentMessage,
    ) -> agent_fabric::Result<Option<AgentResponse>> {
        match message.message_type.as_str() {
            message_types::PING => {
                info!("Tutor {} received ping from {}", self.agent_id, from);
                Ok(Some(AgentResponse::pong(&message.id)))
            }
            message_types::TUTOR_MESSAGE => {
                let tutor_msg: TutorMessage = serde_json::from_slice(&message.payload)
                    .map_err(|e| agent_fabric::AgentFabricError::SerializationError(e.to_string()))?;

                info!("Tutor {} received {} from {}", self.agent_id, tutor_msg.action, from);

                match tutor_msg.action {
                    TutorAction::StartTraining => {
                        self.training_sessions.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let session_id = format!("session-{}", self.get_session_count());
                        Ok(Some(AgentResponse::success(&message.id, Bytes::from(session_id))))
                    }
                    TutorAction::UpdateModel => {
                        let response = "Model updated successfully";
                        Ok(Some(AgentResponse::success(&message.id, Bytes::from(response))))
                    }
                    TutorAction::GetMetrics => {
                        let metrics = format!("Training sessions: {}", self.get_session_count());
                        Ok(Some(AgentResponse::success(&message.id, Bytes::from(metrics))))
                    }
                    TutorAction::StopTraining => {
                        let response = "Training stopped";
                        Ok(Some(AgentResponse::success(&message.id, Bytes::from(response))))
                    }
                }
            }
            _ => {
                warn!("Tutor {} received unknown message type: {}", self.agent_id, message.message_type);
                Ok(Some(AgentResponse::error(&message.id, "Unknown message type")))
            }
        }
    }

    fn supported_message_types(&self) -> Vec<String> {
        vec![
            message_types::PING.to_string(),
            message_types::TUTOR_MESSAGE.to_string(),
        ]
    }
}

impl std::fmt::Display for TutorAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TutorAction::StartTraining => write!(f, "StartTraining"),
            TutorAction::UpdateModel => write!(f, "UpdateModel"),
            TutorAction::GetMetrics => write!(f, "GetMetrics"),
            TutorAction::StopTraining => write!(f, "StopTraining"),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    info!("Starting Agent Ping Example: Twin ↔ Tutor Communication");

    // Create temporary directories for storage
    let twin_storage = TempDir::new()?;
    let tutor_storage = TempDir::new()?;

    // Create agent identifiers
    let twin_id = AgentId::new("twin-001", "node-alpha");
    let tutor_id = AgentId::new("tutor-001", "node-beta");

    info!("Created agents: {} and {}", twin_id, tutor_id);

    // Create agent fabric instances
    let twin_fabric = AgentFabric::new(twin_id.clone());
    let tutor_fabric = AgentFabric::new(tutor_id.clone());

    // Initialize RPC transport for real-time communication
    let twin_rpc_config = HtxConfig {
        listen_addr: "127.0.0.1:8001".parse()?,
        ..Default::default()
    };
    let tutor_rpc_config = HtxConfig {
        listen_addr: "127.0.0.1:8002".parse()?,
        ..Default::default()
    };

    twin_fabric.init_rpc(twin_rpc_config).await?;
    tutor_fabric.init_rpc(tutor_rpc_config).await?;

    // Initialize DTN bridge for offline communication
    twin_fabric.init_dtn(twin_storage.path()).await?;
    tutor_fabric.init_dtn(tutor_storage.path()).await?;

    // Create agent implementations
    let twin_agent = Arc::new(TwinAgent::new(twin_id.clone()));
    let tutor_agent = Arc::new(TutorAgent::new(tutor_id.clone()));

    // Register message handlers
    twin_fabric.register_server(
        "twin-handler".to_string(),
        Box::new(twin_agent.clone())
    ).await;

    tutor_fabric.register_server(
        "tutor-handler".to_string(),
        Box::new(tutor_agent.clone())
    ).await;

    info!("Registered message handlers");

    // Phase 1: RPC Communication (Online)
    info!("\n=== Phase 1: RPC Communication (Online) ===");

    // Twin pings Tutor via RPC
    let ping_message = AgentMessage::ping();
    let rpc_options = DeliveryOptions {
        transport: Transport::Rpc,
        priority: MessagePriority::Normal,
        timeout_ms: Some(5000),
        retry_count: 1,
        require_receipt: false,
    };

    match twin_fabric.send_message(tutor_id.clone(), ping_message, rpc_options).await {
        Ok(Some(response)) => {
            info!("✓ RPC Ping successful: {}", String::from_utf8_lossy(&response.payload.unwrap_or_default()));
        }
        Ok(None) => {
            info!("✓ RPC Ping sent (no response expected)");
        }
        Err(e) => {
            warn!("✗ RPC Ping failed: {}", e);
        }
    }

    // Twin sends training coordination message to Tutor
    let mut training_params = std::collections::HashMap::new();
    training_params.insert("learning_rate".to_string(), 0.001);
    training_params.insert("batch_size".to_string(), 32.0);

    let tutor_msg = TutorMessage {
        session_id: "session-001".to_string(),
        action: TutorAction::StartTraining,
        model_data: Some(Bytes::from("model_weights_v1")),
        parameters: training_params,
    };

    let training_message = AgentMessage::tutor_message(tutor_msg)?;

    match twin_fabric.send_message(tutor_id.clone(), training_message, rpc_options).await {
        Ok(Some(response)) => {
            info!("✓ Training session started: {}", String::from_utf8_lossy(&response.payload.unwrap_or_default()));
        }
        Ok(None) => {
            info!("✓ Training message sent");
        }
        Err(e) => {
            warn!("✗ Training message failed: {}", e);
        }
    }

    // Phase 2: DTN Bundle Communication (Offline Fallback)
    info!("\n=== Phase 2: DTN Bundle Communication (Offline Fallback) ===");

    let bundle_options = DeliveryOptions {
        transport: Transport::Bundle,
        priority: MessagePriority::High,
        timeout_ms: Some(60000), // 1 minute
        retry_count: 3,
        require_receipt: true,
    };

    // Send status query via bundle
    let twin_msg = TwinMessage {
        twin_id: "twin-001".to_string(),
        action: TwinAction::Query,
        data: Bytes::from("status_request"),
    };

    let query_message = AgentMessage::twin_message(twin_msg)?;

    match tutor_fabric.send_message(twin_id.clone(), query_message, bundle_options).await {
        Ok(response) => {
            if let Some(resp) = response {
                info!("✓ Bundle query response: {}", String::from_utf8_lossy(&resp.payload.unwrap_or_default()));
            } else {
                info!("✓ Bundle sent (async delivery)");
            }
        }
        Err(e) => {
            warn!("✗ Bundle delivery failed: {}", e);
        }
    }

    // Phase 3: Auto Transport Selection
    info!("\n=== Phase 3: Auto Transport Selection ===");

    let auto_options = DeliveryOptions {
        transport: Transport::Auto, // Try RPC first, fallback to bundle
        priority: MessagePriority::Normal,
        timeout_ms: Some(5000),
        retry_count: 2,
        require_receipt: false,
    };

    let notification_msg = TwinMessage {
        twin_id: "twin-001".to_string(),
        action: TwinAction::Notify,
        data: Bytes::from("Auto transport test message"),
    };

    let notification_message = AgentMessage::twin_message(notification_msg)?;

    match twin_fabric.send_message(tutor_id.clone(), notification_message, auto_options).await {
        Ok(response) => {
            if let Some(resp) = response {
                info!("✓ Auto transport response: {}", String::from_utf8_lossy(&resp.payload.unwrap_or_default()));
            } else {
                info!("✓ Auto transport message sent");
            }
        }
        Err(e) => {
            warn!("✗ Auto transport failed: {}", e);
        }
    }

    // Phase 4: MLS Group Communication
    info!("\n=== Phase 4: MLS Group Communication ===");

    let group_config = GroupConfig {
        group_id: "training-cohort".to_string(),
        max_members: 10,
        admin_only_add: false,
        admin_only_remove: false,
        require_unanimous_votes: false,
        vote_timeout_seconds: 300,
        ..Default::default()
    };

    // Create training cohort group
    if let Err(e) = twin_fabric.join_group("training-cohort".to_string(), group_config.clone()).await {
        warn!("Failed to create training group: {}", e);
    } else {
        info!("✓ Training cohort group created");

        // Send training coordination message to group
        let group_message = GroupMessage {
            message_id: uuid::Uuid::new_v4().to_string(),
            from: twin_id.clone(),
            message_type: GroupMessageType::Training {
                session_id: "group-session-001".to_string(),
                action: TrainingAction::StartSession,
            },
            payload: Bytes::from("Starting coordinated training session"),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        if let Err(e) = twin_fabric.send_to_group("training-cohort".to_string(), group_message).await {
            warn!("Failed to send group message: {}", e);
        } else {
            info!("✓ Group training message sent");
        }
    }

    // Phase 5: Performance and Statistics
    info!("\n=== Phase 5: Performance Metrics ===");

    // Show agent statistics
    info!("Twin agent updates received: {}", twin_agent.get_update_count());
    info!("Tutor training sessions: {}", tutor_agent.get_session_count());

    // Simulate some delay for async operations
    sleep(Duration::from_millis(100)).await;

    info!("\n=== Agent Ping Example Completed ===");
    info!("Demonstrated:");
    info!("  ✓ Twin ↔ Tutor RPC communication");
    info!("  ✓ DTN bundle fallback delivery");
    info!("  ✓ Auto transport selection");
    info!("  ✓ MLS group coordination");
    info!("  ✓ Message type handling");
    info!("  ✓ Error handling and timeouts");

    Ok(())
}

// Helper functions for demonstration

/// Simulate network failure for testing fallback
#[allow(dead_code)]
async fn simulate_network_failure() {
    warn!("Simulating network failure...");
    sleep(Duration::from_secs(2)).await;
    info!("Network restored");
}

/// Test timeout behavior
#[allow(dead_code)]
async fn test_timeout_behavior() -> Result<(), Box<dyn std::error::Error>> {
    info!("Testing timeout behavior...");

    match timeout(Duration::from_millis(100), sleep(Duration::from_secs(5))).await {
        Ok(_) => info!("Operation completed"),
        Err(_) => info!("Operation timed out as expected"),
    }

    Ok(())
}

/// Demonstrate batch messaging
#[allow(dead_code)]
async fn demonstrate_batch_messaging(
    fabric: &AgentFabric,
    target: AgentId,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Sending batch of messages...");

    let options = DeliveryOptions::default();

    for i in 1..=5 {
        let message = AgentMessage::new(
            "batch-test",
            Bytes::from(format!("Batch message {}", i))
        );

        if let Err(e) = fabric.send_message(target.clone(), message, options.clone()).await {
            warn!("Batch message {} failed: {}", i, e);
        } else {
            info!("Batch message {} sent", i);
        }

        sleep(Duration::from_millis(10)).await;
    }

    Ok(())
}
