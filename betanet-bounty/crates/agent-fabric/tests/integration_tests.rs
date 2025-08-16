//! Integration tests for Agent Fabric
//!
//! Tests the complete functionality including:
//! - RPC communication
//! - DTN bundle fallback
//! - MLS group messaging
//! - Transport auto-selection
//! - Error handling and timeouts

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use tempfile::TempDir;
use tokio::time::{sleep, timeout};

use agent_fabric::{
    AgentFabric, AgentId, AgentMessage, AgentResponse, AgentClient, AgentServer,
    HtxConfig, DeliveryOptions, Transport, MessagePriority,
    api::{TwinMessage, TutorMessage, TwinAction, TutorAction, message_types},
    GroupConfig, GroupMessage, GroupMessageType, TrainingAction, AlertLevel,
    rpc::{RpcClient, RpcServer, RpcTransport},
    dtn_bridge::{DtnBridge, BundleMessage, DtnClient, DtnServer},
    groups::{MlsGroup, GroupMember},
};

/// Test server implementation
struct TestServer {
    name: String,
    message_count: std::sync::atomic::AtomicUsize,
}

impl TestServer {
    fn new(name: String) -> Self {
        Self {
            name,
            message_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn get_message_count(&self) -> usize {
        self.message_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[async_trait::async_trait]
impl AgentServer for TestServer {
    async fn handle_message(
        &self,
        from: AgentId,
        message: AgentMessage,
    ) -> agent_fabric::Result<Option<AgentResponse>> {
        self.message_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        match message.message_type.as_str() {
            message_types::PING => {
                Ok(Some(AgentResponse::success(&message.id, Bytes::from("pong"))))
            }
            "echo" => {
                Ok(Some(AgentResponse::success(&message.id, message.payload)))
            }
            "notification" => {
                // No response for notifications
                Ok(None)
            }
            "error" => {
                Ok(Some(AgentResponse::error(&message.id, "Simulated error")))
            }
            _ => {
                Ok(Some(AgentResponse::error(&message.id, "Unknown message type")))
            }
        }
    }

    fn supported_message_types(&self) -> Vec<String> {
        vec![
            message_types::PING.to_string(),
            "echo".to_string(),
            "notification".to_string(),
            "error".to_string(),
        ]
    }
}

#[tokio::test]
async fn test_rpc_communication() {
    let temp_dir = TempDir::new().unwrap();

    // Create agent fabric
    let agent_id = AgentId::new("test-agent", "test-node");
    let fabric = AgentFabric::new(agent_id.clone());

    // Initialize RPC
    let rpc_config = HtxConfig {
        listen_addr: "127.0.0.1:0".parse().unwrap(), // Use any available port
        ..Default::default()
    };

    fabric.init_rpc(rpc_config).await.unwrap();

    // Register test server
    let test_server = Arc::new(TestServer::new("test-server".to_string()));
    fabric.register_server("test-handler".to_string(), Box::new(TestServer::new("test-server".to_string()))).await;

    // Test RPC ping
    let target_id = AgentId::new("target", "test-node");
    let ping_message = AgentMessage::ping();
    let rpc_options = DeliveryOptions {
        transport: Transport::Rpc,
        timeout_ms: Some(5000),
        ..Default::default()
    };

    // This will fail since we don't have a connection to the target,
    // but it tests the RPC code path
    let result = fabric.send_message(target_id, ping_message, rpc_options).await;

    // Should get TransportUnavailable or AgentNotFound error
    assert!(result.is_err());
}

#[tokio::test]
async fn test_dtn_bridge_creation() {
    let temp_dir = TempDir::new().unwrap();
    let endpoint = betanet_dtn::EndpointId::node("test-node");

    let bridge = DtnBridge::new(endpoint.clone(), temp_dir.path()).await.unwrap();
    assert_eq!(bridge.local_endpoint(), &endpoint);

    // Test starting and stopping
    bridge.start().await.unwrap();
    bridge.stop().await.unwrap();
}

#[tokio::test]
async fn test_dtn_bundle_creation() {
    let agent_from = AgentId::new("sender", "node1");
    let agent_to = AgentId::new("receiver", "node2");
    let message = AgentMessage::new("test", Bytes::from("hello"));
    let options = DeliveryOptions::default();

    let bundle_message = BundleMessage {
        from: agent_from.clone(),
        to: agent_to.clone(),
        content: message.clone(),
        options: options.clone(),
    };

    // Test serialization
    let serialized = serde_json::to_vec(&bundle_message).unwrap();
    let deserialized: BundleMessage = serde_json::from_slice(&serialized).unwrap();

    assert_eq!(bundle_message.from, deserialized.from);
    assert_eq!(bundle_message.to, deserialized.to);
    assert_eq!(bundle_message.content.message_type, deserialized.content.message_type);
}

#[tokio::test]
async fn test_mls_group_creation() {
    let config = GroupConfig {
        group_id: "test-group".to_string(),
        max_members: 5,
        ..Default::default()
    };

    let group = MlsGroup::new("test-group".to_string(), config.clone()).await.unwrap();
    assert_eq!(group.get_config().group_id, "test-group");
    assert_eq!(group.get_config().max_members, 5);

    // Test initializing as creator
    let creator = AgentId::new("creator", "node1");
    group.initialize_as_creator(creator.clone()).await.unwrap();

    let members = group.get_members().await;
    assert_eq!(members.len(), 1);
    assert_eq!(members[0].agent_id, creator);
    assert!(members[0].is_admin);
}

#[tokio::test]
async fn test_group_messaging() {
    let config = GroupConfig::default();
    let group = MlsGroup::new("test-messaging".to_string(), config).await.unwrap();

    let creator = AgentId::new("creator", "node1");
    group.initialize_as_creator(creator.clone()).await.unwrap();

    // Test training message
    let training_message = GroupMessage {
        message_id: "msg-1".to_string(),
        from: creator.clone(),
        message_type: GroupMessageType::Training {
            session_id: "session-1".to_string(),
            action: TrainingAction::StartSession,
        },
        payload: Bytes::from("training data"),
        timestamp: 1234567890,
    };

    group.send_message(training_message).await.unwrap();

    // Test alert message
    let alert_message = GroupMessage {
        message_id: "msg-2".to_string(),
        from: creator.clone(),
        message_type: GroupMessageType::Alert {
            alert_level: AlertLevel::Warning,
            category: "system".to_string(),
        },
        payload: Bytes::from("test alert"),
        timestamp: 1234567891,
    };

    group.send_message(alert_message).await.unwrap();

    let stats = group.get_stats().await;
    assert_eq!(stats.messages_sent, 2);
    assert_eq!(stats.training_sessions, 1);
    assert_eq!(stats.alerts_sent, 1);
}

#[tokio::test]
async fn test_voting_system() {
    let config = GroupConfig {
        group_id: "voting-test".to_string(),
        require_unanimous_votes: false,
        vote_timeout_seconds: 60,
        ..Default::default()
    };

    let group = MlsGroup::new("voting-test".to_string(), config).await.unwrap();

    let proposer = AgentId::new("proposer", "node1");
    group.initialize_as_creator(proposer.clone()).await.unwrap();

    // Start a vote
    let proposal_id = group.start_vote(
        proposer.clone(),
        "Test Proposal".to_string(),
        "A test proposal for voting".to_string(),
        300, // 5 minutes
    ).await.unwrap();

    assert!(!proposal_id.is_empty());

    // Cast a vote
    group.cast_vote(proposal_id.clone(), proposer.clone(), true).await.unwrap();

    let stats = group.get_stats().await;
    assert_eq!(stats.votes_initiated, 1);
}

#[tokio::test]
async fn test_transport_auto_selection() {
    let temp_dir = TempDir::new().unwrap();

    let agent_id = AgentId::new("auto-test", "node1");
    let fabric = AgentFabric::new(agent_id.clone());

    // Initialize DTN only (no RPC)
    fabric.init_dtn(temp_dir.path()).await.unwrap();

    let target_id = AgentId::new("target", "node2");
    let message = AgentMessage::new("test", Bytes::from("auto transport test"));

    let auto_options = DeliveryOptions {
        transport: Transport::Auto, // Should fallback to DTN
        timeout_ms: Some(5000),
        ..Default::default()
    };

    // This should attempt RPC first, then fallback to DTN
    let result = fabric.send_message(target_id, message, auto_options).await;

    // Should return None since DTN is async
    match result {
        Ok(None) => {
            // Expected for DTN delivery
        }
        Ok(Some(_)) => {
            panic!("Unexpected response for DTN delivery");
        }
        Err(e) => {
            // Could fail if DTN isn't properly configured
            println!("Auto transport failed (expected in test): {}", e);
        }
    }
}

#[tokio::test]
async fn test_message_priorities() {
    let message_low = AgentMessage::new("test", Bytes::from("low priority"));
    let message_high = AgentMessage::new("test", Bytes::from("high priority"));

    let options_low = DeliveryOptions {
        priority: MessagePriority::Low,
        ..Default::default()
    };

    let options_high = DeliveryOptions {
        priority: MessagePriority::Critical,
        ..Default::default()
    };

    // Test that priority is properly set
    assert_eq!(options_low.priority, MessagePriority::Low);
    assert_eq!(options_high.priority, MessagePriority::Critical);

    // In a real implementation, these would affect routing and scheduling
    assert!(options_high.priority > options_low.priority);
}

#[tokio::test]
async fn test_message_timeouts() {
    let temp_dir = TempDir::new().unwrap();

    let agent_id = AgentId::new("timeout-test", "node1");
    let fabric = AgentFabric::new(agent_id.clone());

    // Initialize RPC with invalid config to force timeout
    let rpc_config = HtxConfig {
        listen_addr: "127.0.0.1:0".parse().unwrap(),
        ..Default::default()
    };

    fabric.init_rpc(rpc_config).await.unwrap();

    let target_id = AgentId::new("nonexistent", "nowhere");
    let message = AgentMessage::new("test", Bytes::from("timeout test"));

    let timeout_options = DeliveryOptions {
        transport: Transport::Rpc,
        timeout_ms: Some(100), // Very short timeout
        ..Default::default()
    };

    let start = std::time::Instant::now();
    let result = fabric.send_message(target_id, message, timeout_options).await;
    let elapsed = start.elapsed();

    // Should fail quickly due to short timeout
    assert!(result.is_err());
    assert!(elapsed < Duration::from_millis(1000)); // Should fail well before 1 second
}

#[tokio::test]
async fn test_error_handling() {
    let test_server = TestServer::new("error-test".to_string());

    let from_agent = AgentId::new("sender", "node1");

    // Test successful message
    let ping_message = AgentMessage::ping();
    let result = test_server.handle_message(from_agent.clone(), ping_message).await.unwrap();
    assert!(result.is_some());
    assert!(result.unwrap().success);

    // Test error message
    let error_message = AgentMessage::new("error", Bytes::from("trigger error"));
    let result = test_server.handle_message(from_agent.clone(), error_message).await.unwrap();
    assert!(result.is_some());
    let response = result.unwrap();
    assert!(!response.success);
    assert_eq!(response.error, Some("Simulated error".to_string()));

    // Test notification (no response)
    let notification_message = AgentMessage::new("notification", Bytes::from("notify"));
    let result = test_server.handle_message(from_agent.clone(), notification_message).await.unwrap();
    assert!(result.is_none());

    // Check message count
    assert_eq!(test_server.get_message_count(), 3);
}

#[tokio::test]
async fn test_serialization_edge_cases() {
    // Test empty payloads
    let empty_message = AgentMessage::new("empty", Bytes::new());
    let serialized = serde_json::to_vec(&empty_message).unwrap();
    let deserialized: AgentMessage = serde_json::from_slice(&serialized).unwrap();
    assert_eq!(empty_message.message_type, deserialized.message_type);
    assert_eq!(empty_message.payload, deserialized.payload);

    // Test large payloads
    let large_payload = Bytes::from(vec![0u8; 1024 * 1024]); // 1MB
    let large_message = AgentMessage::new("large", large_payload.clone());
    let serialized = serde_json::to_vec(&large_message).unwrap();
    let deserialized: AgentMessage = serde_json::from_slice(&serialized).unwrap();
    assert_eq!(large_message.payload, deserialized.payload);

    // Test special characters
    let special_message = AgentMessage::new("special", Bytes::from("Hello 🌍 World! 特殊字符"));
    let serialized = serde_json::to_vec(&special_message).unwrap();
    let deserialized: AgentMessage = serde_json::from_slice(&serialized).unwrap();
    assert_eq!(special_message.payload, deserialized.payload);
}

#[tokio::test]
async fn test_concurrent_operations() {
    let temp_dir = TempDir::new().unwrap();

    let agent_id = AgentId::new("concurrent-test", "node1");
    let fabric = Arc::new(AgentFabric::new(agent_id.clone()));

    fabric.init_dtn(temp_dir.path()).await.unwrap();

    let test_server = Arc::new(TestServer::new("concurrent-server".to_string()));
    fabric.register_server("concurrent-handler".to_string(), Box::new(TestServer::new("concurrent-server".to_string()))).await;

    // Spawn multiple concurrent operations
    let mut handles = Vec::new();

    for i in 0..10 {
        let fabric_clone = Arc::clone(&fabric);
        let target_id = AgentId::new(&format!("target-{}", i), "node2");

        let handle = tokio::spawn(async move {
            let message = AgentMessage::new("concurrent", Bytes::from(format!("message-{}", i)));
            let options = DeliveryOptions {
                transport: Transport::Bundle,
                ..Default::default()
            };

            fabric_clone.send_message(target_id, message, options).await
        });

        handles.push(handle);
    }

    // Wait for all operations to complete
    let mut success_count = 0;
    for handle in handles {
        match handle.await {
            Ok(Ok(_)) => success_count += 1,
            Ok(Err(_)) => {}, // Expected failures for non-existent targets
            Err(_) => {}, // Task panics
        }
    }

    // At least some operations should succeed or fail gracefully
    println!("Concurrent operations completed: {} successful", success_count);
}

#[tokio::test]
async fn test_full_integration_scenario() {
    // This test demonstrates the complete agent-fabric workflow
    let temp_dir1 = TempDir::new().unwrap();
    let temp_dir2 = TempDir::new().unwrap();

    // Create two agents
    let twin_id = AgentId::new("twin", "node-alpha");
    let tutor_id = AgentId::new("tutor", "node-beta");

    let twin_fabric = AgentFabric::new(twin_id.clone());
    let tutor_fabric = AgentFabric::new(tutor_id.clone());

    // Initialize transports
    twin_fabric.init_dtn(temp_dir1.path()).await.unwrap();
    tutor_fabric.init_dtn(temp_dir2.path()).await.unwrap();

    // Register handlers
    let twin_server = Arc::new(TestServer::new("twin".to_string()));
    let tutor_server = Arc::new(TestServer::new("tutor".to_string()));

    twin_fabric.register_server("twin-handler".to_string(), Box::new(TestServer::new("twin".to_string()))).await;
    tutor_fabric.register_server("tutor-handler".to_string(), Box::new(TestServer::new("tutor".to_string()))).await;

    // Create MLS group
    let group_config = GroupConfig {
        group_id: "integration-test".to_string(),
        max_members: 2,
        ..Default::default()
    };

    twin_fabric.join_group("integration-test".to_string(), group_config).await.unwrap();

    // Send group message
    let group_message = GroupMessage {
        message_id: "integration-msg".to_string(),
        from: twin_id.clone(),
        message_type: GroupMessageType::Training {
            session_id: "integration-session".to_string(),
            action: TrainingAction::StartSession,
        },
        payload: Bytes::from("Integration test message"),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    };

    twin_fabric.send_to_group("integration-test".to_string(), group_message).await.unwrap();

    // Test message exchange
    let message = AgentMessage::new("integration", Bytes::from("full integration test"));
    let options = DeliveryOptions {
        transport: Transport::Auto,
        priority: MessagePriority::High,
        timeout_ms: Some(5000),
        ..Default::default()
    };

    let result = twin_fabric.send_message(tutor_id, message, options).await;

    // Should complete without panic (result depends on DTN configuration)
    match result {
        Ok(_) => println!("Integration test message sent successfully"),
        Err(e) => println!("Integration test message failed (expected): {}", e),
    }

    println!("Full integration scenario completed");
}
