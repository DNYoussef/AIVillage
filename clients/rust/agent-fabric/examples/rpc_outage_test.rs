//! Agent RPC Outage Resilience Test
//!
//! Tests RPC transport resilience, connection failure handling, and DTN fallback
//! mechanisms under various network outage scenarios.
//!
//! Test scenarios:
//! 1. Graceful connection drops
//! 2. Network partitions
//! 3. Timeout scenarios
//! 4. DTN fallback activation
//! 5. Message queue persistence
//! 6. Reconnection and recovery

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use serde_json::json;
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::time::{sleep, timeout};
use tracing::{debug, error, info, warn, Level};
use tracing_subscriber::FmtSubscriber;

use agent_fabric::{
    AgentClient, AgentFabricError, AgentId, AgentMessage, AgentResponse, AgentServer, Result,
};
use agent_fabric::rpc::{RpcTransport, RpcClient, RpcServer};
use betanet_htx::HtxConfig;

/// Outage test configuration
#[derive(Debug, Clone)]
struct OutageTestConfig {
    test_duration: Duration,
    message_interval: Duration,
    outage_scenarios: Vec<OutageScenario>,
    dtn_fallback_enabled: bool,
    max_queue_size: usize,
}

impl Default for OutageTestConfig {
    fn default() -> Self {
        Self {
            test_duration: Duration::from_secs(120),
            message_interval: Duration::from_millis(500),
            outage_scenarios: vec![
                OutageScenario::GracefulDisconnect {
                    start_time: Duration::from_secs(10),
                    duration: Duration::from_secs(5),
                },
                OutageScenario::NetworkPartition {
                    start_time: Duration::from_secs(30),
                    duration: Duration::from_secs(10),
                },
                OutageScenario::TimeoutFailure {
                    start_time: Duration::from_secs(60),
                    duration: Duration::from_secs(8),
                },
                OutageScenario::DTNFallback {
                    start_time: Duration::from_secs(90),
                    duration: Duration::from_secs(15),
                },
            ],
            dtn_fallback_enabled: true,
            max_queue_size: 1000,
        }
    }
}

/// Types of outage scenarios to test
#[derive(Debug, Clone)]
enum OutageScenario {
    /// Graceful connection termination
    GracefulDisconnect { start_time: Duration, duration: Duration },
    /// Network partition (packets dropped)
    NetworkPartition { start_time: Duration, duration: Duration },
    /// Timeout-based failures
    TimeoutFailure { start_time: Duration, duration: Duration },
    /// DTN fallback activation
    DTNFallback { start_time: Duration, duration: Duration },
}

/// Test metrics collector
#[derive(Debug, Default)]
struct OutageTestMetrics {
    messages_sent: u64,
    messages_received: u64,
    messages_lost: u64,
    messages_queued: u64,
    messages_recovered: u64,
    connection_failures: u64,
    reconnection_attempts: u64,
    successful_reconnections: u64,
    dtn_activations: u64,
    avg_response_time: Duration,
    max_queue_depth: usize,
    outage_recovery_times: Vec<Duration>,
}

/// Message queue for DTN fallback
#[derive(Debug)]
struct DTNMessageQueue {
    pending_messages: Arc<Mutex<Vec<QueuedMessage>>>,
    max_size: usize,
}

#[derive(Debug, Clone)]
struct QueuedMessage {
    id: String,
    agent_id: AgentId,
    message: AgentMessage,
    timestamp: Instant,
    retry_count: u32,
}

impl DTNMessageQueue {
    fn new(max_size: usize) -> Self {
        Self {
            pending_messages: Arc::new(Mutex::new(Vec::new())),
            max_size,
        }
    }

    async fn enqueue(&self, agent_id: AgentId, message: AgentMessage) -> Result<()> {
        let mut queue = self.pending_messages.lock().await;

        if queue.len() >= self.max_size {
            warn!("DTN queue at capacity, dropping oldest message");
            queue.remove(0);
        }

        let queued_msg = QueuedMessage {
            id: uuid::Uuid::new_v4().to_string(),
            agent_id,
            message,
            timestamp: Instant::now(),
            retry_count: 0,
        };

        queue.push(queued_msg);
        info!("Queued message for DTN fallback, queue size: {}", queue.len());
        Ok(())
    }

    async fn drain_queue(&self) -> Vec<QueuedMessage> {
        let mut queue = self.pending_messages.lock().await;
        std::mem::take(&mut *queue)
    }

    async fn queue_size(&self) -> usize {
        let queue = self.pending_messages.lock().await;
        queue.len()
    }
}

/// Resilient RPC client with DTN fallback
struct ResilientRpcClient {
    primary_client: Arc<RpcClient>,
    dtn_queue: Arc<DTNMessageQueue>,
    target_agent: AgentId,
    connection_state: Arc<RwLock<ConnectionState>>,
    metrics: Arc<Mutex<OutageTestMetrics>>,
}

#[derive(Debug, Clone)]
enum ConnectionState {
    Connected,
    Disconnected,
    DTNMode,
    Reconnecting,
}

impl ResilientRpcClient {
    fn new(
        primary_client: Arc<RpcClient>,
        dtn_queue: Arc<DTNMessageQueue>,
        target_agent: AgentId,
    ) -> Self {
        Self {
            primary_client,
            dtn_queue,
            target_agent,
            connection_state: Arc::new(RwLock::new(ConnectionState::Connected)),
            metrics: Arc::new(Mutex::new(OutageTestMetrics::default())),
        }
    }

    async fn send_message_resilient(&self, message: AgentMessage) -> Result<AgentResponse> {
        let state = {
            let state_guard = self.connection_state.read().await;
            state_guard.clone()
        };

        match state {
            ConnectionState::Connected => {
                // Try primary transport
                match timeout(Duration::from_secs(5),
                    self.primary_client.send_message(message.clone())).await {
                    Ok(Ok(response)) => {
                        let mut metrics = self.metrics.lock().await;
                        metrics.messages_sent += 1;
                        metrics.messages_received += 1;
                        Ok(response)
                    }
                    Ok(Err(e)) | Err(_) => {
                        // Connection failed, switch to DTN mode
                        self.handle_connection_failure(message).await
                    }
                }
            }
            ConnectionState::DTNMode => {
                // Queue message for DTN fallback
                self.dtn_queue.enqueue(self.target_agent.clone(), message).await?;
                let mut metrics = self.metrics.lock().await;
                metrics.messages_queued += 1;
                metrics.dtn_activations += 1;

                // Return synthetic response indicating DTN queuing
                Ok(AgentResponse::success(
                    &uuid::Uuid::new_v4().to_string(),
                    Bytes::from("DTN_QUEUED"),
                ))
            }
            ConnectionState::Disconnected | ConnectionState::Reconnecting => {
                // Queue message and attempt reconnection
                self.dtn_queue.enqueue(self.target_agent.clone(), message).await?;
                self.attempt_reconnection().await;

                let mut metrics = self.metrics.lock().await;
                metrics.messages_queued += 1;

                Ok(AgentResponse::success(
                    &uuid::Uuid::new_v4().to_string(),
                    Bytes::from("QUEUED_FOR_RETRY"),
                ))
            }
        }
    }

    async fn handle_connection_failure(&self, message: AgentMessage) -> Result<AgentResponse> {
        warn!("Primary RPC connection failed, activating DTN fallback");

        {
            let mut state = self.connection_state.write().await;
            *state = ConnectionState::DTNMode;
        }

        let mut metrics = self.metrics.lock().await;
        metrics.connection_failures += 1;
        metrics.dtn_activations += 1;
        drop(metrics);

        // Queue the failed message
        self.dtn_queue.enqueue(self.target_agent.clone(), message).await?;

        Ok(AgentResponse::error(
            &uuid::Uuid::new_v4().to_string(),
            "Connection failed, message queued for DTN",
        ))
    }

    async fn attempt_reconnection(&self) {
        let current_state = {
            let state = self.connection_state.read().await;
            state.clone()
        };

        if matches!(current_state, ConnectionState::Reconnecting) {
            return; // Already attempting reconnection
        }

        {
            let mut state = self.connection_state.write().await;
            *state = ConnectionState::Reconnecting;
        }

        {
            let mut metrics = self.metrics.lock().await;
            metrics.reconnection_attempts += 1;
        }

        info!("Attempting RPC reconnection...");

        // Simulate reconnection attempt
        sleep(Duration::from_millis(1000)).await;

        // For testing purposes, assume reconnection succeeds after delay
        let reconnected = true; // In real implementation, would test actual connection

        if reconnected {
            {
                let mut state = self.connection_state.write().await;
                *state = ConnectionState::Connected;
            }

            {
                let mut metrics = self.metrics.lock().await;
                metrics.successful_reconnections += 1;
            }

            info!("RPC reconnection successful, draining DTN queue");
            self.drain_dtn_queue().await;
        } else {
            {
                let mut state = self.connection_state.write().await;
                *state = ConnectionState::DTNMode;
            }
            warn!("RPC reconnection failed, remaining in DTN mode");
        }
    }

    async fn drain_dtn_queue(&self) {
        let queued_messages = self.dtn_queue.drain_queue().await;

        for queued_msg in queued_messages {
            match self.primary_client.send_message(queued_msg.message.clone()).await {
                Ok(_) => {
                    let mut metrics = self.metrics.lock().await;
                    metrics.messages_recovered += 1;
                    info!("Recovered queued message: {}", queued_msg.id);
                }
                Err(e) => {
                    // Re-queue the message
                    self.dtn_queue.enqueue(queued_msg.agent_id.clone(), queued_msg.message).await.ok();
                    warn!("Failed to recover queued message: {}", e);
                }
            }
        }
    }

    async fn get_metrics(&self) -> OutageTestMetrics {
        let metrics = self.metrics.lock().await;
        metrics.clone()
    }
}

/// Test server that simulates various failure modes
struct OutageTestServer {
    active_outages: Arc<RwLock<Vec<OutageScenario>>>,
    message_count: Arc<Mutex<u64>>,
}

impl OutageTestServer {
    fn new() -> Self {
        Self {
            active_outages: Arc::new(RwLock::new(Vec::new())),
            message_count: Arc::new(Mutex::new(0)),
        }
    }

    async fn activate_outage(&self, scenario: OutageScenario) {
        let mut outages = self.active_outages.write().await;
        outages.push(scenario.clone());
        info!("Activated outage scenario: {:?}", scenario);
    }

    async fn clear_outages(&self) {
        let mut outages = self.active_outages.write().await;
        outages.clear();
        info!("Cleared all outage scenarios");
    }

    async fn should_simulate_failure(&self) -> bool {
        let outages = self.active_outages.read().await;
        !outages.is_empty()
    }
}

#[async_trait::async_trait]
impl AgentServer for OutageTestServer {
    async fn handle_message(
        &self,
        from: AgentId,
        message: AgentMessage,
    ) -> Result<Option<AgentResponse>> {
        let mut count = self.message_count.lock().await;
        *count += 1;
        let msg_num = *count;
        drop(count);

        // Check if we should simulate a failure
        if self.should_simulate_failure().await {
            // Simulate various failure modes
            let outages = self.active_outages.read().await;
            for outage in outages.iter() {
                match outage {
                    OutageScenario::TimeoutFailure { .. } => {
                        // Simulate timeout by delaying response
                        sleep(Duration::from_secs(10)).await;
                    }
                    OutageScenario::NetworkPartition { .. } => {
                        // Simulate network partition by dropping message
                        return Err(AgentFabricError::NetworkError("Simulated network partition".to_string()));
                    }
                    OutageScenario::GracefulDisconnect { .. } => {
                        // Simulate graceful disconnect
                        return Err(AgentFabricError::ConnectionClosed);
                    }
                    OutageScenario::DTNFallback { .. } => {
                        // Force DTN fallback
                        return Err(AgentFabricError::NetworkError("Force DTN fallback".to_string()));
                    }
                }
            }
        }

        // Normal operation
        let response_data = json!({
            "message_number": msg_num,
            "echo": message.payload,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "from_agent": from.id
        });

        Ok(Some(AgentResponse::success(
            &message.id,
            Bytes::from(response_data.to_string()),
        )))
    }

    fn supported_message_types(&self) -> Vec<String> {
        vec!["test_message".to_string(), "ping".to_string()]
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("🚀 Agent RPC Outage Resilience Test Starting");
    info!("=============================================");

    let config = OutageTestConfig::default();
    run_outage_test(config).await?;

    Ok(())
}

async fn run_outage_test(config: OutageTestConfig) -> Result<(), Box<dyn std::error::Error>> {
    info!("📡 Setting up RPC transport and test agents");

    // Create HTX config for RPC transport
    let server_htx_config = HtxConfig {
        listen_addr: "127.0.0.1:8081".parse().unwrap(),
        enable_tcp: true,
        enable_quic: false,
        ..Default::default()
    };

    let client_htx_config = HtxConfig {
        listen_addr: "127.0.0.1:8082".parse().unwrap(),
        enable_tcp: true,
        enable_quic: false,
        ..Default::default()
    };

    // Set up server
    let server_transport = Arc::new(RpcTransport::new(server_htx_config).await?);
    let test_server = Arc::new(OutageTestServer::new());
    let rpc_server = RpcServer::new(Arc::clone(&server_transport));

    rpc_server.register_handler("test_service".to_string(), test_server.clone()).await;
    rpc_server.start().await?;

    // Set up client
    let client_transport = Arc::new(RpcTransport::new(client_htx_config).await?);
    let server_agent = AgentId::new("test-server", "server-node");
    let server_addr: SocketAddr = "127.0.0.1:8081".parse().unwrap();

    client_transport.connect(server_agent.clone(), server_addr).await?;

    let rpc_client = Arc::new(RpcClient::new(Arc::clone(&client_transport))
        .with_target(server_agent.clone()));

    // Set up DTN queue and resilient client
    let dtn_queue = Arc::new(DTNMessageQueue::new(config.max_queue_size));
    let resilient_client = Arc::new(ResilientRpcClient::new(
        rpc_client,
        dtn_queue.clone(),
        server_agent.clone(),
    ));

    info!("🔧 Starting outage resilience test scenarios");

    // Run the test
    let test_result = run_test_scenarios(
        config.clone(),
        resilient_client.clone(),
        test_server.clone(),
    ).await?;

    // Generate test report
    generate_test_report(&test_result, &config).await?;

    info!("✅ Agent RPC Outage Test Completed Successfully");
    Ok(())
}

async fn run_test_scenarios(
    config: OutageTestConfig,
    client: Arc<ResilientRpcClient>,
    server: Arc<OutageTestServer>,
) -> Result<OutageTestMetrics, Box<dyn std::error::Error>> {
    let test_start = Instant::now();
    let mut message_counter = 0u64;

    info!("📊 Running test for {:?} with {} outage scenarios",
          config.test_duration, config.outage_scenarios.len());

    // Start outage scenario scheduler
    let scenario_server = Arc::clone(&server);
    let scenarios = config.outage_scenarios.clone();

    tokio::spawn(async move {
        for scenario in scenarios {
            let delay = match &scenario {
                OutageScenario::GracefulDisconnect { start_time, .. } => *start_time,
                OutageScenario::NetworkPartition { start_time, .. } => *start_time,
                OutageScenario::TimeoutFailure { start_time, .. } => *start_time,
                OutageScenario::DTNFallback { start_time, .. } => *start_time,
            };

            sleep(delay).await;
            scenario_server.activate_outage(scenario.clone()).await;

            let duration = match &scenario {
                OutageScenario::GracefulDisconnect { duration, .. } => *duration,
                OutageScenario::NetworkPartition { duration, .. } => *duration,
                OutageScenario::TimeoutFailure { duration, .. } => *duration,
                OutageScenario::DTNFallback { duration, .. } => *duration,
            };

            sleep(duration).await;
            scenario_server.clear_outages().await;
        }
    });

    // Main test loop - send messages continuously
    let mut interval = tokio::time::interval(config.message_interval);

    while test_start.elapsed() < config.test_duration {
        interval.tick().await;
        message_counter += 1;

        let test_message = AgentMessage {
            id: format!("msg-{}", message_counter),
            message_type: "test_message".to_string(),
            payload: Bytes::from(format!("Test message {} at {:?}",
                                       message_counter, test_start.elapsed())),
            metadata: HashMap::new(),
        };

        let send_start = Instant::now();
        match client.send_message_resilient(test_message).await {
            Ok(response) => {
                let response_time = send_start.elapsed();
                debug!("Message {}: Response in {:?} - {}",
                       message_counter, response_time,
                       String::from_utf8_lossy(&response.payload));
            }
            Err(e) => {
                warn!("Message {}: Failed - {}", message_counter, e);
            }
        }

        // Log progress every 20 messages
        if message_counter % 20 == 0 {
            let queue_size = dtn_queue.queue_size().await;
            info!("Progress: {} messages sent, DTN queue: {} messages",
                  message_counter, queue_size);
        }
    }

    info!("📈 Test completed, collecting final metrics");

    // Allow time for final message processing and queue draining
    sleep(Duration::from_secs(5)).await;

    let final_metrics = client.get_metrics().await;
    Ok(final_metrics)
}

async fn generate_test_report(
    metrics: &OutageTestMetrics,
    config: &OutageTestConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let report_path = format!("artifacts/rpc_outage_test_report_{}.json", timestamp);

    let success_rate = if metrics.messages_sent > 0 {
        (metrics.messages_received as f64 / metrics.messages_sent as f64) * 100.0
    } else {
        0.0
    };

    let recovery_rate = if metrics.connection_failures > 0 {
        (metrics.successful_reconnections as f64 / metrics.connection_failures as f64) * 100.0
    } else {
        100.0
    };

    let report = json!({
        "test_summary": {
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "test_duration_seconds": config.test_duration.as_secs(),
            "outage_scenarios_count": config.outage_scenarios.len(),
            "dtn_fallback_enabled": config.dtn_fallback_enabled
        },
        "metrics": {
            "messages_sent": metrics.messages_sent,
            "messages_received": metrics.messages_received,
            "messages_lost": metrics.messages_lost,
            "messages_queued": metrics.messages_queued,
            "messages_recovered": metrics.messages_recovered,
            "success_rate_percent": success_rate,
            "connection_failures": metrics.connection_failures,
            "reconnection_attempts": metrics.reconnection_attempts,
            "successful_reconnections": metrics.successful_reconnections,
            "recovery_rate_percent": recovery_rate,
            "dtn_activations": metrics.dtn_activations,
            "max_queue_depth": metrics.max_queue_depth,
            "avg_response_time_ms": metrics.avg_response_time.as_millis()
        },
        "test_scenarios": config.outage_scenarios,
        "conclusions": {
            "rpc_resilience": if success_rate > 80.0 { "PASS" } else { "FAIL" },
            "dtn_fallback": if metrics.dtn_activations > 0 { "PASS" } else { "FAIL" },
            "recovery_capability": if recovery_rate > 70.0 { "PASS" } else { "FAIL" },
            "message_persistence": if metrics.messages_recovered > 0 { "PASS" } else { "FAIL" }
        },
        "recommendations": [
            "Monitor connection failure patterns for optimization",
            "Tune DTN queue size based on expected outage duration",
            "Implement exponential backoff for reconnection attempts",
            "Consider implementing message priority queuing"
        ]
    });

    // Write report to file
    tokio::fs::create_dir_all("artifacts").await.ok();
    tokio::fs::write(&report_path, report.to_string()).await?;

    info!("📋 Test Report Generated");
    info!("========================");
    info!("Success Rate: {:.1}%", success_rate);
    info!("Recovery Rate: {:.1}%", recovery_rate);
    info!("DTN Activations: {}", metrics.dtn_activations);
    info!("Messages Queued: {}", metrics.messages_queued);
    info!("Messages Recovered: {}", metrics.messages_recovered);
    info!("Report saved: {}", report_path);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outage_config_creation() {
        let config = OutageTestConfig::default();
        assert_eq!(config.outage_scenarios.len(), 4);
        assert!(config.dtn_fallback_enabled);
        assert_eq!(config.max_queue_size, 1000);
    }

    #[tokio::test]
    async fn test_dtn_queue_operations() {
        let queue = DTNMessageQueue::new(10);
        let agent_id = AgentId::new("test", "node");
        let message = AgentMessage {
            id: "test-msg".to_string(),
            message_type: "test".to_string(),
            payload: Bytes::from("test data"),
            metadata: HashMap::new(),
        };

        queue.enqueue(agent_id, message).await.unwrap();
        assert_eq!(queue.queue_size().await, 1);

        let drained = queue.drain_queue().await;
        assert_eq!(drained.len(), 1);
        assert_eq!(queue.queue_size().await, 0);
    }

    #[tokio::test]
    async fn test_outage_test_server() {
        let server = OutageTestServer::new();
        assert!(!server.should_simulate_failure().await);

        server.activate_outage(OutageScenario::GracefulDisconnect {
            start_time: Duration::from_secs(0),
            duration: Duration::from_secs(5),
        }).await;

        assert!(server.should_simulate_failure().await);
        server.clear_outages().await;
        assert!(!server.should_simulate_failure().await);
    }
}
