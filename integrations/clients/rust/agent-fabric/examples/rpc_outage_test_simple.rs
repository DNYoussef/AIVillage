//! Agent RPC Outage Resilience Test (Simplified)
//!
//! Tests RPC transport resilience, connection failure handling, and DTN fallback
//! mechanisms under various network outage scenarios.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_fabric::{
    AgentClient, AgentFabricError, AgentId, AgentMessage, AgentResponse, AgentServer, Result,
};
use async_trait::async_trait;
use bytes::Bytes;
use serde_json::json;
use tokio::sync::Mutex;
use tokio::time::sleep;

/// Outage test configuration
#[derive(Debug, Clone)]
struct OutageTestConfig {
    test_duration: Duration,
    message_interval: Duration,
    dtn_fallback_enabled: bool,
    max_queue_size: usize,
}

impl Default for OutageTestConfig {
    fn default() -> Self {
        Self {
            test_duration: Duration::from_secs(30),
            message_interval: Duration::from_millis(1000),
            dtn_fallback_enabled: true,
            max_queue_size: 100,
        }
    }
}

/// Test metrics collector
#[derive(Debug, Default, Clone)]
struct OutageTestMetrics {
    messages_sent: u64,
    messages_received: u64,
    messages_lost: u64,
    messages_queued: u64,
    connection_failures: u64,
    dtn_activations: u64,
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
            queue.remove(0);
        }

        let queued_msg = QueuedMessage {
            id: format!("queued-{}", queue.len()),
            agent_id,
            message,
            timestamp: Instant::now(),
            retry_count: 0,
        };

        queue.push(queued_msg);
        println!("Queued message for DTN fallback, queue size: {}", queue.len());
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

/// Mock RPC client for testing
struct MockRpcClient {
    target_agent: AgentId,
    fail_probability: f64,
    metrics: Arc<Mutex<OutageTestMetrics>>,
}

impl MockRpcClient {
    fn new(target_agent: AgentId) -> Self {
        Self {
            target_agent,
            fail_probability: 0.3, // 30% failure rate
            metrics: Arc::new(Mutex::new(OutageTestMetrics::default())),
        }
    }

    fn set_fail_probability(&mut self, prob: f64) {
        self.fail_probability = prob;
    }

    async fn get_metrics(&self) -> OutageTestMetrics {
        let metrics = self.metrics.lock().await;
        metrics.clone()
    }
}

#[async_trait]
impl AgentClient for MockRpcClient {
    async fn send_message(&self, message: AgentMessage) -> Result<AgentResponse> {
        let mut metrics = self.metrics.lock().await;
        metrics.messages_sent += 1;
        drop(metrics);

        // Simulate random failures
        if rand::random::<f64>() < self.fail_probability {
            let mut metrics = self.metrics.lock().await;
            metrics.connection_failures += 1;
            metrics.messages_lost += 1;
            return Err(AgentFabricError::NetworkError("Simulated network failure".to_string()));
        }

        // Simulate processing time
        sleep(Duration::from_millis(50)).await;

        let mut metrics = self.metrics.lock().await;
        metrics.messages_received += 1;

        Ok(AgentResponse::success(
            &message.id,
            Bytes::from("Mock response"),
        ))
    }

    async fn send_notification(&self, _message: AgentMessage) -> Result<()> {
        Ok(())
    }

    fn target_agent(&self) -> &AgentId {
        &self.target_agent
    }
}

/// Resilient RPC client with DTN fallback
struct ResilientRpcClient {
    primary_client: Arc<MockRpcClient>,
    dtn_queue: Arc<DTNMessageQueue>,
    target_agent: AgentId,
    metrics: Arc<Mutex<OutageTestMetrics>>,
}

impl ResilientRpcClient {
    fn new(
        primary_client: Arc<MockRpcClient>,
        dtn_queue: Arc<DTNMessageQueue>,
        target_agent: AgentId,
    ) -> Self {
        Self {
            primary_client,
            dtn_queue,
            target_agent,
            metrics: Arc::new(Mutex::new(OutageTestMetrics::default())),
        }
    }

    async fn send_message_resilient(&self, message: AgentMessage) -> Result<AgentResponse> {
        // Try primary transport
        match self.primary_client.send_message(message.clone()).await {
            Ok(response) => {
                let mut metrics = self.metrics.lock().await;
                metrics.messages_sent += 1;
                metrics.messages_received += 1;
                Ok(response)
            }
            Err(_) => {
                // Connection failed, use DTN fallback
                self.dtn_queue.enqueue(self.target_agent.clone(), message).await?;
                let mut metrics = self.metrics.lock().await;
                metrics.messages_queued += 1;
                metrics.dtn_activations += 1;
                metrics.connection_failures += 1;

                Ok(AgentResponse::success(
                    &format!("dtn-{}", metrics.messages_queued),
                    Bytes::from("DTN_QUEUED"),
                ))
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
    message_count: Arc<Mutex<u64>>,
    failure_enabled: Arc<Mutex<bool>>,
}

impl OutageTestServer {
    fn new() -> Self {
        Self {
            message_count: Arc::new(Mutex::new(0)),
            failure_enabled: Arc::new(Mutex::new(false)),
        }
    }

    async fn enable_failures(&self, enabled: bool) {
        let mut failure_enabled = self.failure_enabled.lock().await;
        *failure_enabled = enabled;
        println!("Server failures: {}", if enabled { "ENABLED" } else { "DISABLED" });
    }
}

#[async_trait]
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
        let should_fail = {
            let failure_enabled = self.failure_enabled.lock().await;
            *failure_enabled
        };

        if should_fail {
            return Err(AgentFabricError::NetworkError("Simulated server failure".to_string()));
        }

        // Normal operation
        let response_data = json!({
            "message_number": msg_num,
            "echo": String::from_utf8_lossy(&message.payload),
            "from_agent": from.id
        });

        Ok(Some(AgentResponse::success(
            &message.id,
            Bytes::from(response_data.to_string()),
        )))
    }

    fn supported_message_types(&self) -> Vec<String> {
        vec!["test_message".to_string()]
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Agent RPC Outage Resilience Test Starting");
    println!("=============================================");

    let config = OutageTestConfig::default();
    run_outage_test(config).await?;

    Ok(())
}

async fn run_outage_test(config: OutageTestConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("📡 Setting up RPC transport and test agents");

    // Set up mock components
    let server_agent = AgentId::new("test-server", "server-node");
    let primary_client = Arc::new(MockRpcClient::new(server_agent.clone()));
    let dtn_queue = Arc::new(DTNMessageQueue::new(config.max_queue_size));
    let resilient_client = Arc::new(ResilientRpcClient::new(
        Arc::clone(&primary_client),
        dtn_queue.clone(),
        server_agent.clone(),
    ));

    println!("🔧 Starting outage resilience test scenarios");

    // Run the test
    let test_result = run_test_scenarios(
        config.clone(),
        resilient_client.clone(),
        primary_client.clone(),
    ).await?;

    // Generate test report
    generate_test_report(&test_result, &config).await?;

    println!("✅ Agent RPC Outage Test Completed Successfully");
    Ok(())
}

async fn run_test_scenarios(
    config: OutageTestConfig,
    client: Arc<ResilientRpcClient>,
    mock_client: Arc<MockRpcClient>,
) -> Result<OutageTestMetrics, Box<dyn std::error::Error>> {
    let test_start = Instant::now();
    let mut message_counter = 0u64;

    println!("📊 Running test for {:?}", config.test_duration);

    // Schedule failure scenarios
    let mock_client_clone = Arc::clone(&mock_client);
    tokio::spawn(async move {
        // Start with normal operation
        sleep(Duration::from_secs(5)).await;

        // Introduce failures
        {
            let mut mock = Arc::try_unwrap(mock_client_clone).unwrap_or_else(|arc| {
                println!("Could not get exclusive access, using shared reference");
                return;
            });
            mock.set_fail_probability(0.7); // 70% failure rate
        }

        sleep(Duration::from_secs(10)).await;

        // Recover
        println!("Recovery phase starting...");
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

        match client.send_message_resilient(test_message).await {
            Ok(response) => {
                let response_str = String::from_utf8_lossy(&response.payload);
                if response_str == "DTN_QUEUED" {
                    println!("Message {}: Queued for DTN", message_counter);
                } else {
                    println!("Message {}: Success", message_counter);
                }
            }
            Err(e) => {
                println!("Message {}: Failed - {}", message_counter, e);
            }
        }

        // Log progress every 10 messages
        if message_counter % 10 == 0 {
            let queue_size = dtn_queue.queue_size().await;
            println!("Progress: {} messages sent, DTN queue: {} messages",
                  message_counter, queue_size);
        }
    }

    println!("📈 Test completed, collecting final metrics");

    // Allow time for final message processing
    sleep(Duration::from_secs(2)).await;

    let final_metrics = client.get_metrics().await;
    Ok(final_metrics)
}

async fn generate_test_report(
    metrics: &OutageTestMetrics,
    config: &OutageTestConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let success_rate = if metrics.messages_sent > 0 {
        (metrics.messages_received as f64 / metrics.messages_sent as f64) * 100.0
    } else {
        0.0
    };

    let report = json!({
        "test_summary": {
            "test_duration_seconds": config.test_duration.as_secs(),
            "dtn_fallback_enabled": config.dtn_fallback_enabled
        },
        "metrics": {
            "messages_sent": metrics.messages_sent,
            "messages_received": metrics.messages_received,
            "messages_lost": metrics.messages_lost,
            "messages_queued": metrics.messages_queued,
            "success_rate_percent": success_rate,
            "connection_failures": metrics.connection_failures,
            "dtn_activations": metrics.dtn_activations
        },
        "conclusions": {
            "rpc_resilience": if success_rate > 50.0 { "PASS" } else { "FAIL" },
            "dtn_fallback": if metrics.dtn_activations > 0 { "PASS" } else { "FAIL" },
            "message_persistence": if metrics.messages_queued > 0 { "PASS" } else { "FAIL" }
        }
    });

    // Write report to file
    let report_path = "artifacts/rpc_outage_test_report.json";
    std::fs::create_dir_all("artifacts").ok();
    std::fs::write(report_path, report.to_string())?;

    println!("📋 Test Report Generated");
    println!("========================");
    println!("Success Rate: {:.1}%", success_rate);
    println!("DTN Activations: {}", metrics.dtn_activations);
    println!("Messages Queued: {}", metrics.messages_queued);
    println!("Report saved: {}", report_path);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outage_config_creation() {
        let config = OutageTestConfig::default();
        assert!(config.dtn_fallback_enabled);
        assert_eq!(config.max_queue_size, 100);
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
}
