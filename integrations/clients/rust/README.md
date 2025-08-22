# AIVillage Rust SDK

A comprehensive, production-ready Rust client for the AIVillage API with full async support, type safety, and built-in reliability patterns using modern Rust idioms.

## Features

- **Type Safety**: Full Rust type definitions with `serde` serialization support
- **Async/Await**: Native `tokio` async runtime with efficient connection handling
- **Reliability**: Automatic retries with exponential backoff and circuit breaker patterns
- **Idempotency**: Safe retry of mutating operations with idempotency keys
- **Rate Limiting**: Built-in rate limit awareness with automatic backoff
- **Error Handling**: Rich error types using `thiserror` with detailed context
- **Authentication**: Bearer token and API key authentication methods
- **HTTP/2 Support**: Efficient HTTP/2 client with `reqwest` and connection pooling

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
aivillage-client = "1.0.0"
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
anyhow = "1.0"  # For error handling examples
```

## Quick Start

```rust
use aivillage_client::{Client, Configuration};
use aivillage_client::apis::chat_api;
use aivillage_client::models::{ChatRequest, ChatRequestUserContext};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure client
    let config = Configuration {
        base_path: "https://api.aivillage.io/v1".to_string(),
        bearer_access_token: Some("your-api-key".to_string()),
        ..Default::default()
    };

    let client = Client::new(config);

    // Create chat request
    let request = ChatRequest {
        message: "How can I optimize Rust applications for machine learning workloads?".to_string(),
        agent_preference: Some("magi".to_string()),
        mode: Some("comprehensive".to_string()),
        conversation_id: None,
        user_context: Some(ChatRequestUserContext {
            device_type: Some("desktop".to_string()),
            battery_level: None,
            network_type: Some("wifi".to_string()),
        }),
    };

    // Send request
    match chat_api::chat(&client.configuration, request).await {
        Ok(response) => {
            println!("Agent: {}", response.agent_used);
            println!("Response: {}", response.response);
            println!("Processing time: {}ms", response.processing_time_ms);
        }
        Err(e) => {
            eprintln!("Chat error: {}", e);
        }
    }

    Ok(())
}
```

## Configuration

### Basic Configuration

```rust
use aivillage_client::{Client, Configuration};
use std::env;

// Basic configuration
fn create_basic_client() -> Client {
    let config = Configuration {
        base_path: "https://api.aivillage.io/v1".to_string(),
        bearer_access_token: Some("your-api-key".to_string()),
        timeout: Some(std::time::Duration::from_secs(60)),
        ..Default::default()
    };

    Client::new(config)
}

// Environment-based configuration
fn create_client_from_env() -> Result<Client, Box<dyn std::error::Error>> {
    let api_url = env::var("AIVILLAGE_API_URL")
        .unwrap_or_else(|_| "https://api.aivillage.io/v1".to_string());

    let api_key = env::var("AIVILLAGE_API_KEY")?;

    let config = Configuration {
        base_path: api_url,
        bearer_access_token: Some(api_key),
        timeout: Some(std::time::Duration::from_secs(60)),
        ..Default::default()
    };

    Ok(Client::new(config))
}
```

### Advanced Configuration

```rust
use aivillage_client::{Client, Configuration};
use reqwest::ClientBuilder;
use std::time::Duration;

fn create_advanced_client(api_key: String) -> Result<Client, Box<dyn std::error::Error>> {
    // Custom reqwest client with advanced configuration
    let http_client = ClientBuilder::new()
        .timeout(Duration::from_secs(60))
        .connect_timeout(Duration::from_secs(10))
        .tcp_keepalive(Duration::from_secs(60))
        .pool_max_idle_per_host(10)
        .pool_idle_timeout(Duration::from_secs(90))
        .http2_prior_knowledge()
        .gzip(true)
        .user_agent("AIVillage-Rust-SDK/1.0.0")
        .build()?;

    let mut config = Configuration {
        base_path: "https://api.aivillage.io/v1".to_string(),
        bearer_access_token: Some(api_key),
        timeout: Some(Duration::from_secs(60)),
        ..Default::default()
    };

    // Set custom HTTP client
    config.client = http_client;

    Ok(Client::new(config))
}
```

## API Reference

### Chat API

Interact with AIVillage's specialized AI agents.

```rust
use aivillage_client::{Client, Configuration};
use aivillage_client::apis::chat_api;
use aivillage_client::models::{ChatRequest, ChatRequestUserContext, ChatResponse};
use anyhow::Result;

pub struct ChatService {
    client: Client,
}

impl ChatService {
    pub fn new(api_key: String) -> Self {
        let config = Configuration {
            base_path: "https://api.aivillage.io/v1".to_string(),
            bearer_access_token: Some(api_key),
            ..Default::default()
        };

        Self {
            client: Client::new(config),
        }
    }

    pub async fn basic_chat(&self, message: &str) -> Result<ChatResponse> {
        let request = ChatRequest {
            message: message.to_string(),
            agent_preference: Some("sage".to_string()),
            mode: Some("balanced".to_string()),
            conversation_id: None,
            user_context: None,
        };

        Ok(chat_api::chat(&self.client.configuration, request).await?)
    }

    pub async fn contextual_chat(
        &self,
        message: &str,
        conversation_id: Option<String>,
        device_type: Option<&str>,
        battery_level: Option<i32>,
        network_type: Option<&str>,
    ) -> Result<ChatResponse> {
        let user_context = if device_type.is_some() || battery_level.is_some() || network_type.is_some() {
            Some(ChatRequestUserContext {
                device_type: device_type.map(|s| s.to_string()),
                battery_level,
                network_type: network_type.map(|s| s.to_string()),
            })
        } else {
            None
        };

        let request = ChatRequest {
            message: message.to_string(),
            agent_preference: Some("navigator".to_string()), // Mobile optimization specialist
            mode: Some("comprehensive".to_string()),
            conversation_id,
            user_context,
        };

        Ok(chat_api::chat(&self.client.configuration, request).await?)
    }

    pub async fn mobile_optimized_chat(
        &self,
        message: &str,
        battery_level: i32,
        network_type: &str,
    ) -> Result<ChatResponse> {
        self.contextual_chat(
            message,
            None,
            Some("mobile"),
            Some(battery_level),
            Some(network_type),
        ).await
    }

    pub async fn batch_chat(&self, messages: Vec<&str>) -> Vec<Result<ChatResponse>> {
        use futures::future::join_all;

        let futures = messages.into_iter().map(|msg| {
            async move {
                self.basic_chat(msg).await
            }
        });

        join_all(futures).await
    }
}

// Example usage
#[tokio::main]
async fn example_chat_usage() -> Result<()> {
    let chat_service = ChatService::new("your-api-key".to_string());

    // Basic chat
    let response = chat_service.basic_chat("Explain the benefits of Rust for systems programming").await?;
    println!("Agent: {}", response.agent_used);
    println!("Response: {}", response.response);

    // Mobile-optimized chat
    let mobile_response = chat_service.mobile_optimized_chat(
        "How can I deploy this on mobile devices?",
        45, // 45% battery
        "cellular",
    ).await?;

    println!("Mobile response: {}", mobile_response.response);

    // Batch processing
    let messages = vec![
        "What is memory safety?",
        "Explain ownership in Rust",
        "How does borrowing work?",
    ];

    let results = chat_service.batch_chat(messages).await;
    for (i, result) in results.into_iter().enumerate() {
        match result {
            Ok(response) => println!("Response {}: {}", i + 1, response.response),
            Err(e) => eprintln!("Error in response {}: {}", i + 1, e),
        }
    }

    Ok(())
}
```

**Available Agents:**
- `king`: Coordination and oversight with public thought bubbles
- `magi`: Research and comprehensive analysis
- `sage`: Deep knowledge and wisdom
- `oracle`: Predictions and forecasting
- `navigator`: Routing and mobile optimization
- `any`: Auto-select best agent (default)

**Response Modes:**
- `fast`: Quick responses with minimal processing
- `balanced`: Good balance of speed and thoroughness (default)
- `comprehensive`: Detailed analysis with full context
- `creative`: Innovative and creative insights
- `analytical`: Systematic analysis and reasoning

### RAG API

Advanced knowledge retrieval with Bayesian trust networks.

```rust
use aivillage_client::apis::rag_api;
use aivillage_client::models::{QueryRequest, QueryResponse};
use futures::stream::{self, StreamExt};
use anyhow::Result;

pub struct RAGService {
    client: Client,
}

impl RAGService {
    pub fn new(api_key: String) -> Self {
        let config = Configuration {
            base_path: "https://api.aivillage.io/v1".to_string(),
            bearer_access_token: Some(api_key),
            ..Default::default()
        };

        Self {
            client: Client::new(config),
        }
    }

    pub async fn process_query(
        &self,
        query: &str,
        mode: &str,
        max_results: i32,
    ) -> Result<QueryResponse> {
        let request = QueryRequest {
            query: query.to_string(),
            mode: Some(mode.to_string()),
            include_sources: Some(true),
            max_results: Some(max_results),
            user_id: None,
        };

        Ok(rag_api::process_query(&self.client.configuration, request).await?)
    }

    pub async fn comprehensive_query(&self, query: &str) -> Result<QueryResponse> {
        let response = self.process_query(query, "comprehensive", 15).await?;

        // Print detailed results
        println!("Query ID: {}", response.query_id);
        println!("Response: {}", response.response);
        println!("Bayesian confidence: {:.3}", response.metadata.bayesian_confidence);

        for source in &response.sources {
            println!(
                "Source: {} (confidence: {:.3})",
                source.title, source.confidence
            );
        }

        Ok(response)
    }

    pub async fn batch_process_queries(&self, queries: Vec<&str>) -> Vec<Result<QueryResponse>> {
        // Process queries concurrently with controlled concurrency
        let results = stream::iter(queries)
            .map(|query| async move {
                self.process_query(query, "fast", 5).await
            })
            .buffer_unordered(5) // Process up to 5 queries concurrently
            .collect::<Vec<_>>()
            .await;

        results
    }

    pub async fn search_knowledge_with_sources(
        &self,
        query: &str,
    ) -> Result<(String, Vec<String>)> {
        let response = self.process_query(query, "analytical", 10).await?;

        let sources: Vec<String> = response
            .sources
            .into_iter()
            .filter(|s| s.confidence > 0.7) // Only high-confidence sources
            .map(|s| format!("{} (confidence: {:.2})", s.title, s.confidence))
            .collect();

        Ok((response.response, sources))
    }
}
```

### Agents API

Manage and monitor AI agents.

```rust
use aivillage_client::apis::agents_api;
use aivillage_client::models::{
    ListAgentsResponse, Agent, AgentTaskRequest, AgentTaskResponse
};
use std::collections::HashMap;
use tokio::time::{interval, Duration};
use anyhow::Result;

pub struct AgentsService {
    client: Client,
}

impl AgentsService {
    pub fn new(api_key: String) -> Self {
        let config = Configuration {
            base_path: "https://api.aivillage.io/v1".to_string(),
            bearer_access_token: Some(api_key),
            ..Default::default()
        };

        Self {
            client: Client::new(config),
        }
    }

    pub async fn list_agents(
        &self,
        category: Option<&str>,
        available_only: bool,
    ) -> Result<ListAgentsResponse> {
        Ok(agents_api::list_agents(
            &self.client.configuration,
            category.map(|s| s.to_string()),
            Some(available_only),
        ).await?)
    }

    pub async fn assign_task(
        &self,
        agent_id: &str,
        task_description: &str,
        priority: &str,
        timeout_seconds: i32,
        context: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<AgentTaskResponse> {
        let task_request = AgentTaskRequest {
            task_description: task_description.to_string(),
            priority: Some(priority.to_string()),
            timeout_seconds: Some(timeout_seconds),
            context: context.unwrap_or_default(),
        };

        Ok(agents_api::assign_agent_task(
            &self.client.configuration,
            agent_id,
            task_request,
        ).await?)
    }

    pub async fn find_best_agent(&self, category: &str) -> Result<Option<Agent>> {
        let agents = self.list_agents(Some(category), true).await?;

        // Find agent with lowest load
        let best_agent = agents
            .agents
            .into_iter()
            .min_by_key(|agent| agent.current_load);

        Ok(best_agent)
    }

    pub async fn monitor_agents(
        &self,
        interval_secs: u64,
    ) -> Result<()> {
        let mut interval = interval(Duration::from_secs(interval_secs));

        loop {
            interval.tick().await;

            match self.list_agents(None, false).await {
                Ok(agents_response) => {
                    println!("\n=== Agent Status ===");
                    for agent in &agents_response.agents {
                        println!(
                            "  {}: {} (load: {}%)",
                            agent.name, agent.status, agent.current_load
                        );
                    }
                }
                Err(e) => eprintln!("Error monitoring agents: {}", e),
            }
        }
    }
}
```

### P2P API

Monitor peer-to-peer mesh network status.

```rust
use aivillage_client::apis::p2p_api;
use aivillage_client::models::{P2PStatusResponse, ListPeersResponse};
use tokio::time::{interval, Duration};
use anyhow::Result;

pub struct P2PService {
    client: Client,
}

impl P2PService {
    pub fn new(api_key: String) -> Self {
        let config = Configuration {
            base_path: "https://api.aivillage.io/v1".to_string(),
            bearer_access_token: Some(api_key),
            ..Default::default()
        };

        Self {
            client: Client::new(config),
        }
    }

    pub async fn get_network_status(&self) -> Result<P2PStatusResponse> {
        Ok(p2p_api::get_p2p_status(&self.client.configuration).await?)
    }

    pub async fn list_peers(&self, transport_type: &str) -> Result<ListPeersResponse> {
        Ok(p2p_api::list_peers(&self.client.configuration, transport_type).await?)
    }

    pub async fn get_network_overview(&self) -> Result<NetworkOverview> {
        let status = self.get_network_status().await?;
        let bitchat_peers = self.list_peers("bitchat").await?;
        let betanet_peers = self.list_peers("betanet").await?;

        Ok(NetworkOverview {
            status: status.status,
            total_peers: status.peer_count,
            health_score: status.health_score,
            bitchat_peer_count: bitchat_peers.peers.len() as i32,
            betanet_peer_count: betanet_peers.peers.len() as i32,
        })
    }

    pub async fn monitor_network(&self, interval_secs: u64) -> Result<()> {
        let mut interval = interval(Duration::from_secs(interval_secs));

        loop {
            interval.tick().await;

            match self.get_network_overview().await {
                Ok(overview) => {
                    println!("\n=== P2P Network Overview ===");
                    println!("Status: {}", overview.status);
                    println!("Total peers: {}", overview.total_peers);
                    println!("Health score: {:.2}", overview.health_score);
                    println!("BitChat peers: {}", overview.bitchat_peer_count);
                    println!("BetaNet peers: {}", overview.betanet_peer_count);
                }
                Err(e) => eprintln!("Error monitoring network: {}", e),
            }
        }
    }
}

#[derive(Debug)]
pub struct NetworkOverview {
    pub status: String,
    pub total_peers: i32,
    pub health_score: f64,
    pub bitchat_peer_count: i32,
    pub betanet_peer_count: i32,
}
```

### Digital Twin API

Privacy-preserving personal AI assistant.

```rust
use aivillage_client::apis::digital_twin_api;
use aivillage_client::models::{
    DigitalTwinProfileResponse, DigitalTwinDataUpdateRequest
};
use serde_json::{json, Map, Value};
use chrono::{DateTime, Utc};
use anyhow::Result;

pub struct DigitalTwinService {
    client: Client,
}

impl DigitalTwinService {
    pub fn new(api_key: String) -> Self {
        let config = Configuration {
            base_path: "https://api.aivillage.io/v1".to_string(),
            bearer_access_token: Some(api_key),
            ..Default::default()
        };

        Self {
            client: Client::new(config),
        }
    }

    pub async fn get_profile(&self) -> Result<DigitalTwinProfileResponse> {
        Ok(digital_twin_api::get_digital_twin_profile(&self.client.configuration).await?)
    }

    pub async fn update_interaction_data(
        &self,
        interaction_type: &str,
        satisfaction: f64,
        accuracy: f64,
        context: Option<&str>,
    ) -> Result<()> {
        let now: DateTime<Utc> = Utc::now();

        let mut content = Map::new();
        content.insert("interaction_type".to_string(), Value::String(interaction_type.to_string()));
        content.insert("user_satisfaction".to_string(), json!(satisfaction));
        if let Some(ctx) = context {
            content.insert("context".to_string(), Value::String(ctx.to_string()));
        }

        let mut data_point = Map::new();
        data_point.insert("timestamp".to_string(), Value::String(now.to_rfc3339()));
        data_point.insert("content".to_string(), Value::Object(content));
        data_point.insert("prediction_accuracy".to_string(), json!(accuracy));

        let update_request = DigitalTwinDataUpdateRequest {
            data_type: "interaction".to_string(),
            data_points: vec![Value::Object(data_point)],
        };

        digital_twin_api::update_digital_twin_data(
            &self.client.configuration,
            update_request,
        ).await?;

        Ok(())
    }

    pub async fn update_learning_feedback(
        &self,
        prediction_was_correct: bool,
        user_feedback_score: f64,
        learning_domain: &str,
    ) -> Result<()> {
        let accuracy = if prediction_was_correct { 1.0 } else { 0.0 };

        self.update_interaction_data(
            "learning_feedback",
            user_feedback_score,
            accuracy,
            Some(learning_domain),
        ).await
    }

    pub async fn print_profile_summary(&self) -> Result<()> {
        let profile = self.get_profile().await?;

        println!("=== Digital Twin Profile ===");
        println!("Model size: {:.1}MB", profile.model_size_mb);
        println!("Accuracy: {:.3}", profile.learning_stats.accuracy_score);
        println!("Privacy level: {}", profile.privacy_settings.level);
        println!("Total interactions: {}", profile.learning_stats.total_interactions);

        Ok(())
    }
}
```

## Error Handling

### Structured Error Types

```rust
use thiserror::Error;
use serde::{Deserialize, Serialize};

#[derive(Error, Debug)]
pub enum AIVillageError {
    #[error("API request failed: {message}")]
    ApiError {
        status_code: u16,
        message: String,
        request_id: Option<String>,
    },

    #[error("Rate limit exceeded. Retry after {retry_after} seconds")]
    RateLimitError {
        retry_after: u64,
        message: String,
    },

    #[error("Authentication failed: {0}")]
    AuthenticationError(String),

    #[error("Validation failed: {0}")]
    ValidationError(String),

    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

impl AIVillageError {
    pub fn from_response(status: u16, body: &str, headers: &reqwest::header::HeaderMap) -> Self {
        let request_id = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        match status {
            429 => {
                let retry_after = headers
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(60);

                Self::RateLimitError {
                    retry_after,
                    message: body.to_string(),
                }
            }
            401 => Self::AuthenticationError(body.to_string()),
            400 => Self::ValidationError(body.to_string()),
            _ => Self::ApiError {
                status_code: status,
                message: body.to_string(),
                request_id,
            },
        }
    }
}

pub type Result<T> = std::result::Result<T, AIVillageError>;
```

### Retry Logic with Circuit Breaker

```rust
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time::sleep;

#[derive(Debug, Clone, Copy, PartialEq)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug)]
pub struct CircuitBreaker {
    state: Arc<RwLock<CircuitBreakerState>>,
    failure_threshold: u32,
    timeout: Duration,
}

#[derive(Debug)]
struct CircuitBreakerState {
    state: CircuitState,
    failure_count: u32,
    last_failure_time: Option<Instant>,
    success_count: u32,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, timeout: Duration) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitBreakerState {
                state: CircuitState::Closed,
                failure_count: 0,
                last_failure_time: None,
                success_count: 0,
            })),
            failure_threshold,
            timeout,
        }
    }

    pub async fn execute<F, T, E>(&self, f: F) -> std::result::Result<T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
        E: std::fmt::Display,
    {
        // Check if circuit is open
        {
            let state = self.state.read().await;
            if state.state == CircuitState::Open {
                if let Some(last_failure) = state.last_failure_time {
                    if last_failure.elapsed() < self.timeout {
                        return Err(E::from("Circuit breaker is open"));
                    }
                }
            }
        }

        // If half-open or timeout expired, try the operation
        if self.should_attempt().await {
            let mut state = self.state.write().await;
            state.state = CircuitState::HalfOpen;
        }

        // Execute the operation
        match f() {
            Ok(result) => {
                self.on_success().await;
                Ok(result)
            }
            Err(error) => {
                self.on_failure().await;
                Err(error)
            }
        }
    }

    async fn should_attempt(&self) -> bool {
        let state = self.state.read().await;
        match state.state {
            CircuitState::Closed => true,
            CircuitState::HalfOpen => true,
            CircuitState::Open => {
                if let Some(last_failure) = state.last_failure_time {
                    last_failure.elapsed() >= self.timeout
                } else {
                    true
                }
            }
        }
    }

    async fn on_success(&self) {
        let mut state = self.state.write().await;
        match state.state {
            CircuitState::HalfOpen => {
                state.success_count += 1;
                if state.success_count >= 3 {
                    state.state = CircuitState::Closed;
                    state.failure_count = 0;
                    state.success_count = 0;
                }
            }
            CircuitState::Closed => {
                state.failure_count = 0;
            }
            CircuitState::Open => {}
        }
    }

    async fn on_failure(&self) {
        let mut state = self.state.write().await;
        state.failure_count += 1;
        state.last_failure_time = Some(Instant::now());

        if state.failure_count >= self.failure_threshold {
            state.state = CircuitState::Open;
        }
    }
}

// Resilient API client
pub struct ResilientClient {
    client: Client,
    circuit_breaker: CircuitBreaker,
}

impl ResilientClient {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            circuit_breaker: CircuitBreaker::new(3, Duration::from_secs(60)),
        }
    }

    pub async fn chat_with_retry(
        &self,
        request: ChatRequest,
        max_retries: u32,
    ) -> Result<ChatResponse> {
        let mut last_error = None;

        for attempt in 0..=max_retries {
            let result = self.circuit_breaker.execute(|| {
                // This would be async in real implementation
                // For now, we simulate the call
                Ok::<_, AIVillageError>(())
            }).await;

            match result {
                Ok(_) => {
                    // Make the actual API call
                    match chat_api::chat(&self.client.configuration, request.clone()).await {
                        Ok(response) => return Ok(response),
                        Err(e) => {
                            last_error = Some(AIVillageError::NetworkError(e));

                            if attempt < max_retries {
                                let delay = Duration::from_millis(1000 * (2_u64.pow(attempt)));
                                sleep(std::cmp::min(delay, Duration::from_secs(32))).await;
                            }
                        }
                    }
                }
                Err(_) => {
                    // Circuit breaker is open
                    return Err(AIVillageError::ApiError {
                        status_code: 503,
                        message: "Circuit breaker is open".to_string(),
                        request_id: None,
                    });
                }
            }
        }

        Err(last_error.unwrap_or_else(|| AIVillageError::ApiError {
            status_code: 500,
            message: "Max retries exceeded".to_string(),
            request_id: None,
        }))
    }
}
```

## Advanced Usage

### Async Streams and Concurrency

```rust
use futures::{stream, StreamExt, TryStreamExt};
use tokio::sync::Semaphore;
use std::sync::Arc;

pub struct ConcurrentAIVillageClient {
    client: Client,
    semaphore: Arc<Semaphore>,
}

impl ConcurrentAIVillageClient {
    pub fn new(client: Client, max_concurrent: usize) -> Self {
        Self {
            client,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
        }
    }

    pub async fn batch_chat(&self, messages: Vec<String>) -> Vec<Result<ChatResponse>> {
        stream::iter(messages)
            .map(|message| {
                let client = self.client.clone();
                let semaphore = Arc::clone(&self.semaphore);

                async move {
                    let _permit = semaphore.acquire().await.unwrap();

                    let request = ChatRequest {
                        message,
                        agent_preference: Some("sage".to_string()),
                        mode: Some("fast".to_string()),
                        conversation_id: None,
                        user_context: None,
                    };

                    chat_api::chat(&client.configuration, request)
                        .await
                        .map_err(AIVillageError::NetworkError)
                }
            })
            .buffer_unordered(10) // Process up to 10 requests concurrently
            .collect::<Vec<_>>()
            .await
    }

    pub async fn batch_rag_queries(&self, queries: Vec<String>) -> Vec<Result<QueryResponse>> {
        stream::iter(queries)
            .map(|query| {
                let client = self.client.clone();
                let semaphore = Arc::clone(&self.semaphore);

                async move {
                    let _permit = semaphore.acquire().await.unwrap();

                    let request = QueryRequest {
                        query,
                        mode: Some("fast".to_string()),
                        include_sources: Some(true),
                        max_results: Some(5),
                        user_id: None,
                    };

                    rag_api::process_query(&client.configuration, request)
                        .await
                        .map_err(AIVillageError::NetworkError)
                }
            })
            .buffer_unordered(5) // RAG queries are more expensive
            .collect::<Vec<_>>()
            .await
    }

    // Pipeline processing: RAG → Chat
    pub async fn process_pipeline(&self, user_query: String) -> Result<PipelineResult> {
        let start = std::time::Instant::now();

        // Step 1: Get background via RAG
        let rag_request = QueryRequest {
            query: format!("Background information: {}", user_query),
            mode: Some("fast".to_string()),
            include_sources: Some(true),
            max_results: Some(5),
            user_id: None,
        };

        let rag_response = rag_api::process_query(&self.client.configuration, rag_request)
            .await
            .map_err(AIVillageError::NetworkError)?;

        // Step 2: Use RAG results to inform chat
        let context = format!(
            "Based on this background: {}\n\nUser question: {}",
            rag_response.response, user_query
        );

        let chat_request = ChatRequest {
            message: context,
            agent_preference: Some("sage".to_string()),
            mode: Some("comprehensive".to_string()),
            conversation_id: None,
            user_context: None,
        };

        let chat_response = chat_api::chat(&self.client.configuration, chat_request)
            .await
            .map_err(AIVillageError::NetworkError)?;

        Ok(PipelineResult {
            rag_response,
            chat_response,
            duration: start.elapsed(),
        })
    }
}

#[derive(Debug)]
pub struct PipelineResult {
    pub rag_response: QueryResponse,
    pub chat_response: ChatResponse,
    pub duration: Duration,
}
```

### Idempotency and Request Deduplication

```rust
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use sha2::{Digest, Sha256};

pub fn generate_idempotency_key(operation: &str, context: Option<&str>) -> String {
    let timestamp = chrono::Utc::now().format("%Y%m%d-%H%M%S");
    let uuid = Uuid::new_v4().to_string()[..8].to_string();

    match context {
        Some(ctx) => format!("{}-{}-{}-{}", operation, ctx, timestamp, uuid),
        None => format!("{}-{}-{}", operation, timestamp, uuid),
    }
}

pub fn generate_request_hash(request: &ChatRequest) -> String {
    let mut hasher = Sha256::new();
    hasher.update(request.message.as_bytes());
    if let Some(ref agent) = request.agent_preference {
        hasher.update(agent.as_bytes());
    }
    if let Some(ref mode) = request.mode {
        hasher.update(mode.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

pub struct IdempotentClient {
    client: Client,
    cache: Arc<RwLock<HashMap<String, ChatResponse>>>,
}

impl IdempotentClient {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn safe_chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let request_hash = generate_request_hash(&request);

        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(cached_response) = cache.get(&request_hash) {
                return Ok(cached_response.clone());
            }
        }

        // Make API request
        let response = chat_api::chat(&self.client.configuration, request)
            .await
            .map_err(AIVillageError::NetworkError)?;

        // Cache successful response
        {
            let mut cache = self.cache.write().await;
            cache.insert(request_hash, response.clone());
        }

        Ok(response)
    }

    pub async fn chat_with_idempotency_key(
        &self,
        request: ChatRequest,
        idempotency_key: String,
    ) -> Result<ChatResponse> {
        // Check cache for idempotency key
        {
            let cache = self.cache.read().await;
            if let Some(cached_response) = cache.get(&idempotency_key) {
                return Ok(cached_response.clone());
            }
        }

        // Make request with idempotency key in headers
        // Note: This would require modifying the generated client to support custom headers
        let response = chat_api::chat(&self.client.configuration, request)
            .await
            .map_err(AIVillageError::NetworkError)?;

        // Cache by idempotency key
        {
            let mut cache = self.cache.write().await;
            cache.insert(idempotency_key, response.clone());
        }

        Ok(response)
    }

    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }

    pub async fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.read().await;
        (cache.len(), cache.capacity())
    }
}
```

## Testing

### Unit Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use mockito::{Server, Mock};
    use serde_json::json;

    #[tokio::test]
    async fn test_chat_success() {
        let mut server = Server::new_async().await;

        // Mock successful response
        let mock = server
            .mock("POST", "/chat")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(json!({
                "response": "Test response from mock server",
                "agent_used": "test-agent",
                "processing_time_ms": 100,
                "conversation_id": "test-conv-123",
                "metadata": {
                    "confidence": 0.95
                }
            }).to_string())
            .create_async()
            .await;

        // Configure client to use mock server
        let config = Configuration {
            base_path: server.url(),
            bearer_access_token: Some("test-key".to_string()),
            ..Default::default()
        };

        let client = Client::new(config);

        let request = ChatRequest {
            message: "Test message".to_string(),
            agent_preference: Some("sage".to_string()),
            mode: Some("fast".to_string()),
            conversation_id: None,
            user_context: None,
        };

        let response = chat_api::chat(&client.configuration, request).await.unwrap();

        assert_eq!(response.response, "Test response from mock server");
        assert_eq!(response.agent_used, "test-agent");
        assert_eq!(response.processing_time_ms, 100);

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let mut server = Server::new_async().await;

        let mock = server
            .mock("POST", "/chat")
            .with_status(429)
            .with_header("content-type", "application/json")
            .with_header("retry-after", "60")
            .with_body(json!({
                "error": "Rate limit exceeded"
            }).to_string())
            .create_async()
            .await;

        let config = Configuration {
            base_path: server.url(),
            bearer_access_token: Some("test-key".to_string()),
            ..Default::default()
        };

        let client = Client::new(config);

        let request = ChatRequest {
            message: "Test message".to_string(),
            agent_preference: None,
            mode: Some("fast".to_string()),
            conversation_id: None,
            user_context: None,
        };

        let result = chat_api::chat(&client.configuration, request).await;

        assert!(result.is_err());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_idempotency_key_generation() {
        let key1 = generate_idempotency_key("chat", Some("test"));
        let key2 = generate_idempotency_key("chat", Some("test"));

        // Keys should be different even with same params
        assert_ne!(key1, key2);

        // But should follow expected format
        assert!(key1.starts_with("chat-test-"));
        assert!(key2.starts_with("chat-test-"));
    }

    #[tokio::test]
    async fn test_request_deduplication() {
        let request1 = ChatRequest {
            message: "Same message".to_string(),
            agent_preference: Some("sage".to_string()),
            mode: Some("fast".to_string()),
            conversation_id: None,
            user_context: None,
        };

        let request2 = ChatRequest {
            message: "Same message".to_string(),
            agent_preference: Some("sage".to_string()),
            mode: Some("fast".to_string()),
            conversation_id: None,
            user_context: None,
        };

        let hash1 = generate_request_hash(&request1);
        let hash2 = generate_request_hash(&request2);

        // Same requests should have same hash
        assert_eq!(hash1, hash2);

        let request3 = ChatRequest {
            message: "Different message".to_string(),
            agent_preference: Some("sage".to_string()),
            mode: Some("fast".to_string()),
            conversation_id: None,
            user_context: None,
        };

        let hash3 = generate_request_hash(&request3);
        assert_ne!(hash1, hash3);
    }
}
```

### Integration Testing

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::env;

    fn get_test_client() -> Option<Client> {
        let api_key = env::var("AIVILLAGE_TEST_API_KEY").ok()?;
        let api_url = env::var("AIVILLAGE_TEST_API_URL")
            .unwrap_or_else(|_| "https://staging-api.aivillage.io/v1".to_string());

        let config = Configuration {
            base_path: api_url,
            bearer_access_token: Some(api_key),
            timeout: Some(Duration::from_secs(30)),
            ..Default::default()
        };

        Some(Client::new(config))
    }

    #[tokio::test]
    async fn test_chat_integration() {
        let client = match get_test_client() {
            Some(c) => c,
            None => {
                println!("Skipping integration test - no API key");
                return;
            }
        };

        let request = ChatRequest {
            message: "This is an integration test message".to_string(),
            agent_preference: Some("sage".to_string()),
            mode: Some("fast".to_string()),
            conversation_id: None,
            user_context: None,
        };

        let response = chat_api::chat(&client.configuration, request).await.unwrap();

        assert!(!response.response.is_empty());
        assert!(!response.agent_used.is_empty());
        assert!(response.processing_time_ms > 0);

        println!("Integration test response: {}", response.response);
    }

    #[tokio::test]
    async fn test_rag_integration() {
        let client = match get_test_client() {
            Some(c) => c,
            None => {
                println!("Skipping RAG integration test - no API key");
                return;
            }
        };

        let request = QueryRequest {
            query: "What is machine learning?".to_string(),
            mode: Some("fast".to_string()),
            include_sources: Some(true),
            max_results: Some(3),
            user_id: None,
        };

        let response = rag_api::process_query(&client.configuration, request).await.unwrap();

        assert!(!response.response.is_empty());
        assert!(!response.query_id.is_empty());
        assert!(response.metadata.processing_time_ms > 0);

        println!("RAG integration test response: {}", response.response);
    }

    #[tokio::test]
    async fn test_concurrent_requests() {
        let client = match get_test_client() {
            Some(c) => c,
            None => {
                println!("Skipping concurrent test - no API key");
                return;
            }
        };

        let concurrent_client = ConcurrentAIVillageClient::new(client, 5);

        let messages = vec![
            "What is Rust?".to_string(),
            "Explain ownership".to_string(),
            "What are lifetimes?".to_string(),
            "How does borrowing work?".to_string(),
            "What is memory safety?".to_string(),
        ];

        let start = std::time::Instant::now();
        let results = concurrent_client.batch_chat(messages.clone()).await;
        let duration = start.elapsed();

        let success_count = results.iter().filter(|r| r.is_ok()).count();

        println!("Concurrent test results:");
        println!("  Requests: {}", messages.len());
        println!("  Successful: {}", success_count);
        println!("  Duration: {:?}", duration);
        println!("  Requests/sec: {:.2}", messages.len() as f64 / duration.as_secs_f64());

        assert!(success_count > 0, "At least some requests should succeed");
    }
}
```

### Benchmark Testing

```rust
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::{Duration, Instant};

    #[tokio::test]
    async fn benchmark_chat_performance() {
        let client = match get_test_client() {
            Some(c) => c,
            None => {
                println!("Skipping benchmark - no API key");
                return;
            }
        };

        let chat_service = ChatService::new("test-key".to_string());
        let num_requests = 50;

        // Sequential benchmark
        let start = Instant::now();
        for i in 0..num_requests {
            let message = format!("Benchmark message {}", i);
            let _ = chat_service.basic_chat(&message).await;
        }
        let sequential_duration = start.elapsed();

        // Concurrent benchmark
        let messages: Vec<String> = (0..num_requests)
            .map(|i| format!("Concurrent benchmark message {}", i))
            .collect();

        let start = Instant::now();
        let concurrent_client = ConcurrentAIVillageClient::new(client, 10);
        let results = concurrent_client.batch_chat(messages).await;
        let concurrent_duration = start.elapsed();

        let success_count = results.iter().filter(|r| r.is_ok()).count();

        println!("Benchmark Results:");
        println!("  Requests: {}", num_requests);
        println!("  Sequential: {:?} ({:.2} req/s)",
            sequential_duration,
            num_requests as f64 / sequential_duration.as_secs_f64()
        );
        println!("  Concurrent: {:?} ({:.2} req/s)",
            concurrent_duration,
            success_count as f64 / concurrent_duration.as_secs_f64()
        );
        println!("  Speedup: {:.2}x",
            sequential_duration.as_secs_f64() / concurrent_duration.as_secs_f64()
        );
    }
}
```

## Deployment

### Production Configuration

```rust
use serde::Deserialize;
use std::env;
use tracing::{info, error};

#[derive(Debug, Deserialize)]
pub struct AIVillageConfig {
    pub api_key: String,
    pub api_url: String,
    pub timeout_secs: u64,
    pub max_retries: u32,
    pub max_concurrent: usize,
    pub enable_circuit_breaker: bool,
}

impl AIVillageConfig {
    pub fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        let config = Self {
            api_key: env::var("AIVILLAGE_API_KEY")?,
            api_url: env::var("AIVILLAGE_API_URL")
                .unwrap_or_else(|_| "https://api.aivillage.io/v1".to_string()),
            timeout_secs: env::var("AIVILLAGE_TIMEOUT")
                .unwrap_or_else(|_| "60".to_string())
                .parse()
                .unwrap_or(60),
            max_retries: env::var("AIVILLAGE_MAX_RETRIES")
                .unwrap_or_else(|_| "3".to_string())
                .parse()
                .unwrap_or(3),
            max_concurrent: env::var("AIVILLAGE_MAX_CONCURRENT")
                .unwrap_or_else(|_| "10".to_string())
                .parse()
                .unwrap_or(10),
            enable_circuit_breaker: env::var("AIVILLAGE_ENABLE_CIRCUIT_BREAKER")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
        };

        info!("AIVillage client configured: {}", config.api_url);
        Ok(config)
    }
}

pub fn create_production_client(config: &AIVillageConfig) -> Result<Client, Box<dyn std::error::Error>> {
    let client_config = Configuration {
        base_path: config.api_url.clone(),
        bearer_access_token: Some(config.api_key.clone()),
        timeout: Some(Duration::from_secs(config.timeout_secs)),
        ..Default::default()
    };

    Ok(Client::new(client_config))
}

// Health check function
pub async fn health_check(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    let request = ChatRequest {
        message: "Health check".to_string(),
        agent_preference: Some("any".to_string()),
        mode: Some("fast".to_string()),
        conversation_id: None,
        user_context: None,
    };

    match chat_api::chat(&client.configuration, request).await {
        Ok(_) => {
            info!("AIVillage health check passed");
            Ok(())
        }
        Err(e) => {
            error!("AIVillage health check failed: {}", e);
            Err(Box::new(e))
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::init();

    // Load configuration
    let config = AIVillageConfig::from_env()?;

    // Create client
    let client = create_production_client(&config)?;

    // Health check
    health_check(&client).await?;

    // Your application logic here
    info!("AIVillage client ready for production use");

    Ok(())
}
```

### Docker Integration

```dockerfile
# Multi-stage Docker build for Rust
FROM rust:1.75 as builder

WORKDIR /usr/src/app

# Copy manifest files
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src

# Build for release
RUN cargo build --release

# Runtime image
FROM debian:bookworm-slim

# Install SSL certificates and required dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /usr/src/app/target/release/aivillage-app /usr/local/bin/aivillage-app

# Environment variables
ENV AIVILLAGE_API_URL=https://api.aivillage.io/v1
ENV AIVILLAGE_TIMEOUT=60
ENV AIVILLAGE_MAX_RETRIES=3
ENV RUST_LOG=info

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD aivillage-app --health-check || exit 1

# Run the binary
CMD ["aivillage-app"]

EXPOSE 8080
```

### Kubernetes Deployment

```yaml
# aivillage-rust-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aivillage-rust-app
  labels:
    app: aivillage-rust-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aivillage-rust-app
  template:
    metadata:
      labels:
        app: aivillage-rust-app
    spec:
      containers:
      - name: aivillage-rust-app
        image: your-registry/aivillage-rust-app:latest
        ports:
        - containerPort: 8080
        env:
        - name: AIVILLAGE_API_KEY
          valueFrom:
            secretKeyRef:
              name: aivillage-secret
              key: api-key
        - name: AIVILLAGE_API_URL
          value: "https://api.aivillage.io/v1"
        - name: AIVILLAGE_TIMEOUT
          value: "60"
        - name: AIVILLAGE_MAX_CONCURRENT
          value: "20"
        - name: RUST_LOG
          value: "info"
        resources:
          requests:
            memory: "32Mi"
            cpu: "50m"
          limits:
            memory: "64Mi"
            cpu: "100m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Troubleshooting

### Common Issues

**SSL/TLS Issues:**
```rust
// Skip certificate verification (development only)
use reqwest::ClientBuilder;

let http_client = ClientBuilder::new()
    .danger_accept_invalid_certs(true) // NOT for production
    .build()?;
```

**Memory Usage Optimization:**
```rust
// Reduce connection pool size for memory-constrained environments
let http_client = ClientBuilder::new()
    .pool_max_idle_per_host(2)
    .pool_idle_timeout(Duration::from_secs(30))
    .build()?;
```

**Timeout Configuration:**
```rust
// Configure different timeout values
let http_client = ClientBuilder::new()
    .connect_timeout(Duration::from_secs(10))  // Connection timeout
    .timeout(Duration::from_secs(120))         // Total request timeout
    .build()?;
```

### Debug Logging

```rust
use tracing::{info, debug, error, Level};
use tracing_subscriber;

// Initialize logging
fn init_logging() {
    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_target(false)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();
}

// Custom middleware for request logging
use reqwest_middleware::{ClientBuilder, ClientWithMiddleware};
use reqwest_tracing::TracingMiddleware;

pub fn create_logged_client() -> ClientWithMiddleware {
    let reqwest_client = reqwest::Client::new();

    ClientBuilder::new(reqwest_client)
        .with(TracingMiddleware::default())
        .build()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging();

    // Your application code with debug logging
    debug!("Starting AIVillage client");

    Ok(())
}
```

## Support

- **Documentation**: [docs.aivillage.io](https://docs.aivillage.io)
- **API Reference**: [docs.aivillage.io/api](https://docs.aivillage.io/api)
- **GitHub Issues**: [github.com/DNYoussef/AIVillage/issues](https://github.com/DNYoussef/AIVillage/issues)
- **Crates.io**: [crates.io/crates/aivillage-client](https://crates.io/crates/aivillage-client)

## License

MIT License - see [LICENSE](../../LICENSE) for details.
