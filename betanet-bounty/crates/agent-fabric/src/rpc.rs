//! RPC transport using betanet-htx streams
//!
//! Provides real-time agent messaging via HTX protocol streams.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot, Mutex, RwLock};
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

use betanet_htx::{
    accept_tcp, dial_tcp, HtxConfig, HtxError, HtxSession, HtxTcpConnection, Result as HtxResult,
    StreamId,
};

use crate::{
    AgentClient, AgentFabricError, AgentId, AgentMessage, AgentResponse, AgentServer, Result,
};

/// RPC message envelope for HTX transport
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RpcEnvelope {
    pub from: AgentId,
    pub to: AgentId,
    pub message_id: String,
    pub message_type: RpcMessageType,
    pub payload: Bytes,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum RpcMessageType {
    Request,
    Response,
    Notification,
}

/// Pending RPC request
struct PendingRequest {
    sender: oneshot::Sender<AgentResponse>,
    timeout: tokio::time::Instant,
}

/// RPC transport layer using HTX
#[derive(Clone)]
pub struct RpcTransport {
    config: HtxConfig,
    server_addr: SocketAddr,
    connections: Arc<RwLock<HashMap<AgentId, Arc<Mutex<HtxTcpConnection>>>>>,
    pending_requests: Arc<RwLock<HashMap<String, PendingRequest>>>,
    message_handlers: Arc<RwLock<HashMap<String, Box<dyn AgentServer>>>>,
    local_agent_id: AgentId,
}

impl RpcTransport {
    /// Create new RPC transport
    pub async fn new(config: HtxConfig) -> Result<Self> {
        let server_addr = config.listen_addr;
        let local_agent_id = AgentId::new(
            "rpc-transport",
            format!("{}:{}", server_addr.ip(), server_addr.port()),
        );

        Ok(Self {
            config,
            server_addr,
            connections: Arc::new(RwLock::new(HashMap::new())),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            message_handlers: Arc::new(RwLock::new(HashMap::new())),
            local_agent_id,
        })
    }

    /// Start RPC server
    pub async fn start_server(&self) -> Result<()> {
        let listener = TcpListener::bind(self.server_addr)
            .await
            .map_err(|e| AgentFabricError::NetworkError(e.to_string()))?;

        info!("RPC server listening on {}", self.server_addr);

        let connections = Arc::clone(&self.connections);
        let pending_requests = Arc::clone(&self.pending_requests);
        let message_handlers = Arc::clone(&self.message_handlers);
        let config = self.config.clone();

        tokio::spawn(async move {
            loop {
                match accept_tcp(&listener, config.clone()).await {
                    Ok(mut connection) => {
                        info!("Accepted RPC connection");

                        // Perform handshake
                        if let Err(e) = connection.handshake().await {
                            error!("RPC handshake failed: {}", e);
                            continue;
                        }

                        // Start connection handler
                        let conn_connections = Arc::clone(&connections);
                        let conn_pending = Arc::clone(&pending_requests);
                        let conn_handlers = Arc::clone(&message_handlers);

                        tokio::spawn(async move {
                            if let Err(e) = Self::handle_connection(
                                connection,
                                conn_connections,
                                conn_pending,
                                conn_handlers,
                            )
                            .await
                            {
                                error!("Connection handler error: {}", e);
                            }
                        });
                    }
                    Err(e) => {
                        error!("Failed to accept connection: {}", e);
                    }
                }
            }
        });

        // Start cleanup task
        self.start_cleanup_task().await;

        Ok(())
    }

    /// Connect to remote agent
    pub async fn connect(&self, agent_id: AgentId, addr: SocketAddr) -> Result<()> {
        debug!("Connecting to agent {} at {}", agent_id, addr);

        let mut connection = dial_tcp(addr, self.config.clone())
            .await
            .map_err(AgentFabricError::HtxError)?;

        // Perform handshake
        connection
            .handshake()
            .await
            .map_err(AgentFabricError::HtxError)?;

        // Store connection
        {
            let mut connections = self.connections.write().await;
            connections.insert(agent_id.clone(), Arc::new(Mutex::new(connection)));
        }

        info!("Connected to agent {}", agent_id);
        Ok(())
    }

    /// Send RPC request
    pub async fn send_request(
        &self,
        to: AgentId,
        message: AgentMessage,
        timeout_ms: Option<u64>,
    ) -> Result<AgentResponse> {
        let message_id = uuid::Uuid::new_v4().to_string();
        let envelope = RpcEnvelope {
            from: self.local_agent_id.clone(),
            to: to.clone(),
            message_id: message_id.clone(),
            message_type: RpcMessageType::Request,
            payload: serde_json::to_vec(&message)
                .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?
                .into(),
        };

        // Create pending request
        let (sender, receiver) = oneshot::channel();
        let timeout_instant =
            tokio::time::Instant::now() + Duration::from_millis(timeout_ms.unwrap_or(30000));

        {
            let mut pending = self.pending_requests.write().await;
            pending.insert(
                message_id.clone(),
                PendingRequest {
                    sender,
                    timeout: timeout_instant,
                },
            );
        }

        // Send message
        self.send_envelope(to, envelope).await?;

        // Wait for response
        let response = if let Some(timeout_ms) = timeout_ms {
            timeout(Duration::from_millis(timeout_ms), receiver)
                .await
                .map_err(|_| AgentFabricError::NetworkError("Request timeout".to_string()))?
                .map_err(|_| {
                    AgentFabricError::NetworkError("Response channel closed".to_string())
                })?
        } else {
            receiver.await.map_err(|_| {
                AgentFabricError::NetworkError("Response channel closed".to_string())
            })?
        };

        Ok(response)
    }

    /// Send RPC notification (no response expected)
    pub async fn send_notification(&self, to: AgentId, message: AgentMessage) -> Result<()> {
        let envelope = RpcEnvelope {
            from: self.local_agent_id.clone(),
            to: to.clone(),
            message_id: uuid::Uuid::new_v4().to_string(),
            message_type: RpcMessageType::Notification,
            payload: serde_json::to_vec(&message)
                .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?
                .into(),
        };

        self.send_envelope(to, envelope).await
    }

    /// Register message handler
    pub async fn register_handler(&self, service_name: String, handler: Box<dyn AgentServer>) {
        let mut handlers = self.message_handlers.write().await;
        handlers.insert(service_name, handler);
    }

    // Private methods

    async fn send_envelope(&self, to: AgentId, envelope: RpcEnvelope) -> Result<()> {
        let connections = self.connections.read().await;
        if let Some(connection_arc) = connections.get(&to) {
            let mut connection = connection_arc.lock().await;
            let stream_id = connection
                .create_stream()
                .map_err(AgentFabricError::HtxError)?;

            let envelope_bytes = serde_json::to_vec(&envelope)
                .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?;

            connection
                .send(stream_id, &envelope_bytes)
                .await
                .map_err(AgentFabricError::HtxError)?;

            Ok(())
        } else {
            Err(AgentFabricError::AgentNotFound(to))
        }
    }

    async fn handle_connection(
        mut connection: HtxTcpConnection,
        connections: Arc<RwLock<HashMap<AgentId, Arc<Mutex<HtxTcpConnection>>>>>,
        pending_requests: Arc<RwLock<HashMap<String, PendingRequest>>>,
        message_handlers: Arc<RwLock<HashMap<String, Box<dyn AgentServer>>>>,
    ) -> Result<()> {
        loop {
            match connection.recv().await {
                Ok(stream_data) => {
                    for (stream_id, data) in stream_data {
                        if let Err(e) = Self::process_message(
                            StreamId(stream_id),
                            data,
                            &mut connection,
                            Arc::clone(&pending_requests),
                            Arc::clone(&message_handlers),
                        )
                        .await
                        {
                            error!("Error processing message: {}", e);
                        }
                    }
                }
                Err(e) => {
                    error!("Connection receive error: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }

    async fn process_message(
        stream_id: StreamId,
        data: Bytes,
        connection: &mut HtxTcpConnection,
        pending_requests: Arc<RwLock<HashMap<String, PendingRequest>>>,
        message_handlers: Arc<RwLock<HashMap<String, Box<dyn AgentServer>>>>,
    ) -> Result<()> {
        let envelope: RpcEnvelope = serde_json::from_slice(&data)
            .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?;

        match envelope.message_type {
            RpcMessageType::Request => {
                // Handle incoming request
                let message: AgentMessage = serde_json::from_slice(&envelope.payload)
                    .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?;

                let handlers = message_handlers.read().await;
                let response = if let Some(handler) = handlers.get(&message.message_type) {
                    match handler.handle_message(envelope.from.clone(), message).await {
                        Ok(Some(resp)) => resp,
                        Ok(None) => AgentResponse::success(&envelope.message_id, Bytes::new()),
                        Err(e) => AgentResponse::error(&envelope.message_id, e.to_string()),
                    }
                } else {
                    AgentResponse::error(&envelope.message_id, "No handler for message type")
                };

                // Send response back
                let response_envelope = RpcEnvelope {
                    from: envelope.to,
                    to: envelope.from,
                    message_id: envelope.message_id,
                    message_type: RpcMessageType::Response,
                    payload: serde_json::to_vec(&response)
                        .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?
                        .into(),
                };

                let response_bytes = serde_json::to_vec(&response_envelope)
                    .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?;

                connection
                    .send(stream_id.0, &response_bytes)
                    .await
                    .map_err(AgentFabricError::HtxError)?;
            }
            RpcMessageType::Response => {
                // Handle response to our request
                let response: AgentResponse = serde_json::from_slice(&envelope.payload)
                    .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?;

                let mut pending = pending_requests.write().await;
                if let Some(pending_req) = pending.remove(&envelope.message_id) {
                    let _ = pending_req.sender.send(response);
                }
            }
            RpcMessageType::Notification => {
                // Handle notification
                let message: AgentMessage = serde_json::from_slice(&envelope.payload)
                    .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?;

                let handlers = message_handlers.read().await;
                if let Some(handler) = handlers.get(&message.message_type) {
                    if let Err(e) = handler.handle_message(envelope.from, message).await {
                        warn!("Notification handler error: {}", e);
                    }
                }
            }
        }

        Ok(())
    }

    async fn start_cleanup_task(&self) {
        let pending_requests = Arc::clone(&self.pending_requests);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));

            loop {
                interval.tick().await;

                let now = tokio::time::Instant::now();
                let mut pending = pending_requests.write().await;
                let mut to_remove = Vec::new();

                for (id, req) in pending.iter() {
                    if now > req.timeout {
                        to_remove.push(id.clone());
                    }
                }

                for id in to_remove {
                    if let Some(req) = pending.remove(&id) {
                        let _ = req
                            .sender
                            .send(AgentResponse::error(&id, "Request timeout"));
                    }
                }
            }
        });
    }
}

/// RPC client implementation
pub struct RpcClient {
    transport: Arc<RpcTransport>,
    target_agent: AgentId,
}

impl RpcClient {
    pub fn new(transport: Arc<RpcTransport>) -> Self {
        Self {
            transport,
            target_agent: AgentId::new("unknown", "unknown"),
        }
    }

    pub fn with_target(mut self, target: AgentId) -> Self {
        self.target_agent = target;
        self
    }

    pub async fn call(&self, to: AgentId, message: AgentMessage) -> Result<AgentResponse> {
        self.transport.send_request(to, message, None).await
    }

    pub async fn call_with_timeout(
        &self,
        to: AgentId,
        message: AgentMessage,
        timeout_ms: u64,
    ) -> Result<AgentResponse> {
        self.transport
            .send_request(to, message, Some(timeout_ms))
            .await
    }
}

#[async_trait]
impl AgentClient for RpcClient {
    async fn send_message(&self, message: AgentMessage) -> Result<AgentResponse> {
        self.transport
            .send_request(self.target_agent.clone(), message, None)
            .await
    }

    async fn send_notification(&self, message: AgentMessage) -> Result<()> {
        self.transport
            .send_notification(self.target_agent.clone(), message)
            .await
    }

    fn target_agent(&self) -> &AgentId {
        &self.target_agent
    }
}

/// RPC server wrapper
pub struct RpcServer {
    transport: Arc<RpcTransport>,
}

impl RpcServer {
    pub fn new(transport: Arc<RpcTransport>) -> Self {
        Self { transport }
    }

    pub async fn start(&self) -> Result<()> {
        self.transport.start_server().await
    }

    pub async fn register_handler(&self, service_name: String, handler: Box<dyn AgentServer>) {
        self.transport.register_handler(service_name, handler).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TestAgentServer {
        call_count: AtomicUsize,
    }

    impl TestAgentServer {
        fn new() -> Self {
            Self {
                call_count: AtomicUsize::new(0),
            }
        }

        fn call_count(&self) -> usize {
            self.call_count.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl AgentServer for TestAgentServer {
        async fn handle_message(
            &self,
            _from: AgentId,
            message: AgentMessage,
        ) -> Result<Option<AgentResponse>> {
            self.call_count.fetch_add(1, Ordering::Relaxed);

            if message.message_type == "ping" {
                Ok(Some(AgentResponse::success(
                    &message.id,
                    Bytes::from("pong"),
                )))
            } else {
                Ok(Some(AgentResponse::error(
                    &message.id,
                    "Unknown message type",
                )))
            }
        }

        fn supported_message_types(&self) -> Vec<String> {
            vec!["ping".to_string()]
        }
    }

    #[tokio::test]
    async fn test_rpc_transport_creation() {
        let config = HtxConfig {
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            ..Default::default()
        };

        let transport = RpcTransport::new(config).await.unwrap();
        assert!(!transport.local_agent_id.id.is_empty());
    }

    #[test]
    fn test_rpc_envelope_serialization() {
        let envelope = RpcEnvelope {
            from: AgentId::new("sender", "node1"),
            to: AgentId::new("receiver", "node2"),
            message_id: "msg-123".to_string(),
            message_type: RpcMessageType::Request,
            payload: Bytes::from("test payload"),
        };

        let serialized = serde_json::to_vec(&envelope).unwrap();
        let deserialized: RpcEnvelope = serde_json::from_slice(&serialized).unwrap();

        assert_eq!(envelope.from, deserialized.from);
        assert_eq!(envelope.to, deserialized.to);
        assert_eq!(envelope.message_id, deserialized.message_id);
        assert_eq!(envelope.payload, deserialized.payload);
    }

    #[test]
    fn test_rpc_client_creation() {
        let config = HtxConfig::default();
        let transport = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(RpcTransport::new(config))
            .unwrap();

        let client = RpcClient::new(Arc::new(transport))
            .with_target(AgentId::new("test-agent", "test-node"));

        assert_eq!(client.target_agent().id, "test-agent");
        assert_eq!(client.target_agent().node, "test-node");
    }
}
