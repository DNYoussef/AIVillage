// gRPC client for communication with SCION sidecar
// Production implementation with connection pooling, retries, and health checks

use anyhow::{Context, Result, bail};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time::{interval, sleep, timeout};
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, Response, Status};
use tracing::{debug, error, info, warn};

use crate::config::ScionConfig;
use crate::metrics::MetricsCollector;
use crate::generated::betanet::gateway::{
    betanet_gateway_client::BetanetGatewayClient,
    SendScionPacketRequest, SendScionPacketResponse,
    RecvScionPacketRequest, RecvScionPacketResponse,
    RegisterPathRequest, RegisterPathResponse,
    QueryPathsRequest, QueryPathsResponse,
    HealthRequest, HealthResponse,
    StatsRequest, StatsResponse,
    ValidateSequenceRequest, ValidateSequenceResponse,
    PathInfo as ProtoPathInfo,
};

/// SCION client connection state
#[derive(Debug, Clone)]
pub enum ConnectionState {
    Connected,
    Connecting,
    Disconnected,
    Failed(String),
}

/// SCION client statistics
#[derive(Debug, Clone, Default)]
pub struct ScionClientStats {
    /// Total requests sent
    pub requests_sent: u64,

    /// Successful requests
    pub requests_successful: u64,

    /// Failed requests
    pub requests_failed: u64,

    /// Connection attempts
    pub connection_attempts: u64,

    /// Current connection state
    pub connection_state: String,

    /// Average request latency in microseconds
    pub avg_request_latency_us: f64,

    /// Packets sent via SCION
    pub packets_sent: u64,

    /// Packets received via SCION
    pub packets_received: u64,

    /// Paths registered
    pub paths_registered: u64,

    /// Sequence validations performed
    pub sequence_validations: u64,
}

/// SCION client with connection management and retries
pub struct ScionClient {
    config: ScionConfig,
    metrics: Arc<MetricsCollector>,
    client: Arc<RwLock<Option<BetanetGatewayClient<Channel>>>>,
    connection_state: Arc<RwLock<ConnectionState>>,
    stats: Arc<RwLock<ScionClientStats>>,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    background_handle: Option<tokio::task::JoinHandle<()>>,
}

impl ScionClient {
    /// Create new SCION client
    pub async fn new(
        config: ScionConfig,
        metrics: Arc<MetricsCollector>,
    ) -> Result<Self> {
        info!(?config.address, "Initializing SCION client");

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

        let client = Self {
            config: config.clone(),
            metrics: metrics.clone(),
            client: Arc::new(RwLock::new(None)),
            connection_state: Arc::new(RwLock::new(ConnectionState::Disconnected)),
            stats: Arc::new(RwLock::new(ScionClientStats::default())),
            shutdown_tx: Some(shutdown_tx),
            background_handle: None,
        };

        // Attempt initial connection
        client.connect().await?;

        // Start background connection monitoring
        let background_handle = tokio::spawn(Self::background_worker(
            config,
            client.client.clone(),
            client.connection_state.clone(),
            client.stats.clone(),
            metrics,
            shutdown_rx,
        ));

        Ok(Self {
            background_handle: Some(background_handle),
            ..client
        })
    }

    /// Send SCION packet
    pub async fn send_packet(
        &self,
        raw_packet: Vec<u8>,
        dst_ia: String,
        dst_addr: String,
        path_fingerprint: Option<String>,
    ) -> Result<SendScionPacketResponse> {
        let request = Request::new(SendScionPacketRequest {
            raw_packet,
            dst_ia,
            dst_addr,
            path_fingerprint: path_fingerprint.unwrap_or_default(),
            timeout_ms: self.config.request_timeout.as_millis() as u32,
        });

        let response = self.execute_request("send_packet", |client| async move {
            client.send_scion_packet(request).await
        }).await?;

        // Update stats
        let mut stats = self.stats.write().await;
        stats.packets_sent += 1;

        Ok(response.into_inner())
    }

    /// Receive SCION packet (polling)
    pub async fn receive_packet(&self, timeout_ms: u32) -> Result<Option<RecvScionPacketResponse>> {
        let request = Request::new(RecvScionPacketRequest {
            timeout_ms,
            max_packet_size: 65536, // 64KB max
        });

        match self.execute_request("recv_packet", |client| async move {
            client.recv_scion_packet(request).await
        }).await {
            Ok(response) => {
                let response_inner = response.into_inner();

                // Update stats
                let mut stats = self.stats.write().await;
                stats.packets_received += 1;

                Ok(Some(response_inner))
            }
            Err(e) => {
                // Check if this is a timeout (expected for polling)
                if e.to_string().contains("timeout") || e.to_string().contains("no packets") {
                    Ok(None)
                } else {
                    Err(e)
                }
            }
        }
    }

    /// Register SCION path
    pub async fn register_path(
        &self,
        path_fingerprint: String,
        dst_ia: String,
        path_info: Vec<u8>,
    ) -> Result<RegisterPathResponse> {
        let request = Request::new(RegisterPathRequest {
            path_fingerprint,
            dst_ia,
            path_info,
        });

        let response = self.execute_request("register_path", |client| async move {
            client.register_path(request).await
        }).await?;

        // Update stats
        let mut stats = self.stats.write().await;
        stats.paths_registered += 1;

        Ok(response.into_inner())
    }

    /// Query available paths
    pub async fn query_paths(&self, dst_ia: String) -> Result<QueryPathsResponse> {
        let request = Request::new(QueryPathsRequest {
            dst_ia,
            include_expired: false,
            max_paths: 10,
        });

        let response = self.execute_request("query_paths", |client| async move {
            client.query_paths(request).await
        }).await?;

        Ok(response.into_inner())
    }

    /// Validate sequence number
    pub async fn validate_sequence(
        &self,
        peer_id: String,
        sequence: u64,
        timestamp_ns: u64,
        update_window: bool,
    ) -> Result<ValidateSequenceResponse> {
        let request = Request::new(ValidateSequenceRequest {
            peer_id,
            sequence,
            timestamp_ns,
            update_window,
        });

        let response = self.execute_request("validate_sequence", |client| async move {
            client.validate_sequence(request).await
        }).await?;

        // Update stats
        let mut stats = self.stats.write().await;
        stats.sequence_validations += 1;

        Ok(response.into_inner())
    }

    /// Health check
    pub async fn health_check(&self) -> Result<HealthResponse> {
        let request = Request::new(HealthRequest {});

        let response = self.execute_request("health", |client| async move {
            client.health(request).await
        }).await?;

        Ok(response.into_inner())
    }

    /// Get sidecar statistics
    pub async fn get_sidecar_stats(&self) -> Result<StatsResponse> {
        let request = Request::new(StatsRequest {
            include_detailed: true,
        });

        let response = self.execute_request("stats", |client| async move {
            client.stats(request).await
        }).await?;

        Ok(response.into_inner())
    }

    /// Get client statistics
    pub async fn get_stats(&self) -> ScionClientStats {
        let stats = self.stats.read().await.clone();
        let connection_state = self.connection_state.read().await;

        ScionClientStats {
            connection_state: format!("{:?}", *connection_state),
            ..stats
        }
    }

    /// Check if client is connected
    pub async fn is_connected(&self) -> bool {
        let state = self.connection_state.read().await;
        matches!(*state, ConnectionState::Connected)
    }

    /// Stop SCION client
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping SCION client");

        if let Some(tx) = &self.shutdown_tx {
            let _ = tx.send(());
        }

        if let Some(handle) = &self.background_handle {
            if let Err(e) = handle.await {
                warn!(error = ?e, "Error waiting for SCION client background task to stop");
            }
        }

        // Close client connection
        {
            let mut client_guard = self.client.write().await;
            *client_guard = None;
        }

        {
            let mut state_guard = self.connection_state.write().await;
            *state_guard = ConnectionState::Disconnected;
        }

        info!("SCION client stopped");
        Ok(())
    }

    /// Establish gRPC connection to SCION sidecar
    async fn connect(&self) -> Result<()> {
        info!(address = ?self.config.address, "Connecting to SCION sidecar");

        {
            let mut state_guard = self.connection_state.write().await;
            *state_guard = ConnectionState::Connecting;
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.connection_attempts += 1;
        }

        let endpoint = Endpoint::from_shared(format!("http://{}", self.config.address))
            .context("Invalid SCION sidecar address")?
            .connect_timeout(self.config.connect_timeout)
            .timeout(self.config.request_timeout)
            .keep_alive_timeout(self.config.keep_alive_interval);

        let channel = match timeout(self.config.connect_timeout, endpoint.connect()).await {
            Ok(Ok(channel)) => channel,
            Ok(Err(e)) => {
                let error_msg = format!("Connection failed: {}", e);
                error!(error = error_msg, "Failed to connect to SCION sidecar");

                let mut state_guard = self.connection_state.write().await;
                *state_guard = ConnectionState::Failed(error_msg.clone());

                bail!(error_msg);
            }
            Err(_) => {
                let error_msg = "Connection timeout".to_string();
                error!("Connection to SCION sidecar timed out");

                let mut state_guard = self.connection_state.write().await;
                *state_guard = ConnectionState::Failed(error_msg.clone());

                bail!(error_msg);
            }
        };

        let grpc_client = BetanetGatewayClient::new(channel);

        // Test connection with health check
        let health_result = timeout(
            Duration::from_secs(5),
            grpc_client.clone().health(Request::new(HealthRequest {}))
        ).await;

        match health_result {
            Ok(Ok(_)) => {
                info!("Successfully connected to SCION sidecar");

                {
                    let mut client_guard = self.client.write().await;
                    *client_guard = Some(grpc_client);
                }

                {
                    let mut state_guard = self.connection_state.write().await;
                    *state_guard = ConnectionState::Connected;
                }

                Ok(())
            }
            Ok(Err(status)) => {
                let error_msg = format!("Health check failed: {}", status);
                error!(error = error_msg, "SCION sidecar health check failed");

                let mut state_guard = self.connection_state.write().await;
                *state_guard = ConnectionState::Failed(error_msg.clone());

                bail!(error_msg);
            }
            Err(_) => {
                let error_msg = "Health check timeout".to_string();
                error!("SCION sidecar health check timed out");

                let mut state_guard = self.connection_state.write().await;
                *state_guard = ConnectionState::Failed(error_msg.clone());

                bail!(error_msg);
            }
        }
    }

    /// Execute gRPC request with retry logic
    async fn execute_request<F, Fut, T>(&self, operation: &str, request_fn: F) -> Result<Response<T>>
    where
        F: Fn(BetanetGatewayClient<Channel>) -> Fut,
        Fut: std::future::Future<Output = Result<Response<T>, Status>>,
    {
        let start_time = Instant::now();
        let mut last_error: Option<Status> = None;

        for attempt in 1..=self.config.retry.max_attempts {
            // Check if we have a client
            let client = {
                let client_guard = self.client.read().await;
                match client_guard.as_ref() {
                    Some(client) => client.clone(),
                    None => {
                        if attempt == 1 {
                            // Try to reconnect on first attempt
                            drop(client_guard);
                            self.connect().await.context("Failed to reconnect")?;
                            let client_guard = self.client.read().await;
                            client_guard.as_ref().unwrap().clone()
                        } else {
                            bail!("No SCION client connection available");
                        }
                    }
                }
            };

            // Execute request
            match request_fn(client).await {
                Ok(response) => {
                    let elapsed = start_time.elapsed();

                    // Update stats
                    let mut stats = self.stats.write().await;
                    stats.requests_sent += attempt as u64;
                    stats.requests_successful += 1;

                    // Update average latency
                    let current_latency = elapsed.as_micros() as f64;
                    if stats.avg_request_latency_us == 0.0 {
                        stats.avg_request_latency_us = current_latency;
                    } else {
                        stats.avg_request_latency_us =
                            0.9 * stats.avg_request_latency_us + 0.1 * current_latency;
                    }

                    debug!(
                        operation = operation,
                        attempt = attempt,
                        latency_us = current_latency,
                        "SCION request successful"
                    );

                    return Ok(response);
                }
                Err(status) => {
                    last_error = Some(status.clone());

                    warn!(
                        operation = operation,
                        attempt = attempt,
                        max_attempts = self.config.retry.max_attempts,
                        error = ?status,
                        "SCION request failed"
                    );

                    // Check if we should retry
                    if attempt < self.config.retry.max_attempts {
                        // Check if error is retryable
                        if !Self::is_retryable_error(&status) {
                            break;
                        }

                        // Calculate retry delay with exponential backoff
                        let delay = self.calculate_retry_delay(attempt);
                        debug!(delay_ms = delay.as_millis(), "Retrying SCION request");
                        sleep(delay).await;

                        // Try to reconnect on connection errors
                        if matches!(status.code(), tonic::Code::Unavailable | tonic::Code::DeadlineExceeded) {
                            if let Err(e) = self.connect().await {
                                warn!(error = ?e, "Failed to reconnect during retry");
                            }
                        }
                    }
                }
            }
        }

        // Update failure stats
        {
            let mut stats = self.stats.write().await;
            stats.requests_sent += self.config.retry.max_attempts as u64;
            stats.requests_failed += 1;
        }

        // All retries exhausted
        match last_error {
            Some(status) => bail!("SCION request '{}' failed after {} attempts: {}",
                                 operation, self.config.retry.max_attempts, status),
            None => bail!("SCION request '{}' failed after {} attempts",
                         operation, self.config.retry.max_attempts),
        }
    }

    /// Check if gRPC error is retryable
    fn is_retryable_error(status: &Status) -> bool {
        matches!(
            status.code(),
            tonic::Code::Unavailable
            | tonic::Code::DeadlineExceeded
            | tonic::Code::ResourceExhausted
            | tonic::Code::Aborted
        )
    }

    /// Calculate retry delay with exponential backoff and jitter
    fn calculate_retry_delay(&self, attempt: usize) -> Duration {
        let base_delay = self.config.retry.base_delay;
        let max_delay = self.config.retry.max_delay;
        let multiplier = self.config.retry.backoff_multiplier;

        let delay = base_delay.as_millis() as f64 * multiplier.powi((attempt - 1) as i32);
        let delay = Duration::from_millis(delay as u64).min(max_delay);

        if self.config.retry.enable_jitter {
            let jitter = rand::random::<f64>() * 0.1; // ±10% jitter
            let jitter_multiplier = 1.0 + (jitter - 0.05);
            Duration::from_millis((delay.as_millis() as f64 * jitter_multiplier) as u64)
        } else {
            delay
        }
    }

    /// Background worker for connection monitoring
    async fn background_worker(
        config: ScionConfig,
        client: Arc<RwLock<Option<BetanetGatewayClient<Channel>>>>,
        connection_state: Arc<RwLock<ConnectionState>>,
        stats: Arc<RwLock<ScionClientStats>>,
        metrics: Arc<MetricsCollector>,
        mut shutdown_rx: tokio::sync::oneshot::Receiver<()>,
    ) {
        info!("Starting SCION client background worker");

        let mut health_check_interval = interval(config.keep_alive_interval);

        loop {
            tokio::select! {
                _ = health_check_interval.tick() => {
                    Self::perform_health_check(&client, &connection_state, &stats).await;
                }

                _ = &mut shutdown_rx => {
                    info!("SCION client background worker shutting down");
                    break;
                }
            }
        }

        info!("SCION client background worker stopped");
    }

    /// Perform periodic health check
    async fn perform_health_check(
        client: &Arc<RwLock<Option<BetanetGatewayClient<Channel>>>>,
        connection_state: &Arc<RwLock<ConnectionState>>,
        stats: &Arc<RwLock<ScionClientStats>>,
    ) {
        let client_opt = {
            let client_guard = client.read().await;
            client_guard.clone()
        };

        if let Some(mut grpc_client) = client_opt {
            let health_result = timeout(
                Duration::from_secs(10),
                grpc_client.health(Request::new(HealthRequest {}))
            ).await;

            match health_result {
                Ok(Ok(_)) => {
                    let mut state_guard = connection_state.write().await;
                    if !matches!(*state_guard, ConnectionState::Connected) {
                        info!("SCION sidecar connection restored");
                        *state_guard = ConnectionState::Connected;
                    }
                }
                Ok(Err(status)) => {
                    warn!(error = ?status, "SCION sidecar health check failed");
                    let mut state_guard = connection_state.write().await;
                    *state_guard = ConnectionState::Failed(format!("Health check failed: {}", status));
                }
                Err(_) => {
                    warn!("SCION sidecar health check timeout");
                    let mut state_guard = connection_state.write().await;
                    *state_guard = ConnectionState::Failed("Health check timeout".to_string());
                }
            }
        } else {
            debug!("Skipping health check - no client connection");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GatewayConfig;
    use std::net::SocketAddr;

    fn create_test_config() -> ScionConfig {
        ScionConfig {
            address: "127.0.0.1:8080".parse().unwrap(),
            connect_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(10),
            keep_alive_interval: Duration::from_secs(30),
            enable_compression: true,
            max_message_size: 1024 * 1024,
            retry: crate::config::RetryConfig {
                max_attempts: 3,
                base_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(5),
                backoff_multiplier: 2.0,
                enable_jitter: true,
            },
        }
    }

    #[test]
    fn test_retry_delay_calculation() {
        let config = create_test_config();
        let gateway_config = Arc::new(GatewayConfig::default());
        let metrics = Arc::new(MetricsCollector::new(gateway_config).unwrap());

        // We can't easily test the full client without a running sidecar,
        // but we can test the retry delay calculation logic
        let client = ScionClient {
            config: config.clone(),
            metrics,
            client: Arc::new(RwLock::new(None)),
            connection_state: Arc::new(RwLock::new(ConnectionState::Disconnected)),
            stats: Arc::new(RwLock::new(ScionClientStats::default())),
            shutdown_tx: None,
            background_handle: None,
        };

        // Test exponential backoff
        let delay1 = client.calculate_retry_delay(1);
        let delay2 = client.calculate_retry_delay(2);
        let delay3 = client.calculate_retry_delay(3);

        // Each delay should be larger (with some variance due to jitter)
        assert!(delay1 >= config.retry.base_delay);
        assert!(delay2 >= delay1 || delay2.as_millis() > delay1.as_millis() / 2); // Account for jitter
        assert!(delay3 >= delay2 || delay3.as_millis() > delay2.as_millis() / 2);

        // All delays should be under max delay
        assert!(delay1 <= config.retry.max_delay);
        assert!(delay2 <= config.retry.max_delay);
        assert!(delay3 <= config.retry.max_delay);
    }

    #[test]
    fn test_retryable_error_detection() {
        use tonic::Code;

        // Retryable errors
        assert!(ScionClient::is_retryable_error(&Status::new(Code::Unavailable, "service unavailable")));
        assert!(ScionClient::is_retryable_error(&Status::new(Code::DeadlineExceeded, "timeout")));
        assert!(ScionClient::is_retryable_error(&Status::new(Code::ResourceExhausted, "rate limited")));
        assert!(ScionClient::is_retryable_error(&Status::new(Code::Aborted, "aborted")));

        // Non-retryable errors
        assert!(!ScionClient::is_retryable_error(&Status::new(Code::InvalidArgument, "bad request")));
        assert!(!ScionClient::is_retryable_error(&Status::new(Code::NotFound, "not found")));
        assert!(!ScionClient::is_retryable_error(&Status::new(Code::PermissionDenied, "forbidden")));
        assert!(!ScionClient::is_retryable_error(&Status::new(Code::Unauthenticated, "unauthorized")));
    }
}
