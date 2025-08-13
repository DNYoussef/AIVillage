// Prometheus metrics collection for Betanet Gateway
// Production implementation with comprehensive KPI tracking

use anyhow::{Context, Result};
use prometheus::{
    Counter, CounterVec, Gauge, GaugeVec, Histogram, HistogramVec, 
    IntCounter, IntCounterVec, IntGauge, IntGaugeVec,
    register_counter, register_counter_vec, register_gauge, register_gauge_vec,
    register_histogram, register_histogram_vec, register_int_counter, 
    register_int_counter_vec, register_int_gauge, register_int_gauge_vec,
    Encoder, TextEncoder, Registry
};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use hyper::{Body, Method, Request, Response, Server, StatusCode};
use hyper::service::{make_service_fn, service_fn};

use crate::config::{GatewayConfig, MetricsConfig};

/// Comprehensive metrics collector for Betanet Gateway
pub struct MetricsCollector {
    config: MetricsConfig,
    registry: Registry,
    
    // Packet processing metrics
    packets_sent: IntCounterVec,
    packets_received: IntCounterVec,
    packets_dropped: IntCounterVec,
    packet_processing_time: HistogramVec,
    packet_size_bytes: HistogramVec,
    
    // SCION path metrics
    paths_discovered: IntCounter,
    paths_active: IntGauge,
    path_quality_rtt: GaugeVec,
    path_quality_loss: GaugeVec,
    path_failovers: IntCounterVec,
    
    // Anti-replay metrics  
    sequence_validations: IntCounter,
    replays_blocked: IntCounter,
    expired_sequences: IntCounter,
    future_sequences: IntCounter,
    validation_time_us: Histogram,
    active_windows: IntGauge,
    
    // AEAD encryption metrics
    aead_encryptions: IntCounterVec,
    aead_decryptions: IntCounterVec,
    aead_auth_failures: IntCounter,
    aead_key_rotations: IntCounter,
    aead_encryption_time_us: Histogram,
    aead_decryption_time_us: Histogram,
    aead_active_sessions: IntGauge,
    aead_bytes_encrypted: Counter,
    aead_bytes_decrypted: Counter,
    
    // HTX transport metrics
    htx_connections: IntGaugeVec,
    htx_requests: IntCounterVec,
    htx_response_time: HistogramVec,
    htx_bytes_transferred: CounterVec,
    
    // Encryption/compression metrics
    encryption_operations: IntCounterVec,
    compression_ratio: GaugeVec,
    compression_time_us: Histogram,
    
    // System metrics
    memory_usage_bytes: Gauge,
    cpu_usage_percent: Gauge,
    disk_usage_bytes: GaugeVec,
    network_errors: IntCounterVec,
    
    // Performance KPI metrics (for SLA compliance)
    throughput_packets_per_minute: Gauge,
    p95_latency_ms: Gauge,
    availability_percent: Gauge,
    error_rate_percent: Gauge,
    
    // Internal state
    start_time: Instant,
    batch_updates: Arc<RwLock<HashMap<String, f64>>>,
}

impl MetricsCollector {
    /// Create new metrics collector with custom registry
    pub fn new(config: Arc<GatewayConfig>) -> Result<Self> {
        let registry = Registry::new();
        let metrics_config = config.metrics.clone();
        
        // Register packet processing metrics
        let packets_sent = register_int_counter_vec!(
            "betanet_gateway_packets_sent_total",
            "Total number of packets sent",
            &["destination", "path_id", "result"]
        )?;
        
        let packets_received = register_int_counter_vec!(
            "betanet_gateway_packets_received_total", 
            "Total number of packets received",
            &["source", "path_id", "result"]
        )?;
        
        let packets_dropped = register_int_counter_vec!(
            "betanet_gateway_packets_dropped_total",
            "Total number of packets dropped",
            &["reason", "path_id"]
        )?;
        
        let packet_processing_time = register_histogram_vec!(
            "betanet_gateway_packet_processing_seconds",
            "Time spent processing packets",
            &["operation", "path_id"],
            vec![0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.000, 2.500, 5.000, 10.000]
        )?;
        
        let packet_size_bytes = register_histogram_vec!(
            "betanet_gateway_packet_size_bytes",
            "Distribution of packet sizes",
            &["direction"],
            vec![64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0]
        )?;
        
        // Register SCION path metrics
        let paths_discovered = register_int_counter!(
            "betanet_gateway_paths_discovered_total",
            "Total number of SCION paths discovered"
        )?;
        
        let paths_active = register_int_gauge!(
            "betanet_gateway_paths_active",
            "Number of currently active SCION paths"
        )?;
        
        let path_quality_rtt = register_gauge_vec!(
            "betanet_gateway_path_rtt_microseconds",
            "Path round-trip time in microseconds",
            &["path_id", "destination"]
        )?;
        
        let path_quality_loss = register_gauge_vec!(
            "betanet_gateway_path_loss_rate",
            "Path packet loss rate (0.0-1.0)",
            &["path_id", "destination"]
        )?;
        
        let path_failovers = register_int_counter_vec!(
            "betanet_gateway_path_failovers_total",
            "Number of path failovers performed",
            &["from_path", "to_path", "reason"]
        )?;
        
        // Register anti-replay metrics
        let sequence_validations = register_int_counter!(
            "betanet_gateway_sequence_validations_total",
            "Total number of sequence number validations"
        )?;
        
        let replays_blocked = register_int_counter!(
            "betanet_gateway_replays_blocked_total",
            "Total number of replay attacks blocked"
        )?;
        
        let expired_sequences = register_int_counter!(
            "betanet_gateway_expired_sequences_total",
            "Total number of expired sequences rejected"
        )?;
        
        let future_sequences = register_int_counter!(
            "betanet_gateway_future_sequences_total",
            "Total number of far-future sequences rejected"
        )?;
        
        let validation_time_us = register_histogram!(
            "betanet_gateway_validation_time_microseconds",
            "Time spent validating sequence numbers",
            vec![10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0, 100000.0]
        )?;
        
        let active_windows = register_int_gauge!(
            "betanet_gateway_active_windows",
            "Number of active anti-replay windows"
        )?;
        
        // Register AEAD encryption metrics
        let aead_encryptions = register_int_counter_vec!(
            "betanet_gateway_aead_encryptions_total",
            "Total number of AEAD encryption operations",
            &["frame_type", "result"]
        )?;
        
        let aead_decryptions = register_int_counter_vec!(
            "betanet_gateway_aead_decryptions_total",
            "Total number of AEAD decryption operations",
            &["frame_type", "result"]
        )?;
        
        let aead_auth_failures = register_int_counter!(
            "betanet_gateway_aead_auth_failures_total",
            "Total number of AEAD authentication failures"
        )?;
        
        let aead_key_rotations = register_int_counter!(
            "betanet_gateway_aead_key_rotations_total",
            "Total number of AEAD key rotations performed"
        )?;
        
        let aead_encryption_time_us = register_histogram!(
            "betanet_gateway_aead_encryption_time_microseconds",
            "Time spent on AEAD encryption operations",
            vec![10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0]
        )?;
        
        let aead_decryption_time_us = register_histogram!(
            "betanet_gateway_aead_decryption_time_microseconds", 
            "Time spent on AEAD decryption operations",
            vec![10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0]
        )?;
        
        let aead_active_sessions = register_int_gauge!(
            "betanet_gateway_aead_active_sessions",
            "Number of active AEAD sessions"
        )?;
        
        let aead_bytes_encrypted = register_counter!(
            "betanet_gateway_aead_bytes_encrypted_total",
            "Total bytes encrypted with AEAD"
        )?;
        
        let aead_bytes_decrypted = register_counter!(
            "betanet_gateway_aead_bytes_decrypted_total",
            "Total bytes decrypted with AEAD"
        )?;
        
        // Register HTX transport metrics
        let htx_connections = register_int_gauge_vec!(
            "betanet_gateway_htx_connections",
            "Number of active HTX connections",
            &["protocol", "state"]
        )?;
        
        let htx_requests = register_int_counter_vec!(
            "betanet_gateway_htx_requests_total",
            "Total number of HTX requests",
            &["method", "status", "protocol"]
        )?;
        
        let htx_response_time = register_histogram_vec!(
            "betanet_gateway_htx_response_time_seconds",
            "HTX request response time",
            &["method", "status"],
            vec![0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.000, 2.500, 5.000]
        )?;
        
        let htx_bytes_transferred = register_counter_vec!(
            "betanet_gateway_htx_bytes_total",
            "Total bytes transferred over HTX",
            &["direction", "protocol"]
        )?;
        
        // Register encryption/compression metrics
        let encryption_operations = register_int_counter_vec!(
            "betanet_gateway_encryption_operations_total",
            "Total number of encryption/decryption operations",
            &["operation", "result"]
        )?;
        
        let compression_ratio = register_gauge_vec!(
            "betanet_gateway_compression_ratio",
            "Compression ratio achieved (original_size / compressed_size)",
            &["algorithm"]
        )?;
        
        let compression_time_us = register_histogram!(
            "betanet_gateway_compression_time_microseconds",
            "Time spent on compression/decompression",
            vec![100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0, 100000.0, 500000.0]
        )?;
        
        // Register system metrics
        let memory_usage_bytes = register_gauge!(
            "betanet_gateway_memory_usage_bytes",
            "Current memory usage in bytes"
        )?;
        
        let cpu_usage_percent = register_gauge!(
            "betanet_gateway_cpu_usage_percent",
            "Current CPU usage percentage"
        )?;
        
        let disk_usage_bytes = register_gauge_vec!(
            "betanet_gateway_disk_usage_bytes",
            "Current disk usage in bytes",
            &["path", "type"]
        )?;
        
        let network_errors = register_int_counter_vec!(
            "betanet_gateway_network_errors_total",
            "Total number of network errors",
            &["error_type", "component"]
        )?;
        
        // Register performance KPI metrics
        let throughput_packets_per_minute = register_gauge!(
            "betanet_gateway_throughput_packets_per_minute",
            "Current packet throughput per minute"
        )?;
        
        let p95_latency_ms = register_gauge!(
            "betanet_gateway_p95_latency_milliseconds",
            "95th percentile latency in milliseconds"
        )?;
        
        let availability_percent = register_gauge!(
            "betanet_gateway_availability_percent",
            "Service availability percentage"
        )?;
        
        let error_rate_percent = register_gauge!(
            "betanet_gateway_error_rate_percent",
            "Error rate percentage"
        )?;
        
        info!("Metrics collector initialized with {} metrics", registry.gather().len());
        
        Ok(Self {
            config: metrics_config,
            registry,
            packets_sent,
            packets_received,
            packets_dropped,
            packet_processing_time,
            packet_size_bytes,
            paths_discovered,
            paths_active,
            path_quality_rtt,
            path_quality_loss,
            path_failovers,
            sequence_validations,
            replays_blocked,
            expired_sequences,
            future_sequences,
            validation_time_us,
            active_windows,
            aead_encryptions,
            aead_decryptions,
            aead_auth_failures,
            aead_key_rotations,
            aead_encryption_time_us,
            aead_decryption_time_us,
            aead_active_sessions,
            aead_bytes_encrypted,
            aead_bytes_decrypted,
            htx_connections,
            htx_requests,
            htx_response_time,
            htx_bytes_transferred,
            encryption_operations,
            compression_ratio,
            compression_time_us,
            memory_usage_bytes,
            cpu_usage_percent,
            disk_usage_bytes,
            network_errors,
            throughput_packets_per_minute,
            p95_latency_ms,
            availability_percent,
            error_rate_percent,
            start_time: Instant::now(),
            batch_updates: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Start metrics HTTP server
    pub async fn start_server(&self, bind_addr: SocketAddr) -> Result<()> {
        info!(?bind_addr, "Starting metrics server");
        
        let registry = self.registry.clone();
        
        let make_svc = make_service_fn(move |_conn| {
            let registry = registry.clone();
            async move {
                Ok::<_, hyper::Error>(service_fn(move |req| {
                    let registry = registry.clone();
                    async move { handle_metrics_request(req, registry).await }
                }))
            }
        });
        
        let server = Server::bind(&bind_addr).serve(make_svc);
        
        info!("Metrics server listening on http://{}/metrics", bind_addr);
        
        if let Err(e) = server.await {
            error!(error = ?e, "Metrics server failed");
            return Err(e.into());
        }
        
        Ok(())
    }
    
    /// Record packet sent
    pub fn record_packet_sent(&self, destination: &str, path_id: &str, success: bool) {
        let result = if success { "success" } else { "failure" };
        self.packets_sent
            .with_label_values(&[destination, path_id, result])
            .inc();
    }
    
    /// Record packet received  
    pub fn record_packet_received(&self, source: &str, path_id: &str, success: bool) {
        let result = if success { "success" } else { "failure" };
        self.packets_received
            .with_label_values(&[source, path_id, result])
            .inc();
    }
    
    /// Record packet dropped
    pub fn record_packet_dropped(&self, reason: &str, path_id: &str) {
        self.packets_dropped
            .with_label_values(&[reason, path_id])
            .inc();
    }
    
    /// Record packet processing time
    pub fn record_processing_time(&self, operation: &str, path_id: &str, duration: Duration) {
        self.packet_processing_time
            .with_label_values(&[operation, path_id])
            .observe(duration.as_secs_f64());
    }
    
    /// Record packet size
    pub fn record_packet_size(&self, direction: &str, size_bytes: usize) {
        self.packet_size_bytes
            .with_label_values(&[direction])
            .observe(size_bytes as f64);
    }
    
    /// Record path discovery
    pub fn record_path_discovered(&self) {
        self.paths_discovered.inc();
    }
    
    /// Update active paths count
    pub fn update_active_paths(&self, count: i64) {
        self.paths_active.set(count);
    }
    
    /// Update path quality metrics
    pub fn update_path_quality(&self, path_id: &str, destination: &str, rtt_us: f64, loss_rate: f64) {
        self.path_quality_rtt
            .with_label_values(&[path_id, destination])
            .set(rtt_us);
        
        self.path_quality_loss
            .with_label_values(&[path_id, destination])
            .set(loss_rate);
    }
    
    /// Record path failover
    pub fn record_path_failover(&self, from_path: &str, to_path: &str, reason: &str) {
        self.path_failovers
            .with_label_values(&[from_path, to_path, reason])
            .inc();
    }
    
    /// Record sequence validation
    pub fn record_sequence_validation(&self, validation_time: Duration) {
        self.sequence_validations.inc();
        self.validation_time_us.observe(validation_time.as_micros() as f64);
    }
    
    /// Record replay blocked
    pub fn record_replay_blocked(&self) {
        self.replays_blocked.inc();
    }
    
    /// Record expired sequence
    pub fn record_expired_sequence(&self) {
        self.expired_sequences.inc();
    }
    
    /// Record future sequence
    pub fn record_future_sequence(&self) {
        self.future_sequences.inc();
    }
    
    /// Update active windows count
    pub fn update_active_windows(&self, count: i64) {
        self.active_windows.set(count);
    }
    
    /// Record AEAD encryption operation
    pub fn record_aead_encryption(&self, frame_type: &str, success: bool, duration: Duration, bytes: usize) {
        let result = if success { "success" } else { "failure" };
        self.aead_encryptions
            .with_label_values(&[frame_type, result])
            .inc();
        
        self.aead_encryption_time_us.observe(duration.as_micros() as f64);
        
        if success {
            self.aead_bytes_encrypted.inc_by(bytes as f64);
        }
    }
    
    /// Record AEAD decryption operation
    pub fn record_aead_decryption(&self, frame_type: &str, success: bool, duration: Duration, bytes: usize) {
        let result = if success { "success" } else { "failure" };
        self.aead_decryptions
            .with_label_values(&[frame_type, result])
            .inc();
        
        self.aead_decryption_time_us.observe(duration.as_micros() as f64);
        
        if success {
            self.aead_bytes_decrypted.inc_by(bytes as f64);
        } else {
            self.aead_auth_failures.inc();
        }
    }
    
    /// Record AEAD key rotation
    pub fn record_aead_key_rotation(&self) {
        self.aead_key_rotations.inc();
    }
    
    /// Update active AEAD sessions count
    pub fn update_aead_active_sessions(&self, count: i64) {
        self.aead_active_sessions.set(count);
    }
    
    /// Update HTX connection count
    pub fn update_htx_connections(&self, protocol: &str, state: &str, count: i64) {
        self.htx_connections
            .with_label_values(&[protocol, state])
            .set(count);
    }
    
    /// Record HTX request
    pub fn record_htx_request(&self, method: &str, status: &str, protocol: &str, response_time: Duration) {
        self.htx_requests
            .with_label_values(&[method, status, protocol])
            .inc();
        
        self.htx_response_time
            .with_label_values(&[method, status])
            .observe(response_time.as_secs_f64());
    }
    
    /// Record HTX bytes transferred
    pub fn record_htx_bytes(&self, direction: &str, protocol: &str, bytes: f64) {
        self.htx_bytes_transferred
            .with_label_values(&[direction, protocol])
            .inc_by(bytes);
    }
    
    /// Record encryption operation
    pub fn record_encryption_operation(&self, operation: &str, success: bool) {
        let result = if success { "success" } else { "failure" };
        self.encryption_operations
            .with_label_values(&[operation, result])
            .inc();
    }
    
    /// Update compression ratio
    pub fn update_compression_ratio(&self, algorithm: &str, ratio: f64) {
        self.compression_ratio
            .with_label_values(&[algorithm])
            .set(ratio);
    }
    
    /// Record compression time
    pub fn record_compression_time(&self, duration: Duration) {
        self.compression_time_us.observe(duration.as_micros() as f64);
    }
    
    /// Update system memory usage
    pub fn update_memory_usage(&self, bytes: f64) {
        self.memory_usage_bytes.set(bytes);
    }
    
    /// Update CPU usage
    pub fn update_cpu_usage(&self, percent: f64) {
        self.cpu_usage_percent.set(percent);
    }
    
    /// Update disk usage
    pub fn update_disk_usage(&self, path: &str, usage_type: &str, bytes: f64) {
        self.disk_usage_bytes
            .with_label_values(&[path, usage_type])
            .set(bytes);
    }
    
    /// Record network error
    pub fn record_network_error(&self, error_type: &str, component: &str) {
        self.network_errors
            .with_label_values(&[error_type, component])
            .inc();
    }
    
    /// Update performance KPIs
    pub fn update_performance_kpis(
        &self,
        throughput_ppm: f64,
        p95_latency_ms: f64, 
        availability_percent: f64,
        error_rate_percent: f64,
    ) {
        self.throughput_packets_per_minute.set(throughput_ppm);
        self.p95_latency_ms.set(p95_latency_ms);
        self.availability_percent.set(availability_percent);
        self.error_rate_percent.set(error_rate_percent);
    }
    
    /// Get uptime in seconds
    pub fn uptime_seconds(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }
    
    /// Batch update metrics (for performance)
    pub async fn batch_update<F>(&self, update_fn: F)
    where
        F: FnOnce(&mut HashMap<String, f64>),
    {
        let mut batch = self.batch_updates.write().await;
        update_fn(&mut *batch);
        
        // Apply batch updates if collection is enabled
        if self.config.enable_detailed {
            // Apply batched updates to actual metrics
            for (key, value) in batch.drain() {
                match key.as_str() {
                    "memory_usage" => self.memory_usage_bytes.set(*value),
                    "cpu_usage" => self.cpu_usage_percent.set(*value),
                    _ => debug!(key = key, value = value, "Unknown batch metric"),
                }
            }
        }
    }
    
    /// Stop metrics collector
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping metrics collector");
        // Metrics collector doesn't have background tasks to stop
        // Registry cleanup is handled by Drop trait
        Ok(())
    }
}

/// Handle HTTP requests to metrics endpoint
async fn handle_metrics_request(
    req: Request<Body>,
    registry: Registry,
) -> Result<Response<Body>, hyper::Error> {
    match (req.method(), req.uri().path()) {
        (&Method::GET, "/metrics") => {
            let encoder = TextEncoder::new();
            let metric_families = registry.gather();
            
            match encoder.encode_to_string(&metric_families) {
                Ok(metrics_text) => {
                    Ok(Response::builder()
                        .header("Content-Type", encoder.format_type())
                        .body(Body::from(metrics_text))
                        .unwrap())
                }
                Err(e) => {
                    error!(error = ?e, "Failed to encode metrics");
                    Ok(Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .body(Body::from("Internal server error"))
                        .unwrap())
                }
            }
        }
        
        (&Method::GET, "/health") => {
            Ok(Response::builder()
                .body(Body::from("OK"))
                .unwrap())
        }
        
        _ => {
            Ok(Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(Body::from("Not found"))
                .unwrap())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    fn create_test_collector() -> MetricsCollector {
        let config = Arc::new(GatewayConfig::default());
        MetricsCollector::new(config).unwrap()
    }
    
    #[test]
    fn test_metrics_initialization() {
        let collector = create_test_collector();
        
        // Verify metrics are registered
        let families = collector.registry.gather();
        assert!(!families.is_empty());
        
        // Check for key metrics
        let metric_names: Vec<String> = families
            .iter()
            .map(|f| f.get_name().to_string())
            .collect();
        
        assert!(metric_names.contains(&"betanet_gateway_packets_sent_total".to_string()));
        assert!(metric_names.contains(&"betanet_gateway_sequence_validations_total".to_string()));
        assert!(metric_names.contains(&"betanet_gateway_throughput_packets_per_minute".to_string()));
    }
    
    #[test]
    fn test_packet_metrics() {
        let collector = create_test_collector();
        
        // Record some packet metrics
        collector.record_packet_sent("1-ff00:0:110", "path1", true);
        collector.record_packet_received("1-ff00:0:120", "path2", false);
        collector.record_packet_dropped("invalid_checksum", "path1");
        
        // Verify metrics were recorded
        let families = collector.registry.gather();
        let packets_sent_family = families
            .iter()
            .find(|f| f.get_name() == "betanet_gateway_packets_sent_total")
            .unwrap();
        
        assert_eq!(packets_sent_family.get_metric().len(), 1);
        assert_eq!(packets_sent_family.get_metric()[0].get_counter().get_value(), 1.0);
    }
    
    #[test]
    fn test_timing_metrics() {
        let collector = create_test_collector();
        
        let duration = Duration::from_millis(50);
        collector.record_processing_time("encapsulation", "path1", duration);
        collector.record_compression_time(Duration::from_micros(1500));
        
        // Verify histogram metrics were recorded
        let families = collector.registry.gather();
        let processing_time_family = families
            .iter()
            .find(|f| f.get_name() == "betanet_gateway_packet_processing_seconds")
            .unwrap();
        
        assert_eq!(processing_time_family.get_metric().len(), 1);
        assert_eq!(
            processing_time_family.get_metric()[0].get_histogram().get_sample_count(),
            1
        );
    }
    
    #[test]
    fn test_kpi_metrics() {
        let collector = create_test_collector();
        
        collector.update_performance_kpis(
            500000.0, // 500k packets per minute
            125.5,    // 125.5ms p95 latency
            99.9,     // 99.9% availability
            0.1,      // 0.1% error rate
        );
        
        // Check that KPI metrics were updated
        let families = collector.registry.gather();
        let throughput_family = families
            .iter()
            .find(|f| f.get_name() == "betanet_gateway_throughput_packets_per_minute")
            .unwrap();
        
        assert_eq!(throughput_family.get_metric()[0].get_gauge().get_value(), 500000.0);
    }
    
    #[tokio::test]
    async fn test_batch_updates() {
        let collector = create_test_collector();
        
        // Perform batch update
        collector.batch_update(|batch| {
            batch.insert("memory_usage".to_string(), 1024.0 * 1024.0 * 100.0); // 100MB
            batch.insert("cpu_usage".to_string(), 45.5); // 45.5%
        }).await;
        
        // Verify batch updates were applied
        let families = collector.registry.gather();
        let memory_family = families
            .iter()
            .find(|f| f.get_name() == "betanet_gateway_memory_usage_bytes")
            .unwrap();
        
        assert_eq!(memory_family.get_metric()[0].get_gauge().get_value(), 1024.0 * 1024.0 * 100.0);
    }
}