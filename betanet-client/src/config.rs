//! Configuration management for Betanet client

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

/// Main Betanet client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetanetConfig {
    /// Node identification
    pub node_id: String,
    /// Transport configuration
    pub transport: TransportConfig,
    /// Security configuration
    pub security: SecurityConfig,
    /// Chrome fingerprint configuration
    pub chrome_fingerprint: ChromeFingerprintConfig,
    /// Gateway configuration
    pub gateway: GatewayConfig,
    /// Performance tuning
    pub performance: PerformanceConfig,
    /// Integration settings
    pub integration: IntegrationConfig,
}

impl Default for BetanetConfig {
    fn default() -> Self {
        Self {
            node_id: format!("betanet_rust_{}", uuid::Uuid::new_v4()),
            transport: TransportConfig::default(),
            security: SecurityConfig::default(),
            chrome_fingerprint: ChromeFingerprintConfig::default(),
            gateway: GatewayConfig::default(),
            performance: PerformanceConfig::default(),
            integration: IntegrationConfig::default(),
        }
    }
}

/// Transport layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportConfig {
    /// Primary QUIC configuration
    pub quic: QuicConfig,
    /// Fallback TCP configuration
    pub tcp: TcpConfig,
    /// HTTP transport settings
    pub http: HttpConfig,
    /// Fallback behavior
    pub fallback: FallbackConfig,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            quic: QuicConfig::default(),
            tcp: TcpConfig::default(),
            http: HttpConfig::default(),
            fallback: FallbackConfig::default(),
        }
    }
}

/// QUIC transport configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuicConfig {
    /// Listen address for QUIC
    pub listen_addr: SocketAddr,
    /// Maximum concurrent streams
    pub max_concurrent_streams: u32,
    /// Connection idle timeout
    pub idle_timeout: Duration,
    /// Keep alive interval
    pub keep_alive_interval: Duration,
    /// Maximum packet size
    pub max_packet_size: u16,
}

impl Default for QuicConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:4001".parse().unwrap(),
            max_concurrent_streams: 100,
            idle_timeout: Duration::from_secs(30),
            keep_alive_interval: Duration::from_secs(10),
            max_packet_size: 1350,
        }
    }
}

/// TCP transport configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TcpConfig {
    /// Listen address for TCP
    pub listen_addr: SocketAddr,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Read timeout
    pub read_timeout: Duration,
    /// Write timeout
    pub write_timeout: Duration,
    /// TCP keepalive
    pub keepalive: bool,
    /// TCP nodelay
    pub nodelay: bool,
}

impl Default for TcpConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:4002".parse().unwrap(),
            connect_timeout: Duration::from_secs(10),
            read_timeout: Duration::from_secs(30),
            write_timeout: Duration::from_secs(30),
            keepalive: true,
            nodelay: true,
        }
    }
}

/// HTTP transport configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpConfig {
    /// User agent string for Chrome mimicry
    pub user_agent: String,
    /// HTTP version preference
    pub version_preference: HttpVersion,
    /// Request timeout
    pub request_timeout: Duration,
    /// Connection pool size
    pub connection_pool_size: usize,
    /// Custom headers for covert transport
    pub covert_headers: Vec<(String, String)>,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36".to_string(),
            version_preference: HttpVersion::H2,
            request_timeout: Duration::from_secs(30),
            connection_pool_size: 10,
            covert_headers: vec![
                ("Accept".to_string(), "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8".to_string()),
                ("Accept-Language".to_string(), "en-US,en;q=0.5".to_string()),
                ("Accept-Encoding".to_string(), "gzip, deflate, br".to_string()),
            ],
        }
    }
}

/// HTTP version preference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HttpVersion {
    /// HTTP/1.1
    H1,
    /// HTTP/2
    H2,
    /// HTTP/3 (over QUIC)
    H3,
}

/// Fallback transport configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackConfig {
    /// Enable automatic fallback
    pub enabled: bool,
    /// Fallback trigger thresholds
    pub failure_threshold: u32,
    /// Fallback timeout before retry
    pub fallback_timeout: Duration,
    /// Maximum fallback attempts
    pub max_fallback_attempts: u32,
    /// Health check interval
    pub health_check_interval: Duration,
}

impl Default for FallbackConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            failure_threshold: 3,
            fallback_timeout: Duration::from_secs(5),
            max_fallback_attempts: 3,
            health_check_interval: Duration::from_secs(60),
        }
    }
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Noise protocol configuration
    pub noise: NoiseConfig,
    /// TLS configuration
    pub tls: TlsConfig,
    /// Peer verification settings
    pub peer_verification: PeerVerificationConfig,
    /// Rate limiting
    pub rate_limiting: RateLimitingConfig,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            noise: NoiseConfig::default(),
            tls: TlsConfig::default(),
            peer_verification: PeerVerificationConfig::default(),
            rate_limiting: RateLimitingConfig::default(),
        }
    }
}

/// Noise protocol configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseConfig {
    /// Noise pattern (XK, XX, etc.)
    pub pattern: String,
    /// Private key file path
    pub private_key_file: PathBuf,
    /// Enable forward secrecy
    pub forward_secrecy: bool,
    /// Key rotation interval
    pub key_rotation_interval: Duration,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            pattern: "Noise_XK_25519_AESGCM_SHA256".to_string(),
            private_key_file: PathBuf::from("./data/noise_private_key.pem"),
            forward_secrecy: true,
            key_rotation_interval: Duration::from_secs(3600), // 1 hour
        }
    }
}

/// TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Certificate file path
    pub cert_file: PathBuf,
    /// Private key file path
    pub key_file: PathBuf,
    /// CA certificate file path
    pub ca_file: Option<PathBuf>,
    /// Enable mutual TLS
    pub mutual_tls: bool,
    /// TLS version minimum
    pub min_version: String,
    /// Enable session resumption
    pub session_resumption: bool,
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            cert_file: PathBuf::from("./data/tls_cert.pem"),
            key_file: PathBuf::from("./data/tls_key.pem"),
            ca_file: None,
            mutual_tls: true,
            min_version: "TLSv1.3".to_string(),
            session_resumption: true,
        }
    }
}

/// Peer verification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerVerificationConfig {
    /// Enable peer reputation system
    pub reputation_enabled: bool,
    /// Minimum trust score threshold
    pub min_trust_score: f64,
    /// Trust score decay rate
    pub trust_decay_rate: f64,
    /// Blocklist persistence
    pub blocklist_persistence: bool,
}

impl Default for PeerVerificationConfig {
    fn default() -> Self {
        Self {
            reputation_enabled: true,
            min_trust_score: 0.3,
            trust_decay_rate: 0.1,
            blocklist_persistence: true,
        }
    }
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitingConfig {
    /// Maximum connections per minute
    pub max_connections_per_minute: u32,
    /// Maximum messages per minute
    pub max_messages_per_minute: u32,
    /// Rate limit window duration
    pub window_duration: Duration,
    /// Enable adaptive rate limiting
    pub adaptive: bool,
}

impl Default for RateLimitingConfig {
    fn default() -> Self {
        Self {
            max_connections_per_minute: 60,
            max_messages_per_minute: 1000,
            window_duration: Duration::from_secs(60),
            adaptive: true,
        }
    }
}

/// Chrome fingerprint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromeFingerprintConfig {
    /// Enable Chrome fingerprint mimicry
    pub enabled: bool,
    /// Target Chrome version
    pub target_version: String,
    /// Target platform
    pub target_platform: String,
    /// TLS fingerprint configuration
    pub tls_fingerprint: TlsFingerprintConfig,
    /// HTTP fingerprint configuration
    pub http_fingerprint: HttpFingerprintConfig,
}

impl Default for ChromeFingerprintConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            target_version: "120.0.6099.109".to_string(),
            target_platform: "Windows".to_string(),
            tls_fingerprint: TlsFingerprintConfig::default(),
            http_fingerprint: HttpFingerprintConfig::default(),
        }
    }
}

/// TLS fingerprint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsFingerprintConfig {
    /// Cipher suites to mimic
    pub cipher_suites: Vec<String>,
    /// Extensions to include
    pub extensions: Vec<String>,
    /// Elliptic curves
    pub elliptic_curves: Vec<String>,
    /// Signature algorithms
    pub signature_algorithms: Vec<String>,
}

impl Default for TlsFingerprintConfig {
    fn default() -> Self {
        Self {
            cipher_suites: vec![
                "TLS_AES_128_GCM_SHA256".to_string(),
                "TLS_AES_256_GCM_SHA384".to_string(),
                "TLS_CHACHA20_POLY1305_SHA256".to_string(),
            ],
            extensions: vec![
                "server_name".to_string(),
                "supported_groups".to_string(),
                "signature_algorithms".to_string(),
                "key_share".to_string(),
            ],
            elliptic_curves: vec![
                "x25519".to_string(),
                "secp256r1".to_string(),
                "secp384r1".to_string(),
            ],
            signature_algorithms: vec![
                "rsa_pss_rsae_sha256".to_string(),
                "ecdsa_secp256r1_sha256".to_string(),
                "rsa_pss_rsae_sha384".to_string(),
            ],
        }
    }
}

/// HTTP fingerprint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpFingerprintConfig {
    /// Header order to mimic
    pub header_order: Vec<String>,
    /// Connection behavior settings
    pub connection_behavior: ConnectionBehaviorConfig,
    /// Request timing patterns
    pub timing_patterns: TimingPatternsConfig,
}

impl Default for HttpFingerprintConfig {
    fn default() -> Self {
        Self {
            header_order: vec![
                "Host".to_string(),
                "Connection".to_string(),
                "User-Agent".to_string(),
                "Accept".to_string(),
                "Accept-Language".to_string(),
                "Accept-Encoding".to_string(),
            ],
            connection_behavior: ConnectionBehaviorConfig::default(),
            timing_patterns: TimingPatternsConfig::default(),
        }
    }
}

/// Connection behavior configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionBehaviorConfig {
    /// Connection reuse probability
    pub connection_reuse_probability: f64,
    /// Pipelining behavior
    pub enable_pipelining: bool,
    /// Concurrent connections limit
    pub max_concurrent_connections: u32,
}

impl Default for ConnectionBehaviorConfig {
    fn default() -> Self {
        Self {
            connection_reuse_probability: 0.8,
            enable_pipelining: true,
            max_concurrent_connections: 6,
        }
    }
}

/// Request timing patterns configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingPatternsConfig {
    /// Base request delay
    pub base_delay_ms: u64,
    /// Delay variance
    pub delay_variance_ms: u64,
    /// Think time between requests
    pub think_time_ms: u64,
}

impl Default for TimingPatternsConfig {
    fn default() -> Self {
        Self {
            base_delay_ms: 50,
            delay_variance_ms: 20,
            think_time_ms: 100,
        }
    }
}

/// Gateway configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    /// Enable SCION-ish gateway functionality
    pub enabled: bool,
    /// Gateway listen address
    pub listen_addr: SocketAddr,
    /// Local AS number
    pub local_as: u32,
    /// Gateway port
    pub gateway_port: u16,
    /// Beacon interval in seconds
    pub beacon_interval_secs: u64,
    /// Path timeout in seconds
    pub path_timeout_secs: u64,
    /// CBOR control configuration
    pub cbor_control: CborControlConfig,
    /// Path validation settings
    pub path_validation: PathValidationConfig,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            listen_addr: "0.0.0.0:4003".parse().unwrap(),
            local_as: 64512, // Default private AS number
            gateway_port: 8080,
            beacon_interval_secs: 30,
            path_timeout_secs: 300,
            cbor_control: CborControlConfig::default(),
            path_validation: PathValidationConfig::default(),
        }
    }
}

/// CBOR control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CborControlConfig {
    /// Ed25519 signing key file
    pub signing_key_file: PathBuf,
    /// Verification public keys
    pub verification_keys: Vec<String>,
    /// Message TTL
    pub message_ttl_seconds: u32,
}

impl Default for CborControlConfig {
    fn default() -> Self {
        Self {
            signing_key_file: PathBuf::from("./data/ed25519_signing_key.pem"),
            verification_keys: Vec::new(),
            message_ttl_seconds: 300,
        }
    }
}

/// Path validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathValidationConfig {
    /// Enable path validation
    pub enabled: bool,
    /// Maximum path length
    pub max_path_length: u32,
    /// Path verification timeout
    pub verification_timeout: Duration,
}

impl Default for PathValidationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_path_length: 10,
            verification_timeout: Duration::from_secs(5),
        }
    }
}

/// Performance tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Buffer sizes
    pub buffer_sizes: BufferSizesConfig,
    /// Concurrency limits
    pub concurrency: ConcurrencyConfig,
    /// Caching configuration
    pub caching: CachingConfig,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            buffer_sizes: BufferSizesConfig::default(),
            concurrency: ConcurrencyConfig::default(),
            caching: CachingConfig::default(),
        }
    }
}

/// Buffer sizes configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferSizesConfig {
    /// Read buffer size
    pub read_buffer_size: usize,
    /// Write buffer size
    pub write_buffer_size: usize,
    /// Message queue size
    pub message_queue_size: usize,
}

impl Default for BufferSizesConfig {
    fn default() -> Self {
        Self {
            read_buffer_size: 65536,  // 64KB
            write_buffer_size: 65536, // 64KB
            message_queue_size: 1000,
        }
    }
}

/// Concurrency configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrencyConfig {
    /// Number of worker threads
    pub worker_threads: usize,
    /// Maximum concurrent tasks
    pub max_concurrent_tasks: usize,
    /// Task spawn rate limit
    pub task_spawn_rate_limit: u32,
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            worker_threads: num_cpus::get(),
            max_concurrent_tasks: 1000,
            task_spawn_rate_limit: 100,
        }
    }
}

/// Caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingConfig {
    /// Enable message caching
    pub message_cache_enabled: bool,
    /// Message cache size
    pub message_cache_size: usize,
    /// Message cache TTL
    pub message_cache_ttl: Duration,
    /// Enable peer cache
    pub peer_cache_enabled: bool,
    /// Peer cache size
    pub peer_cache_size: usize,
}

impl Default for CachingConfig {
    fn default() -> Self {
        Self {
            message_cache_enabled: true,
            message_cache_size: 10000,
            message_cache_ttl: Duration::from_secs(300),
            peer_cache_enabled: true,
            peer_cache_size: 1000,
        }
    }
}

/// Integration configuration with existing AIVillage systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Python FFI integration
    pub python_ffi: PythonFfiConfig,
    /// Navigator agent integration
    pub navigator: NavigatorConfig,
    /// Metrics and monitoring
    pub monitoring: MonitoringConfig,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            python_ffi: PythonFfiConfig::default(),
            navigator: NavigatorConfig::default(),
            monitoring: MonitoringConfig::default(),
        }
    }
}

/// Python FFI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonFfiConfig {
    /// Enable Python FFI bridge
    pub enabled: bool,
    /// Bridge listen address
    pub bridge_addr: SocketAddr,
    /// Maximum message size
    pub max_message_size: usize,
}

impl Default for PythonFfiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bridge_addr: "127.0.0.1:4004".parse().unwrap(),
            max_message_size: 1048576, // 1MB
        }
    }
}

/// Navigator agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigatorConfig {
    /// Navigator endpoint URL
    pub endpoint: String,
    /// Authentication token
    pub auth_token: Option<String>,
    /// Request timeout
    pub timeout: Duration,
}

impl Default for NavigatorConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:8080/navigator".to_string(),
            auth_token: None,
            timeout: Duration::from_secs(10),
        }
    }
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable Prometheus metrics
    pub prometheus_enabled: bool,
    /// Prometheus listen address
    pub prometheus_addr: SocketAddr,
    /// Enable tracing
    pub tracing_enabled: bool,
    /// Tracing level
    pub tracing_level: String,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            prometheus_enabled: true,
            prometheus_addr: "127.0.0.1:9090".parse().unwrap(),
            tracing_enabled: true,
            tracing_level: "info".to_string(),
        }
    }
}

impl BetanetConfig {
    /// Load configuration from file
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .context("Failed to read configuration file")?;

        toml::from_str(&content)
            .context("Failed to parse configuration file")
    }

    /// Save configuration to file
    pub fn to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .context("Failed to serialize configuration")?;

        std::fs::write(path, content)
            .context("Failed to write configuration file")
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate node ID
        if self.node_id.is_empty() {
            anyhow::bail!("Node ID cannot be empty");
        }

        // Validate transport configuration
        if self.transport.quic.max_concurrent_streams == 0 {
            anyhow::bail!("QUIC max_concurrent_streams must be > 0");
        }

        // Validate security configuration
        if self.security.peer_verification.min_trust_score < 0.0 ||
           self.security.peer_verification.min_trust_score > 1.0 {
            anyhow::bail!("Trust score must be between 0.0 and 1.0");
        }

        // Validate performance configuration
        if self.performance.concurrency.worker_threads == 0 {
            anyhow::bail!("Worker threads must be > 0");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BetanetConfig::default();
        assert!(config.validate().is_ok());
        assert!(!config.node_id.is_empty());
        assert!(config.transport.quic.max_concurrent_streams > 0);
    }

    #[test]
    fn test_config_serialization() {
        let config = BetanetConfig::default();
        let serialized = toml::to_string(&config).unwrap();
        let deserialized: BetanetConfig = toml::from_str(&serialized).unwrap();

        assert_eq!(config.node_id, deserialized.node_id);
        assert_eq!(config.transport.quic.listen_addr, deserialized.transport.quic.listen_addr);
    }
}
