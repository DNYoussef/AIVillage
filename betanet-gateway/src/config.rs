// Configuration management for Betanet Gateway
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

/// Complete gateway configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    pub htx: HTXConfig,
    pub scion: ScionConfig,
    pub anti_replay: AntiReplayConfig,
    pub aead: AeadConfig,
    pub multipath: MultipathConfig,
    pub metrics: MetricsConfig,
    pub performance: PerformanceConfig,
    pub security: SecurityConfig,
}

/// HTX server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HTXConfig {
    /// Server bind address
    pub bind_addr: SocketAddr,
    
    /// TLS certificate path
    pub cert_path: Option<PathBuf>,
    
    /// TLS private key path
    pub key_path: Option<PathBuf>,
    
    /// Enable QUIC transport
    pub enable_quic: bool,
    
    /// Enable HTTP/3
    pub enable_h3: bool,
    
    /// Maximum frame size in bytes
    pub max_frame_size: usize,
    
    /// Connection timeout
    pub connection_timeout: Duration,
    
    /// Keep-alive interval
    pub keep_alive_interval: Duration,
    
    /// Maximum concurrent connections
    pub max_connections: usize,
}

/// SCION sidecar client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScionConfig {
    /// SCION sidecar gRPC address
    pub address: SocketAddr,
    
    /// Connection timeout
    pub connect_timeout: Duration,
    
    /// Request timeout
    pub request_timeout: Duration,
    
    /// Connection keep-alive interval
    pub keep_alive_interval: Duration,
    
    /// Enable gRPC compression
    pub enable_compression: bool,
    
    /// Maximum message size
    pub max_message_size: usize,
    
    /// Retry configuration
    pub retry: RetryConfig,
}

/// Anti-replay protection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntiReplayConfig {
    /// Database path for persistence
    pub db_path: PathBuf,
    
    /// Sliding window size in bits
    pub window_size: usize,
    
    /// Window cleanup TTL
    pub cleanup_ttl: Duration,
    
    /// Background cleanup interval
    pub cleanup_interval: Duration,
    
    /// Database sync interval
    pub sync_interval: Duration,
    
    /// Maximum sequence age before rejection
    pub max_sequence_age: Duration,
}

/// AEAD encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AeadConfig {
    /// Maximum bytes processed per key before rotation
    pub max_bytes_per_key: u64,
    
    /// Maximum time per key before rotation
    pub max_time_per_key: Duration,
}

/// Multipath management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipathConfig {
    /// Path quality measurement interval
    pub measurement_interval: Duration,
    
    /// Path failover threshold (RTT multiplier)
    pub failover_rtt_threshold: f64,
    
    /// Path failover threshold (loss rate)
    pub failover_loss_threshold: f64,
    
    /// Minimum paths to maintain
    pub min_paths: usize,
    
    /// Maximum paths to track
    pub max_paths: usize,
    
    /// Path exploration probability (0.0-1.0)
    pub exploration_probability: f64,
    
    /// Path quality smoothing factor (EWMA alpha)
    pub quality_smoothing: f64,
}

/// Metrics and telemetry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Metrics server bind address
    pub bind_addr: SocketAddr,
    
    /// Enable detailed metrics
    pub enable_detailed: bool,
    
    /// Metrics collection interval
    pub collection_interval: Duration,
    
    /// Maximum metric label cardinality
    pub max_label_cardinality: usize,
    
    /// Enable metric compression
    pub enable_compression: bool,
}

/// Performance tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Worker thread pool size
    pub worker_threads: usize,
    
    /// I/O buffer size
    pub io_buffer_size: usize,
    
    /// Maximum concurrent operations
    pub max_concurrent_ops: usize,
    
    /// Channel buffer sizes
    pub channel_buffer_size: usize,
    
    /// Batch processing size
    pub batch_size: usize,
    
    /// Memory pool configuration
    pub memory_pool: MemoryPoolConfig,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable ChaCha20-Poly1305 AEAD
    pub enable_aead: bool,
    
    /// Key derivation parameters
    pub key_derivation: KeyDerivationConfig,
    
    /// Rate limiting configuration
    pub rate_limiting: RateLimitingConfig,
    
    /// Access control lists
    pub access_control: AccessControlConfig,
}

/// Retry policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_attempts: usize,
    
    /// Base retry delay
    pub base_delay: Duration,
    
    /// Maximum retry delay
    pub max_delay: Duration,
    
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    
    /// Enable jitter
    pub enable_jitter: bool,
}

/// Memory pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolConfig {
    /// Enable memory pooling
    pub enabled: bool,
    
    /// Pool initial capacity
    pub initial_capacity: usize,
    
    /// Pool growth factor
    pub growth_factor: f64,
    
    /// Maximum pool size
    pub max_size: usize,
}

/// Key derivation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyDerivationConfig {
    /// HKDF salt
    pub salt: Vec<u8>,
    
    /// Key derivation info context
    pub info: Vec<u8>,
    
    /// Derived key length
    pub key_length: usize,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitingConfig {
    /// Enable rate limiting
    pub enabled: bool,
    
    /// Requests per second per peer
    pub requests_per_second: u64,
    
    /// Burst capacity
    pub burst_capacity: u64,
    
    /// Rate limit window duration
    pub window_duration: Duration,
}

/// Access control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlConfig {
    /// Enable access control
    pub enabled: bool,
    
    /// Allowed peer IDs (empty = allow all)
    pub allowed_peers: Vec<String>,
    
    /// Blocked peer IDs
    pub blocked_peers: Vec<String>,
    
    /// Allowed IP ranges
    pub allowed_ip_ranges: Vec<String>,
    
    /// Blocked IP ranges
    pub blocked_ip_ranges: Vec<String>,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            htx: HTXConfig::default(),
            scion: ScionConfig::default(),
            anti_replay: AntiReplayConfig::default(),
            aead: AeadConfig::default(),
            multipath: MultipathConfig::default(),
            metrics: MetricsConfig::default(),
            performance: PerformanceConfig::default(),
            security: SecurityConfig::default(),
        }
    }
}

impl Default for HTXConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:8443".parse().unwrap(),
            cert_path: None,
            key_path: None,
            enable_quic: true,
            enable_h3: true,
            max_frame_size: 64 * 1024, // 64KB
            connection_timeout: Duration::from_secs(30),
            keep_alive_interval: Duration::from_secs(10),
            max_connections: 1000,
        }
    }
}

impl Default for ScionConfig {
    fn default() -> Self {
        Self {
            address: "127.0.0.1:8080".parse().unwrap(),
            connect_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(30),
            keep_alive_interval: Duration::from_secs(20),
            enable_compression: true,
            max_message_size: 1024 * 1024, // 1MB
            retry: RetryConfig::default(),
        }
    }
}

impl Default for AntiReplayConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("/var/lib/betanet-gateway/replay.db"),
            window_size: 1024,
            cleanup_ttl: Duration::from_hours(24),
            cleanup_interval: Duration::from_hours(1),
            sync_interval: Duration::from_secs(300), // 5 minutes
            max_sequence_age: Duration::from_hours(1),
        }
    }
}

impl Default for AeadConfig {
    fn default() -> Self {
        Self {
            max_bytes_per_key: 1024 * 1024 * 1024, // 1 GiB
            max_time_per_key: Duration::from_secs(3600), // 1 hour
        }
    }
}

impl Default for MultipathConfig {
    fn default() -> Self {
        Self {
            measurement_interval: Duration::from_secs(5),
            failover_rtt_threshold: 2.0, // 2x normal RTT
            failover_loss_threshold: 0.1, // 10% loss
            min_paths: 1,
            max_paths: 10,
            exploration_probability: 0.1, // 10% exploration
            quality_smoothing: 0.125, // Standard TCP alpha
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:9090".parse().unwrap(),
            enable_detailed: true,
            collection_interval: Duration::from_secs(15),
            max_label_cardinality: 10000,
            enable_compression: true,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            worker_threads: num_cpus::get(),
            io_buffer_size: 64 * 1024, // 64KB
            max_concurrent_ops: 1000,
            channel_buffer_size: 1000,
            batch_size: 100,
            memory_pool: MemoryPoolConfig::default(),
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_aead: true,
            key_derivation: KeyDerivationConfig::default(),
            rate_limiting: RateLimitingConfig::default(),
            access_control: AccessControlConfig::default(),
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
            enable_jitter: true,
        }
    }
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            initial_capacity: 100,
            growth_factor: 1.5,
            max_size: 10000,
        }
    }
}

impl Default for KeyDerivationConfig {
    fn default() -> Self {
        Self {
            salt: b"betanet-gateway-salt-v1".to_vec(),
            info: b"betanet-gateway-aead".to_vec(),
            key_length: 32, // 256-bit keys
        }
    }
}

impl Default for RateLimitingConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default for performance
            requests_per_second: 1000,
            burst_capacity: 5000,
            window_duration: Duration::from_secs(1),
        }
    }
}

impl Default for AccessControlConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default
            allowed_peers: Vec::new(),
            blocked_peers: Vec::new(),
            allowed_ip_ranges: Vec::new(),
            blocked_ip_ranges: Vec::new(),
        }
    }
}

impl GatewayConfig {
    /// Load configuration from TOML file
    pub async fn from_file(path: &std::path::Path) -> Result<Self> {
        let contents = tokio::fs::read_to_string(path).await
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;
        
        let config: Self = toml::from_str(&contents)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))?;
        
        config.validate()
            .with_context(|| "Configuration validation failed")?;
        
        Ok(config)
    }
    
    /// Save configuration to TOML file
    pub async fn to_file(&self, path: &std::path::Path) -> Result<()> {
        let contents = toml::to_string_pretty(self)
            .context("Failed to serialize configuration")?;
        
        tokio::fs::write(path, contents).await
            .with_context(|| format!("Failed to write config file: {}", path.display()))?;
        
        Ok(())
    }
    
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        // Validate HTX config
        if self.htx.max_frame_size == 0 {
            anyhow::bail!("HTX max_frame_size must be greater than 0");
        }
        
        if self.htx.max_frame_size > 16 * 1024 * 1024 {
            anyhow::bail!("HTX max_frame_size too large (max 16MB)");
        }
        
        // Validate anti-replay config
        if self.anti_replay.window_size == 0 {
            anyhow::bail!("Anti-replay window_size must be greater than 0");
        }
        
        if self.anti_replay.window_size > 65536 {
            anyhow::bail!("Anti-replay window_size too large (max 65536)");
        }
        
        // Validate multipath config
        if self.multipath.min_paths == 0 {
            anyhow::bail!("Multipath min_paths must be greater than 0");
        }
        
        if self.multipath.min_paths > self.multipath.max_paths {
            anyhow::bail!("Multipath min_paths cannot exceed max_paths");
        }
        
        if !(0.0..=1.0).contains(&self.multipath.exploration_probability) {
            anyhow::bail!("Multipath exploration_probability must be between 0.0 and 1.0");
        }
        
        // Validate performance config
        if self.performance.worker_threads == 0 {
            anyhow::bail!("Performance worker_threads must be greater than 0");
        }
        
        if self.performance.io_buffer_size == 0 {
            anyhow::bail!("Performance io_buffer_size must be greater than 0");
        }
        
        Ok(())
    }
    
    /// Generate example configuration file
    pub fn example_toml() -> String {
        let example = Self::default();
        toml::to_string_pretty(&example).unwrap_or_else(|_| "# Failed to generate example".to_string())
    }
}

// Helper trait for duration serialization
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;
    
    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_secs())
    }
    
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

// Add num_cpus for performance config default
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4) // Fallback to 4 threads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_default_config_validation() {
        let config = GatewayConfig::default();
        assert!(config.validate().is_ok());
    }
    
    #[tokio::test]
    async fn test_config_file_roundtrip() {
        let config = GatewayConfig::default();
        
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        
        // Save and load
        config.to_file(path).await.unwrap();
        let loaded_config = GatewayConfig::from_file(path).await.unwrap();
        
        // Compare (approximate due to floating point)
        assert_eq!(config.htx.bind_addr, loaded_config.htx.bind_addr);
        assert_eq!(config.scion.address, loaded_config.scion.address);
        assert_eq!(config.anti_replay.window_size, loaded_config.anti_replay.window_size);
    }
    
    #[test]
    fn test_config_validation_errors() {
        let mut config = GatewayConfig::default();
        
        // Test invalid frame size
        config.htx.max_frame_size = 0;
        assert!(config.validate().is_err());
        
        // Test invalid window size
        config = GatewayConfig::default();
        config.anti_replay.window_size = 0;
        assert!(config.validate().is_err());
        
        // Test invalid multipath config
        config = GatewayConfig::default();
        config.multipath.min_paths = 10;
        config.multipath.max_paths = 5;
        assert!(config.validate().is_err());
    }
}