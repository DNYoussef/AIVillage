//! Mixnode configuration

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Mixnode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixnodeConfig {
    /// Listen address
    pub listen_addr: SocketAddr,

    /// Private key file path
    pub private_key_file: Option<PathBuf>,

    /// Number of layers in the mix network
    pub layers: u8,

    /// Enable Sphinx packet processing
    pub enable_sphinx: bool,

    /// Enable VRF-based delays
    pub enable_vrf: bool,

    /// Enable cover traffic generation
    pub enable_cover_traffic: bool,

    /// Minimum delay for packet processing
    pub min_delay: Duration,

    /// Maximum delay for packet processing
    pub max_delay: Duration,

    /// Cover traffic interval
    pub cover_traffic_interval: Duration,

    /// Maximum packet queue size
    pub max_queue_size: usize,

    /// Connection timeout
    pub connection_timeout: Duration,

    /// Network buffer size
    pub buffer_size: usize,
}

impl Default for MixnodeConfig {
    fn default() -> Self {
        Self {
            listen_addr: "127.0.0.1:9001".parse().unwrap(),
            private_key_file: None,
            layers: 3,
            enable_sphinx: true,
            enable_vrf: true,
            enable_cover_traffic: false,
            min_delay: Duration::from_millis(100),
            max_delay: Duration::from_millis(1000),
            cover_traffic_interval: Duration::from_secs(10),
            max_queue_size: 1000,
            connection_timeout: Duration::from_secs(30),
            buffer_size: 8192,
        }
    }
}

impl MixnodeConfig {
    /// Load configuration from file
    pub fn load_from_file(path: &PathBuf) -> crate::Result<Self> {
        let contents = std::fs::read_to_string(path).map_err(crate::MixnodeError::Io)?;
        let config = serde_json::from_str(&contents)
            .map_err(|e| crate::MixnodeError::Config(format!("Failed to parse config: {}", e)))?;
        Ok(config)
    }

    /// Save configuration to file
    pub fn save_to_file(&self, path: &PathBuf) -> crate::Result<()> {
        let contents = serde_json::to_string_pretty(self).map_err(|e| {
            crate::MixnodeError::Config(format!("Failed to serialize config: {}", e))
        })?;
        std::fs::write(path, contents).map_err(crate::MixnodeError::Io)?;
        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> crate::Result<()> {
        if self.layers == 0 {
            return Err(crate::MixnodeError::Config(
                "Layers must be > 0".to_string(),
            ));
        }

        if self.min_delay >= self.max_delay {
            return Err(crate::MixnodeError::Config(
                "min_delay must be < max_delay".to_string(),
            ));
        }

        if self.max_queue_size == 0 {
            return Err(crate::MixnodeError::Config(
                "max_queue_size must be > 0".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MixnodeConfig::default();
        assert_eq!(config.layers, 3);
        assert!(config.enable_sphinx);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        let mut config = MixnodeConfig::default();
        config.layers = 0;
        assert!(config.validate().is_err());

        config.layers = 3;
        config.min_delay = Duration::from_secs(2);
        config.max_delay = Duration::from_secs(1);
        assert!(config.validate().is_err());
    }
}
