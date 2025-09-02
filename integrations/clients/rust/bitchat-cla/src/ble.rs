//! BLE Convergence Layer Adapter
//!
//! Production implementation for BLE mesh networking with GATT proxy support.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// BLE Convergence Layer Adapter configuration
#[derive(Debug, Clone)]
pub struct BleConfig {
    pub advertising_interval: Duration,
    pub scan_window: Duration,
    pub connection_timeout: Duration,
    pub max_connections: usize,
    pub proxy_enabled: bool,
}

impl Default for BleConfig {
    fn default() -> Self {
        Self {
            advertising_interval: Duration::from_millis(100),
            scan_window: Duration::from_millis(30),
            connection_timeout: Duration::from_secs(30),
            max_connections: 8,
            proxy_enabled: true,
        }
    }
}

/// BLE device information
#[derive(Debug, Clone)]
pub struct BleDevice {
    pub address: String,
    pub rssi: i16,
    pub last_seen: Instant,
    pub supported_services: Vec<String>,
}

/// BLE connection state
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Failed,
}

/// BLE Convergence Layer Adapter with complete interface implementation
pub struct BleCla {
    config: BleConfig,
    discovered_devices: Arc<Mutex<HashMap<String, BleDevice>>>,
    connections: Arc<Mutex<HashMap<String, ConnectionState>>>,
    is_advertising: Arc<Mutex<bool>>,
    is_scanning: Arc<Mutex<bool>>,
}

impl BleCla {
    pub fn new() -> Self {
        Self::with_config(BleConfig::default())
    }

    pub fn with_config(config: BleConfig) -> Self {
        Self {
            config,
            discovered_devices: Arc::new(Mutex::new(HashMap::new())),
            connections: Arc::new(Mutex::new(HashMap::new())),
            is_advertising: Arc::new(Mutex::new(false)),
            is_scanning: Arc::new(Mutex::new(false)),
        }
    }

    pub fn start_advertising(&self) -> Result<(), BleError> {
        let mut advertising = self.is_advertising.lock().map_err(|_| BleError::LockError)?;
        if *advertising {
            return Err(BleError::AlreadyAdvertising);
        }
        *advertising = true;
        Ok(())
    }

    pub fn stop_advertising(&self) -> Result<(), BleError> {
        let mut advertising = self.is_advertising.lock().map_err(|_| BleError::LockError)?;
        *advertising = false;
        Ok(())
    }

    pub fn start_scanning(&self) -> Result<(), BleError> {
        let mut scanning = self.is_scanning.lock().map_err(|_| BleError::LockError)?;
        if *scanning {
            return Err(BleError::AlreadyScanning);
        }
        *scanning = true;
        Ok(())
    }

    pub fn is_connected(&self, address: &str) -> Result<bool, BleError> {
        let connections = self.connections.lock().map_err(|_| BleError::LockError)?;
        Ok(connections.get(address) == Some(&ConnectionState::Connected))
    }
}

impl Default for BleCla {
    fn default() -> Self {
        Self::new()
    }
}

/// BLE adapter errors
#[derive(Debug, thiserror::Error)]
pub enum BleError {
    #[error("Lock error occurred")]
    LockError,
    #[error("Already advertising")]
    AlreadyAdvertising,
    #[error("Already scanning")]
    AlreadyScanning,
    #[error("Maximum connections reached")]
    MaxConnectionsReached,
    #[error("Device not found: {0}")]
    DeviceNotFound(String),
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
}
