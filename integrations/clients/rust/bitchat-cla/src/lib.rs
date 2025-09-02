//! BitChat BLE Convergence Layer Adapter for DTN
//!
//! Implements BLE (Bluetooth Low Energy) as a DTN convergence layer with:
//! - BLE advertising and GATT proxy
//! - Packet fragmentation for 20-100 byte MTU
//! - Forward Error Correction (FEC) with XOR fountain codes
//! - Friendship queues for Low Power Node (LPN) support
//! - TTL-based probabilistic rebroadcast

#![deny(clippy::all)]
#![allow(missing_docs)]
#![allow(dead_code)]

// Module structure (to be implemented)
pub mod ble;
pub mod fec;
pub mod fragmentation;
pub mod friendship;
pub mod rebroadcast;

// Re-exports
pub use ble::BleCla;

/// BitChat BLE Convergence Layer Adapter implementation
pub struct BitChatCla {
    ble_adapter: ble::BleCla,
    fec_enabled: bool,
    fragmentation_mtu: usize,
    friendship_enabled: bool,
    rebroadcast_config: rebroadcast::RebroadcastConfig,
}

impl BitChatCla {
    pub fn new() -> Self {
        Self {
            ble_adapter: ble::BleCla::new(),
            fec_enabled: true,
            fragmentation_mtu: 100, // BLE MTU limit
            friendship_enabled: true,
            rebroadcast_config: rebroadcast::RebroadcastConfig::default(),
        }
    }

    pub fn with_config(
        fec_enabled: bool,
        fragmentation_mtu: usize,
        friendship_enabled: bool,
    ) -> Self {
        Self {
            ble_adapter: ble::BleCla::new(),
            fec_enabled,
            fragmentation_mtu,
            friendship_enabled,
            rebroadcast_config: rebroadcast::RebroadcastConfig::default(),
        }
    }

    pub fn configure_rebroadcast(&mut self, config: rebroadcast::RebroadcastConfig) {
        self.rebroadcast_config = config;
    }

    pub fn get_adapter(&self) -> &ble::BleCla {
        &self.ble_adapter
    }
}

impl Default for BitChatCla {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_placeholder() {
        let _cla = BitChatCla::new();
    }
}
