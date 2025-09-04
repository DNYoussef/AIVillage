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
    fragmentation: fragmentation::FragmentationEngine,
    fec_encoder: Option<fec::FecEncoder>,
    fec_decoder: Option<fec::FecDecoder>,
    friendship: Option<friendship::FriendshipEngine>,
    rebroadcast: rebroadcast::RebroadcastEngine,
}

impl BitChatCla {
    pub fn new() -> Self {
        Self::with_config(true, 100, true)
    }

    pub fn with_config(
        fec_enabled: bool,
        fragmentation_mtu: usize,
        friendship_enabled: bool,
    ) -> Self {
        let _ = fragmentation_mtu; // MTU reserved for future use
        Self {
            ble_adapter: ble::BleCla::new(),
            fragmentation: fragmentation::FragmentationEngine::new(),
            fec_encoder: if fec_enabled {
                Some(fec::FecEncoder::new(fec::FecConfig::default()))
            } else {
                None
            },
            fec_decoder: if fec_enabled {
                Some(fec::FecDecoder::new(fec::FecConfig::default()))
            } else {
                None
            },
            friendship: if friendship_enabled {
                Some(friendship::FriendshipEngine::new(friendship::FriendshipConfig::default()))
            } else {
                None
            },
            rebroadcast: rebroadcast::RebroadcastEngine::new(rebroadcast::RebroadcastConfig::default()),
        }
    }

    /// Configure rebroadcasting behaviour
    pub fn configure_rebroadcast(&mut self, config: rebroadcast::RebroadcastConfig) {
        self.rebroadcast = rebroadcast::RebroadcastEngine::new(config);
    }

    /// Access underlying BLE adapter
    pub fn get_adapter(&self) -> &ble::BleCla {
        &self.ble_adapter
    }

    /// Fragment a message and optionally apply FEC
    pub fn prepare_message(&mut self, data: &[u8])
        -> Result<Vec<fragmentation::Fragment>, Box<dyn std::error::Error>>
    {
        let payload = if let Some(encoder) = &self.fec_encoder {
            let packets = encoder.encode(data, 0)?;
            packets.into_iter().flat_map(|p| p.payload).collect::<Vec<u8>>()
        } else {
            data.to_vec()
        };

        Ok(self.fragmentation.fragment_message(&payload)?)
    }

    /// Process incoming fragment and attempt reassembly
    pub fn process_fragment(&mut self, fragment: fragmentation::Fragment)
        -> Result<Option<Vec<u8>>, Box<dyn std::error::Error>>
    {
        let data = self.fragmentation.process_fragment(fragment)?;
        Ok(data)
    }

    /// Access friendship engine if enabled
    pub fn friendship_engine(&mut self) -> Option<&mut friendship::FriendshipEngine> {
        self.friendship.as_mut()
    }

    /// Access rebroadcast engine
    pub fn rebroadcast_engine(&mut self) -> &mut rebroadcast::RebroadcastEngine {
        &mut self.rebroadcast
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
