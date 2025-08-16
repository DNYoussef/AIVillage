//! Betanet Mixnode - High-performance Nym-style mixnode implementation
//!
//! This crate provides a mixnode implementation for anonymous communication
//! using Sphinx packet processing, VRF-based delays, and cover traffic.

#![deny(warnings)]
#![deny(clippy::all)]
#![deny(missing_docs)]

use std::net::SocketAddr;
use std::time::Duration;

use thiserror::Error;

pub mod config;
pub mod crypto;
pub mod delay;
pub mod mixnode;
pub mod packet;
pub mod pipeline;
pub mod routing;

#[cfg(feature = "sphinx")]
pub mod sphinx;

#[cfg(feature = "vrf")]
pub mod vrf_delay;

#[cfg(feature = "vrf")]
pub mod vrf_neighbor;

#[cfg(feature = "cover-traffic")]
pub mod cover;

pub mod rate;

/// Mixnode protocol version
pub const MIXNODE_VERSION: u8 = 1;

/// Maximum packet size
pub const MAX_PACKET_SIZE: usize = 2048;

/// Mixnode errors
#[derive(Debug, Error)]
pub enum MixnodeError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Cryptographic error
    #[error("Crypto error: {0}")]
    Crypto(String),

    /// Packet processing error
    #[error("Packet error: {0}")]
    Packet(String),

    /// Routing error
    #[error("Routing error: {0}")]
    Routing(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Network error
    #[error("Network error: {0}")]
    Network(String),

    /// VRF error
    #[error("VRF error: {0}")]
    Vrf(String),
}

/// Result type for mixnode operations
pub type Result<T> = std::result::Result<T, MixnodeError>;

/// Mixnode statistics
#[derive(Debug, Default, Clone)]
pub struct MixnodeStats {
    /// Packets processed
    pub packets_processed: u64,
    /// Packets forwarded
    pub packets_forwarded: u64,
    /// Packets dropped
    pub packets_dropped: u64,
    /// Cover traffic sent
    pub cover_traffic_sent: u64,
    /// Average processing time (microseconds)
    pub avg_processing_time_us: f64,
    /// Uptime in seconds
    pub uptime_secs: u64,
}

impl MixnodeStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record processed packet
    pub fn record_processed(&mut self, processing_time: Duration) {
        self.packets_processed += 1;
        let time_us = processing_time.as_micros() as f64;
        self.avg_processing_time_us =
            (self.avg_processing_time_us * (self.packets_processed - 1) as f64 + time_us)
            / self.packets_processed as f64;
    }

    /// Record forwarded packet
    pub fn record_forwarded(&mut self) {
        self.packets_forwarded += 1;
    }

    /// Record dropped packet
    pub fn record_dropped(&mut self) {
        self.packets_dropped += 1;
    }

    /// Record cover traffic
    pub fn record_cover_traffic(&mut self) {
        self.cover_traffic_sent += 1;
    }
}

/// Mixnode trait for different implementations
#[async_trait::async_trait]
pub trait Mixnode: Send + Sync {
    /// Start the mixnode
    async fn start(&mut self) -> Result<()>;

    /// Stop the mixnode
    async fn stop(&mut self) -> Result<()>;

    /// Process a packet
    async fn process_packet(&self, packet: &[u8]) -> Result<Option<Vec<u8>>>;

    /// Get node statistics
    fn stats(&self) -> &MixnodeStats;

    /// Get node address
    fn address(&self) -> SocketAddr;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats() {
        let mut stats = MixnodeStats::new();
        assert_eq!(stats.packets_processed, 0);

        stats.record_processed(Duration::from_micros(100));
        assert_eq!(stats.packets_processed, 1);
        assert_eq!(stats.avg_processing_time_us, 100.0);

        stats.record_forwarded();
        assert_eq!(stats.packets_forwarded, 1);
    }
}
