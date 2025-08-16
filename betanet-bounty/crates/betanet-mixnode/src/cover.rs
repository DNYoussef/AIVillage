//! Cover traffic generation

use std::time::Duration;

use tokio::time::{interval, Interval};
use tracing::debug;

use crate::{packet::Packet, Result};

/// Cover traffic generator
pub struct CoverTrafficGenerator {
    interval: Interval,
    packet_size: usize,
    layer: u8,
}

impl CoverTrafficGenerator {
    /// Create new cover traffic generator
    pub fn new(interval_duration: Duration, packet_size: usize, layer: u8) -> Self {
        let interval = interval(interval_duration);
        Self {
            interval,
            packet_size,
            layer,
        }
    }

    /// Generate next cover traffic packet
    pub async fn next_packet(&mut self) -> Result<Packet> {
        self.interval.tick().await;

        debug!("Generating cover traffic packet of {} bytes", self.packet_size);

        let packet = Packet::cover_traffic(self.packet_size, self.layer);
        Ok(packet)
    }

    /// Start cover traffic generation loop
    pub async fn start<F>(&mut self, mut handler: F) -> Result<()>
    where
        F: FnMut(Packet) + Send,
    {
        loop {
            let packet = self.next_packet().await?;
            handler(packet);
        }
    }

    /// Set new interval
    pub fn set_interval(&mut self, duration: Duration) {
        self.interval = interval(duration);
    }

    /// Set packet size
    pub fn set_packet_size(&mut self, size: usize) {
        self.packet_size = size;
    }

    /// Set layer
    pub fn set_layer(&mut self, layer: u8) {
        self.layer = layer;
    }
}

/// Cover traffic configuration
#[derive(Debug, Clone)]
pub struct CoverTrafficConfig {
    /// Enable cover traffic
    pub enabled: bool,
    /// Generation interval
    pub interval: Duration,
    /// Packet size range
    pub min_packet_size: usize,
    /// Maximum packet size
    pub max_packet_size: usize,
    /// Layer to send on
    pub layer: u8,
    /// Burst size (packets per interval)
    pub burst_size: usize,
}

impl Default for CoverTrafficConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval: Duration::from_secs(10),
            min_packet_size: 512,
            max_packet_size: 1024,
            layer: 1,
            burst_size: 1,
        }
    }
}

/// Advanced cover traffic generator with random parameters
pub struct AdvancedCoverTrafficGenerator {
    config: CoverTrafficConfig,
    interval: Interval,
}

impl AdvancedCoverTrafficGenerator {
    /// Create new advanced generator
    pub fn new(config: CoverTrafficConfig) -> Self {
        let interval = interval(config.interval);
        Self { config, interval }
    }

    /// Generate burst of cover traffic packets
    pub async fn generate_burst(&mut self) -> Result<Vec<Packet>> {
        if !self.config.enabled {
            return Ok(vec![]);
        }

        self.interval.tick().await;

        let mut packets = Vec::with_capacity(self.config.burst_size);

        for _ in 0..self.config.burst_size {
            let size = rand::random::<usize>()
                % (self.config.max_packet_size - self.config.min_packet_size + 1)
                + self.config.min_packet_size;

            let packet = Packet::cover_traffic(size, self.config.layer);
            packets.push(packet);
        }

        debug!("Generated {} cover traffic packets", packets.len());
        Ok(packets)
    }

    /// Update configuration
    pub fn update_config(&mut self, config: CoverTrafficConfig) {
        if config.interval != self.config.interval {
            self.interval = interval(config.interval);
        }
        self.config = config;
    }

    /// Check if enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cover_traffic_generator() {
        let mut generator = CoverTrafficGenerator::new(
            Duration::from_millis(100),
            512,
            1
        );

        let packet = generator.next_packet().await.unwrap();
        assert!(packet.is_cover_traffic());
        assert_eq!(packet.layer(), 1);
        assert_eq!(packet.payload.len(), 512);
    }

    #[tokio::test]
    async fn test_advanced_cover_traffic_generator() {
        let config = CoverTrafficConfig {
            enabled: true,
            interval: Duration::from_millis(100),
            min_packet_size: 256,
            max_packet_size: 1024,
            layer: 2,
            burst_size: 3,
        };

        let mut generator = AdvancedCoverTrafficGenerator::new(config);
        let packets = generator.generate_burst().await.unwrap();

        assert_eq!(packets.len(), 3);
        for packet in packets {
            assert!(packet.is_cover_traffic());
            assert_eq!(packet.layer(), 2);
            assert!(packet.payload.len() >= 256);
            assert!(packet.payload.len() <= 1024);
        }
    }

    #[test]
    fn test_cover_traffic_config() {
        let config = CoverTrafficConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.burst_size, 1);
        assert_eq!(config.layer, 1);
    }
}
