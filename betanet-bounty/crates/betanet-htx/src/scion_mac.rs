//! SCION MAC (Message Authentication Code) Implementation
//!
//! Provides cryptographic verification and authentication for SCION packets
//! according to the SCION protocol specification.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use hmac::{Hmac, Mac};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::Sha256;
use thiserror::Error;

/// SCION MAC errors
#[derive(Debug, Error)]
pub enum ScionMacError {
    #[error("MAC verification failed")]
    VerificationFailed,
    #[error("Invalid MAC key: {0}")]
    InvalidKey(String),
    #[error("MAC algorithm not supported: {0}")]
    UnsupportedAlgorithm(String),
    #[error("Key rotation required")]
    KeyRotationRequired,
    #[error("Cryptographic error: {0}")]
    CryptoError(String),
}

/// SCION MAC algorithm types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacAlgorithm {
    HmacSha256,
    HmacSha384,
    HmacSha512,
}

impl MacAlgorithm {
    pub fn key_size(&self) -> usize {
        match self {
            MacAlgorithm::HmacSha256 => 32,
            MacAlgorithm::HmacSha384 => 48,
            MacAlgorithm::HmacSha512 => 64,
        }
    }

    pub fn output_size(&self) -> usize {
        match self {
            MacAlgorithm::HmacSha256 => 32,
            MacAlgorithm::HmacSha384 => 48,
            MacAlgorithm::HmacSha512 => 64,
        }
    }
}

/// SCION MAC key with rotation metadata
#[derive(Debug, Clone)]
pub struct MacKey {
    algorithm: MacAlgorithm,
    key_data: Vec<u8>,
    key_id: u32,
    created_at: SystemTime,
    expires_at: SystemTime,
}

impl MacKey {
    pub fn new(algorithm: MacAlgorithm, key_id: u32, ttl: Duration) -> Result<Self, ScionMacError> {
        let mut key_data = vec![0u8; algorithm.key_size()];
        let mut rng = ChaCha20Rng::from_entropy();
        rng.fill_bytes(&mut key_data);

        let now = SystemTime::now();
        let expires_at = now + ttl;

        Ok(Self {
            algorithm,
            key_data,
            key_id,
            created_at: now,
            expires_at,
        })
    }

    pub fn from_bytes(
        algorithm: MacAlgorithm,
        key_id: u32,
        key_data: Vec<u8>,
        ttl: Duration,
    ) -> Result<Self, ScionMacError> {
        if key_data.len() != algorithm.key_size() {
            return Err(ScionMacError::InvalidKey(format!(
                "Expected {} bytes, got {}",
                algorithm.key_size(),
                key_data.len()
            )));
        }

        let now = SystemTime::now();
        let expires_at = now + ttl;

        Ok(Self {
            algorithm,
            key_data,
            key_id,
            created_at: now,
            expires_at,
        })
    }

    pub fn is_expired(&self) -> bool {
        SystemTime::now() > self.expires_at
    }

    pub fn time_until_expiry(&self) -> Option<Duration> {
        self.expires_at.duration_since(SystemTime::now()).ok()
    }

    pub fn algorithm(&self) -> MacAlgorithm {
        self.algorithm
    }

    pub fn key_id(&self) -> u32 {
        self.key_id
    }
}

/// SCION MAC handler with key rotation support
pub struct ScionMacHandler {
    current_key: MacKey,
    previous_key: Option<MacKey>,
    mac_algorithm: MacAlgorithm,
    key_rotation_interval: Duration,
}

impl ScionMacHandler {
    /// Create new SCION MAC handler
    pub fn new(
        mac_algorithm: MacAlgorithm,
        key_rotation_interval: Duration,
    ) -> Result<Self, ScionMacError> {
        let current_key = MacKey::new(mac_algorithm, 1, key_rotation_interval)?;

        Ok(Self {
            current_key,
            previous_key: None,
            mac_algorithm,
            key_rotation_interval,
        })
    }

    /// Create SCION MAC handler with specific key
    pub fn with_key(
        mac_algorithm: MacAlgorithm,
        key_data: Vec<u8>,
        key_id: u32,
        key_rotation_interval: Duration,
    ) -> Result<Self, ScionMacError> {
        let current_key = MacKey::from_bytes(mac_algorithm, key_id, key_data, key_rotation_interval)?;

        Ok(Self {
            current_key,
            previous_key: None,
            mac_algorithm,
            key_rotation_interval,
        })
    }

    /// Verify MAC for SCION packet
    pub fn verify_mac(&self, data: &[u8], provided_mac: &[u8]) -> Result<(), ScionMacError> {
        // Try current key first
        if self.verify_with_key(&self.current_key, data, provided_mac)? {
            return Ok(());
        }

        // Try previous key if available (for key rotation tolerance)
        if let Some(ref prev_key) = self.previous_key {
            if self.verify_with_key(prev_key, data, provided_mac)? {
                return Ok(());
            }
        }

        Err(ScionMacError::VerificationFailed)
    }

    /// Compute MAC for SCION packet
    pub fn compute_mac(&self, data: &[u8]) -> Result<Bytes, ScionMacError> {
        self.compute_with_key(&self.current_key, data)
    }

    /// Check if key rotation is required
    pub fn mac_key_rotation_required(&self) -> bool {
        // Rotate when current key is 75% through its lifetime
        if let Some(time_left) = self.current_key.time_until_expiry() {
            let quarter_lifetime = self.key_rotation_interval / 4;
            time_left < quarter_lifetime
        } else {
            true // Key expired
        }
    }

    /// Perform MAC key rotation
    pub fn mac_key_rotation(&mut self) -> Result<(), ScionMacError> {
        let new_key_id = self.current_key.key_id() + 1;
        let new_key = MacKey::new(self.mac_algorithm, new_key_id, self.key_rotation_interval)?;

        // Keep current key as previous for verification tolerance
        self.previous_key = Some(self.current_key.clone());
        self.current_key = new_key;

        Ok(())
    }

    /// Get current MAC algorithm specification
    pub fn mac_algorithm(&self) -> MacAlgorithm {
        self.mac_algorithm
    }

    /// Get current key ID
    pub fn current_key_id(&self) -> u32 {
        self.current_key.key_id()
    }

    /// Get MAC size for current algorithm
    pub fn mac_size(&self) -> usize {
        self.mac_algorithm.output_size()
    }

    // Private helper methods

    fn verify_with_key(&self, key: &MacKey, data: &[u8], provided_mac: &[u8]) -> Result<bool, ScionMacError> {
        let computed_mac = self.compute_with_key(key, data)?;

        // Constant-time comparison to prevent timing attacks
        Ok(constant_time_eq(&computed_mac, provided_mac))
    }

    fn compute_with_key(&self, key: &MacKey, data: &[u8]) -> Result<Bytes, ScionMacError> {
        match key.algorithm() {
            MacAlgorithm::HmacSha256 => {
                let mut mac = Hmac::<Sha256>::new_from_slice(&key.key_data)
                    .map_err(|e| ScionMacError::CryptoError(e.to_string()))?;
                mac.update(data);
                Ok(Bytes::copy_from_slice(&mac.finalize().into_bytes()))
            }
            MacAlgorithm::HmacSha384 => {
                let mut mac = Hmac::<sha2::Sha384>::new_from_slice(&key.key_data)
                    .map_err(|e| ScionMacError::CryptoError(e.to_string()))?;
                mac.update(data);
                Ok(Bytes::copy_from_slice(&mac.finalize().into_bytes()))
            }
            MacAlgorithm::HmacSha512 => {
                let mut mac = Hmac::<sha2::Sha512>::new_from_slice(&key.key_data)
                    .map_err(|e| ScionMacError::CryptoError(e.to_string()))?;
                mac.update(data);
                Ok(Bytes::copy_from_slice(&mac.finalize().into_bytes()))
            }
        }
    }
}

/// Constant-time equality comparison to prevent timing attacks
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    result == 0
}

/// SCION packet MAC field utilities
pub struct ScionPacketMac;

impl ScionPacketMac {
    /// Extract MAC field from SCION packet
    pub fn extract_mac(packet: &[u8]) -> Option<&[u8]> {
        // SCION packet MAC is typically in the last 16-64 bytes depending on algorithm
        // This is a simplified implementation - real SCION would parse the packet header
        if packet.len() < 32 {
            return None;
        }
        Some(&packet[packet.len() - 32..])
    }

    /// Get packet data for MAC computation (excluding MAC field)
    pub fn get_mac_data(packet: &[u8]) -> Option<&[u8]> {
        if packet.len() < 32 {
            return None;
        }
        Some(&packet[..packet.len() - 32])
    }

    /// Create packet with MAC appended
    pub fn append_mac(packet_data: &[u8], mac: &[u8]) -> Vec<u8> {
        let mut result = packet_data.to_vec();
        result.extend_from_slice(mac);
        result
    }

    /// Validate SCION packet MAC
    pub fn validate_packet_mac(
        handler: &ScionMacHandler,
        packet: &[u8],
    ) -> Result<(), ScionMacError> {
        let mac_data = Self::get_mac_data(packet)
            .ok_or_else(|| ScionMacError::InvalidKey("Packet too short".to_string()))?;

        let provided_mac = Self::extract_mac(packet)
            .ok_or_else(|| ScionMacError::InvalidKey("Cannot extract MAC".to_string()))?;

        handler.verify_mac(mac_data, provided_mac)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mac_key_creation() {
        let key = MacKey::new(MacAlgorithm::HmacSha256, 1, Duration::from_secs(3600)).unwrap();
        assert_eq!(key.algorithm(), MacAlgorithm::HmacSha256);
        assert_eq!(key.key_id(), 1);
        assert!(!key.is_expired());
        assert_eq!(key.key_data.len(), 32);
    }

    #[test]
    fn test_mac_handler_creation() {
        let handler = ScionMacHandler::new(MacAlgorithm::HmacSha256, Duration::from_secs(3600)).unwrap();
        assert_eq!(handler.mac_algorithm(), MacAlgorithm::HmacSha256);
        assert_eq!(handler.current_key_id(), 1);
        assert_eq!(handler.mac_size(), 32);
    }

    #[test]
    fn test_mac_computation_and_verification() {
        let handler = ScionMacHandler::new(MacAlgorithm::HmacSha256, Duration::from_secs(3600)).unwrap();
        let data = b"test SCION packet data";

        // Compute MAC
        let mac = handler.compute_mac(data).unwrap();
        assert_eq!(mac.len(), 32);

        // Verify MAC
        assert!(handler.verify_mac(data, &mac).is_ok());

        // Verify with wrong data should fail
        let wrong_data = b"wrong data";
        assert!(handler.verify_mac(wrong_data, &mac).is_err());
    }

    #[test]
    fn test_key_rotation() {
        let mut handler = ScionMacHandler::new(MacAlgorithm::HmacSha256, Duration::from_millis(100)).unwrap();
        let original_key_id = handler.current_key_id();

        // Initially no rotation required
        assert!(!handler.mac_key_rotation_required());

        // Wait for key to approach expiry
        std::thread::sleep(Duration::from_millis(80));

        // Now rotation should be required
        assert!(handler.mac_key_rotation_required());

        // Perform rotation
        handler.mac_key_rotation().unwrap();
        assert_eq!(handler.current_key_id(), original_key_id + 1);
    }

    #[test]
    fn test_packet_mac_utilities() {
        let handler = ScionMacHandler::new(MacAlgorithm::HmacSha256, Duration::from_secs(3600)).unwrap();
        let packet_data = b"SCION packet payload";

        // Compute MAC for packet
        let mac = handler.compute_mac(packet_data).unwrap();

        // Create packet with MAC
        let packet_with_mac = ScionPacketMac::append_mac(packet_data, &mac);

        // Validate packet MAC
        assert!(ScionPacketMac::validate_packet_mac(&handler, &packet_with_mac).is_ok());

        // Extract components
        let extracted_mac = ScionPacketMac::extract_mac(&packet_with_mac).unwrap();
        let extracted_data = ScionPacketMac::get_mac_data(&packet_with_mac).unwrap();

        assert_eq!(extracted_mac, &mac[..]);
        assert_eq!(extracted_data, packet_data);
    }

    #[test]
    fn test_constant_time_eq() {
        let a = b"test data";
        let b = b"test data";
        let c = b"different";

        assert!(constant_time_eq(a, b));
        assert!(!constant_time_eq(a, c));
        assert!(!constant_time_eq(a, b"test"));
    }

    #[test]
    fn test_mac_algorithms() {
        for algorithm in [MacAlgorithm::HmacSha256, MacAlgorithm::HmacSha384, MacAlgorithm::HmacSha512] {
            let handler = ScionMacHandler::new(algorithm, Duration::from_secs(3600)).unwrap();
            let data = b"test data for all algorithms";

            let mac = handler.compute_mac(data).unwrap();
            assert_eq!(mac.len(), algorithm.output_size());

            assert!(handler.verify_mac(data, &mac).is_ok());
        }
    }
}
