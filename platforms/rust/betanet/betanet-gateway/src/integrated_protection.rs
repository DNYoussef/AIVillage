// Integrated AEAD + Anti-replay protection for SCION tunnel
// Production implementation combining ChaCha20-Poly1305 encryption with sequence validation

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::aead::{AeadManager, AeadFrame, FrameType};
use crate::anti_replay::{AntiReplayManager, ValidationResult};
use crate::config::{AeadConfig, AntiReplayConfig};
use crate::metrics::MetricsCollector;

/// Protected SCION frame with full AEAD + anti-replay protection
#[derive(Debug, Clone)]
pub struct ProtectedFrame {
    /// Encrypted AEAD frame
    pub aead_frame: AeadFrame,

    /// Anti-replay validation metadata
    pub validation_metadata: Option<ValidationMetadata>,
}

/// Anti-replay validation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetadata {
    /// Peer ID for sequence tracking
    pub peer_id: String,

    /// Frame sequence number
    pub sequence: u64,

    /// Validation timestamp
    pub timestamp_ns: u64,

    /// Whether validation was successful
    pub valid: bool,

    /// Rejection reason if invalid
    pub rejection_reason: String,
}

/// Protection statistics combining AEAD and anti-replay
#[derive(Debug, Clone, Default)]
pub struct ProtectionStats {
    /// Frames successfully protected
    pub frames_protected: u64,

    /// Frames successfully unprotected
    pub frames_unprotected: u64,

    /// AEAD authentication failures
    pub aead_failures: u64,

    /// Anti-replay failures (replays blocked)
    pub replay_failures: u64,

    /// Sequence validation failures
    pub sequence_failures: u64,

    /// Average protection time in microseconds
    pub avg_protection_time_us: u64,

    /// Average validation time in microseconds
    pub avg_validation_time_us: u64,

    /// Total bytes protected
    pub bytes_protected: u64,

    /// Key rotations performed
    pub key_rotations: u64,
}

/// Integrated protection manager combining AEAD and anti-replay
pub struct IntegratedProtectionManager {
    aead: Arc<AeadManager>,
    anti_replay: Arc<AntiReplayManager>,
    metrics: Arc<MetricsCollector>,
    stats: Arc<RwLock<ProtectionStats>>,
}

impl IntegratedProtectionManager {
    /// Create new integrated protection manager
    pub async fn new(
        aead_config: AeadConfig,
        anti_replay_config: AntiReplayConfig,
        metrics: Arc<MetricsCollector>,
        master_key: [u8; 32],
    ) -> Result<Self> {
        info!("Initializing integrated AEAD + anti-replay protection");

        // Initialize AEAD manager
        let aead = Arc::new(AeadManager::new(
            aead_config,
            metrics.clone(),
            master_key,
        ));

        // Initialize anti-replay manager
        let anti_replay = Arc::new(AntiReplayManager::new(
            anti_replay_config,
            metrics.clone(),
        ).await?);

        Ok(Self {
            aead,
            anti_replay,
            metrics,
            stats: Arc::new(RwLock::new(ProtectionStats::default())),
        })
    }

    /// Protect SCION packet with AEAD encryption and sequence numbering
    pub async fn protect_packet(
        &self,
        peer_id: &str,
        packet_data: &[u8],
        frame_type: FrameType,
    ) -> Result<ProtectedFrame> {
        let start_time = Instant::now();

        debug!(
            peer_id = peer_id,
            packet_len = packet_data.len(),
            frame_type = ?frame_type,
            "Protecting SCION packet"
        );

        // Create additional authenticated data with peer ID
        let aad = peer_id.as_bytes();

        // Encrypt frame with AEAD
        let aead_frame = self.aead.encrypt_frame(
            peer_id,
            frame_type,
            packet_data,
            aad,
        ).await.context("AEAD encryption failed")?;

        // Create validation metadata for anti-replay
        let metadata = ValidationMetadata {
            peer_id: peer_id.to_string(),
            sequence: aead_frame.header.sequence,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            valid: true,
            rejection_reason: String::new(),
        };

        // Update protection statistics
        let protection_time = start_time.elapsed().as_micros() as u64;
        let mut stats = self.stats.write().await;
        stats.frames_protected += 1;
        stats.bytes_protected += packet_data.len() as u64;
        stats.avg_protection_time_us = (stats.avg_protection_time_us + protection_time) / 2;
        drop(stats);

        debug!(
            peer_id = peer_id,
            sequence = aead_frame.header.sequence,
            epoch = aead_frame.header.epoch,
            protection_time_us = protection_time,
            "Packet protection completed"
        );

        Ok(ProtectedFrame {
            aead_frame,
            validation_metadata: Some(metadata),
        })
    }

    /// Unprotect SCION packet with anti-replay validation and AEAD decryption
    pub async fn unprotect_packet(
        &self,
        peer_id: &str,
        protected_frame: &ProtectedFrame,
    ) -> Result<Vec<u8>> {
        let start_time = Instant::now();

        debug!(
            peer_id = peer_id,
            sequence = protected_frame.aead_frame.header.sequence,
            frame_type = ?protected_frame.aead_frame.header.frame_type,
            "Unprotecting SCION packet"
        );

        // Extract sequence and timestamp for anti-replay validation
        let sequence = protected_frame.aead_frame.header.sequence;
        let timestamp_ns = if let Some(ref metadata) = protected_frame.validation_metadata {
            metadata.timestamp_ns
        } else {
            // Use current time if no metadata available
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
        };

        // Validate sequence number against anti-replay window
        let validation_result = self.anti_replay.validate_sequence(
            peer_id,
            sequence,
            timestamp_ns,
            true, // Update window on success
        ).await;

        // Check validation result
        if !validation_result.valid {
            let mut stats = self.stats.write().await;
            match validation_result.rejection_reason.as_str() {
                "replay" => stats.replay_failures += 1,
                "expired" | "future" => stats.sequence_failures += 1,
                _ => {}
            }
            stats.avg_validation_time_us =
                (stats.avg_validation_time_us + validation_result.validation_time_us) / 2;
            drop(stats);

            bail!("Anti-replay validation failed: {}", validation_result.rejection_reason);
        }

        // Decrypt AEAD frame
        let plaintext = self.aead.decrypt_frame(peer_id, &protected_frame.aead_frame)
            .await
            .context("AEAD decryption failed")?;

        // Update statistics
        let total_time = start_time.elapsed().as_micros() as u64;
        let mut stats = self.stats.write().await;
        stats.frames_unprotected += 1;
        stats.avg_validation_time_us =
            (stats.avg_validation_time_us + validation_result.validation_time_us) / 2;
        drop(stats);

        debug!(
            peer_id = peer_id,
            sequence = sequence,
            plaintext_len = plaintext.len(),
            total_time_us = total_time,
            validation_time_us = validation_result.validation_time_us,
            "Packet unprotection completed"
        );

        Ok(plaintext)
    }

    /// Check if replay protection detected any replay attempts
    pub async fn check_replay_drops(&self, peer_id: &str) -> u64 {
        let stats = self.anti_replay.get_stats().await;
        stats.replays_blocked
    }

    /// Trigger key rotation for peer
    pub async fn rotate_keys(&self, peer_id: &str) -> Result<()> {
        info!(peer_id = peer_id, "Triggering integrated key rotation");

        // Rotate AEAD keys
        self.aead.rotate_keys(peer_id).await?;

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.key_rotations += 1;

        info!(peer_id = peer_id, "Key rotation completed successfully");
        Ok(())
    }

    /// Check if key rotation is needed
    pub async fn should_rotate_keys(&self, peer_id: &str) -> bool {
        self.aead.should_rotate_keys(peer_id).await
    }

    /// Serialize protected frame for transmission
    pub fn serialize_protected_frame(&self, frame: &ProtectedFrame) -> Result<Vec<u8>> {
        // Use AEAD frame serialization
        self.aead.serialize_frame(&frame.aead_frame)
    }

    /// Deserialize protected frame from wire format
    pub fn deserialize_protected_frame(&self, data: &[u8]) -> Result<ProtectedFrame> {
        let aead_frame = self.aead.deserialize_frame(data)?;

        Ok(ProtectedFrame {
            aead_frame,
            validation_metadata: None, // Will be populated during validation
        })
    }

    /// Get comprehensive protection statistics
    pub async fn get_stats(&self) -> ProtectionStats {
        let mut stats = self.stats.read().await.clone();

        // Add AEAD statistics
        let aead_stats = self.aead.get_stats().await;
        stats.aead_failures = aead_stats.auth_failures;
        stats.key_rotations = aead_stats.key_rotations;

        // Add anti-replay statistics
        let replay_stats = self.anti_replay.get_stats().await;
        stats.replay_failures = replay_stats.replays_blocked;
        stats.sequence_failures = replay_stats.expired_rejected + replay_stats.future_rejected;

        stats
    }

    /// Inject replay attack for testing (DO NOT USE IN PRODUCTION)
    #[cfg(test)]
    pub async fn inject_replay_attack(
        &self,
        peer_id: &str,
        original_frame: &ProtectedFrame,
    ) -> Result<bool> {
        warn!("TESTING: Injecting replay attack for peer {}", peer_id);

        // Try to validate the same sequence number again
        let validation_result = self.anti_replay.validate_sequence(
            peer_id,
            original_frame.aead_frame.header.sequence,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            false, // Don't update window
        ).await;

        // Should be rejected as replay
        Ok(!validation_result.valid && validation_result.rejection_reason == "replay")
    }

    /// Stop integrated protection manager
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping integrated protection manager");

        // Stop anti-replay manager
        self.anti_replay.stop().await?;

        info!("Integrated protection manager stopped");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GatewayConfig;
    use tempfile::TempDir;

    async fn create_test_manager() -> (IntegratedProtectionManager, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_integrated.db");

        let aead_config = AeadConfig {
            max_bytes_per_key: 1024 * 1024, // 1MB for testing
            max_time_per_key: Duration::from_secs(3600),
        };

        let anti_replay_config = AntiReplayConfig {
            db_path,
            window_size: 64,
            cleanup_ttl: Duration::from_secs(3600),
            cleanup_interval: Duration::from_secs(60),
            sync_interval: Duration::from_secs(10),
            max_sequence_age: Duration::from_secs(300),
        };

        let gateway_config = Arc::new(GatewayConfig::default());
        let metrics = Arc::new(MetricsCollector::new(gateway_config).unwrap());
        let master_key = [42u8; 32]; // Test key

        let manager = IntegratedProtectionManager::new(
            aead_config,
            anti_replay_config,
            metrics,
            master_key,
        ).await.unwrap();

        (manager, temp_dir)
    }

    #[tokio::test]
    async fn test_integrated_protect_unprotect() {
        let (manager, _temp_dir) = create_test_manager().await;
        let peer_id = "test_peer";
        let packet_data = b"Hello, integrated SCION protection!";

        // Protect packet
        let protected_frame = manager.protect_packet(
            peer_id,
            packet_data,
            FrameType::ScionData,
        ).await.unwrap();

        // Unprotect packet
        let decrypted = manager.unprotect_packet(peer_id, &protected_frame).await.unwrap();

        assert_eq!(decrypted, packet_data);

        // Check statistics
        let stats = manager.get_stats().await;
        assert_eq!(stats.frames_protected, 1);
        assert_eq!(stats.frames_unprotected, 1);
        assert_eq!(stats.aead_failures, 0);
        assert_eq!(stats.replay_failures, 0);

        manager.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_replay_attack_detection() {
        let (manager, _temp_dir) = create_test_manager().await;
        let peer_id = "replay_peer";
        let packet_data = b"Replay attack test";

        // Protect first packet
        let protected_frame = manager.protect_packet(
            peer_id,
            packet_data,
            FrameType::ScionData,
        ).await.unwrap();

        // First unprotect should succeed
        let decrypted1 = manager.unprotect_packet(peer_id, &protected_frame).await.unwrap();
        assert_eq!(decrypted1, packet_data);

        // Second unprotect of same frame should fail (replay attack)
        let result2 = manager.unprotect_packet(peer_id, &protected_frame).await;
        assert!(result2.is_err());

        // Check that replay was detected
        let stats = manager.get_stats().await;
        assert_eq!(stats.frames_protected, 1);
        assert_eq!(stats.frames_unprotected, 1);
        assert_eq!(stats.replay_failures, 1);

        // Test injection method
        let replay_detected = manager.inject_replay_attack(peer_id, &protected_frame).await.unwrap();
        assert!(replay_detected);

        manager.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_multiple_peers() {
        let (manager, _temp_dir) = create_test_manager().await;
        let peers = ["peer_1", "peer_2", "peer_3"];
        let packet_data = b"Multi-peer test packet";

        // Protect packets for each peer
        let mut protected_frames = Vec::new();
        for peer_id in &peers {
            let frame = manager.protect_packet(
                peer_id,
                packet_data,
                FrameType::ScionData,
            ).await.unwrap();
            protected_frames.push((peer_id, frame));
        }

        // Unprotect packets for each peer
        for (peer_id, frame) in &protected_frames {
            let decrypted = manager.unprotect_packet(peer_id, frame).await.unwrap();
            assert_eq!(decrypted, packet_data);
        }

        // Check statistics
        let stats = manager.get_stats().await;
        assert_eq!(stats.frames_protected, 3);
        assert_eq!(stats.frames_unprotected, 3);
        assert_eq!(stats.aead_failures, 0);
        assert_eq!(stats.replay_failures, 0);

        manager.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_serialization_roundtrip() {
        let (manager, _temp_dir) = create_test_manager().await;
        let peer_id = "serial_peer";
        let packet_data = b"Serialization test data";

        // Protect packet
        let original_frame = manager.protect_packet(
            peer_id,
            packet_data,
            FrameType::Control,
        ).await.unwrap();

        // Serialize
        let serialized = manager.serialize_protected_frame(&original_frame).unwrap();

        // Deserialize
        let deserialized_frame = manager.deserialize_protected_frame(&serialized).unwrap();

        // Unprotect deserialized frame
        let decrypted = manager.unprotect_packet(peer_id, &deserialized_frame).await.unwrap();

        assert_eq!(decrypted, packet_data);

        manager.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_key_rotation() {
        let (manager, _temp_dir) = create_test_manager().await;
        let peer_id = "rotation_peer";

        // Initial packet
        let packet1 = b"Before rotation";
        let frame1 = manager.protect_packet(peer_id, packet1, FrameType::ScionData).await.unwrap();
        let epoch1 = frame1.aead_frame.header.epoch;

        // Trigger key rotation
        manager.rotate_keys(peer_id).await.unwrap();

        // Packet after rotation
        let packet2 = b"After rotation";
        let frame2 = manager.protect_packet(peer_id, packet2, FrameType::ScionData).await.unwrap();
        let epoch2 = frame2.aead_frame.header.epoch;

        // Epochs should be different
        assert_ne!(epoch1, epoch2);

        // Both packets should decrypt correctly
        let decrypted1 = manager.unprotect_packet(peer_id, &frame1).await.unwrap();
        let decrypted2 = manager.unprotect_packet(peer_id, &frame2).await.unwrap();

        assert_eq!(decrypted1, packet1);
        assert_eq!(decrypted2, packet2);

        // Check rotation statistics
        let stats = manager.get_stats().await;
        assert_eq!(stats.key_rotations, 1);

        manager.stop().await.unwrap();
    }
}
