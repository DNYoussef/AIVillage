// AEAD (ChaCha20-Poly1305) protection for SCION tunnel frames
// Production implementation with per-session subkeys derived from Noise XK key material

use anyhow::{Context, Result, bail};
use chacha20poly1305::{
    aead::{Aead, KeyInit, Nonce as AeadNonce, Payload},
    ChaCha20Poly1305, Key, Nonce as ChaChaNonce,
};
use rand::{Rng, thread_rng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use blake3;

use crate::config::AeadConfig;
use crate::metrics::MetricsCollector;

/// AEAD session key material
#[derive(Debug, Clone)]
pub struct SessionKeys {
    /// Encryption key for outbound traffic
    pub tx_key: Key,

    /// Decryption key for inbound traffic
    pub rx_key: Key,

    /// Session creation time
    pub created_at: Instant,

    /// Data processed with this session (for key rotation)
    pub bytes_processed: u64,

    /// Key generation epoch
    pub epoch: u64,
}

/// AEAD frame header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AeadFrameHeader {
    /// Key epoch for rotation
    pub epoch: u64,

    /// Sequence number (64-bit)
    pub sequence: u64,

    /// 12-byte nonce for ChaCha20-Poly1305
    pub nonce: [u8; 12],

    /// Frame type (control/data)
    pub frame_type: FrameType,

    /// Additional authenticated data length
    pub aad_len: u16,
}

/// SCION frame types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum FrameType {
    /// SCION data packet
    ScionData = 0x01,

    /// Control frame (path update, health check, etc.)
    Control = 0x02,

    /// Key rotation frame
    KeyRotation = 0x03,

    /// Heartbeat/keepalive
    Heartbeat = 0x04,
}

/// Encrypted AEAD frame
#[derive(Debug, Clone)]
pub struct AeadFrame {
    /// Frame header (authenticated but not encrypted)
    pub header: AeadFrameHeader,

    /// Additional authenticated data
    pub aad: Vec<u8>,

    /// Encrypted payload + authentication tag
    pub ciphertext: Vec<u8>,
}

/// AEAD statistics
#[derive(Debug, Clone, Default)]
pub struct AeadStats {
    /// Frames encrypted
    pub frames_encrypted: u64,

    /// Frames decrypted successfully
    pub frames_decrypted: u64,

    /// Authentication failures
    pub auth_failures: u64,

    /// Key rotations performed
    pub key_rotations: u64,

    /// Active sessions
    pub active_sessions: u64,

    /// Average encryption time in microseconds
    pub avg_encrypt_time_us: u64,

    /// Average decryption time in microseconds
    pub avg_decrypt_time_us: u64,

    /// Total bytes encrypted
    pub bytes_encrypted: u64,

    /// Total bytes decrypted
    pub bytes_decrypted: u64,
}

/// AEAD session manager with automatic key rotation
pub struct AeadManager {
    config: AeadConfig,
    metrics: Arc<MetricsCollector>,

    /// Active sessions by peer ID
    sessions: Arc<RwLock<HashMap<String, SessionKeys>>>,

    /// Next sequence numbers by peer (for outbound)
    tx_sequences: Arc<RwLock<HashMap<String, u64>>>,

    /// Statistics
    stats: Arc<RwLock<AeadStats>>,

    /// Master key for session derivation (from Noise XK)
    master_key: [u8; 32],

    /// Current key epoch
    current_epoch: Arc<RwLock<u64>>,
}

impl AeadManager {
    /// Create new AEAD manager with master key from Noise XK handshake
    pub fn new(
        config: AeadConfig,
        metrics: Arc<MetricsCollector>,
        master_key: [u8; 32],
    ) -> Self {
        info!("Initializing AEAD manager with ChaCha20-Poly1305");

        Self {
            config,
            metrics,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            tx_sequences: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(AeadStats::default())),
            master_key,
            current_epoch: Arc::new(RwLock::new(1)),
        }
    }

    /// Encrypt SCION packet or control frame
    pub async fn encrypt_frame(
        &self,
        peer_id: &str,
        frame_type: FrameType,
        payload: &[u8],
        aad: &[u8],
    ) -> Result<AeadFrame> {
        let start_time = Instant::now();

        // Get or create session keys
        let session_keys = self.get_or_create_session(peer_id).await?;

        // Get next sequence number
        let sequence = self.get_next_tx_sequence(peer_id).await;

        // Generate random nonce
        let mut nonce = [0u8; 12];
        thread_rng().fill(&mut nonce);

        // Create header
        let header = AeadFrameHeader {
            epoch: session_keys.epoch,
            sequence,
            nonce,
            frame_type,
            aad_len: aad.len() as u16,
        };

        // Serialize header for additional authentication
        let header_bytes = bincode::serialize(&header)
            .context("Failed to serialize AEAD header")?;

        // Create cipher
        let cipher = ChaCha20Poly1305::new(&session_keys.tx_key);
        let nonce = ChaChaNonce::from_slice(&nonce);

        // Combine header + AAD for authentication
        let mut auth_data = header_bytes;
        auth_data.extend_from_slice(aad);

        // Create payload for encryption
        let payload_struct = Payload {
            msg: payload,
            aad: &auth_data,
        };

        // Encrypt
        let ciphertext = cipher.encrypt(nonce, payload_struct)
            .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;

        // Update session stats
        self.update_session_data(peer_id, payload.len() as u64).await?;

        // Update statistics
        let encrypt_time = start_time.elapsed();
        let mut stats = self.stats.write().await;
        stats.frames_encrypted += 1;
        stats.bytes_encrypted += payload.len() as u64;
        stats.avg_encrypt_time_us = (stats.avg_encrypt_time_us + encrypt_time.as_micros() as u64) / 2;
        drop(stats);

        // Record metrics
        let frame_type_str = match frame_type {
            FrameType::ScionData => "scion_data",
            FrameType::Control => "control",
            FrameType::KeyRotation => "key_rotation",
            FrameType::Heartbeat => "heartbeat",
        };
        self.metrics.record_aead_encryption(frame_type_str, true, encrypt_time, payload.len());

        debug!(
            peer_id = peer_id,
            sequence = sequence,
            frame_type = ?frame_type,
            payload_len = payload.len(),
            encrypt_time_us = encrypt_time,
            "Frame encrypted successfully"
        );

        Ok(AeadFrame {
            header,
            aad: aad.to_vec(),
            ciphertext,
        })
    }

    /// Decrypt SCION packet or control frame
    pub async fn decrypt_frame(
        &self,
        peer_id: &str,
        frame: &AeadFrame,
    ) -> Result<Vec<u8>> {
        let start_time = Instant::now();

        // Get session keys
        let session_keys = self.get_session(peer_id).await
            .ok_or_else(|| anyhow::anyhow!("No session keys found for peer: {}", peer_id))?;

        // Check epoch
        if frame.header.epoch != session_keys.epoch {
            // Try to handle key rotation
            if frame.header.frame_type == FrameType::KeyRotation {
                warn!(peer_id = peer_id, expected_epoch = session_keys.epoch,
                      frame_epoch = frame.header.epoch, "Key rotation frame with epoch mismatch");
            } else {
                bail!("Epoch mismatch: expected {}, got {}",
                      session_keys.epoch, frame.header.epoch);
            }
        }

        // Serialize header for authentication verification
        let header_bytes = bincode::serialize(&frame.header)
            .context("Failed to serialize AEAD header")?;

        // Create cipher
        let cipher = ChaCha20Poly1305::new(&session_keys.rx_key);
        let nonce = ChaChaNonce::from_slice(&frame.header.nonce);

        // Combine header + AAD for authentication
        let mut auth_data = header_bytes;
        auth_data.extend_from_slice(&frame.aad);

        // Create payload for decryption
        let payload_struct = Payload {
            msg: &frame.ciphertext,
            aad: &auth_data,
        };

        // Decrypt and verify authentication
        let plaintext = cipher.decrypt(nonce, payload_struct)
            .map_err(|e| {
                // Update auth failure stats
                let stats = self.stats.clone();
                let metrics = self.metrics.clone();
                let frame_type = frame.header.frame_type;

                tokio::spawn(async move {
                    let mut stats_guard = stats.write().await;
                    stats_guard.auth_failures += 1;
                    drop(stats_guard);

                    // Record failed decryption metrics
                    let frame_type_str = match frame_type {
                        FrameType::ScionData => "scion_data",
                        FrameType::Control => "control",
                        FrameType::KeyRotation => "key_rotation",
                        FrameType::Heartbeat => "heartbeat",
                    };
                    metrics.record_aead_decryption(frame_type_str, false, Duration::from_micros(0), 0);
                });

                anyhow::anyhow!("Decryption/authentication failed: {}", e)
            })?;

        // Update session stats
        self.update_session_data(peer_id, plaintext.len() as u64).await?;

        // Update statistics
        let decrypt_time = start_time.elapsed();
        let mut stats = self.stats.write().await;
        stats.frames_decrypted += 1;
        stats.bytes_decrypted += plaintext.len() as u64;
        stats.avg_decrypt_time_us = (stats.avg_decrypt_time_us + decrypt_time.as_micros() as u64) / 2;
        drop(stats);

        // Record successful decryption metrics
        let frame_type_str = match frame.header.frame_type {
            FrameType::ScionData => "scion_data",
            FrameType::Control => "control",
            FrameType::KeyRotation => "key_rotation",
            FrameType::Heartbeat => "heartbeat",
        };
        self.metrics.record_aead_decryption(frame_type_str, true, decrypt_time, plaintext.len());

        debug!(
            peer_id = peer_id,
            sequence = frame.header.sequence,
            frame_type = ?frame.header.frame_type,
            plaintext_len = plaintext.len(),
            decrypt_time_us = decrypt_time,
            "Frame decrypted successfully"
        );

        Ok(plaintext)
    }

    /// Serialize AEAD frame for transmission
    pub fn serialize_frame(&self, frame: &AeadFrame) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();

        // Serialize header
        let header_bytes = bincode::serialize(&frame.header)
            .context("Failed to serialize frame header")?;

        // Frame format: [header_len:u16][header][aad][ciphertext]
        buffer.extend_from_slice(&(header_bytes.len() as u16).to_be_bytes());
        buffer.extend_from_slice(&header_bytes);
        buffer.extend_from_slice(&frame.aad);
        buffer.extend_from_slice(&frame.ciphertext);

        Ok(buffer)
    }

    /// Deserialize AEAD frame from wire format
    pub fn deserialize_frame(&self, data: &[u8]) -> Result<AeadFrame> {
        if data.len() < 2 {
            bail!("Frame data too short");
        }

        // Read header length
        let header_len = u16::from_be_bytes([data[0], data[1]]) as usize;
        if data.len() < 2 + header_len {
            bail!("Insufficient data for header");
        }

        // Deserialize header
        let header: AeadFrameHeader = bincode::deserialize(&data[2..2 + header_len])
            .context("Failed to deserialize frame header")?;

        let aad_start = 2 + header_len;
        let aad_end = aad_start + header.aad_len as usize;

        if data.len() < aad_end {
            bail!("Insufficient data for AAD");
        }

        let aad = data[aad_start..aad_end].to_vec();
        let ciphertext = data[aad_end..].to_vec();

        Ok(AeadFrame {
            header,
            aad,
            ciphertext,
        })
    }

    /// Trigger key rotation for a peer session
    pub async fn rotate_keys(&self, peer_id: &str) -> Result<()> {
        info!(peer_id = peer_id, "Rotating session keys");

        let new_epoch = {
            let mut epoch_guard = self.current_epoch.write().await;
            *epoch_guard += 1;
            *epoch_guard
        };

        // Derive new session keys
        let new_keys = self.derive_session_keys(peer_id, new_epoch)?;

        // Update session
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(peer_id.to_string(), new_keys);
        }

        // Reset sequence numbers
        {
            let mut tx_sequences = self.tx_sequences.write().await;
            tx_sequences.insert(peer_id.to_string(), 0);
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.key_rotations += 1;
        }

        // Record key rotation metrics
        self.metrics.record_aead_key_rotation();

        info!(peer_id = peer_id, epoch = new_epoch, "Key rotation completed");
        Ok(())
    }

    /// Check if session needs key rotation
    pub async fn should_rotate_keys(&self, peer_id: &str) -> bool {
        let sessions = self.sessions.read().await;

        if let Some(session) = sessions.get(peer_id) {
            // Check data limit
            if session.bytes_processed >= self.config.max_bytes_per_key {
                return true;
            }

            // Check time limit
            if session.created_at.elapsed() >= self.config.max_time_per_key {
                return true;
            }
        }

        false
    }

    /// Get AEAD statistics
    pub async fn get_stats(&self) -> AeadStats {
        let mut stats = self.stats.read().await.clone();
        let session_count = self.sessions.read().await.len() as u64;
        stats.active_sessions = session_count;

        // Update Prometheus metrics with current session count
        self.metrics.update_aead_active_sessions(session_count as i64);

        stats
    }

    /// Get or create session keys for peer
    async fn get_or_create_session(&self, peer_id: &str) -> Result<SessionKeys> {
        {
            let sessions = self.sessions.read().await;
            if let Some(keys) = sessions.get(peer_id) {
                return Ok(keys.clone());
            }
        }

        // Create new session
        let epoch = *self.current_epoch.read().await;
        let keys = self.derive_session_keys(peer_id, epoch)?;

        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(peer_id.to_string(), keys.clone());
        }

        info!(peer_id = peer_id, epoch = epoch, "Created new AEAD session");
        Ok(keys)
    }

    /// Get existing session keys
    async fn get_session(&self, peer_id: &str) -> Option<SessionKeys> {
        let sessions = self.sessions.read().await;
        sessions.get(peer_id).cloned()
    }

    /// Get next transmit sequence number
    async fn get_next_tx_sequence(&self, peer_id: &str) -> u64 {
        let mut tx_sequences = self.tx_sequences.write().await;
        let sequence = tx_sequences.entry(peer_id.to_string())
            .and_modify(|seq| *seq += 1)
            .or_insert(1);
        *sequence
    }

    /// Update session data counters
    async fn update_session_data(&self, peer_id: &str, bytes: u64) -> Result<()> {
        let mut sessions = self.sessions.write().await;

        if let Some(session) = sessions.get_mut(peer_id) {
            session.bytes_processed += bytes;

            // Check if rotation is needed
            if session.bytes_processed >= self.config.max_bytes_per_key ||
               session.created_at.elapsed() >= self.config.max_time_per_key {
                drop(sessions);

                // Trigger background key rotation
                let manager = self.clone();
                let peer_id = peer_id.to_string();
                tokio::spawn(async move {
                    if let Err(e) = manager.rotate_keys(&peer_id).await {
                        error!(peer_id = peer_id, error = ?e, "Background key rotation failed");
                    }
                });
            }
        }

        Ok(())
    }

    /// Derive session keys using HKDF from master key
    fn derive_session_keys(&self, peer_id: &str, epoch: u64) -> Result<SessionKeys> {
        // HKDF-like key derivation using BLAKE3
        let mut context = blake3::Hasher::new();
        context.update(&self.master_key);
        context.update(peer_id.as_bytes());
        context.update(&epoch.to_be_bytes());

        // Derive 64 bytes: 32 for TX key, 32 for RX key
        context.update(b"AEAD-SESSION-KEYS");
        let derived = context.finalize();
        let key_material = derived.as_bytes();

        let tx_key = Key::from_slice(&key_material[0..32]);
        let rx_key = Key::from_slice(&key_material[32..64]);

        Ok(SessionKeys {
            tx_key: *tx_key,
            rx_key: *rx_key,
            created_at: Instant::now(),
            bytes_processed: 0,
            epoch,
        })
    }
}

impl Clone for AeadManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            metrics: self.metrics.clone(),
            sessions: self.sessions.clone(),
            tx_sequences: self.tx_sequences.clone(),
            stats: self.stats.clone(),
            master_key: self.master_key,
            current_epoch: self.current_epoch.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GatewayConfig;

    fn create_test_manager() -> AeadManager {
        let config = AeadConfig {
            max_bytes_per_key: 1024 * 1024, // 1MB
            max_time_per_key: Duration::from_secs(3600), // 1 hour
        };

        let gateway_config = Arc::new(GatewayConfig::default());
        let metrics = Arc::new(MetricsCollector::new(gateway_config).unwrap());
        let master_key = [42u8; 32]; // Test key

        AeadManager::new(config, metrics, master_key)
    }

    #[tokio::test]
    async fn test_encrypt_decrypt_roundtrip() {
        let manager = create_test_manager();
        let peer_id = "test_peer";
        let payload = b"Hello, SCION!";
        let aad = b"additional_auth_data";

        // Encrypt
        let encrypted_frame = manager.encrypt_frame(
            peer_id,
            FrameType::ScionData,
            payload,
            aad,
        ).await.unwrap();

        // Decrypt
        let decrypted = manager.decrypt_frame(peer_id, &encrypted_frame).await.unwrap();

        assert_eq!(decrypted, payload);
    }

    #[tokio::test]
    async fn test_authentication_failure() {
        let manager = create_test_manager();
        let peer_id = "test_peer";
        let payload = b"Test message";
        let aad = b"auth_data";

        // Encrypt frame
        let mut frame = manager.encrypt_frame(
            peer_id,
            FrameType::ScionData,
            payload,
            aad,
        ).await.unwrap();

        // Tamper with ciphertext
        frame.ciphertext[0] ^= 0x01;

        // Decryption should fail
        let result = manager.decrypt_frame(peer_id, &frame).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_sequence_numbers() {
        let manager = create_test_manager();
        let peer_id = "seq_peer";

        // Encrypt multiple frames
        for i in 0..5 {
            let frame = manager.encrypt_frame(
                peer_id,
                FrameType::ScionData,
                &[i as u8],
                b"",
            ).await.unwrap();

            assert_eq!(frame.header.sequence, i + 1);
        }
    }

    #[tokio::test]
    async fn test_key_rotation_trigger() {
        let config = AeadConfig {
            max_bytes_per_key: 10, // Very small limit
            max_time_per_key: Duration::from_secs(3600),
        };

        let gateway_config = Arc::new(GatewayConfig::default());
        let metrics = Arc::new(MetricsCollector::new(gateway_config).unwrap());
        let manager = AeadManager::new(config, metrics, [42u8; 32]);

        let peer_id = "rotation_peer";

        // Process data to trigger rotation
        manager.encrypt_frame(peer_id, FrameType::ScionData, &[0u8; 8], b"").await.unwrap();
        manager.encrypt_frame(peer_id, FrameType::ScionData, &[0u8; 8], b"").await.unwrap();

        // Give time for background rotation
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Should trigger key rotation due to byte limit
        assert!(manager.should_rotate_keys(peer_id).await);
    }

    #[tokio::test]
    async fn test_frame_serialization() {
        let manager = create_test_manager();
        let peer_id = "serial_peer";

        // Create frame
        let original_frame = manager.encrypt_frame(
            peer_id,
            FrameType::Control,
            b"control_data",
            b"aad",
        ).await.unwrap();

        // Serialize
        let serialized = manager.serialize_frame(&original_frame).unwrap();

        // Deserialize
        let deserialized_frame = manager.deserialize_frame(&serialized).unwrap();

        // Compare
        assert_eq!(original_frame.header.sequence, deserialized_frame.header.sequence);
        assert_eq!(original_frame.header.frame_type, deserialized_frame.header.frame_type);
        assert_eq!(original_frame.aad, deserialized_frame.aad);
        assert_eq!(original_frame.ciphertext, deserialized_frame.ciphertext);
    }
}
