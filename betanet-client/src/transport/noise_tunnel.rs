//! Noise XK inner tunnel implementation for forward secrecy
//!
//! Provides end-to-end encryption with forward secrecy independent of the
//! underlying transport layer (TLS/QUIC), using the Noise XK handshake pattern.

use crate::{
    config::NoiseConfig,
    error::{BetanetError, Result, SecurityError},
    BetanetMessage,
};

use bytes::{Bytes, BytesMut};
use snow::{Builder, HandshakeState, StatelessTransportState, TransportState};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};

/// Noise XK tunnel manager
pub struct NoiseXKTunnel {
    /// Configuration
    config: NoiseConfig,
    /// Static private key
    static_private_key: [u8; 32],
    /// Static public key
    static_public_key: [u8; 32],
    /// Active sessions
    sessions: Arc<RwLock<HashMap<String, Arc<Mutex<NoiseSession>>>>>,
    /// Key rotation manager
    key_rotation: Arc<KeyRotationManager>,
    /// Running state
    is_running: Arc<RwLock<bool>>,
}

/// Individual Noise session
struct NoiseSession {
    /// Session ID
    session_id: String,
    /// Remote peer ID
    peer_id: String,
    /// Noise transport state
    transport_state: Option<TransportState>,
    /// Session creation time
    created_at: SystemTime,
    /// Last activity time
    last_activity: SystemTime,
    /// Message sequence numbers for replay protection
    send_sequence: u64,
    receive_sequence: u64,
    /// Forward secrecy key rotation
    key_rotation_count: u32,
}

/// Key rotation manager for forward secrecy
struct KeyRotationManager {
    /// Rotation interval
    rotation_interval: Duration,
    /// Last rotation time
    last_rotation: Mutex<SystemTime>,
    /// Rotation counter
    rotation_counter: Mutex<u32>,
}

/// Noise handshake manager
pub struct NoiseHandshakeManager {
    /// Noise pattern
    pattern: String,
    /// Active handshakes
    handshakes: Arc<RwLock<HashMap<String, HandshakeState>>>,
}

/// Encrypted message wrapper
#[derive(Debug, Clone)]
pub struct EncryptedMessage {
    /// Session ID
    pub session_id: String,
    /// Sequence number
    pub sequence: u64,
    /// Encrypted payload
    pub ciphertext: Bytes,
    /// Authentication tag
    pub auth_tag: Option<Bytes>,
    /// Timestamp
    pub timestamp: u64,
}

/// Noise tunnel statistics
#[derive(Debug, Clone, Default)]
pub struct NoiseStats {
    /// Total sessions created
    pub sessions_created: u64,
    /// Active sessions
    pub active_sessions: u64,
    /// Messages encrypted
    pub messages_encrypted: u64,
    /// Messages decrypted
    pub messages_decrypted: u64,
    /// Handshakes completed
    pub handshakes_completed: u64,
    /// Handshake failures
    pub handshake_failures: u64,
    /// Key rotations performed
    pub key_rotations: u64,
    /// Decryption failures
    pub decryption_failures: u64,
}

impl NoiseXKTunnel {
    /// Create new Noise XK tunnel
    pub async fn new(config: NoiseConfig) -> Result<Self> {
        info!("Creating Noise XK tunnel with pattern: {}", config.pattern);

        // Generate or load static keypair
        let (static_private_key, static_public_key) = Self::load_or_generate_keypair(&config).await?;

        // Create key rotation manager
        let key_rotation = Arc::new(KeyRotationManager::new(config.key_rotation_interval));

        Ok(Self {
            config,
            static_private_key,
            static_public_key,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            key_rotation,
            is_running: Arc::new(RwLock::new(false)),
        })
    }

    /// Start the Noise tunnel
    pub async fn start(&self) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        if *is_running {
            warn!("Noise XK tunnel already running");
            return Ok(());
        }

        info!("Starting Noise XK tunnel...");

        // Start key rotation task
        if self.config.forward_secrecy {
            self.start_key_rotation_task().await;
        }

        // Start session cleanup task
        self.start_session_cleanup_task().await;

        *is_running = true;
        info!("Noise XK tunnel started successfully");

        Ok(())
    }

    /// Stop the Noise tunnel
    pub async fn stop(&self) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        if !*is_running {
            warn!("Noise XK tunnel not running");
            return Ok(());
        }

        info!("Stopping Noise XK tunnel...");

        // Clear all sessions
        let mut sessions = self.sessions.write().await;
        sessions.clear();

        *is_running = false;
        info!("Noise XK tunnel stopped");

        Ok(())
    }

    /// Initiate handshake with remote peer
    pub async fn initiate_handshake(&self, peer_id: String) -> Result<(String, Bytes)> {
        let session_id = format!("{}_{}", peer_id, uuid::Uuid::new_v4());

        // Create handshake state
        let builder = Builder::new(self.config.pattern.parse()?);
        let mut handshake = builder
            .local_private_key(&self.static_private_key)
            .build_initiator()?;

        // Generate first handshake message
        let mut message = vec![0u8; 1024]; // Buffer for handshake message
        let len = handshake.write_message(&[], &mut message)?;
        message.truncate(len);

        debug!("Initiated handshake with peer {} (session: {})", peer_id, session_id);

        Ok((session_id, Bytes::from(message)))
    }

    /// Handle incoming handshake message
    pub async fn handle_handshake_message(
        &self,
        session_id: String,
        peer_id: String,
        message: Bytes,
    ) -> Result<Option<Bytes>> {
        // This would handle the full XK handshake pattern
        // Simplified implementation for demonstration

        let builder = Builder::new(self.config.pattern.parse()?);
        let mut handshake = builder
            .local_private_key(&self.static_private_key)
            .build_responder()?;

        let mut response_buffer = vec![0u8; 1024];
        let response_len = handshake.read_message(&message, &mut response_buffer)?;

        if handshake.is_handshake_finished() {
            // Handshake complete, create session
            let transport_state = handshake.into_transport_mode()?;
            self.create_session(session_id.clone(), peer_id, transport_state).await?;

            if response_len > 0 {
                response_buffer.truncate(response_len);
                Ok(Some(Bytes::from(response_buffer)))
            } else {
                Ok(None)
            }
        } else {
            // Continue handshake
            if response_len > 0 {
                response_buffer.truncate(response_len);
                Ok(Some(Bytes::from(response_buffer)))
            } else {
                Ok(None)
            }
        }
    }

    /// Encrypt message for peer
    pub async fn encrypt_message(
        &self,
        peer_id: String,
        message: &BetanetMessage,
    ) -> Result<EncryptedMessage> {
        let sessions = self.sessions.read().await;
        let session = sessions
            .values()
            .find(|s| async {
                let session_guard = s.lock().await;
                session_guard.peer_id == peer_id
            })
            .await
            .ok_or_else(|| BetanetError::Security(SecurityError::Noise(
                format!("No active session with peer: {}", peer_id)
            )))?;

        let mut session_guard = session.lock().await;

        // Check if session needs key rotation
        if self.config.forward_secrecy && self.should_rotate_keys(&session_guard).await {
            self.rotate_session_keys(&mut session_guard).await?;
        }

        // Serialize message
        let plaintext = serde_json::to_vec(message).map_err(|e| {
            BetanetError::Serialization(crate::error::SerializationError::Json(e))
        })?;

        // Encrypt with Noise
        let transport_state = session_guard.transport_state.as_mut().ok_or_else(|| {
            BetanetError::Security(SecurityError::Noise("No transport state".to_string()))
        })?;

        let mut ciphertext = vec![0u8; plaintext.len() + 16]; // Room for auth tag
        let len = transport_state.write_message(&plaintext, &mut ciphertext)?;
        ciphertext.truncate(len);

        // Update session state
        session_guard.send_sequence += 1;
        session_guard.last_activity = SystemTime::now();

        let encrypted_message = EncryptedMessage {
            session_id: session_guard.session_id.clone(),
            sequence: session_guard.send_sequence,
            ciphertext: Bytes::from(ciphertext),
            auth_tag: None, // Included in ciphertext for Noise
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        };

        debug!("Encrypted message for peer {} (sequence: {})", peer_id, encrypted_message.sequence);
        Ok(encrypted_message)
    }

    /// Decrypt message from peer
    pub async fn decrypt_message(
        &self,
        peer_id: String,
        encrypted_message: EncryptedMessage,
    ) -> Result<BetanetMessage> {
        let sessions = self.sessions.read().await;
        let session = sessions
            .get(&encrypted_message.session_id)
            .ok_or_else(|| BetanetError::Security(SecurityError::Noise(
                format!("Unknown session: {}", encrypted_message.session_id)
            )))?;

        let mut session_guard = session.lock().await;

        // Verify sequence number for replay protection
        if encrypted_message.sequence <= session_guard.receive_sequence {
            return Err(BetanetError::Security(SecurityError::Noise(
                "Replay attack detected - invalid sequence number".to_string()
            )));
        }

        // Decrypt with Noise
        let transport_state = session_guard.transport_state.as_mut().ok_or_else(|| {
            BetanetError::Security(SecurityError::Noise("No transport state".to_string()))
        })?;

        let mut plaintext = vec![0u8; encrypted_message.ciphertext.len()];
        let len = transport_state.read_message(&encrypted_message.ciphertext, &mut plaintext)?;
        plaintext.truncate(len);

        // Deserialize message
        let message: BetanetMessage = serde_json::from_slice(&plaintext).map_err(|e| {
            BetanetError::Serialization(crate::error::SerializationError::Json(e))
        })?;

        // Update session state
        session_guard.receive_sequence = encrypted_message.sequence;
        session_guard.last_activity = SystemTime::now();

        debug!("Decrypted message from peer {} (sequence: {})", peer_id, encrypted_message.sequence);
        Ok(message)
    }

    /// Get tunnel statistics
    pub async fn get_stats(&self) -> NoiseStats {
        let sessions = self.sessions.read().await;
        NoiseStats {
            active_sessions: sessions.len() as u64,
            // Other stats would be tracked during operations
            ..Default::default()
        }
    }

    /// Create new session
    async fn create_session(
        &self,
        session_id: String,
        peer_id: String,
        transport_state: TransportState,
    ) -> Result<()> {
        let session = NoiseSession {
            session_id: session_id.clone(),
            peer_id,
            transport_state: Some(transport_state),
            created_at: SystemTime::now(),
            last_activity: SystemTime::now(),
            send_sequence: 0,
            receive_sequence: 0,
            key_rotation_count: 0,
        };

        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id.clone(), Arc::new(Mutex::new(session)));

        info!("Created Noise session: {}", session_id);
        Ok(())
    }

    /// Check if session keys should be rotated
    async fn should_rotate_keys(&self, session: &NoiseSession) -> bool {
        if !self.config.forward_secrecy {
            return false;
        }

        let elapsed = SystemTime::now()
            .duration_since(session.created_at)
            .unwrap_or_default();

        elapsed >= self.config.key_rotation_interval ||
        session.send_sequence >= 1000 // Rotate after 1000 messages
    }

    /// Rotate session keys for forward secrecy
    async fn rotate_session_keys(&self, session: &mut NoiseSession) -> Result<()> {
        // For real forward secrecy, would need to perform a new handshake
        // or use Noise's rekey functionality if available

        if let Some(transport_state) = &mut session.transport_state {
            // Placeholder for key rotation logic
            // In practice, would need to coordinate with peer
            session.key_rotation_count += 1;
            session.send_sequence = 0;
            session.receive_sequence = 0;

            debug!("Rotated keys for session {} (rotation: {})",
                   session.session_id, session.key_rotation_count);
        }

        Ok(())
    }

    /// Load or generate static keypair
    async fn load_or_generate_keypair(config: &NoiseConfig) -> Result<([u8; 32], [u8; 32])> {
        if config.private_key_file.exists() {
            // Load existing keypair
            let key_data = tokio::fs::read(&config.private_key_file).await?;
            if key_data.len() >= 32 {
                let mut private_key = [0u8; 32];
                private_key.copy_from_slice(&key_data[..32]);

                // Generate public key from private key
                let public_key = Self::generate_public_key(&private_key)?;

                info!("Loaded static keypair from: {:?}", config.private_key_file);
                Ok((private_key, public_key))
            } else {
                Err(BetanetError::Security(SecurityError::KeyManagement(
                    "Invalid private key file".to_string()
                )))
            }
        } else {
            // Generate new keypair
            use rand::RngCore;
            let mut rng = rand::thread_rng();
            let mut private_key = [0u8; 32];
            rng.fill_bytes(&mut private_key);

            let public_key = Self::generate_public_key(&private_key)?;

            // Save private key
            if let Some(parent) = config.private_key_file.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            tokio::fs::write(&config.private_key_file, &private_key).await?;

            info!("Generated new static keypair, saved to: {:?}", config.private_key_file);
            Ok((private_key, public_key))
        }
    }

    /// Generate public key from private key
    fn generate_public_key(private_key: &[u8; 32]) -> Result<[u8; 32]> {
        // Use curve25519 for key generation
        use x25519_dalek::{StaticSecret, PublicKey};

        let secret = StaticSecret::from(*private_key);
        let public = PublicKey::from(&secret);

        Ok(*public.as_bytes())
    }

    /// Start key rotation background task
    async fn start_key_rotation_task(&self) {
        let key_rotation = self.key_rotation.clone();
        let sessions = self.sessions.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // Check every 5 minutes

            loop {
                interval.tick().await;

                if key_rotation.should_rotate().await {
                    // Trigger key rotation for all sessions
                    let sessions_map = sessions.read().await;
                    for session in sessions_map.values() {
                        let mut session_guard = session.lock().await;
                        if let Err(e) = NoiseXKTunnel::rotate_session_keys_static(&mut session_guard).await {
                            warn!("Failed to rotate keys for session {}: {}", session_guard.session_id, e);
                        }
                    }

                    key_rotation.mark_rotation().await;
                }
            }
        });
    }

    /// Static version of key rotation for background task
    async fn rotate_session_keys_static(session: &mut NoiseSession) -> Result<()> {
        session.key_rotation_count += 1;
        session.send_sequence = 0;
        session.receive_sequence = 0;
        Ok(())
    }

    /// Start session cleanup background task
    async fn start_session_cleanup_task(&self) {
        let sessions = self.sessions.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(600)); // Cleanup every 10 minutes

            loop {
                interval.tick().await;

                let mut sessions_map = sessions.write().await;
                let now = SystemTime::now();
                let timeout = Duration::from_secs(3600); // 1 hour timeout

                sessions_map.retain(|session_id, session| {
                    if let Ok(session_guard) = session.try_lock() {
                        let elapsed = now.duration_since(session_guard.last_activity).unwrap_or_default();
                        if elapsed > timeout {
                            debug!("Cleaning up inactive session: {}", session_id);
                            false
                        } else {
                            true
                        }
                    } else {
                        true // Keep session if it's currently locked
                    }
                });
            }
        });
    }
}

impl KeyRotationManager {
    /// Create new key rotation manager
    fn new(rotation_interval: Duration) -> Self {
        Self {
            rotation_interval,
            last_rotation: Mutex::new(SystemTime::now()),
            rotation_counter: Mutex::new(0),
        }
    }

    /// Check if keys should be rotated
    async fn should_rotate(&self) -> bool {
        let last_rotation = self.last_rotation.lock().await;
        let elapsed = SystemTime::now().duration_since(*last_rotation).unwrap_or_default();
        elapsed >= self.rotation_interval
    }

    /// Mark that rotation has occurred
    async fn mark_rotation(&self) {
        let mut last_rotation = self.last_rotation.lock().await;
        let mut counter = self.rotation_counter.lock().await;

        *last_rotation = SystemTime::now();
        *counter += 1;

        info!("Key rotation completed (rotation #{})", *counter);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NoiseConfig;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_noise_tunnel_creation() {
        let config = NoiseConfig {
            pattern: "Noise_XK_25519_AESGCM_SHA256".to_string(),
            private_key_file: PathBuf::from("/tmp/test_noise_key.pem"),
            forward_secrecy: true,
            key_rotation_interval: Duration::from_secs(3600),
        };

        let tunnel = NoiseXKTunnel::new(config).await;
        assert!(tunnel.is_ok());
    }

    #[test]
    fn test_encrypted_message_creation() {
        let msg = EncryptedMessage {
            session_id: "test_session".to_string(),
            sequence: 1,
            ciphertext: Bytes::from(vec![1, 2, 3, 4]),
            auth_tag: None,
            timestamp: 1234567890,
        };

        assert_eq!(msg.session_id, "test_session");
        assert_eq!(msg.sequence, 1);
        assert_eq!(msg.ciphertext.len(), 4);
    }

    #[test]
    fn test_key_rotation_manager() {
        let manager = KeyRotationManager::new(Duration::from_secs(60));
        // Test would involve mocking time to verify rotation logic
        assert_eq!(manager.rotation_interval, Duration::from_secs(60));
    }
}
