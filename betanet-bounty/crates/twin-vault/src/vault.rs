//! Twin Vault implementation with append-only log and encrypted storage
//!
//! Provides secure storage with AES-GCM encryption at rest and OS keystore
//! integration for key management.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use chacha20poly1305::{
    aead::{Aead, KeyInit},
    ChaCha20Poly1305, Nonce,
};
use hkdf::Hkdf;
use rand::{rngs::OsRng, RngCore};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use sled::{Db, Tree};
use thiserror::Error;
use tokio::sync::RwLock;

#[cfg(feature = "os-keystore")]
use keyring::Entry;

use crate::crdt::{CrdtOperationFactory, CrdtState, SignedOperation};
use crate::{TwinId, TwinOperation};

/// Encryption key management
#[derive(Debug, Clone)]
pub struct EncryptionKey {
    key: [u8; 32],
    salt: [u8; 16],
}

impl EncryptionKey {
    /// Generate new random key
    pub fn generate() -> Self {
        let mut key = [0u8; 32];
        let mut salt = [0u8; 16];

        let mut rng = OsRng;
        rng.fill_bytes(&mut key);
        rng.fill_bytes(&mut salt);

        Self { key, salt }
    }

    /// Derive key from password using HKDF
    pub fn from_password(password: &str, salt: [u8; 16]) -> Self {
        let hk = Hkdf::<Sha256>::new(Some(&salt), password.as_bytes());
        let mut key = [0u8; 32];
        hk.expand(b"twin-vault-key", &mut key)
            .expect("Invalid length");

        Self { key, salt }
    }

    /// Get encryption cipher
    fn cipher(&self) -> ChaCha20Poly1305 {
        ChaCha20Poly1305::new_from_slice(&self.key).expect("Invalid key length")
    }

    /// Encrypt data
    pub fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>, VaultError> {
        let cipher = self.cipher();
        let nonce = Nonce::from_slice(&self.salt[..12]); // Use first 12 bytes of salt as nonce

        cipher
            .encrypt(nonce, data)
            .map_err(|e| VaultError::EncryptionError(e.to_string()))
    }

    /// Decrypt data
    pub fn decrypt(&self, encrypted_data: &[u8]) -> Result<Vec<u8>, VaultError> {
        let cipher = self.cipher();
        let nonce = Nonce::from_slice(&self.salt[..12]);

        cipher
            .decrypt(nonce, encrypted_data)
            .map_err(|e| VaultError::DecryptionError(e.to_string()))
    }

    /// Serialize key for storage
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.key);
        bytes.extend_from_slice(&self.salt);
        bytes
    }

    /// Deserialize key from storage
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VaultError> {
        if bytes.len() != 48 {
            return Err(VaultError::InvalidKeyFormat);
        }

        let mut key = [0u8; 32];
        let mut salt = [0u8; 16];

        key.copy_from_slice(&bytes[0..32]);
        salt.copy_from_slice(&bytes[32..48]);

        Ok(Self { key, salt })
    }
}

/// OS Keystore integration
pub struct KeyStore {
    #[cfg(feature = "os-keystore")]
    service: String,
}

impl KeyStore {
    pub fn new(service_name: &str) -> Self {
        Self {
            #[cfg(feature = "os-keystore")]
            service: service_name.to_string(),
        }
    }

    /// Store encryption key in OS keystore
    pub fn store_key(&self, twin_id: &TwinId, key: &EncryptionKey) -> Result<(), VaultError> {
        #[cfg(feature = "os-keystore")]
        {
            let entry = Entry::new(&self.service, &twin_id.to_string())
                .map_err(|e| VaultError::KeystoreError(e.to_string()))?;

            let key_bytes = key.to_bytes();
            let base64_key = base64ct::Base64::encode_string(&key_bytes);

            entry
                .set_password(&base64_key)
                .map_err(|e| VaultError::KeystoreError(e.to_string()))?;

            Ok(())
        }

        #[cfg(not(feature = "os-keystore"))]
        {
            // Fallback: just return success for testing
            let _ = (twin_id, key);
            Ok(())
        }
    }

    /// Retrieve encryption key from OS keystore
    pub fn retrieve_key(&self, twin_id: &TwinId) -> Result<Option<EncryptionKey>, VaultError> {
        #[cfg(feature = "os-keystore")]
        {
            let entry = Entry::new(&self.service, &twin_id.to_string())
                .map_err(|e| VaultError::KeystoreError(e.to_string()))?;

            match entry.get_password() {
                Ok(base64_key) => {
                    let key_bytes = base64ct::Base64::decode_vec(&base64_key)
                        .map_err(|e| VaultError::KeystoreError(e.to_string()))?;

                    let key = EncryptionKey::from_bytes(&key_bytes)?;
                    Ok(Some(key))
                }
                Err(keyring::Error::NoEntry) => Ok(None),
                Err(e) => Err(VaultError::KeystoreError(e.to_string())),
            }
        }

        #[cfg(not(feature = "os-keystore"))]
        {
            // Fallback: return None for testing
            let _ = twin_id;
            Ok(None)
        }
    }

    /// Delete key from OS keystore
    pub fn delete_key(&self, twin_id: &TwinId) -> Result<(), VaultError> {
        #[cfg(feature = "os-keystore")]
        {
            let entry = Entry::new(&self.service, &twin_id.to_string())
                .map_err(|e| VaultError::KeystoreError(e.to_string()))?;

            entry
                .delete_password()
                .map_err(|e| VaultError::KeystoreError(e.to_string()))?;

            Ok(())
        }

        #[cfg(not(feature = "os-keystore"))]
        {
            // Fallback: just return success for testing
            let _ = twin_id;
            Ok(())
        }
    }
}

/// Vault configuration
#[derive(Debug, Clone)]
pub struct VaultConfig {
    pub storage_path: PathBuf,
    pub encryption_key: Option<EncryptionKey>,
    pub use_os_keystore: bool,
    pub max_log_size: usize,
    pub auto_compact: bool,
}

impl Default for VaultConfig {
    fn default() -> Self {
        Self {
            storage_path: PathBuf::from("./twin-vault-data"),
            encryption_key: None,
            use_os_keystore: true,
            max_log_size: 100_000, // 100k operations
            auto_compact: true,
        }
    }
}

/// Append-only log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub sequence: u64,
    pub timestamp: u64,
    pub operation: TwinOperation,
    pub result: Option<Bytes>,
    pub actor_id: String,
    pub signature: Vec<u8>,
}

impl LogEntry {
    pub fn new(
        sequence: u64,
        operation: TwinOperation,
        result: Option<Bytes>,
        actor_id: String,
        signature: Vec<u8>,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            sequence,
            timestamp,
            operation,
            result,
            actor_id,
            signature,
        }
    }
}

/// Twin Vault with encrypted storage and append-only log
pub struct TwinVault {
    twin_id: TwinId,
    db: Db,
    log_tree: Tree,
    state_tree: Tree,
    crdt_state: Arc<RwLock<CrdtState>>,
    operation_factory: CrdtOperationFactory,
    encryption_key: EncryptionKey,
    keystore: KeyStore,
    config: VaultConfig,
    sequence_counter: Arc<std::sync::atomic::AtomicU64>,
}

impl TwinVault {
    /// Create new twin vault
    pub async fn new(twin_id: TwinId, config: VaultConfig) -> Result<Self, VaultError> {
        // Create storage directory
        std::fs::create_dir_all(&config.storage_path).map_err(VaultError::IoError)?;

        // Open database
        let db_path = config
            .storage_path
            .join(format!("twin-{}.db", twin_id.to_string().replace(':', "_")));
        let db = sled::open(&db_path).map_err(VaultError::DatabaseError)?;

        // Open trees
        let log_tree = db.open_tree("log").map_err(VaultError::DatabaseError)?;
        let state_tree = db.open_tree("state").map_err(VaultError::DatabaseError)?;

        // Initialize keystore
        let keystore = KeyStore::new("twin-vault");

        // Get or create encryption key
        let encryption_key = if let Some(key) = config.encryption_key.clone() {
            key
        } else if config.use_os_keystore {
            if let Some(key) = keystore.retrieve_key(&twin_id)? {
                key
            } else {
                let key = EncryptionKey::generate();
                keystore.store_key(&twin_id, &key)?;
                key
            }
        } else {
            EncryptionKey::generate()
        };

        // Create operation factory
        let operation_factory = CrdtOperationFactory::new(twin_id.agent_id.to_string());

        // Load or create CRDT state
        let crdt_state = Self::load_crdt_state(&state_tree, &encryption_key).await?;

        // Get current sequence number
        let sequence_counter = Arc::new(std::sync::atomic::AtomicU64::new(
            Self::get_last_sequence(&log_tree)?,
        ));

        Ok(Self {
            twin_id,
            db,
            log_tree,
            state_tree,
            crdt_state: Arc::new(RwLock::new(crdt_state)),
            operation_factory,
            encryption_key,
            keystore,
            config,
            sequence_counter,
        })
    }

    /// Set value in the vault
    pub async fn set(&self, key: String, value: Bytes, timestamp: u64) -> Result<(), VaultError> {
        // Create signed CRDT operation
        let signed_op = self
            .operation_factory
            .create_lww_set(key.clone(), value.clone())?;

        // Apply to CRDT state
        {
            let mut state = self.crdt_state.write().await;
            let lww_map = state.get_lww_map("default");
            lww_map.set_signed(signed_op.clone())?;
        }

        // Create operation
        let operation = TwinOperation::Write {
            key,
            value,
            timestamp,
        };

        // Append to log
        self.append_to_log(operation, None, signed_op.signature)
            .await?;

        // Save state
        self.save_crdt_state().await?;

        Ok(())
    }

    /// Get value from the vault
    pub async fn get(&self, key: &str) -> Result<Option<Bytes>, VaultError> {
        let state = self.crdt_state.read().await;
        let lww_map = state.lww_maps.get("default");

        if let Some(map) = lww_map {
            Ok(map.get(key).cloned())
        } else {
            Ok(None)
        }
    }

    /// Delete value from the vault
    pub async fn delete(&self, key: &str, timestamp: u64) -> Result<(), VaultError> {
        // Create signed CRDT operation
        let signed_op = self.operation_factory.create_lww_delete(key.to_string())?;

        // Apply to CRDT state
        {
            let mut state = self.crdt_state.write().await;
            let lww_map = state.get_lww_map("default");
            lww_map.delete_signed(key.to_string(), signed_op.clone())?;
        }

        // Create operation
        let operation = TwinOperation::Delete {
            key: key.to_string(),
            timestamp,
        };

        // Append to log
        self.append_to_log(operation, None, signed_op.signature)
            .await?;

        // Save state
        self.save_crdt_state().await?;

        Ok(())
    }

    /// Increment counter
    pub async fn increment_counter(
        &self,
        counter_id: &str,
        amount: u64,
        actor_id: &str,
        timestamp: u64,
    ) -> Result<(), VaultError> {
        // Create signed CRDT operation
        let signed_op = self
            .operation_factory
            .create_g_counter_increment(counter_id.to_string(), amount)?;

        // Apply to CRDT state
        {
            let mut state = self.crdt_state.write().await;
            let counter = state.get_g_counter(counter_id);
            counter.increment_signed(signed_op.clone())?;
        }

        // Create operation
        let operation = TwinOperation::Increment {
            counter_id: counter_id.to_string(),
            amount,
            actor_id: actor_id.to_string(),
            timestamp,
        };

        // Append to log
        self.append_to_log(operation, None, signed_op.signature)
            .await?;

        // Save state
        self.save_crdt_state().await?;

        Ok(())
    }

    /// Get counter value
    pub async fn get_counter(&self, counter_id: &str) -> Result<u64, VaultError> {
        let state = self.crdt_state.read().await;
        Ok(state
            .g_counters
            .get(counter_id)
            .map(|c| c.value())
            .unwrap_or(0))
    }

    /// Get all keys
    pub async fn keys(&self) -> Result<Vec<String>, VaultError> {
        let state = self.crdt_state.read().await;
        if let Some(lww_map) = state.lww_maps.get("default") {
            Ok(lww_map.keys())
        } else {
            Ok(Vec::new())
        }
    }

    /// Get CRDT state for synchronization
    pub async fn get_state(&self) -> Result<CrdtState, VaultError> {
        let state = self.crdt_state.read().await;
        Ok(state.clone())
    }

    /// Merge with remote state
    pub async fn merge_state(&self, remote_state: CrdtState) -> Result<(), VaultError> {
        let mut state = self.crdt_state.write().await;
        state.merge(&remote_state)?;
        state.update_sync_timestamp();

        // Save merged state
        drop(state);
        self.save_crdt_state().await?;

        Ok(())
    }

    /// Get vault statistics
    pub async fn stats(&self) -> Result<VaultStats, VaultError> {
        let log_entries = self.log_tree.len();
        let state = self.crdt_state.read().await;
        let total_keys = state
            .lww_maps
            .get("default")
            .map(|m| m.keys().len())
            .unwrap_or(0);
        let total_counters = state.g_counters.len();

        Ok(VaultStats {
            total_operations: log_entries as u64,
            total_keys: total_keys as u64,
            total_counters: total_counters as u64,
            last_sync_timestamp: state.last_sync_timestamp,
            vault_size_bytes: self.estimate_size()?,
        })
    }

    // Private methods

    async fn append_to_log(
        &self,
        operation: TwinOperation,
        result: Option<Bytes>,
        signature: Vec<u8>,
    ) -> Result<(), VaultError> {
        let sequence = self
            .sequence_counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;

        let log_entry = LogEntry::new(
            sequence,
            operation,
            result,
            self.twin_id.agent_id.to_string(),
            signature,
        );

        // Encrypt log entry
        let entry_bytes = bincode::serialize(&log_entry)
            .map_err(|e| VaultError::SerializationError(e.to_string()))?;
        let encrypted_entry = self.encryption_key.encrypt(&entry_bytes)?;

        // Store in log tree
        let key = sequence.to_be_bytes();
        self.log_tree
            .insert(key, encrypted_entry)
            .map_err(VaultError::DatabaseError)?;

        // Flush to disk
        self.log_tree.flush().map_err(VaultError::DatabaseError)?;

        Ok(())
    }

    async fn load_crdt_state(
        state_tree: &Tree,
        encryption_key: &EncryptionKey,
    ) -> Result<CrdtState, VaultError> {
        if let Some(encrypted_state) = state_tree
            .get("current_state")
            .map_err(VaultError::DatabaseError)?
        {
            let state_bytes = encryption_key.decrypt(&encrypted_state)?;
            let state: CrdtState = bincode::deserialize(&state_bytes)
                .map_err(|e| VaultError::SerializationError(e.to_string()))?;

            Ok(state)
        } else {
            Ok(CrdtState::new())
        }
    }

    async fn save_crdt_state(&self) -> Result<(), VaultError> {
        let state = self.crdt_state.read().await;
        let state_bytes = bincode::serialize(&*state)
            .map_err(|e| VaultError::SerializationError(e.to_string()))?;
        let encrypted_state = self.encryption_key.encrypt(&state_bytes)?;

        self.state_tree
            .insert("current_state", encrypted_state)
            .map_err(VaultError::DatabaseError)?;
        self.state_tree.flush().map_err(VaultError::DatabaseError)?;

        Ok(())
    }

    fn get_last_sequence(log_tree: &Tree) -> Result<u64, VaultError> {
        if let Some((key, _)) = log_tree.last().map_err(VaultError::DatabaseError)? {
            let sequence = u64::from_be_bytes(
                key.as_ref()
                    .try_into()
                    .map_err(|_| VaultError::CorruptedData("Invalid sequence key".to_string()))?,
            );
            Ok(sequence)
        } else {
            Ok(0)
        }
    }

    fn estimate_size(&self) -> Result<u64, VaultError> {
        // Use approximation since size_on_disk doesn't exist in this sled version
        let log_size = self.log_tree.len() * 128; // Rough estimate
        let state_size = self.state_tree.len() * 128; // Rough estimate
        Ok((log_size + state_size) as u64)
    }
}

/// Vault statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultStats {
    pub total_operations: u64,
    pub total_keys: u64,
    pub total_counters: u64,
    pub last_sync_timestamp: u64,
    pub vault_size_bytes: u64,
}

/// Vault errors
#[derive(Debug, Error)]
pub enum VaultError {
    #[error("CRDT error: {0}")]
    CrdtError(#[from] crate::crdt::CrdtError),

    #[error("Database error: {0}")]
    DatabaseError(#[from] sled::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Encryption error: {0}")]
    EncryptionError(String),

    #[error("Decryption error: {0}")]
    DecryptionError(String),

    #[error("Keystore error: {0}")]
    KeystoreError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Invalid key format")]
    InvalidKeyFormat,

    #[error("Corrupted data: {0}")]
    CorruptedData(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_vault() -> (TwinVault, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let twin_id = TwinId::new(crate::AgentId::new("test-agent", "test-node"), "test-vault");

        let config = VaultConfig {
            storage_path: temp_dir.path().to_path_buf(),
            encryption_key: Some(EncryptionKey::generate()),
            use_os_keystore: false,
            ..Default::default()
        };

        let vault = TwinVault::new(twin_id, config).await.unwrap();
        (vault, temp_dir)
    }

    #[tokio::test]
    async fn test_vault_basic_operations() {
        let (vault, _temp_dir) = create_test_vault().await;

        // Set a value
        vault
            .set("key1".to_string(), Bytes::from("value1"), 12345)
            .await
            .unwrap();

        // Get the value
        let result = vault.get("key1").await.unwrap();
        assert_eq!(result, Some(Bytes::from("value1")));

        // Delete the value
        vault.delete("key1", 12346).await.unwrap();

        // Verify deletion
        let result = vault.get("key1").await.unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_vault_counter_operations() {
        let (vault, _temp_dir) = create_test_vault().await;

        // Increment counter
        vault
            .increment_counter("counter1", 5, "actor1", 12345)
            .await
            .unwrap();

        // Get counter value
        let value = vault.get_counter("counter1").await.unwrap();
        assert_eq!(value, 5);

        // Increment again
        vault
            .increment_counter("counter1", 3, "actor1", 12346)
            .await
            .unwrap();

        // Check new value
        let value = vault.get_counter("counter1").await.unwrap();
        assert_eq!(value, 8);
    }

    #[tokio::test]
    async fn test_vault_state_merge() {
        let (vault1, _temp_dir1) = create_test_vault().await;
        let (vault2, _temp_dir2) = create_test_vault().await;

        // Add data to both vaults
        vault1
            .set("key1".to_string(), Bytes::from("value1"), 12345)
            .await
            .unwrap();
        vault2
            .increment_counter("counter1", 10, "actor2", 12346)
            .await
            .unwrap();

        // Get state from vault2
        let state2 = vault2.get_state().await.unwrap();

        // Merge into vault1
        vault1.merge_state(state2).await.unwrap();

        // Verify merge
        let value = vault1.get("key1").await.unwrap();
        assert_eq!(value, Some(Bytes::from("value1")));

        let counter = vault1.get_counter("counter1").await.unwrap();
        assert_eq!(counter, 10);
    }

    #[tokio::test]
    async fn test_encryption_key_operations() {
        let key = EncryptionKey::generate();
        let plaintext = b"Hello, world!";

        // Encrypt
        let encrypted = key.encrypt(plaintext).unwrap();
        assert_ne!(encrypted.as_slice(), plaintext);

        // Decrypt
        let decrypted = key.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted.as_slice(), plaintext);
    }

    #[tokio::test]
    async fn test_encryption_key_serialization() {
        let key = EncryptionKey::generate();
        let bytes = key.to_bytes();
        let restored_key = EncryptionKey::from_bytes(&bytes).unwrap();

        let plaintext = b"Test data";
        let encrypted1 = key.encrypt(plaintext).unwrap();
        let encrypted2 = restored_key.encrypt(plaintext).unwrap();

        // Both keys should be able to decrypt each other's data
        let decrypted1 = restored_key.decrypt(&encrypted1).unwrap();
        let decrypted2 = key.decrypt(&encrypted2).unwrap();

        assert_eq!(decrypted1.as_slice(), plaintext);
        assert_eq!(decrypted2.as_slice(), plaintext);
    }

    #[tokio::test]
    async fn test_vault_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let twin_id = TwinId::new(crate::AgentId::new("test-agent", "test-node"), "test-vault");

        let encryption_key = EncryptionKey::generate();

        // Create vault and add data
        {
            let config = VaultConfig {
                storage_path: temp_dir.path().to_path_buf(),
                encryption_key: Some(encryption_key.clone()),
                use_os_keystore: false,
                ..Default::default()
            };

            let vault = TwinVault::new(twin_id.clone(), config).await.unwrap();
            vault
                .set(
                    "persistent_key".to_string(),
                    Bytes::from("persistent_value"),
                    12345,
                )
                .await
                .unwrap();
        }

        // Recreate vault and verify data persists
        {
            let config = VaultConfig {
                storage_path: temp_dir.path().to_path_buf(),
                encryption_key: Some(encryption_key),
                use_os_keystore: false,
                ..Default::default()
            };

            let vault = TwinVault::new(twin_id, config).await.unwrap();
            let result = vault.get("persistent_key").await.unwrap();
            assert_eq!(result, Some(Bytes::from("persistent_value")));
        }
    }

    #[test]
    fn test_keystore_operations() {
        let keystore = KeyStore::new("test-service");
        let twin_id = TwinId::new(
            crate::AgentId::new("test-agent", "test-node"),
            "test-keystore",
        );
        let key = EncryptionKey::generate();

        // Store key (may fail on systems without keystore)
        if keystore.store_key(&twin_id, &key).is_ok() {
            // Retrieve key
            let retrieved = keystore.retrieve_key(&twin_id).unwrap();
            assert!(retrieved.is_some());

            // Delete key
            keystore.delete_key(&twin_id).unwrap();

            // Verify deletion
            let retrieved = keystore.retrieve_key(&twin_id).unwrap();
            assert!(retrieved.is_none());
        }
    }
}
