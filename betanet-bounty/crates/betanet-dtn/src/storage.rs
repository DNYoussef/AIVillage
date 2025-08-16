//! Persistent bundle storage using sled embedded database
//!
//! Provides durable storage for bundles with efficient indexing by various criteria.

use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use sled::{Db, Tree};
use thiserror::Error;
use tokio::sync::RwLock;

use crate::bundle::{Bundle, BundleId, EndpointId};

/// Bundle metadata for efficient queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleMetadata {
    pub id: BundleId,
    pub size: usize,
    pub destination: EndpointId,
    pub source: EndpointId,
    pub creation_time: u64,
    pub lifetime_ms: u64,
    pub priority: u8,
    pub forwarded_count: u32,
    pub stored_at: u64, // Unix timestamp
}

impl BundleMetadata {
    pub fn from_bundle(bundle: &Bundle) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            id: bundle.id(),
            size: bundle.size(),
            destination: bundle.primary.destination.clone(),
            source: bundle.primary.source.clone(),
            creation_time: bundle.primary.creation_timestamp.dtn_time,
            lifetime_ms: bundle.primary.lifetime,
            priority: 1, // Normal priority by default
            forwarded_count: 0,
            stored_at: now,
        }
    }

    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let created_unix_ms = (self.creation_time + 946684800) * 1000; // Convert DTN time to Unix time
        let expiry_time = created_unix_ms + self.lifetime_ms;

        now >= expiry_time
    }
}

/// Persistent bundle storage
pub struct BundleStore {
    db: Db,
    bundles: Tree,
    metadata: Tree,
    by_destination: Tree,
    by_source: Tree,
    stats: Arc<RwLock<StorageStats>>,
}

impl BundleStore {
    /// Open or create a bundle store at the given path
    pub async fn open<P: AsRef<Path>>(path: P) -> Result<Self, StorageError> {
        let db = sled::open(path).map_err(StorageError::DatabaseError)?;

        let bundles = db.open_tree("bundles").map_err(StorageError::DatabaseError)?;
        let metadata = db.open_tree("metadata").map_err(StorageError::DatabaseError)?;
        let by_destination = db.open_tree("by_destination").map_err(StorageError::DatabaseError)?;
        let by_source = db.open_tree("by_source").map_err(StorageError::DatabaseError)?;

        let stats = Arc::new(RwLock::new(StorageStats::default()));

        Ok(Self {
            db,
            bundles,
            metadata,
            by_destination,
            by_source,
            stats,
        })
    }

    /// Create an in-memory store for testing
    pub async fn memory() -> Result<Self, StorageError> {
        let config = sled::Config::new().temporary(true);
        let db = config.open().map_err(StorageError::DatabaseError)?;

        let bundles = db.open_tree("bundles").map_err(StorageError::DatabaseError)?;
        let metadata = db.open_tree("metadata").map_err(StorageError::DatabaseError)?;
        let by_destination = db.open_tree("by_destination").map_err(StorageError::DatabaseError)?;
        let by_source = db.open_tree("by_source").map_err(StorageError::DatabaseError)?;

        let stats = Arc::new(RwLock::new(StorageStats::default()));

        Ok(Self {
            db,
            bundles,
            metadata,
            by_destination,
            by_source,
            stats,
        })
    }

    /// Store a bundle
    pub async fn store(&self, bundle: Bundle) -> Result<(), StorageError> {
        let bundle_id = bundle.id();
        let id_key = self.bundle_id_key(&bundle_id);

        // Check if bundle already exists
        if self.bundles.contains_key(&id_key).map_err(StorageError::DatabaseError)? {
            return Err(StorageError::BundleExists(bundle_id));
        }

        // Encode bundle
        let bundle_data = bundle.encode().map_err(StorageError::BundleError)?;

        // Create metadata
        let metadata = BundleMetadata::from_bundle(&bundle);
        let metadata_data = bincode::serialize(&metadata)
            .map_err(|e| StorageError::SerializationError(e.to_string()))?;

        // Store in multiple indexes
        self.bundles.insert(&id_key, bundle_data.as_ref())
            .map_err(StorageError::DatabaseError)?;

        self.metadata.insert(&id_key, metadata_data)
            .map_err(StorageError::DatabaseError)?;

        // Index by destination
        let dest_key = format!("{}#{}", bundle.primary.destination, bundle_id);
        self.by_destination.insert(dest_key.as_bytes(), &*id_key)
            .map_err(StorageError::DatabaseError)?;

        // Index by source
        let src_key = format!("{}#{}", bundle.primary.source, bundle_id);
        self.by_source.insert(src_key.as_bytes(), &*id_key)
            .map_err(StorageError::DatabaseError)?;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.bundles_stored += 1;
            stats.total_size += bundle_data.len() as u64;
        }

        // Flush to ensure durability
        self.db.flush_async().await.map_err(StorageError::DatabaseError)?;

        Ok(())
    }

    /// Retrieve a bundle by ID
    pub async fn get(&self, bundle_id: &BundleId) -> Result<Option<Bundle>, StorageError> {
        let id_key = self.bundle_id_key(bundle_id);

        if let Some(bundle_data) = self.bundles.get(&id_key).map_err(StorageError::DatabaseError)? {
            let bundle = Bundle::decode(Bytes::from(bundle_data.to_vec()))
                .map_err(StorageError::BundleError)?;

            let mut stats = self.stats.write().await;
            stats.bundles_retrieved += 1;

            Ok(Some(bundle))
        } else {
            Ok(None)
        }
    }

    /// Get bundle metadata without loading the full bundle
    pub async fn get_metadata(&self, bundle_id: &BundleId) -> Result<Option<BundleMetadata>, StorageError> {
        let id_key = self.bundle_id_key(bundle_id);

        if let Some(metadata_data) = self.metadata.get(&id_key).map_err(StorageError::DatabaseError)? {
            let metadata: BundleMetadata = bincode::deserialize(&metadata_data)
                .map_err(|e| StorageError::SerializationError(e.to_string()))?;
            Ok(Some(metadata))
        } else {
            Ok(None)
        }
    }

    /// Remove a bundle from storage
    pub async fn remove(&self, bundle_id: &BundleId) -> Result<bool, StorageError> {
        let id_key = self.bundle_id_key(bundle_id);

        // Get metadata for cleanup
        let metadata = if let Some(metadata_data) = self.metadata.get(&id_key).map_err(StorageError::DatabaseError)? {
            let metadata: BundleMetadata = bincode::deserialize(&metadata_data)
                .map_err(|e| StorageError::SerializationError(e.to_string()))?;
            Some(metadata)
        } else {
            None
        };

        // Remove from main storage
        let removed = self.bundles.remove(&id_key).map_err(StorageError::DatabaseError)?.is_some();

        if removed {
            // Remove metadata
            self.metadata.remove(&id_key).map_err(StorageError::DatabaseError)?;

            if let Some(meta) = metadata {
                // Remove from indexes
                let dest_key = format!("{}#{}", meta.destination, bundle_id);
                self.by_destination.remove(dest_key.as_bytes()).map_err(StorageError::DatabaseError)?;

                let src_key = format!("{}#{}", meta.source, bundle_id);
                self.by_source.remove(src_key.as_bytes()).map_err(StorageError::DatabaseError)?;

                // Update stats
                let mut stats = self.stats.write().await;
                stats.bundles_removed += 1;
                stats.total_size = stats.total_size.saturating_sub(meta.size as u64);
            }
        }

        Ok(removed)
    }

    /// List bundles for a destination
    pub async fn list_for_destination(&self, destination: &EndpointId) -> Result<Vec<BundleId>, StorageError> {
        let prefix = format!("{}#", destination);
        let mut bundle_ids = Vec::new();

        for item in self.by_destination.scan_prefix(prefix.as_bytes()) {
            let (_key, id_bytes) = item.map_err(StorageError::DatabaseError)?;
            let id_str = String::from_utf8_lossy(&id_bytes);
            if let Ok(bundle_id) = self.parse_bundle_id(&id_str) {
                bundle_ids.push(bundle_id);
            }
        }

        Ok(bundle_ids)
    }

    /// List all bundle IDs in storage
    pub async fn list_all(&self) -> Result<Vec<BundleId>, StorageError> {
        let mut bundle_ids = Vec::new();

        for item in self.metadata.iter() {
            let (id_bytes, _metadata_bytes) = item.map_err(StorageError::DatabaseError)?;
            let id_str = String::from_utf8_lossy(&id_bytes);
            if let Ok(bundle_id) = self.parse_bundle_id(&id_str) {
                bundle_ids.push(bundle_id);
            }
        }

        Ok(bundle_ids)
    }

    /// Remove expired bundles
    pub async fn cleanup_expired(&self) -> Result<u64, StorageError> {
        let mut expired_count = 0;
        let mut to_remove = Vec::new();

        // Find expired bundles
        for item in self.metadata.iter() {
            let (_key, metadata_bytes) = item.map_err(StorageError::DatabaseError)?;
            let metadata: BundleMetadata = bincode::deserialize(&metadata_bytes)
                .map_err(|e| StorageError::SerializationError(e.to_string()))?;

            if metadata.is_expired() {
                to_remove.push(metadata.id);
            }
        }

        // Remove expired bundles
        for bundle_id in to_remove {
            if self.remove(&bundle_id).await? {
                expired_count += 1;
            }
        }

        Ok(expired_count)
    }

    /// Get storage statistics
    pub async fn stats(&self) -> StorageStats {
        self.stats.read().await.clone()
    }

    /// Get approximate size of database
    pub async fn size_on_disk(&self) -> Result<u64, StorageError> {
        Ok(self.db.size_on_disk().map_err(StorageError::DatabaseError)?)
    }

    /// Flush all pending writes
    pub async fn flush(&self) -> Result<(), StorageError> {
        self.db.flush_async().await.map_err(StorageError::DatabaseError)?;
        Ok(())
    }

    // Helper methods

    fn bundle_id_key(&self, bundle_id: &BundleId) -> Vec<u8> {
        format!("{}@{}.{}",
               bundle_id.source,
               bundle_id.timestamp.dtn_time,
               bundle_id.timestamp.sequence_number).into_bytes()
    }

    fn parse_bundle_id(&self, key: &str) -> Result<BundleId, StorageError> {
        // Parse "source@time.sequence" format
        if let Some((source_str, timestamp_str)) = key.split_once('@') {
            if let Some((time_str, seq_str)) = timestamp_str.split_once('.') {
                let source: EndpointId = source_str.parse()
                    .map_err(|e| StorageError::ParseError(format!("Invalid source: {}", e)))?;
                let dtn_time: u64 = time_str.parse()
                    .map_err(|e| StorageError::ParseError(format!("Invalid time: {}", e)))?;
                let sequence_number: u64 = seq_str.parse()
                    .map_err(|e| StorageError::ParseError(format!("Invalid sequence: {}", e)))?;

                return Ok(BundleId {
                    source,
                    timestamp: crate::CreationTimestamp { dtn_time, sequence_number },
                });
            }
        }

        Err(StorageError::ParseError(format!("Invalid bundle ID format: {}", key)))
    }
}

/// Storage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StorageStats {
    pub bundles_stored: u64,
    pub bundles_retrieved: u64,
    pub bundles_removed: u64,
    pub total_size: u64,
}

/// Storage-related errors
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] sled::Error),

    #[error("Bundle error: {0}")]
    BundleError(#[from] crate::bundle::BundleError),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Bundle already exists: {0}")]
    BundleExists(BundleId),

    #[error("Bundle not found: {0}")]
    BundleNotFound(BundleId),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Storage is full")]
    StorageFull,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::{Bundle, EndpointId};

    #[tokio::test]
    async fn test_store_and_retrieve() {
        let store = BundleStore::memory().await.unwrap();

        let src = EndpointId::node("node1");
        let dst = EndpointId::node("node2");
        let payload = Bytes::from("Hello, DTN!");

        let bundle = Bundle::new(dst, src, payload.clone(), 60000);
        let bundle_id = bundle.id();

        // Store bundle
        store.store(bundle.clone()).await.unwrap();

        // Retrieve bundle
        let retrieved = store.get(&bundle_id).await.unwrap().unwrap();
        assert_eq!(retrieved.payload.data, crate::bundle::SerializableBytes::from(payload));

        // Check metadata
        let metadata = store.get_metadata(&bundle_id).await.unwrap().unwrap();
        assert_eq!(metadata.id, bundle_id);
        assert_eq!(metadata.destination, bundle.primary.destination);
    }

    #[tokio::test]
    async fn test_list_by_destination() {
        let store = BundleStore::memory().await.unwrap();

        let src = EndpointId::node("node1");
        let dst = EndpointId::node("node2");

        // Store multiple bundles for same destination
        for i in 0..3 {
            let payload = Bytes::from(format!("Message {}", i));
            let bundle = Bundle::new(dst.clone(), src.clone(), payload, 60000);
            store.store(bundle).await.unwrap();
        }

        let bundles = store.list_for_destination(&dst).await.unwrap();
        assert_eq!(bundles.len(), 3);
    }

    #[tokio::test]
    async fn test_remove_bundle() {
        let store = BundleStore::memory().await.unwrap();

        let src = EndpointId::node("node1");
        let dst = EndpointId::node("node2");
        let payload = Bytes::from("Hello, DTN!");

        let bundle = Bundle::new(dst, src, payload, 60000);
        let bundle_id = bundle.id();

        // Store and then remove
        store.store(bundle).await.unwrap();
        assert!(store.remove(&bundle_id).await.unwrap());

        // Should not be found
        assert!(store.get(&bundle_id).await.unwrap().is_none());
        assert!(!store.remove(&bundle_id).await.unwrap());
    }

    #[tokio::test]
    async fn test_duplicate_store() {
        let store = BundleStore::memory().await.unwrap();

        let src = EndpointId::node("node1");
        let dst = EndpointId::node("node2");
        let payload = Bytes::from("Hello, DTN!");

        let bundle = Bundle::new(dst, src, payload, 60000);

        // Store once
        store.store(bundle.clone()).await.unwrap();

        // Store again should fail
        let result = store.store(bundle).await;
        assert!(matches!(result, Err(StorageError::BundleExists(_))));
    }

    #[tokio::test]
    async fn test_stats() {
        let store = BundleStore::memory().await.unwrap();

        let src = EndpointId::node("node1");
        let dst = EndpointId::node("node2");
        let payload = Bytes::from("Hello, DTN!");

        let bundle = Bundle::new(dst, src, payload, 60000);
        let bundle_id = bundle.id();

        store.store(bundle).await.unwrap();

        let stats = store.stats().await;
        assert_eq!(stats.bundles_stored, 1);
        assert!(stats.total_size > 0);

        store.get(&bundle_id).await.unwrap();
        let stats = store.stats().await;
        assert_eq!(stats.bundles_retrieved, 1);
    }
}
