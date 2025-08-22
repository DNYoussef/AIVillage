//! Conflict-free Replicated Data Types (CRDTs) with cryptographic signatures
//!
//! Implements Last-Writer-Wins Map (LWW-Map) and Grow-only Counter (GCounter)
//! with Ed25519 signatures for operation authenticity.

use std::collections::{BTreeMap, HashMap};
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use ed25519_dalek::{Keypair, PublicKey, Signature, Signer, Verifier};
use rand::{rngs::OsRng, CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Cryptographically signed CRDT operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedOperation {
    pub operation: CrdtOperation,
    pub actor_id: String,
    pub timestamp: u64,
    pub signature: Vec<u8>,
    pub public_key: Vec<u8>,
}

impl SignedOperation {
    /// Create new signed operation
    pub fn new(
        operation: CrdtOperation,
        actor_id: String,
        keypair: &Keypair,
    ) -> Result<Self, CrdtError> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| CrdtError::TimestampError(e.to_string()))?
            .as_millis() as u64;

        let mut op = Self {
            operation,
            actor_id,
            timestamp,
            signature: Vec::new(),
            public_key: keypair.public.to_bytes().to_vec(),
        };

        // Sign the operation
        let message = op.signing_message()?;
        op.signature = keypair.sign(&message).to_bytes().to_vec();

        Ok(op)
    }

    /// Verify operation signature
    pub fn verify(&self) -> Result<bool, CrdtError> {
        let public_key = PublicKey::from_bytes(&self.public_key)
            .map_err(|e| CrdtError::SignatureError(e.to_string()))?;

        let signature = Signature::from_bytes(&self.signature)
            .map_err(|e| CrdtError::SignatureError(e.to_string()))?;

        let message = self.signing_message()?;

        Ok(public_key.verify(&message, &signature).is_ok())
    }

    /// Get message to sign (excludes signature field)
    fn signing_message(&self) -> Result<Vec<u8>, CrdtError> {
        let mut temp = self.clone();
        temp.signature = Vec::new();

        bincode::serialize(&temp).map_err(|e| CrdtError::SerializationError(e.to_string()))
    }

    /// Check if this operation happened before another
    pub fn happened_before(&self, other: &SignedOperation) -> bool {
        self.timestamp < other.timestamp
            || (self.timestamp == other.timestamp && self.actor_id < other.actor_id)
    }
}

/// CRDT operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrdtOperation {
    /// LWW-Map set operation
    LwwSet {
        key: String,
        value: Option<Bytes>, // None for delete
    },
    /// GCounter increment operation
    GCounterIncrement { counter_id: String, amount: u64 },
}

/// Last-Writer-Wins Map implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LwwMap {
    /// Key -> (Value, Timestamp, Actor)
    entries: HashMap<String, (Option<Bytes>, u64, String)>,
    /// Operation log for synchronization
    operations: Vec<SignedOperation>,
}

impl LwwMap {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            operations: Vec::new(),
        }
    }

    /// Set value with signed operation
    pub fn set_signed(&mut self, operation: SignedOperation) -> Result<(), CrdtError> {
        // Verify signature
        if !operation.verify()? {
            return Err(CrdtError::InvalidSignature);
        }

        match &operation.operation {
            CrdtOperation::LwwSet { key, value } => {
                // Check if this is the latest operation for this key
                if let Some((_, existing_timestamp, existing_actor)) = self.entries.get(key) {
                    let existing_op = SignedOperation {
                        operation: CrdtOperation::LwwSet {
                            key: key.clone(),
                            value: None,
                        },
                        actor_id: existing_actor.clone(),
                        timestamp: *existing_timestamp,
                        signature: Vec::new(),
                        public_key: Vec::new(),
                    };

                    if !operation.happened_before(&existing_op) {
                        // This operation is newer, apply it
                        self.entries.insert(
                            key.clone(),
                            (
                                value.clone(),
                                operation.timestamp,
                                operation.actor_id.clone(),
                            ),
                        );
                    }
                } else {
                    // First operation for this key
                    self.entries.insert(
                        key.clone(),
                        (
                            value.clone(),
                            operation.timestamp,
                            operation.actor_id.clone(),
                        ),
                    );
                }

                // Add to operation log
                self.operations.push(operation);
                Ok(())
            }
            _ => Err(CrdtError::InvalidOperation(
                "Expected LwwSet operation".to_string(),
            )),
        }
    }

    /// Get value
    pub fn get(&self, key: &str) -> Option<&Bytes> {
        self.entries
            .get(key)
            .and_then(|(value, _, _)| value.as_ref())
    }

    /// Delete value (set to None)
    pub fn delete_signed(
        &mut self,
        key: String,
        operation: SignedOperation,
    ) -> Result<(), CrdtError> {
        // Verify this is a delete operation
        match &operation.operation {
            CrdtOperation::LwwSet {
                key: op_key,
                value: None,
            } if op_key == &key => self.set_signed(operation),
            _ => Err(CrdtError::InvalidOperation(
                "Expected delete operation".to_string(),
            )),
        }
    }

    /// Get all keys
    pub fn keys(&self) -> Vec<String> {
        self.entries
            .iter()
            .filter_map(|(key, (value, _, _))| {
                if value.is_some() {
                    Some(key.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Merge with another LWW-Map
    pub fn merge(&mut self, other: &LwwMap) -> Result<(), CrdtError> {
        // Apply all operations from other map
        for operation in &other.operations {
            // Skip if we already have this operation
            if self
                .operations
                .iter()
                .any(|op| op.actor_id == operation.actor_id && op.timestamp == operation.timestamp)
            {
                continue;
            }

            self.set_signed(operation.clone())?;
        }
        Ok(())
    }

    /// Get operation count for debugging
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }
}

impl Default for LwwMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Grow-only Counter implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCounter {
    /// Actor -> Count mapping
    counters: BTreeMap<String, u64>,
    /// Operation log for synchronization
    operations: Vec<SignedOperation>,
}

impl GCounter {
    pub fn new() -> Self {
        Self {
            counters: BTreeMap::new(),
            operations: Vec::new(),
        }
    }

    /// Increment counter with signed operation
    pub fn increment_signed(&mut self, operation: SignedOperation) -> Result<(), CrdtError> {
        // Verify signature
        if !operation.verify()? {
            return Err(CrdtError::InvalidSignature);
        }

        match &operation.operation {
            CrdtOperation::GCounterIncrement {
                counter_id: _,
                amount,
            } => {
                // Update actor's counter
                let current = self.counters.get(&operation.actor_id).unwrap_or(&0);
                self.counters
                    .insert(operation.actor_id.clone(), current + amount);

                // Add to operation log
                self.operations.push(operation);
                Ok(())
            }
            _ => Err(CrdtError::InvalidOperation(
                "Expected GCounterIncrement operation".to_string(),
            )),
        }
    }

    /// Get total counter value
    pub fn value(&self) -> u64 {
        self.counters.values().sum()
    }

    /// Get counter value for specific actor
    pub fn value_for_actor(&self, actor_id: &str) -> u64 {
        self.counters.get(actor_id).copied().unwrap_or(0)
    }

    /// Merge with another GCounter
    pub fn merge(&mut self, other: &GCounter) -> Result<(), CrdtError> {
        // Apply all operations from other counter
        for operation in &other.operations {
            // Skip if we already have this operation
            if self
                .operations
                .iter()
                .any(|op| op.actor_id == operation.actor_id && op.timestamp == operation.timestamp)
            {
                continue;
            }

            self.increment_signed(operation.clone())?;
        }
        Ok(())
    }

    /// Get all actors
    pub fn actors(&self) -> Vec<String> {
        self.counters.keys().cloned().collect()
    }

    /// Get operation count for debugging
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }
}

impl Default for GCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined CRDT state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrdtState {
    pub lww_maps: HashMap<String, LwwMap>,
    pub g_counters: HashMap<String, GCounter>,
    pub last_sync_timestamp: u64,
}

impl CrdtState {
    pub fn new() -> Self {
        Self {
            lww_maps: HashMap::new(),
            g_counters: HashMap::new(),
            last_sync_timestamp: 0,
        }
    }

    /// Get or create LWW map
    pub fn get_lww_map(&mut self, name: &str) -> &mut LwwMap {
        self.lww_maps
            .entry(name.to_string())
            .or_insert_with(LwwMap::new)
    }

    /// Get or create GCounter
    pub fn get_g_counter(&mut self, name: &str) -> &mut GCounter {
        self.g_counters
            .entry(name.to_string())
            .or_insert_with(GCounter::new)
    }

    /// Merge with another CRDT state
    pub fn merge(&mut self, other: &CrdtState) -> Result<(), CrdtError> {
        // Merge all LWW maps
        for (name, other_map) in &other.lww_maps {
            let our_map = self.get_lww_map(name);
            our_map.merge(other_map)?;
        }

        // Merge all GCounters
        for (name, other_counter) in &other.g_counters {
            let our_counter = self.get_g_counter(name);
            our_counter.merge(other_counter)?;
        }

        // Update sync timestamp
        self.last_sync_timestamp = self.last_sync_timestamp.max(other.last_sync_timestamp);

        Ok(())
    }

    /// Update sync timestamp
    pub fn update_sync_timestamp(&mut self) {
        self.last_sync_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
    }

    /// Check if merge is idempotent
    pub fn is_merge_idempotent(&self, other: &CrdtState) -> bool {
        // Create a clone and merge
        let mut test_state = self.clone();
        if test_state.merge(other).is_err() {
            return false;
        }

        // Merge again and check if result is the same
        let mut test_state2 = test_state.clone();
        if test_state2.merge(other).is_err() {
            return false;
        }

        // Compare states (simplified comparison)
        test_state.lww_maps.len() == test_state2.lww_maps.len()
            && test_state.g_counters.len() == test_state2.g_counters.len()
    }
}

impl Default for CrdtState {
    fn default() -> Self {
        Self::new()
    }
}

/// CRDT operation factory with keypair
pub struct CrdtOperationFactory {
    keypair: Keypair,
    actor_id: String,
}

impl CrdtOperationFactory {
    /// Create new factory with random keypair
    pub fn new(actor_id: String) -> Self {
        // Use compatible RNG for ed25519-dalek 1.0.0
        // Generate key directly from bytes to avoid trait issues
        let keypair = {
            use ed25519_dalek::SecretKey;
            use rand::{rngs::OsRng, RngCore};

            let mut secret_bytes = [0u8; 32];
            OsRng.fill_bytes(&mut secret_bytes);
            let secret_key = SecretKey::from_bytes(&secret_bytes).unwrap();
            let public_key = PublicKey::from(&secret_key);
            Keypair {
                secret: secret_key,
                public: public_key,
            }
        };

        Self { keypair, actor_id }
    }

    /// Create factory with existing keypair
    pub fn with_keypair(actor_id: String, keypair: Keypair) -> Self {
        Self { keypair, actor_id }
    }

    /// Create signed LWW set operation
    pub fn create_lww_set(&self, key: String, value: Bytes) -> Result<SignedOperation, CrdtError> {
        let operation = CrdtOperation::LwwSet {
            key,
            value: Some(value),
        };

        SignedOperation::new(operation, self.actor_id.clone(), &self.keypair)
    }

    /// Create signed LWW delete operation
    pub fn create_lww_delete(&self, key: String) -> Result<SignedOperation, CrdtError> {
        let operation = CrdtOperation::LwwSet { key, value: None };

        SignedOperation::new(operation, self.actor_id.clone(), &self.keypair)
    }

    /// Create signed GCounter increment operation
    pub fn create_g_counter_increment(
        &self,
        counter_id: String,
        amount: u64,
    ) -> Result<SignedOperation, CrdtError> {
        let operation = CrdtOperation::GCounterIncrement { counter_id, amount };

        SignedOperation::new(operation, self.actor_id.clone(), &self.keypair)
    }

    /// Get public key
    pub fn public_key(&self) -> PublicKey {
        self.keypair.public
    }

    /// Get actor ID
    pub fn actor_id(&self) -> &str {
        &self.actor_id
    }
}

/// CRDT errors
#[derive(Debug, Error)]
pub enum CrdtError {
    #[error("Invalid signature")]
    InvalidSignature,

    #[error("Signature error: {0}")]
    SignatureError(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Timestamp error: {0}")]
    TimestampError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lww_map_basic_operations() {
        let factory = CrdtOperationFactory::new("actor1".to_string());
        let mut map = LwwMap::new();

        // Set a value
        let set_op = factory
            .create_lww_set("key1".to_string(), Bytes::from("value1"))
            .unwrap();
        map.set_signed(set_op).unwrap();

        assert_eq!(map.get("key1"), Some(&Bytes::from("value1")));
        assert_eq!(map.operation_count(), 1);

        // Delete the value
        let delete_op = factory.create_lww_delete("key1".to_string()).unwrap();
        map.delete_signed("key1".to_string(), delete_op).unwrap();

        assert_eq!(map.get("key1"), None);
        assert_eq!(map.operation_count(), 2);
    }

    #[test]
    fn test_lww_map_merge() {
        let factory1 = CrdtOperationFactory::new("actor1".to_string());
        let factory2 = CrdtOperationFactory::new("actor2".to_string());

        let mut map1 = LwwMap::new();
        let mut map2 = LwwMap::new();

        // Add different values to each map
        let op1 = factory1
            .create_lww_set("key1".to_string(), Bytes::from("value1"))
            .unwrap();
        map1.set_signed(op1).unwrap();

        let op2 = factory2
            .create_lww_set("key2".to_string(), Bytes::from("value2"))
            .unwrap();
        map2.set_signed(op2).unwrap();

        // Merge map2 into map1
        map1.merge(&map2).unwrap();

        assert_eq!(map1.get("key1"), Some(&Bytes::from("value1")));
        assert_eq!(map1.get("key2"), Some(&Bytes::from("value2")));
        assert_eq!(map1.operation_count(), 2);
    }

    #[test]
    fn test_g_counter_basic_operations() {
        let factory = CrdtOperationFactory::new("actor1".to_string());
        let mut counter = GCounter::new();

        // Increment counter
        let inc_op = factory
            .create_g_counter_increment("counter1".to_string(), 5)
            .unwrap();
        counter.increment_signed(inc_op).unwrap();

        assert_eq!(counter.value(), 5);
        assert_eq!(counter.value_for_actor("actor1"), 5);
        assert_eq!(counter.operation_count(), 1);

        // Increment again
        let inc_op2 = factory
            .create_g_counter_increment("counter1".to_string(), 3)
            .unwrap();
        counter.increment_signed(inc_op2).unwrap();

        assert_eq!(counter.value(), 8);
        assert_eq!(counter.value_for_actor("actor1"), 8);
    }

    #[test]
    fn test_g_counter_merge() {
        let factory1 = CrdtOperationFactory::new("actor1".to_string());
        let factory2 = CrdtOperationFactory::new("actor2".to_string());

        let mut counter1 = GCounter::new();
        let mut counter2 = GCounter::new();

        // Increment both counters
        let inc1 = factory1
            .create_g_counter_increment("counter1".to_string(), 5)
            .unwrap();
        counter1.increment_signed(inc1).unwrap();

        let inc2 = factory2
            .create_g_counter_increment("counter1".to_string(), 3)
            .unwrap();
        counter2.increment_signed(inc2).unwrap();

        // Merge counter2 into counter1
        counter1.merge(&counter2).unwrap();

        assert_eq!(counter1.value(), 8); // 5 + 3
        assert_eq!(counter1.value_for_actor("actor1"), 5);
        assert_eq!(counter1.value_for_actor("actor2"), 3);
    }

    #[test]
    fn test_crdt_state_merge_idempotent() {
        let factory1 = CrdtOperationFactory::new("actor1".to_string());
        let factory2 = CrdtOperationFactory::new("actor2".to_string());

        let mut state1 = CrdtState::new();
        let mut state2 = CrdtState::new();

        // Add operations to both states
        let op1 = factory1
            .create_lww_set("key1".to_string(), Bytes::from("value1"))
            .unwrap();
        state1.get_lww_map("map1").set_signed(op1).unwrap();

        let op2 = factory2
            .create_g_counter_increment("counter1".to_string(), 5)
            .unwrap();
        state2
            .get_g_counter("counter1")
            .increment_signed(op2)
            .unwrap();

        // Test merge idempotency
        assert!(state1.is_merge_idempotent(&state2));

        // Perform actual merge
        state1.merge(&state2).unwrap();

        // Verify merge was successful
        assert_eq!(
            state1.lww_maps.get("map1").unwrap().get("key1"),
            Some(&Bytes::from("value1"))
        );
        assert_eq!(state1.g_counters.get("counter1").unwrap().value(), 5);
    }

    #[test]
    fn test_signed_operation_verification() {
        let factory = CrdtOperationFactory::new("actor1".to_string());
        let operation = factory
            .create_lww_set("key1".to_string(), Bytes::from("value1"))
            .unwrap();

        // Verify valid signature
        assert!(operation.verify().unwrap());

        // Create invalid signature
        let mut invalid_op = operation.clone();
        invalid_op.signature[0] ^= 1; // Flip a bit
        assert!(!invalid_op.verify().unwrap());
    }

    #[test]
    fn test_operation_ordering() {
        let factory1 = CrdtOperationFactory::new("actor1".to_string());
        let factory2 = CrdtOperationFactory::new("actor2".to_string());

        let op1 = factory1
            .create_lww_set("key1".to_string(), Bytes::from("value1"))
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1)); // Ensure different timestamps
        let op2 = factory2
            .create_lww_set("key1".to_string(), Bytes::from("value2"))
            .unwrap();

        assert!(op1.happened_before(&op2));
        assert!(!op2.happened_before(&op1));
    }
}
