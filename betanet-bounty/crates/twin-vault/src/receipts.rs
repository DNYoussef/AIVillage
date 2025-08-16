//! Cryptographic receipts for twin vault operations using COSE signatures
//!
//! Provides verifiable proof of all read and write operations performed
//! on twin state using CBOR Object Signing and Encryption (COSE).

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
#[cfg(feature = "receipts")]
use coset::{
    CoseSign1, CoseSign1Builder, HeaderBuilder, Label, ProtectedHeader,
    CoseKey, CoseKeyBuilder, KeyType, Algorithm, AlgorithmWithUsage,
    iana::{Algorithm as IanaAlgorithm, KeyType as IanaKeyType},
};
use ed25519_dalek::{Keypair, PublicKey, SecretKey, Signature, Signer, Verifier};
use rand::{rngs::OsRng, CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{TwinId, TwinOperation};

/// Receipt for a twin vault operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Receipt {
    /// Unique receipt ID
    pub receipt_id: String,
    /// Twin that performed the operation
    pub twin_id: TwinId,
    /// Operation that was performed
    pub operation: TwinOperation,
    /// Agent that requested the operation
    pub requester: crate::AgentId,
    /// Whether the operation was successful
    pub success: bool,
    /// Result data (for read operations)
    pub result_hash: Option<Vec<u8>>,
    /// Timestamp when receipt was created
    pub timestamp: u64,
    /// COSE signature
    pub cose_signature: Vec<u8>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Receipt {
    /// Create new receipt
    pub fn new(
        twin_id: TwinId,
        operation: TwinOperation,
        requester: crate::AgentId,
        success: bool,
        result_data: Option<&[u8]>,
    ) -> Self {
        let receipt_id = uuid::Uuid::new_v4().to_string();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Hash result data if present
        let result_hash = result_data.map(|data| {
            use sha2::{Sha256, Digest};
            Sha256::digest(data).to_vec()
        });

        Self {
            receipt_id,
            twin_id,
            operation,
            requester,
            success,
            result_hash,
            timestamp,
            cose_signature: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to receipt
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get canonical receipt data for signing (excludes signature)
    pub fn canonical_data(&self) -> Result<Vec<u8>, ReceiptError> {
        let mut receipt_copy = self.clone();
        receipt_copy.cose_signature = Vec::new();

        serde_json::to_vec(&receipt_copy)
            .map_err(|e| ReceiptError::SerializationError(e.to_string()))
    }

    /// Verify receipt signature
    pub fn verify(&self, public_key: &PublicKey) -> Result<bool, ReceiptError> {
        #[cfg(feature = "receipts")]
        {
            let cose_sign1 = CoseSign1::from_slice(&self.cose_signature)
                .map_err(|e| ReceiptError::CoseError(e.to_string()))?;

            let payload = cose_sign1.payload.as_ref()
                .ok_or_else(|| ReceiptError::CoseError("No payload in COSE signature".to_string()))?;

            // Verify the payload matches our canonical data
            let canonical = self.canonical_data()?;
            if payload != &canonical {
                return Ok(false);
            }

            // Get signature from COSE structure
            let signature = Signature::from_bytes(cose_sign1.signature.as_slice())
                .map_err(|e| ReceiptError::SignatureError(e.to_string()))?;

            // Verify signature
            Ok(public_key.verify(&canonical, &signature).is_ok())
        }
        #[cfg(not(feature = "receipts"))]
        {
            // Without COSE support, always return false for verification
            Ok(false)
        }
    }

    /// Get operation type as string
    pub fn operation_type(&self) -> &'static str {
        match self.operation {
            TwinOperation::Read { .. } => "read",
            TwinOperation::Write { .. } => "write",
            TwinOperation::Delete { .. } => "delete",
            TwinOperation::Increment { .. } => "increment",
        }
    }

    /// Check if receipt is expired
    pub fn is_expired(&self, max_age_ms: u64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        (now - self.timestamp) > max_age_ms
    }
}

/// Receipt signer for creating COSE signatures
pub struct ReceiptSigner {
    keypair: Keypair,
    signer_id: String,
}

impl Clone for ReceiptSigner {
    fn clone(&self) -> Self {
        // Clone by recreating from secret key bytes
        let secret_bytes = self.keypair.secret.to_bytes();
        let secret_key = SecretKey::from_bytes(&secret_bytes).unwrap();
        let public_key = PublicKey::from(&secret_key);
        let keypair = Keypair { secret: secret_key, public: public_key };

        Self {
            keypair,
            signer_id: self.signer_id.clone(),
        }
    }
}

impl ReceiptSigner {
    /// Create new receipt signer with random keypair
    pub fn new(signer_id: String) -> Self {
        // Use compatible RNG for ed25519-dalek 1.0.0
        // Generate key directly from bytes to avoid trait issues
        let keypair = {
            use rand::{rngs::OsRng, RngCore};

            let mut secret_bytes = [0u8; 32];
            OsRng.fill_bytes(&mut secret_bytes);
            let secret_key = SecretKey::from_bytes(&secret_bytes).unwrap();
            let public_key = PublicKey::from(&secret_key);
            Keypair { secret: secret_key, public: public_key }
        };

        Self {
            keypair,
            signer_id,
        }
    }

    /// Create receipt signer with existing keypair
    pub fn with_keypair(signer_id: String, keypair: Keypair) -> Self {
        Self {
            keypair,
            signer_id,
        }
    }

    /// Create receipt signer from secret key bytes
    pub fn from_secret_bytes(signer_id: String, secret_bytes: &[u8; 32]) -> Result<Self, ReceiptError> {
        let secret_key = SecretKey::from_bytes(secret_bytes)
            .map_err(|e| ReceiptError::KeyError(e.to_string()))?;
        let public_key = PublicKey::from(&secret_key);
        let keypair = Keypair { secret: secret_key, public: public_key };

        Ok(Self {
            keypair,
            signer_id,
        })
    }

    /// Sign a receipt operation
    pub async fn sign_operation(
        &self,
        twin_id: &TwinId,
        operation: &TwinOperation,
        requester: &crate::AgentId,
        success: bool,
    ) -> Result<Receipt, ReceiptError> {
        self.sign_operation_with_result(twin_id, operation, requester, success, None).await
    }

    /// Sign a receipt operation with result data
    pub async fn sign_operation_with_result(
        &self,
        twin_id: &TwinId,
        operation: &TwinOperation,
        requester: &crate::AgentId,
        success: bool,
        result_data: Option<&[u8]>,
    ) -> Result<Receipt, ReceiptError> {
        let mut receipt = Receipt::new(
            twin_id.clone(),
            operation.clone(),
            requester.clone(),
            success,
            result_data,
        );

        // Add signer metadata
        receipt = receipt.with_metadata("signer_id".to_string(), self.signer_id.clone());
        receipt = receipt.with_metadata("signature_algorithm".to_string(), "Ed25519".to_string());

        // Create COSE signature
        let canonical_data = receipt.canonical_data()?;
        let signature = self.keypair.sign(&canonical_data);

        #[cfg(feature = "receipts")]
        {
            // Build COSE_Sign1 structure
            let protected_header = ProtectedHeader {
                header: HeaderBuilder::new()
                    .algorithm(IanaAlgorithm::EdDSA)
                    .key_id(self.signer_id.as_bytes().to_vec())
                    .build(),
            };

            let cose_sign1 = CoseSign1Builder::new()
                .protected(protected_header)
                .payload(canonical_data)
                .signature(signature.to_bytes().to_vec())
                .build();

            receipt.cose_signature = cose_sign1.to_vec()
                .map_err(|e| ReceiptError::CoseError(e.to_string()))?;
        }
        #[cfg(not(feature = "receipts"))]
        {
            // Without COSE support, store raw signature
            receipt.cose_signature = signature.to_bytes().to_vec();
        }

        Ok(receipt)
    }

    /// Get public key
    pub fn public_key(&self) -> PublicKey {
        self.keypair.public
    }

    /// Get signer ID
    pub fn signer_id(&self) -> &str {
        &self.signer_id
    }

    /// Export secret key bytes (for secure storage)
    pub fn export_secret_key(&self) -> [u8; 32] {
        self.keypair.secret.to_bytes()
    }

    /// Create COSE key for this signer
    #[cfg(feature = "receipts")]
    pub fn to_cose_key(&self) -> Result<CoseKey, ReceiptError> {
        #[cfg(feature = "receipts")]
        {
            let cose_key = CoseKeyBuilder::new_ec2_pub_key(
                IanaKeyType::OKP,
                // Curve Ed25519
                Label::Int(6),
                self.keypair.public.to_bytes().to_vec(),
            )
            .algorithm(IanaAlgorithm::EdDSA)
            .key_id(self.signer_id.as_bytes().to_vec())
            .build();

            Ok(cose_key)
        }
        #[cfg(not(feature = "receipts"))]
        {
            Err(ReceiptError::CoseError("COSE feature disabled".to_string()))
        }
    }
}

/// Receipt verifier for validating COSE signatures
#[derive(Clone)]
pub struct ReceiptVerifier {
    trusted_keys: HashMap<String, PublicKey>,
}

impl ReceiptVerifier {
    /// Create new receipt verifier
    pub fn new() -> Self {
        Self {
            trusted_keys: HashMap::new(),
        }
    }

    /// Add trusted public key
    pub fn add_trusted_key(&mut self, signer_id: String, public_key: PublicKey) {
        self.trusted_keys.insert(signer_id, public_key);
    }

    /// Remove trusted key
    pub fn remove_trusted_key(&mut self, signer_id: &str) {
        self.trusted_keys.remove(signer_id);
    }

    /// Verify receipt signature
    pub fn verify_receipt(&self, receipt: &Receipt) -> Result<ReceiptVerificationResult, ReceiptError> {
        #[cfg(feature = "receipts")]
        {
            // Parse COSE signature
            let cose_sign1 = CoseSign1::from_slice(&receipt.cose_signature)
                .map_err(|e| ReceiptError::CoseError(e.to_string()))?;

            // Extract signer ID from protected header
            let signer_id = if let Some(kid) = &cose_sign1.protected.header.key_id {
                String::from_utf8(kid.clone())
                    .map_err(|e| ReceiptError::CoseError(format!("Invalid key ID: {}", e)))?
            } else {
                return Ok(ReceiptVerificationResult {
                    valid: false,
                    signer_id: None,
                    error: Some("No key ID in COSE signature".to_string()),
                    trusted: false,
                });
            };

            // Check if we have the public key
            let public_key = match self.trusted_keys.get(&signer_id) {
                Some(key) => key,
                None => {
                    return Ok(ReceiptVerificationResult {
                        valid: false,
                        signer_id: Some(signer_id),
                        error: Some("Signer not in trusted keys".to_string()),
                        trusted: false,
                    });
                }
            };

            // Verify signature
            match receipt.verify(public_key) {
                Ok(valid) => Ok(ReceiptVerificationResult {
                    valid,
                    signer_id: Some(signer_id),
                    error: None,
                    trusted: true,
                }),
                Err(e) => Ok(ReceiptVerificationResult {
                    valid: false,
                    signer_id: Some(signer_id),
                    error: Some(e.to_string()),
                    trusted: true,
                }),
            }
        }
        #[cfg(not(feature = "receipts"))]
        {
            // Without COSE support, return basic validation
            Ok(ReceiptVerificationResult {
                valid: false,
                signer_id: Some("unknown".to_string()),
                error: Some("COSE receipts feature disabled".to_string()),
                trusted: false,
            })
        }
    }

    /// Verify receipt and check expiration
    pub fn verify_receipt_with_expiry(
        &self,
        receipt: &Receipt,
        max_age_ms: u64,
    ) -> Result<ReceiptVerificationResult, ReceiptError> {
        if receipt.is_expired(max_age_ms) {
            return Ok(ReceiptVerificationResult {
                valid: false,
                signer_id: None,
                error: Some("Receipt is expired".to_string()),
                trusted: false,
            });
        }

        self.verify_receipt(receipt)
    }

    /// Get list of trusted signer IDs
    pub fn trusted_signers(&self) -> Vec<String> {
        self.trusted_keys.keys().cloned().collect()
    }

    /// Check if signer is trusted
    pub fn is_trusted_signer(&self, signer_id: &str) -> bool {
        self.trusted_keys.contains_key(signer_id)
    }
}

impl Default for ReceiptVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of receipt verification
#[derive(Debug, Clone)]
pub struct ReceiptVerificationResult {
    pub valid: bool,
    pub signer_id: Option<String>,
    pub error: Option<String>,
    pub trusted: bool,
}

impl ReceiptVerificationResult {
    /// Check if receipt is valid and from a trusted signer
    pub fn is_trusted_and_valid(&self) -> bool {
        self.valid && self.trusted
    }
}

/// Receipt store for persisting and querying receipts
pub struct ReceiptStore {
    receipts: HashMap<String, Receipt>,
    receipts_by_twin: HashMap<TwinId, Vec<String>>,
    receipts_by_operation: HashMap<String, Vec<String>>,
}

impl ReceiptStore {
    pub fn new() -> Self {
        Self {
            receipts: HashMap::new(),
            receipts_by_twin: HashMap::new(),
            receipts_by_operation: HashMap::new(),
        }
    }

    /// Store a receipt
    pub fn store_receipt(&mut self, receipt: Receipt) {
        let receipt_id = receipt.receipt_id.clone();
        let twin_id = receipt.twin_id.clone();
        let operation_type = receipt.operation_type().to_string();

        // Store receipt
        self.receipts.insert(receipt_id.clone(), receipt);

        // Index by twin
        self.receipts_by_twin
            .entry(twin_id)
            .or_insert_with(Vec::new)
            .push(receipt_id.clone());

        // Index by operation type
        self.receipts_by_operation
            .entry(operation_type)
            .or_insert_with(Vec::new)
            .push(receipt_id);
    }

    /// Get receipt by ID
    pub fn get_receipt(&self, receipt_id: &str) -> Option<&Receipt> {
        self.receipts.get(receipt_id)
    }

    /// Get all receipts for a twin
    pub fn get_receipts_for_twin(&self, twin_id: &TwinId) -> Vec<&Receipt> {
        self.receipts_by_twin
            .get(twin_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.receipts.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get receipts by operation type
    pub fn get_receipts_by_operation(&self, operation_type: &str) -> Vec<&Receipt> {
        self.receipts_by_operation
            .get(operation_type)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.receipts.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get receipts in time range
    pub fn get_receipts_in_range(&self, start_time: u64, end_time: u64) -> Vec<&Receipt> {
        self.receipts
            .values()
            .filter(|receipt| {
                receipt.timestamp >= start_time && receipt.timestamp <= end_time
            })
            .collect()
    }

    /// Count receipts
    pub fn count(&self) -> usize {
        self.receipts.len()
    }

    /// Clear expired receipts
    pub fn clear_expired(&mut self, max_age_ms: u64) -> usize {
        let expired_ids: Vec<String> = self.receipts
            .values()
            .filter(|receipt| receipt.is_expired(max_age_ms))
            .map(|receipt| receipt.receipt_id.clone())
            .collect();

        let count = expired_ids.len();
        for id in expired_ids {
            self.remove_receipt(&id);
        }
        count
    }

    fn remove_receipt(&mut self, receipt_id: &str) {
        if let Some(receipt) = self.receipts.remove(receipt_id) {
            // Remove from twin index
            if let Some(twin_receipts) = self.receipts_by_twin.get_mut(&receipt.twin_id) {
                twin_receipts.retain(|id| id != receipt_id);
                if twin_receipts.is_empty() {
                    self.receipts_by_twin.remove(&receipt.twin_id);
                }
            }

            // Remove from operation index
            let operation_type = receipt.operation_type().to_string();
            if let Some(op_receipts) = self.receipts_by_operation.get_mut(&operation_type) {
                op_receipts.retain(|id| id != receipt_id);
                if op_receipts.is_empty() {
                    self.receipts_by_operation.remove(&operation_type);
                }
            }
        }
    }
}

impl Default for ReceiptStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Receipt-related errors
#[derive(Debug, Error)]
pub enum ReceiptError {
    #[error("COSE error: {0}")]
    CoseError(String),

    #[error("Signature error: {0}")]
    SignatureError(String),

    #[error("Key error: {0}")]
    KeyError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Invalid receipt format")]
    InvalidFormat,

    #[error("Receipt expired")]
    Expired,

    #[error("Untrusted signer: {0}")]
    UntrustedSigner(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_operation() -> TwinOperation {
        TwinOperation::Write {
            key: "test_key".to_string(),
            value: Bytes::from("test_value"),
            timestamp: 12345,
        }
    }

    fn create_test_twin_id() -> TwinId {
        TwinId::new(
            crate::AgentId::new("test-agent", "test-node"),
            "test-twin",
        )
    }

    #[tokio::test]
    async fn test_receipt_creation_and_verification() {
        let signer = ReceiptSigner::new("test-signer".to_string());
        let twin_id = create_test_twin_id();
        let operation = create_test_operation();
        let requester = crate::AgentId::new("requester", "requester-node");

        // Create receipt
        let receipt = signer.sign_operation(&twin_id, &operation, &requester, true).await.unwrap();

        // Verify receipt
        let is_valid = receipt.verify(&signer.public_key()).unwrap();
        assert!(is_valid);

        // Check receipt properties
        assert_eq!(receipt.twin_id, twin_id);
        assert_eq!(receipt.requester, requester);
        assert!(receipt.success);
        assert_eq!(receipt.operation_type(), "write");
    }

    #[tokio::test]
    async fn test_receipt_verifier() {
        let signer = ReceiptSigner::new("test-signer".to_string());
        let mut verifier = ReceiptVerifier::new();

        // Add trusted key
        verifier.add_trusted_key("test-signer".to_string(), signer.public_key());

        let twin_id = create_test_twin_id();
        let operation = create_test_operation();
        let requester = crate::AgentId::new("requester", "requester-node");

        // Create receipt
        let receipt = signer.sign_operation(&twin_id, &operation, &requester, true).await.unwrap();

        // Verify receipt
        let result = verifier.verify_receipt(&receipt).unwrap();
        assert!(result.is_trusted_and_valid());
        assert_eq!(result.signer_id, Some("test-signer".to_string()));
    }

    #[tokio::test]
    async fn test_untrusted_signer() {
        let signer = ReceiptSigner::new("untrusted-signer".to_string());
        let verifier = ReceiptVerifier::new(); // No trusted keys

        let twin_id = create_test_twin_id();
        let operation = create_test_operation();
        let requester = crate::AgentId::new("requester", "requester-node");

        // Create receipt
        let receipt = signer.sign_operation(&twin_id, &operation, &requester, true).await.unwrap();

        // Verify receipt
        let result = verifier.verify_receipt(&receipt).unwrap();
        assert!(!result.is_trusted_and_valid());
        assert!(!result.trusted);
    }

    #[tokio::test]
    async fn test_receipt_with_result_data() {
        let signer = ReceiptSigner::new("test-signer".to_string());
        let twin_id = create_test_twin_id();
        let operation = TwinOperation::Read {
            key: "test_key".to_string(),
            timestamp: 12345,
        };
        let requester = crate::AgentId::new("requester", "requester-node");
        let result_data = b"read_result_data";

        // Create receipt with result data
        let receipt = signer.sign_operation_with_result(
            &twin_id,
            &operation,
            &requester,
            true,
            Some(result_data),
        ).await.unwrap();

        // Verify receipt
        let is_valid = receipt.verify(&signer.public_key()).unwrap();
        assert!(is_valid);
        assert!(receipt.result_hash.is_some());
        assert_eq!(receipt.operation_type(), "read");
    }

    #[tokio::test]
    async fn test_receipt_store() {
        let signer = ReceiptSigner::new("test-signer".to_string());
        let mut store = ReceiptStore::new();

        let twin_id = create_test_twin_id();
        let operation = create_test_operation();
        let requester = crate::AgentId::new("requester", "requester-node");

        // Create and store receipt
        let receipt = signer.sign_operation(&twin_id, &operation, &requester, true).await.unwrap();
        let receipt_id = receipt.receipt_id.clone();
        store.store_receipt(receipt);

        // Retrieve receipt
        let stored_receipt = store.get_receipt(&receipt_id).unwrap();
        assert_eq!(stored_receipt.receipt_id, receipt_id);

        // Get receipts for twin
        let twin_receipts = store.get_receipts_for_twin(&twin_id);
        assert_eq!(twin_receipts.len(), 1);

        // Get receipts by operation
        let write_receipts = store.get_receipts_by_operation("write");
        assert_eq!(write_receipts.len(), 1);

        assert_eq!(store.count(), 1);
    }

    #[test]
    fn test_receipt_expiration() {
        let twin_id = create_test_twin_id();
        let operation = create_test_operation();
        let requester = crate::AgentId::new("requester", "requester-node");

        let receipt = Receipt::new(twin_id, operation, requester, true, None);

        // Receipt should not be expired immediately
        assert!(!receipt.is_expired(60000)); // 1 minute

        // Receipt should be expired with 0 max age
        assert!(receipt.is_expired(0));
    }

    #[test]
    fn test_signer_key_export_import() {
        let signer1 = ReceiptSigner::new("test-signer".to_string());
        let secret_bytes = signer1.export_secret_key();

        // Create new signer from exported key
        let signer2 = ReceiptSigner::from_secret_bytes("test-signer".to_string(), &secret_bytes).unwrap();

        // Both signers should have the same public key
        assert_eq!(signer1.public_key().to_bytes(), signer2.public_key().to_bytes());
    }

    #[test]
    #[cfg(feature = "receipts")]
    fn test_cose_key_creation() {
        let signer = ReceiptSigner::new("test-signer".to_string());
        let cose_key = signer.to_cose_key().unwrap();

        // Verify COSE key properties
        assert_eq!(cose_key.kty, KeyType::Assigned(IanaKeyType::OKP));
        assert_eq!(cose_key.alg, Some(coset::RegisteredLabelWithPrivate::Assigned(IanaAlgorithm::EdDSA)));
    }
}
