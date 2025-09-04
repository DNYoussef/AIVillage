//! Federated Learning Receipts and Proof of Participation
//!
//! Implements verifiable proof of participation with examples, FLOPs, and
//! energy tracking. Receipts are signed using Ed25519 keys and persisted in a
//! `sled` key-value store for later verification.

use crate::{FederatedError, ParticipantId, Result};
use ed25519_dalek::{Keypair, Signature, Signer, Verifier};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use sled::Db;

/// Federated learning receipt for proof of participation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FLReceipt {
    /// Participant identifier
    pub participant_id: ParticipantId,
    /// Number of training examples processed
    pub num_examples: u32,
    /// FLOPs performed during training
    pub flops: u64,
    /// Energy consumed in joules
    pub energy_joules: f32,
    /// Timestamp of training completion
    pub timestamp: u64,
    /// Cryptographic signature
    pub signature: Vec<u8>,
}

impl FLReceipt {
    /// Create a new unsigned receipt
    pub fn new(
        participant_id: ParticipantId,
        num_examples: u32,
        flops: u64,
        energy_joules: f32,
    ) -> Self {
        Self {
            participant_id,
            num_examples,
            flops,
            energy_joules,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            signature: vec![],
        }
    }

    /// Return the bytes used for signing/verification (receipt without
    /// signature)
    fn signing_bytes(&self) -> Result<Vec<u8>> {
        let mut clone = self.clone();
        clone.signature.clear();
        bincode::serialize(&clone)
            .map_err(|e| FederatedError::SerializationError(e.to_string()))
    }
}

/// Proof of participation manager that signs and stores receipts
pub struct ProofOfParticipation {
    db: Db,
    keypair: Keypair,
}

impl ProofOfParticipation {
    /// Create a new proof manager with an ephemeral sled database
    pub fn new() -> Self {
        let db = sled::Config::new()
            .temporary(true)
            .open()
            .expect("failed to open sled db");
        let keypair = Keypair::generate(&mut OsRng);
        Self { db, keypair }
    }

    /// Generate, sign, and persist a receipt for the given metrics
    pub fn generate_receipt(
        &self,
        participant_id: ParticipantId,
        metrics: &ResourceMetrics,
    ) -> Result<FLReceipt> {
        let mut receipt = FLReceipt::new(
            participant_id,
            metrics.num_examples,
            metrics.flops,
            metrics.energy_joules,
        );
        let data = receipt.signing_bytes()?;
        let sig = self.keypair.sign(&data);
        receipt.signature = sig.to_bytes().to_vec();
        self.store_receipt(&receipt)?;
        Ok(receipt)
    }

    /// Verify that a receipt was signed by this manager
    pub fn verify_receipt(&self, receipt: &FLReceipt) -> Result<bool> {
        let data = receipt.signing_bytes()?;
        let sig = Signature::from_bytes(&receipt.signature)
            .map_err(|e| FederatedError::TrainingError(e.to_string()))?;
        Ok(self.keypair.public.verify(&data, &sig).is_ok())
    }

    /// Retrieve a previously stored receipt
    pub fn get_receipt(
        &self,
        participant: &ParticipantId,
        timestamp: u64,
    ) -> Result<Option<FLReceipt>> {
        let key = Self::key(participant, timestamp);
        match self
            .db
            .get(key)
            .map_err(|e| FederatedError::NetworkError(e.to_string()))?
        {
            Some(bytes) => {
                let receipt: FLReceipt = bincode::deserialize(&bytes)
                    .map_err(|e| FederatedError::SerializationError(e.to_string()))?;
                Ok(Some(receipt))
            }
            None => Ok(None),
        }
    }

    fn store_receipt(&self, receipt: &FLReceipt) -> Result<()> {
        let key = Self::key(&receipt.participant_id, receipt.timestamp);
        let bytes = bincode::serialize(receipt)
            .map_err(|e| FederatedError::SerializationError(e.to_string()))?;
        self.db
            .insert(key, bytes)
            .map_err(|e| FederatedError::NetworkError(e.to_string()))?;
        Ok(())
    }

    fn key(participant: &ParticipantId, timestamp: u64) -> Vec<u8> {
        format!("{}:{}", participant.agent_id, timestamp).into_bytes()
    }
}

/// Resource metrics for receipt generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub num_examples: u32,
    pub flops: u64,
    pub energy_joules: f32,
    pub peak_memory_mb: f32,
}

impl ResourceMetrics {
    pub fn new() -> Self {
        Self {
            num_examples: 0,
            flops: 0,
            energy_joules: 0.0,
            peak_memory_mb: 0.0,
        }
    }
}

