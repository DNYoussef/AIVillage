//! Federated Learning Framework for Betanet
//!
//! Provides robust federated learning across network outages with privacy preservation.
//! Built on top of agent-fabric, twin-vault, and bitchat-cla for secure communication.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//! │    Phones       │    │     Beacon      │    │   Orchestrator  │
//! │  (Edge compute) │    │  (Aggregation)  │    │  (Coordination) │
//! └─────────────────┘    └─────────────────┘    └─────────────────┘
//!          │                       │                       │
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                BitChat P2P (BLE Mesh)                           │
//! └─────────────────────────────────────────────────────────────────┘
//!          │                       │                       │
//! ┌─────────────────────────────────────────────────────────────────┐
//! │         Agent Fabric (RPC + DTN Fallback)                      │
//! └─────────────────────────────────────────────────────────────────┘
//!          │                       │                       │
//! ┌─────────────────────────────────────────────────────────────────┐
//! │               Twin Vault (CRDT State + Receipts)               │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Components
//!
//! - **Orchestrator**: Round coordination via MLS groups and cohort management
//! - **FedAvg+SecAgg**: Secure aggregation with additive masks and DP-SGD
//! - **Gossip**: Peer discovery over BitChat with robust aggregation (trimmed mean/Krum)
//! - **Split Learning**: Early layers on device, later layers on beacon
//! - **Receipts**: Proof of participation with examples, FLOPs, and energy tracking

#![deny(clippy::all)]
#![allow(missing_docs)]

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

// Re-export from dependencies
pub use agent_fabric::{
    AgentFabric, AgentId, AgentMessage, AgentResponse, GroupConfig, GroupMessage,
};
pub use twin_vault::{Receipt, TwinId, TwinManager, TwinOperation};

// Module declarations
pub mod fedavg_secureagg;
pub mod gossip;
pub mod orchestrator;
pub mod receipts;
pub mod split;

// Re-export main types
pub use fedavg_secureagg::{DifferentialPrivacy, FedAvgAggregator, SecureAggregation};
pub use gossip::{GossipProtocol, PeerExchange, RobustAggregation};
pub use orchestrator::{CohortManager, RoundOrchestrator, RoundPlan};
pub use receipts::{FLReceipt, ProofOfParticipation, ResourceMetrics};
pub use split::{BeaconAggregation, DeviceTraining, SplitLearning};

/// Federated Learning round identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RoundId {
    pub session_id: String,
    pub round_number: u64,
    pub epoch: u64,
}

impl RoundId {
    pub fn new(session_id: String, round_number: u64, epoch: u64) -> Self {
        Self {
            session_id,
            round_number,
            epoch,
        }
    }

    pub fn generate(session_id: String) -> Self {
        Self {
            session_id,
            round_number: 0,
            epoch: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    pub fn next_round(&self) -> Self {
        Self {
            session_id: self.session_id.clone(),
            round_number: self.round_number + 1,
            epoch: self.epoch,
        }
    }
}

impl std::fmt::Display for RoundId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:{}",
            self.session_id, self.epoch, self.round_number
        )
    }
}

/// Participant identifier in federated learning
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParticipantId {
    pub agent_id: AgentId,
    pub device_type: DeviceType,
    pub capabilities: DeviceCapabilities,
}

impl ParticipantId {
    pub fn new(
        agent_id: AgentId,
        device_type: DeviceType,
        capabilities: DeviceCapabilities,
    ) -> Self {
        Self {
            agent_id,
            device_type,
            capabilities,
        }
    }
}

impl std::fmt::Display for ParticipantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{:?}", self.agent_id, self.device_type)
    }
}

/// Type of device participating in federated learning
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceType {
    /// Mobile phone with limited compute/battery
    Phone,
    /// Tablet with moderate compute
    Tablet,
    /// Laptop with good compute
    Laptop,
    /// Beacon node for aggregation
    Beacon,
    /// Cloud orchestrator
    Cloud,
}

/// Device computational and communication capabilities
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// Available compute cores
    pub cpu_cores: u32,
    /// Available memory in MB
    pub memory_mb: u32,
    /// Battery level (0.0-1.0, None for wired)
    pub battery_level: Option<f32>,
    /// Estimated FLOPs per second
    pub flops_per_sec: u64,
    /// Network bandwidth in Mbps
    pub bandwidth_mbps: f32,
    /// Supports BLE mesh networking
    pub ble_support: bool,
    /// Supports WiFi Direct
    pub wifi_direct: bool,
    /// Is currently online
    pub is_online: bool,
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            cpu_cores: 4,
            memory_mb: 4096,
            battery_level: Some(0.8),
            flops_per_sec: 1_000_000_000, // 1 GFLOP/s
            bandwidth_mbps: 10.0,
            ble_support: true,
            wifi_direct: true,
            is_online: true,
        }
    }
}

/// Federated learning model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    /// Unique model version
    pub version: String,
    /// Model weights as serialized bytes
    pub weights: Bytes,
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Digital signature for integrity
    pub signature: Option<Bytes>,
}

impl ModelParameters {
    pub fn new(version: String, weights: Bytes, metadata: ModelMetadata) -> Self {
        Self {
            version,
            weights,
            metadata,
            signature: None,
        }
    }

    pub fn size_bytes(&self) -> usize {
        self.weights.len()
    }
}

/// Model metadata for federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model architecture description
    pub architecture: String,
    /// Number of parameters
    pub parameter_count: u64,
    /// Input/output shapes
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    /// Compression type used
    pub compression: CompressionType,
    /// Quantization settings
    pub quantization: QuantizationType,
}

/// Model compression types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompressionType {
    None,
    /// Quantization to 8-bit integers
    Q8,
    /// Top-K sparsification
    TopK {
        k: usize,
    },
    /// Random sparsification with probability
    Random {
        prob: f32,
    },
    /// Gradient compression
    Gradient {
        threshold: f32,
    },
}

/// Quantization types for model parameters
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationType {
    /// No quantization (float32)
    Float32,
    /// 16-bit quantization
    Int16,
    /// 8-bit quantization
    Int8,
    /// 4-bit quantization (aggressive)
    Int4,
}

/// Training configuration for federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Local training epochs
    pub local_epochs: u32,
    /// Batch size for training
    pub batch_size: u32,
    /// Model architecture
    pub model_arch: String,
    /// Dataset configuration
    pub dataset_config: DatasetConfig,
    /// Privacy configuration
    pub privacy_config: PrivacyConfig,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            local_epochs: 5,
            batch_size: 32,
            model_arch: "simple_cnn".to_string(),
            dataset_config: DatasetConfig::default(),
            privacy_config: PrivacyConfig::default(),
            resource_constraints: ResourceConstraints::default(),
        }
    }
}

/// Dataset configuration for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Name of the dataset
    pub name: String,
    /// Number of classes
    pub num_classes: u32,
    /// Data distribution type
    pub distribution: DataDistribution,
    /// Local dataset size
    pub local_size: u32,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            name: "synthetic".to_string(),
            num_classes: 10,
            distribution: DataDistribution::Iid,
            local_size: 1000,
        }
    }
}

/// Data distribution types for federated learning
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataDistribution {
    /// Independent and identically distributed
    Iid,
    /// Non-IID with label skew
    NonIidLabel { alpha: u32 },
    /// Non-IID with feature skew
    NonIidFeature,
    /// Non-IID with quantity skew
    NonIidQuantity,
}

/// Privacy configuration for federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Enable differential privacy
    pub enable_dp: bool,
    /// Differential privacy epsilon
    pub dp_epsilon: f32,
    /// Differential privacy delta
    pub dp_delta: f32,
    /// Clipping norm for gradients
    pub clipping_norm: f32,
    /// Noise multiplier
    pub noise_multiplier: f32,
    /// Enable secure aggregation
    pub enable_secure_agg: bool,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            enable_dp: true,
            dp_epsilon: 1.0,
            dp_delta: 1e-5,
            clipping_norm: 1.0,
            noise_multiplier: 1.0,
            enable_secure_agg: true,
        }
    }
}

/// Resource constraints for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Maximum memory usage in MB
    pub max_memory_mb: u32,
    /// Maximum training time in seconds
    pub max_training_time_sec: u32,
    /// Maximum energy consumption in joules
    pub max_energy_joules: f32,
    /// Minimum battery level to participate
    pub min_battery_level: f32,
    /// Maximum bandwidth usage in MB
    pub max_bandwidth_mb: f32,
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_memory_mb: 1024,
            max_training_time_sec: 300, // 5 minutes
            max_energy_joules: 100.0,
            min_battery_level: 0.2, // 20%
            max_bandwidth_mb: 10.0,
        }
    }
}

/// Result of local training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Round identifier
    pub round_id: RoundId,
    /// Participant identifier
    pub participant_id: ParticipantId,
    /// Updated model parameters
    pub model_update: ModelParameters,
    /// Training metrics
    pub metrics: TrainingMetrics,
    /// Resource usage during training
    pub resource_usage: ResourceUsage,
    /// Timestamp of completion
    pub timestamp: u64,
}

/// Training performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Training loss
    pub training_loss: f32,
    /// Training accuracy
    pub training_accuracy: f32,
    /// Validation loss (if available)
    pub validation_loss: Option<f32>,
    /// Validation accuracy (if available)
    pub validation_accuracy: Option<f32>,
    /// Number of training examples
    pub num_examples: u32,
    /// Training time in seconds
    pub training_time_sec: f32,
}

/// Resource usage during training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Peak memory usage in MB
    pub peak_memory_mb: f32,
    /// Energy consumed in joules
    pub energy_joules: f32,
    /// FLOPs performed
    pub flops: u64,
    /// Bytes sent/received
    pub bytes_sent: u64,
    pub bytes_received: u64,
    /// Battery drain (start - end level)
    pub battery_drain: f32,
}

/// Aggregation result from a federated learning round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationResult {
    /// Round identifier
    pub round_id: RoundId,
    /// Global model after aggregation
    pub global_model: ModelParameters,
    /// Aggregation statistics
    pub stats: AggregationStats,
    /// List of participating devices
    pub participants: Vec<ParticipantId>,
    /// Timestamp of completion
    pub timestamp: u64,
}

/// Statistics from model aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationStats {
    /// Number of participants
    pub num_participants: u32,
    /// Total training examples across participants
    pub total_examples: u64,
    /// Average training loss
    pub avg_training_loss: f32,
    /// Average training accuracy
    pub avg_training_accuracy: f32,
    /// Model improvement metric
    pub improvement: f32,
    /// Convergence status
    pub converged: bool,
}

/// Federated learning session configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FLSession {
    /// Unique session identifier
    pub session_id: String,
    /// Training configuration
    pub config: TrainingConfig,
    /// Target number of participants
    pub target_participants: u32,
    /// Minimum participants to start
    pub min_participants: u32,
    /// Maximum number of rounds
    pub max_rounds: u64,
    /// Round timeout in seconds
    pub round_timeout_sec: u64,
    /// Convergence threshold
    pub convergence_threshold: f32,
    /// Session status
    pub status: SessionStatus,
    /// Created timestamp
    pub created_at: u64,
}

impl FLSession {
    pub fn new(session_id: String, config: TrainingConfig) -> Self {
        Self {
            session_id,
            config,
            target_participants: 10,
            min_participants: 3,
            max_rounds: 100,
            round_timeout_sec: 600, // 10 minutes
            convergence_threshold: 0.001,
            status: SessionStatus::Created,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

/// Status of a federated learning session
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionStatus {
    Created,
    WaitingForParticipants,
    InProgress,
    Converged,
    MaxRoundsReached,
    Failed,
    Cancelled,
}

/// Federated learning errors
#[derive(Debug, Error)]
pub enum FederatedError {
    #[error("Agent fabric error: {0}")]
    AgentFabricError(#[from] agent_fabric::AgentFabricError),

    #[error("Twin vault error: {0}")]
    TwinVaultError(#[from] twin_vault::TwinVaultError),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Training error: {0}")]
    TrainingError(String),

    #[error("Aggregation error: {0}")]
    AggregationError(String),

    #[error("Participant not found: {0}")]
    ParticipantNotFound(ParticipantId),

    #[error("Round not found: {0}")]
    RoundNotFound(RoundId),

    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("Insufficient participants: got {got}, need {need}")]
    InsufficientParticipants { got: u32, need: u32 },

    #[error("Round timeout")]
    RoundTimeout,

    #[error("Resource constraint violation: {0}")]
    ResourceConstraintViolation(String),

    #[error("Privacy violation: {0}")]
    PrivacyViolation(String),

    #[error("Network error: {0}")]
    NetworkError(String),
}

/// Result type for federated learning operations
pub type Result<T> = std::result::Result<T, FederatedError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_id_creation() {
        let round_id = RoundId::new("session-001".to_string(), 1, 1234567890);
        assert_eq!(round_id.session_id, "session-001");
        assert_eq!(round_id.round_number, 1);
        assert_eq!(round_id.epoch, 1234567890);
    }

    #[test]
    fn test_round_id_generation() {
        let round_id = RoundId::generate("session-002".to_string());
        assert_eq!(round_id.session_id, "session-002");
        assert_eq!(round_id.round_number, 0);
        assert!(round_id.epoch > 0);
    }

    #[test]
    fn test_round_id_next() {
        let round_id = RoundId::new("session-003".to_string(), 5, 1234567890);
        let next_round = round_id.next_round();
        assert_eq!(next_round.session_id, "session-003");
        assert_eq!(next_round.round_number, 6);
        assert_eq!(next_round.epoch, 1234567890);
    }

    #[test]
    fn test_device_capabilities_default() {
        let caps = DeviceCapabilities::default();
        assert_eq!(caps.cpu_cores, 4);
        assert_eq!(caps.memory_mb, 4096);
        assert_eq!(caps.battery_level, Some(0.8));
        assert!(caps.ble_support);
        assert!(caps.wifi_direct);
        assert!(caps.is_online);
    }

    #[test]
    fn test_model_parameters() {
        let metadata = ModelMetadata {
            architecture: "test_model".to_string(),
            parameter_count: 1000,
            input_shape: vec![28, 28, 1],
            output_shape: vec![10],
            compression: CompressionType::None,
            quantization: QuantizationType::Float32,
        };

        let params =
            ModelParameters::new("v1.0".to_string(), Bytes::from(vec![1, 2, 3, 4]), metadata);

        assert_eq!(params.version, "v1.0");
        assert_eq!(params.size_bytes(), 4);
        assert!(params.signature.is_none());
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.local_epochs, 5);
        assert_eq!(config.batch_size, 32);
        assert!(config.privacy_config.enable_dp);
        assert!(config.privacy_config.enable_secure_agg);
    }

    #[test]
    fn test_fl_session_creation() {
        let config = TrainingConfig::default();
        let session = FLSession::new("test-session".to_string(), config);

        assert_eq!(session.session_id, "test-session");
        assert_eq!(session.status, SessionStatus::Created);
        assert_eq!(session.target_participants, 10);
        assert_eq!(session.min_participants, 3);
        assert!(session.created_at > 0);
    }

    #[test]
    fn test_participant_id() {
        let agent_id = AgentId::new("phone-001", "mobile-node");
        let capabilities = DeviceCapabilities::default();
        let participant = ParticipantId::new(agent_id.clone(), DeviceType::Phone, capabilities);

        assert_eq!(participant.agent_id, agent_id);
        assert_eq!(participant.device_type, DeviceType::Phone);
    }
}
