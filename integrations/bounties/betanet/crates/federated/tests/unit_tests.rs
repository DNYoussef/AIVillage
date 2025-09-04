//! Unit tests for individual federated learning components

use federated::*;
use ndarray::Array1;

#[test]
fn test_round_id_creation_and_display() {
    let round_id = RoundId::new("test-session".to_string(), 5, 1234567890);

    assert_eq!(round_id.session_id, "test-session");
    assert_eq!(round_id.round_number, 5);
    assert_eq!(round_id.epoch, 1234567890);

    let display_str = format!("{}", round_id);
    assert_eq!(display_str, "test-session:1234567890:5");

    let next = round_id.next_round();
    assert_eq!(next.round_number, 6);
    assert_eq!(next.session_id, round_id.session_id);
    assert_eq!(next.epoch, round_id.epoch);
}

#[test]
fn test_participant_id_display() {
    let agent_id = agent_fabric::AgentId::new("test-phone", "mobile");
    let capabilities = DeviceCapabilities::default();
    let participant = ParticipantId::new(agent_id, DeviceType::Phone, capabilities);

    let display_str = format!("{}", participant);
    assert!(display_str.contains("test-phone"));
    assert!(display_str.contains("Phone"));
}

#[test]
fn test_device_capabilities_equality() {
    let caps1 = DeviceCapabilities {
        cpu_cores: 4,
        memory_mb: 4096,
        battery_level: Some(0.8),
        flops_per_sec: 1_000_000_000,
        bandwidth_mbps: 10.0,
        ble_support: true,
        wifi_direct: true,
        is_online: true,
    };

    let caps2 = DeviceCapabilities {
        cpu_cores: 4,
        memory_mb: 4096,
        battery_level: Some(0.8),
        flops_per_sec: 1_000_000_000,
        bandwidth_mbps: 10.0,
        ble_support: true,
        wifi_direct: true,
        is_online: true,
    };

    assert_eq!(caps1, caps2);
    assert_eq!(caps1, caps1.clone());
}

#[test]
fn test_compression_types() {
    let none = CompressionType::None;
    let q8 = CompressionType::Q8;
    let topk = CompressionType::TopK { k: 10 };
    let random = CompressionType::Random { prob: 0.5 };
    let gradient = CompressionType::Gradient { threshold: 0.1 };

    // Test partial equality (note: we removed Eq derive due to f32 fields)
    assert_eq!(none, CompressionType::None);
    assert_eq!(q8, CompressionType::Q8);

    // Test cloning
    assert_eq!(topk, topk.clone());
    assert_eq!(random, random.clone());
    assert_eq!(gradient, gradient.clone());
}

#[test]
fn test_training_config_defaults() {
    let config = TrainingConfig::default();

    assert_eq!(config.learning_rate, 0.01);
    assert_eq!(config.local_epochs, 5);
    assert_eq!(config.batch_size, 32);
    assert_eq!(config.model_arch, "simple_cnn");
    assert!(config.privacy_config.enable_dp);
    assert!(config.privacy_config.enable_secure_agg);
}

#[test]
fn test_privacy_config_defaults() {
    let config = PrivacyConfig::default();

    assert!(config.enable_dp);
    assert_eq!(config.dp_epsilon, 1.0);
    assert_eq!(config.dp_delta, 1e-5);
    assert_eq!(config.clipping_norm, 1.0);
    assert_eq!(config.noise_multiplier, 1.0);
    assert!(config.enable_secure_agg);
}

#[test]
fn test_fl_session_creation() {
    let config = TrainingConfig::default();
    let session = FLSession::new("test-session".to_string(), config);

    assert_eq!(session.session_id, "test-session");
    assert_eq!(session.status, SessionStatus::Created);
    assert_eq!(session.target_participants, 10);
    assert_eq!(session.min_participants, 3);
    assert_eq!(session.max_rounds, 100);
    assert!(session.created_at > 0);
}

#[test]
fn test_model_parameters() {
    let test_weights = vec![1.0f32, 2.0, 3.0, 4.0];
    let weights_bytes = bincode::serialize(&test_weights).unwrap();

    let metadata = ModelMetadata {
        architecture: "test_model".to_string(),
        parameter_count: 4,
        input_shape: vec![28, 28, 1],
        output_shape: vec![10],
        compression: CompressionType::None,
        quantization: QuantizationType::Float32,
    };

    let params = ModelParameters::new(
        "v1.0".to_string(),
        bytes::Bytes::from(weights_bytes),
        metadata,
    );

    assert_eq!(params.version, "v1.0");
    assert!(params.size_bytes() > 0);
    assert!(params.signature.is_none());
}

#[test]
fn test_fedavg_config() {
    let config = FedAvgConfig::default();

    assert_eq!(config.min_participants, 2);
    assert_eq!(config.max_participants, 100);
    assert_eq!(config.aggregation_timeout_sec, 600);
    assert!(config.enable_validation);
}

#[test]
fn test_resource_metrics() {
    let metrics = ResourceMetrics::new();

    assert_eq!(metrics.num_examples, 0);
    assert_eq!(metrics.flops, 0);
    assert_eq!(metrics.energy_joules, 0.0);
    assert_eq!(metrics.peak_memory_mb, 0.0);
}

#[test]
fn test_fl_receipt_creation() {
    let agent_id = agent_fabric::AgentId::new("test-participant", "mobile");
    let participant =
        ParticipantId::new(agent_id, DeviceType::Phone, DeviceCapabilities::default());

    let receipt = FLReceipt::new(participant.clone(), 1000, 1_000_000, 50.0);

    assert_eq!(receipt.participant_id, participant);
    assert_eq!(receipt.num_examples, 1000);
    assert_eq!(receipt.flops, 1_000_000);
    assert_eq!(receipt.energy_joules, 50.0);
    assert!(receipt.timestamp > 0);
    assert!(receipt.signature.is_empty()); // No signature by default
}

#[test]
fn test_receipt_sign_verify_and_storage() {
    let pop = ProofOfParticipation::new();
    let participant = ParticipantId::new(
        agent_fabric::AgentId::new("tester", "device"),
        DeviceType::Phone,
        DeviceCapabilities::default(),
    );
    let metrics = ResourceMetrics {
        num_examples: 10,
        flops: 100,
        energy_joules: 1.0,
        peak_memory_mb: 1.0,
    };

    let receipt = pop
        .generate_receipt(participant.clone(), &metrics)
        .unwrap();
    assert!(pop.verify_receipt(&receipt).unwrap());
    let stored = pop
        .get_receipt(&participant, receipt.timestamp)
        .unwrap()
        .unwrap();
    assert_eq!(stored.num_examples, metrics.num_examples);
}

#[test]
fn test_gossip_peer_exchange_membership() {
    let mut g1 = GossipProtocol::new();
    let mut g2 = GossipProtocol::new();
    let p1 = ParticipantId::new(
        agent_fabric::AgentId::new("p1", "dev"),
        DeviceType::Phone,
        DeviceCapabilities::default(),
    );
    let p2 = ParticipantId::new(
        agent_fabric::AgentId::new("p2", "dev"),
        DeviceType::Tablet,
        DeviceCapabilities::default(),
    );
    g1.add_peer(p1.clone());
    g2.add_peer(p2.clone());
    PeerExchange::exchange(&mut g1, &g2.peers());
    assert_eq!(g1.peers().len(), 2);
    assert!(g1
        .peers()
        .iter()
        .any(|p| p.agent_id.to_string() == p2.agent_id.to_string()));
}

#[test]
fn test_split_learning_workflow() {
    let mut split = SplitLearning::new();
    let batches = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
    let agg = split.train_round(batches.clone()).unwrap();
    assert_eq!(agg, vec![1.5, 3.0]);
    let replay = split.replay_microbatches().unwrap();
    assert_eq!(replay, batches);
}
