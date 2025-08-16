//! Mock communication test for federated learning framework
//!
//! Tests core functionality without external dependencies that require newer Rust versions

use federated::*;
use ndarray::Array1;
use bytes::Bytes;

/// Mock FL communication test
#[test]
fn test_mock_fl_communication() {
    println!("🚀 Starting mock FL communication test...");

    // Test 1: Create participants
    let participants = create_test_participants();
    println!("✅ Created {} participants", participants.len());

    for (i, participant) in participants.iter().enumerate() {
        println!("  Participant {}: {} ({:?})",
                 i + 1, participant.agent_id, participant.device_type);
    }

    // Test 2: Create round and training results
    let round_id = RoundId::new("test-session".to_string(), 1, 1234567890);
    let training_results = create_test_training_results(&round_id, &participants);
    println!("✅ Generated {} training results for round {}",
             training_results.len(), round_id);

    // Test 3: Test model parameter serialization/deserialization
    for (i, result) in training_results.iter().enumerate() {
        let weights: Array1<f32> = bincode::deserialize(&result.model_update.weights)
            .expect(&format!("Failed to deserialize weights for participant {}", i));
        println!("  Participant {} weights: {:?}", i, weights.as_slice().unwrap());
    }

    // Test 4: Test compression algorithms
    test_compression_algorithms();

    // Test 5: Test differential privacy
    test_differential_privacy_basic();

    // Test 6: Test aggregation statistics
    test_aggregation_stats(&training_results);

    // Test 7: Test participant health
    test_participant_health_basic();

    // Test 8: Test receipts
    test_receipts_basic(&participants);

    println!("✅ Mock FL communication test COMPLETED SUCCESSFULLY!");
}

fn create_test_participants() -> Vec<ParticipantId> {
    vec![
        ParticipantId::new(
            agent_fabric::AgentId::new("mobile-001", "android-phone"),
            DeviceType::Phone,
            DeviceCapabilities {
                cpu_cores: 4,
                memory_mb: 4096,
                battery_level: Some(0.85),
                flops_per_sec: 1_000_000_000,
                bandwidth_mbps: 15.0,
                ble_support: true,
                wifi_direct: true,
                is_online: true,
            },
        ),
        ParticipantId::new(
            agent_fabric::AgentId::new("mobile-002", "ios-phone"),
            DeviceType::Phone,
            DeviceCapabilities {
                cpu_cores: 6,
                memory_mb: 6144,
                battery_level: Some(0.67),
                flops_per_sec: 1_500_000_000,
                bandwidth_mbps: 20.0,
                ble_support: true,
                wifi_direct: true,
                is_online: true,
            },
        ),
        ParticipantId::new(
            agent_fabric::AgentId::new("tablet-001", "android-tablet"),
            DeviceType::Tablet,
            DeviceCapabilities {
                cpu_cores: 8,
                memory_mb: 8192,
                battery_level: Some(0.92),
                flops_per_sec: 2_000_000_000,
                bandwidth_mbps: 25.0,
                ble_support: true,
                wifi_direct: true,
                is_online: true,
            },
        ),
        ParticipantId::new(
            agent_fabric::AgentId::new("laptop-001", "edge-device"),
            DeviceType::Laptop,
            DeviceCapabilities {
                cpu_cores: 8,
                memory_mb: 16384,
                battery_level: None, // Wired
                flops_per_sec: 5_000_000_000,
                bandwidth_mbps: 100.0,
                ble_support: false,
                wifi_direct: false,
                is_online: true,
            },
        ),
    ]
}

fn create_test_training_results(round_id: &RoundId, participants: &[ParticipantId]) -> Vec<TrainingResult> {
    participants.iter().enumerate().map(|(i, participant)| {
        // Create different model weights for each participant to simulate real training
        let base_weights = vec![
            0.1 + (i as f32 * 0.05),
            0.2 + (i as f32 * 0.03),
            0.3 + (i as f32 * 0.02),
            0.4 + (i as f32 * 0.01),
            -0.1 + (i as f32 * 0.04),
        ];

        let weights = Array1::from(base_weights);
        let serialized_weights = bincode::serialize(&weights)
            .expect("Failed to serialize weights");

        TrainingResult {
            round_id: round_id.clone(),
            participant_id: participant.clone(),
            model_update: ModelParameters::new(
                format!("model-v1-participant-{}", i),
                Bytes::from(serialized_weights),
                ModelMetadata {
                    architecture: "test_cnn".to_string(),
                    parameter_count: 5,
                    input_shape: vec![32, 32, 3],
                    output_shape: vec![10],
                    compression: CompressionType::None,
                    quantization: QuantizationType::Float32,
                },
            ),
            metrics: TrainingMetrics {
                training_loss: 0.8 - (i as f32 * 0.1), // Decreasing loss
                training_accuracy: 0.6 + (i as f32 * 0.08), // Increasing accuracy
                validation_loss: Some(0.85 - (i as f32 * 0.08)),
                validation_accuracy: Some(0.55 + (i as f32 * 0.09)),
                num_examples: 1000 + (i * 300) as u32,
                training_time_sec: 45.0 + (i as f32 * 15.0),
            },
            resource_usage: ResourceUsage {
                peak_memory_mb: 80.0 + (i as f32 * 25.0),
                energy_joules: 35.0 + (i as f32 * 12.0),
                flops: 800_000 + (i * 150_000) as u64,
                bytes_sent: 2048 + (i * 512) as u64,
                bytes_received: 4096 + (i * 1024) as u64,
                battery_drain: 0.03 + (i as f32 * 0.01),
            },
            timestamp: 1234567890 + (i * 60) as u64,
        }
    }).collect()
}

fn test_compression_algorithms() {
    println!("🗜️  Testing compression algorithms...");

    let test_weights = Array1::from(vec![0.8, -0.3, 0.9, 0.1, -0.7, 0.4, -0.1, 0.6]);

    // Create secure aggregation for testing
    let privacy_config = PrivacyConfig { enable_dp: false, ..Default::default() };
    let secure_agg = SecureAggregation::new(
        privacy_config,
        CompressionType::None,
        QuantizationType::Float32,
    );

    // Test quantization
    let quantized = secure_agg.quantize_weights(&test_weights, 8)
        .expect("Quantization failed");
    println!("  Original: {:?}", test_weights.as_slice().unwrap());
    println!("  Quantized (8-bit): {:?}", quantized.as_slice().unwrap());

    // Test top-k sparsification
    let sparse_topk = secure_agg.top_k_sparsification(&test_weights, 4)
        .expect("Top-K failed");
    let non_zero_count = sparse_topk.iter().filter(|&&x| x != 0.0).count();
    println!("  Top-K (k=4): {:?}, non-zero: {}", sparse_topk.as_slice().unwrap(), non_zero_count);
    assert_eq!(non_zero_count, 4);

    // Test gradient compression
    let compressed = secure_agg.gradient_compression(&test_weights, 0.5)
        .expect("Gradient compression failed");
    println!("  Gradient compressed (threshold=0.5): {:?}", compressed.as_slice().unwrap());

    println!("  ✅ Compression algorithms working correctly");
}

fn test_differential_privacy_basic() {
    println!("🔒 Testing differential privacy...");

    let dp = DifferentialPrivacy::new(1.0, 1e-5, 1.0, 1.0);
    let gradients = Array1::from(vec![3.0, -2.0, 4.0, 1.0]);

    let dp_gradients = dp.apply_dp_sgd(&gradients)
        .expect("DP-SGD failed");

    let original_norm = gradients.mapv(|x| x * x).sum().sqrt();
    let dp_norm = dp_gradients.mapv(|x| x * x).sum().sqrt();

    println!("  Original gradients: {:?} (L2 norm: {:.3})",
             gradients.as_slice().unwrap(), original_norm);
    println!("  DP gradients: {:?} (L2 norm: {:.3})",
             dp_gradients.as_slice().unwrap(), dp_norm);

    // Test privacy cost computation
    let (eps, delta) = dp.compute_privacy_cost(5, 3);
    println!("  Privacy cost (5 rounds, 3 participants): ε={:.3}, δ={:.6}", eps, delta);

    // Clipped norm should be <= clipping_norm + small tolerance for noise
    assert!(dp_norm <= 1.5);
    println!("  ✅ Differential privacy working correctly");
}

fn test_aggregation_stats(training_results: &[TrainingResult]) {
    println!("📊 Testing aggregation statistics...");

    let privacy_config = PrivacyConfig::default();
    let secure_agg = SecureAggregation::new(
        privacy_config,
        CompressionType::None,
        QuantizationType::Float32,
    );

    let stats = secure_agg.compute_aggregation_stats(training_results);

    println!("  Participants: {}", stats.num_participants);
    println!("  Total examples: {}", stats.total_examples);
    println!("  Average training loss: {:.4}", stats.avg_training_loss);
    println!("  Average training accuracy: {:.4}", stats.avg_training_accuracy);
    println!("  Converged: {}", stats.converged);

    assert_eq!(stats.num_participants, training_results.len() as u32);
    assert!(stats.total_examples > 0);
    assert!(stats.avg_training_loss >= 0.0);
    assert!(stats.avg_training_accuracy >= 0.0 && stats.avg_training_accuracy <= 1.0);

    println!("  ✅ Aggregation statistics working correctly");
}

fn test_participant_health_basic() {
    println!("🏥 Testing participant health...");

    let mut health = ParticipantHealth::new();

    // Test initial state
    assert!(health.is_healthy());
    assert_eq!(health.consecutive_failures, 0);
    assert_eq!(health.success_rate, 1.0);

    // Test failure tracking
    health.update(false);
    health.update(false);
    assert!(health.is_healthy()); // Still healthy after 2 failures

    health.update(false);
    assert!(!health.is_healthy()); // Unhealthy after 3 consecutive failures

    println!("  After 3 failures:");
    println!("    Success rate: {:.2}", health.success_rate);
    println!("    Consecutive failures: {}", health.consecutive_failures);
    println!("    Is healthy: {}", health.is_healthy());

    // Test recovery
    health.update(true);
    assert!(health.is_healthy()); // Healthy again after success

    println!("  After recovery: healthy = {}", health.is_healthy());
    println!("  ✅ Participant health tracking working correctly");
}

fn test_receipts_basic(participants: &[ParticipantId]) {
    println!("🧾 Testing receipts system...");

    let pop = ProofOfParticipation::new();

    for (i, participant) in participants.iter().take(2).enumerate() {
        let metrics = ResourceMetrics {
            num_examples: 1000 + (i * 500) as u32,
            flops: 1_000_000 + (i * 250_000) as u64,
            energy_joules: 45.0 + (i as f32 * 15.0),
            peak_memory_mb: 120.0 + (i as f32 * 30.0),
        };

        let receipt = pop.generate_receipt(participant.clone(), &metrics)
            .expect("Failed to generate receipt");

        println!("  Receipt for {}:", participant.agent_id);
        println!("    Examples: {}", receipt.num_examples);
        println!("    FLOPs: {}", receipt.flops);
        println!("    Energy: {:.1}J", receipt.energy_joules);
        println!("    Timestamp: {}", receipt.timestamp);

        assert_eq!(receipt.participant_id, *participant);
        assert_eq!(receipt.num_examples, metrics.num_examples);
        assert!(receipt.timestamp > 0);
    }

    println!("  ✅ Receipts system working correctly");
}

/// Test round plan creation and metadata
#[test]
fn test_round_plan_functionality() {
    println!("📋 Testing round plan functionality...");

    let participants = create_test_participants();
    let round_id = RoundId::new("test-session".to_string(), 3, 1234567890);
    let config = TrainingConfig::default();

    let mut plan = RoundPlan::new(
        round_id.clone(),
        participants.clone(),
        config,
        std::time::Duration::from_secs(600),
    );

    // Test basic properties
    assert_eq!(plan.round_id, round_id);
    assert_eq!(plan.participants.len(), participants.len());
    assert_eq!(plan.timeout, std::time::Duration::from_secs(600));
    assert!(plan.global_model.is_none());
    assert!(plan.metadata.is_empty());

    // Test metadata addition
    plan = plan.with_metadata("experiment_id".to_string(), "exp-001".to_string());
    plan = plan.with_metadata("researcher".to_string(), "alice".to_string());

    assert_eq!(plan.metadata.get("experiment_id"), Some(&"exp-001".to_string()));
    assert_eq!(plan.metadata.get("researcher"), Some(&"alice".to_string()));

    println!("  Round: {}", plan.round_id);
    println!("  Participants: {}", plan.participants.len());
    println!("  Timeout: {:?}", plan.timeout);
    println!("  Metadata: {:?}", plan.metadata);

    println!("  ✅ Round plan functionality working correctly");
}

/// Test cohort manager
#[test]
fn test_cohort_manager_functionality() {
    println!("👥 Testing cohort manager...");

    let mut cohort = CohortManager::new("test-session".to_string(), 2, 5);
    let participants = create_test_participants();

    // Test initial state
    assert!(!cohort.can_start_round());
    assert_eq!(cohort.active_participants(), 0);

    // Add participants
    for participant in &participants {
        cohort.add_participant(participant.clone())
            .expect("Failed to add participant");
    }

    assert!(cohort.can_start_round());
    assert_eq!(cohort.active_participants(), participants.len() as u32);

    // Test round start
    let round_id = RoundId::new("test-session".to_string(), 1, 1234567890);
    cohort.start_round(round_id.clone());

    assert_eq!(cohort.current_round, 1);
    assert!(!cohort.is_round_complete()); // No results yet

    // Add some training results
    let training_results = create_test_training_results(&round_id, &participants);
    for result in training_results.iter().take(3) { // Add 3 out of 4 results
        cohort.add_training_result(result.clone());
    }

    // Should be complete with 3 out of 4 participants (75% > 70% threshold)
    assert!(cohort.is_round_complete());

    let collected_results = cohort.get_training_results();
    assert_eq!(collected_results.len(), 3);

    println!("  Session: {}", cohort.session_id);
    println!("  Active participants: {}", cohort.active_participants());
    println!("  Current round: {}", cohort.current_round);
    println!("  Round complete: {}", cohort.is_round_complete());
    println!("  Training results: {}", collected_results.len());

    println!("  ✅ Cohort manager working correctly");
}
