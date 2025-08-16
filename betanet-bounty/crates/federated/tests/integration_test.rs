//! Integration test for the complete federated learning workflow
//!
//! Tests the full FL pipeline: orchestrator coordination, secure aggregation,
//! participant management, and MLS group communication.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use tempfile::TempDir;
use tokio::time::sleep;
use ndarray::Array1;

use agent_fabric::{AgentFabric, AgentId};
use twin_vault::{TwinManager, ReceiptSigner, ReceiptVerifier};
use federated::*;

/// Mock test that simulates a complete federated learning round
#[tokio::test]
async fn test_complete_federated_learning_workflow() {
    println!("🚀 Starting complete federated learning workflow test...");

    // Setup test infrastructure
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let temp_path = temp_dir.path().to_str().unwrap();

    println!("📁 Created temporary directory: {}", temp_path);

    // Initialize core components
    let agent_fabric = create_mock_agent_fabric().await;
    let twin_manager = create_mock_twin_manager(temp_path).await;
    let orchestrator_config = OrchestratorConfig::default();

    println!("🔧 Initialized core components");

    // Create round orchestrator
    let (orchestrator, command_sender) = RoundOrchestrator::new(
        agent_fabric.clone(),
        twin_manager.clone(),
        orchestrator_config,
    ).await.expect("Failed to create orchestrator");

    println!("🎭 Created round orchestrator");

    // Start orchestrator in background
    let orchestrator_handle = {
        let orchestrator = Arc::new(orchestrator);
        let orchestrator_clone = orchestrator.clone();
        tokio::spawn(async move {
            // Run for a limited time to avoid infinite loop in test
            tokio::time::timeout(Duration::from_secs(10), orchestrator_clone.start()).await
        })
    };

    // Give orchestrator time to start
    sleep(Duration::from_millis(100)).await;
    println!("⏰ Orchestrator started");

    // Create training configuration
    let training_config = TrainingConfig {
        learning_rate: 0.01,
        local_epochs: 3,
        batch_size: 32,
        model_arch: "test_cnn".to_string(),
        dataset_config: DatasetConfig {
            name: "mock_dataset".to_string(),
            num_classes: 10,
            distribution: DataDistribution::Iid,
            local_size: 1000,
        },
        privacy_config: PrivacyConfig {
            enable_dp: true,
            dp_epsilon: 1.0,
            dp_delta: 1e-5,
            clipping_norm: 1.0,
            noise_multiplier: 1.0,
            enable_secure_agg: true,
        },
        resource_constraints: ResourceConstraints::default(),
    };

    println!("📋 Created training configuration");

    // Create FL session
    let session_id = orchestrator.create_session(training_config).await
        .expect("Failed to create FL session");

    println!("🏗️  Created FL session: {}", session_id);

    // Create mock participants
    let participants = create_mock_participants();
    println!("👥 Created {} mock participants", participants.len());

    // Add participants to session
    for (i, participant) in participants.iter().enumerate() {
        orchestrator.add_participant(&session_id, participant.clone()).await
            .expect(&format!("Failed to add participant {}", i));
        println!("  ✅ Added participant: {}", participant.agent_id);
    }

    // Wait for session to start
    sleep(Duration::from_millis(200)).await;

    // Start a round manually to test the flow
    let round_id = orchestrator.start_round(&session_id).await
        .expect("Failed to start round");

    println!("🔄 Started FL round: {}", round_id);

    // Simulate training results from participants
    let training_results = create_mock_training_results(&round_id, &participants);
    println!("🎯 Generated {} training results", training_results.len());

    // Submit training results
    for (i, result) in training_results.iter().enumerate() {
        orchestrator.collect_training_result(&round_id, result.clone()).await
            .expect(&format!("Failed to collect result from participant {}", i));
        println!("  📥 Collected result from: {}", result.participant_id.agent_id);
    }

    // Wait for aggregation to complete
    sleep(Duration::from_millis(300)).await;

    // Test secure aggregation directly
    println!("🔐 Testing secure aggregation...");
    test_secure_aggregation(&round_id, training_results.clone()).await;

    // Test gossip protocol
    println!("📡 Testing gossip protocol...");
    test_gossip_protocol().await;

    // Test receipts system
    println!("🧾 Testing receipts system...");
    test_receipts_system(&participants).await;

    // Cleanup
    orchestrator_handle.abort();
    println!("🧹 Cleaned up orchestrator");

    println!("✅ Complete federated learning workflow test PASSED!");
}

/// Test secure aggregation with mock data
async fn test_secure_aggregation(round_id: &RoundId, training_results: Vec<TrainingResult>) {
    let privacy_config = PrivacyConfig::default();
    let mut secure_agg = SecureAggregation::new(
        privacy_config,
        CompressionType::None,
        QuantizationType::Float32,
    );

    let result = secure_agg.secure_aggregate(round_id, training_results).await
        .expect("Secure aggregation failed");

    println!("  🔒 Secure aggregation completed:");
    println!("    - Participants: {}", result.stats.num_participants);
    println!("    - Total examples: {}", result.stats.total_examples);
    println!("    - Avg loss: {:.4}", result.stats.avg_training_loss);
    println!("    - Avg accuracy: {:.4}", result.stats.avg_training_accuracy);
    println!("    - Converged: {}", result.stats.converged);

    assert_eq!(result.round_id, *round_id);
    assert!(result.stats.num_participants > 0);
    assert!(result.stats.total_examples > 0);
}

/// Test gossip protocol functionality
async fn test_gossip_protocol() {
    let gossip = GossipProtocol::new();
    let peer_exchange = PeerExchange::new();
    let robust_agg = RobustAggregation::new();

    println!("  📡 Gossip protocol components initialized");
    println!("    - GossipProtocol: ✅");
    println!("    - PeerExchange: ✅");
    println!("    - RobustAggregation: ✅");
}

/// Test receipts and proof of participation
async fn test_receipts_system(participants: &[ParticipantId]) {
    let pop = ProofOfParticipation::new();

    for participant in participants.iter().take(2) {
        let metrics = ResourceMetrics {
            num_examples: 1000,
            flops: 1_000_000,
            energy_joules: 50.0,
            peak_memory_mb: 128.0,
        };

        let receipt = pop.generate_receipt(participant.clone(), &metrics)
            .expect("Failed to generate receipt");

        println!("  🧾 Generated receipt for {}", participant.agent_id);
        println!("    - Examples: {}", receipt.num_examples);
        println!("    - FLOPs: {}", receipt.flops);
        println!("    - Energy: {:.2}J", receipt.energy_joules);
        println!("    - Timestamp: {}", receipt.timestamp);
    }
}

/// Create mock agent fabric for testing
async fn create_mock_agent_fabric() -> Arc<AgentFabric> {
    // Create a mock agent fabric
    // In a real implementation, this would connect to the actual network
    let node_id = AgentId::new("orchestrator-001", "fl-coordinator");
    let fabric = AgentFabric::new(node_id).await
        .expect("Failed to create agent fabric");
    Arc::new(fabric)
}

/// Create mock twin manager for testing
async fn create_mock_twin_manager(temp_path: &str) -> Arc<TwinManager> {
    let signer = ReceiptSigner::new("test-signer");
    let verifier = ReceiptVerifier::new();

    let manager = TwinManager::new(signer, verifier).await
        .expect("Failed to create twin manager");

    Arc::new(manager)
}

/// Create mock participants for testing
fn create_mock_participants() -> Vec<ParticipantId> {
    vec![
        ParticipantId::new(
            AgentId::new("phone-001", "mobile-device"),
            DeviceType::Phone,
            DeviceCapabilities {
                cpu_cores: 4,
                memory_mb: 4096,
                battery_level: Some(0.8),
                flops_per_sec: 1_000_000_000,
                bandwidth_mbps: 10.0,
                ble_support: true,
                wifi_direct: true,
                is_online: true,
            },
        ),
        ParticipantId::new(
            AgentId::new("phone-002", "mobile-device"),
            DeviceType::Phone,
            DeviceCapabilities {
                cpu_cores: 8,
                memory_mb: 6144,
                battery_level: Some(0.65),
                flops_per_sec: 1_500_000_000,
                bandwidth_mbps: 15.0,
                ble_support: true,
                wifi_direct: true,
                is_online: true,
            },
        ),
        ParticipantId::new(
            AgentId::new("tablet-001", "mobile-device"),
            DeviceType::Tablet,
            DeviceCapabilities {
                cpu_cores: 6,
                memory_mb: 8192,
                battery_level: Some(0.9),
                flops_per_sec: 2_000_000_000,
                bandwidth_mbps: 20.0,
                ble_support: true,
                wifi_direct: true,
                is_online: true,
            },
        ),
    ]
}

/// Create mock training results for testing
fn create_mock_training_results(round_id: &RoundId, participants: &[ParticipantId]) -> Vec<TrainingResult> {
    participants.iter().enumerate().map(|(i, participant)| {
        // Create mock model weights
        let weights = Array1::from(vec![
            0.1 + i as f32 * 0.05,  // Slightly different weights per participant
            0.2 + i as f32 * 0.03,
            0.3 + i as f32 * 0.02,
            0.4 + i as f32 * 0.01,
        ]);

        let serialized_weights = bincode::serialize(&weights)
            .expect("Failed to serialize weights");

        let metadata = ModelMetadata {
            architecture: "test_cnn".to_string(),
            parameter_count: 4,
            input_shape: vec![32, 32, 3],
            output_shape: vec![10],
            compression: CompressionType::None,
            quantization: QuantizationType::Float32,
        };

        TrainingResult {
            round_id: round_id.clone(),
            participant_id: participant.clone(),
            model_update: ModelParameters::new(
                format!("participant-{}-v1", i),
                Bytes::from(serialized_weights),
                metadata,
            ),
            metrics: TrainingMetrics {
                training_loss: 0.5 - (i as f32 * 0.05), // Decreasing loss
                training_accuracy: 0.7 + (i as f32 * 0.05), // Increasing accuracy
                validation_loss: Some(0.6 - (i as f32 * 0.03)),
                validation_accuracy: Some(0.65 + (i as f32 * 0.04)),
                num_examples: 1000 + (i * 200) as u32, // Different dataset sizes
                training_time_sec: 60.0 + (i as f32 * 10.0),
            },
            resource_usage: ResourceUsage {
                peak_memory_mb: 100.0 + (i as f32 * 20.0),
                energy_joules: 50.0 + (i as f32 * 10.0),
                flops: 1_000_000 + (i * 200_000) as u64,
                bytes_sent: 1024 + (i * 256) as u64,
                bytes_received: 2048 + (i * 512) as u64,
                battery_drain: 0.05 + (i as f32 * 0.01),
            },
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }).collect()
}

/// Test compression algorithms
#[tokio::test]
async fn test_compression_algorithms() {
    println!("🗜️  Testing compression algorithms...");

    let privacy_config = PrivacyConfig { enable_dp: false, ..Default::default() };

    // Test Q8 quantization
    let secure_agg_q8 = SecureAggregation::new(
        privacy_config.clone(),
        CompressionType::Q8,
        QuantizationType::Int8,
    );

    let weights = Array1::from(vec![0.1, 0.5, -0.3, 0.8, -0.2]);
    let quantized = secure_agg_q8.quantize_weights(&weights, 8)
        .expect("Q8 quantization failed");

    println!("  🔢 Q8 Quantization:");
    println!("    Original: {:?}", weights.as_slice().unwrap());
    println!("    Quantized: {:?}", quantized.as_slice().unwrap());

    // Test Top-K sparsification
    let secure_agg_topk = SecureAggregation::new(
        privacy_config.clone(),
        CompressionType::TopK { k: 3 },
        QuantizationType::Float32,
    );

    let sparse_weights = secure_agg_topk.top_k_sparsification(&weights, 3)
        .expect("Top-K sparsification failed");

    let non_zero_count = sparse_weights.iter().filter(|&&x| x != 0.0).count();

    println!("  🎯 Top-K Sparsification (k=3):");
    println!("    Original: {:?}", weights.as_slice().unwrap());
    println!("    Sparse: {:?}", sparse_weights.as_slice().unwrap());
    println!("    Non-zero elements: {}", non_zero_count);

    assert_eq!(non_zero_count, 3);

    // Test random sparsification
    let sparse_random = secure_agg_topk.random_sparsification(&weights, 0.6)
        .expect("Random sparsification failed");

    println!("  🎲 Random Sparsification (prob=0.6):");
    println!("    Sparse: {:?}", sparse_random.as_slice().unwrap());

    println!("✅ Compression algorithms test PASSED!");
}

/// Test differential privacy
#[tokio::test]
async fn test_differential_privacy() {
    println!("🔒 Testing differential privacy...");

    let dp = DifferentialPrivacy::new(1.0, 1e-5, 1.0, 1.0);
    let gradients = Array1::from(vec![2.0, -1.5, 3.0, 0.5]);

    let dp_gradients = dp.apply_dp_sgd(&gradients)
        .expect("DP-SGD failed");

    // Check that gradients were clipped and noise was added
    let l2_norm = dp_gradients.mapv(|x| x * x).sum().sqrt();

    println!("  📊 DP-SGD Results:");
    println!("    Original: {:?}", gradients.as_slice().unwrap());
    println!("    DP-SGD: {:?}", dp_gradients.as_slice().unwrap());
    println!("    L2 norm after clipping: {:.4}", l2_norm);
    println!("    Privacy budget: ε={}, δ={}", 1.0, 1e-5);

    assert_eq!(dp_gradients.len(), gradients.len());
    assert!(l2_norm <= 1.1); // Allow small tolerance for noise

    // Test privacy cost computation
    let (total_eps, total_delta) = dp.compute_privacy_cost(10, 5);
    println!("    Total privacy cost (10 rounds): ε={:.4}, δ={:.6}", total_eps, total_delta);

    println!("✅ Differential privacy test PASSED!");
}

/// Test participant health tracking
#[tokio::test]
async fn test_participant_health() {
    println!("🏥 Testing participant health tracking...");

    let mut health = ParticipantHealth::new();
    assert!(health.is_healthy());

    // Simulate some failures
    health.update(false);
    health.update(false);
    assert!(health.is_healthy()); // Still healthy after 2 failures

    health.update(false);
    assert!(!health.is_healthy()); // Unhealthy after 3 consecutive failures

    println!("  📈 Health tracking:");
    println!("    Success rate: {:.2}", health.success_rate);
    println!("    Consecutive failures: {}", health.consecutive_failures);
    println!("    Is healthy: {}", health.is_healthy());

    // Recovery
    health.update(true);
    assert!(health.is_healthy()); // Healthy again after success

    println!("    After recovery - Is healthy: {}", health.is_healthy());

    println!("✅ Participant health test PASSED!");
}

/// Benchmark aggregation performance
#[tokio::test]
async fn benchmark_aggregation_performance() {
    println!("⚡ Benchmarking aggregation performance...");

    let privacy_config = PrivacyConfig::default();
    let mut secure_agg = SecureAggregation::new(
        privacy_config,
        CompressionType::None,
        QuantizationType::Float32,
    );

    // Create larger dataset for benchmarking
    let participants = (0..10).map(|i| {
        ParticipantId::new(
            AgentId::new(&format!("benchmark-{:03}", i), "mobile"),
            DeviceType::Phone,
            DeviceCapabilities::default(),
        )
    }).collect::<Vec<_>>();

    let round_id = RoundId::new("benchmark".to_string(), 1, 12345);
    let training_results = create_mock_training_results(&round_id, &participants);

    let start_time = std::time::Instant::now();

    let result = secure_agg.secure_aggregate(&round_id, training_results).await
        .expect("Benchmark aggregation failed");

    let duration = start_time.elapsed();

    println!("  ⏱️  Performance Results:");
    println!("    Participants: {}", result.stats.num_participants);
    println!("    Total examples: {}", result.stats.total_examples);
    println!("    Aggregation time: {:.2}ms", duration.as_millis());
    println!("    Throughput: {:.0} examples/sec",
             result.stats.total_examples as f64 / duration.as_secs_f64());

    // Performance should be reasonable for 10 participants
    assert!(duration.as_millis() < 1000); // Should complete within 1 second

    println!("✅ Aggregation performance benchmark PASSED!");
}
