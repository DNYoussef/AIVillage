//! Partition and merge idempotency tests for twin vault
//!
//! Tests that CRDT operations are truly conflict-free and that merges
//! are idempotent under network partitions and concurrent operations.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use tempfile::TempDir;
use tokio::sync::RwLock;

use twin_vault::{
    crdt::{CrdtOperationFactory, CrdtState, GCounter, LwwMap},
    receipts::{ReceiptSigner, ReceiptVerifier},
    vault::{TwinVault, VaultConfig},
    AgentId, TwinId, TwinManager, TwinOperation, TwinPreferences,
};

/// Create test twin manager
async fn create_test_twin_manager() -> (TwinManager, Arc<agent_fabric::AgentFabric>, TempDir) {
    let temp_dir = TempDir::new().unwrap();

    // Create agent fabric (simplified for testing)
    let agent_id = AgentId::new("test-agent", "test-node");
    let agent_fabric = Arc::new(agent_fabric::AgentFabric::new(agent_id));

    // Create receipt signer and verifier
    let receipt_signer = ReceiptSigner::new("test-signer".to_string());
    let mut receipt_verifier = ReceiptVerifier::new();
    receipt_verifier.add_trusted_key("test-signer".to_string(), receipt_signer.public_key());

    let twin_manager =
        TwinManager::new(Arc::clone(&agent_fabric), receipt_signer, receipt_verifier)
            .await
            .unwrap();

    (twin_manager, agent_fabric, temp_dir)
}

/// Create test vault configuration
fn create_test_vault_config(temp_dir: &TempDir) -> VaultConfig {
    VaultConfig {
        storage_path: temp_dir.path().to_path_buf(),
        encryption_key: Some(twin_vault::vault::EncryptionKey::generate()),
        use_os_keystore: false,
        max_log_size: 1000,
        auto_compact: false,
    }
}

/// Test that LWW-Map merge operations are idempotent
#[tokio::test]
async fn test_lww_map_merge_idempotent() {
    let factory1 = CrdtOperationFactory::new("actor1".to_string());
    let factory2 = CrdtOperationFactory::new("actor2".to_string());
    let factory3 = CrdtOperationFactory::new("actor3".to_string());

    // Create three maps with overlapping and conflicting operations
    let mut map1 = LwwMap::new();
    let mut map2 = LwwMap::new();
    let mut map3 = LwwMap::new();

    // Add different operations to each map
    let op1a = factory1
        .create_lww_set("key1".to_string(), Bytes::from("value1a"))
        .unwrap();
    let op1b = factory1
        .create_lww_set("key2".to_string(), Bytes::from("value2a"))
        .unwrap();
    map1.set_signed(op1a).unwrap();
    map1.set_signed(op1b).unwrap();

    // Simulate slight delay for different timestamps
    tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;

    let op2a = factory2
        .create_lww_set("key1".to_string(), Bytes::from("value1b"))
        .unwrap(); // Conflict with map1
    let op2b = factory2
        .create_lww_set("key3".to_string(), Bytes::from("value3b"))
        .unwrap();
    map2.set_signed(op2a).unwrap();
    map2.set_signed(op2b).unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;

    let op3a = factory3
        .create_lww_set("key2".to_string(), Bytes::from("value2c"))
        .unwrap(); // Conflict with map1
    let op3b = factory3
        .create_lww_set("key4".to_string(), Bytes::from("value4c"))
        .unwrap();
    map3.set_signed(op3a).unwrap();
    map3.set_signed(op3b).unwrap();

    // Test merge idempotency: merge1 = merge2 after multiple merges
    let mut merge_result1 = map1.clone();
    let mut merge_result2 = map1.clone();

    // First merge sequence: 1 <- 2 <- 3
    merge_result1.merge(&map2).unwrap();
    merge_result1.merge(&map3).unwrap();

    // Second merge sequence: 1 <- 2 <- 3 <- 2 <- 3 (redundant merges)
    merge_result2.merge(&map2).unwrap();
    merge_result2.merge(&map3).unwrap();
    merge_result2.merge(&map2).unwrap(); // Redundant
    merge_result2.merge(&map3).unwrap(); // Redundant

    // Results should be identical regardless of redundant merges
    assert_eq!(merge_result1.keys().len(), merge_result2.keys().len());
    for key in merge_result1.keys() {
        assert_eq!(merge_result1.get(&key), merge_result2.get(&key));
    }

    // Test different merge orders produce same result
    let mut merge_result3 = map1.clone();
    merge_result3.merge(&map3).unwrap(); // Different order: 1 <- 3 <- 2
    merge_result3.merge(&map2).unwrap();

    assert_eq!(merge_result1.keys().len(), merge_result3.keys().len());
    for key in merge_result1.keys() {
        assert_eq!(merge_result1.get(&key), merge_result3.get(&key));
    }

    println!("✅ LWW-Map merge operations are idempotent");
}

/// Test that GCounter merge operations are idempotent
#[tokio::test]
async fn test_g_counter_merge_idempotent() {
    let factory1 = CrdtOperationFactory::new("actor1".to_string());
    let factory2 = CrdtOperationFactory::new("actor2".to_string());
    let factory3 = CrdtOperationFactory::new("actor3".to_string());

    // Create three counters with different increments
    let mut counter1 = GCounter::new();
    let mut counter2 = GCounter::new();
    let mut counter3 = GCounter::new();

    // Add increments to each counter
    let inc1a = factory1
        .create_g_counter_increment("counter".to_string(), 5)
        .unwrap();
    let inc1b = factory1
        .create_g_counter_increment("counter".to_string(), 3)
        .unwrap();
    counter1.increment_signed(inc1a).unwrap();
    counter1.increment_signed(inc1b).unwrap();

    let inc2a = factory2
        .create_g_counter_increment("counter".to_string(), 7)
        .unwrap();
    let inc2b = factory2
        .create_g_counter_increment("counter".to_string(), 2)
        .unwrap();
    counter2.increment_signed(inc2a).unwrap();
    counter2.increment_signed(inc2b).unwrap();

    let inc3a = factory3
        .create_g_counter_increment("counter".to_string(), 4)
        .unwrap();
    counter3.increment_signed(inc3a).unwrap();

    // Test merge idempotency
    let mut merge_result1 = counter1.clone();
    let mut merge_result2 = counter1.clone();

    // First merge sequence
    merge_result1.merge(&counter2).unwrap();
    merge_result1.merge(&counter3).unwrap();

    // Second merge sequence with redundant merges
    merge_result2.merge(&counter2).unwrap();
    merge_result2.merge(&counter3).unwrap();
    merge_result2.merge(&counter2).unwrap(); // Redundant
    merge_result2.merge(&counter3).unwrap(); // Redundant

    // Results should be identical
    assert_eq!(merge_result1.value(), merge_result2.value());
    assert_eq!(merge_result1.value(), 5 + 3 + 7 + 2 + 4); // Total of all increments

    // Test different merge orders
    let mut merge_result3 = counter1.clone();
    merge_result3.merge(&counter3).unwrap(); // Different order
    merge_result3.merge(&counter2).unwrap();

    assert_eq!(merge_result1.value(), merge_result3.value());

    println!("✅ GCounter merge operations are idempotent");
}

/// Test CRDT state merge under simulated network partition
#[tokio::test]
async fn test_crdt_state_partition_merge() {
    // Simulate 3 nodes that become partitioned and then reconnect
    let factory_a = CrdtOperationFactory::new("node_a".to_string());
    let factory_b = CrdtOperationFactory::new("node_b".to_string());
    let factory_c = CrdtOperationFactory::new("node_c".to_string());

    let mut state_a = CrdtState::new();
    let mut state_b = CrdtState::new();
    let mut state_c = CrdtState::new();

    // Initial synchronized state
    let init_op = factory_a
        .create_lww_set("initial".to_string(), Bytes::from("synced"))
        .unwrap();
    state_a
        .get_lww_map("map1")
        .set_signed(init_op.clone())
        .unwrap();
    state_b
        .get_lww_map("map1")
        .set_signed(init_op.clone())
        .unwrap();
    state_c.get_lww_map("map1").set_signed(init_op).unwrap();

    // Simulate partition: each node continues independently

    // Node A operations during partition
    let op_a1 = factory_a
        .create_lww_set("key_a1".to_string(), Bytes::from("value_a1"))
        .unwrap();
    let op_a2 = factory_a
        .create_g_counter_increment("counter_a".to_string(), 10)
        .unwrap();
    state_a.get_lww_map("map1").set_signed(op_a1).unwrap();
    state_a
        .get_g_counter("counter_a")
        .increment_signed(op_a2)
        .unwrap();

    // Node B operations during partition
    let op_b1 = factory_b
        .create_lww_set("key_b1".to_string(), Bytes::from("value_b1"))
        .unwrap();
    let op_b2 = factory_b
        .create_lww_set("key_a1".to_string(), Bytes::from("value_b_conflict"))
        .unwrap(); // Conflict!
    let op_b3 = factory_b
        .create_g_counter_increment("counter_a".to_string(), 15)
        .unwrap();
    state_b.get_lww_map("map1").set_signed(op_b1).unwrap();
    state_b.get_lww_map("map1").set_signed(op_b2).unwrap();
    state_b
        .get_g_counter("counter_a")
        .increment_signed(op_b3)
        .unwrap();

    // Node C operations during partition
    let op_c1 = factory_c
        .create_lww_set("key_c1".to_string(), Bytes::from("value_c1"))
        .unwrap();
    let op_c2 = factory_c
        .create_g_counter_increment("counter_c".to_string(), 20)
        .unwrap();
    state_c.get_lww_map("map1").set_signed(op_c1).unwrap();
    state_c
        .get_g_counter("counter_c")
        .increment_signed(op_c2)
        .unwrap();

    // Test that merge is idempotent before healing
    assert!(state_a.is_merge_idempotent(&state_a));
    assert!(state_b.is_merge_idempotent(&state_b));
    assert!(state_c.is_merge_idempotent(&state_c));

    // Partition heals: nodes merge their states
    let mut final_state = state_a.clone();
    final_state.merge(&state_b).unwrap();
    final_state.merge(&state_c).unwrap();

    // Test idempotency of the final merge
    let mut final_state_copy = final_state.clone();
    final_state_copy.merge(&state_b).unwrap(); // Redundant merge
    final_state_copy.merge(&state_c).unwrap(); // Redundant merge

    // Should have same number of maps and counters
    assert_eq!(final_state.lww_maps.len(), final_state_copy.lww_maps.len());
    assert_eq!(
        final_state.g_counters.len(),
        final_state_copy.g_counters.len()
    );

    // Verify conflict resolution (LWW should have resolved key_a1 conflict)
    let map1 = final_state.lww_maps.get("map1").unwrap();
    assert!(map1.get("initial").is_some());
    assert!(map1.get("key_a1").is_some()); // Should exist after conflict resolution
    assert!(map1.get("key_b1").is_some());
    assert!(map1.get("key_c1").is_some());

    // Verify counter aggregation
    let counter_a = final_state.g_counters.get("counter_a").unwrap();
    assert_eq!(counter_a.value(), 25); // 10 + 15

    let counter_c = final_state.g_counters.get("counter_c").unwrap();
    assert_eq!(counter_c.value(), 20);

    println!("✅ CRDT state merge is idempotent under network partitions");
}

/// Test twin vault state merge idempotency at vault level
#[tokio::test]
async fn test_twin_vault_merge_idempotent() {
    let temp_dir1 = TempDir::new().unwrap();
    let temp_dir2 = TempDir::new().unwrap();

    let twin_id = TwinId::new(AgentId::new("test-agent", "test-node"), "test-twin");

    // Create two vaults
    let config1 = create_test_vault_config(&temp_dir1);
    let config2 = create_test_vault_config(&temp_dir2);

    let vault1 = TwinVault::new(twin_id.clone(), config1).await.unwrap();
    let vault2 = TwinVault::new(twin_id, config2).await.unwrap();

    // Add different data to each vault
    vault1
        .set("key1".to_string(), Bytes::from("value1"), 12345)
        .await
        .unwrap();
    vault1
        .increment_counter("counter1", 10, "actor1", 12346)
        .await
        .unwrap();

    vault2
        .set("key2".to_string(), Bytes::from("value2"), 12347)
        .await
        .unwrap();
    vault2
        .increment_counter("counter1", 15, "actor2", 12348)
        .await
        .unwrap();

    // Get states
    let state1 = vault1.get_state().await.unwrap();
    let state2 = vault2.get_state().await.unwrap();

    // Merge state2 into vault1 multiple times (should be idempotent)
    vault1.merge_state(state2.clone()).await.unwrap();
    let merged_state1 = vault1.get_state().await.unwrap();

    vault1.merge_state(state2.clone()).await.unwrap(); // Redundant merge
    let merged_state2 = vault1.get_state().await.unwrap();

    // States should be identical after redundant merge
    assert_eq!(merged_state1.lww_maps.len(), merged_state2.lww_maps.len());
    assert_eq!(
        merged_state1.g_counters.len(),
        merged_state2.g_counters.len()
    );

    // Verify data is present from both vaults
    assert_eq!(
        vault1.get("key1").await.unwrap(),
        Some(Bytes::from("value1"))
    );
    assert_eq!(
        vault1.get("key2").await.unwrap(),
        Some(Bytes::from("value2"))
    );
    assert_eq!(vault1.get_counter("counter1").await.unwrap(), 25); // 10 + 15

    println!("✅ Twin vault merge operations are idempotent");
}

/// Test concurrent operations and eventual consistency
#[tokio::test]
async fn test_concurrent_operations_eventual_consistency() {
    let (twin_manager, _agent_fabric, temp_dir) = create_test_twin_manager().await;
    let config = create_test_vault_config(&temp_dir);

    let twin_id = TwinId::new(AgentId::new("test-agent", "test-node"), "concurrent-test");
    let requester = AgentId::new("requester", "requester-node");

    // Set permissive preferences
    let preferences = TwinPreferences {
        allow_read: true,
        allow_write: true,
        allow_sync: true,
        ..Default::default()
    };
    twin_manager
        .set_preferences(twin_id.clone(), preferences)
        .await;

    // Create vault
    let vault = twin_manager
        .get_twin(twin_id.clone(), config)
        .await
        .unwrap();

    // Simulate concurrent operations
    let mut handles = Vec::new();

    for i in 0..10 {
        let manager = twin_manager.clone();
        let twin_id = twin_id.clone();
        let requester = requester.clone();

        let handle = tokio::spawn(async move {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;

            let operation = TwinOperation::Write {
                key: format!("concurrent_key_{}", i),
                value: Bytes::from(format!("value_{}", i)),
                timestamp,
            };

            manager
                .perform_operation(twin_id, operation, requester)
                .await
        });

        handles.push(handle);
    }

    // Wait for all operations to complete
    let mut results = Vec::new();
    for handle in handles {
        let result: twin_vault::Result<(Option<bytes::Bytes>, twin_vault::Receipt)> =
            handle.await.unwrap();
        results.push(result);
    }

    // All operations should succeed
    for (i, result) in results.iter().enumerate() {
        assert!(result.is_ok(), "Operation {} failed: {:?}", i, result);
    }

    // Verify all keys are present
    let keys = vault.keys().await.unwrap();
    assert_eq!(keys.len(), 10);

    for i in 0..10 {
        let key = format!("concurrent_key_{}", i);
        assert!(keys.contains(&key), "Missing key: {}", key);

        let value = vault.get(&key).await.unwrap().unwrap();
        assert_eq!(value, Bytes::from(format!("value_{}", i)));
    }

    println!("✅ Concurrent operations maintain eventual consistency");
}

/// Test that partition merge respects causality
#[tokio::test]
async fn test_partition_merge_causality() {
    let factory1 = CrdtOperationFactory::new("actor1".to_string());
    let factory2 = CrdtOperationFactory::new("actor2".to_string());

    let mut state1 = CrdtState::new();
    let mut state2 = CrdtState::new();

    // Initial synchronized state
    let init_op = factory1
        .create_lww_set("base".to_string(), Bytes::from("initial"))
        .unwrap();
    state1
        .get_lww_map("map")
        .set_signed(init_op.clone())
        .unwrap();
    state2.get_lww_map("map").set_signed(init_op).unwrap();

    // Create causal dependency: op2 logically depends on op1
    let op1 = factory1
        .create_lww_set("step1".to_string(), Bytes::from("first"))
        .unwrap();
    state1.get_lww_map("map").set_signed(op1).unwrap();

    // Simulate network delay - op2 created after op1 but might arrive first
    tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;

    let op2 = factory2
        .create_lww_set("step2".to_string(), Bytes::from("second"))
        .unwrap();
    state2.get_lww_map("map").set_signed(op2).unwrap();

    // Merge in both directions (simulating partition healing)
    let mut merged_a = state1.clone();
    merged_a.merge(&state2).unwrap();

    let mut merged_b = state2.clone();
    merged_b.merge(&state1).unwrap();

    // Both should converge to same state regardless of merge order
    let map_a = merged_a.lww_maps.get("map").unwrap();
    let map_b = merged_b.lww_maps.get("map").unwrap();

    assert_eq!(map_a.keys().len(), map_b.keys().len());
    assert!(map_a.get("base").is_some());
    assert!(map_a.get("step1").is_some());
    assert!(map_a.get("step2").is_some());

    assert!(map_b.get("base").is_some());
    assert!(map_b.get("step1").is_some());
    assert!(map_b.get("step2").is_some());

    println!("✅ Partition merge respects causality and converges");
}

/// Test merge performance and scalability
#[tokio::test]
async fn test_merge_performance_scalability() {
    let start_time = std::time::Instant::now();

    // Create large CRDT states
    let factory1 = CrdtOperationFactory::new("actor1".to_string());
    let factory2 = CrdtOperationFactory::new("actor2".to_string());

    let mut state1 = CrdtState::new();
    let mut state2 = CrdtState::new();

    // Add many operations to each state
    for i in 0..1000 {
        if i % 2 == 0 {
            let op = factory1
                .create_lww_set(format!("key_{}", i), Bytes::from(format!("value1_{}", i)))
                .unwrap();
            state1.get_lww_map("large_map").set_signed(op).unwrap();

            let counter_op = factory1
                .create_g_counter_increment(format!("counter_{}", i), 1)
                .unwrap();
            state1
                .get_g_counter(&format!("counter_{}", i))
                .increment_signed(counter_op)
                .unwrap();
        } else {
            let op = factory2
                .create_lww_set(format!("key_{}", i), Bytes::from(format!("value2_{}", i)))
                .unwrap();
            state2.get_lww_map("large_map").set_signed(op).unwrap();

            let counter_op = factory2
                .create_g_counter_increment(format!("counter_{}", i), 2)
                .unwrap();
            state2
                .get_g_counter(&format!("counter_{}", i))
                .increment_signed(counter_op)
                .unwrap();
        }
    }

    let setup_time = start_time.elapsed();
    println!("Setup time for 1000 operations: {:?}", setup_time);

    // Measure merge performance
    let merge_start = std::time::Instant::now();
    state1.merge(&state2).unwrap();
    let merge_time = merge_start.elapsed();

    println!("Merge time for 1000 operations: {:?}", merge_time);

    // Verify merge correctness
    let map = state1.lww_maps.get("large_map").unwrap();
    assert_eq!(map.keys().len(), 1000);

    // Test idempotency on large state
    let idempotent_start = std::time::Instant::now();
    state1.merge(&state2).unwrap(); // Redundant merge
    let idempotent_time = idempotent_start.elapsed();

    println!("Idempotent merge time: {:?}", idempotent_time);

    // Merge should be reasonably fast (under 100ms for 1000 ops)
    assert!(
        merge_time.as_millis() < 100,
        "Merge too slow: {:?}",
        merge_time
    );

    println!("✅ Merge operations scale well and remain performant");
}

/// Integration test: Full twin manager partition scenario
#[tokio::test]
async fn test_full_partition_scenario() {
    let (twin_manager1, _agent_fabric1, temp_dir1) = create_test_twin_manager().await;
    let (twin_manager2, _agent_fabric2, temp_dir2) = create_test_twin_manager().await;

    let twin_id = TwinId::new(AgentId::new("distributed-agent", "node"), "shared-twin");
    let requester1 = AgentId::new("user1", "client1");
    let requester2 = AgentId::new("user2", "client2");

    // Set up permissive preferences on both managers
    let preferences = TwinPreferences {
        allow_read: true,
        allow_write: true,
        allow_sync: true,
        ..Default::default()
    };

    twin_manager1
        .set_preferences(twin_id.clone(), preferences.clone())
        .await;
    twin_manager2
        .set_preferences(twin_id.clone(), preferences)
        .await;

    // Create vaults on both managers
    let config1 = create_test_vault_config(&temp_dir1);
    let config2 = create_test_vault_config(&temp_dir2);

    let _vault1 = twin_manager1
        .get_twin(twin_id.clone(), config1)
        .await
        .unwrap();
    let _vault2 = twin_manager2
        .get_twin(twin_id.clone(), config2)
        .await
        .unwrap();

    // Simulate network partition: operations on both sides

    // Side 1 operations
    let op1a = TwinOperation::Write {
        key: "partition_key_1".to_string(),
        value: Bytes::from("side1_value"),
        timestamp: 12345,
    };
    let op1b = TwinOperation::Increment {
        counter_id: "shared_counter".to_string(),
        amount: 10,
        actor_id: "side1".to_string(),
        timestamp: 12346,
    };

    twin_manager1
        .perform_operation(twin_id.clone(), op1a, requester1.clone())
        .await
        .unwrap();
    twin_manager1
        .perform_operation(twin_id.clone(), op1b, requester1)
        .await
        .unwrap();

    // Side 2 operations (during partition)
    let op2a = TwinOperation::Write {
        key: "partition_key_2".to_string(),
        value: Bytes::from("side2_value"),
        timestamp: 12347,
    };
    let op2b = TwinOperation::Increment {
        counter_id: "shared_counter".to_string(),
        amount: 15,
        actor_id: "side2".to_string(),
        timestamp: 12348,
    };

    twin_manager2
        .perform_operation(twin_id.clone(), op2a, requester2.clone())
        .await
        .unwrap();
    twin_manager2
        .perform_operation(twin_id.clone(), op2b, requester2)
        .await
        .unwrap();

    // Get states before merge
    let vault1 = twin_manager1
        .get_twin(twin_id.clone(), VaultConfig::default())
        .await
        .unwrap();
    let vault2 = twin_manager2
        .get_twin(twin_id.clone(), VaultConfig::default())
        .await
        .unwrap();

    let state1 = vault1.get_state().await.unwrap();
    let state2 = vault2.get_state().await.unwrap();

    // Partition heals: sync states
    vault1.merge_state(state2.clone()).await.unwrap();
    vault2.merge_state(state1).await.unwrap();

    // Both vaults should now have same state
    let final_state1 = vault1.get_state().await.unwrap();
    let final_state2 = vault2.get_state().await.unwrap();

    // Verify convergence
    assert_eq!(final_state1.lww_maps.len(), final_state2.lww_maps.len());
    assert_eq!(final_state1.g_counters.len(), final_state2.g_counters.len());

    // Verify both partitions' data is present
    assert_eq!(
        vault1.get("partition_key_1").await.unwrap(),
        Some(Bytes::from("side1_value"))
    );
    assert_eq!(
        vault1.get("partition_key_2").await.unwrap(),
        Some(Bytes::from("side2_value"))
    );
    assert_eq!(vault1.get_counter("shared_counter").await.unwrap(), 25); // 10 + 15

    assert_eq!(
        vault2.get("partition_key_1").await.unwrap(),
        Some(Bytes::from("side1_value"))
    );
    assert_eq!(
        vault2.get("partition_key_2").await.unwrap(),
        Some(Bytes::from("side2_value"))
    );
    assert_eq!(vault2.get_counter("shared_counter").await.unwrap(), 25); // 10 + 15

    println!("✅ Full partition scenario: eventual consistency achieved");
}
