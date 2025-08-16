//! Receipt verification tests for twin vault operations
//!
//! Tests that COSE-signed receipts are properly generated, verifiable,
//! and provide cryptographic proof of operations performed on twin state.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use ed25519_dalek::Keypair;
use rand::rngs::OsRng;
use tempfile::TempDir;

use twin_vault::{
    AgentId, TwinId, TwinOperation, TwinManager, TwinPreferences,
    receipts::{Receipt, ReceiptSigner, ReceiptVerifier, ReceiptStore, ReceiptVerificationResult},
    vault::VaultConfig,
};

/// Create test twin manager with receipt signer/verifier
async fn create_test_twin_manager_with_receipts() -> (TwinManager, ReceiptSigner, ReceiptVerifier, TempDir) {
    let temp_dir = TempDir::new().unwrap();

    // Create agent fabric
    let agent_id = AgentId::new("test-agent", "test-node");
    let agent_fabric = std::sync::Arc::new(agent_fabric::AgentFabric::new(agent_id));

    // Create receipt signer
    let receipt_signer = ReceiptSigner::new("test-signer".to_string());

    // Create receipt verifier with trusted key
    let mut receipt_verifier = ReceiptVerifier::new();
    receipt_verifier.add_trusted_key("test-signer".to_string(), receipt_signer.public_key());

    let twin_manager = TwinManager::new(
        agent_fabric,
        receipt_signer.clone(),
        receipt_verifier.clone(),
    ).await.unwrap();

    (twin_manager, receipt_signer, receipt_verifier, temp_dir)
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

/// Test basic receipt creation and verification
#[tokio::test]
async fn test_basic_receipt_verification() {
    let signer = ReceiptSigner::new("test-signer".to_string());
    let mut verifier = ReceiptVerifier::new();
    verifier.add_trusted_key("test-signer".to_string(), signer.public_key());

    let twin_id = TwinId::new(AgentId::new("test-agent", "test-node"), "test-twin");
    let requester = AgentId::new("requester", "requester-node");

    let operation = TwinOperation::Write {
        key: "test_key".to_string(),
        value: Bytes::from("test_value"),
        timestamp: 12345,
    };

    // Create receipt
    let receipt = signer.sign_operation(&twin_id, &operation, &requester, true).await.unwrap();

    // Verify receipt
    let verification = verifier.verify_receipt(&receipt).unwrap();
    assert!(verification.is_trusted_and_valid());
    assert_eq!(verification.signer_id, Some("test-signer".to_string()));

    // Test direct verification
    let is_valid = receipt.verify(&signer.public_key()).unwrap();
    assert!(is_valid);

    println!("✅ Basic receipt verification works");
}

/// Test receipt verification with untrusted signer
#[tokio::test]
async fn test_untrusted_signer_verification() {
    let trusted_signer = ReceiptSigner::new("trusted-signer".to_string());
    let untrusted_signer = ReceiptSigner::new("untrusted-signer".to_string());

    let mut verifier = ReceiptVerifier::new();
    verifier.add_trusted_key("trusted-signer".to_string(), trusted_signer.public_key());
    // Note: untrusted_signer is NOT added to verifier

    let twin_id = TwinId::new(AgentId::new("test-agent", "test-node"), "test-twin");
    let requester = AgentId::new("requester", "requester-node");

    let operation = TwinOperation::Read {
        key: "test_key".to_string(),
        timestamp: 12345,
    };

    // Create receipt with untrusted signer
    let receipt = untrusted_signer.sign_operation(&twin_id, &operation, &requester, true).await.unwrap();

    // Verification should fail for untrusted signer
    let verification = verifier.verify_receipt(&receipt).unwrap();
    assert!(!verification.is_trusted_and_valid());
    assert!(!verification.trusted);
    assert_eq!(verification.signer_id, Some("untrusted-signer".to_string()));

    println!("✅ Untrusted signer verification correctly fails");
}

/// Test receipt verification with tampered data
#[tokio::test]
async fn test_tampered_receipt_verification() {
    let signer = ReceiptSigner::new("test-signer".to_string());
    let mut verifier = ReceiptVerifier::new();
    verifier.add_trusted_key("test-signer".to_string(), signer.public_key());

    let twin_id = TwinId::new(AgentId::new("test-agent", "test-node"), "test-twin");
    let requester = AgentId::new("requester", "requester-node");

    let operation = TwinOperation::Delete {
        key: "test_key".to_string(),
        timestamp: 12345,
    };

    // Create valid receipt
    let mut receipt = signer.sign_operation(&twin_id, &operation, &requester, true).await.unwrap();

    // Tamper with receipt data
    receipt.success = false; // Change success flag

    // Verification should fail for tampered receipt
    let verification = verifier.verify_receipt(&receipt).unwrap();
    assert!(!verification.is_trusted_and_valid());
    assert!(verification.trusted); // Signer is trusted
    assert!(!verification.valid); // But signature is invalid

    println!("✅ Tampered receipt verification correctly fails");
}

/// Test receipt with result data verification
#[tokio::test]
async fn test_receipt_with_result_data() {
    let signer = ReceiptSigner::new("result-signer".to_string());
    let mut verifier = ReceiptVerifier::new();
    verifier.add_trusted_key("result-signer".to_string(), signer.public_key());

    let twin_id = TwinId::new(AgentId::new("data-agent", "data-node"), "data-twin");
    let requester = AgentId::new("reader", "reader-node");

    let operation = TwinOperation::Read {
        key: "data_key".to_string(),
        timestamp: 12345,
    };

    let result_data = b"sensitive_read_result";

    // Create receipt with result data
    let receipt = signer.sign_operation_with_result(
        &twin_id,
        &operation,
        &requester,
        true,
        Some(result_data)
    ).await.unwrap();

    // Verify receipt
    let verification = verifier.verify_receipt(&receipt).unwrap();
    assert!(verification.is_trusted_and_valid());

    // Check that result hash is present
    assert!(receipt.result_hash.is_some());

    // Verify hash matches the data
    use sha2::{Sha256, Digest};
    let expected_hash = Sha256::digest(result_data).to_vec();
    assert_eq!(receipt.result_hash.unwrap(), expected_hash);

    println!("✅ Receipt with result data verification works");
}

/// Test receipt expiration handling
#[tokio::test]
async fn test_receipt_expiration() {
    let signer = ReceiptSigner::new("expiry-signer".to_string());
    let mut verifier = ReceiptVerifier::new();
    verifier.add_trusted_key("expiry-signer".to_string(), signer.public_key());

    let twin_id = TwinId::new(AgentId::new("test-agent", "test-node"), "test-twin");
    let requester = AgentId::new("requester", "requester-node");

    let operation = TwinOperation::Increment {
        counter_id: "test_counter".to_string(),
        amount: 5,
        actor_id: "test_actor".to_string(),
        timestamp: 12345,
    };

    // Create receipt
    let receipt = signer.sign_operation(&twin_id, &operation, &requester, true).await.unwrap();

    // Test immediate verification (should pass)
    let verification = verifier.verify_receipt(&receipt).unwrap();
    assert!(verification.is_trusted_and_valid());

    // Test with very short expiry (should fail)
    let verification_with_expiry = verifier.verify_receipt_with_expiry(&receipt, 0).unwrap();
    assert!(!verification_with_expiry.valid);
    assert!(verification_with_expiry.error.as_ref().unwrap().contains("expired"));

    // Test with long expiry (should pass)
    let verification_with_long_expiry = verifier.verify_receipt_with_expiry(&receipt, 60000).unwrap();
    assert!(verification_with_long_expiry.is_trusted_and_valid());

    println!("✅ Receipt expiration handling works correctly");
}

/// Test receipt store functionality
#[tokio::test]
async fn test_receipt_store_operations() {
    let signer = ReceiptSigner::new("store-signer".to_string());
    let mut store = ReceiptStore::new();

    let twin_id1 = TwinId::new(AgentId::new("agent1", "node1"), "twin1");
    let twin_id2 = TwinId::new(AgentId::new("agent2", "node2"), "twin2");
    let requester = AgentId::new("requester", "requester-node");

    // Create multiple receipts
    let operations = vec![
        TwinOperation::Write {
            key: "key1".to_string(),
            value: Bytes::from("value1"),
            timestamp: 12345,
        },
        TwinOperation::Read {
            key: "key1".to_string(),
            timestamp: 12346,
        },
        TwinOperation::Delete {
            key: "key1".to_string(),
            timestamp: 12347,
        },
    ];

    let mut receipts = Vec::new();

    // Create receipts for twin1
    for operation in &operations {
        let receipt = signer.sign_operation(&twin_id1, operation, &requester, true).await.unwrap();
        receipts.push(receipt.clone());
        store.store_receipt(receipt);
    }

    // Create one receipt for twin2
    let receipt2 = signer.sign_operation(&twin_id2, &operations[0], &requester, true).await.unwrap();
    store.store_receipt(receipt2.clone());

    // Test retrieval by ID
    let stored_receipt = store.get_receipt(&receipts[0].receipt_id);
    assert!(stored_receipt.is_some());
    assert_eq!(stored_receipt.unwrap().receipt_id, receipts[0].receipt_id);

    // Test retrieval by twin
    let twin1_receipts = store.get_receipts_for_twin(&twin_id1);
    assert_eq!(twin1_receipts.len(), 3);

    let twin2_receipts = store.get_receipts_for_twin(&twin_id2);
    assert_eq!(twin2_receipts.len(), 1);

    // Test retrieval by operation type
    let write_receipts = store.get_receipts_by_operation("write");
    assert_eq!(write_receipts.len(), 2); // One for each twin

    let read_receipts = store.get_receipts_by_operation("read");
    assert_eq!(read_receipts.len(), 1);

    let delete_receipts = store.get_receipts_by_operation("delete");
    assert_eq!(delete_receipts.len(), 1);

    // Test count
    assert_eq!(store.count(), 4);

    // Test time range retrieval
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
    let recent_receipts = store.get_receipts_in_range(now - 10000, now + 10000);
    assert_eq!(recent_receipts.len(), 4); // All receipts should be recent

    println!("✅ Receipt store operations work correctly");
}

/// Test full integration: twin manager operations with receipt verification
#[tokio::test]
async fn test_full_receipt_integration() {
    let (twin_manager, _signer, verifier, temp_dir) = create_test_twin_manager_with_receipts().await;
    let config = create_test_vault_config(&temp_dir);

    let twin_id = TwinId::new(AgentId::new("integration-agent", "integration-node"), "integration-twin");
    let requester = AgentId::new("requester", "requester-node");

    // Set permissive preferences
    let preferences = TwinPreferences {
        allow_read: true,
        allow_write: true,
        allow_sync: true,
        ..Default::default()
    };
    twin_manager.set_preferences(twin_id.clone(), preferences).await;

    // Create vault
    let _vault = twin_manager.get_twin(twin_id.clone(), config).await.unwrap();

    // Perform operations and collect receipts
    let operations = vec![
        TwinOperation::Write {
            key: "integration_key1".to_string(),
            value: Bytes::from("integration_value1"),
            timestamp: 12345,
        },
        TwinOperation::Write {
            key: "integration_key2".to_string(),
            value: Bytes::from("integration_value2"),
            timestamp: 12346,
        },
        TwinOperation::Read {
            key: "integration_key1".to_string(),
            timestamp: 12347,
        },
        TwinOperation::Increment {
            counter_id: "integration_counter".to_string(),
            amount: 10,
            actor_id: "integration_actor".to_string(),
            timestamp: 12348,
        },
    ];

    let mut receipts = Vec::new();

    for operation in operations {
        let (result, receipt) = twin_manager
            .perform_operation(twin_id.clone(), operation, requester.clone())
            .await
            .unwrap();

        // Verify receipt immediately
        let verification = verifier.verify_receipt(&receipt).unwrap();
        assert!(verification.is_trusted_and_valid(), "Receipt verification failed");

        receipts.push((result, receipt));
    }

    // Verify all receipts
    assert_eq!(receipts.len(), 4);

    // First two operations (writes) should have no result
    assert!(receipts[0].0.is_none());
    assert!(receipts[1].0.is_none());

    // Third operation (read) should have result
    assert!(receipts[2].0.is_some());
    assert_eq!(receipts[2].0.as_ref().unwrap(), &Bytes::from("integration_value1"));

    // Fourth operation (increment) should have no result
    assert!(receipts[3].0.is_none());

    // Verify receipt metadata
    for (i, (_, receipt)) in receipts.iter().enumerate() {
        assert_eq!(receipt.twin_id, twin_id);
        assert_eq!(receipt.requester, requester);
        assert!(receipt.success);
        assert!(!receipt.receipt_id.is_empty());

        // Check metadata
        assert_eq!(receipt.metadata.get("signer_id"), Some(&"test-signer".to_string()));
        assert_eq!(receipt.metadata.get("signature_algorithm"), Some(&"Ed25519".to_string()));

        println!("Receipt {}: {} - {}", i, receipt.operation_type(), receipt.receipt_id);
    }

    println!("✅ Full receipt integration works correctly");
}

/// Test receipt verification under concurrent operations
#[tokio::test]
async fn test_concurrent_receipt_verification() {
    let (twin_manager, _signer, verifier, temp_dir) = create_test_twin_manager_with_receipts().await;
    let config = create_test_vault_config(&temp_dir);

    let twin_id = TwinId::new(AgentId::new("concurrent-agent", "concurrent-node"), "concurrent-twin");

    // Set permissive preferences
    let preferences = TwinPreferences {
        allow_read: true,
        allow_write: true,
        allow_sync: true,
        ..Default::default()
    };
    twin_manager.set_preferences(twin_id.clone(), preferences).await;

    // Create vault
    let _vault = twin_manager.get_twin(twin_id.clone(), config).await.unwrap();

    // Spawn concurrent operations
    let mut handles = Vec::new();

    for i in 0..10 {
        let manager = twin_manager.clone();
        let twin_id = twin_id.clone();
        let requester = AgentId::new(&format!("requester_{}", i), "concurrent-node");

        let handle = tokio::spawn(async move {
            let operation = TwinOperation::Write {
                key: format!("concurrent_key_{}", i),
                value: Bytes::from(format!("concurrent_value_{}", i)),
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64,
            };

            manager.perform_operation(twin_id, operation, requester).await
        });

        handles.push(handle);
    }

    // Collect all results
    let mut all_receipts = Vec::new();

    for handle in handles {
        let result = handle.await.unwrap().unwrap();
        all_receipts.push(result.1); // Extract receipt
    }

    // Verify all receipts concurrently
    let mut verification_handles = Vec::new();

    for receipt in all_receipts {
        let verifier = verifier.clone();
        let handle = tokio::spawn(async move {
            verifier.verify_receipt(&receipt)
        });
        verification_handles.push(handle);
    }

    // Check all verifications
    for (i, handle) in verification_handles.into_iter().enumerate() {
        let verification = handle.await.unwrap().unwrap();
        assert!(verification.is_trusted_and_valid(), "Verification {} failed", i);
    }

    println!("✅ Concurrent receipt verification works correctly");
}

/// Test cross-signer receipt verification
#[tokio::test]
async fn test_cross_signer_verification() {
    let signer1 = ReceiptSigner::new("signer1".to_string());
    let signer2 = ReceiptSigner::new("signer2".to_string());
    let signer3 = ReceiptSigner::new("signer3".to_string());

    let mut verifier = ReceiptVerifier::new();
    verifier.add_trusted_key("signer1".to_string(), signer1.public_key());
    verifier.add_trusted_key("signer2".to_string(), signer2.public_key());
    // Note: signer3 is not trusted

    let twin_id = TwinId::new(AgentId::new("cross-agent", "cross-node"), "cross-twin");
    let requester = AgentId::new("requester", "requester-node");

    let operation = TwinOperation::Write {
        key: "cross_key".to_string(),
        value: Bytes::from("cross_value"),
        timestamp: 12345,
    };

    // Create receipts with different signers
    let receipt1 = signer1.sign_operation(&twin_id, &operation, &requester, true).await.unwrap();
    let receipt2 = signer2.sign_operation(&twin_id, &operation, &requester, true).await.unwrap();
    let receipt3 = signer3.sign_operation(&twin_id, &operation, &requester, true).await.unwrap();

    // Verify receipts
    let verification1 = verifier.verify_receipt(&receipt1).unwrap();
    let verification2 = verifier.verify_receipt(&receipt2).unwrap();
    let verification3 = verifier.verify_receipt(&receipt3).unwrap();

    // Trusted signers should verify
    assert!(verification1.is_trusted_and_valid());
    assert_eq!(verification1.signer_id, Some("signer1".to_string()));

    assert!(verification2.is_trusted_and_valid());
    assert_eq!(verification2.signer_id, Some("signer2".to_string()));

    // Untrusted signer should not verify
    assert!(!verification3.is_trusted_and_valid());
    assert!(!verification3.trusted);
    assert_eq!(verification3.signer_id, Some("signer3".to_string()));

    // Test signer management
    assert!(verifier.is_trusted_signer("signer1"));
    assert!(verifier.is_trusted_signer("signer2"));
    assert!(!verifier.is_trusted_signer("signer3"));

    let trusted_signers = verifier.trusted_signers();
    assert_eq!(trusted_signers.len(), 2);
    assert!(trusted_signers.contains(&"signer1".to_string()));
    assert!(trusted_signers.contains(&"signer2".to_string()));

    println!("✅ Cross-signer verification works correctly");
}

/// Test receipt store with expiration cleanup
#[tokio::test]
async fn test_receipt_store_expiration_cleanup() {
    let signer = ReceiptSigner::new("cleanup-signer".to_string());
    let mut store = ReceiptStore::new();

    let twin_id = TwinId::new(AgentId::new("cleanup-agent", "cleanup-node"), "cleanup-twin");
    let requester = AgentId::new("requester", "requester-node");

    // Create receipts with artificial timestamps
    let old_timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64 - 120000; // 2 minutes ago

    // Create old receipt
    let operation1 = TwinOperation::Write {
        key: "old_key".to_string(),
        value: Bytes::from("old_value"),
        timestamp: old_timestamp,
    };
    let mut old_receipt = signer.sign_operation(&twin_id, &operation1, &requester, true).await.unwrap();
    old_receipt.timestamp = old_timestamp; // Manually set old timestamp
    store.store_receipt(old_receipt);

    // Create recent receipt
    let operation2 = TwinOperation::Write {
        key: "new_key".to_string(),
        value: Bytes::from("new_value"),
        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64,
    };
    let new_receipt = signer.sign_operation(&twin_id, &operation2, &requester, true).await.unwrap();
    store.store_receipt(new_receipt);

    // Verify initial count
    assert_eq!(store.count(), 2);

    // Clean up expired receipts (older than 1 minute)
    let cleaned_count = store.clear_expired(60000);
    assert_eq!(cleaned_count, 1); // Should remove the old receipt

    // Verify final count
    assert_eq!(store.count(), 1);

    // Verify the remaining receipt is the new one
    let remaining_receipts = store.get_receipts_by_operation("write");
    assert_eq!(remaining_receipts.len(), 1);
    assert!(remaining_receipts[0].timestamp > old_timestamp);

    println!("✅ Receipt store expiration cleanup works correctly");
}
