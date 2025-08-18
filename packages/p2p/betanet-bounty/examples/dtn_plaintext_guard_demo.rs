//! DTN Plaintext Guard Demonstration
//!
//! This example demonstrates the DTN security invariant that prevents
//! plaintext data from leaking through gateway boundaries.
//!
//! Usage: cargo run --example dtn_plaintext_guard_demo

use bytes::Bytes;
use betanet_dtn::{Bundle, EndpointId, PayloadBlock};

fn main() {
    println!("🔒 DTN Plaintext Guard Demonstration");
    println!("====================================\n");

    // Test various payload types
    let test_cases = vec![
        // Plaintext samples (should be detected)
        ("HTTP Request", "GET /api/users HTTP/1.1\r\nHost: example.com\r\n\r\n", false),
        ("JSON Data", r#"{"username": "alice", "password": "secret123", "role": "admin"}"#, false),
        ("XML Config", r#"<?xml version="1.0"?><config><database>mysql://localhost</database></config>"#, false),
        ("Email Message", "From: alice@corp.com\nTo: bob@corp.com\nSubject: Quarterly Report\n\nAttached is the Q4 financial data.", false),
        ("SSH Public Key", "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDTgBK...", false),
        ("Shell Script", "#!/bin/bash\necho 'Deploying to production'\nsudo systemctl restart nginx", false),

        // Ciphertext-like samples (should pass)
        ("Random Binary", &generate_random_hex(64), true),
        ("High Entropy Data", &generate_high_entropy_hex(48), true),
        ("Compressed-like", &generate_compressed_like_hex(72), true),
    ];

    println!("Testing {} payload samples...\n", test_cases.len());

    let mut plaintext_detected = 0;
    let mut ciphertext_passed = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;

    for (description, payload_str, should_be_ciphertext) in test_cases {
        let payload = if payload_str.starts_with("\\x") {
            // Handle hex-encoded payloads
            Bytes::from(hex_decode(payload_str))
        } else {
            Bytes::from(payload_str.as_bytes())
        };

        let payload_block = PayloadBlock::new(payload.clone());
        let is_detected_as_ciphertext = payload_block.is_ciphertext();

        let status = if should_be_ciphertext == is_detected_as_ciphertext {
            if should_be_ciphertext {
                ciphertext_passed += 1;
                "✅ PASS (Ciphertext allowed)"
            } else {
                plaintext_detected += 1;
                "✅ PASS (Plaintext blocked)"
            }
        } else if should_be_ciphertext {
            false_positives += 1;
            "❌ FALSE POSITIVE (Ciphertext blocked)"
        } else {
            false_negatives += 1;
            "❌ FALSE NEGATIVE (Plaintext allowed)"
        };

        println!("📦 {:<20} | {}", description, status);

        // Show payload details for interesting cases
        if payload.len() <= 100 {
            let preview = if payload_str.len() > 60 {
                format!("{}...", &payload_str[..60])
            } else {
                payload_str.to_string()
            };
            println!("   Preview: {}", preview);
        }

        // Show security metrics
        if !should_be_ciphertext || !is_detected_as_ciphertext {
            let entropy = betanet_dtn::bundle::calculate_entropy(&payload);
            let chi_squared = betanet_dtn::bundle::chi_squared_test(&payload);
            let printable_ratio = payload.iter()
                .filter(|&&b| b.is_ascii_graphic() || b == b' ')
                .count() as f64 / payload.len() as f64;

            println!("   Metrics: entropy={:.2}, chi²={:.1}, printable={:.1}%",
                entropy, chi_squared, printable_ratio * 100.0);
        }

        println!();
    }

    // Summary
    println!("📊 Test Results Summary:");
    println!("========================");
    println!("   Plaintext correctly detected: {} ✅", plaintext_detected);
    println!("   Ciphertext correctly allowed: {} ✅", ciphertext_passed);
    println!("   False positives (ciphertext blocked): {} ❌", false_positives);
    println!("   False negatives (plaintext allowed): {} ❌", false_negatives);

    let total = plaintext_detected + ciphertext_passed + false_positives + false_negatives;
    let accuracy = (plaintext_detected + ciphertext_passed) as f64 / total as f64 * 100.0;

    println!("\n🎯 Overall Accuracy: {:.1}%", accuracy);

    if false_positives == 0 && false_negatives == 0 {
        println!("🏆 PERFECT SECURITY: No plaintext leakage detected!");
        println!("✅ DTN gateway invariant successfully enforced");
    } else if false_negatives > 0 {
        println!("⚠️  SECURITY RISK: {} plaintext samples were not detected", false_negatives);
    } else {
        println!("⚠️  PERFORMANCE ISSUE: {} legitimate ciphertext samples were blocked", false_positives);
    }

    println!("\n🔐 DTN Plaintext Guard Demonstration Complete!");
}

/// Generate random hex string for testing
fn generate_random_hex(bytes: usize) -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let random_bytes: Vec<u8> = (0..bytes).map(|_| rng.gen()).collect();
    format!("\\x{}", hex::encode(random_bytes))
}

/// Generate high-entropy hex string (simulating good ciphertext)
fn generate_high_entropy_hex(bytes: usize) -> String {
    let high_entropy_bytes: Vec<u8> = (0..bytes)
        .map(|i| ((i * 179 + 97) % 256) as u8)
        .collect();
    format!("\\x{}", hex::encode(high_entropy_bytes))
}

/// Generate compressed-like hex string
fn generate_compressed_like_hex(bytes: usize) -> String {
    let mut data = vec![0x1f, 0x8b, 0x08, 0x00]; // gzip-like header
    let random_bytes: Vec<u8> = (0..bytes-4).map(|i| ((i * 73 + 31) % 256) as u8).collect();
    data.extend(random_bytes);
    format!("\\x{}", hex::encode(data))
}

/// Simple hex decoder for test data
fn hex_decode(hex_str: &str) -> Vec<u8> {
    if hex_str.starts_with("\\x") {
        hex::decode(&hex_str[2..]).unwrap_or_default()
    } else {
        hex::decode(hex_str).unwrap_or_default()
    }
}
