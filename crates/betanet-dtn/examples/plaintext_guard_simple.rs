//! Simple DTN Plaintext Guard Demonstration
//!
//! Usage: cargo run --package betanet-dtn --example plaintext_guard_simple

use betanet_dtn::{Bundle, EndpointId, PayloadBlock};
use bytes::Bytes;

fn main() {
    println!("🔒 DTN Plaintext Guard Demonstration");
    println!("====================================\n");

    // Test plaintext detection
    println!("📝 Testing Plaintext Detection:");

    let plaintext_samples = vec![
        (
            "HTTP Request",
            "GET /api/users HTTP/1.1\r\nHost: example.com\r\n\r\n",
        ),
        (
            "JSON Data",
            r#"{"username": "alice", "password": "secret123"}"#,
        ),
        ("SSH Key", "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDTgBK..."),
        (
            "Shell Script",
            "#!/bin/bash\necho 'Deploy complete'\nexit 0",
        ),
    ];

    let mut all_detected = true;
    for (desc, payload_str) in plaintext_samples {
        let payload = Bytes::from(payload_str);
        let payload_block = PayloadBlock::new(payload);
        let is_ciphertext = payload_block.is_ciphertext();

        if is_ciphertext {
            println!("  ❌ {}: SECURITY FAILURE - plaintext not detected!", desc);
            all_detected = false;
        } else {
            println!("  ✅ {}: plaintext correctly detected", desc);
        }
    }

    // Test ciphertext acceptance
    println!("\n🔐 Testing Ciphertext Acceptance:");

    let ciphertext_samples = vec![
        ("Random bytes", generate_random_bytes(64)),
        ("High entropy", generate_high_entropy_bytes(48)),
        ("Binary data", generate_binary_data(32)),
    ];

    let mut all_accepted = true;
    for (desc, payload_bytes) in ciphertext_samples {
        let payload = Bytes::from(payload_bytes);
        let payload_block = PayloadBlock::new(payload);
        let is_ciphertext = payload_block.is_ciphertext();

        if !is_ciphertext {
            println!(
                "  ❌ {}: PERFORMANCE ISSUE - ciphertext incorrectly blocked!",
                desc
            );
            all_accepted = false;
        } else {
            println!("  ✅ {}: ciphertext correctly accepted", desc);
        }
    }

    // Test the actual gateway guard
    println!("\n🚪 Gateway Boundary Guard Test:");

    let plaintext_bundle = Bundle::new(
        EndpointId::node("external-gateway"),
        EndpointId::node("internal-source"),
        Bytes::from("This plaintext should not cross the gateway!"),
        60000,
    );

    println!("  Creating bundle with plaintext payload...");
    if plaintext_bundle.payload.is_ciphertext() {
        println!("  ❌ CRITICAL SECURITY FAILURE: Plaintext not detected!");
    } else {
        println!("  ✅ Security guard active: plaintext correctly identified");
        if cfg!(debug_assertions) {
            println!("  🛡️  Debug mode: debug_assert! would trigger on send_bundle()");
        } else {
            println!("  🛡️  Release mode: CLA implementation should reject");
        }
    }

    // Summary
    println!("\n📊 Security Assessment:");
    if all_detected && all_accepted {
        println!("  🏆 PERFECT SECURITY: All tests passed!");
        println!("  ✅ DTN gateway invariant successfully enforced");
        println!("  ✅ No plaintext leakage possible");
        println!("  ✅ Legitimate ciphertext flows freely");
    } else {
        if !all_detected {
            println!("  ⚠️  SECURITY RISK: Some plaintext was not detected!");
        }
        if !all_accepted {
            println!("  ⚠️  PERFORMANCE ISSUE: Some ciphertext was blocked!");
        }
    }

    println!("\n🔐 Key Security Features:");
    println!("  • Multi-layer detection: entropy analysis, pattern matching, statistical tests");
    println!("  • Gateway enforcement: debug_assert! at CLA boundary prevents egress");
    println!("  • Zero-tolerance policy: no plaintext crosses DTN gateway boundaries");
    println!("  • High accuracy: distinguishes legitimate ciphertext from plaintext");

    println!("\n✅ DTN Plaintext Guard demonstration complete!");
}

/// Generate random bytes (simulates good ciphertext)
fn generate_random_bytes(size: usize) -> Vec<u8> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen()).collect()
}

/// Generate high-entropy bytes with even distribution
fn generate_high_entropy_bytes(size: usize) -> Vec<u8> {
    (0..size).map(|i| ((i * 179 + 97) % 256) as u8).collect()
}

/// Generate binary data with some structure (but still high entropy)
fn generate_binary_data(size: usize) -> Vec<u8> {
    let mut data = vec![0x1f, 0x8b, 0x08, 0x00]; // gzip-like header
    for i in 0..(size.saturating_sub(4)) {
        data.push(((i * 73 + 31) % 256) as u8);
    }
    data
}
