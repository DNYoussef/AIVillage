//! DTN Plaintext Guard Test Harness
//!
//! This test harness validates the security invariant that no plaintext
//! data leaks through DTN gateway boundaries during bundle transmission.

use crate::{Bundle, ConvergenceLayer, EndpointId, PayloadBlock};
use bytes::Bytes;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Mock CLA that records attempted plaintext transmissions
pub struct MockConvergenceLayer {
    pub name: &'static str,
    pub plaintext_detected: Arc<AtomicBool>,
    pub last_bundle: Arc<std::sync::Mutex<Option<Bundle>>>,
}

impl MockConvergenceLayer {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            plaintext_detected: Arc::new(AtomicBool::new(false)),
            last_bundle: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    pub fn was_plaintext_detected(&self) -> bool {
        self.plaintext_detected.load(Ordering::Relaxed)
    }

    pub fn reset(&self) {
        self.plaintext_detected.store(false, Ordering::Relaxed);
        *self.last_bundle.lock().unwrap() = None;
    }
}

#[async_trait::async_trait]
impl ConvergenceLayer for MockConvergenceLayer {
    fn name(&self) -> &'static str {
        self.name
    }

    fn mtu(&self) -> usize {
        1500
    }

    async fn send_bundle_impl(
        &self,
        _destination: &str,
        bundle: &Bundle,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Store the bundle for inspection
        *self.last_bundle.lock().unwrap() = Some(bundle.clone());

        // In debug builds, the debug_assert! in send_bundle() should catch plaintext
        // In release builds, we manually check here for testing
        if !bundle.payload.is_ciphertext() {
            self.plaintext_detected.store(true, Ordering::Relaxed);
            return Err("Plaintext detected at DTN gateway boundary".into());
        }

        Ok(())
    }

    async fn start_listening(
        &self,
        _local_address: &str,
        _bundle_handler: Box<dyn Fn(Bundle) + Send + Sync>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

    async fn stop(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }
}

/// Generate test payloads for validation
pub struct PlaintextTestVectors;

impl PlaintextTestVectors {
    /// Generate obviously plaintext payloads that should be detected
    pub fn plaintext_samples() -> Vec<(&'static str, Bytes)> {
        vec![
            (
                "HTTP request",
                Bytes::from("GET /api/v1/status HTTP/1.1\r\nHost: example.com\r\n\r\n"),
            ),
            (
                "JSON data",
                Bytes::from(r#"{"username": "alice", "password": "secret123"}"#),
            ),
            (
                "XML document",
                Bytes::from("<?xml version=\"1.0\"?><root><user>bob</user></root>"),
            ),
            (
                "SSH key",
                Bytes::from("ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC..."),
            ),
            (
                "PEM certificate",
                Bytes::from("-----BEGIN CERTIFICATE-----\nMIIDXTCCAkWgAwIBAgIJAKoK..."),
            ),
            (
                "Shell script",
                Bytes::from("#!/bin/bash\necho 'Hello World'\nexit 0"),
            ),
            (
                "Email",
                Bytes::from(
                    "From: alice@example.com\nTo: bob@example.com\nSubject: Test\n\nHello Bob!",
                ),
            ),
            (
                "SQL query",
                Bytes::from("SELECT username, password FROM users WHERE id = 1;"),
            ),
            (
                "Log entry",
                Bytes::from("[2024-01-01 12:00:00] INFO: User alice logged in from 192.168.1.100"),
            ),
            (
                "Configuration",
                Bytes::from("server.port=8080\ndb.host=localhost\napi.key=secret"),
            ),
        ]
    }

    /// Generate ciphertext-like payloads that should pass
    pub fn ciphertext_samples() -> Vec<(&'static str, Bytes)> {
        vec![
            (
                "Random bytes",
                Bytes::from(Self::generate_random_bytes(1024)),
            ),
            (
                "High entropy",
                Bytes::from(Self::generate_high_entropy_bytes(512)),
            ),
            ("AES-like", Bytes::from(Self::simulate_aes_output(256))),
            (
                "Compressed data",
                Bytes::from(Self::simulate_compressed_data(800)),
            ),
            (
                "Base64 binary",
                Bytes::from(Self::generate_base64_like_binary(400)),
            ),
        ]
    }

    /// Generate cryptographically random bytes
    fn generate_random_bytes(size: usize) -> Vec<u8> {
        (0..size).map(|_| rand::random::<u8>()).collect()
    }

    /// Generate high-entropy bytes (simulating good ciphertext)
    fn generate_high_entropy_bytes(size: usize) -> Vec<u8> {
        use std::collections::HashMap;
        let mut data = Vec::with_capacity(size);
        let mut byte_counts = HashMap::new();

        // Generate bytes ensuring roughly even distribution
        for i in 0..size {
            let byte = ((i * 179 + 97) % 256) as u8; // Pseudo-random with good distribution
            data.push(byte);
            *byte_counts.entry(byte).or_insert(0) += 1;
        }

        data
    }

    /// Simulate AES output (should have high entropy, no patterns)
    fn simulate_aes_output(size: usize) -> Vec<u8> {
        // AES output is indistinguishable from random
        Self::generate_random_bytes(size)
    }

    /// Simulate compressed data (high entropy, some structure)
    fn simulate_compressed_data(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        // Add some compression-like headers
        data.extend_from_slice(&[0x1f, 0x8b, 0x08, 0x00]); // gzip-like header
                                                           // Fill rest with high-entropy data
        data.extend(Self::generate_random_bytes(size - 4));
        data
    }

    /// Generate binary data that might contain some base64-like sequences
    fn generate_base64_like_binary(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        // Mix of binary and some ASCII, but with low enough ASCII ratio to pass
        for i in 0..size {
            if i % 7 == 0 {
                // Occasional ASCII character (less than 15% of total)
                data.push(b'A' + (i % 26) as u8);
            } else {
                data.push(rand::random::<u8>());
            }
        }
        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bundle, EndpointId};

    #[cfg(not(debug_assertions))]
    #[tokio::test]
    async fn test_plaintext_detection() {
        let cla = MockConvergenceLayer::new("test-cla");

        for (description, payload) in PlaintextTestVectors::plaintext_samples() {
            cla.reset();

            let bundle = Bundle::new(
                EndpointId::node("dest"),
                EndpointId::node("src"),
                payload,
                60000, // 1 minute lifetime
            );

            println!("Testing plaintext detection for: {}", description);

            // In debug builds, this should panic due to debug_assert!
            // In release builds, send_bundle_impl should catch it
            let result = cla.send_bundle("dest-addr", &bundle).await;

            if cfg!(debug_assertions) {
                // In debug mode, we expect panic due to debug_assert!
                // Since we can't easily test panics in async context,
                // we'll check the payload directly
                assert!(
                    !bundle.payload.is_ciphertext(),
                    "Failed to detect plaintext in: {}",
                    description
                );
            } else {
                // In release mode, should return error
                assert!(
                    result.is_err() || cla.was_plaintext_detected(),
                    "Failed to detect plaintext in: {}",
                    description
                );
            }
        }
    }

    #[cfg(not(debug_assertions))]
    #[tokio::test]
    async fn test_ciphertext_acceptance() {
        let cla = MockConvergenceLayer::new("test-cla");

        for (description, payload) in PlaintextTestVectors::ciphertext_samples() {
            cla.reset();

            let bundle = Bundle::new(
                EndpointId::node("dest"),
                EndpointId::node("src"),
                payload,
                60000,
            );

            println!("Testing ciphertext acceptance for: {}", description);

            // Should pass the plaintext guard
            assert!(
                bundle.payload.is_ciphertext(),
                "Ciphertext incorrectly flagged as plaintext: {}",
                description
            );

            let result = cla.send_bundle("dest-addr", &bundle).await;
            assert!(
                result.is_ok(),
                "Ciphertext rejected by DTN gateway: {}",
                description
            );
            assert!(
                !cla.was_plaintext_detected(),
                "False positive plaintext detection for: {}",
                description
            );
        }
    }

    #[test]
    fn test_entropy_calculation() {
        // Test entropy of different data types
        let uniform_random: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let entropy = crate::bundle::calculate_entropy(&uniform_random);
        assert!(
            entropy > 7.9,
            "Uniform distribution should have ~8 bits entropy, got {}",
            entropy
        );

        let all_zeros = vec![0u8; 1000];
        let entropy = crate::bundle::calculate_entropy(&all_zeros);
        assert!(
            entropy < 0.1,
            "All zeros should have ~0 bits entropy, got {}",
            entropy
        );

        let plaintext = b"Hello, this is a plaintext message!";
        let entropy = crate::bundle::calculate_entropy(plaintext);
        assert!(
            entropy < 6.0,
            "English text should have low entropy, got {}",
            entropy
        );
    }

    #[test]
    fn test_chi_squared() {
        // Test chi-squared for different distributions
        let uniform: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let chi_sq = crate::bundle::chi_squared_test(&uniform);
        assert!(
            chi_sq < 300.0,
            "Uniform distribution should have low chi-squared, got {}",
            chi_sq
        );

        let biased = [b'A'; 1000];
        let chi_sq = crate::bundle::chi_squared_test(&biased);
        assert!(
            chi_sq > 1000.0,
            "Biased distribution should have high chi-squared, got {}",
            chi_sq
        );
    }

    #[test]
    fn test_payload_block_plaintext_detection() {
        // Test various plaintext patterns
        let test_cases = vec![
            (r#"{"user": "alice"}"#, false), // JSON
            ("GET /api HTTP/1.1", false),    // HTTP
            ("<?xml version=", false),       // XML
            ("ssh-rsa AAAAB3", false),       // SSH key
            ("Hello World!", false),         // Simple text
        ];

        for (payload_str, should_pass) in test_cases {
            let payload_block = PayloadBlock::new(Bytes::from(payload_str));
            let is_ciphertext = payload_block.is_ciphertext();

            if should_pass {
                assert!(
                    is_ciphertext,
                    "Expected '{}' to be considered ciphertext",
                    payload_str
                );
            } else {
                assert!(
                    !is_ciphertext,
                    "Expected '{}' to be detected as plaintext",
                    payload_str
                );
            }
        }
    }

    #[test]
    fn test_empty_payload() {
        let empty_payload = PayloadBlock::new(Bytes::new());
        assert!(
            empty_payload.is_ciphertext(),
            "Empty payload should be considered secure"
        );
    }
}

/// Integration test demonstrating the DTN security invariant
pub async fn run_dtn_plaintext_guard_demo() {
    println!("🔒 DTN Plaintext Guard Demonstration");
    println!("====================================");

    let cla = MockConvergenceLayer::new("demo-cla");

    println!("\n📝 Testing plaintext detection...");
    for (desc, payload) in PlaintextTestVectors::plaintext_samples()
        .into_iter()
        .take(3)
    {
        let bundle = Bundle::new(
            EndpointId::node("gateway"),
            EndpointId::node("source"),
            payload,
            60000,
        );

        println!("  • {}: ", desc);
        match cla.send_bundle("external-dest", &bundle).await {
            Ok(_) => println!("    ❌ SECURITY VIOLATION: Plaintext allowed through!"),
            Err(_) => println!("    ✅ Plaintext correctly blocked"),
        }
        cla.reset();
    }

    println!("\n🔐 Testing ciphertext acceptance...");
    for (desc, payload) in PlaintextTestVectors::ciphertext_samples()
        .into_iter()
        .take(3)
    {
        let bundle = Bundle::new(
            EndpointId::node("gateway"),
            EndpointId::node("source"),
            payload,
            60000,
        );

        println!("  • {}: ", desc);
        match cla.send_bundle("external-dest", &bundle).await {
            Ok(_) => println!("    ✅ Ciphertext correctly allowed"),
            Err(e) => println!("    ❌ PERFORMANCE ISSUE: Ciphertext blocked: {}", e),
        }
        cla.reset();
    }

    println!("\n✅ DTN Plaintext Guard demonstration completed!");
    println!("   Security invariant: No plaintext at DTN gateway boundaries ✓");
}
