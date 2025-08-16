//! Noise XK Protocol Compliance Checks
//!
//! Validates Noise XK implementation for proper:
//! - Handshake pattern compliance (e,es -> e,ee -> s,se)
//! - Key rotation thresholds and rate limiting
//! - Cryptographic parameters (X25519, ChaCha20-Poly1305, Blake2s)
//! - Security measures (nonce handling, error states)
//! - Fragmentation and MTU discovery
//! - Transport state management

use crate::{LintIssue, SeverityLevel, Result};
use crate::checks::{CheckRule, CheckContext};
use regex::Regex;

/// Noise XK handshake pattern compliance
pub struct NoiseXkHandshakeRule;

impl CheckRule for NoiseXkHandshakeRule {
    fn name(&self) -> &str {
        "noise-xk-handshake"
    }

    fn description(&self) -> &str {
        "Check Noise XK handshake pattern implementation for proper message sequence"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check Noise-related files
        if !context.file_path.to_string_lossy().contains("noise") &&
           !context.content.contains("NoiseXK") &&
           !context.content.contains("Noise_XK") {
            return Ok(issues);
        }

        // Check for correct Noise pattern string
        if context.content.contains("Noise_XK") {
            let pattern_regex = Regex::new(r#"Noise_XK_(\w+)_(\w+)_(\w+)"#).unwrap();
            for (line_num, line) in context.content.lines().enumerate() {
                if let Some(captures) = pattern_regex.captures(line) {
                    let key_exchange = captures.get(1).map_or("", |m| m.as_str());
                    let cipher = captures.get(2).map_or("", |m| m.as_str());
                    let hash = captures.get(3).map_or("", |m| m.as_str());

                    // Validate cryptographic primitives
                    if key_exchange != "25519" {
                        issues.push(LintIssue::new(
                            "NOISE001".to_string(),
                            SeverityLevel::Error,
                            format!("Insecure key exchange '{}' - should use '25519' (X25519)", key_exchange),
                            self.name().to_string(),
                        ).with_location(context.file_path.clone(), line_num + 1, 0));
                    }

                    if cipher != "ChaChaPoly" {
                        issues.push(LintIssue::new(
                            "NOISE002".to_string(),
                            SeverityLevel::Error,
                            format!("Insecure cipher '{}' - should use 'ChaChaPoly' (ChaCha20-Poly1305)", cipher),
                            self.name().to_string(),
                        ).with_location(context.file_path.clone(), line_num + 1, 0));
                    }

                    if hash != "BLAKE2s" && hash != "SHA256" {
                        issues.push(LintIssue::new(
                            "NOISE003".to_string(),
                            SeverityLevel::Warning,
                            format!("Hash function '{}' - recommend 'BLAKE2s' or 'SHA256'", hash),
                            self.name().to_string(),
                        ).with_location(context.file_path.clone(), line_num + 1, 0));
                    }
                }
            }
        }

        // Check for proper handshake state management
        if context.content.contains("HandshakePhase") {
            let required_phases = ["Uninitialized", "Message1", "Message2", "Message3", "Transport", "Failed"];
            let mut missing_phases = Vec::new();

            for &phase in &required_phases {
                if !context.content.contains(phase) {
                    missing_phases.push(phase);
                }
            }

            if !missing_phases.is_empty() {
                issues.push(LintIssue::new(
                    "NOISE004".to_string(),
                    SeverityLevel::Error,
                    format!("Missing handshake phases: {}", missing_phases.join(", ")),
                    self.name().to_string(),
                ));
            }
        }

        // Check for proper XK message sequence validation
        let message_sequence_checks = [
            ("create_message_1", "is_initiator", "Only initiator can send message 1"),
            ("create_message_2", "!self.is_initiator", "Only responder can send message 2"),
            ("create_message_3", "is_initiator", "Only initiator can send message 3"),
            ("process_message_2", "is_initiator", "Only initiator can process message 2"),
            ("process_message_3", "!self.is_initiator", "Only responder can process message 3"),
        ];

        for (method, condition, _description) in &message_sequence_checks {
            if context.content.contains(method) {
                // Check if proper role validation exists
                let method_start = context.content.find(method);
                if let Some(start) = method_start {
                    let method_section = &context.content[start..start + 500.min(context.content.len() - start)];
                    if !method_section.contains(condition) {
                        issues.push(LintIssue::new(
                            "NOISE005".to_string(),
                            SeverityLevel::Error,
                            format!("Method '{}' missing role validation ({})", method, condition),
                            self.name().to_string(),
                        ));
                    }
                }
            }
        }

        // Check for handshake state transition validation
        if context.content.contains("phase") && context.content.contains("HandshakePhase") {
            let state_checks = [
                ("create_message_1", "Uninitialized"),
                ("create_message_2", "Message1"),
                ("create_message_3", "Message2"),
                ("process_message_3", "Message2"),
            ];

            for (method, expected_state) in &state_checks {
                if context.content.contains(method) {
                    let method_start = context.content.find(method);
                    if let Some(start) = method_start {
                        let method_section = &context.content[start..start + 800.min(context.content.len() - start)];
                        if !method_section.contains(&format!("phase.*{}", expected_state)) &&
                           !method_section.contains(&format!("{}.*phase", expected_state)) {
                            issues.push(LintIssue::new(
                                "NOISE006".to_string(),
                                SeverityLevel::Warning,
                                format!("Method '{}' should validate phase is '{}'", method, expected_state),
                                self.name().to_string(),
                            ));
                        }
                    }
                }
            }
        }

        Ok(issues)
    }
}

/// Noise XK key rotation compliance
pub struct NoiseXkKeyRotationRule;

impl CheckRule for NoiseXkKeyRotationRule {
    fn name(&self) -> &str {
        "noise-xk-key-rotation"
    }

    fn description(&self) -> &str {
        "Check Noise XK key rotation thresholds and rate limiting implementation"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check key rotation related files
        if !context.content.contains("KeyRotation") &&
           !context.content.contains("rekey") &&
           !context.content.contains("KEY_UPDATE") {
            return Ok(issues);
        }

        // Check rekey thresholds
        let threshold_checks = [
            ("REKEY_BYTES_THRESHOLD", "8 * 1024 * 1024 * 1024", "8 GiB"),
            ("REKEY_FRAMES_THRESHOLD", "65_536", "65,536 frames"),
            ("REKEY_TIME_THRESHOLD", "3600", "1 hour"),
        ];

        for (constant, expected_value, description) in &threshold_checks {
            if context.content.contains(constant) {
                let threshold_regex = Regex::new(&format!(r"{}\s*:\s*\w+\s*=\s*([^;]+)", constant)).unwrap();
                for (line_num, line) in context.content.lines().enumerate() {
                    if let Some(captures) = threshold_regex.captures(line) {
                        if let Some(value) = captures.get(1) {
                            let value_str = value.as_str().trim();
                            if !value_str.contains(expected_value) {
                                issues.push(LintIssue::new(
                                    "NOISE007".to_string(),
                                    SeverityLevel::Warning,
                                    format!("{} value '{}' differs from spec recommendation ({})", constant, value_str, description),
                                    self.name().to_string(),
                                ).with_location(context.file_path.clone(), line_num + 1, 0));
                            }
                        }
                    }
                }
            }
        }

        // Check KEY_UPDATE rate limiting parameters
        let rate_limit_checks = [
            ("KEY_UPDATE_MIN_INTERVAL_SECS", 30, "minimum 30 seconds"),
            ("KEY_UPDATE_MIN_INTERVAL_FRAMES", 4096, "minimum 4096 frames"),
            ("KEY_UPDATE_ACCEPT_WINDOW_SECS", 2, "2 second window"),
            ("KEY_UPDATE_TOKEN_BUCKET_SIZE", 10, "burst size 10"),
        ];

        for (constant, expected_min, description) in &rate_limit_checks {
            if context.content.contains(constant) {
                let const_regex = Regex::new(&format!(r"{}\s*:\s*\w+\s*=\s*(\d+)", constant)).unwrap();
                for (line_num, line) in context.content.lines().enumerate() {
                    if let Some(captures) = const_regex.captures(line) {
                        if let Some(value_str) = captures.get(1) {
                            if let Ok(value) = value_str.as_str().parse::<u32>() {
                                if (value as i32) < *expected_min {
                                    issues.push(LintIssue::new(
                                        "NOISE008".to_string(),
                                        SeverityLevel::Warning,
                                        format!("{} value {} below recommended minimum ({})", constant, value, description),
                                        self.name().to_string(),
                                    ).with_location(context.file_path.clone(), line_num + 1, 0));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Check for proper rate limiting implementation
        if context.content.contains("can_initiate_key_update") {
            let required_checks = [
                "KEY_UPDATE_MIN_INTERVAL_SECS",
                "KEY_UPDATE_MIN_INTERVAL_FRAMES",
                "token_bucket",
            ];

            for check in &required_checks {
                if !context.content.contains(check) {
                    issues.push(LintIssue::new(
                        "NOISE009".to_string(),
                        SeverityLevel::Error,
                        format!("KEY_UPDATE rate limiting missing '{}' check", check),
                        self.name().to_string(),
                    ));
                }
            }
        }

        // Check for sliding window rate limiting
        if context.content.contains("accept_window") || context.content.contains("should_accept") {
            if !context.content.contains("recent_updates") || !context.content.contains("VecDeque") {
                issues.push(LintIssue::new(
                    "NOISE010".to_string(),
                    SeverityLevel::Error,
                    "Sliding window rate limiting implementation missing proper window tracking".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        Ok(issues)
    }
}

/// Noise XK fragmentation compliance
pub struct NoiseXkFragmentationRule;

impl CheckRule for NoiseXkFragmentationRule {
    fn name(&self) -> &str {
        "noise-xk-fragmentation"
    }

    fn description(&self) -> &str {
        "Check Noise XK handshake fragmentation for MTU resilience"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check fragmentation related files
        if !context.content.contains("fragment") &&
           !context.content.contains("HANDSHAKE_FRAGMENT_SIZE") &&
           !context.content.contains("HandshakeFragment") {
            return Ok(issues);
        }

        // Check fragment size configuration
        let fragment_size_regex = Regex::new(r"HANDSHAKE_FRAGMENT_SIZE\s*:\s*usize\s*=\s*(\d+)").unwrap();
        for (line_num, line) in context.content.lines().enumerate() {
            if let Some(captures) = fragment_size_regex.captures(line) {
                if let Some(size_str) = captures.get(1) {
                    if let Ok(size) = size_str.as_str().parse::<u32>() {
                        if size < 576 {
                            issues.push(LintIssue::new(
                                "NOISE011".to_string(),
                                SeverityLevel::Error,
                                format!("Fragment size {} below IPv4 minimum MTU (576)", size),
                                self.name().to_string(),
                            ).with_location(context.file_path.clone(), line_num + 1, 0));
                        }

                        if size > 1500 {
                            issues.push(LintIssue::new(
                                "NOISE012".to_string(),
                                SeverityLevel::Warning,
                                format!("Fragment size {} above Ethernet MTU (1500) - may cause fragmentation", size),
                                self.name().to_string(),
                            ).with_location(context.file_path.clone(), line_num + 1, 0));
                        }

                        // Recommend conservative size for wide compatibility
                        if size > 1200 {
                            issues.push(LintIssue::new(
                                "NOISE013".to_string(),
                                SeverityLevel::Info,
                                format!("Fragment size {} - consider 1200 bytes for wider compatibility", size),
                                self.name().to_string(),
                            ).with_location(context.file_path.clone(), line_num + 1, 0));
                        }
                    }
                }
            }
        }

        // Check for proper fragment structure
        if context.content.contains("HandshakeFragment") {
            let required_fields = ["fragment_id", "total_fragments", "fragment_index", "data"];
            let mut missing_fields = Vec::new();

            for &field in &required_fields {
                if !context.content.contains(field) {
                    missing_fields.push(field);
                }
            }

            if !missing_fields.is_empty() {
                issues.push(LintIssue::new(
                    "NOISE014".to_string(),
                    SeverityLevel::Error,
                    format!("HandshakeFragment missing required fields: {}", missing_fields.join(", ")),
                    self.name().to_string(),
                ));
            }
        }

        // Check for reassembly logic
        if context.content.contains("HandshakeReassembler") {
            let required_methods = ["fragment_message", "add_fragment"];
            let mut missing_methods = Vec::new();

            for &method in &required_methods {
                if !context.content.contains(method) {
                    missing_methods.push(method);
                }
            }

            if !missing_methods.is_empty() {
                issues.push(LintIssue::new(
                    "NOISE015".to_string(),
                    SeverityLevel::Error,
                    format!("HandshakeReassembler missing required methods: {}", missing_methods.join(", ")),
                    self.name().to_string(),
                ));
            }

            // Check for proper fragment bounds checking
            if context.content.contains("add_fragment") && !context.content.contains("fragment_index as usize") {
                issues.push(LintIssue::new(
                    "NOISE016".to_string(),
                    SeverityLevel::Warning,
                    "Fragment reassembly should validate fragment_index bounds".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        Ok(issues)
    }
}

/// Noise XK security compliance
pub struct NoiseXkSecurityRule;

impl CheckRule for NoiseXkSecurityRule {
    fn name(&self) -> &str {
        "noise-xk-security"
    }

    fn description(&self) -> &str {
        "Check Noise XK security measures and error handling"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check security-related Noise files
        if !context.content.contains("NoiseXK") &&
           !context.content.contains("encrypt") &&
           !context.content.contains("decrypt") {
            return Ok(issues);
        }

        // Check key length validation
        if context.content.contains("InvalidKeyLength") {
            if !context.content.contains("key.len() != 32") {
                issues.push(LintIssue::new(
                    "NOISE017".to_string(),
                    SeverityLevel::Error,
                    "Missing proper X25519 key length validation (32 bytes)".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        // Check for message size limits
        if context.content.contains("MAX_MESSAGE_SIZE") {
            let max_size_regex = Regex::new(r"MAX_MESSAGE_SIZE\s*:\s*usize\s*=\s*(\d+)").unwrap();
            for (line_num, line) in context.content.lines().enumerate() {
                if let Some(captures) = max_size_regex.captures(line) {
                    if let Some(size_str) = captures.get(1) {
                        if let Ok(size) = size_str.as_str().parse::<u32>() {
                            if size > 65535 {
                                issues.push(LintIssue::new(
                                    "NOISE018".to_string(),
                                    SeverityLevel::Warning,
                                    format!("MAX_MESSAGE_SIZE {} exceeds Noise protocol limit (65535)", size),
                                    self.name().to_string(),
                                ).with_location(context.file_path.clone(), line_num + 1, 0));
                            }

                            if size < 1024 {
                                issues.push(LintIssue::new(
                                    "NOISE019".to_string(),
                                    SeverityLevel::Warning,
                                    format!("MAX_MESSAGE_SIZE {} too small for practical use", size),
                                    self.name().to_string(),
                                ).with_location(context.file_path.clone(), line_num + 1, 0));
                            }
                        }
                    }
                }
            }
        }

        // Check for proper error handling in encryption/decryption
        if context.content.contains("fn encrypt") {
            let encrypt_section = if let Some(start) = context.content.find("fn encrypt") {
                &context.content[start..start + 1000.min(context.content.len() - start)]
            } else {
                ""
            };

            if !encrypt_section.contains("HandshakeNotComplete") {
                issues.push(LintIssue::new(
                    "NOISE020".to_string(),
                    SeverityLevel::Error,
                    "encrypt() missing handshake completion check".to_string(),
                    self.name().to_string(),
                ));
            }

            if !encrypt_section.contains("MessageTooLarge") {
                issues.push(LintIssue::new(
                    "NOISE021".to_string(),
                    SeverityLevel::Error,
                    "encrypt() missing message size validation".to_string(),
                    self.name().to_string(),
                ));
            }

            if !encrypt_section.contains("should_rekey") {
                issues.push(LintIssue::new(
                    "NOISE022".to_string(),
                    SeverityLevel::Warning,
                    "encrypt() should check rekey requirements".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        // Check for secure random number generation
        if context.content.contains("rand") || context.content.contains("random") {
            if context.content.contains("thread_rng") && !context.content.contains("OsRng") {
                issues.push(LintIssue::new(
                    "NOISE023".to_string(),
                    SeverityLevel::Critical,
                    "Using thread_rng instead of cryptographically secure OsRng".to_string(),
                    self.name().to_string(),
                ));
            }

            if context.content.contains("rand::random") {
                issues.push(LintIssue::new(
                    "NOISE024".to_string(),
                    SeverityLevel::Critical,
                    "Using rand::random() instead of cryptographically secure OsRng".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        // Check for nonce overflow protection
        if context.content.contains("nonce") || context.content.contains("NonceOverflow") {
            if !context.content.contains("overflow") && context.content.contains("encrypt") {
                issues.push(LintIssue::new(
                    "NOISE025".to_string(),
                    SeverityLevel::Error,
                    "Missing nonce overflow detection and handling".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        // Check for proper state cleanup on errors
        if context.content.contains("reset") {
            let reset_section = if let Some(start) = context.content.find("fn reset") {
                &context.content[start..start + 500.min(context.content.len() - start)]
            } else {
                ""
            };

            if !reset_section.contains("handshake = None") || !reset_section.contains("transport = None") {
                issues.push(LintIssue::new(
                    "NOISE026".to_string(),
                    SeverityLevel::Warning,
                    "reset() should clear all cryptographic state (handshake, transport)".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        Ok(issues)
    }
}

/// Noise XK transport state compliance
pub struct NoiseXkTransportRule;

impl CheckRule for NoiseXkTransportRule {
    fn name(&self) -> &str {
        "noise-xk-transport"
    }

    fn description(&self) -> &str {
        "Check Noise XK transport state management and lifecycle"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check transport-related files
        if !context.content.contains("transport") &&
           !context.content.contains("TransportState") &&
           !context.content.contains("into_transport_mode") {
            return Ok(issues);
        }

        // Check for proper transport state initialization
        if context.content.contains("into_transport_mode") {
            // Should only be called after message 3 processing
            let transport_context = if let Some(start) = context.content.find("into_transport_mode") {
                let start_context = start.saturating_sub(300);
                let end_context = (start + 300).min(context.content.len());
                &context.content[start_context..end_context]
            } else {
                ""
            };

            if !transport_context.contains("Message3") && !transport_context.contains("message_3") {
                issues.push(LintIssue::new(
                    "NOISE027".to_string(),
                    SeverityLevel::Warning,
                    "into_transport_mode() should only be called after message 3 processing".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        // Check transport readiness validation
        if context.content.contains("is_transport_ready") {
            let ready_check = if let Some(start) = context.content.find("is_transport_ready") {
                &context.content[start..start + 200.min(context.content.len() - start)]
            } else {
                ""
            };

            if !ready_check.contains("Transport") {
                issues.push(LintIssue::new(
                    "NOISE028".to_string(),
                    SeverityLevel::Error,
                    "is_transport_ready() should check for HandshakePhase::Transport".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        // Check that transport operations validate handshake completion
        let transport_methods = ["encrypt", "decrypt", "initiate_key_update"];
        for method in &transport_methods {
            if context.content.contains(&format!("fn {}", method)) {
                let method_section = if let Some(start) = context.content.find(&format!("fn {}", method)) {
                    &context.content[start..start + 400.min(context.content.len() - start)]
                } else {
                    ""
                };

                if !method_section.contains("is_transport_ready") && !method_section.contains("HandshakeNotComplete") {
                    issues.push(LintIssue::new(
                        "NOISE029".to_string(),
                        SeverityLevel::Error,
                        format!("{}() should validate handshake completion before operation", method),
                        self.name().to_string(),
                    ));
                }
            }
        }

        // Check for proper frame counting
        if context.content.contains("frames_sent") || context.content.contains("frames_received") {
            if context.content.contains("encrypt") && !context.content.contains("record_frame_sent") {
                issues.push(LintIssue::new(
                    "NOISE030".to_string(),
                    SeverityLevel::Warning,
                    "encrypt() should record frame count for rekey tracking".to_string(),
                    self.name().to_string(),
                ));
            }

            if context.content.contains("decrypt") && !context.content.contains("record_frame_received") {
                issues.push(LintIssue::new(
                    "NOISE031".to_string(),
                    SeverityLevel::Warning,
                    "decrypt() should record frame count for rekey tracking".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        // Check for proper byte counting
        if context.content.contains("bytes_sent") || context.content.contains("bytes_received") {
            if context.content.contains("encrypt") && !context.content.contains("bytes_sent +=") {
                issues.push(LintIssue::new(
                    "NOISE032".to_string(),
                    SeverityLevel::Warning,
                    "encrypt() should update bytes_sent counter for rekey tracking".to_string(),
                    self.name().to_string(),
                ));
            }

            if context.content.contains("decrypt") && !context.content.contains("bytes_received +=") {
                issues.push(LintIssue::new(
                    "NOISE033".to_string(),
                    SeverityLevel::Warning,
                    "decrypt() should update bytes_received counter for rekey tracking".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        Ok(issues)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn create_test_context(content: &str) -> CheckContext {
        CheckContext {
            file_path: PathBuf::from("test_noise.rs"),
            content: content.to_string(),
        }
    }

    #[tokio::test]
    async fn test_handshake_rule() {
        let rule = NoiseXkHandshakeRule;

        // Test correct pattern
        let content = r#"
            const NOISE_PATTERN: &str = "Noise_XK_25519_ChaChaPoly_BLAKE2s";
        "#;
        let context = create_test_context(content);
        let issues = rule.check(&context).await.unwrap();
        assert_eq!(issues.len(), 0);

        // Test incorrect cipher
        let content = r#"
            const NOISE_PATTERN: &str = "Noise_XK_25519_AES256_BLAKE2s";
        "#;
        let context = create_test_context(content);
        let issues = rule.check(&context).await.unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].id, "NOISE002");
    }

    #[tokio::test]
    async fn test_key_rotation_rule() {
        let rule = NoiseXkKeyRotationRule;

        // Test incorrect threshold
        let content = r#"
            const REKEY_BYTES_THRESHOLD: u64 = 1024 * 1024; // Too small
        "#;
        let context = create_test_context(content);
        let issues = rule.check(&context).await.unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].id, "NOISE007");
    }

    #[tokio::test]
    async fn test_security_rule() {
        let rule = NoiseXkSecurityRule;

        // Test insecure random generation
        let content = r#"
            let mut rng = rand::thread_rng();
            let random_bytes = rand::random::<[u8; 32]>();
        "#;
        let context = create_test_context(content);
        let issues = rule.check(&context).await.unwrap();
        assert!(issues.len() >= 2); // Should flag both issues
        assert!(issues.iter().any(|i| i.severity == SeverityLevel::Critical));
    }

    #[tokio::test]
    async fn test_fragmentation_rule() {
        let rule = NoiseXkFragmentationRule;

        // Test fragment size too small
        let content = r#"
            const HANDSHAKE_FRAGMENT_SIZE: usize = 400; // Too small
        "#;
        let context = create_test_context(content);
        let issues = rule.check(&context).await.unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].id, "NOISE011");
    }

    #[tokio::test]
    async fn test_transport_rule() {
        let rule = NoiseXkTransportRule;

        // Test missing transport readiness check
        let content = r#"
            pub fn encrypt(&mut self, plaintext: &[u8]) -> Result<Bytes, NoiseError> {
                // Missing is_transport_ready() check
                let transport = self.transport.as_mut().unwrap();
                transport.write_message(plaintext, &mut buffer)
            }
        "#;
        let context = create_test_context(content);
        let issues = rule.check(&context).await.unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].id, "NOISE029");
    }
}
