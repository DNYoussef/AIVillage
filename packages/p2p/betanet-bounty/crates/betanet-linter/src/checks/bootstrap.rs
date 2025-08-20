//! Bootstrap PoW linting rules
//!
//! Ensures proper Argon2id PoW implementation and parameter validation

use crate::checks::{CheckContext, CheckRule};
use crate::{LintIssue, Result, SeverityLevel};
use async_trait::async_trait;
use once_cell::sync::Lazy;
use regex::Regex;

static MEMORY_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"Mobile.*memory_kb:\s*(\d+)").unwrap());

static TARGET_TIME_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"Mobile.*target_time_ms:\s*(\d+)").unwrap());

static RATE_LIMIT_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"rate_limit.*:\s*(\d+)").unwrap());

/// Check that Argon2id is properly advertised and configured
pub struct Argon2idAdvertisementRule;

#[async_trait]
impl CheckRule for Argon2idAdvertisementRule {
    fn name(&self) -> &str {
        "argon2id-advertisement"
    }

    fn description(&self) -> &str {
        "Ensure Argon2id PoW is advertised and properly configured"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check bootstrap-related files
        if !context.file_path.to_string_lossy().contains("bootstrap")
            && !context.content.contains("PoW")
            && !context.content.contains("argon2")
        {
            return Ok(issues);
        }

        // Check if Argon2id is properly advertised
        let has_argon2id_enum =
            context.content.contains("supports_argon2id") || context.content.contains("Argon2id");

        if context.content.contains("PoW") && !has_argon2id_enum {
            issues.push(LintIssue::new(
                "BS001".to_string(),
                SeverityLevel::Error,
                "PoW implementation without Argon2id support advertisement".to_string(),
                self.name().to_string(),
            ));
        }

        // Check for proper device class awareness
        if context.content.contains("DeviceClass") {
            let device_classes = ["Mobile", "Desktop", "Server", "Embedded"];
            let mut found_classes = 0;

            for class in &device_classes {
                if context.content.contains(class) {
                    found_classes += 1;
                }
            }

            if found_classes < 3 {
                issues.push(LintIssue::new(
                    "BS002".to_string(),
                    SeverityLevel::Warning,
                    format!(
                        "DeviceClass implementation missing key device types (found {}/4)",
                        found_classes
                    ),
                    self.name().to_string(),
                ));
            }
        }

        // Check for user agent parsing
        if context.content.contains("DeviceClass") && !context.content.contains("user_agent") {
            issues.push(LintIssue::new(
                "BS003".to_string(),
                SeverityLevel::Warning,
                "DeviceClass without user agent detection capability".to_string(),
                self.name().to_string(),
            ));
        }

        Ok(issues)
    }
}

/// Check Argon2id parameters are sane for different device classes
pub struct Argon2idParameterRule;

#[async_trait]
impl CheckRule for Argon2idParameterRule {
    fn name(&self) -> &str {
        "argon2id-parameters"
    }

    fn description(&self) -> &str {
        "Validate Argon2id parameters are appropriate for device classes"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check files with Argon2 parameters
        if !context.content.contains("Argon2Params")
            && !context.content.contains("memory_kb")
            && !context.content.contains("iterations")
        {
            return Ok(issues);
        }

        // Check mobile parameters (should be <300ms target)
        if context.content.contains("Mobile") {
            self.check_mobile_params(context, &mut issues);
        }

        // Check server parameters (should be 10x+ harder than mobile)
        if context.content.contains("Server") {
            self.check_server_params(context, &mut issues);
        }

        // Check for proper parameter validation
        if context.content.contains("Argon2Params") && !context.content.contains("validate") {
            issues.push(LintIssue::new(
                "BS004".to_string(),
                SeverityLevel::Error,
                "Argon2Params without validation function".to_string(),
                self.name().to_string(),
            ));
        }

        // Check for abuse factor scaling
        if context.content.contains("Argon2Params") && !context.content.contains("scale_for_abuse")
        {
            issues.push(LintIssue::new(
                "BS005".to_string(),
                SeverityLevel::Warning,
                "Argon2Params without abuse factor scaling".to_string(),
                self.name().to_string(),
            ));
        }

        Ok(issues)
    }
}

impl Argon2idParameterRule {
    fn check_mobile_params(&self, context: &CheckContext, issues: &mut Vec<LintIssue>) {
        // Check mobile memory usage (should be reasonable for mobile devices)
        for (line_num, line) in context.content.lines().enumerate() {
            if let Some(captures) = MEMORY_REGEX.captures(line) {
                if let Some(memory_str) = captures.get(1) {
                    if let Ok(memory_kb) = memory_str.as_str().parse::<u32>() {
                        if memory_kb > 32768 {
                            // > 32 MB for mobile
                            issues.push(LintIssue::new(
                                "BS006".to_string(),
                                SeverityLevel::Error,
                                format!("Mobile Argon2id memory usage {}KB exceeds recommended 32MB limit", memory_kb),
                                self.name().to_string(),
                            ).with_location(context.file_path.clone(), line_num + 1, 0));
                        }
                    }
                }
            }
        }

        // Check mobile target time (should be <300ms)
        for (line_num, line) in context.content.lines().enumerate() {
            if let Some(captures) = TARGET_TIME_REGEX.captures(line) {
                if let Some(time_str) = captures.get(1) {
                    if let Ok(time_ms) = time_str.as_str().parse::<u32>() {
                        if time_ms >= 300 {
                            issues.push(
                                LintIssue::new(
                                    "BS007".to_string(),
                                    SeverityLevel::Error,
                                    format!(
                                        "Mobile target time {}ms exceeds 300ms requirement",
                                        time_ms
                                    ),
                                    self.name().to_string(),
                                )
                                .with_location(
                                    context.file_path.clone(),
                                    line_num + 1,
                                    0,
                                ),
                            );
                        }
                    }
                }
            }
        }
    }

    fn check_server_params(&self, context: &CheckContext, issues: &mut Vec<LintIssue>) {
        // Extract mobile and server memory params for comparison
        let mobile_memory = self.extract_device_memory(context, "Mobile");
        let server_memory = self.extract_device_memory(context, "Server");

        if let (Some(mobile_kb), Some(server_kb)) = (mobile_memory, server_memory) {
            let ratio = server_kb as f64 / mobile_kb as f64;
            if ratio < 10.0 {
                issues.push(LintIssue::new(
                    "BS008".to_string(),
                    SeverityLevel::Error,
                    format!("Server PoW ({}KB) is only {:.1}x harder than mobile ({}KB), should be 10x+",
                           server_kb, ratio, mobile_kb),
                    self.name().to_string(),
                ));
            }
        }

        // Check server iterations vs mobile
        let mobile_iterations = self.extract_device_iterations(context, "Mobile");
        let server_iterations = self.extract_device_iterations(context, "Server");

        if let (Some(mobile_iter), Some(server_iter)) = (mobile_iterations, server_iterations) {
            let ratio = server_iter as f64 / mobile_iter as f64;
            if ratio < 2.0 {
                issues.push(LintIssue::new(
                    "BS009".to_string(),
                    SeverityLevel::Warning,
                    format!(
                        "Server iterations ({}) should be significantly higher than mobile ({})",
                        server_iter, mobile_iter
                    ),
                    self.name().to_string(),
                ));
            }
        }
    }

    fn extract_device_memory(&self, context: &CheckContext, device: &str) -> Option<u32> {
        let regex_pattern = format!(r"{}.*memory_kb:\s*(\d+)", device);
        let regex = Regex::new(&regex_pattern).ok()?;

        for line in context.content.lines() {
            if let Some(captures) = regex.captures(line) {
                if let Some(memory_str) = captures.get(1) {
                    return memory_str.as_str().parse::<u32>().ok();
                }
            }
        }
        None
    }

    fn extract_device_iterations(&self, context: &CheckContext, device: &str) -> Option<u32> {
        let regex_pattern = format!(r"{}.*iterations:\s*(\d+)", device);
        let regex = Regex::new(&regex_pattern).ok()?;

        for line in context.content.lines() {
            if let Some(captures) = regex.captures(line) {
                if let Some(iter_str) = captures.get(1) {
                    return iter_str.as_str().parse::<u32>().ok();
                }
            }
        }
        None
    }
}

/// Check for proper CPU PoW fallback with rate limiting
pub struct CpuPoWFallbackRule;

#[async_trait]
impl CheckRule for CpuPoWFallbackRule {
    fn name(&self) -> &str {
        "cpu-pow-fallback"
    }

    fn description(&self) -> &str {
        "Ensure proper CPU PoW fallback with strict rate limiting"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Check if this file implements PoW
        if !context.content.contains("PoW")
            && !context.content.contains("CpuPoW")
            && !context.content.contains("bootstrap")
        {
            return Ok(issues);
        }

        // Check for CPU PoW fallback implementation
        if context.content.contains("supports_argon2id") && !context.content.contains("CpuPoW") {
            issues.push(LintIssue::new(
                "BS010".to_string(),
                SeverityLevel::Error,
                "Argon2id implementation without CPU PoW fallback".to_string(),
                self.name().to_string(),
            ));
        }

        // Check for rate limiting in CPU PoW
        if context.content.contains("CpuPoW") {
            if !context.content.contains("rate_limit") {
                issues.push(LintIssue::new(
                    "BS011".to_string(),
                    SeverityLevel::Error,
                    "CPU PoW implementation without rate limiting".to_string(),
                    self.name().to_string(),
                ));
            }

            // Check for reasonable rate limits
            for (line_num, line) in context.content.lines().enumerate() {
                if let Some(captures) = RATE_LIMIT_REGEX.captures(line) {
                    if let Some(rate_str) = captures.get(1) {
                        if let Ok(rate) = rate_str.as_str().parse::<u32>() {
                            if rate > 100 {
                                issues.push(LintIssue::new(
                                    "BS012".to_string(),
                                    SeverityLevel::Warning,
                                    format!("CPU PoW rate limit {} is too high for abuse prevention", rate),
                                    self.name().to_string(),
                                ).with_location(context.file_path.clone(), line_num + 1, 0));
                            }
                        }
                    }
                }
            }
        }

        Ok(issues)
    }
}

/// Check for proper negotiation protocol implementation
pub struct BootstrapNegotiationRule;

#[async_trait]
impl CheckRule for BootstrapNegotiationRule {
    fn name(&self) -> &str {
        "bootstrap-negotiation"
    }

    fn description(&self) -> &str {
        "Validate bootstrap negotiation protocol implementation"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check bootstrap negotiation files
        if !context.content.contains("BootstrapMessage")
            && !context.content.contains("ClientHello")
            && !context.content.contains("negotiation")
        {
            return Ok(issues);
        }

        // Check for complete negotiation message types
        let required_messages = ["ClientHello", "Challenge", "Solution", "Result"];

        let mut found_messages = 0;
        for message in &required_messages {
            if context.content.contains(message) {
                found_messages += 1;
            }
        }

        if found_messages < required_messages.len() {
            issues.push(LintIssue::new(
                "BS013".to_string(),
                SeverityLevel::Error,
                format!(
                    "Incomplete bootstrap negotiation protocol (found {}/{} message types)",
                    found_messages,
                    required_messages.len()
                ),
                self.name().to_string(),
            ));
        }

        // Check for capability announcement
        if context.content.contains("ClientHello") && !context.content.contains("supports_argon2id")
        {
            issues.push(LintIssue::new(
                "BS014".to_string(),
                SeverityLevel::Error,
                "ClientHello without Argon2id capability announcement".to_string(),
                self.name().to_string(),
            ));
        }

        // Check for challenge expiry
        if context.content.contains("Challenge") && !context.content.contains("expires_at") {
            issues.push(LintIssue::new(
                "BS015".to_string(),
                SeverityLevel::Warning,
                "PoW Challenge without expiry mechanism".to_string(),
                self.name().to_string(),
            ));
        }

        Ok(issues)
    }
}

/// Check for abuse tracking and progressive difficulty scaling
pub struct AbuseTrackingRule;

#[async_trait]
impl CheckRule for AbuseTrackingRule {
    fn name(&self) -> &str {
        "abuse-tracking"
    }

    fn description(&self) -> &str {
        "Ensure proper abuse tracking and progressive difficulty scaling"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check abuse tracking related code
        if !context.content.contains("AbuseTracker")
            && !context.content.contains("abuse_factor")
            && !context.content.contains("scale_for_abuse")
        {
            return Ok(issues);
        }

        // Check for abuse tracking in bootstrap manager
        if context.content.contains("BootstrapManager") && !context.content.contains("AbuseTracker")
        {
            issues.push(LintIssue::new(
                "BS016".to_string(),
                SeverityLevel::Error,
                "BootstrapManager without abuse tracking capability".to_string(),
                self.name().to_string(),
            ));
        }

        // Check for failure/success tracking
        if context.content.contains("AbuseTracker") {
            if !context.content.contains("record_failure") {
                issues.push(LintIssue::new(
                    "BS017".to_string(),
                    SeverityLevel::Error,
                    "AbuseTracker without failure recording".to_string(),
                    self.name().to_string(),
                ));
            }

            if !context.content.contains("record_success") {
                issues.push(LintIssue::new(
                    "BS018".to_string(),
                    SeverityLevel::Error,
                    "AbuseTracker without success recording".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        // Check for progressive scaling
        if context.content.contains("abuse_factor") && !context.content.contains("scale_for_abuse")
        {
            issues.push(LintIssue::new(
                "BS019".to_string(),
                SeverityLevel::Warning,
                "Abuse factor calculation without parameter scaling".to_string(),
                self.name().to_string(),
            ));
        }

        // Check for cleanup mechanism
        if context.content.contains("AbuseTracker") && !context.content.contains("cleanup") {
            issues.push(LintIssue::new(
                "BS020".to_string(),
                SeverityLevel::Warning,
                "AbuseTracker without cleanup mechanism for old entries".to_string(),
                self.name().to_string(),
            ));
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
            file_path: PathBuf::from("test_bootstrap.rs"),
            content: content.to_string(),
        }
    }

    #[tokio::test]
    async fn test_argon2id_advertisement_rule() {
        let rule = Argon2idAdvertisementRule;

        // Test missing Argon2id support
        let context = create_test_context("PoW implementation without argon2");
        let issues = rule.check(&context).await.unwrap();
        assert!(!issues.is_empty());
        assert_eq!(issues[0].id, "BS001");

        // Test proper implementation
        let context = create_test_context("PoW with supports_argon2id: true");
        let issues = rule.check(&context).await.unwrap();
        assert!(issues.iter().all(|i| i.id != "BS001"));
    }

    #[tokio::test]
    async fn test_argon2id_parameter_rule() {
        let rule = Argon2idParameterRule;

        // Test mobile parameters too high
        let context = create_test_context(
            r#"
            Mobile => Self {
                memory_kb: 65536,  // Too high for mobile
                target_time_ms: 500, // Too high for mobile
            }
        "#,
        );
        let issues = rule.check(&context).await.unwrap();
        assert!(issues.iter().any(|i| i.id == "BS006"));
        assert!(issues.iter().any(|i| i.id == "BS007"));
    }

    #[tokio::test]
    async fn test_cpu_pow_fallback_rule() {
        let rule = CpuPoWFallbackRule;

        // Test missing CPU fallback
        let context = create_test_context("supports_argon2id but no CpuPoW");
        let issues = rule.check(&context).await.unwrap();
        assert!(issues.iter().any(|i| i.id == "BS010"));

        // Test missing rate limiting
        let context = create_test_context("CpuPoW without rate_limit");
        let issues = rule.check(&context).await.unwrap();
        assert!(issues.iter().any(|i| i.id == "BS011"));
    }

    #[tokio::test]
    async fn test_bootstrap_negotiation_rule() {
        let rule = BootstrapNegotiationRule;

        // Test incomplete negotiation
        let context = create_test_context("ClientHello and Challenge only");
        let issues = rule.check(&context).await.unwrap();
        assert!(issues.iter().any(|i| i.id == "BS013"));

        // Test missing capability announcement
        let context = create_test_context("ClientHello without supports_argon2id");
        let issues = rule.check(&context).await.unwrap();
        assert!(issues.iter().any(|i| i.id == "BS014"));
    }

    #[tokio::test]
    async fn test_abuse_tracking_rule() {
        let rule = AbuseTrackingRule;

        // Test missing abuse tracker
        let context = create_test_context("BootstrapManager without AbuseTracker");
        let issues = rule.check(&context).await.unwrap();
        assert!(issues.iter().any(|i| i.id == "BS016"));

        // Test incomplete tracking
        let context = create_test_context("AbuseTracker without record_failure");
        let issues = rule.check(&context).await.unwrap();
        assert!(issues.iter().any(|i| i.id == "BS017"));
    }
}
