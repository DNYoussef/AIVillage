//! SCION Bridge Compliance Checks
//!
//! Validates SCION gateway integration for proper:
//! - Gateway infrastructure compatibility with AI Village components
//! - SCION protocol path selection and validation
//! - Bridge configuration and security measures
//! - Container orchestration and networking
//! - Integration with navigator agent and dual-path transport

use crate::{LintIssue, SeverityLevel, Result};
use crate::checks::{CheckRule, CheckContext};
use regex::Regex;
use async_trait::async_trait;

/// SCION gateway infrastructure compliance
pub struct ScionGatewayRule;

#[async_trait]
impl CheckRule for ScionGatewayRule {
    fn name(&self) -> &str {
        "scion-gateway-infrastructure"
    }

    fn description(&self) -> &str {
        "Check SCION gateway infrastructure compatibility and configuration"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check SCION-related files
        if !context.file_path.to_string_lossy().contains("scion") &&
           !context.file_path.to_string_lossy().contains("gateway") &&
           !context.content.contains("SCION") &&
           !context.content.contains("Gateway") {
            return Ok(issues);
        }

        // Check for required SCION daemon configuration
        if context.content.contains("scion") || context.content.contains("SCION") {
            let required_scion_configs = [
                ("general.id", "SCION AS identifier"),
                ("dispatcher.socket", "dispatcher socket path"),
                ("path_service.address", "path service endpoint"),
                ("control_service.address", "control service endpoint"),
            ];

            for (config_key, description) in &required_scion_configs {
                if !context.content.contains(config_key) {
                    issues.push(LintIssue::new(
                        "SCION001".to_string(),
                        SeverityLevel::Error,
                        format!("Missing required SCION configuration: {}", description),
                        self.name().to_string(),
                    ));
                }
            }
        }

        // Check gateway bridge configuration
        if context.content.contains("Gateway") && context.content.contains("Bridge") {
            let bridge_requirements = [
                ("listen_address", "gateway listen address"),
                ("upstream_scion_addr", "upstream SCION address"),
                ("max_path_age", "path expiration policy"),
                ("path_selection_policy", "path selection strategy"),
            ];

            for (requirement, description) in &bridge_requirements {
                if !context.content.contains(requirement) {
                    issues.push(LintIssue::new(
                        "SCION002".to_string(),
                        SeverityLevel::Warning,
                        format!("Gateway bridge missing: {}", description),
                        self.name().to_string(),
                    ));
                }
            }
        }

        // Check for hardcoded SCION AS numbers (should be configurable)
        let as_regex = Regex::new(r#""?(?:scion_as|as_id|autonomous_system)"?\s*[:=]\s*"?(\d+)-\d+:\d+:\d+"?"#).unwrap();
        for (line_num, line) in context.content.lines().enumerate() {
            if let Some(captures) = as_regex.captures(line) {
                if let Some(as_prefix) = captures.get(1) {
                    if as_prefix.as_str() == "1" {
                        issues.push(LintIssue::new(
                            "SCION003".to_string(),
                            SeverityLevel::Warning,
                            "Hardcoded test AS number detected - should use production AS ID".to_string(),
                            self.name().to_string(),
                        ).with_location(context.file_path.clone(), line_num + 1, 0));
                    }
                }
            }
        }

        // Check for proper Docker network configuration
        if context.file_path.to_string_lossy().contains("docker") {
            if context.content.contains("network_mode: host") {
                issues.push(LintIssue::new(
                    "SCION004".to_string(),
                    SeverityLevel::Error,
                    "SCION gateway should not use host networking for security isolation".to_string(),
                    self.name().to_string(),
                ));
            }

            if !context.content.contains("networks:") && context.content.contains("scion") {
                issues.push(LintIssue::new(
                    "SCION005".to_string(),
                    SeverityLevel::Error,
                    "SCION container missing network configuration".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        Ok(issues)
    }
}

/// SCION path selection compliance
pub struct ScionPathSelectionRule;

#[async_trait]
impl CheckRule for ScionPathSelectionRule {
    fn name(&self) -> &str {
        "scion-path-selection"
    }

    fn description(&self) -> &str {
        "Check SCION path selection algorithm compliance and security"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check path selection related code
        if !context.content.contains("path") && !context.content.contains("Path") &&
           !context.content.contains("routing") && !context.content.contains("Routing") {
            return Ok(issues);
        }

        // Check for proper path validation
        if context.content.contains("select_path") || context.content.contains("choose_path") {
            let path_validation_checks = [
                ("verify_path_signature", "path signature verification"),
                ("check_path_expiry", "path expiration validation"),
                ("validate_hop_fields", "hop field validation"),
                ("anti_replay_check", "replay attack protection"),
            ];

            for (check_function, description) in &path_validation_checks {
                if !context.content.contains(check_function) {
                    issues.push(LintIssue::new(
                        "SCION006".to_string(),
                        SeverityLevel::Error,
                        format!("Path selection missing: {}", description),
                        self.name().to_string(),
                    ));
                }
            }
        }

        // Check for path diversity requirements
        if context.content.contains("PathSelection") || context.content.contains("path_selection") {
            if !context.content.contains("diversity") && !context.content.contains("disjoint") {
                issues.push(LintIssue::new(
                    "SCION007".to_string(),
                    SeverityLevel::Warning,
                    "Path selection should consider path diversity for resilience".to_string(),
                    self.name().to_string(),
                ));
            }

            // Check for latency vs security tradeoff configuration
            if !context.content.contains("latency_weight") && !context.content.contains("security_weight") {
                issues.push(LintIssue::new(
                    "SCION008".to_string(),
                    SeverityLevel::Warning,
                    "Path selection missing latency vs security weight configuration".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        // Check for path caching implementation
        if context.content.contains("PathCache") || context.content.contains("path_cache") {
            let cache_requirements = [
                ("max_cache_size", "cache size limits"),
                ("cache_ttl", "time-to-live configuration"),
                ("path_invalidation", "cache invalidation logic"),
            ];

            for (requirement, description) in &cache_requirements {
                if !context.content.contains(requirement) {
                    issues.push(LintIssue::new(
                        "SCION009".to_string(),
                        SeverityLevel::Warning,
                        format!("Path cache missing: {}", description),
                        self.name().to_string(),
                    ));
                }
            }
        }

        Ok(issues)
    }
}

/// SCION security compliance
pub struct ScionSecurityRule;

#[async_trait]
impl CheckRule for ScionSecurityRule {
    fn name(&self) -> &str {
        "scion-security"
    }

    fn description(&self) -> &str {
        "Check SCION bridge security measures and authentication"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check security-related SCION code
        if !context.content.contains("scion") && !context.content.contains("SCION") &&
           !context.content.contains("auth") && !context.content.contains("security") {
            return Ok(issues);
        }

        // Check for proper certificate validation
        if context.content.contains("certificate") || context.content.contains("cert") {
            if !context.content.contains("verify_certificate") && !context.content.contains("validate_cert") {
                issues.push(LintIssue::new(
                    "SCION010".to_string(),
                    SeverityLevel::Error,
                    "SCION certificate handling missing validation logic".to_string(),
                    self.name().to_string(),
                ));
            }

            // Check for certificate revocation checking
            if !context.content.contains("revocation") && !context.content.contains("CRL") {
                issues.push(LintIssue::new(
                    "SCION011".to_string(),
                    SeverityLevel::Warning,
                    "SCION certificate validation missing revocation checking".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        // Check for MAC (Message Authentication Code) validation
        if context.content.contains("mac") || context.content.contains("MAC") {
            let mac_requirements = [
                ("verify_mac", "MAC verification"),
                ("mac_key_rotation", "MAC key rotation"),
                ("mac_algorithm", "MAC algorithm specification"),
            ];

            for (requirement, description) in &mac_requirements {
                if !context.content.contains(requirement) {
                    issues.push(LintIssue::new(
                        "SCION012".to_string(),
                        SeverityLevel::Error,
                        format!("SCION MAC handling missing: {}", description),
                        self.name().to_string(),
                    ));
                }
            }
        }

        // Check for proper EPIC (SCION packet-carried forwarding state) validation
        if context.content.contains("EPIC") || context.content.contains("epic") {
            if !context.content.contains("epic_proof") && !context.content.contains("validate_epic") {
                issues.push(LintIssue::new(
                    "SCION013".to_string(),
                    SeverityLevel::Error,
                    "EPIC implementation missing proof validation".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        // Check for DRKey (Dynamically Recreatable Key) implementation
        if context.content.contains("DRKey") || context.content.contains("drkey") {
            let drkey_requirements = [
                ("derive_key", "key derivation"),
                ("key_hierarchy", "key hierarchy management"),
                ("epoch_validation", "epoch-based validation"),
            ];

            for (requirement, description) in &drkey_requirements {
                if !context.content.contains(requirement) {
                    issues.push(LintIssue::new(
                        "SCION014".to_string(),
                        SeverityLevel::Warning,
                        format!("DRKey implementation missing: {}", description),
                        self.name().to_string(),
                    ));
                }
            }
        }

        // Check for rate limiting on SCION paths
        if context.content.contains("rate_limit") || context.content.contains("throttle") {
            if !context.content.contains("per_path_limit") && !context.content.contains("path_based_rate") {
                issues.push(LintIssue::new(
                    "SCION015".to_string(),
                    SeverityLevel::Warning,
                    "Rate limiting should be applied per SCION path for DoS protection".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        Ok(issues)
    }
}

/// SCION integration compliance
pub struct ScionIntegrationRule;

#[async_trait]
impl CheckRule for ScionIntegrationRule {
    fn name(&self) -> &str {
        "scion-integration"
    }

    fn description(&self) -> &str {
        "Check SCION integration with AI Village components"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check integration-related code
        if !context.content.contains("integration") && !context.content.contains("bridge") &&
           !context.content.contains("transport") && !context.content.contains("navigator") {
            return Ok(issues);
        }

        // Check for proper Navigator agent integration
        if context.content.contains("Navigator") || context.content.contains("navigator") {
            if context.content.contains("scion") || context.content.contains("SCION") {
                let navigation_requirements = [
                    ("scion_path_available", "SCION path availability check"),
                    ("fallback_transport", "fallback transport mechanism"),
                    ("path_quality_metrics", "path quality assessment"),
                ];

                for (requirement, description) in &navigation_requirements {
                    if !context.content.contains(requirement) {
                        issues.push(LintIssue::new(
                            "SCION016".to_string(),
                            SeverityLevel::Warning,
                            format!("Navigator-SCION integration missing: {}", description),
                            self.name().to_string(),
                        ));
                    }
                }
            }
        }

        // Check for dual-path transport integration
        if context.content.contains("dual_path") || context.content.contains("DualPath") {
            if !context.content.contains("scion_primary") && !context.content.contains("scion_secondary") {
                issues.push(LintIssue::new(
                    "SCION017".to_string(),
                    SeverityLevel::Warning,
                    "Dual-path transport should consider SCION as primary/secondary option".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        // Check for resource management integration
        if context.content.contains("resource_management") || context.content.contains("ResourceManagement") {
            if context.content.contains("scion") && !context.content.contains("scion_bandwidth_limit") {
                issues.push(LintIssue::new(
                    "SCION018".to_string(),
                    SeverityLevel::Warning,
                    "Resource management missing SCION bandwidth limiting".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        // Check for federation manager compatibility
        if context.content.contains("federation") || context.content.contains("Federation") {
            if context.content.contains("scion") || context.content.contains("SCION") {
                if !context.content.contains("cross_as_communication") {
                    issues.push(LintIssue::new(
                        "SCION019".to_string(),
                        SeverityLevel::Error,
                        "Federation manager missing cross-AS communication support".to_string(),
                        self.name().to_string(),
                    ));
                }
            }
        }

        // Check for monitoring and observability
        if context.content.contains("monitor") || context.content.contains("metrics") {
            if context.content.contains("scion") {
                let monitoring_requirements = [
                    ("path_latency_metric", "path latency monitoring"),
                    ("path_availability_metric", "path availability tracking"),
                    ("gateway_health_check", "gateway health monitoring"),
                ];

                for (requirement, description) in &monitoring_requirements {
                    if !context.content.contains(requirement) {
                        issues.push(LintIssue::new(
                            "SCION020".to_string(),
                            SeverityLevel::Warning,
                            format!("SCION monitoring missing: {}", description),
                            self.name().to_string(),
                        ));
                    }
                }
            }
        }

        Ok(issues)
    }
}

/// SCION container deployment compliance
pub struct ScionDeploymentRule;

#[async_trait]
impl CheckRule for ScionDeploymentRule {
    fn name(&self) -> &str {
        "scion-deployment"
    }

    fn description(&self) -> &str {
        "Check SCION container deployment and orchestration compliance"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check deployment-related files
        if !context.file_path.to_string_lossy().contains("docker") &&
           !context.file_path.to_string_lossy().contains("compose") &&
           !context.file_path.to_string_lossy().contains("k8s") &&
           !context.file_path.to_string_lossy().contains("deploy") {
            return Ok(issues);
        }

        // Check Docker composition files
        if context.content.contains("services:") && context.content.contains("scion") {
            // Check for proper volume mounts
            if !context.content.contains("volumes:") {
                issues.push(LintIssue::new(
                    "SCION021".to_string(),
                    SeverityLevel::Error,
                    "SCION service missing volume configuration for persistent data".to_string(),
                    self.name().to_string(),
                ));
            }

            // Check for resource limits
            if !context.content.contains("mem_limit") && !context.content.contains("cpus") {
                issues.push(LintIssue::new(
                    "SCION022".to_string(),
                    SeverityLevel::Warning,
                    "SCION service missing resource limits".to_string(),
                    self.name().to_string(),
                ));
            }

            // Check for health checks
            if !context.content.contains("healthcheck:") {
                issues.push(LintIssue::new(
                    "SCION023".to_string(),
                    SeverityLevel::Warning,
                    "SCION service missing health check configuration".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        // Check Kubernetes deployment files
        if context.content.contains("kind: Deployment") && context.content.contains("scion") {
            // Check for proper security context
            if !context.content.contains("securityContext:") {
                issues.push(LintIssue::new(
                    "SCION024".to_string(),
                    SeverityLevel::Error,
                    "SCION Kubernetes deployment missing security context".to_string(),
                    self.name().to_string(),
                ));
            }

            // Check for network policies
            if !context.content.contains("NetworkPolicy") {
                issues.push(LintIssue::new(
                    "SCION025".to_string(),
                    SeverityLevel::Warning,
                    "SCION deployment missing network policy for isolation".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        // Check for service mesh integration
        if context.content.contains("istio") || context.content.contains("linkerd") {
            if context.content.contains("scion") && !context.content.contains("ServiceEntry") {
                issues.push(LintIssue::new(
                    "SCION026".to_string(),
                    SeverityLevel::Warning,
                    "SCION service mesh integration missing ServiceEntry configuration".to_string(),
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
            file_path: PathBuf::from("test_scion.rs"),
            content: content.to_string(),
        }
    }

    #[tokio::test]
    async fn test_scion_gateway_rule() {
        let rule = ScionGatewayRule;

        // Test missing SCION configuration
        let content = r#"
            # SCION daemon configuration
            [general]
            # Missing id configuration
            [dispatcher]
            # Missing socket configuration
        "#;
        let context = create_test_context(content);
        let issues = rule.check(&context).await.unwrap();
        assert!(issues.len() > 0);
        assert!(issues.iter().any(|i| i.id == "SCION001"));

        // Test hardcoded test AS number
        let content = r#"
            scion_as = "1-ff00:0:110"
        "#;
        let context = create_test_context(content);
        let issues = rule.check(&context).await.unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].id, "SCION003");
    }

    #[tokio::test]
    async fn test_scion_path_selection_rule() {
        let rule = ScionPathSelectionRule;

        // Test missing path validation
        let content = r#"
            fn select_path(paths: &[Path]) -> Option<Path> {
                // Missing validation checks
                paths.first().cloned()
            }
        "#;
        let context = create_test_context(content);
        let issues = rule.check(&context).await.unwrap();
        assert!(issues.len() > 0);
        assert!(issues.iter().any(|i| i.id == "SCION006"));
    }

    #[tokio::test]
    async fn test_scion_security_rule() {
        let rule = ScionSecurityRule;

        // Test missing certificate validation
        let content = r#"
            struct Certificate {
                data: Vec<u8>,
            }

            fn process_certificate(cert: Certificate) {
                // Missing verify_certificate call
                println!("Processing certificate");
            }
        "#;
        let context = create_test_context(content);
        let issues = rule.check(&context).await.unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].id, "SCION010");
    }

    #[tokio::test]
    async fn test_scion_integration_rule() {
        let rule = ScionIntegrationRule;

        // Test missing Navigator integration
        let content = r#"
            struct Navigator {
                scion_enabled: bool,
            }

            impl Navigator {
                fn select_transport(&self) -> Transport {
                    // Missing scion_path_available check
                    Transport::Default
                }
            }
        "#;
        let context = create_test_context(content);
        let issues = rule.check(&context).await.unwrap();
        assert!(issues.len() > 0);
        assert!(issues.iter().any(|i| i.id == "SCION016"));
    }

    #[tokio::test]
    async fn test_scion_deployment_rule() {
        let rule = ScionDeploymentRule;

        // Test missing Docker configuration
        let content = r#"
            services:
              scion-gateway:
                image: scion/gateway:latest
                # Missing volumes, mem_limit, healthcheck
                ports:
                  - "8080:8080"
        "#;
        let context = CheckContext {
            file_path: PathBuf::from("docker-compose.yml"),
            content: content.to_string(),
        };
        let issues = rule.check(&context).await.unwrap();
        assert!(issues.len() >= 2); // Missing volumes and resource limits
        assert!(issues.iter().any(|i| i.id == "SCION021"));
        assert!(issues.iter().any(|i| i.id == "SCION022"));
    }
}
