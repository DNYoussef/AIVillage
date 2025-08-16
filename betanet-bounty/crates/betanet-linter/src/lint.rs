//! Core linting functionality

use walkdir::WalkDir;
use crate::{LinterConfig, LintResults, LintIssue, SeverityLevel, Result};
use crate::checks::{CheckRule, CheckContext, UnsafeCodeRule, BlockingCallRule, TemplateReuseLintRule, KSDistanceLintRule, EpsilonPolicyLintRule};
use crate::checks::bootstrap::{Argon2idAdvertisementRule, Argon2idParameterRule, CpuPoWFallbackRule, BootstrapNegotiationRule, AbuseTrackingRule};
use crate::checks::tls_mirror::{TlsTemplateCacheRule, TlsSiteClassificationRule, TlsMixtureModelRule, TlsCoverTrafficRule, TlsAntiFingerprintRule};
use crate::checks::noise_xk::{NoiseXkHandshakeRule, NoiseXkKeyRotationRule, NoiseXkFragmentationRule, NoiseXkSecurityRule, NoiseXkTransportRule};
use crate::checks::frame_format::{FrameStructureRule, FrameTypeRule, VarintEncodingRule, FrameBufferRule, FrameValidationRule};
use crate::checks::scion_bridge::{ScionGatewayRule, ScionPathSelectionRule, ScionSecurityRule, ScionIntegrationRule, ScionDeploymentRule};

/// Main linter
pub struct Linter {
    config: LinterConfig,
    rules: Vec<Box<dyn CheckRule>>,
}

impl Linter {
    /// Create new linter
    pub fn new(config: LinterConfig) -> Self {
        let mut rules: Vec<Box<dyn CheckRule>> = vec![];

        // Add all available rules
        rules.push(Box::new(UnsafeCodeRule));
        rules.push(Box::new(BlockingCallRule));
        rules.push(Box::new(TemplateReuseLintRule));
        rules.push(Box::new(KSDistanceLintRule));
        rules.push(Box::new(EpsilonPolicyLintRule));

        // Bootstrap rules
        rules.push(Box::new(Argon2idAdvertisementRule));
        rules.push(Box::new(Argon2idParameterRule));
        rules.push(Box::new(CpuPoWFallbackRule));
        rules.push(Box::new(BootstrapNegotiationRule));
        rules.push(Box::new(AbuseTrackingRule));

        // TLS mirror rules
        rules.push(Box::new(TlsTemplateCacheRule));
        rules.push(Box::new(TlsSiteClassificationRule));
        rules.push(Box::new(TlsMixtureModelRule));
        rules.push(Box::new(TlsCoverTrafficRule));
        rules.push(Box::new(TlsAntiFingerprintRule));

        // Noise XK rules
        rules.push(Box::new(NoiseXkHandshakeRule));
        rules.push(Box::new(NoiseXkKeyRotationRule));
        rules.push(Box::new(NoiseXkFragmentationRule));
        rules.push(Box::new(NoiseXkSecurityRule));
        rules.push(Box::new(NoiseXkTransportRule));

        // Frame format rules
        rules.push(Box::new(FrameStructureRule));
        rules.push(Box::new(FrameTypeRule));
        rules.push(Box::new(VarintEncodingRule));
        rules.push(Box::new(FrameBufferRule));
        rules.push(Box::new(FrameValidationRule));

        // SCION bridge rules
        rules.push(Box::new(ScionGatewayRule));
        rules.push(Box::new(ScionPathSelectionRule));
        rules.push(Box::new(ScionSecurityRule));
        rules.push(Box::new(ScionIntegrationRule));
        rules.push(Box::new(ScionDeploymentRule));

        Self { config, rules }
    }

    /// Run all linting checks
    pub async fn run(&self) -> Result<LintResults> {
        let mut results = LintResults::new();

        // Walk directory and find relevant files
        for entry in WalkDir::new(&self.config.target_dir)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.file_type().is_file() {
                if let Some(ext) = entry.path().extension() {
                    match ext.to_str() {
                        Some("rs") => self.check_rust_file(entry.path(), &mut results).await?,
                        Some("toml") => self.check_toml_file(entry.path(), &mut results).await?,
                        _ => {}
                    }
                }
                results.files_checked += 1;
            }
        }

        Ok(results)
    }

    /// Check specific rule
    pub async fn check_rule(&self, rule_name: &str) -> Result<LintResults> {
        let mut results = LintResults::new();

        // Find the specific rule
        let target_rule = self.rules.iter().find(|rule| rule.name() == rule_name);

        if let Some(rule) = target_rule {
            // Walk directory and apply this specific rule
            for entry in WalkDir::new(&self.config.target_dir)
                .follow_links(true)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                if entry.file_type().is_file() {
                    if let Some(ext) = entry.path().extension() {
                        match ext.to_str() {
                            Some("rs") | Some("toml") => {
                                if let Ok(content) = std::fs::read_to_string(entry.path()) {
                                    let context = CheckContext {
                                        file_path: entry.path().to_path_buf(),
                                        content,
                                    };

                                    let rule_issues = rule.check(&context).await?;
                                    for issue in rule_issues {
                                        results.add_issue(issue);
                                    }
                                    results.rules_executed += 1;
                                }
                            }
                            _ => {}
                        }
                    }
                    results.files_checked += 1;
                }
            }
        } else {
            results.add_issue(LintIssue::new(
                "U001".to_string(),
                SeverityLevel::Warning,
                format!("Unknown rule: {}", rule_name),
                "unknown-rule".to_string(),
            ));
        }

        Ok(results)
    }

    async fn check_rust_file(&self, path: &std::path::Path, results: &mut LintResults) -> Result<()> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::LinterError::Io(e))?;

        let context = CheckContext {
            file_path: path.to_path_buf(),
            content,
        };

        // Run all rules on this file
        for rule in &self.rules {
            let rule_issues = rule.check(&context).await?;
            for issue in rule_issues {
                results.add_issue(issue);
            }
            results.rules_executed += 1;
        }

        Ok(())
    }

    async fn check_toml_file(&self, path: &std::path::Path, results: &mut LintResults) -> Result<()> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::LinterError::Io(e))?;

        let context = CheckContext {
            file_path: path.to_path_buf(),
            content,
        };

        // Run applicable rules on TOML files (fewer rules apply)
        for rule in &self.rules {
            // Only run certain rules on TOML files
            match rule.name() {
                "template-cache-reuse-ratio" | "ks-distance-distribution" | "epsilon-privacy-policy" |
                "argon2id-advertisement" | "argon2id-parameters" | "cpu-pow-fallback" |
                "bootstrap-negotiation" | "abuse-tracking" |
                "tls-template-cache" | "tls-site-classification" | "tls-mixture-model" |
                "tls-cover-traffic" | "tls-anti-fingerprint" |
                "noise-xk-handshake" | "noise-xk-key-rotation" | "noise-xk-fragmentation" |
                "noise-xk-security" | "noise-xk-transport" |
                "frame-structure" | "frame-type" | "varint-encoding" |
                "frame-buffer" | "frame-validation" |
                "scion-gateway-infrastructure" | "scion-path-selection" | "scion-security" |
                "scion-integration" | "scion-deployment" => {
                    let rule_issues = rule.check(&context).await?;
                    for issue in rule_issues {
                        results.add_issue(issue);
                    }
                    results.rules_executed += 1;
                }
                _ => {} // Skip other rules for TOML files
            }
        }

        Ok(())
    }
}
