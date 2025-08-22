//! Core linting functionality

use crate::checks::bootstrap::{
    AbuseTrackingRule, Argon2idAdvertisementRule, Argon2idParameterRule, BootstrapNegotiationRule,
    CpuPoWFallbackRule,
};
use crate::checks::frame_format::{
    FrameBufferRule, FrameStructureRule, FrameTypeRule, FrameValidationRule, VarintEncodingRule,
};
use crate::checks::noise_xk::{
    NoiseXkFragmentationRule, NoiseXkHandshakeRule, NoiseXkKeyRotationRule, NoiseXkSecurityRule,
    NoiseXkTransportRule,
};
use crate::checks::scion_bridge::{
    ScionDeploymentRule, ScionGatewayRule, ScionIntegrationRule, ScionPathSelectionRule,
    ScionSecurityRule,
};
use crate::checks::tls_mirror::{
use crate::checks::compliance::Section11ComplianceChecker;
    TlsAntiFingerprintRule, TlsCoverTrafficRule, TlsMixtureModelRule, TlsSiteClassificationRule,
    TlsTemplateCacheRule,
};
use crate::checks::{
    BlockingCallRule, CheckContext, CheckRule, CryptographyBestPracticesRule,
    DependencySecurityRule, EpsilonPolicyLintRule, KSDistanceLintRule, TemplateReuseLintRule,
    UnsafeCodeRule,
};
use crate::{LintIssue, LintResults, LinterConfig, Result, SeverityLevel};
use walkdir::WalkDir;

/// Main linter
pub struct Linter {
    config: LinterConfig,
    rules: Vec<Box<dyn CheckRule>>,
}

impl Linter {
    /// Create new linter
    pub fn new(config: LinterConfig) -> Self {
        let rules: Vec<Box<dyn CheckRule>> = vec![
            // Add all available rules
            Box::new(UnsafeCodeRule),
            Box::new(BlockingCallRule),
            Box::new(TemplateReuseLintRule),
            Box::new(KSDistanceLintRule),
            Box::new(EpsilonPolicyLintRule),
            Box::new(DependencySecurityRule),
            Box::new(CryptographyBestPracticesRule),
            // Bootstrap rules
            Box::new(Argon2idAdvertisementRule),
            Box::new(Argon2idParameterRule),
            Box::new(CpuPoWFallbackRule),
            Box::new(BootstrapNegotiationRule),
            Box::new(AbuseTrackingRule),
            // TLS mirror rules
            Box::new(TlsTemplateCacheRule),
            Box::new(TlsSiteClassificationRule),
            Box::new(TlsMixtureModelRule),
            Box::new(TlsCoverTrafficRule),
            Box::new(TlsAntiFingerprintRule),
            // Noise XK rules
            Box::new(NoiseXkHandshakeRule),
            Box::new(NoiseXkKeyRotationRule),
            Box::new(NoiseXkFragmentationRule),
            Box::new(NoiseXkSecurityRule),
            Box::new(NoiseXkTransportRule),
            // Frame format rules
            Box::new(FrameStructureRule),
            Box::new(FrameTypeRule),
            Box::new(VarintEncodingRule),
            Box::new(FrameBufferRule),
            Box::new(FrameValidationRule),
            // SCION bridge rules
            Box::new(ScionGatewayRule),
            Box::new(ScionPathSelectionRule),
            Box::new(ScionSecurityRule),
            Box::new(ScionIntegrationRule),
            Box::new(ScionDeploymentRule),
        ];

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

    async fn check_rust_file(
        &self,
        path: &std::path::Path,
        results: &mut LintResults,
    ) -> Result<()> {
        let content = std::fs::read_to_string(path).map_err(crate::LinterError::Io)?;

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

    async fn check_toml_file(
        &self,
        path: &std::path::Path,
        results: &mut LintResults,
    ) -> Result<()> {
        let content = std::fs::read_to_string(path).map_err(crate::LinterError::Io)?;

        let context = CheckContext {
            file_path: path.to_path_buf(),
            content,
        };

        // Run applicable rules on TOML files (fewer rules apply)
        for rule in &self.rules {
            // Only run certain rules on TOML files
            match rule.name() {
                "template-cache-reuse-ratio"
                | "ks-distance-distribution"
                | "epsilon-privacy-policy"
                | "dependency-security"
                | "cryptography-best-practices"
                | "argon2id-advertisement"
                | "argon2id-parameters"
                | "cpu-pow-fallback"
                | "bootstrap-negotiation"
                | "abuse-tracking"
                | "tls-template-cache"
                | "tls-site-classification"
                | "tls-mixture-model"
                | "tls-cover-traffic"
                | "tls-anti-fingerprint"
                | "noise-xk-handshake"
                | "noise-xk-key-rotation"
                | "noise-xk-fragmentation"
                | "noise-xk-security"
                | "noise-xk-transport"
                | "frame-structure"
                | "frame-type"
                | "varint-encoding"
                | "frame-buffer"
                | "frame-validation"
                | "scion-gateway-infrastructure"
                | "scion-path-selection"
                | "scion-security"
                | "scion-integration"
                | "scion-deployment" => {
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
