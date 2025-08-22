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
    TlsAntiFingerprintRule, TlsCoverTrafficRule, TlsMixtureModelRule, TlsSiteClassificationRule,
    TlsTemplateCacheRule,
};
use crate::checks::compliance::Section11ComplianceChecker;
use crate::checks::{
    BlockingCallRule, CheckContext, CheckRule, CryptographyBestPracticesRule,
    DependencySecurityRule, EpsilonPolicyLintRule, KSDistanceLintRule, TemplateReuseLintRule,
    UnsafeCodeRule,
};
use crate::{LintIssue, LintResults, LinterConfig, Result, SeverityLevel};
use std::process::Command;
use walkdir::WalkDir;

/// Security binary scanner for detecting pre-fix vulnerabilities
pub struct SecurityBinaryScanner;

impl SecurityBinaryScanner {
    /// Scan a binary for security vulnerabilities
    pub async fn scan_binary(&self, binary_path: &std::path::Path) -> Result<Vec<LintIssue>> {
        let mut issues = Vec::new();

        // Check if it's actually a binary
        if !self.is_binary_file(binary_path) {
            return Ok(issues);
        }

        // Check version stamp for vulnerable versions
        if let Ok(version_info) = self.extract_version_info(binary_path).await {
            issues.extend(self.check_vulnerable_versions(&version_info, binary_path).await);
        }

        // Check for vulnerable symbol patterns
        if let Ok(symbols) = self.extract_symbols(binary_path).await {
            issues.extend(self.check_vulnerable_symbols(&symbols, binary_path).await);
        }

        // Check build timestamp against security fix commits
        if let Ok(build_time) = self.extract_build_timestamp(binary_path).await {
            issues.extend(self.check_build_timestamp(&build_time, binary_path).await);
        }

        Ok(issues)
    }

    fn is_binary_file(&self, path: &std::path::Path) -> bool {
        // Check file extension and magic bytes
        if let Some(ext) = path.extension() {
            match ext.to_str() {
                Some("exe") | Some("dll") | Some("so") | Some("dylib") => return true,
                _ => {}
            }
        }

        // Check for binary without extension (common on Unix)
        if let Ok(metadata) = std::fs::metadata(path) {
            if metadata.is_file() && self.has_binary_magic(path) {
                return true;
            }
        }

        false
    }

    fn has_binary_magic(&self, path: &std::path::Path) -> bool {
        if let Ok(mut file) = std::fs::File::open(path) {
            let mut buffer = [0u8; 16];
            if let Ok(_) = std::io::Read::read(&mut file, &mut buffer) {
                // Check for common binary magic numbers
                match &buffer[0..4] {
                    [0x7f, 0x45, 0x4c, 0x46] => return true, // ELF
                    [0x4d, 0x5a, _, _] => return true,       // PE/DOS
                    [0xfe, 0xed, 0xfa, 0xce] => return true, // Mach-O 32-bit
                    [0xfe, 0xed, 0xfa, 0xcf] => return true, // Mach-O 64-bit
                    _ => {}
                }
            }
        }
        false
    }

    async fn extract_version_info(&self, binary_path: &std::path::Path) -> Result<String> {
        // Try to extract version information using various methods
        let mut version_info = String::new();

        // Method 1: Try to run the binary with --version
        if let Ok(output) = Command::new(binary_path)
            .arg("--version")
            .output()
        {
            version_info.push_str(&String::from_utf8_lossy(&output.stdout));
        }

        // Method 2: Use strings command to extract version strings
        #[cfg(unix)]
        if let Ok(output) = Command::new("strings")
            .arg(binary_path)
            .output()
        {
            let strings_output = String::from_utf8_lossy(&output.stdout);
            for line in strings_output.lines() {
                if line.contains("version") || line.contains("v0.1.") || line.contains("betanet") {
                    version_info.push_str(line);
                    version_info.push('\n');
                }
            }
        }

        // Method 3: Use objdump to extract build info
        #[cfg(unix)]
        if let Ok(output) = Command::new("objdump")
            .args(&["-s", "-j", ".comment"])
            .arg(binary_path)
            .output()
        {
            version_info.push_str(&String::from_utf8_lossy(&output.stdout));
        }

        Ok(version_info)
    }

    async fn extract_symbols(&self, binary_path: &std::path::Path) -> Result<Vec<String>> {
        let mut symbols = Vec::new();

        // Use nm command to extract symbols
        #[cfg(unix)]
        if let Ok(output) = Command::new("nm")
            .arg(binary_path)
            .output()
        {
            let nm_output = String::from_utf8_lossy(&output.stdout);
            for line in nm_output.lines() {
                if let Some(symbol) = line.split_whitespace().last() {
                    symbols.push(symbol.to_string());
                }
            }
        }

        // Windows equivalent using dumpbin
        #[cfg(windows)]
        if let Ok(output) = Command::new("dumpbin")
            .args(&["/symbols", binary_path.to_str().unwrap_or("")])
            .output()
        {
            let dumpbin_output = String::from_utf8_lossy(&output.stdout);
            for line in dumpbin_output.lines() {
                // Parse dumpbin symbol output
                if line.contains("SECT") || line.contains("External") {
                    if let Some(symbol) = line.split('|').last() {
                        symbols.push(symbol.trim().to_string());
                    }
                }
            }
        }

        Ok(symbols)
    }

    async fn extract_build_timestamp(&self, binary_path: &std::path::Path) -> Result<u64> {
        // Try to extract build timestamp from binary metadata
        if let Ok(metadata) = std::fs::metadata(binary_path) {
            if let Ok(modified) = metadata.modified() {
                if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                    return Ok(duration.as_secs());
                }
            }
        }

        // For PE files, extract timestamp from PE header
        #[cfg(windows)]
        {
            // This would require parsing PE headers, simplified for now
            if let Ok(_file) = std::fs::File::open(binary_path) {
                // Simplified PE timestamp extraction would go here
                return Ok(0);
            }
        }

        Ok(0)
    }

    async fn check_vulnerable_versions(&self, version_info: &str, binary_path: &std::path::Path) -> Vec<LintIssue> {
        let mut issues = Vec::new();

        // Known vulnerable version patterns
        let vulnerable_patterns = [
            ("v0.1.0", "Contains critical Sphinx nonce vulnerability (CVE-2025-SPHINX)"),
            ("v0.1.1", "Contains Noise key renegotiation vulnerability (CVE-2025-NOISE)"),
            ("betanet-htx 0.1.0", "HTX transport contains signature verification bypass"),
            ("betanet-mixnode 0.1.0", "Mixnode contains Ed25519 key generation vulnerability"),
        ];

        for (pattern, description) in &vulnerable_patterns {
            if version_info.contains(pattern) {
                issues.push(LintIssue::new(
                    "SEC-BIN-001".to_string(),
                    SeverityLevel::Critical,
                    format!("Vulnerable binary detected: {} - {}", pattern, description),
                    "security-binary-scan".to_string(),
                ).with_location(binary_path.to_path_buf(), 0, 0));
            }
        }

        // Check for absence of fixed version indicators
        if version_info.contains("betanet") && !version_info.contains("v0.1.2") && !version_info.contains("secure") {
            issues.push(LintIssue::new(
                "SEC-BIN-002".to_string(),
                SeverityLevel::Error,
                "Binary appears to be from pre-security-fix version - upgrade required".to_string(),
                "security-binary-scan".to_string(),
            ).with_location(binary_path.to_path_buf(), 0, 0));
        }

        issues
    }

    async fn check_vulnerable_symbols(&self, symbols: &[String], binary_path: &std::path::Path) -> Vec<LintIssue> {
        let mut issues = Vec::new();

        // Known vulnerable symbol patterns
        let vulnerable_symbols = [
            ("fake_verify", "Contains fake Ed25519 signature verification"),
            ("stub_key_gen", "Contains stubbed key generation"),
            ("zero_nonce", "Contains zero nonce vulnerability"),
            ("STUB_IMPL", "Contains stubbed cryptographic implementation"),
            ("INSECURE_", "Contains insecure implementation markers"),
        ];

        for symbol in symbols {
            for (vuln_pattern, description) in &vulnerable_symbols {
                if symbol.contains(vuln_pattern) {
                    issues.push(LintIssue::new(
                        "SEC-BIN-003".to_string(),
                        SeverityLevel::Critical,
                        format!("Vulnerable symbol found: {} - {}", symbol, description),
                        "security-binary-scan".to_string(),
                    ).with_location(binary_path.to_path_buf(), 0, 0));
                }
            }
        }

        // Check for missing security symbols that should be present
        let required_security_symbols = [
            "hkdf_derive",
            "ed25519_verify",
            "real_key_gen",
            "secure_nonce",
        ];

        for required_symbol in &required_security_symbols {
            if !symbols.iter().any(|s| s.contains(required_symbol)) {
                issues.push(LintIssue::new(
                    "SEC-BIN-004".to_string(),
                    SeverityLevel::Warning,
                    format!("Missing expected security symbol: {} - may indicate pre-fix binary", required_symbol),
                    "security-binary-scan".to_string(),
                ).with_location(binary_path.to_path_buf(), 0, 0));
            }
        }

        issues
    }

    async fn check_build_timestamp(&self, build_timestamp: &u64, binary_path: &std::path::Path) -> Vec<LintIssue> {
        let mut issues = Vec::new();

        // Security fix commit timestamps (Unix epoch)
        // These would be the actual commit timestamps from the security fixes
        let sphinx_fix_timestamp = 1692144000; // Approximate timestamp for commit 085abb6e
        let noise_fix_timestamp = 1692230400;  // Approximate timestamp for commit 5d9057d0

        if *build_timestamp > 0 {
            if *build_timestamp < sphinx_fix_timestamp {
                issues.push(LintIssue::new(
                    "SEC-BIN-005".to_string(),
                    SeverityLevel::Critical,
                    format!("Binary built before Sphinx security fixes ({})",
                           chrono::DateTime::from_timestamp(*build_timestamp as i64, 0)
                               .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                               .unwrap_or_else(|| "unknown".to_string())),
                    "security-binary-scan".to_string(),
                ).with_location(binary_path.to_path_buf(), 0, 0));
            } else if *build_timestamp < noise_fix_timestamp {
                issues.push(LintIssue::new(
                    "SEC-BIN-006".to_string(),
                    SeverityLevel::Error,
                    format!("Binary built before Noise security fixes ({})",
                           chrono::DateTime::from_timestamp(*build_timestamp as i64, 0)
                               .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                               .unwrap_or_else(|| "unknown".to_string())),
                    "security-binary-scan".to_string(),
                ).with_location(binary_path.to_path_buf(), 0, 0));
            }
        }

        issues
    }
}

/// Main linter
pub struct Linter {
    config: LinterConfig,
    rules: Vec<Box<dyn CheckRule>>,
    security_scanner: SecurityBinaryScanner,
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
            // Compliance rules
            Box::new(Section11ComplianceChecker),
        ];

        Self {
            config,
            rules,
            security_scanner: SecurityBinaryScanner,
        }
    }

    /// Run all linting checks
    pub async fn run(&self) -> Result<LintResults> {
        let mut results = LintResults::new();

        // Walk directory and find relevant files, skipping ignored paths
        for entry in WalkDir::new(&self.config.target_dir)
            .follow_links(true)
            .into_iter()
            .filter_entry(|e| !self.is_ignored(e.path()))
            .filter_map(|e| e.ok())
        {
            if entry.file_type().is_file() {
                if let Some(ext) = entry.path().extension() {
                    match ext.to_str() {
                        Some("rs") => self.check_rust_file(entry.path(), &mut results).await?,
                        Some("toml") => self.check_toml_file(entry.path(), &mut results).await?,
                        Some("exe") | Some("dll") | Some("so") | Some("dylib") => {
                            self.check_binary_file(entry.path(), &mut results).await?
                        }
                        _ => {}
                    }
                } else {
                    // Check files without extensions that might be binaries
                    self.check_potential_binary(entry.path(), &mut results).await?;
                }
                results.files_checked += 1;
            }
        }

        Ok(results)
    }

    /// Check specific rule
    pub async fn check_rule(&self, rule_name: &str) -> Result<LintResults> {
        let mut results = LintResults::new();

        // Handle special security-scan rule
        if rule_name == "security-scan" {
            return self.security_scan().await;
        }

        // Find the specific rule
        let target_rule = self.rules.iter().find(|rule| rule.name() == rule_name);

        if let Some(rule) = target_rule {
            // Walk directory and apply this specific rule
            for entry in WalkDir::new(&self.config.target_dir)
                .follow_links(true)
                .into_iter()
                .filter_entry(|e| !self.is_ignored(e.path()))
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

    /// Dedicated security scan for binaries
    pub async fn security_scan(&self) -> Result<LintResults> {
        let mut results = LintResults::new();

        for entry in WalkDir::new(&self.config.target_dir)
            .follow_links(true)
            .into_iter()
            .filter_entry(|e| !self.is_ignored(e.path()))
            .filter_map(|e| e.ok())
        {
            if entry.file_type().is_file() {
                // Check all potential binaries
                let security_issues = self.security_scanner.scan_binary(entry.path()).await?;
                for issue in security_issues {
                    results.add_issue(issue);
                }
                results.files_checked += 1;
            }
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

    async fn check_binary_file(
        &self,
        path: &std::path::Path,
        results: &mut LintResults,
    ) -> Result<()> {
        let security_issues = self.security_scanner.scan_binary(path).await?;
        for issue in security_issues {
            results.add_issue(issue);
        }
        Ok(())
    }

    async fn check_potential_binary(
        &self,
        path: &std::path::Path,
        results: &mut LintResults,
    ) -> Result<()> {
        // Only check files that might be binaries
        if self.security_scanner.is_binary_file(path) {
            let security_issues = self.security_scanner.scan_binary(path).await?;
            for issue in security_issues {
                results.add_issue(issue);
            }
        }
        Ok(())
    }

    fn is_ignored(&self, path: &std::path::Path) -> bool {
        if let Ok(relative) = path.strip_prefix(&self.config.target_dir) {
            for ancestor in relative.ancestors() {
                if let Some(name) = ancestor.file_name() {
                    if self
                        .config
                        .ignored_paths
                        .iter()
                        .any(|p| p.as_os_str() == name)
                    {
                        return true;
                    }
                }
            }
        }
        false
    }
}
