//! TLS Mirror Compliance Checks
//!
//! Validates TLS camouflage implementation for proper:
//! - Template cache configuration and behavior
//! - Stochastic reuse probability settings
//! - Site classification and behavioral profiles
//! - Mixture model parameter ranges
//! - Cover traffic generation
//! - Anti-fingerprinting measures

use crate::checks::{CheckContext, CheckRule};
use crate::{LintIssue, Result, SeverityLevel};
use async_trait::async_trait;
use once_cell::sync::Lazy;
use regex::Regex;

static TLS_REUSE_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"stochastic_reuse_probability:\s*([0-9.]+)").unwrap());

static TTL_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"default_ttl:\s*Duration::from_secs\((\d+)\)").unwrap());

static CACHE_SIZE_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"max_cache_size:\s*(\d+)").unwrap());

static HARDCODED_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"if\s+origin\s*==\s*"([^"]+)""#).unwrap());

static LOGNORMAL_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"mu:\s*([0-9.]+),\s*sigma:\s*([0-9.]+)").unwrap());

static WEIGHT_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"weight:\s*([0-9.]+)").unwrap());

static NUM_ENTRIES_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"gen_range\((\d+)\.\.=?(\d+)\)").unwrap());

static POOL_LIMIT_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"while.*len\(\)\s*>\s*(\d+)").unwrap());

static CHROME_VERSION_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"Chrome[._]?(\d+)").unwrap());

/// TLS mirror template cache compliance
pub struct TlsTemplateCacheRule;

#[async_trait]
impl CheckRule for TlsTemplateCacheRule {
    fn name(&self) -> &str {
        "tls-template-cache"
    }

    fn description(&self) -> &str {
        "Check TLS template cache configuration for proper TTL and reuse ratios"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check TLS-related files
        if !context.file_path.to_string_lossy().contains("tls")
            && !context.content.contains("TemplateCache")
            && !context.content.contains("template_cache")
        {
            return Ok(issues);
        }

        // Check stochastic reuse probability
        for (line_num, line) in context.content.lines().enumerate() {
            if let Some(captures) = TLS_REUSE_REGEX.captures(line) {
                if let Some(prob_str) = captures.get(1) {
                    if let Ok(probability) = prob_str.as_str().parse::<f64>() {
                        if !(0.7..=0.95).contains(&probability) {
                            issues.push(LintIssue::new(
                                "TLS001".to_string(),
                                SeverityLevel::Warning,
                                format!("Stochastic reuse probability {} outside recommended range 0.7-0.95", probability),
                                self.name().to_string(),
                            ).with_location(context.file_path.clone(), line_num + 1, 0));
                        }

                        if probability > 0.98 {
                            issues.push(LintIssue::new(
                                "TLS002".to_string(),
                                SeverityLevel::Error,
                                format!("Extremely high reuse probability {} may create detectable patterns", probability),
                                self.name().to_string(),
                            ).with_location(context.file_path.clone(), line_num + 1, 0));
                        }
                    }
                }
            }
        }

        // Check TTL configuration
        for (line_num, line) in context.content.lines().enumerate() {
            if let Some(captures) = TTL_REGEX.captures(line) {
                if let Some(ttl_str) = captures.get(1) {
                    if let Ok(ttl_secs) = ttl_str.as_str().parse::<u64>() {
                        if ttl_secs < 600 {
                            // 10 minutes
                            issues.push(LintIssue::new(
                                "TLS003".to_string(),
                                SeverityLevel::Warning,
                                format!("TTL {} seconds too short, may cause excessive template regeneration", ttl_secs),
                                self.name().to_string(),
                            ).with_location(context.file_path.clone(), line_num + 1, 0));
                        }

                        if ttl_secs > 7200 {
                            // 2 hours
                            issues.push(
                                LintIssue::new(
                                    "TLS004".to_string(),
                                    SeverityLevel::Warning,
                                    format!(
                                        "TTL {} seconds too long, may create stale templates",
                                        ttl_secs
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

        // Check for hardcoded template data (anti-pattern)
        if context.content.contains("template_data: vec![]") {
            issues.push(LintIssue::new(
                "TLS005".to_string(),
                SeverityLevel::Error,
                "Hardcoded empty template data - templates should be dynamically generated"
                    .to_string(),
                self.name().to_string(),
            ));
        }

        // Check max cache size configuration
        for (line_num, line) in context.content.lines().enumerate() {
            if let Some(captures) = CACHE_SIZE_REGEX.captures(line) {
                if let Some(size_str) = captures.get(1) {
                    if let Ok(cache_size) = size_str.as_str().parse::<u32>() {
                        if cache_size < 100 {
                            issues.push(
                                LintIssue::new(
                                    "TLS006".to_string(),
                                    SeverityLevel::Warning,
                                    format!(
                                        "Cache size {} too small, may cause frequent evictions",
                                        cache_size
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

                        if cache_size > 2000 {
                            issues.push(
                                LintIssue::new(
                                    "TLS007".to_string(),
                                    SeverityLevel::Warning,
                                    format!(
                                        "Cache size {} excessive, may consume too much memory",
                                        cache_size
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

        Ok(issues)
    }
}

/// TLS mirror site classification compliance
pub struct TlsSiteClassificationRule;

#[async_trait]
impl CheckRule for TlsSiteClassificationRule {
    fn name(&self) -> &str {
        "tls-site-classification"
    }

    fn description(&self) -> &str {
        "Check TLS site classification for proper behavioral profiles"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check site classification related files
        if !context.content.contains("SiteClass") && !context.content.contains("from_origin") {
            return Ok(issues);
        }

        // Check for hardcoded site classifications (anti-pattern)
        for (line_num, line) in context.content.lines().enumerate() {
            if HARDCODED_REGEX.is_match(line) {
                issues.push(LintIssue::new(
                    "TLS008".to_string(),
                    SeverityLevel::Warning,
                    "Hardcoded origin classification may not scale - consider pattern-based classification".to_string(),
                    self.name().to_string(),
                ).with_location(context.file_path.clone(), line_num + 1, 0));
            }
        }

        // Check for missing Unknown fallback
        if context.content.contains("enum SiteClass") && !context.content.contains("Unknown") {
            issues.push(LintIssue::new(
                "TLS009".to_string(),
                SeverityLevel::Error,
                "SiteClass enum missing Unknown variant for unclassified origins".to_string(),
                self.name().to_string(),
            ));
        }

        // Check for balanced site class coverage
        let site_classes = [
            "CDN", "Social", "Commerce", "News", "Tech", "Finance", "Gaming", "Stream",
        ];
        let mut covered_classes = 0;
        for class in &site_classes {
            if context.content.contains(class) {
                covered_classes += 1;
            }
        }

        if covered_classes < 5 && context.content.contains("SiteClass") {
            issues.push(LintIssue::new(
                "TLS010".to_string(),
                SeverityLevel::Warning,
                format!("Only {} site classes covered, consider broader classification for better camouflage", covered_classes),
                self.name().to_string(),
            ));
        }

        Ok(issues)
    }
}

/// TLS mirror mixture model compliance
pub struct TlsMixtureModelRule;

#[async_trait]
impl CheckRule for TlsMixtureModelRule {
    fn name(&self) -> &str {
        "tls-mixture-model"
    }

    fn description(&self) -> &str {
        "Check TLS mixture model parameters for realistic distributions"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check mixture model files
        if !context.content.contains("MixtureModel")
            && !context.content.contains("LogNormalComponent")
        {
            return Ok(issues);
        }

        // Check log-normal parameters for realistic ranges
        for (line_num, line) in context.content.lines().enumerate() {
            if let Some(captures) = LOGNORMAL_REGEX.captures(line) {
                if let (Some(mu_str), Some(sigma_str)) = (captures.get(1), captures.get(2)) {
                    if let (Ok(mu), Ok(sigma)) = (
                        mu_str.as_str().parse::<f64>(),
                        sigma_str.as_str().parse::<f64>(),
                    ) {
                        // Check mu parameter (log-scale location)
                        if !(3.0..=16.0).contains(&mu) {
                            issues.push(LintIssue::new(
                                "TLS011".to_string(),
                                SeverityLevel::Warning,
                                format!("Log-normal μ={} outside typical web traffic range 3.0-16.0", mu),
                                self.name().to_string(),
                            ).with_location(context.file_path.clone(), line_num + 1, 0));
                        }

                        // Check sigma parameter (log-scale spread)
                        if !(0.1..=2.5).contains(&sigma) {
                            issues.push(LintIssue::new(
                                "TLS012".to_string(),
                                SeverityLevel::Warning,
                                format!("Log-normal σ={} outside realistic variability range 0.1-2.5", sigma),
                                self.name().to_string(),
                            ).with_location(context.file_path.clone(), line_num + 1, 0));
                        }
                    }
                }
            }
        }

        // Check mixture component weights
        let mut total_weight = 0.0;
        let mut weight_count = 0;

        for line in context.content.lines() {
            if let Some(captures) = WEIGHT_REGEX.captures(line) {
                if let Some(weight_str) = captures.get(1) {
                    if let Ok(weight) = weight_str.as_str().parse::<f64>() {
                        total_weight += weight;
                        weight_count += 1;

                        if !(0.0..=1.0).contains(&weight) {
                            issues.push(LintIssue::new(
                                "TLS013".to_string(),
                                SeverityLevel::Error,
                                format!("Mixture component weight {} invalid, must be in range [0.0, 1.0]", weight),
                                self.name().to_string(),
                            ));
                        }
                    }
                }
            }
        }

        // Check if weights approximately sum to 1.0 for each mixture
        if weight_count > 0 && weight_count % 2 == 0 {
            // Assuming pairs for timing/size
            let expected_sum = (weight_count / 2) as f64; // Each mixture should sum to 1.0
            if (total_weight - expected_sum).abs() > 0.1 {
                issues.push(LintIssue::new(
                    "TLS014".to_string(),
                    SeverityLevel::Warning,
                    format!(
                        "Mixture component weights sum to {:.2}, should approximately sum to {:.1}",
                        total_weight, expected_sum
                    ),
                    self.name().to_string(),
                ));
            }
        }

        Ok(issues)
    }
}

/// TLS mirror cover traffic compliance
pub struct TlsCoverTrafficRule;

#[async_trait]
impl CheckRule for TlsCoverTrafficRule {
    fn name(&self) -> &str {
        "tls-cover-traffic"
    }

    fn description(&self) -> &str {
        "Check TLS cover traffic generation for proper randomization and volume"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check cover traffic related files
        if !context.content.contains("CoverTraffic")
            && !context.content.contains("cover_pool")
            && !context.content.contains("generate_cover")
        {
            return Ok(issues);
        }

        // Check cover traffic volume configuration
        for (line_num, line) in context.content.lines().enumerate() {
            if line.contains("cover") && line.contains("entries") {
                if let Some(captures) = NUM_ENTRIES_REGEX.captures(line) {
                    if let (Some(min_str), Some(max_str)) = (captures.get(1), captures.get(2)) {
                        if let (Ok(min_entries), Ok(max_entries)) = (
                            min_str.as_str().parse::<u32>(),
                            max_str.as_str().parse::<u32>(),
                        ) {
                            if min_entries < 3 {
                                issues.push(LintIssue::new(
                                    "TLS015".to_string(),
                                    SeverityLevel::Warning,
                                    format!("Minimum cover entries {} too low, may not provide adequate cover", min_entries),
                                    self.name().to_string(),
                                ).with_location(context.file_path.clone(), line_num + 1, 0));
                            }

                            if max_entries > 50 {
                                issues.push(LintIssue::new(
                                    "TLS016".to_string(),
                                    SeverityLevel::Warning,
                                    format!("Maximum cover entries {} too high, may waste resources", max_entries),
                                    self.name().to_string(),
                                ).with_location(context.file_path.clone(), line_num + 1, 0));
                            }

                            if (max_entries - min_entries) < 3 {
                                issues.push(LintIssue::new(
                                    "TLS017".to_string(),
                                    SeverityLevel::Warning,
                                    "Cover traffic volume range too narrow, may create predictable patterns".to_string(),
                                    self.name().to_string(),
                                ).with_location(context.file_path.clone(), line_num + 1, 0));
                            }
                        }
                    }
                }
            }
        }

        // Check cover pool size limits
        for (line_num, line) in context.content.lines().enumerate() {
            if line.contains("cover_pool") {
                if let Some(captures) = POOL_LIMIT_REGEX.captures(line) {
                    if let Some(limit_str) = captures.get(1) {
                        if let Ok(limit) = limit_str.as_str().parse::<u32>() {
                            if limit < 50 {
                                issues.push(LintIssue::new(
                                    "TLS018".to_string(),
                                    SeverityLevel::Warning,
                                    format!("Cover pool limit {} too small, may exhaust cover traffic", limit),
                                    self.name().to_string(),
                                ).with_location(context.file_path.clone(), line_num + 1, 0));
                            }

                            if limit > 1000 {
                                issues.push(LintIssue::new(
                                    "TLS019".to_string(),
                                    SeverityLevel::Warning,
                                    format!("Cover pool limit {} excessive, may consume too much memory", limit),
                                    self.name().to_string(),
                                ).with_location(context.file_path.clone(), line_num + 1, 0));
                            }
                        }
                    }
                }
            }
        }

        // Check for deterministic cover traffic generation (anti-pattern)
        if context.content.contains("cover-") && context.content.contains("example.com") {
            // Check if server names are too predictable
            let predictable_names = ["cover-1", "cover-test", "cover-static"];
            for name in &predictable_names {
                if context.content.contains(name) {
                    issues.push(LintIssue::new(
                        "TLS020".to_string(),
                        SeverityLevel::Warning,
                        format!(
                            "Predictable cover traffic server name '{}' may be detectable",
                            name
                        ),
                        self.name().to_string(),
                    ));
                }
            }
        }

        Ok(issues)
    }
}

/// TLS mirror anti-fingerprinting compliance
pub struct TlsAntiFingerprintRule;

#[async_trait]
impl CheckRule for TlsAntiFingerprintRule {
    fn name(&self) -> &str {
        "tls-anti-fingerprint"
    }

    fn description(&self) -> &str {
        "Check TLS anti-fingerprinting measures for proper randomization and diversity"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Only check anti-fingerprinting related files
        if !context.content.contains("fingerprint")
            && !context.content.contains("Chrome")
            && !context.content.contains("ja3")
            && !context.content.contains("ClientHello")
        {
            return Ok(issues);
        }

        // Check for hardcoded Chrome version (anti-pattern)
        let mut hardcoded_versions = 0;
        for (line_num, line) in context.content.lines().enumerate() {
            if let Some(captures) = CHROME_VERSION_REGEX.captures(line) {
                if let Some(version_str) = captures.get(1) {
                    if let Ok(version) = version_str.as_str().parse::<u32>() {
                        hardcoded_versions += 1;

                        // Check if version is realistic (Chrome versions)
                        if !(90..=130).contains(&version) {
                            issues.push(
                                LintIssue::new(
                                    "TLS021".to_string(),
                                    SeverityLevel::Warning,
                                    format!(
                                        "Chrome version {} unrealistic, may be detectable",
                                        version
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

        if hardcoded_versions > 3 {
            issues.push(LintIssue::new(
                "TLS022".to_string(),
                SeverityLevel::Warning,
                "Multiple hardcoded Chrome versions may reduce fingerprint diversity".to_string(),
                self.name().to_string(),
            ));
        }

        // Check for cipher suite diversity
        let cipher_patterns = [
            "TLS_AES_128",
            "TLS_AES_256",
            "TLS_CHACHA20",
            "ECDHE_RSA",
            "ECDHE_ECDSA",
        ];
        let mut cipher_count = 0;
        for pattern in &cipher_patterns {
            if context.content.contains(pattern) {
                cipher_count += 1;
            }
        }

        if cipher_count < 3 && context.content.contains("cipher_suites") {
            issues.push(LintIssue::new(
                "TLS023".to_string(),
                SeverityLevel::Warning,
                format!(
                    "Only {} cipher suite types detected, consider more diversity",
                    cipher_count
                ),
                self.name().to_string(),
            ));
        }

        // Check for extension diversity
        let extension_patterns = [
            "SERVER_NAME",
            "SUPPORTED_GROUPS",
            "SIGNATURE_ALGORITHMS",
            "ALPN",
            "KEY_SHARE",
        ];
        let mut extension_count = 0;
        for pattern in &extension_patterns {
            if context.content.contains(pattern) {
                extension_count += 1;
            }
        }

        if extension_count < 4 && context.content.contains("extensions") {
            issues.push(LintIssue::new(
                "TLS024".to_string(),
                SeverityLevel::Warning,
                format!(
                    "Only {} extension types detected, consider more diversity",
                    extension_count
                ),
                self.name().to_string(),
            ));
        }

        // Check for deterministic randomization (anti-pattern)
        if context.content.contains("deterministic for testing") {
            issues.push(LintIssue::new(
                "TLS025".to_string(),
                SeverityLevel::Critical,
                "Deterministic randomization detected - must use cryptographically secure randomness in production".to_string(),
                self.name().to_string(),
            ));
        }

        // Check for JA3/JA4 hash validation
        if (context.content.contains("ja3") || context.content.contains("ja4"))
            && !context.content.contains("hash")
            && !context.content.contains("md5")
        {
            issues.push(LintIssue::new(
                "TLS026".to_string(),
                SeverityLevel::Warning,
                "JA3/JA4 fingerprinting without hash computation may not be effective".to_string(),
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
            file_path: PathBuf::from("test_tls.rs"),
            content: content.to_string(),
        }
    }

    #[tokio::test]
    async fn test_template_cache_rule() {
        let rule = TlsTemplateCacheRule;

        // Test valid configuration
        let content = r#"
            TemplateCacheConfig {
                stochastic_reuse_probability: 0.85,
                default_ttl: Duration::from_secs(1800),
                max_cache_size: 500,
            }
        "#;
        let context = create_test_context(content);
        let issues = rule.check(&context).await.unwrap();
        assert_eq!(issues.len(), 0);

        // Test invalid reuse probability
        let content = r#"
            stochastic_reuse_probability: 0.99,
        "#;
        let context = create_test_context(content);
        let issues = rule.check(&context).await.unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].id, "TLS002");
    }

    #[tokio::test]
    async fn test_mixture_model_rule() {
        let rule = TlsMixtureModelRule;

        // Test invalid log-normal parameters
        let content = r#"
            LogNormalComponent { weight: 0.6, mu: 1.5, sigma: 3.0 },
        "#;
        let context = create_test_context(content);
        let issues = rule.check(&context).await.unwrap();
        assert!(issues.len() >= 2); // Should flag both mu and sigma
    }

    #[tokio::test]
    async fn test_anti_fingerprint_rule() {
        let rule = TlsAntiFingerprintRule;

        // Test deterministic randomization
        let content = r#"
            // deterministic for testing
            random.iter_mut().enumerate()
        "#;
        let context = create_test_context(content);
        let issues = rule.check(&context).await.unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].severity, SeverityLevel::Critical);
    }
}
