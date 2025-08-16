//! Linting rule implementations

use crate::{LintIssue, Result, SeverityLevel};
use async_trait::async_trait;
use once_cell::sync::Lazy;
use regex::Regex;

pub mod bootstrap;
pub mod frame_format;
pub mod noise_xk;
pub mod scion_bridge;
pub mod tls_mirror;

static TEMPLATE_REUSE_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"stochastic_reuse_probability:\s*([0-9.]+)").unwrap());

static KS_HISTOGRAM_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"histogram.*=.*vec!\[").unwrap());

static KS_LOGNORMAL_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"mu:\s*([0-9.]+).*sigma:\s*([0-9.]+)").unwrap());

static KEY_SIZE_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"key_size.*=.*(\d+)").unwrap());

static SECRET_REGEXES: Lazy<Vec<Regex>> = Lazy::new(|| {
    vec![
        Regex::new(r#"password\s*=\s*"[^"]+""#).unwrap(),
        Regex::new(r#"secret\s*=\s*"[^"]+""#).unwrap(),
        Regex::new(r#"key\s*=\s*"[^"]+""#).unwrap(),
        Regex::new(r#"token\s*=\s*"[^"]+""#).unwrap(),
    ]
});

static EPSILON_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"epsilon_max.*[:=]\s*([0-9.]+)").unwrap());

/// Check rule trait
#[async_trait]
pub trait CheckRule: Send + Sync {
    /// Rule name
    fn name(&self) -> &str;

    /// Rule description
    fn description(&self) -> &str;

    /// Execute rule check
    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>>;
}

/// Check context
pub struct CheckContext {
    /// File being checked
    pub file_path: std::path::PathBuf,
    /// File content
    pub content: String,
}

/// Security rule for unsafe code
pub struct UnsafeCodeRule;

#[async_trait]
impl CheckRule for UnsafeCodeRule {
    fn name(&self) -> &str {
        "unsafe-code"
    }

    fn description(&self) -> &str {
        "Check for unsafe code blocks"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        if context.content.contains("unsafe") {
            issues.push(LintIssue::new(
                "S001".to_string(),
                SeverityLevel::Warning,
                "Unsafe code detected".to_string(),
                self.name().to_string(),
            ));
        }

        Ok(issues)
    }
}

/// Performance rule for blocking calls
pub struct BlockingCallRule;

#[async_trait]
impl CheckRule for BlockingCallRule {
    fn name(&self) -> &str {
        "blocking-calls"
    }

    fn description(&self) -> &str {
        "Check for blocking calls in async code"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        if context.content.contains("std::thread::sleep") {
            issues.push(LintIssue::new(
                "P001".to_string(),
                SeverityLevel::Error,
                "Blocking sleep call in async context".to_string(),
                self.name().to_string(),
            ));
        }

        Ok(issues)
    }
}

/// Privacy/Camouflage Rules

/// Check template cache calibration reuse ratio
pub struct TemplateReuseLintRule;

#[async_trait]
impl CheckRule for TemplateReuseLintRule {
    fn name(&self) -> &str {
        "template-cache-reuse-ratio"
    }

    fn description(&self) -> &str {
        "Check that TLS template cache has appropriate calibration reuse ratio"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Check if this is a TLS camouflage related file
        if !context.file_path.to_string_lossy().contains("tls")
            && !context.content.contains("TemplateCache")
        {
            return Ok(issues);
        }

        // Look for stochastic_reuse_probability configuration
        for (line_num, line) in context.content.lines().enumerate() {
            if let Some(captures) = TEMPLATE_REUSE_REGEX.captures(line) {
                if let Some(prob_str) = captures.get(1) {
                    if let Ok(probability) = prob_str.as_str().parse::<f64>() {
                        // Check if probability is in acceptable range
                        if !(0.7..=0.95).contains(&probability) {
                            issues.push(LintIssue::new(
                                "PRIV001".to_string(),
                                SeverityLevel::Warning,
                                format!("Template reuse probability {} is outside recommended range 0.7-0.95", probability),
                                self.name().to_string(),
                            ).with_location(context.file_path.clone(), line_num + 1, 0));
                        }
                    }
                }
            }
        }

        // Check for hardcoded template values (anti-pattern)
        if context.content.contains("template_data: vec![]") {
            issues.push(LintIssue::new(
                "PRIV002".to_string(),
                SeverityLevel::Error,
                "Hardcoded empty template data detected - templates should be dynamically generated".to_string(),
                self.name().to_string(),
            ));
        }

        Ok(issues)
    }
}

/// Check KS (Kolmogorov-Smirnov) distance for padding/timing distributions
pub struct KSDistanceLintRule;

#[async_trait]
impl CheckRule for KSDistanceLintRule {
    fn name(&self) -> &str {
        "ks-distance-distribution"
    }

    fn description(&self) -> &str {
        "Check KS distance for realistic timing/padding distributions"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Check for mixture model implementations
        if !context.content.contains("MixtureModel") && !context.content.contains("sample_") {
            return Ok(issues);
        }

        // Look for histogram or distribution definitions
        let found_histogram = KS_HISTOGRAM_REGEX.is_match(&context.content);

        if !found_histogram && context.content.contains("sample_") {
            issues.push(LintIssue::new(
                "PRIV003".to_string(),
                SeverityLevel::Error,
                "Sampling function found without proper distribution histogram - may fail KS test"
                    .to_string(),
                self.name().to_string(),
            ));
        }

        // Check for realistic log-normal parameters
        for (line_num, line) in context.content.lines().enumerate() {
            if let Some(captures) = KS_LOGNORMAL_REGEX.captures(line) {
                if let (Some(mu_str), Some(sigma_str)) = (captures.get(1), captures.get(2)) {
                    if let (Ok(mu), Ok(sigma)) = (
                        mu_str.as_str().parse::<f64>(),
                        sigma_str.as_str().parse::<f64>(),
                    ) {
                        // Check if parameters are realistic for web traffic
                        if !(3.0..=15.0).contains(&mu) || !(0.1..=2.5).contains(&sigma) {
                            issues.push(LintIssue::new(
                                "PRIV004".to_string(),
                                SeverityLevel::Warning,
                                format!("Log-normal parameters μ={}, σ={} may not match real web traffic distributions", mu, sigma),
                                self.name().to_string(),
                            ).with_location(context.file_path.clone(), line_num + 1, 0));
                        }
                    }
                }
            }
        }

        Ok(issues)
    }
}

/// Check for dependency vulnerabilities
pub struct DependencySecurityRule;

#[async_trait]
impl CheckRule for DependencySecurityRule {
    fn name(&self) -> &str {
        "dependency-security"
    }

    fn description(&self) -> &str {
        "Check for known vulnerable dependencies and security advisories"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Check Cargo.toml files for known vulnerable versions
        if context.file_path.to_string_lossy().contains("Cargo.toml") {
            // Look for common vulnerable dependency patterns
            let vulnerable_deps = [
                ("chrono", "0.4.19", "Segmentation fault vulnerability"),
                ("openssl", "0.10.38", "Security vulnerabilities in OpenSSL"),
                ("regex", "1.5.4", "ReDoS vulnerability patterns"),
                ("serde_yaml", "0.8.23", "Stack overflow vulnerability"),
                (
                    "time",
                    "0.1.44",
                    "Potential for incorrect time calculations",
                ),
            ];

            for (dep_name, vuln_version, description) in &vulnerable_deps {
                let dep_regex =
                    regex::Regex::new(&format!(r#"{}\s*=\s*"({})""#, dep_name, vuln_version))
                        .unwrap();
                for (line_num, line) in context.content.lines().enumerate() {
                    if dep_regex.is_match(line) {
                        issues.push(
                            LintIssue::new(
                                "SEC001".to_string(),
                                SeverityLevel::Critical,
                                format!(
                                    "Vulnerable dependency {} version {} detected: {}",
                                    dep_name, vuln_version, description
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

            // Check for dev-dependencies that shouldn't be in production
            let dev_only_deps = ["proptest", "criterion", "quickcheck"];
            for dep in &dev_only_deps {
                if context.content.contains(&format!("dependencies]\n{}", dep))
                    || context.content.contains(&format!("dependencies.{}", dep))
                {
                    issues.push(LintIssue::new(
                        "SEC002".to_string(),
                        SeverityLevel::Warning,
                        format!("Development dependency '{}' found in main dependencies - should be dev-dependency", dep),
                        self.name().to_string(),
                    ));
                }
            }
        }

        Ok(issues)
    }
}

/// Check for crypto implementation best practices
pub struct CryptographyBestPracticesRule;

#[async_trait]
impl CheckRule for CryptographyBestPracticesRule {
    fn name(&self) -> &str {
        "cryptography-best-practices"
    }

    fn description(&self) -> &str {
        "Check for cryptographic implementation best practices and common pitfalls"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Check for weak crypto patterns
        let weak_patterns = [
            ("MD5", "MD5 is cryptographically broken"),
            ("SHA1", "SHA1 is cryptographically weak"),
            ("RC4", "RC4 is insecure"),
            ("DES", "DES has insufficient key length"),
            (
                "rand::thread_rng",
                "Consider using ChaCha20Rng for reproducible randomness",
            ),
        ];

        for (pattern, warning) in &weak_patterns {
            if context.content.contains(pattern) {
                issues.push(LintIssue::new(
                    "CRYPTO001".to_string(),
                    SeverityLevel::Warning,
                    format!("Weak cryptography detected: {} - {}", pattern, warning),
                    self.name().to_string(),
                ));
            }
        }

        // Check for proper key sizes
        for (line_num, line) in context.content.lines().enumerate() {
            if let Some(captures) = KEY_SIZE_REGEX.captures(line) {
                if let Some(size_str) = captures.get(1) {
                    if let Ok(size) = size_str.as_str().parse::<u32>() {
                        if size < 128 {
                            issues.push(
                                LintIssue::new(
                                    "CRYPTO002".to_string(),
                                    SeverityLevel::Error,
                                    format!(
                                        "Key size {} bits too small - use at least 128 bits",
                                        size
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

        // Check for hardcoded secrets/keys
        for regex in SECRET_REGEXES.iter() {
            for (line_num, line) in context.content.lines().enumerate() {
                if regex.is_match(line) && !line.contains("example") && !line.contains("test") {
                    issues.push(
                        LintIssue::new(
                            "CRYPTO003".to_string(),
                            SeverityLevel::Critical,
                            "Hardcoded secret detected - use environment variables or secure vault"
                                .to_string(),
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

        Ok(issues)
    }
}

/// Check epsilon privacy policy compliance
pub struct EpsilonPolicyLintRule;

#[async_trait]
impl CheckRule for EpsilonPolicyLintRule {
    fn name(&self) -> &str {
        "epsilon-privacy-policy"
    }

    fn description(&self) -> &str {
        "Check epsilon privacy budget policy compliance"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        // Check for privacy budget related code
        if !context.content.contains("epsilon") && !context.content.contains("PrivacyBudget") {
            return Ok(issues);
        }

        // Look for epsilon_max configurations
        for (line_num, line) in context.content.lines().enumerate() {
            if let Some(captures) = EPSILON_REGEX.captures(line) {
                if let Some(eps_str) = captures.get(1) {
                    if let Ok(epsilon) = eps_str.as_str().parse::<f64>() {
                        // Check if epsilon is within acceptable bounds for differential privacy
                        if epsilon > 10.0 {
                            issues.push(LintIssue::new(
                                "PRIV008".to_string(),
                                SeverityLevel::Error,
                                format!("Epsilon value {} exceeds maximum recommended value of 10.0 for differential privacy", epsilon),
                                self.name().to_string(),
                            ).with_location(context.file_path.clone(), line_num + 1, 0));
                        } else if epsilon > 1.0 {
                            issues.push(LintIssue::new(
                                "PRIV009".to_string(),
                                SeverityLevel::Warning,
                                format!("Epsilon value {} is high - consider stronger privacy protection", epsilon),
                                self.name().to_string(),
                            ).with_location(context.file_path.clone(), line_num + 1, 0));
                        }
                    }
                }
            }
        }

        // Check for budget exhaustion handling
        if context.content.contains("BudgetExceeded") || context.content.contains("epsilon_used") {
            let has_overflow_handling = context.content.contains("allow_budget_overdraft")
                || context.content.contains("reset_budget")
                || context.content.contains("release_budget");

            if !has_overflow_handling {
                issues.push(LintIssue::new(
                    "PRIV010".to_string(),
                    SeverityLevel::Warning,
                    "Privacy budget management without proper overflow/reset handling".to_string(),
                    self.name().to_string(),
                ));
            }
        }

        Ok(issues)
    }
}
