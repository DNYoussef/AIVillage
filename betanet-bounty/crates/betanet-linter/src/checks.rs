//! Linting rule implementations

use crate::{LintIssue, SeverityLevel, Result};
use regex::Regex;
use async_trait::async_trait;

pub mod bootstrap;
pub mod tls_mirror;
pub mod noise_xk;
pub mod frame_format;
pub mod scion_bridge;

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
        if !context.file_path.to_string_lossy().contains("tls") &&
           !context.content.contains("TemplateCache") {
            return Ok(issues);
        }

        // Look for stochastic_reuse_probability configuration
        let reuse_regex = Regex::new(r"stochastic_reuse_probability:\s*([0-9.]+)").unwrap();

        for (line_num, line) in context.content.lines().enumerate() {
            if let Some(captures) = reuse_regex.captures(line) {
                if let Some(prob_str) = captures.get(1) {
                    if let Ok(probability) = prob_str.as_str().parse::<f64>() {
                        // Check if probability is in acceptable range
                        if probability < 0.7 || probability > 0.95 {
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
        let histogram_regex = Regex::new(r"histogram.*=.*vec!\[").unwrap();
        let found_histogram = histogram_regex.is_match(&context.content);

        if !found_histogram && context.content.contains("sample_") {
            issues.push(LintIssue::new(
                "PRIV003".to_string(),
                SeverityLevel::Error,
                "Sampling function found without proper distribution histogram - may fail KS test".to_string(),
                self.name().to_string(),
            ));
        }

        // Check for realistic log-normal parameters
        let lognormal_regex = Regex::new(r"mu:\s*([0-9.]+).*sigma:\s*([0-9.]+)").unwrap();

        for (line_num, line) in context.content.lines().enumerate() {
            if let Some(captures) = lognormal_regex.captures(line) {
                if let (Some(mu_str), Some(sigma_str)) = (captures.get(1), captures.get(2)) {
                    if let (Ok(mu), Ok(sigma)) = (mu_str.as_str().parse::<f64>(), sigma_str.as_str().parse::<f64>()) {
                        // Check if parameters are realistic for web traffic
                        if mu < 3.0 || mu > 15.0 || sigma < 0.1 || sigma > 2.5 {
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
        let epsilon_regex = Regex::new(r"epsilon_max.*[:=]\s*([0-9.]+)").unwrap();

        for (line_num, line) in context.content.lines().enumerate() {
            if let Some(captures) = epsilon_regex.captures(line) {
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
            let has_overflow_handling = context.content.contains("allow_budget_overdraft") ||
                                        context.content.contains("reset_budget") ||
                                        context.content.contains("release_budget");

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
