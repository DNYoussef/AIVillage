//! Betanet Linter - Spec-compliance linter for Betanet components
//!
//! This crate provides linting and validation for Betanet protocol implementations,
//! including SBOM generation and compliance checking.

#![deny(warnings)]
#![deny(clippy::all)]
#![deny(missing_docs)]

use std::path::PathBuf;

use thiserror::Error;

pub mod checks;
pub mod lint;
pub mod report;

#[cfg(feature = "sbom")]
pub mod sbom;

/// Linter errors
#[derive(Debug, Error)]
pub enum LinterError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Parse error
    #[error("Parse error: {0}")]
    Parse(String),

    /// Validation error
    #[error("Validation error: {0}")]
    Validation(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// SBOM generation error
    #[error("SBOM error: {0}")]
    Sbom(String),
}

/// Result type for linter operations
pub type Result<T> = std::result::Result<T, LinterError>;

/// Linter configuration
#[derive(Debug, Clone)]
pub struct LinterConfig {
    /// Target directory to lint
    pub target_dir: PathBuf,
    /// Enable all checks
    pub enable_all_checks: bool,
    /// Generate SBOM
    pub generate_sbom: bool,
    /// Output format
    pub output_format: OutputFormat,
    /// Severity level
    pub severity_level: SeverityLevel,
}

/// Output format
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputFormat {
    /// Human-readable text
    Text,
    /// JSON format
    Json,
    /// SARIF format
    Sarif,
}

/// Severity level
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum SeverityLevel {
    /// Info level
    Info,
    /// Warning level
    Warning,
    /// Error level
    Error,
    /// Critical level
    Critical,
}

impl Default for LinterConfig {
    fn default() -> Self {
        Self {
            target_dir: PathBuf::from("."),
            enable_all_checks: true,
            generate_sbom: false,
            output_format: OutputFormat::Text,
            severity_level: SeverityLevel::Info,
        }
    }
}

/// Lint issue
#[derive(Debug, Clone)]
pub struct LintIssue {
    /// Issue ID
    pub id: String,
    /// Severity level
    pub severity: SeverityLevel,
    /// Issue message
    pub message: String,
    /// File path
    pub file_path: Option<PathBuf>,
    /// Line number
    pub line_number: Option<usize>,
    /// Column number
    pub column_number: Option<usize>,
    /// Rule that triggered this issue
    pub rule_name: String,
}

impl LintIssue {
    /// Create new lint issue
    pub fn new(id: String, severity: SeverityLevel, message: String, rule_name: String) -> Self {
        Self {
            id,
            severity,
            message,
            file_path: None,
            line_number: None,
            column_number: None,
            rule_name,
        }
    }

    /// Set file location
    pub fn with_location(mut self, file_path: PathBuf, line: usize, column: usize) -> Self {
        self.file_path = Some(file_path);
        self.line_number = Some(line);
        self.column_number = Some(column);
        self
    }
}

/// Lint results
#[derive(Debug, Clone)]
pub struct LintResults {
    /// List of issues found
    pub issues: Vec<LintIssue>,
    /// Files checked
    pub files_checked: usize,
    /// Rules executed
    pub rules_executed: usize,
}

impl LintResults {
    /// Create new empty results
    pub fn new() -> Self {
        Self {
            issues: vec![],
            files_checked: 0,
            rules_executed: 0,
        }
    }

    /// Add issue
    pub fn add_issue(&mut self, issue: LintIssue) {
        self.issues.push(issue);
    }

    /// Get issues by severity
    pub fn issues_by_severity(&self, severity: &SeverityLevel) -> Vec<&LintIssue> {
        self.issues
            .iter()
            .filter(|i| &i.severity == severity)
            .collect()
    }

    /// Check if has issues above severity level
    pub fn has_issues_above(&self, severity: &SeverityLevel) -> bool {
        self.issues.iter().any(|i| i.severity >= *severity)
    }

    /// Get summary statistics
    pub fn summary(&self) -> (usize, usize, usize, usize) {
        let critical = self.issues_by_severity(&SeverityLevel::Critical).len();
        let errors = self.issues_by_severity(&SeverityLevel::Error).len();
        let warnings = self.issues_by_severity(&SeverityLevel::Warning).len();
        let info = self.issues_by_severity(&SeverityLevel::Info).len();

        (critical, errors, warnings, info)
    }
}

impl Default for LintResults {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lint_issue() {
        let issue = LintIssue::new(
            "TEST001".to_string(),
            SeverityLevel::Warning,
            "Test message".to_string(),
            "test-rule".to_string(),
        );

        assert_eq!(issue.id, "TEST001");
        assert_eq!(issue.severity, SeverityLevel::Warning);
        assert!(issue.file_path.is_none());
    }

    #[test]
    fn test_lint_results() {
        let mut results = LintResults::new();

        results.add_issue(LintIssue::new(
            "E001".to_string(),
            SeverityLevel::Error,
            "Error message".to_string(),
            "error-rule".to_string(),
        ));

        results.add_issue(LintIssue::new(
            "W001".to_string(),
            SeverityLevel::Warning,
            "Warning message".to_string(),
            "warning-rule".to_string(),
        ));

        assert_eq!(results.issues.len(), 2);
        assert!(results.has_issues_above(&SeverityLevel::Warning));

        let (critical, errors, warnings, info) = results.summary();
        assert_eq!(critical, 0);
        assert_eq!(errors, 1);
        assert_eq!(warnings, 1);
        assert_eq!(info, 0);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(SeverityLevel::Critical > SeverityLevel::Error);
        assert!(SeverityLevel::Error > SeverityLevel::Warning);
        assert!(SeverityLevel::Warning > SeverityLevel::Info);
    }
}
