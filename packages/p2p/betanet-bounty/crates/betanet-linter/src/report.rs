//! Report generation

use crate::{LintResults, OutputFormat, Result};
use serde_json::json;

/// Report generator
pub struct ReportGenerator {
    format: OutputFormat,
}

impl ReportGenerator {
    /// Create new report generator
    pub fn new(format: OutputFormat) -> Self {
        Self { format }
    }

    /// Generate report
    pub fn generate(&self, results: &LintResults) -> Result<String> {
        match self.format {
            OutputFormat::Text => self.generate_text(results),
            OutputFormat::Json => self.generate_json(results),
            OutputFormat::Sarif => self.generate_sarif(results),
        }
    }

    fn generate_text(&self, results: &LintResults) -> Result<String> {
        let mut output = String::new();

        let (critical, errors, warnings, info) = results.summary();

        output.push_str("Betanet Linter Results\n");
        output.push_str("======================\n\n");
        output.push_str(&format!("Files checked: {}\n", results.files_checked));
        output.push_str(&format!("Rules executed: {}\n", results.rules_executed));
        output.push_str(&format!(
            "Critical: {}, Errors: {}, Warnings: {}, Info: {}\n\n",
            critical, errors, warnings, info
        ));

        for issue in &results.issues {
            output.push_str(&format!(
                "[{}] {}: {} ({})\n",
                issue.severity_str(),
                issue.id,
                issue.message,
                issue.rule_name
            ));
            if let Some(path) = &issue.file_path {
                if let (Some(line), Some(col)) = (issue.line_number, issue.column_number) {
                    output.push_str(&format!("  at {}:{}:{}\n", path.display(), line, col));
                } else {
                    output.push_str(&format!("  in {}\n", path.display()));
                }
            }
            output.push('\n');
        }

        Ok(output)
    }

    fn generate_json(&self, results: &LintResults) -> Result<String> {
        let json_results = json!({
            "summary": {
                "files_checked": results.files_checked,
                "rules_executed": results.rules_executed,
                "total_issues": results.issues.len()
            },
            "issues": results.issues.iter().map(|issue| {
                json!({
                    "id": issue.id,
                    "severity": format!("{:?}", issue.severity),
                    "message": issue.message,
                    "rule": issue.rule_name,
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "column_number": issue.column_number
                })
            }).collect::<Vec<_>>()
        });

        serde_json::to_string_pretty(&json_results)
            .map_err(|e| crate::LinterError::Parse(e.to_string()))
    }

    fn generate_sarif(&self, results: &LintResults) -> Result<String> {
        // Simplified SARIF format
        let sarif = json!({
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "betanet-linter",
                        "version": "0.1.0"
                    }
                },
                "results": results.issues.iter().map(|issue| {
                    json!({
                        "ruleId": issue.rule_name,
                        "message": {
                            "text": issue.message
                        },
                        "level": match issue.severity {
                            crate::SeverityLevel::Critical => "error",
                            crate::SeverityLevel::Error => "error",
                            crate::SeverityLevel::Warning => "warning",
                            crate::SeverityLevel::Info => "note"
                        }
                    })
                }).collect::<Vec<_>>()
            }]
        });

        serde_json::to_string_pretty(&sarif).map_err(|e| crate::LinterError::Parse(e.to_string()))
    }
}

trait SeverityStr {
    fn severity_str(&self) -> &'static str;
}

impl SeverityStr for crate::LintIssue {
    fn severity_str(&self) -> &'static str {
        match self.severity {
            crate::SeverityLevel::Critical => "CRITICAL",
            crate::SeverityLevel::Error => "ERROR",
            crate::SeverityLevel::Warning => "WARNING",
            crate::SeverityLevel::Info => "INFO",
        }
    }
}
