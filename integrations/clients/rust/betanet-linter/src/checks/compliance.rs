//! Compliance checking rules

use crate::checks::{CheckContext, CheckRule};
use crate::{LintIssue, Result, SeverityLevel};
use async_trait::async_trait;
use regex::Regex;

/// Section 11 compliance checker.
///
/// Currently validates two concrete rules:
///  * Legacy transition headers must not appear in code.
///  * Targets declared via `target("...")` must not point to legacy or insecure nodes.
pub struct Section11ComplianceChecker;

#[async_trait]
impl CheckRule for Section11ComplianceChecker {
    fn name(&self) -> &str {
        "section-11-compliance"
    }

    fn description(&self) -> &str {
        "Check Section 11 compliance requirements"
    }

    async fn check(&self, context: &CheckContext) -> Result<Vec<LintIssue>> {
        let mut issues = vec![];

        let legacy_header_re = Regex::new(r"legacy_transition_header").unwrap();
        let target_re = Regex::new(r#"target\((?P<q>"|')(?P<val>[^"']+)(?P=q)\)"#).unwrap();

        for (line_num, line) in context.lines() {
            if legacy_header_re.is_match(line) {
                issues.push(
                    LintIssue::new(
                        "SEC11-LEGACY".to_string(),
                        SeverityLevel::Error,
                        "Legacy transition header usage detected".to_string(),
                        self.name().to_string(),
                    )
                    .with_location(context.file_path.clone(), line_num, 0),
                );
            }

            if let Some(caps) = target_re.captures(line) {
                let target = &caps["val"];
                if target.contains("legacy") || target.contains("insecure") {
                    issues.push(
                        LintIssue::new(
                            "SEC11-TARGET".to_string(),
                            SeverityLevel::Error,
                            format!("Disallowed target `{}` in configuration", target),
                            self.name().to_string(),
                        )
                        .with_location(context.file_path.clone(), line_num, 0),
                    );
                }
            }
        }

        Ok(issues)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn ctx(content: &str) -> CheckContext {
        CheckContext {
            file_path: PathBuf::from("test.rs"),
            content: content.to_string(),
        }
    }

    #[tokio::test]
    async fn flags_legacy_header() {
        let rule = Section11ComplianceChecker;
        let src = "fn main() { legacy_transition_header(); }";
        let issues = rule.check(&ctx(src)).await.unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].id, "SEC11-LEGACY");
        assert_eq!(issues[0].line_number, Some(1));
    }

    #[tokio::test]
    async fn flags_insecure_target() {
        let rule = Section11ComplianceChecker;
        let src = "target(\"legacy-node\")";
        let issues = rule.check(&ctx(src)).await.unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].id, "SEC11-TARGET");
    }

    #[tokio::test]
    async fn passes_compliant_code() {
        let rule = Section11ComplianceChecker;
        let src = "target(\"secure-node\")\nfn main() { }";
        let issues = rule.check(&ctx(src)).await.unwrap();
        assert!(issues.is_empty());
    }
}
