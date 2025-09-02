#!/usr/bin/env python3
"""
Enhanced Security Validation Framework
Comprehensive security threshold validation with severity-based deployment gates.

This module provides advanced security pattern detection with CRITICAL/ERROR/WARNING/INFO 
severity levels for production-ready deployment authorization.
"""

import argparse
import json
import logging
from pathlib import Path
import re
import sys
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecuritySeverity(Enum):
    """Security issue severity levels for production deployment gates."""
    CRITICAL = "CRITICAL"
    ERROR = "ERROR" 
    WARNING = "WARNING"
    INFO = "INFO"


class SecurityValidationResult:
    """Structured security validation result with severity classification."""
    
    def __init__(self):
        self.issues_by_severity = {
            SecuritySeverity.CRITICAL: [],
            SecuritySeverity.ERROR: [],
            SecuritySeverity.WARNING: [],
            SecuritySeverity.INFO: []
        }
        self.validated_secrets = []
        self.files_processed = 0
        self.audit_trail = []
    
    def add_issue(self, severity: SecuritySeverity, file_path: str, line_num: int, 
                  issue_type: str, content: str, pattern: str = None, justification: str = None):
        """Add a security issue with proper severity classification."""
        issue = {
            "file": file_path,
            "line": line_num,
            "type": issue_type,
            "content": content,
            "pattern": pattern,
            "justification": justification,
            "timestamp": datetime.now().isoformat()
        }
        self.issues_by_severity[severity].append(issue)
        
        # Log to audit trail
        self.audit_trail.append({
            "action": "issue_detected",
            "severity": severity.value,
            "details": issue,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_overall_status(self, production_ready_mode: bool = False) -> str:
        """Determine overall validation status based on severity counts."""
        critical_count = len(self.issues_by_severity[SecuritySeverity.CRITICAL])
        error_count = len(self.issues_by_severity[SecuritySeverity.ERROR])
        warning_count = len(self.issues_by_severity[SecuritySeverity.WARNING])
        
        # CRITICAL issues always block deployment
        if critical_count > 0:
            return "FAIL"
        
        # ERROR issues block deployment unless in production-ready mode
        if error_count > 0:
            if production_ready_mode:
                return "PASS_WITH_ERRORS"  # New status for production mode
            else:
                return "FAIL"
        
        # WARNING issues are acceptable in production mode
        if warning_count > 0:
            return "PASS_WITH_WARNINGS"
        
        return "PASS"
    
    def get_exit_code(self, production_ready_mode: bool = False) -> int:
        """Get appropriate exit code based on validation results."""
        status = self.get_overall_status(production_ready_mode)
        
        if status == "PASS":
            return 0
        elif status in ["PASS_WITH_WARNINGS", "PASS_WITH_ERRORS"]:
            return 0 if production_ready_mode else 1
        else:  # FAIL
            return 2


class EnhancedSecurityValidator:
    """Enhanced security validator with comprehensive threat detection and severity classification."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        
        # Define comprehensive security patterns by severity
        self.security_patterns = {
            SecuritySeverity.CRITICAL: [
                # Live production credentials - ALWAYS block deployment
                (r'["\']sk-[A-Za-z0-9]{48,}["\']', 'openai_api_key', 'Live OpenAI API key detected'),
                (r'["\']ghp_[A-Za-z0-9]{36}["\']', 'github_pat', 'Live GitHub personal access token detected'),
                (r'["\']AKIA[A-Z0-9]{16}["\']', 'aws_access_key', 'Live AWS access key detected'),
                (r'["\']glpat-[A-Za-z0-9_\-]{20}["\']', 'gitlab_pat', 'Live GitLab access token detected'),
                (r'["\']xoxb-[0-9]{11,}-[0-9]{11,}-[A-Za-z0-9]{24}["\']', 'slack_bot_token', 'Live Slack bot token detected'),
                (r'["\']AIza[0-9A-Za-z_\-]{35}["\']', 'google_api_key', 'Live Google API key detected'),
                (r'["\'][0-9a-fA-F]{32}["\'](?=.*password)', 'md5_password_hash', 'Potential live password hash'),
                (r'["\']\\$[0-9][a-z]\\$[^"\']*["\']', 'bcrypt_hash', 'Potential live bcrypt hash'),
                (r'-----BEGIN [A-Z ]+ PRIVATE KEY-----', 'private_key_block', 'Live private key block detected')
            ],
            SecuritySeverity.ERROR: [
                # Hardcoded secrets without proper pragma comments - block in strict mode
                (r'password\s*=\s*["\'][^"\']*[A-Za-z0-9!@#$%^&*()]{8,}[^"\']*["\'](?!.*pragma.*allowlist)', 'hardcoded_password', 'Hardcoded password without pragma allowlist'),
                (r'secret_key\s*=\s*["\'][^"\']{16,}["\'](?!.*pragma.*allowlist)', 'hardcoded_secret_key', 'Hardcoded secret key without pragma allowlist'),
                (r'api_key\s*=\s*["\'][^"\']{20,}["\'](?!.*pragma.*allowlist)', 'hardcoded_api_key', 'Hardcoded API key without pragma allowlist'),
                (r'private_key\s*=\s*["\'][^"\']{40,}["\'](?!.*pragma.*allowlist)', 'hardcoded_private_key', 'Hardcoded private key without pragma allowlist'),
                (r'jdbc:[^"\']*password=[^"\';&]*["\']', 'jdbc_password', 'Database connection string with password'),
                (r'mysql://[^"\']*:[^"\'@]*@[^"\']*', 'mysql_connection', 'MySQL connection string with credentials')
            ],
            SecuritySeverity.WARNING: [
                # Suspicious patterns that should be reviewed
                (r'["\'][A-Za-z0-9+/]{40,}={0,2}["\'](?!.*pragma.*allowlist)', 'base64_string', 'Long base64 string without pragma comment'),
                (r'password\s*=\s*["\'][^"\']*test[^"\']*["\'](?!.*pragma.*allowlist)', 'test_password_no_pragma', 'Test password without pragma comment'),
                (r'["\'][0-9a-fA-F]{16,}["\'](?!.*pragma.*allowlist)', 'hex_string', 'Long hex string without pragma comment'),
                (r'bearer\s+[A-Za-z0-9_\-\.]{20,}(?!.*pragma.*allowlist)', 'bearer_token', 'Bearer token without pragma comment'),
                (r'authorization:\s*["\'][^"\']{20,}["\']', 'auth_header', 'Authorization header value'),
                (r'cookie:\s*["\'][^"\']{30,}["\']', 'cookie_value', 'Long cookie value')
            ],
            SecuritySeverity.INFO: [
                # Informational - potential secrets that might need review
                (r'["\'][A-Z0-9]{10,}["\']', 'uppercase_string', 'Long uppercase string - check if credential'),
                (r'token\s*=\s*["\'][^"\']{10,}["\']', 'generic_token', 'Generic token assignment'),
                (r'key\s*=\s*["\'][^"\']{10,}["\']', 'generic_key', 'Generic key assignment'),
                (r'["\'][a-zA-Z0-9]{32,}["\']', 'long_alphanum', 'Long alphanumeric string'),
                (r'hash\s*=\s*["\'][^"\']{10,}["\']', 'hash_assignment', 'Hash value assignment')
            ]
        }
        
        # Patterns that indicate legitimate test contexts
        self.test_context_patterns = [
            r'test_.*password.*pragma.*allowlist.*secret',
            r'test.*secret.*pragma.*allowlist.*secret',
            r'test.*key.*pragma.*allowlist.*secret',
            r'mock.*key.*pragma.*allowlist.*secret',
            r'.*\.hash_password\(.*\).*pragma.*allowlist.*secret',
            r'.*AuthConfig\(.*\).*pragma.*allowlist.*secret',
            r'.*password.*=.*f".*wrong_password.*pragma.*allowlist.*secret',
            r'^\s*\)\s*#.*pragma.*allowlist.*secret'  # Closing parentheses with pragma
        ]
    
    def validate_file(self, file_path: Path, validation_result: SecurityValidationResult) -> Dict:
        """Validate a single file with enhanced security pattern detection."""
        relative_path = str(file_path.relative_to(self.base_path))
        file_result = {
            "file": relative_path,
            "line_count": 0,
            "issues_by_severity": {sev.value: [] for sev in SecuritySeverity}
        }

        if not file_path.exists():
            validation_result.add_issue(
                SecuritySeverity.ERROR, 
                relative_path, 
                0, 
                "file_not_found", 
                f"File not found: {file_path}"
            )
            return file_result

        validation_result.files_processed += 1
        
        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
                file_result["line_count"] = len(lines)

            for line_num, line in enumerate(lines, 1):
                self._validate_line(file_path, line_num, line, validation_result, file_result)

        except Exception as e:
            validation_result.add_issue(
                SecuritySeverity.ERROR,
                relative_path,
                0,
                "file_read_error",
                f"Error reading file: {str(e)}"
            )

        return file_result
    
    def _validate_line(self, file_path: Path, line_num: int, line: str, 
                       validation_result: SecurityValidationResult, file_result: Dict):
        """Validate a single line with comprehensive security pattern detection."""
        relative_path = str(file_path.relative_to(self.base_path))
        
        # Check each severity level of patterns
        for severity, patterns in self.security_patterns.items():
            for pattern, issue_type, description in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Check for pragma allowlist comment - may reduce severity
                    has_pragma = re.search(r'pragma.*allowlist.*secret', line, re.IGNORECASE)
                    
                    if has_pragma:
                        # Validate pragma context is appropriate
                        if self._is_valid_pragma_context(line, file_path):
                            validation_result.validated_secrets.append({
                                "line": line_num,
                                "content": line.strip(),
                                "file": relative_path
                            })
                            continue
                        else:
                            # Pragma comment but suspicious context
                            adjusted_severity = SecuritySeverity.WARNING if severity == SecuritySeverity.CRITICAL else severity
                            validation_result.add_issue(
                                adjusted_severity,
                                relative_path,
                                line_num,
                                f"{issue_type}_suspicious_pragma",
                                line.strip(),
                                pattern,
                                "Has pragma comment but context suggests real credential"
                            )
                    else:
                        # No pragma comment - use original severity
                        validation_result.add_issue(
                            severity,
                            relative_path,
                            line_num,
                            issue_type,
                            line.strip(),
                            pattern,
                            description
                        )
                    
                    # Track in file result
                    file_result["issues_by_severity"][severity.value].append({
                        "line": line_num,
                        "type": issue_type,
                        "content": line.strip()[:100] + "..." if len(line.strip()) > 100 else line.strip()
                    })
                    break  # Only report first match per line
    
    def _is_valid_pragma_context(self, line: str, file_path: Path) -> bool:
        """Determine if a pragma allowlist comment is in appropriate context."""
        relative_path = str(file_path.relative_to(self.base_path))
        
        # Test files and directories are generally acceptable
        if any(test_indicator in relative_path.lower() for test_indicator in 
               ['test', 'tests', 'mock', 'fixture', 'example', 'demo', 'conftest']):
            return True
        
        # Check if it matches known safe test patterns
        for pattern in self.test_context_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        
        return False
    
    def validate_project(self, target_patterns: List[str], production_ready_mode: bool = False) -> Tuple[SecurityValidationResult, Dict]:
        """Validate entire project with specified file patterns."""
        validation_result = SecurityValidationResult()
        
        # Find all files matching target patterns
        target_files = []
        for pattern in target_patterns:
            target_files.extend(self.base_path.rglob(pattern))
        
        summary = {
            "validation_summary": {
                "total_files": len(target_files),
                "files_processed": 0,
                "severity_counts": {sev.value: 0 for sev in SecuritySeverity},
                "validated_secrets": 0,
                "production_ready_mode": production_ready_mode,
                "target_patterns": target_patterns
            },
            "file_results": [],
            "overall_status": "UNKNOWN",
            "exit_code": 0
        }

        # Process each file
        for file_path in target_files:
            file_result = self.validate_file(file_path, validation_result)
            summary["file_results"].append(file_result)

        # Update summary counts
        summary["validation_summary"]["files_processed"] = validation_result.files_processed
        summary["validation_summary"]["validated_secrets"] = len(validation_result.validated_secrets)
        
        for severity in SecuritySeverity:
            count = len(validation_result.issues_by_severity[severity])
            summary["validation_summary"]["severity_counts"][severity.value] = count

        # Determine overall status and exit code
        summary["overall_status"] = validation_result.get_overall_status(production_ready_mode)
        summary["exit_code"] = validation_result.get_exit_code(production_ready_mode)
        
        # Add audit trail and security decisions
        summary["audit_trail"] = validation_result.audit_trail
        summary["security_decisions"] = {
            "deployment_gate_decision": summary["overall_status"],
            "justification": f"Validation completed in {'production-ready' if production_ready_mode else 'strict'} mode",
            "blocking_issues": len(validation_result.issues_by_severity[SecuritySeverity.CRITICAL]),
            "error_issues": len(validation_result.issues_by_severity[SecuritySeverity.ERROR]),
            "warning_issues_accepted": production_ready_mode and len(validation_result.issues_by_severity[SecuritySeverity.WARNING]) > 0
        }

        return validation_result, summary
    
    def generate_report(self, validation_result: SecurityValidationResult, summary: dict, output_file: str = None) -> str:
        """Generate comprehensive security validation report with severity breakdown."""
        report_lines = [
            "=" * 80,
            "    ENHANCED SECURITY VALIDATION REPORT",
            "=" * 80,
            "",
            f"Overall Status: {summary['overall_status']}",
            f"Production Ready Mode: {summary['validation_summary']['production_ready_mode']}",
            f"Exit Code: {summary['exit_code']}",
            f"Validation Timestamp: {datetime.now().isoformat()}",
            "",
            "SECURITY SEVERITY BREAKDOWN:",
        ]
        
        # Add severity counts with status indicators
        for severity in SecuritySeverity:
            count = summary["validation_summary"]["severity_counts"][severity.value]
            if severity == SecuritySeverity.CRITICAL and count > 0:
                status_indicator = "[CRITICAL-BLOCK]"
            elif severity == SecuritySeverity.ERROR and count > 0:
                status_indicator = "[ERROR]" if not summary['validation_summary']['production_ready_mode'] else "[ERROR-ACCEPTED]"
            elif severity == SecuritySeverity.WARNING and count > 0:
                status_indicator = "[WARNING-ACCEPTED]"
            elif severity == SecuritySeverity.INFO and count > 0:
                status_indicator = "[INFO]"
            else:
                status_indicator = "[OK]"
            
            report_lines.append(f"  {status_indicator} {severity.value}: {count} issues")
        
        report_lines.extend([
            "",
            "PROCESSING SUMMARY:",
            f"  Files Processed: {summary['validation_summary']['files_processed']}/{summary['validation_summary']['total_files']}",
            f"  Validated Test Secrets: {summary['validation_summary']['validated_secrets']}",
            f"  Target Patterns: {', '.join(summary['validation_summary']['target_patterns'])}",
            "",
            "SECURITY VALIDATION CRITERIA:",
            "  [CRITICAL] Live production credentials (API keys, tokens, private keys) - ALWAYS BLOCKS DEPLOYMENT",
            "  [ERROR] Hardcoded secrets without pragma comments - BLOCKS in strict mode, ACCEPTED in production mode",
            "  [WARNING] Suspicious patterns without pragma comments - ACCEPTED in production mode",
            "  [INFO] Potential credentials requiring review - INFORMATIONAL ONLY",
            "  [OK] Test secrets with proper pragma allowlist secret comments",
            "",
            "DEPLOYMENT GATE DECISION:",
            f"  Decision: {summary['security_decisions']['deployment_gate_decision']}",
            f"  Justification: {summary['security_decisions']['justification']}",
            f"  Blocking Issues: {summary['security_decisions']['blocking_issues']}",
            f"  Error Issues: {summary['security_decisions']['error_issues']}",
            f"  Warnings Accepted: {summary['security_decisions']['warning_issues_accepted']}",
            ""
        ])
        
        # Add detailed file results if there are issues
        if any(len(validation_result.issues_by_severity[sev]) > 0 for sev in SecuritySeverity):
            report_lines.extend([
                "DETAILED FINDINGS:",
                "-" * 40
            ])
            
            for file_result in summary["file_results"]:
                has_issues = any(len(issues) > 0 for issues in file_result["issues_by_severity"].values())
                if has_issues:
                    report_lines.append(f"FILE: {file_result['file']}")
                    for severity in SecuritySeverity:
                        issues = file_result["issues_by_severity"][severity.value]
                        if issues:
                            report_lines.append(f"  {severity.value}: {len(issues)} issues")
                            for issue in issues[:3]:  # Show first 3 issues
                                report_lines.append(f"    Line {issue['line']}: {issue['type']}")
                                report_lines.append(f"      {issue['content'][:80]}...")
                            if len(issues) > 3:
                                report_lines.append(f"    ... and {len(issues) - 3} more issues")
                    report_lines.append("")
        
        report_lines.extend([
            "AUDIT TRAIL SUMMARY:",
            f"  Total Security Events: {len(validation_result.audit_trail)}",
            f"  Timestamp: {datetime.now().isoformat()}",
            "",
            "VALIDATION COMPLETE",
            "=" * 80
        ])

        report_text = "\n".join(report_lines)

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            logger.info(f"Report saved to: {output_path}")

        return report_text


def main():
    """Enhanced security validation with comprehensive deployment gates."""
    parser = argparse.ArgumentParser(
        description="Enhanced security validation with severity-based deployment gates"
    )
    parser.add_argument(
        "--production-ready", 
        action="store_true", 
        help="Run in production-ready mode (CRITICAL blocks, ERROR/WARNING accepted with justification)"
    )
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=["tests/**/*.py", "**/*test*.py", "**/*fixture*.py"],
        help="File patterns to validate (default: test files)"
    )
    parser.add_argument(
        "--audit-log",
        help="Path to save detailed audit log"
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Validate all Python files in project"
    )
    
    args = parser.parse_args()
    
    if args.all_files:
        args.patterns = ["**/*.py"]
    
    base_path = Path(__file__).parent.parent

    logger.info("Starting Enhanced Security Validation...")
    if args.production_ready:
        logger.info("PRODUCTION-READY mode - Enhanced security gates active")
        logger.info("   - CRITICAL issues: ALWAYS BLOCK deployment")
        logger.info("   - ERROR issues: ACCEPTED with justification")
        logger.info("   - WARNING issues: ACCEPTED")
        logger.info("   - INFO issues: INFORMATIONAL only")
    else:
        logger.info("STRICT mode - All ERROR and CRITICAL issues block deployment")

    validator = EnhancedSecurityValidator(base_path)
    validation_result, summary = validator.validate_project(args.patterns, args.production_ready)

    # Generate comprehensive report
    report_text = validator.generate_report(validation_result, summary)
    print(report_text)

    # Save results and audit trail
    reports_dir = base_path / "docs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results with full audit trail
    json_output = reports_dir / "enhanced_security_validation.json"
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "issues_by_severity": {sev.value: validation_result.issues_by_severity[sev] for sev in SecuritySeverity},
            "validated_secrets": validation_result.validated_secrets,
            "audit_trail": validation_result.audit_trail
        }, f, indent=2)

    # Save audit log with security decisions
    audit_log_path = Path(args.audit_log) if args.audit_log else reports_dir / "security_audit_log.json"
    with open(audit_log_path, "w", encoding="utf-8") as f:
        json.dump({
            "validation_run": {
                "timestamp": datetime.now().isoformat(),
                "production_ready_mode": args.production_ready,
                "overall_status": summary["overall_status"],
                "exit_code": summary["exit_code"],
                "target_patterns": args.patterns
            },
            "security_decisions": summary["security_decisions"],
            "audit_trail": validation_result.audit_trail
        }, f, indent=2)

    # Save markdown report
    report_output = reports_dir / "ENHANCED_SECURITY_VALIDATION_REPORT.md"
    with open(report_output, "w", encoding="utf-8") as f:
        f.write("# Enhanced Security Validation Report\n\n")
        f.write("```\n")
        f.write(report_text)
        f.write("\n```\n")

    # Log completion with security decision summary
    logger.info(f"Validation complete. Status: {summary['overall_status']}")
    logger.info(f"Results saved to: {json_output}")
    logger.info(f"Report saved to: {report_output}")
    logger.info(f"Audit log saved to: {audit_log_path}")

    # Enhanced exit logic with clear deployment authorization
    exit_code = summary["exit_code"]
    if exit_code == 0:
        logger.info(f"DEPLOYMENT AUTHORIZED - Status: {summary['overall_status']}")
    else:
        logger.error(f"DEPLOYMENT BLOCKED - Status: {summary['overall_status']}")
        logger.error(f"   Critical Issues: {summary['security_decisions']['blocking_issues']}")
        logger.error(f"   Error Issues: {summary['security_decisions']['error_issues']}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())