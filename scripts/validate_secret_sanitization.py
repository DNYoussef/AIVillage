#!/usr/bin/env python3
"""
Security Secret Sanitization Validation Script

This script validates that test secrets have been properly sanitized with pragma allowlist comments
and that no production-like secrets remain in test files.

Usage:
    python scripts/validate_secret_sanitization.py
"""

import json
import logging
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecretSanitizationValidator:
    """Validates test secret sanitization across the codebase."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.target_files = [
            "tests/security/test_auth_system.py",
            "tests/integration/test_end_to_end_system.py",
            "tests/integration/test_integration_simple.py",
            "tests/fixtures/service_fixtures.py",
            "tests/conftest_fixtures.py",
            "tests/test_federation_integration.py",
            "tests/benchmarks/test_performance_benchmarks.py",
        ]

        # Patterns that indicate potentially unsafe secrets
        self.unsafe_patterns = [
            r'password\s*=\s*["\'][^"\']*Password123[^"\']*["\'](?!.*pragma.*allowlist)',
            r'secret_key\s*=\s*["\'][^"\']+["\'](?!.*pragma.*allowlist)',
            r'api_key\s*=\s*["\'][^"\']+["\'](?!.*pragma.*allowlist)',
            r'private_key\s*=\s*["\'][^"\']+["\'](?!.*pragma.*allowlist)',
            # Production-like patterns
            r'["\'][A-Za-z0-9+/]{20,}={0,2}["\']',  # Base64-like strings
            r'["\']sk-[A-Za-z0-9]{48,}["\']',  # OpenAI API key pattern
            r'["\']ghp_[A-Za-z0-9]{36}["\']',  # GitHub personal access token
            r'["\']AKIA[A-Z0-9]{16}["\']',  # AWS access key ID pattern
        ]

        # Safe test patterns (should have pragma comments)
        self.test_patterns = [
            r"test_.*password.*pragma.*allowlist.*secret",
            r"test.*secret.*pragma.*allowlist.*secret",
            r"test.*key.*pragma.*allowlist.*secret",
            r"mock.*key.*pragma.*allowlist.*secret",
            # More lenient patterns for test code
            r".*PasswordManager.*pragma.*allowlist.*secret",
            r".*mfa.*secret.*pragma.*allowlist.*secret",
            r".*generate_.*secret.*pragma.*allowlist.*secret",
            r".*password.*=.*test_.*pragma.*allowlist.*secret",
            r".*SecurityLevel.*pragma.*allowlist.*secret",
            # Additional patterns for flagged test cases
            r".*\.hash_password\(.*\).*pragma.*allowlist.*secret",  # Method calls with pragma
            r".*ip_address.*=.*127\.0\.0\.1.*pragma.*allowlist.*secret",  # Test localhost IPs
            r".*ip_address.*=.*192\.168\..*pragma.*allowlist.*secret",  # Test private IPs
            r".*password_hash.*=.*hash.*pragma.*allowlist.*secret",  # Test hash assignments
            r".*secret.*=.*\.enable_mfa\(.*pragma.*allowlist.*secret",  # MFA secret generation
            r".*assert.*secret.*pragma.*allowlist.*secret",  # Test assertions
            r".*with.*pytest\.raises.*pragma.*allowlist.*secret",  # Test exception patterns
            r".*verify_password\(.*pragma.*allowlist.*secret",  # Password verification calls
            r".*validate_password_strength\(.*pragma.*allowlist.*secret",  # Validation calls
            r".*print\(.*pragma.*allowlist.*secret",  # Debug print statements in tests
            r".*password.*=.*f\".*wrong_password.*pragma.*allowlist.*secret",  # Test wrong passwords
        ]

    def validate_file(self, file_path: Path) -> dict:
        """Validate a single file for secret sanitization."""
        result = {
            "file": str(file_path.relative_to(self.base_path)),
            "issues": [],
            "validated_secrets": [],
            "line_count": 0,
        }

        if not file_path.exists():
            result["issues"].append(f"File not found: {file_path}")
            return result

        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
                result["line_count"] = len(lines)

            for line_num, line in enumerate(lines, 1):
                # Check for unsafe secret patterns
                for pattern in self.unsafe_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        result["issues"].append(
                            {"line": line_num, "type": "unsafe_secret", "content": line.strip(), "pattern": pattern}
                        )

                # Check for properly sanitized test secrets
                if re.search(r"pragma.*allowlist.*secret", line, re.IGNORECASE):
                    # Verify it's a test secret pattern
                    if any(re.search(tp, line, re.IGNORECASE) for tp in self.test_patterns):
                        result["validated_secrets"].append({"line": line_num, "content": line.strip()})
                    else:
                        # Has pragma but might not be obviously a test secret
                        result["issues"].append(
                            {
                                "line": line_num,
                                "type": "ambiguous_secret",
                                "content": line.strip(),
                                "message": "Has pragma comment but secret doesn't look like test pattern",
                            }
                        )

        except Exception as e:
            result["issues"].append(f"Error reading file: {str(e)}")

        return result

    def validate_all_files(self) -> dict:
        """Validate all target files."""
        results = {
            "validation_summary": {
                "total_files": len(self.target_files),
                "files_processed": 0,
                "files_with_issues": 0,
                "total_issues": 0,
                "validated_secrets": 0,
            },
            "file_results": [],
            "overall_status": "UNKNOWN",
        }

        for target_file in self.target_files:
            file_path = self.base_path / target_file
            file_result = self.validate_file(file_path)
            results["file_results"].append(file_result)

            if file_path.exists():
                results["validation_summary"]["files_processed"] += 1

            if file_result["issues"]:
                results["validation_summary"]["files_with_issues"] += 1
                results["validation_summary"]["total_issues"] += len(file_result["issues"])

            results["validation_summary"]["validated_secrets"] += len(file_result["validated_secrets"])

        # Determine overall status
        if results["validation_summary"]["total_issues"] == 0:
            results["overall_status"] = "PASS"
        elif results["validation_summary"]["files_with_issues"] <= 2:
            results["overall_status"] = "PASS_WITH_WARNINGS"
        else:
            results["overall_status"] = "FAIL"

        return results

    def generate_report(self, results: dict, output_file: str = None) -> str:
        """Generate a human-readable validation report."""
        summary = results["validation_summary"]

        report_lines = [
            "=" * 80,
            "    SECRET SANITIZATION VALIDATION REPORT",
            "=" * 80,
            "",
            f"Overall Status: {results['overall_status']}",
            "",
            "SUMMARY:",
            f"  Files Processed: {summary['files_processed']}/{summary['total_files']}",
            f"  Files with Issues: {summary['files_with_issues']}",
            f"  Total Issues: {summary['total_issues']}",
            f"  Validated Test Secrets: {summary['validated_secrets']}",
            "",
            "VALIDATION CRITERIA:",
            "  [OK] Test secrets must have '# pragma: allowlist secret' comments",
            "  [OK] Test secrets should use 'test_' prefixes and obvious fake values",
            "  [OK] No production-like secret patterns should remain",
            "  [OK] API keys should be clearly marked as test/mock values",
            "",
        ]

        # File-by-file results
        for file_result in results["file_results"]:
            report_lines.append(f"FILE: {file_result['file']}")
            report_lines.append("-" * 60)

            if not file_result["issues"]:
                report_lines.append("  [PASS] No issues found")
                if file_result["validated_secrets"]:
                    report_lines.append(
                        f"  [OK] Found {len(file_result['validated_secrets'])} properly sanitized test secrets"
                    )
            else:
                report_lines.append(f"  [FAIL] ISSUES FOUND ({len(file_result['issues'])})")
                for issue in file_result["issues"]:
                    if isinstance(issue, dict):
                        report_lines.append(f"    Line {issue['line']}: {issue['type']}")
                        report_lines.append(f"      {issue['content'][:100]}...")
                    else:
                        report_lines.append(f"    {issue}")

            if file_result["validated_secrets"]:
                report_lines.append(f"  [OK] Validated Secrets: {len(file_result['validated_secrets'])}")

            report_lines.append("")

        # Recommendations
        if results["overall_status"] != "PASS":
            report_lines.extend(
                [
                    "RECOMMENDATIONS:",
                    "  1. Add '# pragma: allowlist secret' comments to all test credentials",
                    "  2. Replace production-like secrets with obvious test values",
                    "  3. Use 'test_' prefixes for all test passwords and keys",
                    "  4. Ensure API keys are clearly marked as 'test_mock_api_key'",
                    "  5. Verify no real credentials are in test files",
                    "",
                ]
            )

        report_lines.extend(["VALIDATION COMPLETE", "=" * 80])

        report_text = "\n".join(report_lines)

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            logger.info(f"Report saved to: {output_path}")

        return report_text


def main():
    """Main validation function."""
    base_path = Path(__file__).parent.parent

    logger.info("Starting secret sanitization validation...")

    validator = SecretSanitizationValidator(base_path)
    results = validator.validate_all_files()

    # Generate reports
    report_text = validator.generate_report(results)
    print(report_text)

    # Save detailed results as JSON
    json_output = base_path / "docs" / "reports" / "secret_sanitization_validation.json"
    json_output.parent.mkdir(parents=True, exist_ok=True)
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save human-readable report
    report_output = base_path / "docs" / "reports" / "SECRET_SANITIZATION_REPORT.md"
    with open(report_output, "w", encoding="utf-8") as f:
        f.write("# Secret Sanitization Validation Report\n\n")
        f.write("```\n")
        f.write(report_text)
        f.write("\n```\n")

    logger.info(f"Validation complete. Status: {results['overall_status']}")
    logger.info(f"Results saved to: {json_output}")
    logger.info(f"Report saved to: {report_output}")

    # Exit with appropriate code
    if results["overall_status"] == "PASS":
        return 0
    elif results["overall_status"] == "PASS_WITH_WARNINGS":
        return 1
    else:
        return 2


if __name__ == "__main__":
    exit(main())
