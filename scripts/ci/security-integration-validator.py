#!/usr/bin/env python3
"""
Security Integration Validator
Validates that all security enhancements are properly integrated into CI/CD pipelines.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path
import subprocess  # nosec B404
import sys

import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    PASS = "PASS"  # nosec B105 - enum value, not password
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIP = "SKIP"


@dataclass
class ValidationCheck:
    name: str
    description: str
    category: str
    result: ValidationResult
    message: str
    recommendation: str | None = None


class SecurityIntegrationValidator:
    """Validates security integration across CI/CD pipelines."""

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.workflows_dir = self.root_dir / ".github" / "workflows"
        self.config_dir = self.root_dir / "config"
        self.scripts_dir = self.root_dir / "scripts"
        self.checks = []

    def validate_workflow_files(self) -> list[ValidationCheck]:
        """Validate GitHub workflow files have security enhancements."""
        checks = []

        expected_workflows = [
            "main-ci.yml",
            "scion_production.yml",
            "architectural-quality.yml",
            "image-security-scan.yml",
            "p2p-test-suite.yml",
        ]

        for workflow in expected_workflows:
            workflow_path = self.workflows_dir / workflow
            if not workflow_path.exists():
                checks.append(
                    ValidationCheck(
                        name=f"workflow_exists_{workflow}",
                        description=f"Workflow file {workflow} exists",
                        category="workflows",
                        result=ValidationResult.FAIL,
                        message=f"Workflow file {workflow} not found",
                        recommendation=f"Create {workflow} workflow file",
                    )
                )
                continue

            # Validate workflow content
            try:
                with open(workflow_path, encoding="utf-8") as f:
                    content = f.read()

                workflow_checks = self._validate_workflow_content(workflow, content)
                checks.extend(workflow_checks)

            except Exception as e:
                checks.append(
                    ValidationCheck(
                        name=f"workflow_readable_{workflow}",
                        description=f"Workflow file {workflow} is readable",
                        category="workflows",
                        result=ValidationResult.FAIL,
                        message=f"Could not read {workflow}: {e}",
                        recommendation="Check file permissions and syntax",
                    )
                )

        return checks

    def _validate_workflow_content(self, workflow_name: str, content: str) -> list[ValidationCheck]:
        """Validate specific workflow content for security enhancements."""
        checks = []

        # Common security requirements for all workflows
        common_requirements = {
            "detect-secrets": "detect-secrets tool integration",
            "bandit": "Bandit security scanning",
            "safety": "Safety dependency scanning",
            "security-gate": "Security gate validation",
        }

        for requirement, description in common_requirements.items():
            if requirement in content:
                checks.append(
                    ValidationCheck(
                        name=f"{workflow_name}_{requirement}",
                        description=f"{workflow_name} includes {description}",
                        category="security_integration",
                        result=ValidationResult.PASS,
                        message=f"✅ {description} found in {workflow_name}",
                    )
                )
            else:
                checks.append(
                    ValidationCheck(
                        name=f"{workflow_name}_{requirement}",
                        description=f"{workflow_name} includes {description}",
                        category="security_integration",
                        result=ValidationResult.WARNING,
                        message=f"⚠️ {description} not found in {workflow_name}",
                        recommendation=f"Add {description} to {workflow_name}",
                    )
                )

        # Workflow-specific validations
        if workflow_name == "main-ci.yml":
            checks.extend(self._validate_main_ci_specifics(content))
        elif workflow_name == "scion_production.yml":
            checks.extend(self._validate_scion_production_specifics(content))
        elif workflow_name == "architectural-quality.yml":
            checks.extend(self._validate_architectural_quality_specifics(content))
        elif workflow_name == "image-security-scan.yml":
            checks.extend(self._validate_image_security_specifics(content))
        elif workflow_name == "p2p-test-suite.yml":
            checks.extend(self._validate_p2p_security_specifics(content))

        return checks

    def _validate_main_ci_specifics(self, content: str) -> list[ValidationCheck]:
        """Validate main CI workflow specific enhancements."""
        checks = []

        main_ci_requirements = [
            ("secret_detection_check", "Secret Detection Check"),
            ("cryptographic_algorithm_validation", "Cryptographic Algorithm Validation"),
            ("anti_pattern_detection", "Anti-Pattern Detection"),
            ("security_compliance_report", "Security Compliance Report"),
            ("security-gate:", "Security Gate job"),
        ]

        for requirement, description in main_ci_requirements:
            if requirement in content:
                checks.append(
                    ValidationCheck(
                        name=f"main_ci_{requirement}",
                        description=f"Main CI includes {description}",
                        category="main_ci_security",
                        result=ValidationResult.PASS,
                        message=f"✅ {description} found in main CI",
                    )
                )
            else:
                checks.append(
                    ValidationCheck(
                        name=f"main_ci_{requirement}",
                        description=f"Main CI includes {description}",
                        category="main_ci_security",
                        result=ValidationResult.FAIL,
                        message=f"❌ {description} not found in main CI",
                        recommendation=f"Add {description} to main CI workflow",
                    )
                )

        return checks

    def _validate_scion_production_specifics(self, content: str) -> list[ValidationCheck]:
        """Validate SCION production workflow specific enhancements."""
        checks = []

        scion_requirements = [
            ("security-preflight:", "Security Pre-Flight stage"),
            ("emergency_bypass", "Emergency bypass handling"),
            ("security-compliance:", "Security compliance stage"),
            ("deployment-gate:", "Deployment gate stage"),
            ("production-security-gate", "Production security gate"),
        ]

        for requirement, description in scion_requirements:
            if requirement in content:
                checks.append(
                    ValidationCheck(
                        name=f"scion_{requirement.replace(':', '')}",
                        description=f"SCION includes {description}",
                        category="scion_security",
                        result=ValidationResult.PASS,
                        message=f"✅ {description} found in SCION production",
                    )
                )
            else:
                checks.append(
                    ValidationCheck(
                        name=f"scion_{requirement.replace(':', '')}",
                        description=f"SCION includes {description}",
                        category="scion_security",
                        result=ValidationResult.FAIL,
                        message=f"❌ {description} not found in SCION production",
                        recommendation=f"Add {description} to SCION production workflow",
                    )
                )

        return checks

    def _validate_architectural_quality_specifics(self, content: str) -> list[ValidationCheck]:
        """Validate architectural quality workflow specific enhancements."""
        checks = []

        arch_requirements = [
            ("Security Pre-Check", "Security pre-check for architecture"),
            ("Enhanced Architectural Fitness", "Enhanced architectural fitness functions"),
            ("connascence", "Connascence analysis"),
            ("Enhanced Quality Gates", "Enhanced quality gates"),
            ("Enhanced Quality Gate Status Check", "Enhanced quality gate status check"),
        ]

        for requirement, description in arch_requirements:
            if requirement in content:
                checks.append(
                    ValidationCheck(
                        name=f"arch_{requirement.replace(' ', '_').lower()}",
                        description=f"Architecture workflow includes {description}",
                        category="architecture_security",
                        result=ValidationResult.PASS,
                        message=f"✅ {description} found in architecture workflow",
                    )
                )
            else:
                checks.append(
                    ValidationCheck(
                        name=f"arch_{requirement.replace(' ', '_').lower()}",
                        description=f"Architecture workflow includes {description}",
                        category="architecture_security",
                        result=ValidationResult.WARNING,
                        message=f"⚠️ {description} not found in architecture workflow",
                        recommendation=f"Add {description} to architecture workflow",
                    )
                )

        return checks

    def _validate_image_security_specifics(self, content: str) -> list[ValidationCheck]:
        """Validate image security workflow specific enhancements."""
        checks = []

        image_requirements = [
            ("Enhanced Vulnerability Analysis", "Enhanced vulnerability analysis"),
            ("security-gate:", "Security gate evaluation"),
            ("security-summary-", "Security summary generation"),
            ("Enhanced Security Gate", "Enhanced security gate"),
        ]

        for requirement, description in image_requirements:
            if requirement in content:
                checks.append(
                    ValidationCheck(
                        name=f"image_{requirement.replace(':', '').replace(' ', '_').lower()}",
                        description=f"Image security includes {description}",
                        category="image_security",
                        result=ValidationResult.PASS,
                        message=f"✅ {description} found in image security workflow",
                    )
                )
            else:
                checks.append(
                    ValidationCheck(
                        name=f"image_{requirement.replace(':', '').replace(' ', '_').lower()}",
                        description=f"Image security includes {description}",
                        category="image_security",
                        result=ValidationResult.FAIL,
                        message=f"❌ {description} not found in image security workflow",
                        recommendation=f"Add {description} to image security workflow",
                    )
                )

        return checks

    def _validate_p2p_security_specifics(self, content: str) -> list[ValidationCheck]:
        """Validate P2P test suite specific enhancements."""
        checks = []

        p2p_requirements = [
            ("security-preflight:", "Security pre-flight stage"),
            ("P2P Security Pre-Flight", "P2P security pre-flight check"),
            ("Enhanced P2P Network Security", "Enhanced P2P network security tests"),
            ("Comprehensive Security Scanning", "Comprehensive security scanning"),
            ("security-gate:", "Security gate evaluation stage"),
            ("Final Security Gate Evaluation", "Final security gate evaluation"),
        ]

        for requirement, description in p2p_requirements:
            if requirement in content:
                checks.append(
                    ValidationCheck(
                        name=f"p2p_{requirement.replace(':', '').replace(' ', '_').lower()}",
                        description=f"P2P suite includes {description}",
                        category="p2p_security",
                        result=ValidationResult.PASS,
                        message=f"✅ {description} found in P2P test suite",
                    )
                )
            else:
                checks.append(
                    ValidationCheck(
                        name=f"p2p_{requirement.replace(':', '').replace(' ', '_').lower()}",
                        description=f"P2P suite includes {description}",
                        category="p2p_security",
                        result=ValidationResult.FAIL,
                        message=f"❌ {description} not found in P2P test suite",
                        recommendation=f"Add {description} to P2P test suite",
                    )
                )

        return checks

    def validate_security_config_files(self) -> list[ValidationCheck]:
        """Validate security configuration files exist and are valid."""
        checks = []

        config_files = {
            "config/security/security-gate-config.yaml": "Security gate configuration",
            "config/security/emergency-procedures.yaml": "Emergency procedures configuration",
        }

        for config_file, description in config_files.items():
            config_path = self.root_dir / config_file

            if not config_path.exists():
                checks.append(
                    ValidationCheck(
                        name=f"config_{config_path.name}",
                        description=f"{description} file exists",
                        category="configuration",
                        result=ValidationResult.FAIL,
                        message=f"❌ {config_file} not found",
                        recommendation=f"Create {config_file}",
                    )
                )
                continue

            # Validate YAML syntax
            try:
                with open(config_path) as f:
                    yaml.safe_load(f)

                checks.append(
                    ValidationCheck(
                        name=f"config_{config_path.name}_valid",
                        description=f"{description} has valid YAML syntax",
                        category="configuration",
                        result=ValidationResult.PASS,
                        message=f"✅ {config_file} has valid syntax",
                    )
                )

            except yaml.YAMLError as e:
                checks.append(
                    ValidationCheck(
                        name=f"config_{config_path.name}_valid",
                        description=f"{description} has valid YAML syntax",
                        category="configuration",
                        result=ValidationResult.FAIL,
                        message=f"❌ {config_file} has invalid YAML: {e}",
                        recommendation=f"Fix YAML syntax in {config_file}",
                    )
                )

        return checks

    def validate_security_scripts(self) -> list[ValidationCheck]:
        """Validate security scripts exist and are executable."""
        checks = []

        script_files = {
            "scripts/ci/security-gate-validator.py": "Security gate validator script",
            "scripts/ci/compliance-reporter.py": "Compliance reporter script",
            "scripts/ci/emergency-bypass-manager.py": "Emergency bypass manager script",
            "scripts/operational/enhanced_artifact_collector.py": "Enhanced artifact collector script",
        }

        for script_file, description in script_files.items():
            script_path = self.root_dir / script_file

            if not script_path.exists():
                checks.append(
                    ValidationCheck(
                        name=f"script_{script_path.name}",
                        description=f"{description} exists",
                        category="scripts",
                        result=ValidationResult.FAIL,
                        message=f"❌ {script_file} not found",
                        recommendation=f"Create {script_file}",
                    )
                )
                continue

            # Check if file is executable (Python files should be)
            if not script_path.is_file():
                checks.append(
                    ValidationCheck(
                        name=f"script_{script_path.name}_file",
                        description=f"{description} is a file",
                        category="scripts",
                        result=ValidationResult.FAIL,
                        message=f"❌ {script_file} is not a regular file",
                        recommendation=f"Ensure {script_file} is a regular file",
                    )
                )
                continue

            # Check Python syntax
            try:
                result = subprocess.run(  # nosec B603 B607
                    ["python", "-m", "py_compile", str(script_path)], capture_output=True, text=True, cwd=self.root_dir
                )

                if result.returncode == 0:
                    checks.append(
                        ValidationCheck(
                            name=f"script_{script_path.name}_syntax",
                            description=f"{description} has valid Python syntax",
                            category="scripts",
                            result=ValidationResult.PASS,
                            message=f"✅ {script_file} has valid Python syntax",
                        )
                    )
                else:
                    checks.append(
                        ValidationCheck(
                            name=f"script_{script_path.name}_syntax",
                            description=f"{description} has valid Python syntax",
                            category="scripts",
                            result=ValidationResult.FAIL,
                            message=f"❌ {script_file} has syntax errors: {result.stderr}",
                            recommendation=f"Fix Python syntax errors in {script_file}",
                        )
                    )

            except FileNotFoundError:
                checks.append(
                    ValidationCheck(
                        name=f"script_{script_path.name}_python",
                        description="Python interpreter available for validation",
                        category="scripts",
                        result=ValidationResult.SKIP,
                        message="⏭️ Python interpreter not found, skipping syntax check",
                        recommendation="Install Python to validate script syntax",
                    )
                )

        return checks

    def validate_security_tools_integration(self) -> list[ValidationCheck]:
        """Validate that security tools are properly integrated."""
        checks = []

        # Check for secrets baseline
        secrets_baseline = self.root_dir / ".secrets.baseline"
        if secrets_baseline.exists():
            checks.append(
                ValidationCheck(
                    name="secrets_baseline_exists",
                    description="Secrets baseline file exists",
                    category="security_tools",
                    result=ValidationResult.PASS,
                    message="✅ .secrets.baseline found",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    name="secrets_baseline_exists",
                    description="Secrets baseline file exists",
                    category="security_tools",
                    result=ValidationResult.WARNING,
                    message="⚠️ .secrets.baseline not found",
                    recommendation="Run 'detect-secrets scan --update .secrets.baseline' to create baseline",
                )
            )

        # Check for security requirements
        security_reqs = self.root_dir / "config" / "requirements" / "requirements-security.txt"
        if security_reqs.exists():
            checks.append(
                ValidationCheck(
                    name="security_requirements_exists",
                    description="Security requirements file exists",
                    category="security_tools",
                    result=ValidationResult.PASS,
                    message="✅ Security requirements file found",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    name="security_requirements_exists",
                    description="Security requirements file exists",
                    category="security_tools",
                    result=ValidationResult.WARNING,
                    message="⚠️ Security requirements file not found",
                    recommendation="Create config/requirements/requirements-security.txt with security tool dependencies",
                )
            )

        return checks

    def validate_integration_points(self) -> list[ValidationCheck]:
        """Validate integration points between security components."""
        checks = []

        # Check that workflows reference security configuration
        main_ci = self.workflows_dir / "main-ci.yml"
        if main_ci.exists():
            try:
                with open(main_ci, encoding="utf-8") as f:
                    content = f.read()

                if "requirements-security.txt" in content:
                    checks.append(
                        ValidationCheck(
                            name="main_ci_security_reqs_integration",
                            description="Main CI integrates security requirements",
                            category="integration",
                            result=ValidationResult.PASS,
                            message="✅ Main CI references security requirements",
                        )
                    )
                else:
                    checks.append(
                        ValidationCheck(
                            name="main_ci_security_reqs_integration",
                            description="Main CI integrates security requirements",
                            category="integration",
                            result=ValidationResult.WARNING,
                            message="⚠️ Main CI does not reference security requirements",
                            recommendation="Add security requirements installation to main CI",
                        )
                    )

            except Exception as e:
                logger.error(f"Error validating main CI integration: {e}")

        # Check that emergency procedures are integrated
        scion_prod = self.workflows_dir / "scion_production.yml"
        if scion_prod.exists():
            try:
                with open(scion_prod, encoding="utf-8") as f:
                    content = f.read()

                if "emergency_bypass" in content:
                    checks.append(
                        ValidationCheck(
                            name="scion_emergency_procedures_integration",
                            description="SCION production integrates emergency procedures",
                            category="integration",
                            result=ValidationResult.PASS,
                            message="✅ SCION production includes emergency bypass handling",
                        )
                    )
                else:
                    checks.append(
                        ValidationCheck(
                            name="scion_emergency_procedures_integration",
                            description="SCION production integrates emergency procedures",
                            category="integration",
                            result=ValidationResult.FAIL,
                            message="❌ SCION production missing emergency bypass handling",
                            recommendation="Add emergency bypass handling to SCION production workflow",
                        )
                    )

            except Exception as e:
                logger.error(f"Error validating SCION integration: {e}")

        return checks

    def run_validation(self) -> tuple[list[ValidationCheck], dict[str, int]]:
        """Run complete security integration validation."""
        logger.info("Starting security integration validation...")

        all_checks = []

        # Run all validation categories
        all_checks.extend(self.validate_workflow_files())
        all_checks.extend(self.validate_security_config_files())
        all_checks.extend(self.validate_security_scripts())
        all_checks.extend(self.validate_security_tools_integration())
        all_checks.extend(self.validate_integration_points())

        # Calculate summary statistics
        summary = {
            "total": len(all_checks),
            "pass": len([c for c in all_checks if c.result == ValidationResult.PASS]),
            "fail": len([c for c in all_checks if c.result == ValidationResult.FAIL]),
            "warning": len([c for c in all_checks if c.result == ValidationResult.WARNING]),
            "skip": len([c for c in all_checks if c.result == ValidationResult.SKIP]),
        }

        logger.info(
            f"Validation complete: {summary['pass']} passed, {summary['fail']} failed, {summary['warning']} warnings, {summary['skip']} skipped"
        )

        return all_checks, summary

    def generate_report(
        self, checks: list[ValidationCheck], summary: dict[str, int], output_file: str | None = None
    ) -> str:
        """Generate validation report."""

        # Determine overall status
        if summary["fail"] > 0:
            overall_status = "❌ FAILED"
        elif summary["warning"] > 0:
            overall_status = "⚠️ WARNINGS"
        else:
            overall_status = "✅ PASSED"

        report = f"""# Security Integration Validation Report

**Overall Status:** {overall_status}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total Checks:** {summary['total']}
- **Passed:** ✅ {summary['pass']}
- **Failed:** ❌ {summary['fail']}
- **Warnings:** ⚠️ {summary['warning']}
- **Skipped:** ⏭️ {summary['skip']}

## Results by Category

"""

        # Group checks by category
        categories = {}
        for check in checks:
            if check.category not in categories:
                categories[check.category] = []
            categories[check.category].append(check)

        for category, category_checks in categories.items():
            report += f"### {category.title().replace('_', ' ')}\n\n"

            for check in category_checks:
                result_emoji = {
                    ValidationResult.PASS: "✅",
                    ValidationResult.FAIL: "❌",
                    ValidationResult.WARNING: "⚠️",
                    ValidationResult.SKIP: "⏭️",
                }[check.result]

                report += f"- {result_emoji} **{check.name}**: {check.message}\n"
                if check.recommendation and check.result in [ValidationResult.FAIL, ValidationResult.WARNING]:
                    report += f"  - *Recommendation: {check.recommendation}*\n"

            report += "\n"

        # Add failed checks summary
        failed_checks = [c for c in checks if c.result == ValidationResult.FAIL]
        if failed_checks:
            report += "## Critical Issues (Must Fix)\n\n"
            for check in failed_checks:
                report += f"- **{check.name}**: {check.message}\n"
                if check.recommendation:
                    report += f"  - *Action Required: {check.recommendation}*\n"
            report += "\n"

        # Add recommendations
        warning_checks = [c for c in checks if c.result == ValidationResult.WARNING]
        if warning_checks:
            report += "## Recommendations (Should Fix)\n\n"
            for check in warning_checks:
                report += f"- **{check.name}**: {check.message}\n"
                if check.recommendation:
                    report += f"  - *Suggestion: {check.recommendation}*\n"
            report += "\n"

        if not failed_checks and not warning_checks:
            report += (
                "## 🎉 Excellent!\n\nAll security integrations are properly configured and functioning correctly.\n"
            )

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Validation report written to {output_file}")

        return report


def main():
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Security Integration Validator")
    parser.add_argument("--root-dir", default=".", help="Root directory to validate")
    parser.add_argument("--output", help="Output report file")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown", help="Output format")
    parser.add_argument("--fail-on-warnings", action="store_true", help="Exit with error on warnings")
    parser.add_argument("--quiet", action="store_true", help="Suppress output except for errors")

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    validator = SecurityIntegrationValidator(args.root_dir)
    checks, summary = validator.run_validation()

    if args.format == "json":
        json_report = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "checks": [
                {
                    "name": check.name,
                    "description": check.description,
                    "category": check.category,
                    "result": check.result.value,
                    "message": check.message,
                    "recommendation": check.recommendation,
                }
                for check in checks
            ],
        }

        output_file = args.output or "security-integration-validation.json"
        with open(output_file, "w") as f:
            json.dump(json_report, f, indent=2)

        if not args.quiet:
            print(f"JSON report written to {output_file}")
    else:
        report_content = validator.generate_report(checks, summary, args.output)
        if not args.output and not args.quiet:
            print(report_content)

    # Print summary to console
    if not args.quiet:
        print("\nValidation Summary:")
        print(
            f"Total: {summary['total']}, Passed: {summary['pass']}, Failed: {summary['fail']}, Warnings: {summary['warning']}, Skipped: {summary['skip']}"
        )

    # Exit with appropriate code
    if summary["fail"] > 0:
        logger.error("Validation failed - security integrations have critical issues")
        sys.exit(1)
    elif summary["warning"] > 0:
        if args.fail_on_warnings:
            logger.error("Validation failed - warnings found and fail-on-warnings enabled")
            sys.exit(1)
        else:
            logger.warning("Validation passed with warnings")
            sys.exit(0)
    else:
        logger.info("Validation passed - all security integrations are properly configured")
        sys.exit(0)


if __name__ == "__main__":
    main()
