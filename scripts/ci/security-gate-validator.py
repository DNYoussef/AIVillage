#!/usr/bin/env python3
"""
Security Gate Validator
Validates security gates across CI/CD pipelines and enforces security policies.
"""

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SecurityGateResult(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class SecurityViolation:
    category: str
    severity: str
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    recommendation: Optional[str] = None


@dataclass
class SecurityGateReport:
    gate_name: str
    result: SecurityGateResult
    violations: List[SecurityViolation]
    metrics: Dict[str, any]
    timestamp: str


class SecurityGateValidator:
    """Main security gate validator class."""

    def __init__(self, config_path: str = "config/security/security-gate-config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.violations = []

    def _load_config(self) -> Dict:
        """Load security gate configuration."""
        try:
            if self.config_path.exists():
                with open(self.config_path, "r") as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                return self._default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._default_config()

    def _default_config(self) -> Dict:
        """Default security gate configuration."""
        return {
            "security_gates": {
                "main_ci": {
                    "enabled": True,
                    "thresholds": {
                        "vulnerability_scanning": {"max_critical": 0, "max_high": 2},
                        "secret_detection": {"block_on_new_secrets": True},
                    },
                }
            }
        }

    def validate_secret_detection(self, gate_config: Dict) -> List[SecurityViolation]:
        """Validate secret detection requirements."""
        violations = []
        secret_config = gate_config.get("thresholds", {}).get("secret_detection", {})

        if secret_config.get("block_on_new_secrets", True):
            try:
                # Run detect-secrets scan
                result = subprocess.run(
                    ["detect-secrets", "scan", "--baseline", ".secrets.baseline"],
                    capture_output=True,
                    text=True,
                    cwd=".",
                )

                if result.returncode != 0:
                    violations.append(
                        SecurityViolation(
                            category="secret_detection",
                            severity="CRITICAL",
                            message="New secrets detected in codebase",
                            recommendation="Review and remediate detected secrets using detect-secrets audit",
                        )
                    )

            except FileNotFoundError:
                violations.append(
                    SecurityViolation(
                        category="secret_detection",
                        severity="HIGH",
                        message="detect-secrets tool not found",
                        recommendation="Install detect-secrets: pip install detect-secrets",
                    )
                )

        return violations

    def validate_vulnerability_scanning(self, gate_config: Dict) -> List[SecurityViolation]:
        """Validate vulnerability scanning results."""
        violations = []
        vuln_config = gate_config.get("thresholds", {}).get("vulnerability_scanning", {})

        # Check for security scan results
        security_reports = ["bandit-report.json", "safety-report.json", "semgrep-report.json"]

        for report_file in security_reports:
            if Path(report_file).exists():
                violations.extend(self._analyze_security_report(report_file, vuln_config))

        return violations

    def _analyze_security_report(self, report_file: str, config: Dict) -> List[SecurityViolation]:
        """Analyze security report file."""
        violations = []

        try:
            with open(report_file, "r") as f:
                report_data = json.load(f)

            if "bandit" in report_file:
                violations.extend(self._analyze_bandit_report(report_data, config))
            elif "safety" in report_file:
                violations.extend(self._analyze_safety_report(report_data, config))
            elif "semgrep" in report_file:
                violations.extend(self._analyze_semgrep_report(report_data, config))

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Could not analyze {report_file}: {e}")

        return violations

    def _analyze_bandit_report(self, report_data: Dict, config: Dict) -> List[SecurityViolation]:
        """Analyze Bandit security scan results."""
        violations = []
        results = report_data.get("results", [])

        critical_count = sum(1 for r in results if r.get("issue_severity") == "HIGH")
        high_count = sum(1 for r in results if r.get("issue_severity") == "MEDIUM")

        max_critical = config.get("max_critical", 0)
        max_high = config.get("max_high", 2)

        if critical_count > max_critical:
            violations.append(
                SecurityViolation(
                    category="vulnerability_scanning",
                    severity="CRITICAL",
                    message=f"Critical security issues found: {critical_count} > {max_critical}",
                    recommendation="Review and fix critical security issues identified by Bandit",
                )
            )

        if high_count > max_high:
            violations.append(
                SecurityViolation(
                    category="vulnerability_scanning",
                    severity="HIGH",
                    message=f"High severity security issues found: {high_count} > {max_high}",
                    recommendation="Review and fix high severity security issues",
                )
            )

        return violations

    def _analyze_safety_report(self, report_data: Dict, config: Dict) -> List[SecurityViolation]:
        """Analyze Safety dependency scan results."""
        violations = []

        if isinstance(report_data, list):
            critical_vulns = [v for v in report_data if "high" in v.get("vulnerability", "").lower()]

            if len(critical_vulns) > config.get("max_critical", 0):
                violations.append(
                    SecurityViolation(
                        category="dependency_security",
                        severity="CRITICAL",
                        message=f"Critical dependency vulnerabilities found: {len(critical_vulns)}",
                        recommendation="Update vulnerable dependencies to secure versions",
                    )
                )

        return violations

    def _analyze_semgrep_report(self, report_data: Dict, config: Dict) -> List[SecurityViolation]:
        """Analyze Semgrep SAST results."""
        violations = []
        results = report_data.get("results", [])

        error_count = sum(1 for r in results if r.get("extra", {}).get("severity") == "ERROR")

        if error_count > 0:
            violations.append(
                SecurityViolation(
                    category="static_analysis",
                    severity="HIGH",
                    message=f"SAST security errors found: {error_count}",
                    recommendation="Review and fix security issues identified by Semgrep",
                )
            )

        return violations

    def validate_architecture_quality(self, gate_config: Dict) -> List[SecurityViolation]:
        """Validate architectural quality gates."""
        violations = []
        arch_config = gate_config.get("thresholds", {})

        # Check coupling metrics
        if Path("coupling_report.json").exists():
            try:
                with open("coupling_report.json", "r") as f:
                    coupling_data = json.load(f)

                max_coupling = coupling_data.get("max_coupling", 0)
                threshold = arch_config.get("coupling_threshold", 12.0)

                if max_coupling > threshold:
                    violations.append(
                        SecurityViolation(
                            category="architecture_quality",
                            severity="MEDIUM",
                            message=f"Coupling threshold exceeded: {max_coupling} > {threshold}",
                            recommendation="Refactor to reduce coupling between modules",
                        )
                    )

            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning("Could not analyze coupling metrics")

        return violations

    def validate_gate(self, gate_name: str) -> SecurityGateReport:
        """Validate a specific security gate."""
        gate_config = self.config.get("security_gates", {}).get(gate_name, {})

        if not gate_config.get("enabled", True):
            return SecurityGateReport(
                gate_name=gate_name,
                result=SecurityGateResult.PASSED,
                violations=[],
                metrics={},
                timestamp=self._get_timestamp(),
            )

        all_violations = []

        # Run validation checks
        all_violations.extend(self.validate_secret_detection(gate_config))
        all_violations.extend(self.validate_vulnerability_scanning(gate_config))
        all_violations.extend(self.validate_architecture_quality(gate_config))

        # Determine result
        critical_violations = [v for v in all_violations if v.severity == "CRITICAL"]
        high_violations = [v for v in all_violations if v.severity == "HIGH"]

        if gate_config.get("fail_fast", True) and critical_violations:
            result = SecurityGateResult.FAILED
        elif critical_violations or len(high_violations) > 3:
            result = SecurityGateResult.FAILED
        elif high_violations:
            result = SecurityGateResult.WARNING
        else:
            result = SecurityGateResult.PASSED

        metrics = {
            "total_violations": len(all_violations),
            "critical_violations": len(critical_violations),
            "high_violations": len(high_violations),
            "medium_violations": len([v for v in all_violations if v.severity == "MEDIUM"]),
        }

        return SecurityGateReport(
            gate_name=gate_name,
            result=result,
            violations=all_violations,
            metrics=metrics,
            timestamp=self._get_timestamp(),
        )

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime

        return datetime.now().isoformat()

    def generate_report(self, report: SecurityGateReport, output_file: Optional[str] = None) -> str:
        """Generate security gate report."""
        report_content = f"""# Security Gate Report: {report.gate_name}

**Status:** {report.result.value.upper()}
**Timestamp:** {report.timestamp}
**Total Violations:** {report.metrics.get('total_violations', 0)}

## Security Metrics
- Critical Violations: {report.metrics.get('critical_violations', 0)}
- High Violations: {report.metrics.get('high_violations', 0)}
- Medium Violations: {report.metrics.get('medium_violations', 0)}

## Violations
"""

        if report.violations:
            for violation in report.violations:
                report_content += f"""
### {violation.severity}: {violation.category}
**Message:** {violation.message}
**Recommendation:** {violation.recommendation or 'Review and remediate'}
"""
                if violation.file_path:
                    report_content += f"**File:** {violation.file_path}"
                    if violation.line_number:
                        report_content += f":{violation.line_number}"
                    report_content += "\n"
        else:
            report_content += "\nNo violations found ✅\n"

        if output_file:
            with open(output_file, "w") as f:
                f.write(report_content)

        return report_content


def main():
    parser = argparse.ArgumentParser(description="Security Gate Validator")
    parser.add_argument("--gate", required=True, help="Security gate to validate")
    parser.add_argument(
        "--config", default="config/security/security-gate-config.yaml", help="Security gate configuration file"
    )
    parser.add_argument("--output", help="Output report file")
    parser.add_argument("--fail-on-violations", action="store_true", help="Exit with error code on security violations")
    parser.add_argument("--json-output", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    validator = SecurityGateValidator(args.config)
    report = validator.validate_gate(args.gate)

    if args.json_output:
        json_report = {
            "gate_name": report.gate_name,
            "result": report.result.value,
            "violations": [
                {
                    "category": v.category,
                    "severity": v.severity,
                    "message": v.message,
                    "file_path": v.file_path,
                    "line_number": v.line_number,
                    "recommendation": v.recommendation,
                }
                for v in report.violations
            ],
            "metrics": report.metrics,
            "timestamp": report.timestamp,
        }

        output_file = args.output or f"security-gate-{args.gate}.json"
        with open(output_file, "w") as f:
            json.dump(json_report, f, indent=2)

        print(f"Security gate report written to {output_file}")
    else:
        report_content = validator.generate_report(report, args.output)
        if not args.output:
            print(report_content)

    # Exit with appropriate code
    if args.fail_on_violations and report.result == SecurityGateResult.FAILED:
        logger.error(f"Security gate {args.gate} failed with {len(report.violations)} violations")
        sys.exit(1)
    elif report.result == SecurityGateResult.WARNING:
        logger.warning(f"Security gate {args.gate} passed with warnings")
        sys.exit(0)
    else:
        logger.info(f"Security gate {args.gate} passed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
