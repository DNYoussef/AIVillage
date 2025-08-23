#!/usr/bin/env python3
"""
Compliance Reporter
Generates comprehensive security and compliance reports from CI/CD pipeline results.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ComplianceMetric:
    name: str
    value: Any
    threshold: Optional[Any] = None
    status: str = "unknown"  # passed, failed, warning
    category: str = "general"


@dataclass
class ComplianceReport:
    timestamp: str
    commit_sha: str
    pipeline_type: str
    overall_status: str
    metrics: List[ComplianceMetric]
    security_summary: Dict[str, Any]
    recommendations: List[str]
    compliance_score: int


class ComplianceReporter:
    """Generate comprehensive compliance reports."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.timestamp = datetime.now().isoformat()

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load compliance configuration."""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Could not load config: {e}")

        return {
            "compliance": {
                "thresholds": {
                    "security_score_minimum": 85,
                    "vulnerability_max_critical": 0,
                    "vulnerability_max_high": 2,
                    "coverage_minimum": 80,
                },
                "categories": ["security", "quality", "performance", "architecture"],
            }
        }

    def collect_security_metrics(self) -> List[ComplianceMetric]:
        """Collect security-related compliance metrics."""
        metrics = []

        # Secret detection metrics
        if Path(".secrets.baseline").exists():
            metrics.append(
                ComplianceMetric(
                    name="secret_detection_baseline", value="present", status="passed", category="security"
                )
            )
        else:
            metrics.append(
                ComplianceMetric(
                    name="secret_detection_baseline", value="missing", status="failed", category="security"
                )
            )

        # Vulnerability scanning metrics
        vuln_metrics = self._collect_vulnerability_metrics()
        metrics.extend(vuln_metrics)

        # Security scanning metrics
        security_scan_metrics = self._collect_security_scan_metrics()
        metrics.extend(security_scan_metrics)

        return metrics

    def _collect_vulnerability_metrics(self) -> List[ComplianceMetric]:
        """Collect vulnerability scanning metrics."""
        metrics = []

        # Check for various security report files
        report_files = {
            "bandit-report.json": "bandit",
            "safety-report.json": "safety",
            "trivy-results.json": "trivy",
            "semgrep-report.json": "semgrep",
        }

        for report_file, tool in report_files.items():
            if Path(report_file).exists():
                try:
                    with open(report_file, "r") as f:
                        report_data = json.load(f)

                    if tool == "bandit":
                        critical_count = sum(
                            1 for r in report_data.get("results", []) if r.get("issue_severity") == "HIGH"
                        )
                        high_count = sum(
                            1 for r in report_data.get("results", []) if r.get("issue_severity") == "MEDIUM"
                        )

                        metrics.append(
                            ComplianceMetric(
                                name=f"{tool}_critical_issues",
                                value=critical_count,
                                threshold=self.config.get("compliance", {})
                                .get("thresholds", {})
                                .get("vulnerability_max_critical", 0),
                                status="passed" if critical_count == 0 else "failed",
                                category="security",
                            )
                        )

                        metrics.append(
                            ComplianceMetric(
                                name=f"{tool}_high_issues",
                                value=high_count,
                                threshold=self.config.get("compliance", {})
                                .get("thresholds", {})
                                .get("vulnerability_max_high", 2),
                                status="passed" if high_count <= 2 else "failed",
                                category="security",
                            )
                        )

                    elif tool == "safety" and isinstance(report_data, list):
                        vuln_count = len(report_data)
                        metrics.append(
                            ComplianceMetric(
                                name=f"{tool}_vulnerabilities",
                                value=vuln_count,
                                threshold=0,
                                status="passed" if vuln_count == 0 else "warning",
                                category="security",
                            )
                        )

                    elif tool == "trivy":
                        # Analyze Trivy results
                        total_vulns = 0
                        critical_vulns = 0
                        high_vulns = 0

                        for result in report_data.get("Results", []):
                            for vuln in result.get("Vulnerabilities", []):
                                total_vulns += 1
                                severity = vuln.get("Severity", "UNKNOWN")
                                if severity == "CRITICAL":
                                    critical_vulns += 1
                                elif severity == "HIGH":
                                    high_vulns += 1

                        metrics.append(
                            ComplianceMetric(
                                name=f"{tool}_total_vulnerabilities", value=total_vulns, category="security"
                            )
                        )

                        metrics.append(
                            ComplianceMetric(
                                name=f"{tool}_critical_vulnerabilities",
                                value=critical_vulns,
                                threshold=0,
                                status="passed" if critical_vulns == 0 else "failed",
                                category="security",
                            )
                        )

                        metrics.append(
                            ComplianceMetric(
                                name=f"{tool}_high_vulnerabilities",
                                value=high_vulns,
                                threshold=2,
                                status="passed" if high_vulns <= 2 else "failed",
                                category="security",
                            )
                        )

                except Exception as e:
                    logger.warning(f"Could not parse {report_file}: {e}")

        return metrics

    def _collect_security_scan_metrics(self) -> List[ComplianceMetric]:
        """Collect security scanning tool metrics."""
        metrics = []

        # Check for detect-secrets
        try:
            result = subprocess.run(["detect-secrets", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                metrics.append(
                    ComplianceMetric(name="detect_secrets_installed", value=True, status="passed", category="security")
                )
        except FileNotFoundError:
            metrics.append(
                ComplianceMetric(name="detect_secrets_installed", value=False, status="failed", category="security")
            )

        # Check for bandit
        try:
            result = subprocess.run(["bandit", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                metrics.append(
                    ComplianceMetric(name="bandit_installed", value=True, status="passed", category="security")
                )
        except FileNotFoundError:
            metrics.append(
                ComplianceMetric(name="bandit_installed", value=False, status="warning", category="security")
            )

        return metrics

    def collect_quality_metrics(self) -> List[ComplianceMetric]:
        """Collect code quality metrics."""
        metrics = []

        # Check for architectural quality reports
        if Path("quality_gate_result.json").exists():
            try:
                with open("quality_gate_result.json", "r") as f:
                    quality_data = json.load(f)

                metrics.append(
                    ComplianceMetric(
                        name="quality_gate_passed",
                        value=quality_data.get("passed", False),
                        status="passed" if quality_data.get("passed", False) else "failed",
                        category="quality",
                    )
                )

                metrics.append(
                    ComplianceMetric(
                        name="quality_score",
                        value=quality_data.get("overall_score", 0),
                        threshold=self.config.get("compliance", {})
                        .get("thresholds", {})
                        .get("security_score_minimum", 85),
                        status="passed" if quality_data.get("overall_score", 0) >= 85 else "failed",
                        category="quality",
                    )
                )

            except Exception as e:
                logger.warning(f"Could not parse quality gate results: {e}")

        # Check for coupling metrics
        if Path("coupling_report.json").exists():
            try:
                with open("coupling_report.json", "r") as f:
                    coupling_data = json.load(f)

                max_coupling = coupling_data.get("max_coupling", 0)
                metrics.append(
                    ComplianceMetric(
                        name="max_coupling",
                        value=max_coupling,
                        threshold=12.0,
                        status="passed" if max_coupling <= 12.0 else "failed",
                        category="architecture",
                    )
                )

            except Exception as e:
                logger.warning(f"Could not parse coupling metrics: {e}")

        return metrics

    def collect_test_metrics(self) -> List[ComplianceMetric]:
        """Collect test coverage and test results metrics."""
        metrics = []

        # Check for coverage reports
        coverage_files = ["coverage.xml", ".coverage", "coverage.json"]
        for coverage_file in coverage_files:
            if Path(coverage_file).exists():
                metrics.append(
                    ComplianceMetric(name="test_coverage_present", value=True, status="passed", category="quality")
                )
                break
        else:
            metrics.append(
                ComplianceMetric(name="test_coverage_present", value=False, status="warning", category="quality")
            )

        return metrics

    def calculate_compliance_score(self, metrics: List[ComplianceMetric]) -> int:
        """Calculate overall compliance score."""
        if not metrics:
            return 0

        total_weight = 0
        weighted_score = 0

        # Define category weights
        weights = {"security": 40, "quality": 25, "architecture": 20, "performance": 10, "general": 5}

        for metric in metrics:
            weight = weights.get(metric.category, 5)
            total_weight += weight

            if metric.status == "passed":
                weighted_score += weight
            elif metric.status == "warning":
                weighted_score += weight * 0.5
            # failed = 0 points

        return int((weighted_score / total_weight) * 100) if total_weight > 0 else 0

    def generate_recommendations(self, metrics: List[ComplianceMetric]) -> List[str]:
        """Generate compliance recommendations based on metrics."""
        recommendations = []

        failed_metrics = [m for m in metrics if m.status == "failed"]
        warning_metrics = [m for m in metrics if m.status == "warning"]

        # Security recommendations
        security_failed = [m for m in failed_metrics if m.category == "security"]
        if security_failed:
            recommendations.append("🔒 CRITICAL: Address security vulnerabilities immediately")

            for metric in security_failed:
                if "secret" in metric.name:
                    recommendations.append("- Implement detect-secrets scanning and remediate exposed secrets")
                elif "critical" in metric.name and metric.value > 0:
                    recommendations.append(f"- Fix {metric.value} critical security vulnerabilities")
                elif "high" in metric.name and metric.value > metric.threshold:
                    recommendations.append(f"- Address {metric.value} high severity security issues")

        # Quality recommendations
        quality_failed = [m for m in failed_metrics if m.category == "quality"]
        if quality_failed:
            recommendations.append("📊 Improve code quality metrics")

            for metric in quality_failed:
                if "quality_score" in metric.name:
                    recommendations.append(f"- Improve quality score from {metric.value} to above {metric.threshold}")
                elif "coverage" in metric.name:
                    recommendations.append("- Increase test coverage")

        # Architecture recommendations
        arch_failed = [m for m in failed_metrics if m.category == "architecture"]
        if arch_failed:
            recommendations.append("🏗️ Address architectural concerns")

            for metric in arch_failed:
                if "coupling" in metric.name:
                    recommendations.append(f"- Reduce coupling from {metric.value} to below {metric.threshold}")
                elif "complexity" in metric.name:
                    recommendations.append("- Refactor complex components")

        # Tool installation recommendations
        tool_warnings = [m for m in warning_metrics if "installed" in m.name]
        if tool_warnings:
            recommendations.append("🛠️ Install missing security tools:")
            for metric in tool_warnings:
                tool_name = metric.name.replace("_installed", "")
                recommendations.append(f"- Install {tool_name}")

        if not recommendations:
            recommendations.append("✅ All compliance checks passed - no recommendations needed")

        return recommendations

    def generate_report(self, pipeline_type: str, commit_sha: str) -> ComplianceReport:
        """Generate comprehensive compliance report."""
        logger.info("Generating compliance report...")

        # Collect all metrics
        all_metrics = []
        all_metrics.extend(self.collect_security_metrics())
        all_metrics.extend(self.collect_quality_metrics())
        all_metrics.extend(self.collect_test_metrics())

        # Calculate compliance score
        compliance_score = self.calculate_compliance_score(all_metrics)

        # Determine overall status
        failed_count = len([m for m in all_metrics if m.status == "failed"])
        critical_failed = len([m for m in all_metrics if m.status == "failed" and m.category == "security"])

        if critical_failed > 0:
            overall_status = "failed"
        elif failed_count > 3:
            overall_status = "failed"
        elif failed_count > 0:
            overall_status = "warning"
        else:
            overall_status = "passed"

        # Generate security summary
        security_metrics = [m for m in all_metrics if m.category == "security"]
        security_summary = {
            "total_security_metrics": len(security_metrics),
            "security_passed": len([m for m in security_metrics if m.status == "passed"]),
            "security_failed": len([m for m in security_metrics if m.status == "failed"]),
            "security_warnings": len([m for m in security_metrics if m.status == "warning"]),
            "critical_vulnerabilities": sum(
                m.value for m in security_metrics if "critical" in m.name and isinstance(m.value, int)
            ),
            "high_vulnerabilities": sum(
                m.value for m in security_metrics if "high" in m.name and isinstance(m.value, int)
            ),
        }

        # Generate recommendations
        recommendations = self.generate_recommendations(all_metrics)

        return ComplianceReport(
            timestamp=self.timestamp,
            commit_sha=commit_sha,
            pipeline_type=pipeline_type,
            overall_status=overall_status,
            metrics=all_metrics,
            security_summary=security_summary,
            recommendations=recommendations,
            compliance_score=compliance_score,
        )

    def save_report(self, report: ComplianceReport, output_file: str, format: str = "json"):
        """Save compliance report to file."""
        if format == "json":
            with open(output_file, "w") as f:
                json.dump(asdict(report), f, indent=2, default=str)

        elif format == "yaml":
            with open(output_file, "w") as f:
                yaml.dump(asdict(report), f, default_flow_style=False)

        elif format == "markdown":
            self._save_markdown_report(report, output_file)

        logger.info(f"Compliance report saved to {output_file}")

    def _save_markdown_report(self, report: ComplianceReport, output_file: str):
        """Save report as markdown format."""
        content = f"""# Compliance Report

**Timestamp:** {report.timestamp}
**Commit SHA:** {report.commit_sha}
**Pipeline Type:** {report.pipeline_type}
**Overall Status:** {report.overall_status.upper()}
**Compliance Score:** {report.compliance_score}/100

## Security Summary
- Total Security Metrics: {report.security_summary['total_security_metrics']}
- Security Passed: {report.security_summary['security_passed']}
- Security Failed: {report.security_summary['security_failed']}
- Security Warnings: {report.security_summary['security_warnings']}
- Critical Vulnerabilities: {report.security_summary['critical_vulnerabilities']}
- High Vulnerabilities: {report.security_summary['high_vulnerabilities']}

## Detailed Metrics

| Metric | Category | Value | Threshold | Status |
|--------|----------|-------|-----------|---------|"""

        for metric in report.metrics:
            threshold_str = str(metric.threshold) if metric.threshold is not None else "N/A"
            status_emoji = {"passed": "✅", "failed": "❌", "warning": "⚠️"}.get(metric.status, "❓")
            content += f"\n| {metric.name} | {metric.category} | {metric.value} | {threshold_str} | {status_emoji} {metric.status} |"

        content += "\n\n## Recommendations\n\n"
        for rec in report.recommendations:
            content += f"- {rec}\n"

        with open(output_file, "w") as f:
            f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Generate compliance reports")
    parser.add_argument("--pipeline-type", default="ci", help="Pipeline type (ci, production, etc.)")
    parser.add_argument("--commit-sha", default=os.environ.get("GITHUB_SHA", "unknown"), help="Commit SHA")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--format", choices=["json", "yaml", "markdown"], default="json", help="Output format")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--fail-on-violations", action="store_true", help="Exit with error code on compliance failures")

    args = parser.parse_args()

    reporter = ComplianceReporter(args.config)
    report = reporter.generate_report(args.pipeline_type, args.commit_sha)

    reporter.save_report(report, args.output, args.format)

    # Print summary
    print(f"\nCompliance Report Summary:")
    print(f"Overall Status: {report.overall_status.upper()}")
    print(f"Compliance Score: {report.compliance_score}/100")
    print(
        f"Security Issues: {report.security_summary['security_failed']} failed, {report.security_summary['security_warnings']} warnings"
    )

    # Exit with appropriate code
    if args.fail_on_violations and report.overall_status == "failed":
        logger.error("Compliance check failed")
        sys.exit(1)
    elif report.overall_status == "warning":
        logger.warning("Compliance check passed with warnings")
        sys.exit(0)
    else:
        logger.info("Compliance check passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
