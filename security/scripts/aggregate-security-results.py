#!/usr/bin/env python3
"""
Aggregate Security Results Script
Consolidates vulnerability scan results from multiple ecosystems and tools
into a unified security report for AIVillage project.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Vulnerability:
    """Represents a single vulnerability finding"""

    id: str
    package: str
    version: str
    ecosystem: str
    severity: str
    title: str
    description: str
    cve_id: str = ""
    cvss_score: float = 0.0
    fixed_version: str = ""
    source: str = ""
    references: List[str] = None

    def __post_init__(self):
        if self.references is None:
            self.references = []


@dataclass
class SecurityReport:
    """Aggregated security report"""

    scan_timestamp: str
    project_name: str = "AIVillage"
    total_dependencies: int = 0
    vulnerabilities: List[Vulnerability] = None
    summary: Dict[str, int] = None
    ecosystems: Dict[str, Dict[str, int]] = None
    recommendations: List[str] = None

    def __post_init__(self):
        if self.vulnerabilities is None:
            self.vulnerabilities = []
        if self.summary is None:
            self.summary = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        if self.ecosystems is None:
            self.ecosystems = {}
        if self.recommendations is None:
            self.recommendations = []


class SecurityResultsAggregator:
    """Aggregates security scan results from multiple tools and ecosystems"""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.report = SecurityReport(scan_timestamp=datetime.now(timezone.utc).isoformat())

    def process_all_results(self):
        """Process all security scan results"""
        logger.info("Starting security results aggregation...")

        # Process Python results
        self._process_python_results()

        # Process Node.js results
        self._process_nodejs_results()

        # Process Rust results
        self._process_rust_results()

        # Process Go results
        self._process_go_results()

        # Process container results
        self._process_container_results()

        # Generate summary and recommendations
        self._generate_summary()
        self._generate_recommendations()

        logger.info(f"Aggregation complete. Found {len(self.report.vulnerabilities)} vulnerabilities.")

    def _process_python_results(self):
        """Process Python security scan results"""
        logger.info("Processing Python security results...")

        # Process pip-audit results
        pip_audit_file = self.results_dir / "python-security-results" / "pip-audit.json"
        if pip_audit_file.exists():
            self._process_pip_audit(pip_audit_file)

        # Process safety results
        safety_file = self.results_dir / "python-security-results" / "safety.json"
        if safety_file.exists():
            self._process_safety(safety_file)

        # Process bandit results
        bandit_file = self.results_dir / "python-security-results" / "bandit.json"
        if bandit_file.exists():
            self._process_bandit(bandit_file)

        # Process semgrep results
        semgrep_file = self.results_dir / "python-security-results" / "semgrep.json"
        if semgrep_file.exists():
            self._process_semgrep(semgrep_file)

    def _process_nodejs_results(self):
        """Process Node.js security scan results"""
        logger.info("Processing Node.js security results...")

        # Process npm audit results
        npm_audit_file = self.results_dir / "nodejs-security-results" / "npm-audit.json"
        if npm_audit_file.exists():
            self._process_npm_audit(npm_audit_file)

        # Process retire.js results
        retire_file = self.results_dir / "nodejs-security-results" / "retire.json"
        if retire_file.exists():
            self._process_retire(retire_file)

    def _process_rust_results(self):
        """Process Rust security scan results"""
        logger.info("Processing Rust security results...")

        # Process cargo audit results
        cargo_audit_file = self.results_dir / "rust-security-results" / "cargo-audit.json"
        if cargo_audit_file.exists():
            self._process_cargo_audit(cargo_audit_file)

    def _process_go_results(self):
        """Process Go security scan results"""
        logger.info("Processing Go security results...")

        # Process govulncheck results
        govulncheck_file = self.results_dir / "go-security-results" / "govulncheck.json"
        if govulncheck_file.exists():
            self._process_govulncheck(govulncheck_file)

    def _process_container_results(self):
        """Process container security scan results"""
        logger.info("Processing container security results...")

        # Process Trivy results
        trivy_file = self.results_dir / "container-security-results" / "trivy.json"
        if trivy_file.exists():
            self._process_trivy(trivy_file)

    def _process_pip_audit(self, file_path: Path):
        """Process pip-audit JSON results"""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            for vuln in data.get("vulnerabilities", []):
                package = vuln.get("package", "")
                version = vuln.get("installed_version", "")

                for advisory in vuln.get("vulnerabilities", []):
                    vulnerability = Vulnerability(
                        id=advisory.get("id", ""),
                        package=package,
                        version=version,
                        ecosystem="python",
                        severity=self._normalize_severity(advisory.get("severity", "unknown")),
                        title=advisory.get("summary", ""),
                        description=advisory.get("description", ""),
                        cve_id=advisory.get("aliases", [])[0] if advisory.get("aliases") else "",
                        fixed_version=advisory.get("fixed_version", ""),
                        source="pip-audit",
                        references=advisory.get("references", []),
                    )
                    self.report.vulnerabilities.append(vulnerability)

        except Exception as e:
            logger.error(f"Error processing pip-audit results: {e}")

    def _process_safety(self, file_path: Path):
        """Process safety JSON results"""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            for vuln in data.get("vulnerabilities", []):
                vulnerability = Vulnerability(
                    id=vuln.get("vulnerability_id", ""),
                    package=vuln.get("package_name", ""),
                    version=vuln.get("analyzed_version", ""),
                    ecosystem="python",
                    severity=self._normalize_severity(vuln.get("severity", "unknown")),
                    title=vuln.get("advisory", ""),
                    description=vuln.get("advisory", ""),
                    cve_id=vuln.get("CVE", ""),
                    source="safety",
                )
                self.report.vulnerabilities.append(vulnerability)

        except Exception as e:
            logger.error(f"Error processing safety results: {e}")

    def _process_bandit(self, file_path: Path):
        """Process bandit JSON results"""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            for result in data.get("results", []):
                vulnerability = Vulnerability(
                    id=result.get("test_id", ""),
                    package="static-analysis",
                    version="",
                    ecosystem="python",
                    severity=self._normalize_severity(result.get("issue_severity", "unknown")),
                    title=result.get("test_name", ""),
                    description=result.get("issue_text", ""),
                    source="bandit",
                )
                self.report.vulnerabilities.append(vulnerability)

        except Exception as e:
            logger.error(f"Error processing bandit results: {e}")

    def _process_semgrep(self, file_path: Path):
        """Process semgrep JSON results"""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            for result in data.get("results", []):
                vulnerability = Vulnerability(
                    id=result.get("check_id", ""),
                    package="static-analysis",
                    version="",
                    ecosystem="python",
                    severity=self._normalize_severity(result.get("extra", {}).get("severity", "unknown")),
                    title=result.get("extra", {}).get("message", ""),
                    description=result.get("extra", {}).get("message", ""),
                    source="semgrep",
                )
                self.report.vulnerabilities.append(vulnerability)

        except Exception as e:
            logger.error(f"Error processing semgrep results: {e}")

    def _process_npm_audit(self, file_path: Path):
        """Process npm audit JSON results"""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            for vuln_id, vuln in data.get("vulnerabilities", {}).items():
                vulnerability = Vulnerability(
                    id=vuln_id,
                    package=vuln.get("name", ""),
                    version=vuln.get("range", ""),
                    ecosystem="nodejs",
                    severity=self._normalize_severity(vuln.get("severity", "unknown")),
                    title=vuln.get("title", ""),
                    description=vuln.get("url", ""),
                    cve_id=vuln.get("cves", [])[0] if vuln.get("cves") else "",
                    source="npm-audit",
                )
                self.report.vulnerabilities.append(vulnerability)

        except Exception as e:
            logger.error(f"Error processing npm audit results: {e}")

    def _process_retire(self, file_path: Path):
        """Process retire.js JSON results"""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            for result in data:
                for vuln in result.get("results", []):
                    for vulnerability_data in vuln.get("vulnerabilities", []):
                        vulnerability = Vulnerability(
                            id=vulnerability_data.get("id", ""),
                            package=result.get("file", ""),
                            version=vuln.get("version", ""),
                            ecosystem="nodejs",
                            severity=self._normalize_severity(vulnerability_data.get("severity", "unknown")),
                            title=vulnerability_data.get("summary", ""),
                            description=vulnerability_data.get("info", [{}])[0].get("summary", ""),
                            source="retire.js",
                        )
                        self.report.vulnerabilities.append(vulnerability)

        except Exception as e:
            logger.error(f"Error processing retire.js results: {e}")

    def _process_cargo_audit(self, file_path: Path):
        """Process cargo audit JSON results"""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            for vuln in data.get("vulnerabilities", {}).get("list", []):
                for package in vuln.get("versions", {}):
                    vulnerability = Vulnerability(
                        id=vuln.get("advisory", {}).get("id", ""),
                        package=package,
                        version=vuln.get("versions", {}).get(package, {}).get("version", ""),
                        ecosystem="rust",
                        severity=self._normalize_severity("high"),  # Cargo audit doesn't provide severity
                        title=vuln.get("advisory", {}).get("title", ""),
                        description=vuln.get("advisory", {}).get("description", ""),
                        source="cargo-audit",
                    )
                    self.report.vulnerabilities.append(vulnerability)

        except Exception as e:
            logger.error(f"Error processing cargo audit results: {e}")

    def _process_govulncheck(self, file_path: Path):
        """Process govulncheck JSON results"""
        try:
            with open(file_path, "r") as f:
                content = f.read()
                # govulncheck produces JSON lines format
                for line in content.strip().split("\n"):
                    if line.strip():
                        data = json.loads(line)
                        if data.get("finding"):
                            finding = data["finding"]
                            vulnerability = Vulnerability(
                                id=finding.get("osv", ""),
                                package=finding.get("symbol", ""),
                                version="",
                                ecosystem="go",
                                severity="high",  # Go vulnerabilities are typically high
                                title=finding.get("summary", ""),
                                description=finding.get("description", ""),
                                source="govulncheck",
                            )
                            self.report.vulnerabilities.append(vulnerability)

        except Exception as e:
            logger.error(f"Error processing govulncheck results: {e}")

    def _process_trivy(self, file_path: Path):
        """Process Trivy JSON results"""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            for result in data.get("Results", []):
                for vuln in result.get("Vulnerabilities", []):
                    vulnerability = Vulnerability(
                        id=vuln.get("VulnerabilityID", ""),
                        package=vuln.get("PkgName", ""),
                        version=vuln.get("InstalledVersion", ""),
                        ecosystem="container",
                        severity=self._normalize_severity(vuln.get("Severity", "unknown")),
                        title=vuln.get("Title", ""),
                        description=vuln.get("Description", ""),
                        cvss_score=vuln.get("CVSS", {}).get("nvd", {}).get("V3Score", 0.0),
                        fixed_version=vuln.get("FixedVersion", ""),
                        source="trivy",
                    )
                    self.report.vulnerabilities.append(vulnerability)

        except Exception as e:
            logger.error(f"Error processing Trivy results: {e}")

    def _normalize_severity(self, severity: str) -> str:
        """Normalize severity levels across different tools"""
        severity = severity.lower().strip()

        if severity in ["critical", "crit"]:
            return "critical"
        elif severity in ["high", "error"]:
            return "high"
        elif severity in ["medium", "moderate", "warning"]:
            return "medium"
        elif severity in ["low", "info", "note"]:
            return "low"
        else:
            return "info"

    def _generate_summary(self):
        """Generate vulnerability summary statistics"""
        # Count by severity
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        ecosystem_counts = {}

        for vuln in self.report.vulnerabilities:
            severity_counts[vuln.severity] = severity_counts.get(vuln.severity, 0) + 1

            if vuln.ecosystem not in ecosystem_counts:
                ecosystem_counts[vuln.ecosystem] = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
            ecosystem_counts[vuln.ecosystem][vuln.severity] += 1

        self.report.summary = severity_counts
        self.report.ecosystems = ecosystem_counts

    def _generate_recommendations(self):
        """Generate security recommendations based on findings"""
        recommendations = []

        critical_count = self.report.summary.get("critical", 0)
        high_count = self.report.summary.get("high", 0)

        if critical_count > 0:
            recommendations.append(
                f"🚨 CRITICAL: {critical_count} critical vulnerabilities require immediate attention"
            )
            recommendations.append("- Halt production deployments until critical issues are resolved")
            recommendations.append("- Implement emergency patching procedures")

        if high_count > 0:
            recommendations.append(
                f"⚠️  HIGH: {high_count} high-severity vulnerabilities should be addressed within 7 days"
            )

        # Ecosystem-specific recommendations
        for ecosystem, counts in self.report.ecosystems.items():
            total = sum(counts.values())
            if total > 0:
                recommendations.append(
                    f"📦 {ecosystem.upper()}: {total} vulnerabilities found - consider dependency updates"
                )

        # General recommendations
        recommendations.extend(
            [
                "🔄 Enable automated dependency updates where possible",
                "🛡️  Implement vulnerability scanning in CI/CD pipeline",
                "📊 Monitor dependency health continuously",
                "🔒 Consider using Software Bill of Materials (SBOM) for better visibility",
            ]
        )

        self.report.recommendations = recommendations

    def save_results(self, output_dir: str):
        """Save aggregated results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save full report
        with open(output_path / "security-report.json", "w") as f:
            json.dump(asdict(self.report), f, indent=2, default=str)

        # Save summary for GitHub Actions
        with open(output_path / "security-summary.json", "w") as f:
            json.dump(self.report.summary, f, indent=2)

        # Save critical vulnerabilities for alerting
        critical_vulns = [v for v in self.report.vulnerabilities if v.severity == "critical"]
        if critical_vulns:
            with open(output_path / "critical-vulnerabilities.json", "w") as f:
                json.dump(
                    {
                        "critical": len(critical_vulns),
                        "high": len([v for v in self.report.vulnerabilities if v.severity == "high"]),
                        "vulnerabilities": [asdict(v) for v in critical_vulns],
                    },
                    f,
                    indent=2,
                    default=str,
                )

        logger.info(f"Results saved to {output_path}")


def main():
    """Main execution function"""
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "security/results"

    aggregator = SecurityResultsAggregator(results_dir)
    aggregator.process_all_results()
    aggregator.save_results("security/reports")

    print("SUCCESS: Security aggregation complete!")
    print(f"Total vulnerabilities: {len(aggregator.report.vulnerabilities)}")
    print(f"Critical: {aggregator.report.summary['critical']}")
    print(f"High: {aggregator.report.summary['high']}")
    print(f"Medium: {aggregator.report.summary['medium']}")
    print(f"Low: {aggregator.report.summary['low']}")


if __name__ == "__main__":
    main()
