#!/usr/bin/env python3
"""
AIVillage Artifacts Validation Script

Validates collected operational artifacts for completeness, quality, and compliance.
Used in CI/CD pipeline to ensure artifact collection meets standards.
"""

import json
import logging
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArtifactsValidator:
    """Validates operational artifacts collection."""

    def __init__(self, artifacts_dir: str = "artifacts"):
        """Initialize validator with artifacts directory."""
        self.artifacts_dir = Path(artifacts_dir)
        self.validation_results = []

    def validate_all_artifacts(self) -> tuple[bool, dict]:
        """
        Validate all collected artifacts.

        Returns:
            Tuple of (success, validation_report)
        """
        logger.info("Starting artifacts validation...")

        validation_report = {
            "validation_timestamp": "2025-08-19T00:00:00Z",
            "artifacts_directory": str(self.artifacts_dir),
            "validations": [],
            "summary": {"total_validations": 0, "passed": 0, "failed": 0, "warnings": 0},
        }

        # Validate coverage artifacts
        coverage_result = self._validate_coverage_artifacts()
        validation_report["validations"].append(coverage_result)

        # Validate security artifacts
        security_result = self._validate_security_artifacts()
        validation_report["validations"].append(security_result)

        # Validate SBOM artifacts
        sbom_result = self._validate_sbom_artifacts()
        validation_report["validations"].append(sbom_result)

        # Validate performance artifacts
        performance_result = self._validate_performance_artifacts()
        validation_report["validations"].append(performance_result)

        # Validate quality artifacts
        quality_result = self._validate_quality_artifacts()
        validation_report["validations"].append(quality_result)

        # Validate container artifacts
        container_result = self._validate_container_artifacts()
        validation_report["validations"].append(container_result)

        # Validate compliance artifacts
        compliance_result = self._validate_compliance_artifacts()
        validation_report["validations"].append(compliance_result)

        # Calculate summary
        for result in validation_report["validations"]:
            validation_report["summary"]["total_validations"] += 1
            if result["status"] == "passed":
                validation_report["summary"]["passed"] += 1
            elif result["status"] == "failed":
                validation_report["summary"]["failed"] += 1
            else:
                validation_report["summary"]["warnings"] += 1

        success = validation_report["summary"]["failed"] == 0

        logger.info(f"Validation completed: {validation_report['summary']}")
        return success, validation_report

    def _validate_coverage_artifacts(self) -> dict:
        """Validate coverage artifacts."""
        logger.info("Validating coverage artifacts...")

        coverage_dir = self.artifacts_dir / "coverage"
        issues = []
        warnings = []

        # Check for coverage.xml
        xml_file = coverage_dir / "coverage.xml"
        if xml_file.exists():
            try:
                # Parse coverage XML and extract coverage percentage
                tree = ET.parse(xml_file)
                root = tree.getroot()

                coverage_attr = root.get("line-rate")
                if coverage_attr:
                    coverage_pct = float(coverage_attr) * 100
                    if coverage_pct < 80:
                        warnings.append(f"Coverage below 80%: {coverage_pct:.1f}%")
                    logger.info(f"Coverage: {coverage_pct:.1f}%")
                else:
                    issues.append("Coverage percentage not found in XML")
            except Exception as e:
                issues.append(f"Failed to parse coverage.xml: {e}")
        else:
            issues.append("coverage.xml not found")

        # Check for HTML coverage report
        html_dir = coverage_dir / "htmlcov"
        if not html_dir.exists():
            warnings.append("HTML coverage report not found")

        # Check for JSON coverage report
        json_file = coverage_dir / "coverage.json"
        if not json_file.exists():
            warnings.append("JSON coverage report not found")

        status = "failed" if issues else ("warning" if warnings else "passed")

        return {
            "category": "coverage",
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "artifacts_found": len(list(coverage_dir.glob("*"))) if coverage_dir.exists() else 0,
        }

    def _validate_security_artifacts(self) -> dict:
        """Validate security artifacts."""
        logger.info("Validating security artifacts...")

        security_dir = self.artifacts_dir / "security"
        issues = []
        warnings = []

        # Check for Bandit report
        bandit_file = security_dir / "bandit-report.json"
        if bandit_file.exists():
            try:
                with open(bandit_file) as f:
                    bandit_data = json.load(f)

                high_issues = len([r for r in bandit_data.get("results", []) if r.get("issue_severity") == "HIGH"])
                if high_issues > 0:
                    warnings.append(f"Bandit found {high_issues} high severity issues")
            except Exception as e:
                issues.append(f"Failed to parse bandit report: {e}")
        else:
            warnings.append("Bandit security report not found")

        # Check for Safety report
        safety_file = security_dir / "safety-report.json"
        if safety_file.exists():
            try:
                with open(safety_file) as f:
                    safety_data = json.load(f)

                vulns = len(safety_data.get("vulnerabilities", []))
                if vulns > 0:
                    warnings.append(f"Safety found {vulns} dependency vulnerabilities")
            except Exception as e:
                issues.append(f"Failed to parse safety report: {e}")
        else:
            warnings.append("Safety dependency report not found")

        # Check for Semgrep report
        semgrep_file = security_dir / "semgrep-report.json"
        if semgrep_file.exists():
            try:
                with open(semgrep_file) as f:
                    semgrep_data = json.load(f)

                findings = len(semgrep_data.get("results", []))
                if findings > 10:  # Threshold
                    warnings.append(f"Semgrep found {findings} issues")
            except Exception as e:
                issues.append(f"Failed to parse semgrep report: {e}")
        else:
            warnings.append("Semgrep static analysis report not found")

        status = "failed" if issues else ("warning" if warnings else "passed")

        return {
            "category": "security",
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "artifacts_found": len(list(security_dir.glob("*"))) if security_dir.exists() else 0,
        }

    def _validate_sbom_artifacts(self) -> dict:
        """Validate SBOM artifacts."""
        logger.info("Validating SBOM artifacts...")

        sbom_dir = self.artifacts_dir / "sbom"
        issues = []
        warnings = []

        # Check for SPDX SBOM
        spdx_file = sbom_dir / "aivillage-sbom.spdx"
        if spdx_file.exists():
            try:
                # Basic validation - check file size
                size = spdx_file.stat().st_size
                if size < 100:  # Too small
                    warnings.append("SPDX SBOM file seems too small")
            except Exception as e:
                issues.append(f"Failed to validate SPDX SBOM: {e}")
        else:
            # Check for alternative SBOM formats
            basic_sbom = sbom_dir / "basic-sbom.txt"
            if not basic_sbom.exists():
                warnings.append("No SBOM artifacts found")

        # Check for CycloneDX SBOM
        cyclonedx_file = sbom_dir / "aivillage-sbom.json"
        if cyclonedx_file.exists():
            try:
                with open(cyclonedx_file) as f:
                    sbom_data = json.load(f)

                components = sbom_data.get("components", [])
                if len(components) < 10:  # Threshold
                    warnings.append("SBOM has very few components")
            except Exception as e:
                issues.append(f"Failed to parse CycloneDX SBOM: {e}")

        status = "failed" if issues else ("warning" if warnings else "passed")

        return {
            "category": "sbom",
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "artifacts_found": len(list(sbom_dir.glob("*"))) if sbom_dir.exists() else 0,
        }

    def _validate_performance_artifacts(self) -> dict:
        """Validate performance artifacts."""
        logger.info("Validating performance artifacts...")

        performance_dir = self.artifacts_dir / "performance"
        issues = []
        warnings = []

        # Check for benchmark results
        benchmark_file = performance_dir / "benchmark-results.json"
        if benchmark_file.exists():
            try:
                with open(benchmark_file) as f:
                    benchmark_data = json.load(f)

                benchmarks = benchmark_data.get("benchmarks", [])
                if len(benchmarks) == 0:
                    warnings.append("No benchmarks found in results")

                # Check for regressions
                for benchmark in benchmarks:
                    stats = benchmark.get("stats", {})
                    mean_time = stats.get("mean", 0)
                    if mean_time > 5.0:  # 5 second threshold
                        warnings.append(f"Slow benchmark: {benchmark.get('name')} ({mean_time:.2f}s)")
            except Exception as e:
                issues.append(f"Failed to parse benchmark results: {e}")
        else:
            warnings.append("Benchmark results not found")

        # Check for memory profile
        memory_file = performance_dir / "memory-profile.txt"
        if not memory_file.exists():
            warnings.append("Memory profile not found")

        status = "failed" if issues else ("warning" if warnings else "passed")

        return {
            "category": "performance",
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "artifacts_found": len(list(performance_dir.glob("*"))) if performance_dir.exists() else 0,
        }

    def _validate_quality_artifacts(self) -> dict:
        """Validate code quality artifacts."""
        logger.info("Validating quality artifacts...")

        quality_dir = self.artifacts_dir / "quality"
        issues = []
        warnings = []

        # Check for Ruff report
        ruff_file = quality_dir / "ruff-report.json"
        if ruff_file.exists():
            try:
                with open(ruff_file) as f:
                    ruff_data = json.load(f)

                if len(ruff_data) > 100:  # Too many issues
                    warnings.append(f"Ruff found {len(ruff_data)} linting issues")
            except Exception as e:
                issues.append(f"Failed to parse Ruff report: {e}")
        else:
            warnings.append("Ruff linting report not found")

        # Check for MyPy report
        mypy_file = quality_dir / "mypy-report.txt"
        if mypy_file.exists():
            try:
                with open(mypy_file) as f:
                    content = f.read()

                error_count = content.count("error:")
                if error_count > 50:  # Threshold
                    warnings.append(f"MyPy found {error_count} type errors")
            except Exception as e:
                issues.append(f"Failed to parse MyPy report: {e}")
        else:
            warnings.append("MyPy type checking report not found")

        # Check for complexity report
        complexity_file = quality_dir / "complexity-report.json"
        if complexity_file.exists():
            try:
                with open(complexity_file) as f:
                    complexity_data = json.load(f)

                # Look for high complexity functions
                high_complexity = 0
                for file_data in complexity_data.values():
                    if isinstance(file_data, list):
                        for func in file_data:
                            if func.get("complexity", 0) > 10:
                                high_complexity += 1

                if high_complexity > 10:
                    warnings.append(f"Found {high_complexity} high-complexity functions")
            except Exception as e:
                issues.append(f"Failed to parse complexity report: {e}")
        else:
            warnings.append("Code complexity report not found")

        status = "failed" if issues else ("warning" if warnings else "passed")

        return {
            "category": "quality",
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "artifacts_found": len(list(quality_dir.glob("*"))) if quality_dir.exists() else 0,
        }

    def _validate_container_artifacts(self) -> dict:
        """Validate container security artifacts."""
        logger.info("Validating container artifacts...")

        container_dir = self.artifacts_dir / "containers"
        issues = []
        warnings = []

        # Check for Trivy report
        trivy_file = container_dir / "trivy-report.json"
        if trivy_file.exists():
            try:
                with open(trivy_file) as f:
                    trivy_data = json.load(f)

                results = trivy_data.get("Results", [])
                total_vulns = sum(len(r.get("Vulnerabilities", [])) for r in results)

                if total_vulns > 0:
                    warnings.append(f"Trivy found {total_vulns} container vulnerabilities")
            except Exception as e:
                issues.append(f"Failed to parse Trivy report: {e}")
        else:
            warnings.append("Trivy container security report not found")

        # Check for Grype report
        grype_file = container_dir / "grype-report.json"
        if grype_file.exists():
            try:
                with open(grype_file) as f:
                    grype_data = json.load(f)

                matches = grype_data.get("matches", [])
                if len(matches) > 0:
                    warnings.append(f"Grype found {len(matches)} vulnerability matches")
            except Exception as e:
                issues.append(f"Failed to parse Grype report: {e}")
        else:
            warnings.append("Grype vulnerability report not found")

        status = "failed" if issues else ("warning" if warnings else "passed")

        return {
            "category": "container",
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "artifacts_found": len(list(container_dir.glob("*"))) if container_dir.exists() else 0,
        }

    def _validate_compliance_artifacts(self) -> dict:
        """Validate compliance artifacts."""
        logger.info("Validating compliance artifacts...")

        compliance_dir = self.artifacts_dir / "compliance"
        issues = []
        warnings = []

        # Check for GDPR compliance report
        gdpr_file = compliance_dir / "gdpr-compliance.json"
        if gdpr_file.exists():
            try:
                with open(gdpr_file) as f:
                    gdpr_data = json.load(f)

                checks = gdpr_data.get("checks", {})
                failed_checks = [k for k, v in checks.items() if v not in ["implemented", "configured"]]

                if failed_checks:
                    warnings.append(f"GDPR compliance issues: {', '.join(failed_checks)}")
            except Exception as e:
                issues.append(f"Failed to parse GDPR report: {e}")
        else:
            warnings.append("GDPR compliance report not found")

        # Check for SOC2 compliance report
        soc2_file = compliance_dir / "soc2-compliance.json"
        if soc2_file.exists():
            try:
                with open(soc2_file) as f:
                    soc2_data = json.load(f)

                criteria = soc2_data.get("trust_services_criteria", {})
                failed_criteria = [k for k, v in criteria.items() if v == "partial"]

                if failed_criteria:
                    warnings.append(f"SOC2 partial compliance: {', '.join(failed_criteria)}")
            except Exception as e:
                issues.append(f"Failed to parse SOC2 report: {e}")
        else:
            warnings.append("SOC2 compliance report not found")

        status = "failed" if issues else ("warning" if warnings else "passed")

        return {
            "category": "compliance",
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "artifacts_found": len(list(compliance_dir.glob("*"))) if compliance_dir.exists() else 0,
        }


def main():
    """Main validation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate AIVillage operational artifacts")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory")
    parser.add_argument("--output", help="Output validation report file")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")

    args = parser.parse_args()

    print("🔍 AIVillage Artifacts Validation")
    print("=" * 50)

    # Initialize validator
    validator = ArtifactsValidator(args.artifacts_dir)

    # Run validation
    success, report = validator.validate_all_artifacts()

    # Output results
    print("\n📊 Validation Summary:")
    print(f"  Total Validations: {report['summary']['total_validations']}")
    print(f"  Passed: {report['summary']['passed']}")
    print(f"  Failed: {report['summary']['failed']}")
    print(f"  Warnings: {report['summary']['warnings']}")

    # Show detailed results
    print("\n📋 Detailed Results:")
    for validation in report["validations"]:
        status_emoji = (
            "✅" if validation["status"] == "passed" else ("⚠️" if validation["status"] == "warning" else "❌")
        )
        print(
            f"  {status_emoji} {validation['category'].title()}: {validation['status']} ({validation['artifacts_found']} artifacts)"
        )

        for issue in validation["issues"]:
            print(f"    ❌ {issue}")
        for warning in validation["warnings"]:
            print(f"    ⚠️  {warning}")

    # Save report if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Validation report saved: {args.output}")

    # Exit with appropriate code
    if not success or (args.strict and report["summary"]["warnings"] > 0):
        print("\n❌ Validation failed")
        sys.exit(1)
    else:
        print("\n✅ Validation passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
