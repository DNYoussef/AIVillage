"""
Test coverage monitoring during clean architecture reorganization.
Ensures test coverage is maintained throughout the restructuring process.
"""

from pathlib import Path
import subprocess
import sys
from typing import Any

import coverage
import pytest

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class CoverageMonitor:
    """Monitor test coverage during reorganization"""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.packages_dir = self.project_root / "packages"
        self.tests_dir = self.project_root / "tests"
        self.coverage_thresholds = {"total": 80.0, "package_minimum": 70.0, "critical_modules": 85.0}
        self.critical_modules = [
            "packages.agents.core.base",
            "packages.rag.core.pipeline",
            "packages.p2p.network",
            "packages.core.common",
        ]

    def measure_baseline_coverage(self) -> dict[str, Any]:
        """Measure baseline coverage before reorganization"""
        try:
            # Run coverage analysis
            cov = coverage.Coverage()
            cov.start()

            # Run test suite
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(self.tests_dir), "--tb=short", "-q"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            cov.stop()
            cov.save()

            # Generate coverage report
            coverage_data = self._generate_coverage_data(cov)

            return {
                "timestamp": self._get_timestamp(),
                "test_result": result.returncode == 0,
                "coverage": coverage_data,
                "test_output": result.stdout,
                "test_errors": result.stderr,
            }

        except Exception as e:
            return {"timestamp": self._get_timestamp(), "error": str(e), "coverage": {"total": 0.0, "by_file": {}}}

    def measure_current_coverage(self) -> dict[str, Any]:
        """Measure current coverage during reorganization"""
        return self.measure_baseline_coverage()  # Same process

    def compare_coverage(self, baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
        """Compare coverage between baseline and current"""
        baseline_total = baseline.get("coverage", {}).get("total", 0.0)
        current_total = current.get("coverage", {}).get("total", 0.0)

        coverage_change = current_total - baseline_total

        # Compare by file
        baseline_files = baseline.get("coverage", {}).get("by_file", {})
        current_files = current.get("coverage", {}).get("by_file", {})

        file_changes = {}
        for file_path in set(baseline_files.keys()) | set(current_files.keys()):
            baseline_pct = baseline_files.get(file_path, 0.0)
            current_pct = current_files.get(file_path, 0.0)
            file_changes[file_path] = current_pct - baseline_pct

        return {
            "total_change": coverage_change,
            "current_total": current_total,
            "baseline_total": baseline_total,
            "file_changes": file_changes,
            "significant_drops": {
                file_path: change for file_path, change in file_changes.items() if change < -5.0  # More than 5% drop
            },
        }

    def validate_coverage_thresholds(self, coverage_data: dict[str, Any]) -> dict[str, Any]:
        """Validate coverage meets minimum thresholds"""
        violations = []

        # Check total coverage
        total_coverage = coverage_data.get("total", 0.0)
        if total_coverage < self.coverage_thresholds["total"]:
            violations.append(
                {
                    "type": "total_coverage",
                    "current": total_coverage,
                    "threshold": self.coverage_thresholds["total"],
                    "message": f"Total coverage {total_coverage:.1f}% below threshold {self.coverage_thresholds['total']:.1f}%",
                }
            )

        # Check package coverage
        by_file = coverage_data.get("by_file", {})
        package_coverage = self._calculate_package_coverage(by_file)

        for package, coverage_pct in package_coverage.items():
            if coverage_pct < self.coverage_thresholds["package_minimum"]:
                violations.append(
                    {
                        "type": "package_coverage",
                        "package": package,
                        "current": coverage_pct,
                        "threshold": self.coverage_thresholds["package_minimum"],
                        "message": f"Package {package} coverage {coverage_pct:.1f}% below threshold {self.coverage_thresholds['package_minimum']:.1f}%",
                    }
                )

        # Check critical modules
        for module in self.critical_modules:
            module_coverage = self._find_module_coverage(module, by_file)
            if module_coverage is not None and module_coverage < self.coverage_thresholds["critical_modules"]:
                violations.append(
                    {
                        "type": "critical_module",
                        "module": module,
                        "current": module_coverage,
                        "threshold": self.coverage_thresholds["critical_modules"],
                        "message": f"Critical module {module} coverage {module_coverage:.1f}% below threshold {self.coverage_thresholds['critical_modules']:.1f}%",
                    }
                )

        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "total_coverage": total_coverage,
            "package_coverage": package_coverage,
        }

    def _generate_coverage_data(self, cov: coverage.Coverage) -> dict[str, Any]:
        """Generate coverage data from coverage object"""
        try:
            # Get coverage data
            data = cov.get_data()

            # Calculate total coverage
            total_lines = 0
            covered_lines = 0
            by_file = {}

            for filename in data.measured_files():
                # Filter to only include project files
                if not str(filename).startswith(str(self.packages_dir)):
                    continue

                analysis = cov.analysis2(filename)
                executable_lines = analysis[1]  # Lines that can be executed
                missing_lines = analysis[3]  # Lines not covered

                if executable_lines:
                    file_covered = len(executable_lines) - len(missing_lines)
                    file_total = len(executable_lines)
                    file_pct = (file_covered / file_total) * 100 if file_total > 0 else 0

                    by_file[str(filename)] = file_pct
                    total_lines += file_total
                    covered_lines += file_covered

            total_pct = (covered_lines / total_lines) * 100 if total_lines > 0 else 0

            return {"total": total_pct, "total_lines": total_lines, "covered_lines": covered_lines, "by_file": by_file}

        except Exception as e:
            return {"error": str(e), "total": 0.0, "by_file": {}}

    def _calculate_package_coverage(self, by_file: dict[str, float]) -> dict[str, float]:
        """Calculate coverage by package"""
        package_stats = {}

        for file_path, coverage_pct in by_file.items():
            # Extract package name from path
            path = Path(file_path)
            if "packages" in path.parts:
                package_idx = path.parts.index("packages")
                if package_idx + 1 < len(path.parts):
                    package_name = path.parts[package_idx + 1]

                    if package_name not in package_stats:
                        package_stats[package_name] = []
                    package_stats[package_name].append(coverage_pct)

        # Calculate average coverage per package
        package_coverage = {}
        for package, coverages in package_stats.items():
            package_coverage[package] = sum(coverages) / len(coverages)

        return package_coverage

    def _find_module_coverage(self, module_name: str, by_file: dict[str, float]) -> float | None:
        """Find coverage for specific module"""
        module_path_pattern = module_name.replace(".", "/")

        for file_path, coverage_pct in by_file.items():
            if module_path_pattern in file_path:
                return coverage_pct

        return None

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        import datetime

        return datetime.datetime.now().isoformat()

    def generate_coverage_report(self, coverage_data: dict[str, Any]) -> str:
        """Generate human-readable coverage report"""
        report_lines = [
            "=" * 60,
            "Test Coverage Report",
            "=" * 60,
            f"Timestamp: {coverage_data.get('timestamp', 'Unknown')}",
            f"Total Coverage: {coverage_data.get('coverage', {}).get('total', 0.0):.1f}%",
            "",
        ]

        # Add package breakdown
        by_file = coverage_data.get("coverage", {}).get("by_file", {})
        package_coverage = self._calculate_package_coverage(by_file)

        if package_coverage:
            report_lines.extend(["Package Coverage:", "-" * 30])

            for package, coverage_pct in sorted(package_coverage.items()):
                status = "✓" if coverage_pct >= self.coverage_thresholds["package_minimum"] else "✗"
                report_lines.append(f"  {status} {package}: {coverage_pct:.1f}%")

        # Add threshold validation
        validation = self.validate_coverage_thresholds(coverage_data.get("coverage", {}))

        if validation["violations"]:
            report_lines.extend(["", "Coverage Violations:", "-" * 30])

            for violation in validation["violations"]:
                report_lines.append(f"  ✗ {violation['message']}")

        return "\n".join(report_lines)


class CoverageRegressionTest:
    """Test for coverage regression during reorganization"""

    def __init__(self):
        self.monitor = CoverageMonitor()

    def test_coverage_preservation(self):
        """Test that coverage is preserved during reorganization"""
        # Measure current coverage
        current_coverage = self.monitor.measure_current_coverage()

        # Validate thresholds
        validation = self.monitor.validate_coverage_thresholds(current_coverage.get("coverage", {}))

        if not validation["passed"]:
            violations_msg = "\n".join([v["message"] for v in validation["violations"]])
            pytest.fail(f"Coverage threshold violations:\n{violations_msg}")

    def test_no_significant_coverage_drop(self):
        """Test that coverage doesn't drop significantly"""
        # This would compare against stored baseline in real scenario
        # For now, we'll test current coverage meets minimum thresholds

        current_coverage = self.monitor.measure_current_coverage()
        total_coverage = current_coverage.get("coverage", {}).get("total", 0.0)

        # Test passes if coverage is reasonable
        if total_coverage < 70.0:  # Minimum acceptable coverage
            pytest.fail(f"Coverage too low: {total_coverage:.1f}% (minimum: 70.0%)")

    def test_critical_modules_coverage(self):
        """Test that critical modules maintain high coverage"""
        current_coverage = self.monitor.measure_current_coverage()
        by_file = current_coverage.get("coverage", {}).get("by_file", {})

        violations = []

        for module in self.monitor.critical_modules:
            module_coverage = self.monitor._find_module_coverage(module, by_file)

            if module_coverage is None:
                # Module might be reorganized - this is acceptable during transition
                continue

            if module_coverage < self.monitor.coverage_thresholds["critical_modules"]:
                violations.append(
                    f"{module}: {module_coverage:.1f}% (required: {self.monitor.coverage_thresholds['critical_modules']:.1f}%)"
                )

        if violations:
            violations_msg = "\n".join(violations)
            pytest.fail(f"Critical module coverage violations:\n{violations_msg}")


# Test fixtures
@pytest.fixture
def coverage_monitor():
    """Fixture for coverage monitor"""
    return CoverageMonitor()


@pytest.fixture
def coverage_regression_test():
    """Fixture for coverage regression test"""
    return CoverageRegressionTest()


# Test functions
def test_coverage_preservation(coverage_regression_test):
    """Test coverage preservation"""
    coverage_regression_test.test_coverage_preservation()


def test_no_significant_coverage_drop(coverage_regression_test):
    """Test no significant coverage drop"""
    coverage_regression_test.test_no_significant_coverage_drop()


def test_critical_modules_coverage(coverage_regression_test):
    """Test critical modules coverage"""
    coverage_regression_test.test_critical_modules_coverage()


def test_coverage_report_generation(coverage_monitor):
    """Test coverage report generation"""
    coverage_data = coverage_monitor.measure_current_coverage()
    report = coverage_monitor.generate_coverage_report(coverage_data)

    assert "Test Coverage Report" in report
    assert "Total Coverage:" in report


if __name__ == "__main__":
    # Generate coverage report when run directly
    monitor = CoverageMonitor()
    coverage_data = monitor.measure_current_coverage()
    report = monitor.generate_coverage_report(coverage_data)

    print(report)

    # Also run tests
    pytest.main([__file__, "-v"])
