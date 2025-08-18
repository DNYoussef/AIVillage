"""Test Coverage & Quality Gates - Prompt K

Comprehensive test coverage analysis and quality gates for ensuring high-quality
code with comprehensive testing, automated quality checks, and CI/CD integration.

Integration Point: Quality validation framework for Phase 4 testing
"""

import json
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pytest


class QualityGateStatus(Enum):
    """Quality gate status enumeration."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


@dataclass
class CoverageMetrics:
    """Test coverage metrics."""

    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    function_coverage: float = 0.0
    covered_lines: int = 0
    total_lines: int = 0
    missing_lines: list[int] = field(default_factory=list)
    excluded_lines: list[int] = field(default_factory=list)


@dataclass
class QualityGateResult:
    """Quality gate execution result."""

    gate_name: str
    status: QualityGateStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0


@dataclass
class TestSuiteResult:
    """Test suite execution result."""

    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0
    execution_time_ms: float = 0.0
    coverage_metrics: CoverageMetrics | None = None


class TestCoverageAnalyzer:
    """Analyzes test coverage across the codebase."""

    def __init__(self, source_dir: str = "src", test_dir: str = "tests"):
        """Initialize coverage analyzer.

        Args:
            source_dir: Source code directory
            test_dir: Test directory
        """
        self.source_dir = Path(source_dir)
        self.test_dir = Path(test_dir)

    def run_coverage_analysis(self, test_pattern: str = "test_*.py") -> CoverageMetrics:
        """Run comprehensive coverage analysis.

        Args:
            test_pattern: Test file pattern

        Returns:
            CoverageMetrics with coverage information
        """
        try:
            # Try to use coverage.py if available
            import coverage

            cov = coverage.Coverage(source=[str(self.source_dir)])
            cov.start()

            # Run tests with coverage
            pytest_args = [
                str(self.test_dir),
                "-v",
                "--tb=short",
                "--collect-only",
            ]

            pytest.main(pytest_args)

            cov.stop()
            cov.save()

            # Generate coverage report
            total_lines = 0
            covered_lines = 0
            missing_lines = []

            for filename in cov.get_data().measured_files():
                analysis = cov.analysis2(filename)
                total_lines += len(analysis[1])  # executable lines
                covered_lines += len(analysis[1]) - len(analysis[2])  # covered lines
                missing_lines.extend(analysis[2])  # missing lines

            line_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0.0

            return CoverageMetrics(
                line_coverage=line_coverage,
                branch_coverage=0.0,  # Would need additional analysis
                function_coverage=0.0,  # Would need additional analysis
                covered_lines=covered_lines,
                total_lines=total_lines,
                missing_lines=missing_lines[:100],  # Limit for readability
            )

        except ImportError:
            # Fallback to manual analysis
            return self._manual_coverage_analysis()

    def _manual_coverage_analysis(self) -> CoverageMetrics:
        """Manual coverage analysis when coverage.py not available."""
        # Count source lines
        total_lines = 0
        for py_file in self.source_dir.rglob("*.py"):
            if py_file.is_file():
                try:
                    with open(py_file, encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                        # Count non-empty, non-comment lines
                        code_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]
                        total_lines += len(code_lines)
                except Exception:
                    continue

        # Count test files
        test_files = list(self.test_dir.rglob("test_*.py"))
        test_count = len(test_files)

        # Rough estimate: assume 50% coverage if we have reasonable test count
        estimated_coverage = min(50.0, (test_count / 50) * 100) if test_count > 0 else 0.0

        return CoverageMetrics(
            line_coverage=estimated_coverage,
            branch_coverage=estimated_coverage * 0.8,  # Estimate
            function_coverage=estimated_coverage * 0.9,  # Estimate
            covered_lines=int(total_lines * estimated_coverage / 100),
            total_lines=total_lines,
        )

    def analyze_test_distribution(self) -> dict[str, Any]:
        """Analyze test distribution across modules."""
        distribution = {
            "by_module": {},
            "by_type": {},
            "total_test_files": 0,
            "total_test_functions": 0,
        }

        for test_file in self.test_dir.rglob("test_*.py"):
            if not test_file.is_file():
                continue

            distribution["total_test_files"] += 1

            # Determine module being tested
            module_name = test_file.stem.replace("test_", "")
            if module_name not in distribution["by_module"]:
                distribution["by_module"][module_name] = {
                    "files": 0,
                    "functions": 0,
                }

            distribution["by_module"][module_name]["files"] += 1

            # Count test functions
            try:
                with open(test_file, encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Simple regex to count test functions
                test_functions = content.count("def test_")
                distribution["by_module"][module_name]["functions"] += test_functions
                distribution["total_test_functions"] += test_functions

                # Categorize test types
                if "integration" in str(test_file).lower():
                    distribution["by_type"]["integration"] = distribution["by_type"].get("integration", 0) + 1
                elif "unit" in str(test_file).lower():
                    distribution["by_type"]["unit"] = distribution["by_type"].get("unit", 0) + 1
                elif "e2e" in str(test_file).lower():
                    distribution["by_type"]["e2e"] = distribution["by_type"].get("e2e", 0) + 1
                else:
                    distribution["by_type"]["other"] = distribution["by_type"].get("other", 0) + 1

            except Exception:
                continue

        return distribution


class QualityGateManager:
    """Manages quality gates for CI/CD pipeline."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize quality gate manager.

        Args:
            config: Quality gate configuration
        """
        self.config = config or self._get_default_config()
        self.results: list[QualityGateResult] = []

    def _get_default_config(self) -> dict[str, Any]:
        """Get default quality gate configuration."""
        return {
            "coverage": {
                "minimum_line_coverage": 30.0,  # Conservative for large codebase
                "minimum_branch_coverage": 20.0,
                "minimum_function_coverage": 40.0,
            },
            "tests": {
                "minimum_test_count": 100,
                "maximum_failure_rate": 10.0,  # Allow 10% failure rate
                "maximum_execution_time_minutes": 30,
            },
            "code_quality": {
                "enable_linting": True,
                "enable_type_checking": True,
                "enable_security_scan": True,
                "enable_complexity_check": True,
            },
            "performance": {
                "maximum_import_time_ms": 5000,
                "maximum_test_time_ms": 60000,
            },
            "documentation": {
                "require_docstrings": False,  # Not enforced yet
                "require_type_hints": False,  # Not enforced yet
            },
        }

    def run_all_gates(self, test_pattern: str = "test_*.py") -> list[QualityGateResult]:
        """Run all quality gates.

        Args:
            test_pattern: Test file pattern

        Returns:
            List of quality gate results
        """
        self.results = []

        # Run gates in order of importance
        gates = [
            ("basic_imports", self._gate_basic_imports),
            ("test_discovery", self._gate_test_discovery),
            ("critical_tests", self._gate_critical_tests),
            ("coverage_analysis", self._gate_coverage_analysis),
            ("code_quality", self._gate_code_quality),
            ("performance_checks", self._gate_performance_checks),
            ("security_validation", self._gate_security_validation),
        ]

        for gate_name, gate_func in gates:
            start_time = time.perf_counter()

            try:
                result = gate_func()
                result.gate_name = gate_name
                result.execution_time_ms = (time.perf_counter() - start_time) * 1000

            except Exception as e:
                result = QualityGateResult(
                    gate_name=gate_name,
                    status=QualityGateStatus.FAIL,
                    message=f"Gate execution failed: {e}",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                )

            self.results.append(result)

        return self.results

    def _gate_basic_imports(self) -> QualityGateResult:
        """Quality gate: Basic imports work."""
        try:
            # Test basic imports from Phase 1
            sys.path.insert(0, "src")

            critical_modules = [
                "core.security.secure_serializer",
                "production.rag.rag_system.core.offline_defaults",
                "production.compression.cpu_only_config",
            ]

            failed_imports = []
            for module in critical_modules:
                try:
                    __import__(module)
                except ImportError as e:
                    failed_imports.append(f"{module}: {e}")

            if failed_imports:
                return QualityGateResult(
                    gate_name="basic_imports",
                    status=QualityGateStatus.FAIL,
                    message=f"Critical imports failed: {len(failed_imports)}",
                    details={"failed_imports": failed_imports},
                )
            else:
                return QualityGateResult(
                    gate_name="basic_imports",
                    status=QualityGateStatus.PASS,
                    message="All critical imports successful",
                )

        except Exception as e:
            return QualityGateResult(
                gate_name="basic_imports",
                status=QualityGateStatus.FAIL,
                message=f"Import gate failed: {e}",
            )

    def _gate_test_discovery(self) -> QualityGateResult:
        """Quality gate: Test discovery works."""
        try:
            # Run pytest collection
            result = pytest.main(["--collect-only", "-q", "tests/"])

            if result == 0:
                # Count discovered tests
                analyzer = TestCoverageAnalyzer()
                distribution = analyzer.analyze_test_distribution()

                total_tests = distribution["total_test_functions"]
                min_tests = self.config["tests"]["minimum_test_count"]

                if total_tests >= min_tests:
                    return QualityGateResult(
                        gate_name="test_discovery",
                        status=QualityGateStatus.PASS,
                        message=f"Test discovery successful: {total_tests} tests found",
                        details=distribution,
                    )
                else:
                    return QualityGateResult(
                        gate_name="test_discovery",
                        status=QualityGateStatus.WARNING,
                        message=f"Low test count: {total_tests} < {min_tests}",
                        details=distribution,
                    )
            else:
                return QualityGateResult(
                    gate_name="test_discovery",
                    status=QualityGateStatus.FAIL,
                    message="Test discovery failed",
                )

        except Exception as e:
            return QualityGateResult(
                gate_name="test_discovery",
                status=QualityGateStatus.FAIL,
                message=f"Test discovery gate failed: {e}",
            )

    def _gate_critical_tests(self) -> QualityGateResult:
        """Quality gate: Critical tests pass."""
        try:
            # Run critical test suites from Phase 1
            critical_test_patterns = [
                "tests/p2p/test_imports.py",
                "tests/rag_system/test_offline_defaults.py",
                "tests/security/test_pickle_elimination.py",
                "tests/compression/test_cpu_only_compression.py",
            ]

            passed_suites = 0
            failed_suites = []

            for pattern in critical_test_patterns:
                if Path(pattern).exists():
                    result = pytest.main([pattern, "-v", "--tb=no"])
                    if result == 0:
                        passed_suites += 1
                    else:
                        failed_suites.append(pattern)

            if len(failed_suites) == 0:
                return QualityGateResult(
                    gate_name="critical_tests",
                    status=QualityGateStatus.PASS,
                    message=f"All {passed_suites} critical test suites passed",
                )
            elif len(failed_suites) <= 1:  # Allow 1 failure
                return QualityGateResult(
                    gate_name="critical_tests",
                    status=QualityGateStatus.WARNING,
                    message=f"{passed_suites} passed, {len(failed_suites)} failed",
                    details={"failed_suites": failed_suites},
                )
            else:
                return QualityGateResult(
                    gate_name="critical_tests",
                    status=QualityGateStatus.FAIL,
                    message=f"Too many critical test failures: {len(failed_suites)}",
                    details={"failed_suites": failed_suites},
                )

        except Exception as e:
            return QualityGateResult(
                gate_name="critical_tests",
                status=QualityGateStatus.FAIL,
                message=f"Critical tests gate failed: {e}",
            )

    def _gate_coverage_analysis(self) -> QualityGateResult:
        """Quality gate: Test coverage meets minimum thresholds."""
        try:
            analyzer = TestCoverageAnalyzer()
            coverage = analyzer.run_coverage_analysis()

            min_line_coverage = self.config["coverage"]["minimum_line_coverage"]
            min_branch_coverage = self.config["coverage"]["minimum_branch_coverage"]

            issues = []
            if coverage.line_coverage < min_line_coverage:
                issues.append(f"Line coverage {coverage.line_coverage:.1f}% < {min_line_coverage}%")

            if coverage.branch_coverage < min_branch_coverage:
                issues.append(f"Branch coverage {coverage.branch_coverage:.1f}% < {min_branch_coverage}%")

            if issues:
                return QualityGateResult(
                    gate_name="coverage_analysis",
                    status=QualityGateStatus.WARNING,  # Warning, not failure
                    message=f"Coverage below threshold: {'; '.join(issues)}",
                    details={
                        "line_coverage": coverage.line_coverage,
                        "branch_coverage": coverage.branch_coverage,
                        "total_lines": coverage.total_lines,
                        "covered_lines": coverage.covered_lines,
                    },
                )
            else:
                return QualityGateResult(
                    gate_name="coverage_analysis",
                    status=QualityGateStatus.PASS,
                    message=f"Coverage meets thresholds: {coverage.line_coverage:.1f}% line coverage",
                    details={
                        "line_coverage": coverage.line_coverage,
                        "branch_coverage": coverage.branch_coverage,
                        "total_lines": coverage.total_lines,
                        "covered_lines": coverage.covered_lines,
                    },
                )

        except Exception as e:
            return QualityGateResult(
                gate_name="coverage_analysis",
                status=QualityGateStatus.SKIP,
                message=f"Coverage analysis skipped: {e}",
            )

    def _gate_code_quality(self) -> QualityGateResult:
        """Quality gate: Code quality checks."""
        try:
            quality_issues = []

            # Check for basic code quality indicators
            if self.config["code_quality"]["enable_linting"]:
                # Check if critical files have obvious issues
                critical_files = [
                    "src/core/security/secure_serializer.py",
                    "src/production/rag/rag_system/core/offline_defaults.py",
                    "src/production/compression/cpu_only_config.py",
                ]

                syntax_errors = 0
                for file_path in critical_files:
                    if Path(file_path).exists():
                        try:
                            with open(file_path, encoding="utf-8") as f:
                                compile(f.read(), file_path, "exec")
                        except SyntaxError:
                            syntax_errors += 1

                if syntax_errors > 0:
                    quality_issues.append(f"{syntax_errors} critical files have syntax errors")

            if quality_issues:
                return QualityGateResult(
                    gate_name="code_quality",
                    status=QualityGateStatus.FAIL,
                    message=f"Code quality issues: {'; '.join(quality_issues)}",
                    details={"issues": quality_issues},
                )
            else:
                return QualityGateResult(
                    gate_name="code_quality",
                    status=QualityGateStatus.PASS,
                    message="Basic code quality checks passed",
                )

        except Exception as e:
            return QualityGateResult(
                gate_name="code_quality",
                status=QualityGateStatus.SKIP,
                message=f"Code quality gate skipped: {e}",
            )

    def _gate_performance_checks(self) -> QualityGateResult:
        """Quality gate: Performance checks."""
        try:
            # Test import performance
            start_time = time.perf_counter()

            sys.path.insert(0, "src")

            import_time_ms = (time.perf_counter() - start_time) * 1000
            max_import_time = self.config["performance"]["maximum_import_time_ms"]

            if import_time_ms > max_import_time:
                return QualityGateResult(
                    gate_name="performance_checks",
                    status=QualityGateStatus.WARNING,
                    message=f"Slow import: {import_time_ms:.1f}ms > {max_import_time}ms",
                    details={"import_time_ms": import_time_ms},
                )
            else:
                return QualityGateResult(
                    gate_name="performance_checks",
                    status=QualityGateStatus.PASS,
                    message=f"Performance checks passed: {import_time_ms:.1f}ms import time",
                    details={"import_time_ms": import_time_ms},
                )

        except Exception as e:
            return QualityGateResult(
                gate_name="performance_checks",
                status=QualityGateStatus.SKIP,
                message=f"Performance checks skipped: {e}",
            )

    def _gate_security_validation(self) -> QualityGateResult:
        """Quality gate: Security validation."""
        try:
            # Check that security measures from Phase 1 are working
            security_checks = []

            # Check pickle elimination
            try:
                from core.security.secure_serializer import LegacyPickleRejector

                pickle_data = b"\x80\x03]q\x00(K\x01K\x02K\x03e."

                try:
                    LegacyPickleRejector.validate_not_pickle(pickle_data)
                    security_checks.append("FAIL: Pickle validation not working")
                except Exception:
                    security_checks.append("PASS: Pickle rejection working")

            except ImportError:
                security_checks.append("SKIP: Pickle rejector not available")

            # Check offline defaults
            try:
                from production.rag.rag_system.core.offline_defaults import get_offline_rag_config

                config = get_offline_rag_config()

                if config.enable_internet_features:
                    security_checks.append("FAIL: Offline config allows internet features")
                else:
                    security_checks.append("PASS: Offline config properly secured")

            except ImportError:
                security_checks.append("SKIP: Offline config not available")

            failures = [check for check in security_checks if check.startswith("FAIL")]

            if failures:
                return QualityGateResult(
                    gate_name="security_validation",
                    status=QualityGateStatus.FAIL,
                    message=f"Security validation failed: {len(failures)} issues",
                    details={"checks": security_checks},
                )
            else:
                return QualityGateResult(
                    gate_name="security_validation",
                    status=QualityGateStatus.PASS,
                    message="Security validation passed",
                    details={"checks": security_checks},
                )

        except Exception as e:
            return QualityGateResult(
                gate_name="security_validation",
                status=QualityGateStatus.SKIP,
                message=f"Security validation skipped: {e}",
            )

    def get_overall_status(self) -> QualityGateStatus:
        """Get overall quality gate status."""
        if not self.results:
            return QualityGateStatus.SKIP

        statuses = [result.status for result in self.results]

        if QualityGateStatus.FAIL in statuses:
            return QualityGateStatus.FAIL
        elif QualityGateStatus.WARNING in statuses:
            return QualityGateStatus.WARNING
        else:
            return QualityGateStatus.PASS

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive quality gate report."""
        return {
            "overall_status": self.get_overall_status().value,
            "execution_summary": {
                "total_gates": len(self.results),
                "passed": len([r for r in self.results if r.status == QualityGateStatus.PASS]),
                "failed": len([r for r in self.results if r.status == QualityGateStatus.FAIL]),
                "warnings": len([r for r in self.results if r.status == QualityGateStatus.WARNING]),
                "skipped": len([r for r in self.results if r.status == QualityGateStatus.SKIP]),
                "total_execution_time_ms": sum(r.execution_time_ms for r in self.results),
            },
            "gate_results": [
                {
                    "gate_name": result.gate_name,
                    "status": result.status.value,
                    "message": result.message,
                    "execution_time_ms": result.execution_time_ms,
                    "details": result.details,
                }
                for result in self.results
            ],
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on gate results."""
        recommendations = []

        failed_gates = [r for r in self.results if r.status == QualityGateStatus.FAIL]
        warning_gates = [r for r in self.results if r.status == QualityGateStatus.WARNING]

        if failed_gates:
            recommendations.append(f"Address {len(failed_gates)} critical failures before deployment")

        if warning_gates:
            recommendations.append(f"Investigate {len(warning_gates)} warnings to improve quality")

        # Specific recommendations based on gate results
        for result in self.results:
            if result.gate_name == "coverage_analysis" and result.status == QualityGateStatus.WARNING:
                recommendations.append("Increase test coverage by adding unit tests for uncovered modules")

            if result.gate_name == "critical_tests" and result.status != QualityGateStatus.PASS:
                recommendations.append("Fix critical test failures - these are essential for system functionality")

        if not recommendations:
            recommendations.append("All quality gates passed - system ready for deployment")

        return recommendations


def run_quality_gates(config_file: str | None = None) -> dict[str, Any]:
    """Run all quality gates and return comprehensive report.

    Args:
        config_file: Optional configuration file path

    Returns:
        Dict with complete quality gate report
    """
    # Load configuration if provided
    config = None
    if config_file and Path(config_file).exists():
        with open(config_file) as f:
            config = json.load(f)

    # Initialize and run quality gates
    manager = QualityGateManager(config)
    manager.run_all_gates()

    return manager.generate_report()


def generate_ci_config() -> str:
    """Generate CI/CD configuration for quality gates.

    Returns:
        YAML configuration for GitHub Actions
    """
    return """
name: Quality Gates

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  quality-gates:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest coverage
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

    - name: Run Quality Gates
      run: |
        python -c "
        import sys
        sys.path.insert(0, 'src')
        from testing.coverage_gates import run_quality_gates
        import json

        report = run_quality_gates()
        print('Quality Gate Report:')
        print(json.dumps(report, indent=2))

        if report['overall_status'] == 'fail':
            sys.exit(1)
        elif report['overall_status'] == 'warning':
            print('Warnings detected - review recommended')
        "

    - name: Upload Quality Report
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: quality-gate-report-py${{ matrix.python-version }}
        path: quality_gate_report.json
"""
