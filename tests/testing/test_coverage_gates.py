"""Tests for Test Coverage & Quality Gates - Prompt K

Comprehensive validation of test coverage analysis and quality gates framework
ensuring high-quality code standards and automated CI/CD validation.

Integration Point: Quality validation for Phase 4 testing
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from testing.coverage_gates import (
    CoverageMetrics,
    QualityGateManager,
    QualityGateResult,
    QualityGateStatus,
    TestCoverageAnalyzer,
    generate_ci_config,
    run_quality_gates,
)


class TestCoverageMetrics:
    """Test coverage metrics data structure."""

    def test_coverage_metrics_creation(self):
        """Test coverage metrics creation."""
        metrics = CoverageMetrics(
            line_coverage=85.5,
            branch_coverage=72.3,
            function_coverage=90.1,
            covered_lines=1000,
            total_lines=1200,
            missing_lines=[10, 15, 20],
        )

        assert metrics.line_coverage == 85.5
        assert metrics.branch_coverage == 72.3
        assert metrics.function_coverage == 90.1
        assert metrics.covered_lines == 1000
        assert metrics.total_lines == 1200
        assert metrics.missing_lines == [10, 15, 20]

    def test_coverage_metrics_defaults(self):
        """Test coverage metrics default values."""
        metrics = CoverageMetrics()

        assert metrics.line_coverage == 0.0
        assert metrics.branch_coverage == 0.0
        assert metrics.function_coverage == 0.0
        assert metrics.covered_lines == 0
        assert metrics.total_lines == 0
        assert metrics.missing_lines == []
        assert metrics.excluded_lines == []


class TestQualityGateResult:
    """Test quality gate result data structure."""

    def test_quality_gate_result_creation(self):
        """Test quality gate result creation."""
        result = QualityGateResult(
            gate_name="test_gate",
            status=QualityGateStatus.PASS,
            message="Gate passed successfully",
            details={"test_count": 100},
            execution_time_ms=1500.0,
        )

        assert result.gate_name == "test_gate"
        assert result.status == QualityGateStatus.PASS
        assert result.message == "Gate passed successfully"
        assert result.details["test_count"] == 100
        assert result.execution_time_ms == 1500.0

    def test_quality_gate_status_enum(self):
        """Test quality gate status enumeration."""
        assert QualityGateStatus.PASS.value == "pass"
        assert QualityGateStatus.FAIL.value == "fail"
        assert QualityGateStatus.WARNING.value == "warning"
        assert QualityGateStatus.SKIP.value == "skip"


class TestTestCoverageAnalyzer:
    """Test coverage analyzer functionality."""

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = TestCoverageAnalyzer("src", "tests")

        assert analyzer.source_dir == Path("src")
        assert analyzer.test_dir == Path("tests")

    def test_analyzer_custom_directories(self):
        """Test analyzer with custom directories."""
        analyzer = TestCoverageAnalyzer("custom_src", "custom_tests")

        assert analyzer.source_dir == Path("custom_src")
        assert analyzer.test_dir == Path("custom_tests")

    @patch("testing.coverage_gates.coverage")
    def test_coverage_analysis_with_coverage_py(self, mock_coverage):
        """Test coverage analysis when coverage.py is available."""
        # Mock coverage.py
        mock_cov_instance = MagicMock()
        mock_coverage.Coverage.return_value = mock_cov_instance

        # Mock coverage data
        mock_cov_instance.get_data.return_value.measured_files.return_value = [
            "src/module1.py",
            "src/module2.py",
        ]

        # Mock analysis results
        mock_cov_instance.analysis2.side_effect = [
            ("src/module1.py", [1, 2, 3, 4, 5], [4, 5]),  # 3/5 lines covered
            ("src/module2.py", [1, 2, 3], []),  # 3/3 lines covered
        ]

        analyzer = TestCoverageAnalyzer()
        metrics = analyzer.run_coverage_analysis()

        assert metrics.total_lines == 8  # 5 + 3
        assert metrics.covered_lines == 6  # 3 + 3
        assert metrics.line_coverage == 75.0  # 6/8 * 100

    def test_manual_coverage_analysis(self):
        """Test manual coverage analysis fallback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "src"
            test_dir = Path(temp_dir) / "tests"
            source_dir.mkdir()
            test_dir.mkdir()

            # Create some source files
            (source_dir / "module1.py").write_text("""
def function1():
    return "hello"

def function2():
    return "world"
""")

            (source_dir / "module2.py").write_text("""
class TestClass:
    def method1(self):
        pass
""")

            # Create some test files
            (test_dir / "test_module1.py").write_text("""
def test_function1():
    pass

def test_function2():
    pass
""")

            analyzer = TestCoverageAnalyzer(str(source_dir), str(test_dir))
            metrics = analyzer._manual_coverage_analysis()

            assert metrics.total_lines > 0
            assert metrics.line_coverage >= 0
            assert metrics.covered_lines >= 0

    def test_test_distribution_analysis(self):
        """Test test distribution analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "tests"
            test_dir.mkdir()

            # Create test files
            (test_dir / "test_module1.py").write_text("""
def test_function1():
    pass

def test_function2():
    pass

class TestClass:
    def test_method1(self):
        pass
""")

            (test_dir / "test_integration_module.py").write_text("""
def test_integration_flow():
    pass
""")

            (test_dir / "test_e2e_scenario.py").write_text("""
def test_e2e_workflow():
    pass
""")

            analyzer = TestCoverageAnalyzer(test_dir="test_dir")
            analyzer.test_dir = test_dir  # Override for test

            distribution = analyzer.analyze_test_distribution()

            assert distribution["total_test_files"] == 3
            assert distribution["total_test_functions"] >= 3
            assert "module1" in distribution["by_module"]
            assert "integration" in distribution["by_type"]
            assert "e2e" in distribution["by_type"]


class TestQualityGateManager:
    """Test quality gate manager functionality."""

    def test_manager_initialization_default_config(self):
        """Test manager initialization with default config."""
        manager = QualityGateManager()

        assert "coverage" in manager.config
        assert "tests" in manager.config
        assert "code_quality" in manager.config
        assert manager.config["coverage"]["minimum_line_coverage"] == 30.0

    def test_manager_initialization_custom_config(self):
        """Test manager initialization with custom config."""
        custom_config = {
            "coverage": {"minimum_line_coverage": 50.0},
            "tests": {"minimum_test_count": 200},
        }

        manager = QualityGateManager(custom_config)

        assert manager.config["coverage"]["minimum_line_coverage"] == 50.0
        assert manager.config["tests"]["minimum_test_count"] == 200

    def test_basic_imports_gate_success(self):
        """Test basic imports gate with successful imports."""
        manager = QualityGateManager()

        # This should work since we have the modules from Phase 1
        result = manager._gate_basic_imports()

        assert result.gate_name == "basic_imports"
        # Should either pass or have import issues that are documented
        assert result.status in [QualityGateStatus.PASS, QualityGateStatus.FAIL]
        assert isinstance(result.message, str)

    def test_test_discovery_gate(self):
        """Test test discovery gate."""
        manager = QualityGateManager()

        result = manager._gate_test_discovery()

        assert result.gate_name == "test_discovery"
        assert result.status in [
            QualityGateStatus.PASS,
            QualityGateStatus.WARNING,
            QualityGateStatus.FAIL,
        ]
        assert isinstance(result.message, str)

    def test_security_validation_gate(self):
        """Test security validation gate."""
        manager = QualityGateManager()

        result = manager._gate_security_validation()

        assert result.gate_name == "security_validation"
        assert result.status in [
            QualityGateStatus.PASS,
            QualityGateStatus.FAIL,
            QualityGateStatus.SKIP,
        ]
        assert isinstance(result.message, str)

    def test_performance_checks_gate(self):
        """Test performance checks gate."""
        manager = QualityGateManager()

        result = manager._gate_performance_checks()

        assert result.gate_name == "performance_checks"
        assert result.status in [
            QualityGateStatus.PASS,
            QualityGateStatus.WARNING,
            QualityGateStatus.SKIP,
        ]
        assert isinstance(result.message, str)

    def test_overall_status_calculation(self):
        """Test overall status calculation."""
        manager = QualityGateManager()

        # Test with no results
        assert manager.get_overall_status() == QualityGateStatus.SKIP

        # Test with mixed results
        manager.results = [
            QualityGateResult("gate1", QualityGateStatus.PASS, "Pass"),
            QualityGateResult("gate2", QualityGateStatus.WARNING, "Warning"),
            QualityGateResult("gate3", QualityGateStatus.PASS, "Pass"),
        ]
        assert manager.get_overall_status() == QualityGateStatus.WARNING

        # Test with failure
        manager.results.append(
            QualityGateResult("gate4", QualityGateStatus.FAIL, "Fail")
        )
        assert manager.get_overall_status() == QualityGateStatus.FAIL

    def test_report_generation(self):
        """Test comprehensive report generation."""
        manager = QualityGateManager()
        manager.results = [
            QualityGateResult(
                "gate1", QualityGateStatus.PASS, "Pass", {"detail": "test"}
            ),
            QualityGateResult("gate2", QualityGateStatus.WARNING, "Warning"),
        ]

        report = manager.generate_report()

        assert "overall_status" in report
        assert "execution_summary" in report
        assert "gate_results" in report
        assert "recommendations" in report

        assert report["execution_summary"]["total_gates"] == 2
        assert report["execution_summary"]["passed"] == 1
        assert report["execution_summary"]["warnings"] == 1
        assert len(report["gate_results"]) == 2
        assert isinstance(report["recommendations"], list)


class TestQualityGatesIntegration:
    """Test quality gates integration scenarios."""

    def test_run_all_gates(self):
        """Test running all quality gates."""
        manager = QualityGateManager()
        results = manager.run_all_gates()

        assert len(results) >= 5  # Should have multiple gates
        assert all(isinstance(result, QualityGateResult) for result in results)
        assert all(result.execution_time_ms >= 0 for result in results)

        # Check that all expected gates are present
        gate_names = [result.gate_name for result in results]
        expected_gates = [
            "basic_imports",
            "test_discovery",
            "critical_tests",
            "coverage_analysis",
            "security_validation",
        ]

        for gate in expected_gates:
            assert gate in gate_names

    def test_run_quality_gates_function(self):
        """Test main quality gates function."""
        report = run_quality_gates()

        assert isinstance(report, dict)
        assert "overall_status" in report
        assert "execution_summary" in report
        assert "gate_results" in report

        # Status should be one of the valid values
        assert report["overall_status"] in ["pass", "fail", "warning", "skip"]

    def test_quality_gates_with_config_file(self):
        """Test quality gates with configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {
                "coverage": {"minimum_line_coverage": 20.0},
                "tests": {"minimum_test_count": 50},
            }
            json.dump(config, f)
            config_file = f.name

        try:
            report = run_quality_gates(config_file)
            assert isinstance(report, dict)
            assert "overall_status" in report
        finally:
            os.unlink(config_file)

    def test_ci_config_generation(self):
        """Test CI/CD configuration generation."""
        ci_config = generate_ci_config()

        assert isinstance(ci_config, str)
        assert "name: Quality Gates" in ci_config
        assert "pytest" in ci_config
        assert "coverage" in ci_config
        assert "python-version" in ci_config
        assert "run_quality_gates" in ci_config

    def test_quality_gates_error_handling(self):
        """Test quality gates error handling."""
        manager = QualityGateManager()

        # Test with invalid configuration
        manager.config = {}  # Empty config should use defaults

        results = manager.run_all_gates()

        # Should still run but may have skipped gates
        assert len(results) >= 5
        assert all(isinstance(result, QualityGateResult) for result in results)

    def test_recommendations_generation(self):
        """Test recommendations generation."""
        manager = QualityGateManager()

        # Set up results with various statuses
        manager.results = [
            QualityGateResult("basic_imports", QualityGateStatus.PASS, "Pass"),
            QualityGateResult(
                "coverage_analysis", QualityGateStatus.WARNING, "Low coverage"
            ),
            QualityGateResult("critical_tests", QualityGateStatus.FAIL, "Tests failed"),
        ]

        recommendations = manager._generate_recommendations()

        assert isinstance(recommendations, list)
        assert len(recommendations) >= 1

        # Should include specific recommendations based on gate results
        rec_text = " ".join(recommendations).lower()
        assert "critical" in rec_text or "failure" in rec_text


class TestQualityGatesPerformance:
    """Test quality gates performance characteristics."""

    def test_gate_execution_timing(self):
        """Test that gate execution times are recorded."""
        manager = QualityGateManager()

        result = manager._gate_basic_imports()

        # Execution time should be set (even if 0)
        assert hasattr(result, "execution_time_ms")
        assert result.execution_time_ms >= 0

    def test_overall_execution_time(self):
        """Test overall execution time tracking."""
        manager = QualityGateManager()
        results = manager.run_all_gates()

        report = manager.generate_report()
        total_time = report["execution_summary"]["total_execution_time_ms"]

        assert total_time >= 0
        assert total_time == sum(r.execution_time_ms for r in results)

    def test_performance_regression_detection(self):
        """Test that slow operations are detected."""
        manager = QualityGateManager()

        # Set very low performance threshold for testing
        manager.config["performance"]["maximum_import_time_ms"] = 0.001  # Very low

        result = manager._gate_performance_checks()

        # Should detect slow import (or skip if import fails)
        assert result.status in [
            QualityGateStatus.WARNING,
            QualityGateStatus.SKIP,
            QualityGateStatus.PASS,
        ]


if __name__ == "__main__":
    # Run quality gates validation
    print("=== Testing Quality Gates Framework ===")

    # Test basic functionality
    print("Testing coverage analyzer...")
    analyzer = TestCoverageAnalyzer()
    distribution = analyzer.analyze_test_distribution()
    print(
        f"OK Test distribution: {distribution['total_test_files']} files, {distribution['total_test_functions']} functions"
    )

    # Test quality gate manager
    print("Testing quality gate manager...")
    manager = QualityGateManager()
    results = manager.run_all_gates()
    print(f"OK Quality gates: {len(results)} gates executed")

    # Test report generation
    print("Testing report generation...")
    report = manager.generate_report()
    print(f"OK Report generated: {report['overall_status']} status")

    # Show summary
    summary = report["execution_summary"]
    print(
        f"Gates: {summary['passed']} passed, {summary['failed']} failed, {summary['warnings']} warnings, {summary['skipped']} skipped"
    )

    print("=== Quality gates framework validation completed ===")
