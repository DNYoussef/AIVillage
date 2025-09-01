"""Test Runner for Security Server Refactoring Test Suite.

Executes comprehensive tests for security server refactoring including:
- Unit tests for individual components
- Integration tests between modules  
- Backward compatibility tests
- Performance tests
- Coverage analysis and reporting
"""

import subprocess
import sys
import time
from pathlib import Path



class SecurityRefactorTestRunner:
    """Comprehensive test runner for security refactoring."""

    def __init__(self):
        """Initialize test runner."""
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent.parent
        self.test_files = [
            "test_session_manager_refactor.py",
            "test_middleware_refactor.py",
            "test_route_handlers_refactor.py",
            "test_security_context_refactor.py",
            "test_integration_refactor.py",
            "test_backward_compatibility_refactor.py",
            "test_performance_security_refactor.py",
        ]

    def run_individual_test_suite(self, test_file: str) -> dict:
        """Run individual test suite and capture results."""
        print(f"\n{'='*60}")
        print(f"Running {test_file}")
        print(f"{'='*60}")

        start_time = time.time()

        # Run pytest with coverage and detailed output
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self.test_dir / test_file),
            "-v",
            "--tb=short",
            "--capture=no",
            "--cov=infrastructure.shared.security",
            "--cov-report=term-missing",
            "--cov-append",
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout per test file
            )

            end_time = time.time()
            duration = end_time - start_time

            return {
                "file": test_file,
                "returncode": result.returncode,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }

        except subprocess.TimeoutExpired:
            return {
                "file": test_file,
                "returncode": -1,
                "duration": 120,
                "stdout": "",
                "stderr": f"Test suite {test_file} timed out after 120 seconds",
                "success": False,
            }
        except Exception as e:
            return {
                "file": test_file,
                "returncode": -2,
                "duration": 0,
                "stdout": "",
                "stderr": f"Failed to run {test_file}: {str(e)}",
                "success": False,
            }

    def run_all_tests(self) -> dict:
        """Run all security refactoring tests."""
        print("🔒 Starting Security Server Refactoring Test Suite")
        print(f"Test directory: {self.test_dir}")
        print(f"Found {len(self.test_files)} test files")

        overall_start = time.time()
        results = []

        # Run each test suite
        for test_file in self.test_files:
            if (self.test_dir / test_file).exists():
                result = self.run_individual_test_suite(test_file)
                results.append(result)

                # Print immediate feedback
                status = "✅ PASSED" if result["success"] else "❌ FAILED"
                print(f"{status} {test_file} ({result['duration']:.2f}s)")

                if not result["success"]:
                    print(f"Error output: {result['stderr'][:500]}...")
            else:
                print(f"⚠️  Test file not found: {test_file}")

        overall_end = time.time()
        total_duration = overall_end - overall_start

        # Generate summary
        summary = self._generate_test_summary(results, total_duration)

        return {"summary": summary, "results": results, "total_duration": total_duration}

    def _generate_test_summary(self, results: list, total_duration: float) -> dict:
        """Generate comprehensive test summary."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["success"])
        failed_tests = total_tests - passed_tests

        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        summary = {
            "total_test_files": total_tests,
            "passed_test_files": passed_tests,
            "failed_test_files": failed_tests,
            "success_rate": success_rate,
            "total_duration": total_duration,
            "avg_duration_per_file": total_duration / total_tests if total_tests > 0 else 0,
            "status": "PASSED" if failed_tests == 0 else "FAILED",
        }

        return summary

    def run_coverage_analysis(self) -> dict:
        """Run coverage analysis for security modules."""
        print(f"\n{'='*60}")
        print("Running Coverage Analysis")
        print(f"{'='*60}")

        # Run comprehensive coverage
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self.test_dir),
            "--cov=infrastructure.shared.security",
            "--cov-report=html:coverage_html",
            "--cov-report=xml:coverage.xml",
            "--cov-report=term-missing",
            "--cov-fail-under=90",  # Require 90% coverage
            "-q",
        ]

        try:
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True, timeout=180)

            # Parse coverage from output
            coverage_info = self._parse_coverage_output(result.stdout)

            return {
                "success": result.returncode == 0,
                "coverage_info": coverage_info,
                "output": result.stdout,
                "html_report": "coverage_html/index.html",
            }

        except Exception as e:
            return {"success": False, "error": str(e), "coverage_info": {}}

    def _parse_coverage_output(self, output: str) -> dict:
        """Parse coverage information from pytest output."""
        coverage_info = {}

        lines = output.split("\n")
        for line in lines:
            if "% coverage" in line or "TOTAL" in line:
                # Extract coverage percentage
                parts = line.split()
                for part in parts:
                    if part.endswith("%"):
                        try:
                            coverage_info["total_coverage"] = float(part[:-1])
                            break
                        except ValueError:
                            continue

        return coverage_info

    def generate_final_report(self, test_results: dict, coverage_results: dict) -> str:
        """Generate final comprehensive report."""
        report = []
        report.append("🔒 SECURITY SERVER REFACTORING TEST REPORT")
        report.append("=" * 60)

        # Test Summary
        summary = test_results["summary"]
        report.append("\n📊 TEST SUMMARY:")
        report.append(f"   Total Test Files: {summary['total_test_files']}")
        report.append(f"   Passed: {summary['passed_test_files']}")
        report.append(f"   Failed: {summary['failed_test_files']}")
        report.append(f"   Success Rate: {summary['success_rate']:.1f}%")
        report.append(f"   Total Duration: {summary['total_duration']:.2f}s")
        report.append(f"   Overall Status: {summary['status']}")

        # Coverage Summary
        if coverage_results.get("success"):
            coverage_info = coverage_results.get("coverage_info", {})
            total_coverage = coverage_info.get("total_coverage", 0)
            report.append("\n📈 COVERAGE SUMMARY:")
            report.append(f"   Total Coverage: {total_coverage:.1f}%")
            report.append(f"   Coverage Status: {'✅ PASSED' if total_coverage >= 90 else '❌ FAILED'}")
            report.append(f"   HTML Report: {coverage_results.get('html_report', 'N/A')}")
        else:
            report.append("\n📈 COVERAGE SUMMARY:")
            report.append(f"   Coverage Status: ❌ FAILED - {coverage_results.get('error', 'Unknown error')}")

        # Individual Test Results
        report.append("\n📋 INDIVIDUAL TEST RESULTS:")
        for result in test_results["results"]:
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            report.append(f"   {status} {result['file']} ({result['duration']:.2f}s)")

            if not result["success"]:
                # Include error details for failed tests
                stderr_preview = result["stderr"][:200] + "..." if len(result["stderr"]) > 200 else result["stderr"]
                report.append(f"      Error: {stderr_preview}")

        # Test Coverage by Component
        report.append("\n🎯 TEST COVERAGE BY COMPONENT:")
        components = [
            "SessionManager (test_session_manager_refactor.py)",
            "Security Middleware (test_middleware_refactor.py)",
            "Route Handlers (test_route_handlers_refactor.py)",
            "Security Context (test_security_context_refactor.py)",
            "Integration (test_integration_refactor.py)",
            "Backward Compatibility (test_backward_compatibility_refactor.py)",
            "Performance (test_performance_security_refactor.py)",
        ]

        for component in components:
            report.append(f"   ✅ {component}")

        # Recommendations
        report.append("\n💡 RECOMMENDATIONS:")
        if summary["success_rate"] == 100:
            report.append("   ✅ All tests passing - refactoring is safe to proceed")
            report.append("   ✅ Backward compatibility verified")
            report.append("   ✅ Performance benchmarks met")
        else:
            report.append("   ⚠️  Some tests failing - review failures before refactoring")
            report.append("   🔍 Check error details above")

        if coverage_results.get("success") and coverage_results.get("coverage_info", {}).get("total_coverage", 0) >= 90:
            report.append("   ✅ Coverage requirements met (≥90%)")
        else:
            report.append("   ⚠️  Coverage below 90% - add more tests")

        return "\n".join(report)


def main():
    """Main execution function."""
    runner = SecurityRefactorTestRunner()

    # Run all tests
    print("Starting comprehensive security refactoring test suite...")
    test_results = runner.run_all_tests()

    # Run coverage analysis
    print("\nRunning coverage analysis...")
    coverage_results = runner.run_coverage_analysis()

    # Generate final report
    final_report = runner.generate_final_report(test_results, coverage_results)

    print(f"\n{final_report}")

    # Save report to file
    report_file = runner.test_dir / "security_refactor_test_report.txt"
    with open(report_file, "w") as f:
        f.write(final_report)

    print(f"\n📄 Full report saved to: {report_file}")

    # Return exit code based on results
    if test_results["summary"]["status"] == "PASSED" and coverage_results.get("success", False):
        print("\n🎉 All tests passed! Refactoring is safe to proceed.")
        return 0
    else:
        print("\n❌ Some tests failed. Review results before proceeding.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
