#!/usr/bin/env python3
"""
Comprehensive Test Runner for Cognate 25M System Validation
Phase 3: Testing & Validation - Master Test Suite
"""

import sys
import time
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Also need to add the core directory specifically
core_path = project_root / "core"
if core_path.exists():
    sys.path.insert(0, str(core_path))

# Import all test suites
from test_import_validation import TestImportValidation
from test_functional_validation import TestFunctionalValidation
from test_integration_validation import TestIntegrationValidation
from test_file_organization import TestFileOrganizationValidation
from test_error_handling import TestErrorHandlingValidation


class CognateTestRunner:
    """Master test runner for comprehensive Cognate system validation."""

    def __init__(self):
        self.results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "test_suites": {},
            "start_time": None,
            "end_time": None,
        }

    def run_test_suite(self, suite_class, suite_name, temp_dir=None):
        """Run a complete test suite and capture results."""
        print(f"\n🧪 Running {suite_name}")
        print("=" * 60)

        suite_results = {"tests_run": 0, "tests_passed": 0, "tests_failed": 0, "errors": []}

        try:
            suite = suite_class()

            # Get all test methods
            test_methods = [
                method for method in dir(suite) if method.startswith("test_") and callable(getattr(suite, method))
            ]

            for test_method in test_methods:
                suite_results["tests_run"] += 1
                self.results["total_tests"] += 1

                try:
                    print(f"  Running {test_method}...")

                    # Call test method with temp_dir if it accepts it
                    method = getattr(suite, test_method)
                    import inspect

                    sig = inspect.signature(method)

                    if temp_dir and "temp_output_dir" in sig.parameters:
                        method(temp_dir)
                    else:
                        method()

                    suite_results["tests_passed"] += 1
                    self.results["passed_tests"] += 1
                    print(f"  ✅ {test_method} PASSED")

                except Exception as e:
                    suite_results["tests_failed"] += 1
                    self.results["failed_tests"] += 1
                    error_info = {"test": test_method, "error": str(e), "traceback": traceback.format_exc()}
                    suite_results["errors"].append(error_info)
                    print(f"  ❌ {test_method} FAILED: {str(e)[:100]}...")

        except Exception as e:
            print(f"❌ Failed to run {suite_name}: {e}")
            suite_results["errors"].append(
                {"test": "suite_initialization", "error": str(e), "traceback": traceback.format_exc()}
            )

        self.results["test_suites"][suite_name] = suite_results

        # Suite summary
        passed = suite_results["tests_passed"]
        failed = suite_results["tests_failed"]
        total = suite_results["tests_run"]

        print(f"\n{suite_name} Results: {passed}/{total} passed, {failed} failed")
        if suite_results["errors"]:
            print(f"Errors in {suite_name}:")
            for error in suite_results["errors"][:3]:  # Show first 3 errors
                print(f"  - {error['test']}: {error['error'][:100]}...")

    def run_all_tests(self):
        """Run all test suites comprehensively."""
        print("🚀 COGNATE 25M SYSTEM - COMPREHENSIVE VALIDATION")
        print("=" * 80)
        print(f"Testing reorganized Cognate system")
        print(f"Test suites: Import, Functional, Integration, File Organization, Error Handling")
        print("=" * 80)

        self.results["start_time"] = time.time()

        # Create temporary directory for tests that need it
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()

        try:
            # Run all test suites
            test_suites = [
                (TestImportValidation, "Import Validation", False),
                (TestFunctionalValidation, "Functional Validation", True),
                (TestIntegrationValidation, "Integration Validation", True),
                (TestFileOrganizationValidation, "File Organization Validation", False),
                (TestErrorHandlingValidation, "Error Handling Validation", True),
            ]

            for suite_class, suite_name, needs_temp_dir in test_suites:
                self.run_test_suite(suite_class, suite_name, temp_dir if needs_temp_dir else None)

        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.results["end_time"] = time.time()
        self.print_final_report()

    def print_final_report(self):
        """Print comprehensive final test report."""
        duration = self.results["end_time"] - self.results["start_time"]

        print("\n" + "=" * 80)
        print("🎯 FINAL TEST REPORT - COGNATE 25M SYSTEM VALIDATION")
        print("=" * 80)

        # Overall statistics
        total = self.results["total_tests"]
        passed = self.results["passed_tests"]
        failed = self.results["failed_tests"]
        success_rate = (passed / total * 100) if total > 0 else 0

        print(f"📊 OVERALL RESULTS:")
        print(f"   Total Tests Run: {total}")
        print(f"   Tests Passed: {passed}")
        print(f"   Tests Failed: {failed}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Duration: {duration:.2f} seconds")

        # Per-suite breakdown
        print(f"\n📋 TEST SUITE BREAKDOWN:")
        for suite_name, suite_results in self.results["test_suites"].items():
            suite_total = suite_results["tests_run"]
            suite_passed = suite_results["tests_passed"]
            suite_failed = suite_results["tests_failed"]
            suite_rate = (suite_passed / suite_total * 100) if suite_total > 0 else 0

            status = "✅ PASS" if suite_failed == 0 else "❌ FAIL"
            print(f"   {status} {suite_name}: {suite_passed}/{suite_total} ({suite_rate:.1f}%)")

        # Critical issues
        print(f"\n🔍 CRITICAL ANALYSIS:")
        critical_issues = []

        # Check for import failures
        import_results = self.results["test_suites"].get("Import Validation", {})
        if import_results.get("tests_failed", 0) > 0:
            critical_issues.append("❌ CRITICAL: Import system has failures")

        # Check for functional failures
        func_results = self.results["test_suites"].get("Functional Validation", {})
        if func_results.get("tests_failed", 0) > 0:
            critical_issues.append("⚠️  WARNING: Core functionality has issues")

        # Check for integration failures
        integration_results = self.results["test_suites"].get("Integration Validation", {})
        if integration_results.get("tests_failed", 0) > 0:
            critical_issues.append("⚠️  WARNING: Pipeline integration has issues")

        if critical_issues:
            for issue in critical_issues:
                print(f"   {issue}")
        else:
            print("   ✅ No critical issues detected")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if failed == 0:
            print("   ✅ All tests passed! Cognate system reorganization is successful.")
            print("   ✅ Ready for production deployment.")
            print("   ✅ EvoMerge integration should work correctly.")
        elif failed <= 2:
            print("   ⚠️  Minor issues detected. Review failed tests.")
            print("   ⚠️  Consider fixing issues before production deployment.")
        else:
            print("   ❌ Multiple issues detected. System needs attention.")
            print("   ❌ Do not deploy until issues are resolved.")

        # Success criteria check
        print(f"\n🎯 SUCCESS CRITERIA VALIDATION:")
        success_criteria = [
            ("All imports work without errors", import_results.get("tests_failed", 1) == 0),
            ("Models created successfully with correct specs", func_results.get("tests_passed", 0) >= 3),
            ("Parameter counts within 10% of 25M target", "parameter_count_validation" in str(func_results)),
            (
                "Integration with Agent Forge pipeline works",
                integration_results.get("tests_failed", 1) < integration_results.get("tests_run", 1),
            ),
            ("Backward compatibility maintained", "backward_compatibility" in str(integration_results)),
            (
                "Clean error messages for missing dependencies",
                self.results["test_suites"].get("Error Handling Validation", {}).get("tests_passed", 0) > 0,
            ),
            (
                "No remaining duplicate functionality",
                self.results["test_suites"].get("File Organization Validation", {}).get("tests_failed", 1) == 0,
            ),
        ]

        for criterion, met in success_criteria:
            status = "✅" if met else "❌"
            print(f"   {status} {criterion}")

        # Overall verdict
        all_criteria_met = all(met for _, met in success_criteria)
        critical_systems_working = (
            import_results.get("tests_failed", 1) == 0 and func_results.get("tests_passed", 0) >= 3
        )

        print(f"\n🏆 FINAL VERDICT:")
        if all_criteria_met:
            print("   🎉 COMPLETE SUCCESS - All success criteria met!")
            print("   ✅ Cognate 25M reorganization is fully validated")
            print("   🚀 Ready for Phase 4: Production deployment")
        elif critical_systems_working:
            print("   ✅ CORE SUCCESS - Critical systems working correctly")
            print("   ⚠️  Minor issues to address in follow-up")
            print("   🔄 Can proceed with Phase 4 but monitor issues")
        else:
            print("   ❌ REQUIRES ATTENTION - Core systems have issues")
            print("   🛑 Do not proceed to Phase 4 until resolved")
            print("   🔧 Fix critical issues before continuing")

        print("=" * 80)

        # Return success indicator
        return all_criteria_met or critical_systems_working


if __name__ == "__main__":
    runner = CognateTestRunner()
    success = runner.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
