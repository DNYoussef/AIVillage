#!/usr/bin/env python3
"""
Automated Test Runner for Phase 6 Baking System
===============================================

Comprehensive test runner that executes all Phase 6 baking tests:
- Unit tests for all components
- Integration tests for cross-phase compatibility
- Performance validation tests
- Quality preservation tests
- Inference capability tests
- System working verification

Features:
- Parallel test execution
- Detailed reporting
- Performance metrics collection
- Quality gate validation
- HTML report generation
"""

import unittest
import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import subprocess
import platform

# Setup paths
TESTS_DIR = Path(__file__).parent
REPO_ROOT = TESTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"

# Add source to Python path
sys.path.insert(0, str(SRC_DIR))


class TestResult:
    """Container for individual test results"""
    def __init__(self, test_name: str, test_class: str, module: str):
        self.test_name = test_name
        self.test_class = test_class
        self.module = module
        self.status = "pending"
        self.duration = 0.0
        self.error_message = ""
        self.traceback = ""
        self.start_time = 0.0
        self.end_time = 0.0

    def mark_started(self):
        """Mark test as started"""
        self.start_time = time.time()
        self.status = "running"

    def mark_completed(self, success: bool, error_message: str = "", traceback_str: str = ""):
        """Mark test as completed"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = "passed" if success else "failed"
        self.error_message = error_message
        self.traceback = traceback_str


class TestSuite:
    """Container for a collection of related tests"""
    def __init__(self, name: str, description: str, test_modules: List[str]):
        self.name = name
        self.description = description
        self.test_modules = test_modules
        self.results: List[TestResult] = []
        self.total_duration = 0.0
        self.passed_count = 0
        self.failed_count = 0
        self.skipped_count = 0


class Phase6TestRunner:
    """Comprehensive test runner for Phase 6 baking system"""

    def __init__(self, parallel: bool = True, max_workers: int = 4, verbose: bool = True):
        self.parallel = parallel
        self.max_workers = max_workers
        self.verbose = verbose

        # Setup logging
        self.logger = self._setup_logging()

        # Test suites
        self.test_suites = self._define_test_suites()

        # Results storage
        self.all_results: List[TestResult] = []
        self.suite_results: Dict[str, TestSuite] = {}

        # System information
        self.system_info = self._collect_system_info()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("Phase6TestRunner")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _define_test_suites(self) -> Dict[str, TestSuite]:
        """Define all test suites"""
        return {
            "unit": TestSuite(
                name="Unit Tests",
                description="Unit tests for individual Phase 6 components",
                test_modules=[
                    "tests.unit.test_phase6_baking_architecture"
                ]
            ),
            "integration": TestSuite(
                name="Integration Tests",
                description="Integration tests for Phase 5/7 compatibility",
                test_modules=[
                    "tests.integration.test_phase6_integration"
                ]
            ),
            "performance": TestSuite(
                name="Performance Tests",
                description="Performance validation and 2-5x speedup tests",
                test_modules=[
                    "tests.performance.test_phase6_performance"
                ]
            ),
            "quality": TestSuite(
                name="Quality Tests",
                description="Quality preservation and theater detection tests",
                test_modules=[
                    "tests.quality.test_phase6_quality"
                ]
            ),
            "inference": TestSuite(
                name="Inference Tests",
                description="Inference capability and real-time performance tests",
                test_modules=[
                    "tests.inference.test_phase6_inference"
                ]
            )
        }

    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for the test report"""
        import torch

        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["gpu_name"] = torch.cuda.get_device_name(0)

        return info

    def discover_tests_in_module(self, module_path: str) -> List[TestResult]:
        """Discover all tests in a given module"""
        tests = []

        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(
                module_path.replace(".", "_"),
                TESTS_DIR / f"{module_path.replace('.', '/')}.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Discover test classes
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                    issubclass(attr, unittest.TestCase) and
                    attr != unittest.TestCase):

                    # Discover test methods
                    for method_name in dir(attr):
                        if method_name.startswith("test_"):
                            test_result = TestResult(
                                test_name=method_name,
                                test_class=attr_name,
                                module=module_path
                            )
                            tests.append(test_result)

        except Exception as e:
            self.logger.error(f"Failed to discover tests in {module_path}: {str(e)}")

        return tests

    def run_single_test(self, test_result: TestResult) -> TestResult:
        """Run a single test and capture results"""
        test_result.mark_started()

        try:
            # Create test suite for this specific test
            module_path = TESTS_DIR / f"{test_result.module.replace('.', '/')}.py"
            spec = importlib.util.spec_from_file_location(
                test_result.module.replace(".", "_"),
                module_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get the test class and create instance
            test_class = getattr(module, test_result.test_class)
            test_instance = test_class()

            # Run setup if it exists
            if hasattr(test_instance, 'setUp'):
                test_instance.setUp()

            # Run the actual test
            test_method = getattr(test_instance, test_result.test_name)
            test_method()

            # Run teardown if it exists
            if hasattr(test_instance, 'tearDown'):
                test_instance.tearDown()

            test_result.mark_completed(True)

        except unittest.SkipTest as e:
            test_result.mark_completed(True)
            test_result.status = "skipped"
            test_result.error_message = str(e)

        except Exception as e:
            error_message = str(e)
            traceback_str = traceback.format_exc()
            test_result.mark_completed(False, error_message, traceback_str)

        return test_result

    def run_test_suite(self, suite_name: str) -> TestSuite:
        """Run all tests in a test suite"""
        suite = self.test_suites[suite_name]
        self.logger.info(f"Running test suite: {suite.name}")

        # Discover all tests
        all_tests = []
        for module_path in suite.test_modules:
            module_tests = self.discover_tests_in_module(module_path)
            all_tests.extend(module_tests)

        suite.results = all_tests

        if not all_tests:
            self.logger.warning(f"No tests found in suite: {suite.name}")
            return suite

        start_time = time.time()

        if self.parallel and len(all_tests) > 1:
            # Run tests in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_test = {
                    executor.submit(self.run_single_test, test): test
                    for test in all_tests
                }

                for future in as_completed(future_to_test):
                    test = future_to_test[future]
                    try:
                        result = future.result()
                        if self.verbose:
                            status_symbol = "✓" if result.status == "passed" else "✗" if result.status == "failed" else "⊝"
                            print(f"  {status_symbol} {result.test_class}.{result.test_name} ({result.duration:.2f}s)")
                    except Exception as e:
                        self.logger.error(f"Test {test.test_name} raised exception: {e}")
        else:
            # Run tests sequentially
            for test in all_tests:
                result = self.run_single_test(test)
                if self.verbose:
                    status_symbol = "✓" if result.status == "passed" else "✗" if result.status == "failed" else "⊝"
                    print(f"  {status_symbol} {result.test_class}.{result.test_name} ({result.duration:.2f}s)")

        suite.total_duration = time.time() - start_time

        # Calculate statistics
        suite.passed_count = sum(1 for r in suite.results if r.status == "passed")
        suite.failed_count = sum(1 for r in suite.results if r.status == "failed")
        suite.skipped_count = sum(1 for r in suite.results if r.status == "skipped")

        self.logger.info(f"Suite {suite.name} completed: "
                        f"{suite.passed_count} passed, "
                        f"{suite.failed_count} failed, "
                        f"{suite.skipped_count} skipped "
                        f"({suite.total_duration:.2f}s)")

        return suite

    def run_all_tests(self, selected_suites: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run all test suites"""
        self.logger.info("Starting Phase 6 baking system tests")
        self.logger.info(f"System: {self.system_info['platform']}")
        self.logger.info(f"PyTorch: {self.system_info['pytorch_version']}")
        self.logger.info(f"CUDA: {self.system_info['cuda_available']}")

        if selected_suites is None:
            selected_suites = list(self.test_suites.keys())

        overall_start_time = time.time()

        # Run each test suite
        for suite_name in selected_suites:
            if suite_name not in self.test_suites:
                self.logger.warning(f"Unknown test suite: {suite_name}")
                continue

            print(f"\n{self.test_suites[suite_name].name}")
            print("=" * 60)

            suite_result = self.run_test_suite(suite_name)
            self.suite_results[suite_name] = suite_result
            self.all_results.extend(suite_result.results)

        overall_duration = time.time() - overall_start_time

        # Generate summary
        total_passed = sum(suite.passed_count for suite in self.suite_results.values())
        total_failed = sum(suite.failed_count for suite in self.suite_results.values())
        total_skipped = sum(suite.skipped_count for suite in self.suite_results.values())
        total_tests = total_passed + total_failed + total_skipped

        summary = {
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "skipped": total_skipped,
            "duration": overall_duration,
            "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
            "suites": self.suite_results,
            "system_info": self.system_info
        }

        self._print_summary(summary)
        return summary

    def _print_summary(self, summary: Dict[str, Any]):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("PHASE 6 BAKING SYSTEM TEST SUMMARY")
        print("=" * 80)

        print(f"Total Tests:    {summary['total_tests']}")
        print(f"Passed:         {summary['passed']} ({'✓' if summary['passed'] > 0 else ''})")
        print(f"Failed:         {summary['failed']} ({'✗' if summary['failed'] > 0 else ''})")
        print(f"Skipped:        {summary['skipped']} ({'⊝' if summary['skipped'] > 0 else ''})")
        print(f"Success Rate:   {summary['success_rate']:.1f}%")
        print(f"Duration:       {summary['duration']:.2f} seconds")

        print("\nSuite Breakdown:")
        for suite_name, suite in summary['suites'].items():
            status = "PASS" if suite.failed_count == 0 else "FAIL"
            print(f"  {suite.name:20} {status:4} ({suite.passed_count:2}/{suite.passed_count + suite.failed_count:2}) [{suite.total_duration:6.2f}s]")

        if summary['failed'] > 0:
            print("\nFailed Tests:")
            for suite in summary['suites'].values():
                failed_tests = [r for r in suite.results if r.status == "failed"]
                for test in failed_tests:
                    print(f"  ✗ {test.module}.{test.test_class}.{test.test_name}")
                    print(f"    Error: {test.error_message}")

        print("=" * 80)

    def generate_json_report(self, output_path: Path) -> Path:
        """Generate JSON test report"""
        report_data = {
            "timestamp": self.system_info["timestamp"],
            "system_info": self.system_info,
            "summary": {
                "total_tests": len(self.all_results),
                "passed": sum(1 for r in self.all_results if r.status == "passed"),
                "failed": sum(1 for r in self.all_results if r.status == "failed"),
                "skipped": sum(1 for r in self.all_results if r.status == "skipped"),
                "duration": sum(suite.total_duration for suite in self.suite_results.values())
            },
            "suites": {},
            "tests": []
        }

        # Add suite information
        for suite_name, suite in self.suite_results.items():
            report_data["suites"][suite_name] = {
                "name": suite.name,
                "description": suite.description,
                "passed": suite.passed_count,
                "failed": suite.failed_count,
                "skipped": suite.skipped_count,
                "duration": suite.total_duration
            }

        # Add individual test results
        for result in self.all_results:
            report_data["tests"].append({
                "name": result.test_name,
                "class": result.test_class,
                "module": result.module,
                "status": result.status,
                "duration": result.duration,
                "error_message": result.error_message,
                "traceback": result.traceback
            })

        # Write report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        return output_path

    def validate_quality_gates(self) -> Dict[str, bool]:
        """Validate quality gates for Phase 6 baking system"""
        quality_gates = {
            "unit_tests_pass": True,
            "integration_tests_pass": True,
            "performance_requirements_met": True,
            "quality_preservation_validated": True,
            "inference_capability_verified": True,
            "overall_success_rate": True
        }

        # Check unit tests
        if "unit" in self.suite_results:
            quality_gates["unit_tests_pass"] = self.suite_results["unit"].failed_count == 0

        # Check integration tests
        if "integration" in self.suite_results:
            quality_gates["integration_tests_pass"] = self.suite_results["integration"].failed_count == 0

        # Check performance tests
        if "performance" in self.suite_results:
            quality_gates["performance_requirements_met"] = self.suite_results["performance"].failed_count == 0

        # Check quality tests
        if "quality" in self.suite_results:
            quality_gates["quality_preservation_validated"] = self.suite_results["quality"].failed_count == 0

        # Check inference tests
        if "inference" in self.suite_results:
            quality_gates["inference_capability_verified"] = self.suite_results["inference"].failed_count == 0

        # Check overall success rate
        total_tests = len(self.all_results)
        passed_tests = sum(1 for r in self.all_results if r.status == "passed")
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        quality_gates["overall_success_rate"] = success_rate >= 95.0  # 95% success rate required

        return quality_gates


def main():
    """Main entry point for test runner"""
    parser = argparse.ArgumentParser(description="Phase 6 Baking System Test Runner")

    parser.add_argument(
        "--suites", "-s",
        nargs="+",
        choices=["unit", "integration", "performance", "quality", "inference"],
        help="Test suites to run (default: all)"
    )

    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        default=True,
        help="Run tests in parallel (default: True)"
    )

    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run tests sequentially"
    )

    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=TESTS_DIR / "results" / f"test_report_{int(time.time())}.json",
        help="Output path for JSON report"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Verbose output"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet output"
    )

    args = parser.parse_args()

    # Configure parallel execution
    parallel = args.parallel and not args.sequential
    verbose = args.verbose and not args.quiet

    # Create test runner
    runner = Phase6TestRunner(
        parallel=parallel,
        max_workers=args.workers,
        verbose=verbose
    )

    # Run tests
    try:
        summary = runner.run_all_tests(selected_suites=args.suites)

        # Generate report
        report_path = runner.generate_json_report(args.output)
        print(f"\nDetailed report saved to: {report_path}")

        # Validate quality gates
        quality_gates = runner.validate_quality_gates()
        print("\nQuality Gates:")
        for gate, passed in quality_gates.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {gate:35} {status}")

        # Exit with appropriate code
        all_gates_pass = all(quality_gates.values())
        exit_code = 0 if all_gates_pass else 1

        if not all_gates_pass:
            print("\n❌ Some quality gates failed. Phase 6 baking system requires attention.")
        else:
            print("\n✅ All quality gates passed. Phase 6 baking system is working properly.")

        return exit_code

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        return 130

    except Exception as e:
        print(f"\n\nTest runner failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())