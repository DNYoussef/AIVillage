#!/usr/bin/env python3
"""
Quiet-STaR Test Suite Runner

Comprehensive test execution script with advanced reporting and analysis.
Provides multiple execution modes and detailed quality metrics.
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import xml.etree.ElementTree as ET


class QuietSTaRTestRunner:
    """Advanced test runner for Quiet-STaR test suite."""

    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.results = {}
        self.start_time = None
        self.end_time = None

    def run_tests(self,
                  mode: str = "comprehensive",
                  parallel: bool = True,
                  coverage: bool = True,
                  report_format: str = "both",
                  fail_fast: bool = False,
                  verbose: bool = True) -> Dict[str, Any]:
        """
        Run tests with specified configuration.

        Args:
            mode: Test execution mode (quick, comprehensive, performance, etc.)
            parallel: Enable parallel test execution
            coverage: Enable coverage reporting
            report_format: Output format (html, xml, json, both)
            fail_fast: Stop on first failure
            verbose: Verbose output

        Returns:
            Test execution results
        """
        self.start_time = time.time()

        print("=" * 80)
        print("QUIET-STAR TEST SUITE EXECUTION")
        print("=" * 80)
        print(f"Mode: {mode}")
        print(f"Parallel: {parallel}")
        print(f"Coverage: {coverage}")
        print(f"Report Format: {report_format}")
        print("=" * 80)

        # Build pytest command
        cmd = self._build_pytest_command(
            mode, parallel, coverage, report_format, fail_fast, verbose
        )

        print(f"Executing: {' '.join(cmd)}")
        print("-" * 80)

        # Execute tests
        try:
            result = subprocess.run(
                cmd,
                cwd=self.test_dir,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )

            self.end_time = time.time()
            duration = self.end_time - self.start_time

            # Process results
            self.results = self._process_results(result, duration)

            # Print summary
            self._print_summary()

            return self.results

        except subprocess.TimeoutExpired:
            print("❌ Tests timed out after 30 minutes")
            return {"status": "timeout", "duration": 1800}

        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            return {"status": "error", "error": str(e)}

    def _build_pytest_command(self,
                              mode: str,
                              parallel: bool,
                              coverage: bool,
                              report_format: str,
                              fail_fast: bool,
                              verbose: bool) -> List[str]:
        """Build pytest command based on configuration."""
        cmd = [sys.executable, "-m", "pytest"]

        # Add test file
        cmd.append("test_quietstar.py")

        # Mode-specific arguments
        if mode == "quick":
            cmd.extend(["-m", "not slow", "--tb=short"])
        elif mode == "unit":
            cmd.extend(["-m", "unit"])
        elif mode == "integration":
            cmd.extend(["-m", "integration"])
        elif mode == "performance":
            cmd.extend(["-m", "performance", "--benchmark-only"])
        elif mode == "property":
            cmd.extend(["-m", "property"])
        elif mode == "contract":
            cmd.extend(["-m", "contract"])
        elif mode == "comprehensive":
            cmd.extend(["--tb=short"])
        elif mode == "debug":
            cmd.extend(["-v", "--tb=long", "-s"])

        # Parallel execution
        if parallel and mode != "debug":
            cmd.extend(["-n", "auto"])

        # Fail fast
        if fail_fast:
            cmd.extend(["-x"])

        # Verbose output
        if verbose:
            cmd.extend(["-v"])

        # Coverage configuration
        if coverage:
            cmd.extend([
                "--cov=quiet_star",
                "--cov-config=.coveragerc",
                "--cov-fail-under=85"
            ])

            # Coverage report formats
            if report_format in ["html", "both"]:
                cmd.extend(["--cov-report=html:htmlcov"])
            if report_format in ["xml", "both"]:
                cmd.extend(["--cov-report=xml:coverage.xml"])

            cmd.extend(["--cov-report=term-missing"])

        # Additional reporting
        if report_format in ["html", "both"]:
            cmd.extend(["--html=report.html", "--self-contained-html"])
        if report_format in ["xml", "both"]:
            cmd.extend(["--junit-xml=test-results.xml"])

        # Performance and timing
        cmd.extend(["--durations=10"])

        return cmd

    def _process_results(self, result: subprocess.CompletedProcess, duration: float) -> Dict[str, Any]:
        """Process test execution results."""
        # Parse output for test statistics
        output_lines = result.stdout.split('\n')

        # Extract test counts
        test_counts = self._extract_test_counts(output_lines)

        # Extract coverage information
        coverage_info = self._extract_coverage_info(output_lines)

        # Parse XML results if available
        xml_results = self._parse_xml_results()

        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "exit_code": result.returncode,
            "duration": duration,
            "test_counts": test_counts,
            "coverage": coverage_info,
            "xml_results": xml_results,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    def _extract_test_counts(self, output_lines: List[str]) -> Dict[str, int]:
        """Extract test count information from pytest output."""
        counts = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0
        }

        for line in output_lines:
            if "passed" in line and "failed" in line:
                # Parse line like "5 failed, 23 passed, 2 skipped in 45.2s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed" and i > 0:
                        counts["passed"] = int(parts[i-1])
                    elif part == "failed" and i > 0:
                        counts["failed"] = int(parts[i-1])
                    elif part == "skipped" and i > 0:
                        counts["skipped"] = int(parts[i-1])
                    elif part == "error" and i > 0:
                        counts["errors"] = int(parts[i-1])

        counts["total"] = sum([counts["passed"], counts["failed"],
                              counts["skipped"], counts["errors"]])
        return counts

    def _extract_coverage_info(self, output_lines: List[str]) -> Dict[str, Any]:
        """Extract coverage information from pytest output."""
        coverage = {
            "total_coverage": 0.0,
            "missing_lines": 0,
            "covered_lines": 0
        }

        for line in output_lines:
            if "%" in line and "TOTAL" in line:
                # Parse total coverage line
                parts = line.split()
                for part in parts:
                    if "%" in part:
                        coverage["total_coverage"] = float(part.rstrip('%'))

        return coverage

    def _parse_xml_results(self) -> Optional[Dict[str, Any]]:
        """Parse XML test results if available."""
        xml_path = self.test_dir / "test-results.xml"
        if not xml_path.exists():
            return None

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            return {
                "tests": int(root.get("tests", 0)),
                "failures": int(root.get("failures", 0)),
                "errors": int(root.get("errors", 0)),
                "skipped": int(root.get("skipped", 0)),
                "time": float(root.get("time", 0.0))
            }
        except Exception as e:
            print(f"Warning: Could not parse XML results: {e}")
            return None

    def _print_summary(self):
        """Print execution summary."""
        print("\n" + "=" * 80)
        print("TEST EXECUTION SUMMARY")
        print("=" * 80)

        status = self.results.get("status", "unknown")
        duration = self.results.get("duration", 0)
        test_counts = self.results.get("test_counts", {})
        coverage = self.results.get("coverage", {})

        # Overall status
        status_emoji = "✅" if status == "passed" else "❌"
        print(f"Status: {status_emoji} {status.upper()}")
        print(f"Duration: {duration:.2f} seconds")

        # Test counts
        total = test_counts.get("total", 0)
        passed = test_counts.get("passed", 0)
        failed = test_counts.get("failed", 0)
        skipped = test_counts.get("skipped", 0)

        print(f"\nTest Results:")
        print(f"  Total: {total}")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")
        print(f"  Skipped: {skipped}")

        if total > 0:
            pass_rate = (passed / total) * 100
            print(f"  Pass Rate: {pass_rate:.1f}%")

        # Coverage
        total_coverage = coverage.get("total_coverage", 0)
        print(f"\nCoverage: {total_coverage:.1f}%")

        # Quality assessment
        print(f"\nQuality Assessment:")
        if status == "passed" and total_coverage >= 85:
            print("  ✅ All quality gates passed")
        else:
            print("  ❌ Quality gates failed")
            if status != "passed":
                print("    - Tests failed")
            if total_coverage < 85:
                print(f"    - Coverage below 85% ({total_coverage:.1f}%)")

        print("=" * 80)

    def generate_report(self, output_file: str = "test_report.json"):
        """Generate detailed JSON report."""
        report = {
            "timestamp": time.time(),
            "test_suite": "Quiet-STaR Comprehensive Test Suite",
            "execution": self.results,
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "working_directory": str(self.test_dir)
            }
        }

        output_path = self.test_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📊 Detailed report saved to: {output_path}")
        return output_path


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Quiet-STaR Test Suite Runner")

    parser.add_argument(
        "--mode",
        choices=["quick", "comprehensive", "unit", "integration", "performance",
                "property", "contract", "debug"],
        default="comprehensive",
        help="Test execution mode"
    )

    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel execution"
    )

    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting"
    )

    parser.add_argument(
        "--report-format",
        choices=["html", "xml", "json", "both"],
        default="both",
        help="Report output format"
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )

    parser.add_argument(
        "--report-file",
        default="test_report.json",
        help="Output file for detailed report"
    )

    args = parser.parse_args()

    # Determine test directory
    test_dir = Path(__file__).parent

    # Create runner and execute tests
    runner = QuietSTaRTestRunner(test_dir)

    results = runner.run_tests(
        mode=args.mode,
        parallel=not args.no_parallel,
        coverage=not args.no_coverage,
        report_format=args.report_format,
        fail_fast=args.fail_fast,
        verbose=not args.quiet
    )

    # Generate detailed report
    runner.generate_report(args.report_file)

    # Exit with appropriate code
    exit_code = results.get("exit_code", 1)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()