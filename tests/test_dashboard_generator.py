#!/usr/bin/env python3
"""Generate comprehensive test dashboard for AIVillage project."""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


class TestDashboard:
    def __init__(self):
        self.results = {
            "generated": datetime.now().isoformat(),
            "summary": {},
            "test_suites": {},
            "performance": {},
            "coverage": {},
            "issues": []
        }

    def run_test_suite(self, name: str, command: list[str], timeout: int = 60) -> dict:
        """Run a test suite and capture results."""
        print(f"Running {name}...")
        start_time = time.time()

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd="."
            )

            duration = time.time() - start_time

            return {
                "name": name,
                "command": " ".join(command),
                "duration": duration,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "status": "PASSED" if result.returncode == 0 else "FAILED"
            }

        except subprocess.TimeoutExpired:
            return {
                "name": name,
                "command": " ".join(command),
                "duration": timeout,
                "return_code": -1,
                "stdout": "",
                "stderr": "Test timed out",
                "status": "TIMEOUT"
            }
        except Exception as e:
            return {
                "name": name,
                "command": " ".join(command),
                "duration": 0,
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "status": "ERROR"
            }

    def run_sprint4_tests(self):
        """Run Sprint 4 distributed infrastructure tests."""
        print("=== Sprint 4 Integration Tests ===")

        result = self.run_test_suite(
            "Sprint 4 - Distributed Infrastructure",
            ["python", "scripts/create_integration_tests.py"],
            timeout=120
        )

        # Parse Sprint 4 results
        if "Passed:" in result["stdout"]:
            passed_line = [line for line in result["stdout"].split('\n') if "Passed:" in line][0]
            try:
                # Parse "Passed: 3/6 tests"
                parts = passed_line.split("Passed: ")[1].split("/")
                passed = int(parts[0].strip())
                total = int(parts[1].split()[0].strip())  # Remove "tests" word
                result["tests_passed"] = passed
                result["tests_total"] = total
                result["pass_rate"] = (passed / total) * 100 if total > 0 else 0
            except (ValueError, IndexError):
                result["tests_passed"] = 0
                result["tests_total"] = 0
                result["pass_rate"] = 0

        self.results["test_suites"]["sprint4"] = result

    def run_core_tests(self):
        """Run core module tests."""
        print("=== Core Module Tests ===")

        test_files = [
            ("Compression Pipeline", ["python", "-m", "pytest", "tests/test_compression_only.py", "-v", "--tb=no"]),
            ("Pipeline Simple", ["python", "-m", "pytest", "tests/test_pipeline_simple.py", "-v", "--tb=no"]),
            ("Evolution System", ["python", "-m", "pytest", "tests/test_corrected_evolution.py", "-v", "--tb=no"]),
            ("King Agent", ["python", "-m", "pytest", "tests/test_king_agent.py", "-v", "--tb=no"]),
        ]

        for name, command in test_files:
            result = self.run_test_suite(name, command, timeout=90)

            # Parse pytest results
            if "passed" in result["stdout"] or "failed" in result["stdout"]:
                lines = result["stdout"].split('\n')
                summary_line = [line for line in lines if " passed" in line or " failed" in line][-1]
                if "passed" in summary_line:
                    passed = int(summary_line.split(" passed")[0].split("=")[-1].strip())
                    result["tests_passed"] = passed
                    result["tests_total"] = passed
                    result["pass_rate"] = 100.0

            self.results["test_suites"][name.lower().replace(" ", "_")] = result

    def check_system_health(self):
        """Check overall system health."""
        print("=== System Health Check ===")

        health_checks = [
            ("Python Import Check", ["python", "-c", "import agent_forge; print('SUCCESS')"]),
            ("Configuration Check", ["python", "-c", "import agent_forge.compression; print('SUCCESS')"]),
            ("Dependency Check", ["python", "-c", "import torch, transformers; print('SUCCESS')"]),
        ]

        for name, command in health_checks:
            result = self.run_test_suite(name, command, timeout=30)
            self.results["test_suites"][name.lower().replace(" ", "_")] = result

    def run_linting_check(self):
        """Run code quality checks."""
        print("=== Code Quality Check ===")

        result = self.run_test_suite(
            "Ruff Linting",
            ["ruff", "check", "scripts/", "--output-format=json"],
            timeout=60
        )

        # Parse ruff results
        if result["return_code"] == 0:
            result["issues_found"] = 0
        else:
            try:
                issues = json.loads(result["stdout"])
                result["issues_found"] = len(issues)
                self.results["issues"].extend(issues[:10])  # Top 10 issues
            except BaseException:
                result["issues_found"] = "Unknown"

        self.results["test_suites"]["code_quality"] = result

    def generate_summary(self):
        """Generate test summary."""
        total_suites = len(self.results["test_suites"])
        passed_suites = sum(1 for suite in self.results["test_suites"].values()
                            if suite["status"] == "PASSED")

        total_tests = sum(suite.get("tests_total", 0) for suite in self.results["test_suites"].values())
        total_passed = sum(suite.get("tests_passed", 0) for suite in self.results["test_suites"].values())

        self.results["summary"] = {
            "total_test_suites": total_suites,
            "passed_test_suites": passed_suites,
            "suite_pass_rate": (passed_suites / total_suites * 100) if total_suites > 0 else 0,
            "total_individual_tests": total_tests,
            "total_passed_tests": total_passed,
            "overall_pass_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
            "total_issues": len(self.results["issues"])
        }

    def generate_report(self):
        """Generate comprehensive test report."""
        print("\n" + "=" * 80)
        print("AIVillage Test Dashboard - Comprehensive Results")
        print("=" * 80)

        # Summary
        summary = self.results["summary"]
        print(f"\n📊 OVERALL SUMMARY")
        print(
            f"   Test Suites: {
                summary['passed_test_suites']}/{
                summary['total_test_suites']} passed ({
                summary['suite_pass_rate']:.1f}%)")
        print(
            f"   Individual Tests: {
                summary['total_passed_tests']}/{
                summary['total_individual_tests']} passed ({
                summary['overall_pass_rate']:.1f}%)")
        print(f"   Code Quality Issues: {summary['total_issues']}")

        # Detailed results
        print(f"\n🔍 DETAILED RESULTS")
        for _name, suite in self.results["test_suites"].items():
            status_emoji = "✅" if suite["status"] == "PASSED" else "❌" if suite["status"] == "FAILED" else "⏱️" if suite["status"] == "TIMEOUT" else "⚠️"
            duration = f"({suite['duration']:.1f}s)"

            print(f"   {status_emoji} {suite['name']} {duration}")

            if suite.get("tests_total"):
                print(
                    f"      └─ {
                        suite['tests_passed']}/{
                        suite['tests_total']} tests passed ({
                        suite.get(
                            'pass_rate',
                            0):.1f}%)")

            if suite["status"] != "PASSED" and suite["stderr"]:
                error_preview = suite["stderr"][:100] + "..." if len(suite["stderr"]) > 100 else suite["stderr"]
                print(f"      └─ Error: {error_preview}")

        # Sprint 4 specific results
        if "sprint4" in self.results["test_suites"]:
            sprint4 = self.results["test_suites"]["sprint4"]
            print(f"\n🚀 SPRINT 4 - DISTRIBUTED INFRASTRUCTURE")
            print(f"   Status: {sprint4['status']}")
            print(f"   Tests: {sprint4.get('tests_passed', 0)}/{sprint4.get('tests_total', 0)} passed")

            if "stdout" in sprint4:
                # Extract specific test results
                lines = sprint4["stdout"].split('\n')
                test_results = [line for line in lines if "[PASS]" in line or "[FAIL]" in line]
                for result in test_results:
                    status_emoji = "✅" if "[PASS]" in result else "❌"
                    test_name = result.replace("[PASS]", "").replace("[FAIL]", "").strip()
                    print(f"   {status_emoji} {test_name}")

        # Performance insights
        print(f"\n⚡ PERFORMANCE INSIGHTS")
        longest_test = max(self.results["test_suites"].values(), key=lambda x: x["duration"])
        print(f"   Longest test: {longest_test['name']} ({longest_test['duration']:.1f}s)")

        total_duration = sum(suite["duration"] for suite in self.results["test_suites"].values())
        print(f"   Total test time: {total_duration:.1f}s")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        if summary["overall_pass_rate"] >= 90:
            print("   🎉 Excellent test coverage! System is in great shape.")
        elif summary["overall_pass_rate"] >= 70:
            print("   👍 Good test coverage. Consider addressing failing tests.")
        else:
            print("   ⚠️ Test coverage needs improvement. Focus on fixing failing tests.")

        if summary["total_issues"] > 0:
            print(f"   🔧 Address {summary['total_issues']} code quality issues for better maintainability.")

        print("\n" + "=" * 80)

        return self.results

    def save_report(self, filename: str = "test_dashboard_results.json"):
        """Save detailed results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📄 Detailed results saved to: {filename}")


def main():
    """Run comprehensive test dashboard."""
    dashboard = TestDashboard()

    # Run all test suites
    dashboard.run_sprint4_tests()
    dashboard.run_core_tests()
    dashboard.check_system_health()
    dashboard.run_linting_check()

    # Generate summary and report
    dashboard.generate_summary()
    results = dashboard.generate_report()
    dashboard.save_report()

    return results


if __name__ == "__main__":
    main()
