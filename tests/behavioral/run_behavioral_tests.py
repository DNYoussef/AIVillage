#!/usr/bin/env python3
"""
Agent Forge Behavioral Test Suite Runner

Comprehensive test execution script that runs all behavioral tests with:
- Automated test discovery and categorization
- Service availability checking
- Performance monitoring and reporting
- Test coverage analysis
- CI/CD integration support
- Detailed result reporting and analytics

Usage:
    python run_behavioral_tests.py [options]
    
Options:
    --category <category>   Run specific test category (pipeline, training, integration, etc.)
    --performance          Run only performance benchmarks
    --integration          Run only integration tests
    --skip-slow            Skip slow tests
    --coverage             Generate coverage report
    --report-format <fmt>  Report format: console, json, html (default: console)
    --parallel             Run tests in parallel
    --ci                   CI/CD mode with structured output
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "core"))

# Test suite configuration
TEST_CATEGORIES = {
    "pipeline": {
        "description": "Agent Forge 8-phase pipeline tests",
        "files": ["test_agent_forge_pipeline_behavioral.py"],
        "markers": ["behavioral", "pipeline"],
        "estimated_duration": "5 minutes",
    },
    "training": {
        "description": "Real Cognate training system tests",
        "files": ["test_cognate_training_system.py"],
        "markers": ["training", "behavioral"],
        "estimated_duration": "3 minutes",
    },
    "integration": {
        "description": "WebSocket and UI integration tests",
        "files": ["test_integration_websocket_ui.py"],
        "markers": ["integration", "requires_websocket"],
        "estimated_duration": "4 minutes",
    },
    "contracts": {
        "description": "Behavioral contract validation tests",
        "files": ["test_behavioral_contracts.py"],
        "markers": ["contracts", "behavioral"],
        "estimated_duration": "2 minutes",
    },
    "performance": {
        "description": "Performance benchmarks and load tests",
        "files": ["test_performance_benchmarks.py"],
        "markers": ["performance", "slow"],
        "estimated_duration": "8 minutes",
    },
    "error_recovery": {
        "description": "Error scenarios and recovery tests",
        "files": ["test_error_recovery_scenarios.py"],
        "markers": ["error_recovery", "behavioral"],
        "estimated_duration": "3 minutes",
    },
}

logger = logging.getLogger(__name__)


class TestSuiteRunner:
    """Comprehensive test suite runner with monitoring and reporting."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_dir = project_root / "tests" / "behavioral"
        self.results = {
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0,
            "categories_run": [],
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "error_tests": 0,
            "coverage_percent": None,
            "performance_summary": {},
            "service_availability": {},
            "detailed_results": {},
        }

    def check_service_availability(self) -> Dict[str, bool]:
        """Check availability of required services."""
        import requests

        services = {
            "controller_api": "http://localhost:8087/health",
            "websocket_server": "http://localhost:8085/",
            "chat_api": "http://localhost:8084/health",
        }

        availability = {}

        for service_name, endpoint in services.items():
            try:
                response = requests.get(endpoint, timeout=5)
                availability[service_name] = response.status_code == 200
                logger.info(f"✅ {service_name}: Available")
            except Exception as e:
                availability[service_name] = False
                logger.info(f"❌ {service_name}: Not available ({type(e).__name__})")

        self.results["service_availability"] = availability
        return availability

    def run_pytest_command(
        self,
        test_files: List[str] = None,
        markers: List[str] = None,
        extra_args: List[str] = None,
        coverage: bool = False,
        parallel: bool = False,
    ) -> Tuple[int, str, str]:
        """Run pytest with specified parameters."""

        cmd = [sys.executable, "-m", "pytest"]

        # Add test directory
        if test_files:
            for file in test_files:
                cmd.append(str(self.test_dir / file))
        else:
            cmd.append(str(self.test_dir))

        # Add markers
        if markers:
            marker_expr = " and ".join(markers)
            cmd.extend(["-m", marker_expr])

        # Add coverage if requested
        if coverage:
            cmd.extend(
                [
                    "--cov=core",
                    "--cov=infrastructure",
                    "--cov-report=term-missing",
                    "--cov-report=json:tests/data/coverage.json",
                ]
            )

        # Add parallel execution if requested
        if parallel:
            import psutil

            cpu_count = psutil.cpu_count()
            cmd.extend(["-n", str(min(cpu_count, 4))])  # Max 4 parallel processes

        # Add extra arguments
        if extra_args:
            cmd.extend(extra_args)

        # Add standard arguments
        cmd.extend(
            [
                "--tb=short",
                "--durations=10",
                "--json-report",
                f"--json-report-file={self.test_dir / 'test_results.json'}",
            ]
        )

        logger.info(f"Running command: {' '.join(cmd)}")

        # Execute pytest
        start_time = time.time()
        process = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.project_root))
        duration = time.time() - start_time

        logger.info(f"Pytest completed in {duration:.1f}s with exit code {process.returncode}")

        return process.returncode, process.stdout, process.stderr

    def parse_test_results(self) -> Dict[str, Any]:
        """Parse pytest JSON results."""
        results_file = self.test_dir / "test_results.json"

        if not results_file.exists():
            logger.warning("Test results JSON file not found")
            return {}

        try:
            with open(results_file) as f:
                data = json.load(f)

            summary = data.get("summary", {})

            return {
                "total": summary.get("total", 0),
                "passed": summary.get("passed", 0),
                "failed": summary.get("failed", 0),
                "skipped": summary.get("skipped", 0),
                "error": summary.get("error", 0),
                "duration": data.get("duration", 0),
                "tests": data.get("tests", []),
            }

        except Exception as e:
            logger.error(f"Failed to parse test results: {e}")
            return {}

    def analyze_performance_results(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance test results."""
        performance_summary = {
            "benchmarks_run": 0,
            "avg_response_time_ms": None,
            "memory_efficiency": None,
            "throughput_metrics": {},
            "failed_benchmarks": [],
        }

        if not test_results.get("tests"):
            return performance_summary

        performance_tests = [test for test in test_results["tests"] if "performance" in test.get("nodeid", "").lower()]

        performance_summary["benchmarks_run"] = len(performance_tests)

        # Analyze individual performance tests
        for test in performance_tests:
            if test.get("outcome") == "failed":
                performance_summary["failed_benchmarks"].append(
                    {"test_name": test.get("nodeid", ""), "failure_reason": test.get("call", {}).get("longrepr", "")}
                )

        return performance_summary

    def generate_coverage_report(self) -> Optional[float]:
        """Generate and parse coverage report."""
        coverage_file = PROJECT_ROOT / "tests" / "data" / "coverage.json"

        if not coverage_file.exists():
            return None

        try:
            with open(coverage_file) as f:
                coverage_data = json.load(f)

            total_coverage = coverage_data.get("totals", {}).get("percent_covered")

            if total_coverage is not None:
                logger.info(f"Test coverage: {total_coverage:.1f}%")
                return total_coverage

        except Exception as e:
            logger.error(f"Failed to parse coverage report: {e}")

        return None

    def run_category(self, category: str, **kwargs) -> bool:
        """Run tests for a specific category."""
        if category not in TEST_CATEGORIES:
            logger.error(f"Unknown test category: {category}")
            return False

        cat_config = TEST_CATEGORIES[category]
        logger.info(f"🚀 Running {category} tests: {cat_config['description']}")
        logger.info(f"   Estimated duration: {cat_config['estimated_duration']}")

        # Run the tests
        exit_code, stdout, stderr = self.run_pytest_command(
            test_files=cat_config["files"], markers=cat_config.get("markers"), **kwargs
        )

        success = exit_code == 0

        if success:
            logger.info(f"✅ {category} tests completed successfully")
        else:
            logger.error(f"❌ {category} tests failed with exit code {exit_code}")
            if stderr:
                logger.error(f"   Error output: {stderr[:500]}...")

        # Parse results
        test_results = self.parse_test_results()

        self.results["categories_run"].append(
            {"category": category, "success": success, "exit_code": exit_code, "results": test_results}
        )

        return success

    def run_all_categories(self, skip_categories: List[str] = None, **kwargs) -> bool:
        """Run all test categories."""
        skip_categories = skip_categories or []

        logger.info("🚀 Running complete Agent Forge Behavioral Test Suite")

        total_success = True

        for category in TEST_CATEGORIES.keys():
            if category in skip_categories:
                logger.info(f"⏭️  Skipping {category} tests")
                continue

            success = self.run_category(category, **kwargs)
            if not success:
                total_success = False

        return total_success

    def generate_report(self, format: str = "console") -> str:
        """Generate test results report."""
        if format == "console":
            return self.generate_console_report()
        elif format == "json":
            return self.generate_json_report()
        elif format == "html":
            return self.generate_html_report()
        else:
            raise ValueError(f"Unknown report format: {format}")

    def generate_console_report(self) -> str:
        """Generate console-formatted report."""
        report_lines = ["=" * 80, "🧪 AGENT FORGE BEHAVIORAL TEST SUITE RESULTS", "=" * 80, ""]

        # Summary
        total_categories = len(self.results["categories_run"])
        successful_categories = sum(1 for cat in self.results["categories_run"] if cat["success"])

        report_lines.extend(
            [
                "📊 SUMMARY",
                f"   Categories run: {total_categories}",
                f"   Successful: {successful_categories}",
                f"   Failed: {total_categories - successful_categories}",
                f"   Duration: {self.results['duration_seconds']:.1f}s",
                "",
            ]
        )

        # Service availability
        if self.results["service_availability"]:
            report_lines.append("🔗 SERVICE AVAILABILITY")
            for service, available in self.results["service_availability"].items():
                status = "✅ Available" if available else "❌ Unavailable"
                report_lines.append(f"   {service}: {status}")
            report_lines.append("")

        # Category results
        report_lines.append("📋 CATEGORY RESULTS")
        for category_result in self.results["categories_run"]:
            category = category_result["category"]
            success = category_result["success"]
            results = category_result.get("results", {})

            status = "✅ PASSED" if success else "❌ FAILED"
            report_lines.append(f"   {category}: {status}")

            if results:
                report_lines.append(f"      Total tests: {results.get('total', 0)}")
                report_lines.append(f"      Passed: {results.get('passed', 0)}")
                report_lines.append(f"      Failed: {results.get('failed', 0)}")
                report_lines.append(f"      Skipped: {results.get('skipped', 0)}")
                report_lines.append(f"      Duration: {results.get('duration', 0):.1f}s")

            report_lines.append("")

        # Coverage
        if self.results["coverage_percent"]:
            report_lines.extend(
                ["📈 TEST COVERAGE", f"   Overall coverage: {self.results['coverage_percent']:.1f}%", ""]
            )

        # Performance summary
        if self.results["performance_summary"]:
            perf = self.results["performance_summary"]
            report_lines.extend(
                [
                    "⚡ PERFORMANCE BENCHMARKS",
                    f"   Benchmarks run: {perf.get('benchmarks_run', 0)}",
                    f"   Failed benchmarks: {len(perf.get('failed_benchmarks', []))}",
                    "",
                ]
            )

        report_lines.extend(["=" * 80, f"🏁 Test suite completed at {datetime.now().isoformat()}", "=" * 80])

        return "\n".join(report_lines)

    def generate_json_report(self) -> str:
        """Generate JSON-formatted report."""
        return json.dumps(self.results, indent=2, default=str)

    def generate_html_report(self) -> str:
        """Generate HTML-formatted report."""
        # Basic HTML template
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Agent Forge Behavioral Test Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .success { color: green; }
        .failure { color: red; }
        .summary { margin: 20px 0; }
        .category { margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 Agent Forge Behavioral Test Results</h1>
        <p>Generated: {timestamp}</p>
    </div>
    
    <div class="summary">
        <h2>📊 Summary</h2>
        <p>Duration: {duration:.1f}s</p>
        <p>Categories: {total_categories}</p>
        <p>Overall Status: <span class="{overall_class}">{overall_status}</span></p>
    </div>
    
    <div class="categories">
        <h2>📋 Category Results</h2>
        {category_results}
    </div>
</body>
</html>
"""

        # Generate category results HTML
        category_html = []
        for cat_result in self.results["categories_run"]:
            success_class = "success" if cat_result["success"] else "failure"
            status = "PASSED" if cat_result["success"] else "FAILED"

            category_html.append(
                f"""
            <div class="category">
                <h3>{cat_result["category"]}: <span class="{success_class}">{status}</span></h3>
                <p>Tests: {cat_result.get("results", {}).get("total", "N/A")}</p>
            </div>
            """
            )

        # Determine overall status
        successful_categories = sum(1 for cat in self.results["categories_run"] if cat["success"])
        total_categories = len(self.results["categories_run"])
        overall_success = successful_categories == total_categories and total_categories > 0

        return html_template.format(
            timestamp=datetime.now().isoformat(),
            duration=self.results["duration_seconds"],
            total_categories=total_categories,
            overall_class="success" if overall_success else "failure",
            overall_status="ALL TESTS PASSED" if overall_success else "SOME TESTS FAILED",
            category_results="\n".join(category_html),
        )

    def run(
        self,
        category: Optional[str] = None,
        skip_slow: bool = False,
        coverage: bool = False,
        parallel: bool = False,
        report_format: str = "console",
        ci_mode: bool = False,
    ) -> int:
        """Run the test suite with specified options."""

        self.results["start_time"] = datetime.now()

        logger.info("🚀 Starting Agent Forge Behavioral Test Suite")

        # Check service availability
        self.check_service_availability()

        # Prepare pytest arguments
        extra_args = []
        if skip_slow:
            extra_args.extend(["-m", "not slow"])

        if ci_mode:
            extra_args.extend(["--tb=line", "--quiet"])

        # Run tests
        success = True

        try:
            if category:
                success = self.run_category(category, extra_args=extra_args, coverage=coverage, parallel=parallel)
            else:
                skip_categories = []
                if skip_slow:
                    skip_categories.append("performance")

                success = self.run_all_categories(
                    skip_categories=skip_categories, extra_args=extra_args, coverage=coverage, parallel=parallel
                )

            # Generate coverage report
            if coverage:
                self.results["coverage_percent"] = self.generate_coverage_report()

            # Analyze performance results
            if not skip_slow:
                for cat_result in self.results["categories_run"]:
                    if cat_result["category"] == "performance":
                        self.results["performance_summary"] = self.analyze_performance_results(
                            cat_result.get("results", {})
                        )
                        break

        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            success = False

        finally:
            self.results["end_time"] = datetime.now()
            self.results["duration_seconds"] = (self.results["end_time"] - self.results["start_time"]).total_seconds()

        # Generate and display report
        report = self.generate_report(report_format)

        if report_format == "console":
            print(report)
        else:
            # Save report to file
            report_file = self.test_dir / f"test_report.{report_format}"
            with open(report_file, "w") as f:
                f.write(report)
            logger.info(f"Report saved to: {report_file}")

        return 0 if success else 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Agent Forge Behavioral Test Suite Runner")

    parser.add_argument("--category", choices=list(TEST_CATEGORIES.keys()), help="Run specific test category")

    parser.add_argument("--performance", action="store_true", help="Run only performance benchmarks")

    parser.add_argument("--integration", action="store_true", help="Run only integration tests")

    parser.add_argument("--skip-slow", action="store_true", help="Skip slow tests")

    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")

    parser.add_argument("--report-format", choices=["console", "json", "html"], default="console", help="Report format")

    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")

    parser.add_argument("--ci", action="store_true", help="CI/CD mode with structured output")

    parser.add_argument("--list-categories", action="store_true", help="List available test categories")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # List categories if requested
    if args.list_categories:
        print("📋 Available Test Categories:")
        for category, config in TEST_CATEGORIES.items():
            print(f"   {category}: {config['description']}")
            print(f"      Duration: {config['estimated_duration']}")
            print(f"      Files: {', '.join(config['files'])}")
            print()
        return 0

    # Determine category
    category = args.category
    if args.performance:
        category = "performance"
    elif args.integration:
        category = "integration"

    # Create and run test suite
    runner = TestSuiteRunner(PROJECT_ROOT)

    exit_code = runner.run(
        category=category,
        skip_slow=args.skip_slow,
        coverage=args.coverage,
        parallel=args.parallel,
        report_format=args.report_format,
        ci_mode=args.ci,
    )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
