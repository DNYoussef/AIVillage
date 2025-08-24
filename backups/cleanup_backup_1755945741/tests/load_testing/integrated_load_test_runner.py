#!/usr/bin/env python3
"""
Integrated Load Test Runner for AIVillage Production Validation
==============================================================

Master orchestrator that coordinates all load testing infrastructure:
- Production load testing
- Soak testing for stability
- Performance regression detection
- Resource monitoring
- Automated reporting and alerts

Usage:
    python integrated_load_test_runner.py --full-suite
    python integrated_load_test_runner.py --quick-validation
    python integrated_load_test_runner.py --production-readiness
"""

import argparse
import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import our load testing modules
try:
    from .performance_regression_detector import PerformanceBenchmarkRunner, PerformanceRegressionDetector
    from .production_load_test_suite import ProductionLoadTestSuite, create_test_profiles
    from .soak_test_orchestrator import SoakTestConfig, SoakTestOrchestrator
except ImportError:
    from performance_regression_detector import PerformanceBenchmarkRunner, PerformanceRegressionDetector
    from production_load_test_suite import ProductionLoadTestSuite, create_test_profiles
    from soak_test_orchestrator import SoakTestConfig, SoakTestOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("integrated_load_test.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class TestSuiteConfig:
    """Configuration for integrated test suite"""

    suite_name: str = "aivillage_load_test_suite"
    base_url: str = "http://localhost:8000"
    output_dir: Path = field(default_factory=lambda: Path("load_test_results"))

    # Test selection
    run_quick_load_test: bool = True
    run_full_load_test: bool = True
    run_soak_test: bool = False
    run_regression_test: bool = True

    # Test parameters
    load_test_profiles: list[str] = field(default_factory=lambda: ["quick", "basic"])
    soak_test_duration_hours: float = 1.0

    # Thresholds
    max_acceptable_error_rate: float = 0.02
    max_acceptable_p99_latency: float = 2000.0
    max_acceptable_memory_mb: float = 1024.0

    # Alerts and notifications
    enable_alerts: bool = True
    alert_webhook_url: str | None = None
    slack_webhook_url: str | None = None

    # CI/CD integration
    ci_mode: bool = False
    generate_junit_xml: bool = False
    generate_html_report: bool = True


@dataclass
class TestResult:
    """Individual test result"""

    test_name: str
    test_type: str
    status: str  # "PASS", "FAIL", "WARNING", "SKIP"
    duration_seconds: float
    metrics: dict[str, Any]
    error_message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegratedTestReport:
    """Complete integrated test report"""

    suite_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    overall_status: str
    test_results: list[TestResult]
    summary: dict[str, Any]
    environment: dict[str, str]
    recommendations: list[str] = field(default_factory=list)


class SystemHealthChecker:
    """Pre-flight system health validation"""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def check_system_health(self) -> dict[str, Any]:
        """Comprehensive system health check"""
        logger.info("Performing pre-test system health check...")

        health_checks = {
            "system_reachable": await self._check_system_reachable(),
            "endpoints_available": await self._check_critical_endpoints(),
            "resource_availability": await self._check_resource_availability(),
            "dependencies_ready": await self._check_dependencies(),
        }

        overall_healthy = all(check["status"] == "healthy" for check in health_checks.values())

        return {"overall_healthy": overall_healthy, "checks": health_checks, "timestamp": datetime.now().isoformat()}

    async def _check_system_reachable(self) -> dict[str, Any]:
        """Check if system is reachable"""
        try:
            # Simple connectivity test
            import urllib.request

            with urllib.request.urlopen(f"{self.base_url}/health", timeout=10) as response:
                if response.getcode() == 200:
                    return {"status": "healthy", "message": "System reachable"}
                else:
                    return {"status": "unhealthy", "message": f"HTTP {response.getcode()}"}

        except Exception as e:
            return {"status": "unhealthy", "message": f"Connection failed: {str(e)}"}

    async def _check_critical_endpoints(self) -> dict[str, Any]:
        """Check critical API endpoints"""
        endpoints = ["/health", "/v1/agents/health", "/v1/rag/health"]
        results = {}

        for endpoint in endpoints:
            try:
                import urllib.request

                with urllib.request.urlopen(f"{self.base_url}{endpoint}", timeout=5) as response:
                    results[endpoint] = response.getcode() == 200

            except Exception:
                results[endpoint] = False

        healthy_count = sum(1 for healthy in results.values() if healthy)
        total_count = len(endpoints)

        if healthy_count == total_count:
            status = "healthy"
            message = "All endpoints healthy"
        elif healthy_count > total_count * 0.5:
            status = "degraded"
            message = f"{healthy_count}/{total_count} endpoints healthy"
        else:
            status = "unhealthy"
            message = f"Only {healthy_count}/{total_count} endpoints healthy"

        return {"status": status, "message": message, "details": results}

    async def _check_resource_availability(self) -> dict[str, Any]:
        """Check system resource availability"""
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Check if resources are adequate for testing
            issues = []
            if cpu_percent > 80:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            if memory.percent > 90:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            if disk.percent > 95:
                issues.append(f"Low disk space: {disk.percent:.1f}% used")

            if issues:
                return {"status": "degraded", "message": "Resource constraints detected", "issues": issues}
            else:
                return {
                    "status": "healthy",
                    "message": "Adequate resources available",
                    "details": {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "disk_percent": disk.percent,
                    },
                }

        except ImportError:
            return {"status": "unknown", "message": "psutil not available, cannot check resources"}

    async def _check_dependencies(self) -> dict[str, Any]:
        """Check external dependencies"""
        # Check if required Python packages are available
        required_packages = ["asyncio", "json", "pathlib"]
        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            return {"status": "unhealthy", "message": f'Missing packages: {", ".join(missing_packages)}'}
        else:
            return {"status": "healthy", "message": "All dependencies available"}


class AlertManager:
    """Manage alerts and notifications"""

    def __init__(self, config: TestSuiteConfig):
        self.config = config

    async def send_alert(self, report: IntegratedTestReport):
        """Send alert based on test results"""
        if not self.config.enable_alerts:
            return

        if report.overall_status == "FAIL":
            await self._send_failure_alert(report)
        elif report.overall_status == "WARNING":
            await self._send_warning_alert(report)

    async def _send_failure_alert(self, report: IntegratedTestReport):
        """Send failure alert"""
        logger.error("Sending failure alert...")

        message = f"""
🚨 AIVillage Load Test FAILURE 🚨

Suite: {report.suite_name}
Duration: {report.duration_seconds:.1f}s
Failed Tests: {len([r for r in report.test_results if r.status == 'FAIL'])}

Critical Issues:
{self._format_critical_issues(report)}

View full report: {self.config.output_dir}/integrated_report.json
        """.strip()

        await self._send_notification(message, "critical")

    async def _send_warning_alert(self, report: IntegratedTestReport):
        """Send warning alert"""
        logger.warning("Sending warning alert...")

        message = f"""
⚠️ AIVillage Load Test WARNING ⚠️

Suite: {report.suite_name}
Duration: {report.duration_seconds:.1f}s
Warnings: {len([r for r in report.test_results if r.status == 'WARNING'])}

Issues detected - review recommended.
View full report: {self.config.output_dir}/integrated_report.json
        """.strip()

        await self._send_notification(message, "warning")

    def _format_critical_issues(self, report: IntegratedTestReport) -> str:
        """Format critical issues for alerts"""
        failed_tests = [r for r in report.test_results if r.status == "FAIL"]
        issues = []

        for test in failed_tests[:3]:  # Show only first 3 failures
            issues.append(f"- {test.test_name}: {test.error_message or 'Failed'}")

        if len(failed_tests) > 3:
            issues.append(f"- ... and {len(failed_tests) - 3} more failures")

        return "\n".join(issues)

    async def _send_notification(self, message: str, severity: str):
        """Send notification via configured channels"""
        # Webhook notification
        if self.config.alert_webhook_url:
            await self._send_webhook(message, severity)

        # Slack notification
        if self.config.slack_webhook_url:
            await self._send_slack(message, severity)

        # Console notification
        if severity == "critical":
            logger.error(f"ALERT: {message}")
        else:
            logger.warning(f"ALERT: {message}")

    async def _send_webhook(self, message: str, severity: str):
        """Send webhook notification"""
        try:
            import json
            import urllib.request

            payload = {
                "message": message,
                "severity": severity,
                "timestamp": datetime.now().isoformat(),
                "source": "aivillage_load_test",
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.config.alert_webhook_url, data=data, headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                if response.getcode() == 200:
                    logger.info("Webhook alert sent successfully")
                else:
                    logger.error(f"Webhook alert failed: HTTP {response.getcode()}")

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

    async def _send_slack(self, message: str, severity: str):
        """Send Slack notification"""
        try:
            import json
            import urllib.request

            emoji = "🚨" if severity == "critical" else "⚠️"
            payload = {"text": f"{emoji} {message}", "username": "AIVillage Load Test", "icon_emoji": ":robot_face:"}

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.config.slack_webhook_url, data=data, headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                if response.getcode() == 200:
                    logger.info("Slack alert sent successfully")
                else:
                    logger.error(f"Slack alert failed: HTTP {response.getcode()}")

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")


class ReportGenerator:
    """Generate comprehensive test reports"""

    def __init__(self, config: TestSuiteConfig):
        self.config = config

    def generate_reports(self, report: IntegratedTestReport):
        """Generate all configured report formats"""
        # JSON report (always generated)
        self._generate_json_report(report)

        # HTML report
        if self.config.generate_html_report:
            self._generate_html_report(report)

        # JUnit XML for CI/CD
        if self.config.generate_junit_xml:
            self._generate_junit_xml(report)

    def _generate_json_report(self, report: IntegratedTestReport):
        """Generate JSON report"""
        report_file = self.config.output_dir / "integrated_report.json"

        with open(report_file, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)

        logger.info(f"JSON report saved to {report_file}")

    def _generate_html_report(self, report: IntegratedTestReport):
        """Generate HTML report"""
        html_content = self._create_html_content(report)
        report_file = self.config.output_dir / "integrated_report.html"

        with open(report_file, "w") as f:
            f.write(html_content)

        logger.info(f"HTML report saved to {report_file}")

    def _generate_junit_xml(self, report: IntegratedTestReport):
        """Generate JUnit XML report"""
        xml_content = self._create_junit_xml(report)
        report_file = self.config.output_dir / "junit_report.xml"

        with open(report_file, "w") as f:
            f.write(xml_content)

        logger.info(f"JUnit XML report saved to {report_file}")

    def _create_html_content(self, report: IntegratedTestReport) -> str:
        """Create HTML report content"""
        status_color = {"PASS": "#28a745", "FAIL": "#dc3545", "WARNING": "#ffc107", "SKIP": "#6c757d"}

        # Test results table rows
        test_rows = ""
        for test in report.test_results:
            color = status_color.get(test.status, "#6c757d")
            test_rows += f"""
            <tr>
                <td>{test.test_name}</td>
                <td>{test.test_type}</td>
                <td style="color: {color}; font-weight: bold;">{test.status}</td>
                <td>{test.duration_seconds:.1f}s</td>
                <td>{test.error_message or '-'}</td>
            </tr>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AIVillage Load Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .summary {{ background: #e9ecef; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .pass {{ color: #28a745; }}
                .fail {{ color: #dc3545; }}
                .warning {{ color: #ffc107; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AIVillage Load Test Report</h1>
                <p><strong>Suite:</strong> {report.suite_name}</p>
                <p><strong>Duration:</strong> {report.duration_seconds:.1f} seconds</p>
                <p><strong>Status:</strong> <span class="{report.overall_status.lower()}">{report.overall_status}</span></p>
                <p><strong>Generated:</strong> {report.end_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Tests:</strong> {report.summary['total_tests']}</p>
                <p><strong>Passed:</strong> {report.summary['passed']}</p>
                <p><strong>Failed:</strong> {report.summary['failed']}</p>
                <p><strong>Warnings:</strong> {report.summary['warnings']}</p>
                <p><strong>Skipped:</strong> {report.summary['skipped']}</p>
            </div>

            <h2>Test Results</h2>
            <table>
                <tr>
                    <th>Test Name</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Error Message</th>
                </tr>
                {test_rows}
            </table>

            <h2>Recommendations</h2>
            <ul>
                {"".join(f"<li>{rec}</li>" for rec in report.recommendations)}
            </ul>
        </body>
        </html>
        """

        return html

    def _create_junit_xml(self, report: IntegratedTestReport) -> str:
        """Create JUnit XML content"""
        test_cases = ""

        for test in report.test_results:
            if test.status == "FAIL":
                failure_elem = f'<failure message="{test.error_message or "Test failed"}">{test.error_message or "Test failed"}</failure>'
            else:
                failure_elem = ""

            test_cases += f"""
            <testcase name="{test.test_name}" classname="{test.test_type}" time="{test.duration_seconds:.3f}">
                {failure_elem}
            </testcase>
            """

        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
        <testsuite name="{report.suite_name}"
                   tests="{report.summary['total_tests']}"
                   failures="{report.summary['failed']}"
                   errors="0"
                   time="{report.duration_seconds:.3f}"
                   timestamp="{report.start_time.isoformat()}">
            {test_cases}
        </testsuite>
        """

        return xml


class IntegratedLoadTestRunner:
    """Main integrated load test runner"""

    def __init__(self, config: TestSuiteConfig):
        self.config = config
        self.test_results = []
        self.health_checker = SystemHealthChecker(config.base_url)
        self.alert_manager = AlertManager(config)
        self.report_generator = ReportGenerator(config)

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    async def run_full_suite(self) -> IntegratedTestReport:
        """Run complete integrated test suite"""
        logger.info(f"Starting integrated load test suite: {self.config.suite_name}")
        start_time = datetime.now()

        try:
            # Pre-flight health check
            health_check = await self.health_checker.check_system_health()
            if not health_check["overall_healthy"]:
                logger.error("System health check failed - aborting tests")
                return self._create_failed_report(start_time, "System health check failed")

            # Run test sequence
            if self.config.run_regression_test:
                await self._run_regression_tests()

            if self.config.run_quick_load_test:
                await self._run_quick_load_tests()

            if self.config.run_full_load_test:
                await self._run_full_load_tests()

            if self.config.run_soak_test:
                await self._run_soak_tests()

        except Exception as e:
            logger.error(f"Test suite execution failed: {e}")
            self.test_results.append(
                TestResult(
                    test_name="test_suite_execution",
                    test_type="system",
                    status="FAIL",
                    duration_seconds=0.0,
                    metrics={},
                    error_message=str(e),
                )
            )

        # Generate final report
        end_time = datetime.now()
        report = self._create_final_report(start_time, end_time)

        # Generate reports and alerts
        self.report_generator.generate_reports(report)
        await self.alert_manager.send_alert(report)

        return report

    async def _run_regression_tests(self):
        """Run performance regression tests"""
        logger.info("Running performance regression tests...")
        start_time = time.time()

        try:
            detector = PerformanceRegressionDetector(self.config.output_dir / "performance_data")
            benchmark_runner = PerformanceBenchmarkRunner({})

            # Run current benchmark
            current_benchmark = await benchmark_runner.run_full_benchmark()

            # Try to load baseline
            baseline = detector.load_baseline("baseline")

            if baseline:
                # Compare against baseline
                regression_report = detector.detect_regressions(baseline, current_benchmark)

                status = "PASS"
                error_message = None

                if regression_report.overall_verdict == "FAIL":
                    status = "FAIL"
                    error_message = f"Performance regressions detected: {regression_report.summary['worst_regression']}"
                elif regression_report.overall_verdict == "WARNING":
                    status = "WARNING"
                    error_message = f"Performance warnings: {regression_report.summary['warnings']} metrics"

                self.test_results.append(
                    TestResult(
                        test_name="performance_regression_test",
                        test_type="performance",
                        status=status,
                        duration_seconds=time.time() - start_time,
                        metrics=regression_report.summary,
                        error_message=error_message,
                    )
                )
            else:
                # No baseline - save current as baseline
                detector.save_baseline(current_benchmark)
                self.test_results.append(
                    TestResult(
                        test_name="performance_baseline_establishment",
                        test_type="performance",
                        status="PASS",
                        duration_seconds=time.time() - start_time,
                        metrics={"baseline_established": True},
                    )
                )

        except Exception as e:
            logger.error(f"Regression test failed: {e}")
            self.test_results.append(
                TestResult(
                    test_name="performance_regression_test",
                    test_type="performance",
                    status="FAIL",
                    duration_seconds=time.time() - start_time,
                    metrics={},
                    error_message=str(e),
                )
            )

    async def _run_quick_load_tests(self):
        """Run quick load tests"""
        logger.info("Running quick load tests...")

        profiles = create_test_profiles()

        for profile_name in ["quick"]:
            if profile_name not in profiles:
                continue

            start_time = time.time()

            try:
                config = profiles[profile_name]
                config.base_url = self.config.base_url

                test_suite = ProductionLoadTestSuite(config)
                metrics = await test_suite.run_load_test()

                status = "PASS" if metrics.passed else "FAIL"
                error_message = "; ".join(metrics.failure_reasons) if not metrics.passed else None

                self.test_results.append(
                    TestResult(
                        test_name=f"load_test_{profile_name}",
                        test_type="load",
                        status=status,
                        duration_seconds=time.time() - start_time,
                        metrics=asdict(metrics),
                        error_message=error_message,
                    )
                )

            except Exception as e:
                logger.error(f"Load test {profile_name} failed: {e}")
                self.test_results.append(
                    TestResult(
                        test_name=f"load_test_{profile_name}",
                        test_type="load",
                        status="FAIL",
                        duration_seconds=time.time() - start_time,
                        metrics={},
                        error_message=str(e),
                    )
                )

    async def _run_full_load_tests(self):
        """Run full load tests"""
        logger.info("Running full load tests...")

        profiles = create_test_profiles()

        for profile_name in self.config.load_test_profiles:
            if profile_name not in profiles or profile_name == "quick":
                continue

            start_time = time.time()

            try:
                config = profiles[profile_name]
                config.base_url = self.config.base_url

                test_suite = ProductionLoadTestSuite(config)
                metrics = await test_suite.run_load_test()

                status = "PASS" if metrics.passed else "FAIL"
                error_message = "; ".join(metrics.failure_reasons) if not metrics.passed else None

                self.test_results.append(
                    TestResult(
                        test_name=f"load_test_{profile_name}",
                        test_type="load",
                        status=status,
                        duration_seconds=time.time() - start_time,
                        metrics=asdict(metrics),
                        error_message=error_message,
                    )
                )

            except Exception as e:
                logger.error(f"Load test {profile_name} failed: {e}")
                self.test_results.append(
                    TestResult(
                        test_name=f"load_test_{profile_name}",
                        test_type="load",
                        status="FAIL",
                        duration_seconds=time.time() - start_time,
                        metrics={},
                        error_message=str(e),
                    )
                )

    async def _run_soak_tests(self):
        """Run soak tests"""
        logger.info("Running soak tests...")
        start_time = time.time()

        try:
            soak_config = SoakTestConfig(
                duration_hours=self.config.soak_test_duration_hours,
                base_url=self.config.base_url,
                output_dir=self.config.output_dir / "soak_test",
                concurrent_users=20,  # Reduce load for stability testing
                request_rate_per_second=5.0,
            )

            orchestrator = SoakTestOrchestrator(soak_config)
            metrics = await orchestrator.run_soak_test()

            # Evaluate soak test success
            test_passed = orchestrator._evaluate_test_success()
            status = "PASS" if test_passed else "FAIL"

            error_message = None
            if not test_passed:
                issues = []
                if metrics.error_rate > soak_config.max_error_rate:
                    issues.append(f"High error rate: {metrics.error_rate:.3f}")
                if metrics.memory_growth_rate_mb_per_hour > soak_config.max_memory_growth_mb_per_hour:
                    issues.append(f"Memory leak: {metrics.memory_growth_rate_mb_per_hour:.1f} MB/hour")
                error_message = "; ".join(issues)

            self.test_results.append(
                TestResult(
                    test_name="soak_test",
                    test_type="stability",
                    status=status,
                    duration_seconds=time.time() - start_time,
                    metrics=asdict(metrics),
                    error_message=error_message,
                )
            )

        except Exception as e:
            logger.error(f"Soak test failed: {e}")
            self.test_results.append(
                TestResult(
                    test_name="soak_test",
                    test_type="stability",
                    status="FAIL",
                    duration_seconds=time.time() - start_time,
                    metrics={},
                    error_message=str(e),
                )
            )

    def _create_final_report(self, start_time: datetime, end_time: datetime) -> IntegratedTestReport:
        """Create final integrated test report"""
        duration = (end_time - start_time).total_seconds()

        # Calculate summary
        total_tests = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.status == "PASS")
        failed = sum(1 for r in self.test_results if r.status == "FAIL")
        warnings = sum(1 for r in self.test_results if r.status == "WARNING")
        skipped = sum(1 for r in self.test_results if r.status == "SKIP")

        # Determine overall status
        if failed > 0:
            overall_status = "FAIL"
        elif warnings > 0:
            overall_status = "WARNING"
        else:
            overall_status = "PASS"

        # Generate recommendations
        recommendations = self._generate_recommendations()

        return IntegratedTestReport(
            suite_name=self.config.suite_name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            overall_status=overall_status,
            test_results=self.test_results,
            summary={
                "total_tests": total_tests,
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "skipped": skipped,
                "success_rate": (passed / total_tests * 100) if total_tests > 0 else 0,
            },
            environment=self._get_environment_info(),
            recommendations=recommendations,
        )

    def _create_failed_report(self, start_time: datetime, error_message: str) -> IntegratedTestReport:
        """Create report for failed test suite"""
        return IntegratedTestReport(
            suite_name=self.config.suite_name,
            start_time=start_time,
            end_time=datetime.now(),
            duration_seconds=0.0,
            overall_status="FAIL",
            test_results=[
                TestResult(
                    test_name="test_suite_setup",
                    test_type="system",
                    status="FAIL",
                    duration_seconds=0.0,
                    metrics={},
                    error_message=error_message,
                )
            ],
            summary={"total_tests": 1, "passed": 0, "failed": 1, "warnings": 0, "skipped": 0, "success_rate": 0},
            environment=self._get_environment_info(),
            recommendations=["Fix system health issues before running load tests"],
        )

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        failed_tests = [r for r in self.test_results if r.status == "FAIL"]
        warning_tests = [r for r in self.test_results if r.status == "WARNING"]

        if failed_tests:
            recommendations.append("Address failing tests before deploying to production")

        if warning_tests:
            recommendations.append("Review performance warnings and consider optimization")

        # Specific recommendations based on test types
        load_failures = [r for r in failed_tests if r.test_type == "load"]
        if load_failures:
            recommendations.append("Investigate system capacity and scaling configuration")

        performance_issues = [r for r in failed_tests + warning_tests if r.test_type == "performance"]
        if performance_issues:
            recommendations.append("Profile application performance and optimize bottlenecks")

        stability_issues = [r for r in failed_tests if r.test_type == "stability"]
        if stability_issues:
            recommendations.append("Check for memory leaks and resource management issues")

        if not recommendations:
            recommendations.append("All tests passed - system is ready for production deployment")

        return recommendations

    def _get_environment_info(self) -> dict[str, str]:
        """Get environment information"""
        import platform

        return {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "test_runner_version": "1.0.0",
            "base_url": self.config.base_url,
        }


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AIVillage Integrated Load Test Runner")

    # Test selection
    parser.add_argument("--full-suite", action="store_true", help="Run complete test suite")
    parser.add_argument("--quick-validation", action="store_true", help="Run quick validation tests")
    parser.add_argument("--production-readiness", action="store_true", help="Run production readiness tests")

    # Configuration
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for testing")
    parser.add_argument("--output-dir", type=Path, default="load_test_results", help="Output directory")
    parser.add_argument("--soak-duration", type=float, default=1.0, help="Soak test duration in hours")

    # Alerts and reporting
    parser.add_argument("--alert-webhook", help="Alert webhook URL")
    parser.add_argument("--slack-webhook", help="Slack webhook URL")
    parser.add_argument("--ci-mode", action="store_true", help="CI/CD mode (generate JUnit XML)")

    args = parser.parse_args()

    # Create configuration
    config = TestSuiteConfig(
        base_url=args.base_url,
        output_dir=args.output_dir,
        soak_test_duration_hours=args.soak_duration,
        alert_webhook_url=args.alert_webhook,
        slack_webhook_url=args.slack_webhook,
        ci_mode=args.ci_mode,
        generate_junit_xml=args.ci_mode,
    )

    # Configure test selection
    if args.quick_validation:
        config.run_quick_load_test = True
        config.run_full_load_test = False
        config.run_soak_test = False
        config.load_test_profiles = ["quick"]
    elif args.production_readiness:
        config.run_full_load_test = True
        config.run_soak_test = True
        config.load_test_profiles = ["basic", "stress"]
    elif args.full_suite:
        config.run_full_load_test = True
        config.run_soak_test = True
        config.load_test_profiles = ["quick", "basic", "stress", "scale"]

    logger.info("Starting integrated load test runner")
    logger.info(f"Configuration: {config.load_test_profiles}, soak: {config.run_soak_test}")

    # Run test suite
    runner = IntegratedLoadTestRunner(config)
    report = await runner.run_full_suite()

    # Print summary
    print("\n" + "=" * 80)
    print("INTEGRATED LOAD TEST SUITE RESULTS")
    print("=" * 80)
    print(f"Suite: {report.suite_name}")
    print(f"Duration: {report.duration_seconds:.1f}s")
    print(f"Overall Status: {report.overall_status}")
    print(
        f"Tests: {report.summary['total_tests']} total, {report.summary['passed']} passed, {report.summary['failed']} failed"
    )
    print(f"Success Rate: {report.summary['success_rate']:.1f}%")

    if report.recommendations:
        print("\nRecommendations:")
        for rec in report.recommendations:
            print(f"  • {rec}")

    print("=" * 80)

    # Return appropriate exit code
    if report.overall_status == "PASS":
        print("✅ All load tests passed - system ready for production")
        return 0
    elif report.overall_status == "WARNING":
        print("⚠️  Load tests completed with warnings - review recommended")
        return 1 if config.ci_mode else 0  # Warnings are non-fatal in non-CI mode
    else:
        print("❌ Load tests failed - system not ready for production")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
