#!/usr/bin/env python3
"""
Comprehensive Test Coverage Dashboard for AIVillage
Real-time monitoring and analysis of test coverage across all components.
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class CoverageMetrics:
    """Container for coverage metrics."""

    total_lines: int
    covered_lines: int
    coverage_percent: float
    missing_lines: list[int]
    file_path: str
    timestamp: str


@dataclass
class TestSuiteMetrics:
    """Container for test suite metrics."""

    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    execution_time: float
    test_files: list[str]


class TestCoverageDashboard:
    """Comprehensive test coverage dashboard."""

    def __init__(self, project_root: str | None = None):
        self.project_root = Path(project_root or os.getcwd())
        self.coverage_data = {}
        self.test_metrics = {}
        self.critical_components = self._identify_critical_components()

    def _identify_critical_components(self) -> dict[str, list[str]]:
        """Identify critical components that require high test coverage."""
        return {
            "sprint6_infrastructure": [
                "src/core/p2p/",
                "src/core/resources/",
                "src/production/agent_forge/evolution/infrastructure_aware_evolution.py",
                "src/production/agent_forge/evolution/resource_constrained_evolution.py",
                "src/production/agent_forge/evolution/evolution_coordination_protocol.py",
            ],
            "sprint6_p2p_layer": [
                "src/core/p2p/p2p_node.py",
                "src/core/p2p/peer_discovery.py",
                "src/core/p2p/message_protocol.py",
                "src/core/p2p/encryption_layer.py",
            ],
            "sprint6_resource_management": [
                "src/core/resources/device_profiler.py",
                "src/core/resources/resource_monitor.py",
                "src/core/resources/constraint_manager.py",
                "src/core/resources/adaptive_loader.py",
            ],
            "sprint6_evolution_systems": [
                "src/production/agent_forge/evolution/base.py",
                "src/production/agent_forge/evolution/dual_evolution_system.py",
                "src/production/agent_forge/evolution/magi_architectural_evolution.py",
                "src/production/agent_forge/evolution/nightly_evolution_orchestrator.py",
            ],
            "production_critical": [
                "production/compression/",
                "production/evolution/",
                "production/rag/",
                "mcp_servers/hyperag/server.py",
                "mcp_servers/hyperag/protocol.py",
            ],
            "core_systems": [
                "agent_forge/core/",
                "agent_forge/orchestrator.py",
                "communications/protocol.py",
                "digital_twin/core/",
            ],
            "security_components": [
                "mcp_servers/hyperag/auth.py",
                "digital_twin/security/",
                "mcp_servers/hyperag/guardian/",
            ],
            "integration_points": [
                "experimental/services/",
                "mcp_servers/hyperag/memory/",
                "hyperag/education/",
            ],
        }

    def run_coverage_analysis(self) -> dict:
        """Run comprehensive coverage analysis."""
        print("🔍 Running comprehensive test coverage analysis...")

        # Step 1: Run Sprint 6 validation
        sprint6_results = self._run_sprint6_validation()

        # Step 2: Run Sprint 6 infrastructure tests
        infrastructure_results = self._run_sprint6_infrastructure_tests()

        # Step 3: Run Sprint 6 performance tests
        performance_results = self._run_sprint6_performance_tests()

        # Step 4: Run pytest with coverage
        coverage_results = self._run_pytest_coverage()

        # Step 5: Analyze coverage by component
        component_coverage = self._analyze_component_coverage()

        # Step 6: Identify gaps and priorities
        coverage_gaps = self._identify_coverage_gaps()

        # Step 7: Generate recommendations
        recommendations = self._generate_recommendations(coverage_gaps)

        # Step 8: Create visual dashboard
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "sprint6_validation": sprint6_results,
            "sprint6_infrastructure": infrastructure_results,
            "sprint6_performance": performance_results,
            "overall_coverage": coverage_results,
            "component_coverage": component_coverage,
            "coverage_gaps": coverage_gaps,
            "recommendations": recommendations,
            "test_suite_health": self._analyze_test_suite_health(),
        }

        return dashboard_data

    def _run_pytest_coverage(self) -> dict:
        """Run pytest with coverage collection."""
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--cov=.",
            "--cov-report=json:coverage_report.json",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-config=.coveragerc",
            "-v",
            "--tb=short",
            "--durations=10",
        ]

        print(f"Executing: {' '.join(cmd)}")

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=self.project_root,
                check=False,  # 10 minute timeout
            )
            execution_time = time.time() - start_time

            print(f"✅ Pytest completed in {execution_time:.2f} seconds")
            print(f"Exit code: {result.returncode}")

            if result.returncode != 0:
                print(f"⚠️ Pytest warnings/errors:\n{result.stderr}")

            # Load coverage results
            coverage_file = self.project_root / "coverage_report.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)

                return {
                    "success": True,
                    "execution_time": execution_time,
                    "coverage_data": coverage_data,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            return {
                "success": False,
                "error": "Coverage report not generated",
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Pytest execution timed out",
                "execution_time": 600,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _analyze_component_coverage(self) -> dict:
        """Analyze coverage by critical components."""
        component_metrics = {}

        for category, components in self.critical_components.items():
            category_metrics = {
                "total_files": 0,
                "covered_files": 0,
                "average_coverage": 0.0,
                "files": {},
            }

            total_coverage = 0
            file_count = 0

            for component_path in components:
                files = self._find_python_files(component_path)

                for file_path in files:
                    file_coverage = self._get_file_coverage(file_path)
                    category_metrics["files"][str(file_path)] = file_coverage

                    if file_coverage is not None:
                        total_coverage += file_coverage["coverage_percent"]
                        file_count += 1

                        if file_coverage["coverage_percent"] > 0:
                            category_metrics["covered_files"] += 1

                    category_metrics["total_files"] += 1

            if file_count > 0:
                category_metrics["average_coverage"] = total_coverage / file_count

            component_metrics[category] = category_metrics

        return component_metrics

    def _find_python_files(self, path_pattern: str) -> list[Path]:
        """Find Python files matching pattern."""
        if path_pattern.endswith("/"):
            # Directory pattern
            pattern = self.project_root / path_pattern / "**/*.py"
            return list(Path().glob(str(pattern)))
        # Specific file
        file_path = self.project_root / path_pattern
        return [file_path] if file_path.exists() else []

    def _get_file_coverage(self, file_path: Path) -> dict | None:
        """Get coverage data for specific file."""
        # This would integrate with coverage.py data
        # For now, return mock data structure

        if not file_path.exists():
            return None

        # Count lines in file
        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
                total_lines = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
        except Exception:
            total_lines = 0

        # Mock coverage calculation (would use actual coverage data)
        coverage_percent = 0.0
        covered_lines = 0
        missing_lines = []

        return {
            "coverage_percent": coverage_percent,
            "total_lines": total_lines,
            "covered_lines": covered_lines,
            "missing_lines": missing_lines,
            "file_path": str(file_path),
            "has_tests": self._file_has_tests(file_path),
        }

    def _file_has_tests(self, file_path: Path) -> bool:
        """Check if file has corresponding tests."""
        # Check for test files
        test_patterns = [
            self.project_root / "tests" / f"test_{file_path.stem}.py",
            self.project_root / "tests" / file_path.parent.name / f"test_{file_path.stem}.py",
            file_path.parent / f"test_{file_path.stem}.py",
        ]

        return any(test_path.exists() for test_path in test_patterns)

    def _identify_coverage_gaps(self) -> dict:
        """Identify critical coverage gaps."""
        gaps = {
            "critical_untested": [],
            "low_coverage": [],
            "missing_integration_tests": [],
            "security_gaps": [],
        }

        # Find untested critical files
        for category, components in self.critical_components.items():
            for component_path in components:
                files = self._find_python_files(component_path)

                for file_path in files:
                    if not self._file_has_tests(file_path):
                        gaps["critical_untested"].append(
                            {
                                "file": str(file_path),
                                "category": category,
                                "priority": ("high" if "production_critical" in category else "medium"),
                            }
                        )

        return gaps

    def _generate_recommendations(self, coverage_gaps: dict) -> list[dict]:
        """Generate recommendations for improving coverage."""
        recommendations = []

        # High priority recommendations
        if coverage_gaps["critical_untested"]:
            recommendations.append(
                {
                    "priority": "critical",
                    "title": "Create tests for production-critical components",
                    "description": f"Found {len(coverage_gaps['critical_untested'])} untested critical files",
                    "action_items": [
                        f"Create test file for {gap['file']}" for gap in coverage_gaps["critical_untested"][:5]
                    ],
                    "estimated_effort": "2-3 days",
                    "impact": "High - Essential for production reliability",
                }
            )

        # Coverage improvement recommendations
        recommendations.append(
            {
                "priority": "high",
                "title": "Improve MCP server test coverage",
                "description": "MCP servers are core to the system but lack comprehensive tests",
                "action_items": [
                    "Create integration tests for server.py",
                    "Add unit tests for protocol.py",
                    "Test authentication and authorization",
                    "Add performance tests for concurrent connections",
                ],
                "estimated_effort": "3-4 days",
                "impact": "High - Critical for system stability",
            }
        )

        recommendations.append(
            {
                "priority": "high",
                "title": "Implement compression pipeline testing",
                "description": "Production compression needs comprehensive test coverage",
                "action_items": [
                    "Test compression algorithms",
                    "Add performance benchmarks",
                    "Test model loading and saving",
                    "Add integration tests with evolution system",
                ],
                "estimated_effort": "2-3 days",
                "impact": "High - Production critical component",
            }
        )

        recommendations.append(
            {
                "priority": "medium",
                "title": "Add integration test framework",
                "description": "Need end-to-end integration testing",
                "action_items": [
                    "Create integration test fixtures",
                    "Add multi-component workflow tests",
                    "Test external API integrations",
                    "Add load testing capabilities",
                ],
                "estimated_effort": "1-2 weeks",
                "impact": "Medium - Improves system reliability",
            }
        )

        return recommendations

    def _analyze_test_suite_health(self) -> dict:
        """Analyze overall test suite health."""
        test_files = list(self.project_root.rglob("test_*.py"))
        test_files.extend(list(self.project_root.rglob("*_test.py")))

        health_metrics = {
            "total_test_files": len(test_files),
            "test_distribution": {},
            "test_quality_issues": [],
            "performance_issues": [],
        }

        # Analyze test distribution
        for test_file in test_files:
            category = test_file.parent.name
            if category not in health_metrics["test_distribution"]:
                health_metrics["test_distribution"][category] = 0
            health_metrics["test_distribution"][category] += 1

        return health_metrics

    def _run_sprint6_validation(self) -> dict:
        """Run Sprint 6 validation script."""
        print("🚀 Running Sprint 6 validation...")

        validation_script = self.project_root / "validate_sprint6.py"
        if not validation_script.exists():
            return {
                "success": False,
                "error": "validate_sprint6.py not found",
                "timestamp": datetime.now().isoformat(),
            }

        try:
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, str(validation_script)],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=self.project_root,
                check=False,
            )
            execution_time = time.time() - start_time

            return {
                "success": result.returncode == 0,
                "execution_time": execution_time,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat(),
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Sprint 6 validation timed out",
                "execution_time": 300,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _run_sprint6_infrastructure_tests(self) -> dict:
        """Run Sprint 6 infrastructure tests."""
        print("🏗️ Running Sprint 6 infrastructure tests...")

        test_files = [
            "tests/test_sprint6_infrastructure.py",
            "tests/communications/test_p2p.py",
            "tests/core/test_integration.py",
        ]

        results = {
            "total_files": len(test_files),
            "passed_files": 0,
            "failed_files": 0,
            "results": {},
            "timestamp": datetime.now().isoformat(),
        }

        for test_file in test_files:
            test_path = self.project_root / test_file
            if not test_path.exists():
                results["results"][test_file] = {
                    "success": False,
                    "error": "Test file not found",
                }
                results["failed_files"] += 1
                continue

            try:
                start_time = time.time()
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        str(test_path),
                        "-v",
                        "--tb=short",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=180,
                    cwd=self.project_root,
                    check=False,
                )
                execution_time = time.time() - start_time

                if result.returncode == 0:
                    results["passed_files"] += 1
                else:
                    results["failed_files"] += 1

                results["results"][test_file] = {
                    "success": result.returncode == 0,
                    "execution_time": execution_time,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }

            except subprocess.TimeoutExpired:
                results["results"][test_file] = {
                    "success": False,
                    "error": "Test execution timed out",
                    "execution_time": 180,
                }
                results["failed_files"] += 1
            except Exception as e:
                results["results"][test_file] = {"success": False, "error": str(e)}
                results["failed_files"] += 1

        return results

    def _run_sprint6_performance_tests(self) -> dict:
        """Run Sprint 6 performance tests."""
        print("⚡ Running Sprint 6 performance tests...")

        performance_test = "tests/test_sprint6_performance.py"
        test_path = self.project_root / performance_test

        if not test_path.exists():
            return {
                "success": False,
                "error": "Performance test file not found",
                "timestamp": datetime.now().isoformat(),
            }

        try:
            start_time = time.time()
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(test_path),
                    "-v",
                    "--benchmark-only",
                    "--tb=short",
                ],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=self.project_root,
                check=False,
            )
            execution_time = time.time() - start_time

            return {
                "success": result.returncode == 0,
                "execution_time": execution_time,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat(),
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Performance tests timed out",
                "execution_time": 300,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def run_sprint6_test_suite(self) -> dict:
        """Run comprehensive Sprint 6 test suite."""
        print("🧪 Running comprehensive Sprint 6 test suite...")

        # Start time tracking
        suite_start_time = time.time()

        # Run all Sprint 6 tests
        validation_results = self._run_sprint6_validation()
        infrastructure_results = self._run_sprint6_infrastructure_tests()
        performance_results = self._run_sprint6_performance_tests()

        # Calculate overall results
        total_execution_time = time.time() - suite_start_time

        # Determine overall success
        overall_success = (
            validation_results.get("success", False)
            and infrastructure_results.get("passed_files", 0) == infrastructure_results.get("total_files", 1)
            and performance_results.get("success", False)
        )

        suite_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_success": overall_success,
            "total_execution_time": total_execution_time,
            "validation": validation_results,
            "infrastructure": infrastructure_results,
            "performance": performance_results,
            "summary": {
                "validation_passed": validation_results.get("success", False),
                "infrastructure_tests_passed": infrastructure_results.get("passed_files", 0),
                "infrastructure_tests_failed": infrastructure_results.get("failed_files", 0),
                "performance_tests_passed": performance_results.get("success", False),
            },
        }

        # Save results
        results_file = self.project_root / "sprint6_test_results.json"
        with open(results_file, "w") as f:
            json.dump(suite_results, f, indent=2)

        print(f"✅ Sprint 6 test suite completed in {total_execution_time:.2f} seconds")
        print(f"📊 Overall success: {overall_success}")
        print(f"📄 Results saved to: {results_file}")

        return suite_results

    def generate_html_dashboard(self, dashboard_data: dict) -> str:
        """Generate HTML dashboard."""
        # Determine Sprint 6 status
        sprint6_status = "🟢 Healthy"
        sprint6_class = "success"

        if "sprint6_validation" in dashboard_data:
            validation_success = dashboard_data["sprint6_validation"].get("success", False)
            infrastructure_success = dashboard_data.get("sprint6_infrastructure", {}).get("passed_files", 0) > 0
            performance_success = dashboard_data.get("sprint6_performance", {}).get("success", False)

            if validation_success and infrastructure_success and performance_success:
                sprint6_status = "🟢 All Systems Operational"
                sprint6_class = "success"
            elif validation_success and infrastructure_success:
                sprint6_status = "🟡 Core Systems OK, Performance Issues"
                sprint6_class = "warning"
            elif validation_success:
                sprint6_status = "🟡 Validation OK, Infrastructure Issues"
                sprint6_class = "warning"
            else:
                sprint6_status = "🔴 Critical Issues Detected"
                sprint6_class = "critical"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Sprint 6 Test Dashboard - AIVillage</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; background: #f8f9fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 30px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); }}
        .header h1 {{ margin: 0 0 10px 0; font-size: 2.5em; font-weight: 700; }}
        .header p {{ margin: 0; opacity: 0.9; font-size: 1.1em; }}
        .status-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); transition: transform 0.2s ease; }}
        .metric-card:hover {{ transform: translateY(-2px); }}
        .metric-card h2 {{ margin: 0 0 15px 0; color: #2c3e50; font-size: 1.4em; }}
        .metric-card h3 {{ margin: 0 0 10px 0; color: #57606f; font-size: 0.9em; text-transform: uppercase; letter-spacing: 0.5px; }}
        .status-value {{ font-size: 2.2em; font-weight: 700; margin-bottom: 8px; }}
        .status-description {{ color: #747d8c; font-size: 0.9em; line-height: 1.4; }}
        .critical {{ border-left: 5px solid #e74c3c; }}
        .warning {{ border-left: 5px solid #f39c12; }}
        .success {{ border-left: 5px solid #27ae60; }}
        .info {{ border-left: 5px solid #3498db; }}
        .recommendation {{ background: #fff3cd; padding: 20px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #f39c12; }}
        .test-results {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e1e8ed; }}
        th {{ background-color: #2c3e50; color: white; font-weight: 600; }}
        .progress-bar {{ width: 100%; background: #e1e8ed; border-radius: 6px; height: 8px; overflow: hidden; margin: 10px 0; }}
        .progress-fill {{ height: 100%; background: linear-gradient(90deg, #27ae60, #2ecc71); transition: width 0.3s ease; }}
        .test-status-pass {{ color: #27ae60; font-weight: 600; }}
        .test-status-fail {{ color: #e74c3c; font-weight: 600; }}
        .test-status-warn {{ color: #f39c12; font-weight: 600; }}
        .execution-time {{ font-size: 0.9em; color: #7f8c8d; }}
        .collapsible {{ cursor: pointer; padding: 10px; background: #ecf0f1; border: none; text-align: left; width: 100%; font-size: 1em; }}
        .collapsible:hover {{ background: #d5dbdb; }}
        .content {{ display: none; padding: 15px; background: #f8f9fa; border-left: 3px solid #3498db; }}
        .content pre {{ background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; font-size: 0.9em; }}
    </style>
    <script>
        function toggleContent(id) {{
            var content = document.getElementById(id);
            content.style.display = content.style.display === 'block' ? 'none' : 'block';
        }}

        function refreshDashboard() {{
            location.reload();
        }}

        setTimeout(refreshDashboard, 300000); // Auto-refresh every 5 minutes
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Sprint 6 Test Dashboard</h1>
            <p>Real-time monitoring for P2P, Resource Management, and Evolution Infrastructure • Generated: {dashboard_data["timestamp"]}</p>
        </div>

        <div class="status-grid">
            <div class="metric-card {sprint6_class}">
                <h3>Sprint 6 Infrastructure Status</h3>
                <div class="status-value">{sprint6_status}</div>
                <div class="status-description">Overall system health assessment</div>
            </div>

            <div class="metric-card info">
                <h3>Components Analyzed</h3>
                <div class="status-value">{len(self.critical_components)}</div>
                <div class="status-description">Critical system categories monitored</div>
            </div>
        </div>"""

        # Add Sprint 6 test results sections
        if "sprint6_validation" in dashboard_data:
            validation = dashboard_data["sprint6_validation"]
            validation_status = "✅ Passed" if validation.get("success", False) else "❌ Failed"
            validation_class = "success" if validation.get("success", False) else "critical"

            html_content += f"""
        <div class="test-results {validation_class}">
            <h2>🔧 Sprint 6 Validation Results</h2>
            <p><strong>Status:</strong> <span class="test-status-{"pass" if validation.get("success", False) else "fail"}">{validation_status}</span></p>
            <p><strong>Execution Time:</strong> <span class="execution-time">{validation.get("execution_time", 0):.2f} seconds</span></p>

            <button class="collapsible" onclick="toggleContent('validation-details')">View Validation Output</button>
            <div id="validation-details" class="content">
                <h4>Standard Output:</h4>
                <pre>{validation.get("stdout", "No output")}</pre>
                {f"<h4>Error Output:</h4><pre>{validation.get('stderr', '')}</pre>" if validation.get("stderr") else ""}
            </div>
        </div>"""

        if "sprint6_infrastructure" in dashboard_data:
            infra = dashboard_data["sprint6_infrastructure"]

            html_content += f"""
        <div class="test-results">
            <h2>🏗️ Sprint 6 Infrastructure Tests</h2>
            <div class="status-grid">
                <div class="metric-card success">
                    <h3>Passed Tests</h3>
                    <div class="status-value test-status-pass">{infra.get("passed_files", 0)}</div>
                </div>
                <div class="metric-card {"critical" if infra.get("failed_files", 0) > 0 else "info"}">
                    <h3>Failed Tests</h3>
                    <div class="status-value test-status-{"fail" if infra.get("failed_files", 0) > 0 else "pass"}">{infra.get("failed_files", 0)}</div>
                </div>
                <div class="metric-card info">
                    <h3>Total Files</h3>
                    <div class="status-value">{infra.get("total_files", 0)}</div>
                </div>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {(infra.get("passed_files", 0) / max(infra.get("total_files", 1), 1)) * 100:.1f}%"></div>
            </div>

            <table>
                <tr><th>Test File</th><th>Status</th><th>Execution Time</th><th>Details</th></tr>"""

            for test_file, result in infra.get("results", {}).items():
                status_icon = "✅" if result.get("success", False) else "❌"
                status_class = "test-status-pass" if result.get("success", False) else "test-status-fail"
                exec_time = result.get("execution_time", 0)

                html_content += f"""
                <tr>
                    <td>{test_file.split("/")[-1]}</td>
                    <td class="{status_class}">{status_icon}</td>
                    <td class="execution-time">{exec_time:.2f}s</td>
                    <td><button class="collapsible" onclick="toggleContent('{test_file.replace("/", "-").replace(".", "-")}')">View</button></td>
                </tr>"""

            html_content += "</table>"

            # Add collapsible details for each test
            for test_file, result in infra.get("results", {}).items():
                content_id = test_file.replace("/", "-").replace(".", "-")
                html_content += f"""
            <div id="{content_id}" class="content">
                <h4>{test_file}</h4>
                <p><strong>Status:</strong> {"Success" if result.get("success", False) else "Failed"}</p>
                <p><strong>Execution Time:</strong> {result.get("execution_time", 0):.2f} seconds</p>
                {f"<p><strong>Error:</strong> {result.get('error', '')}</p>" if result.get("error") else ""}
                {f"<h5>Output:</h5><pre>{result.get('stdout', 'No output')}</pre>" if result.get("stdout") else ""}
                {f"<h5>Errors:</h5><pre>{result.get('stderr', '')}</pre>" if result.get("stderr") else ""}
            </div>"""

            html_content += "</div>"

        if "sprint6_performance" in dashboard_data:
            perf = dashboard_data["sprint6_performance"]
            perf_status = "✅ Passed" if perf.get("success", False) else "❌ Failed"
            perf_class = "success" if perf.get("success", False) else "warning"

            html_content += f"""
        <div class="test-results {perf_class}">
            <h2>⚡ Sprint 6 Performance Tests</h2>
            <p><strong>Status:</strong> <span class="test-status-{"pass" if perf.get("success", False) else "fail"}">{perf_status}</span></p>
            <p><strong>Execution Time:</strong> <span class="execution-time">{perf.get("execution_time", 0):.2f} seconds</span></p>

            <button class="collapsible" onclick="toggleContent('performance-details')">View Performance Results</button>
            <div id="performance-details" class="content">
                <h4>Benchmark Output:</h4>
                <pre>{perf.get("stdout", "No output")}</pre>
                {f"<h4>Error Output:</h4><pre>{perf.get('stderr', '')}</pre>" if perf.get("stderr") else ""}
            </div>
        </div>"""

        html_content += """
        <div class="test-results">
            <h2>🎯 Critical Component Coverage</h2>
            <table>
            <tr><th>Component Category</th><th>Files</th><th>Avg Coverage</th><th>Status</th></tr>
"""

        for category, metrics in dashboard_data["component_coverage"].items():
            coverage = metrics["average_coverage"]
            status = "🟢 Good" if coverage > 80 else "🟡 Medium" if coverage > 50 else "🔴 Needs Work"

            html_content += f"""
            <tr>
                <td>{category.replace("_", " ").title()}</td>
                <td>{metrics["total_files"]}</td>
                <td>{coverage:.1f}%</td>
                <td>{status}</td>
            </tr>
"""

        html_content += """
        </table>
    </div>

    <div class="metric-card critical">
        <h2>⚠️ Priority Recommendations</h2>
"""

        for rec in dashboard_data["recommendations"][:3]:
            html_content += f"""
        <div class="recommendation">
            <h3>{rec["title"]}</h3>
            <p>{rec["description"]}</p>
            <p><strong>Priority:</strong> {rec["priority"].upper()}</p>
            <p><strong>Estimated Effort:</strong> {rec["estimated_effort"]}</p>
            <p><strong>Impact:</strong> {rec["impact"]}</p>
        </div>
"""

        html_content += (
            """
    </div>

    <div class="metric-card">
        <h2>📈 Test Suite Health</h2>
        <p><strong>Total Test Files:</strong> """
            + str(dashboard_data["test_suite_health"]["total_test_files"])
            + """</p>
        <p><strong>Test Distribution:</strong></p>
        <ul>"""
        )

        for category, count in dashboard_data["test_suite_health"]["test_distribution"].items():
            html_content += f"<li>{category}: {count} files</li>"

        html_content += """
        </ul>
    </div>

    <div class="metric-card">
        <h2>🚀 Next Steps</h2>
        <ol>
            <li>Focus on production-critical components first</li>
            <li>Create MCP server integration tests</li>
            <li>Add compression pipeline comprehensive tests</li>
            <li>Implement end-to-end workflow tests</li>
            <li>Set up automated coverage monitoring</li>
        </ol>
    </div>

    <div class="metric-card">
        <h2>📋 Quick Actions</h2>
        <p>To improve coverage immediately:</p>
        <ul>
            <li>Run: <code>python -m pytest --cov=. --cov-report=html</code></li>
            <li>View detailed report: <code>open htmlcov/index.html</code></li>
            <li>Create missing test files for critical components</li>
            <li>Add integration tests for multi-component workflows</li>
        </ul>
    </div>
</body>
</html>
"""

        return html_content

    def save_dashboard(self, dashboard_data: dict, output_dir: Path | None = None) -> dict[str, str]:
        """Save dashboard data and reports."""
        if output_dir is None:
            output_dir = self.project_root / "coverage_reports"

        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON data
        json_file = output_dir / f"coverage_analysis_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(dashboard_data, f, indent=2)

        # Save HTML dashboard
        html_content = self.generate_html_dashboard(dashboard_data)
        html_file = output_dir / f"coverage_dashboard_{timestamp}.html"
        with open(html_file, "w") as f:
            f.write(html_content)

        # Save markdown summary
        md_content = self._generate_markdown_summary(dashboard_data)
        md_file = output_dir / f"coverage_summary_{timestamp}.md"
        with open(md_file, "w") as f:
            f.write(md_content)

        return {
            "json_report": str(json_file),
            "html_dashboard": str(html_file),
            "markdown_summary": str(md_file),
        }

    def _generate_markdown_summary(self, dashboard_data: dict) -> str:
        """Generate markdown summary report."""
        return (
            f"""# Test Coverage Analysis Summary

Generated: {dashboard_data["timestamp"]}

## Overview
- **Analysis Status**: Complete
- **Components Analyzed**: {len(self.critical_components)} categories
- **Critical Untested Files**: {len(dashboard_data["coverage_gaps"]["critical_untested"])}

## Component Coverage

| Category | Files | Avg Coverage | Status |
|----------|-------|--------------|--------|
"""
            + "\n".join(
                [
                    f"| {cat.replace('_', ' ').title()} | {metrics['total_files']} | "
                    f"{metrics['average_coverage']:.1f}% | "
                    f"{'🟢' if metrics['average_coverage'] > 80 else '🟡' if metrics['average_coverage'] > 50 else '🔴'} |"
                    for cat, metrics in dashboard_data["component_coverage"].items()
                ]
            )
            + """

## Priority Recommendations

"""
            + "\n".join(
                [
                    f"### {rec['priority'].upper()}: {rec['title']}\n{rec['description']}\n**Effort**: {rec['estimated_effort']}\n"
                    for rec in dashboard_data["recommendations"][:3]
                ]
            )
            + """

## Action Items
1. Create tests for production-critical components
2. Add MCP server integration tests
3. Implement compression pipeline tests
4. Set up automated coverage monitoring
5. Create integration test framework

## Quick Start
```bash
# Run coverage analysis
python -m pytest --cov=. --cov-report=html

# View results
open htmlcov/index.html

# Create new test files
python test_coverage_dashboard.py
```
"""
        )


def main():
    """Main execution function."""
    print("🚀 Initializing AIVillage Test Coverage Dashboard")

    dashboard = TestCoverageDashboard()

    # Run comprehensive analysis
    print("📊 Running coverage analysis...")
    dashboard_data = dashboard.run_coverage_analysis()

    # Save reports
    print("💾 Saving dashboard reports...")
    saved_files = dashboard.save_dashboard(dashboard_data)

    print("✅ Coverage analysis complete!")
    print("📄 Reports saved:")
    for report_type, file_path in saved_files.items():
        print(f"   {report_type}: {file_path}")

    # Print summary
    print("\n📋 Quick Summary:")
    print(f"   - Critical untested files: {len(dashboard_data['coverage_gaps']['critical_untested'])}")
    print(f"   - Total test files: {dashboard_data['test_suite_health']['total_test_files']}")
    print(f"   - Priority recommendations: {len(dashboard_data['recommendations'])}")

    # Show top priority files
    untested = dashboard_data["coverage_gaps"]["critical_untested"]
    if untested:
        print("\n🎯 Top priority files needing tests:")
        for i, gap in enumerate(untested[:5], 1):
            print(f"   {i}. {gap['file']} ({gap['category']})")

    print(f"\n🌐 Open dashboard: {saved_files['html_dashboard']}")


if __name__ == "__main__":
    main()
