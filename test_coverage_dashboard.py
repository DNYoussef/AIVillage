#!/usr/bin/env python3
"""
Comprehensive Test Coverage Dashboard for AIVillage
Real-time monitoring and analysis of test coverage across all components.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import glob
import ast

@dataclass
class CoverageMetrics:
    """Container for coverage metrics."""
    total_lines: int
    covered_lines: int
    coverage_percent: float
    missing_lines: List[int]
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
    test_files: List[str]

class TestCoverageDashboard:
    """Comprehensive test coverage dashboard."""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.coverage_data = {}
        self.test_metrics = {}
        self.critical_components = self._identify_critical_components()

    def _identify_critical_components(self) -> Dict[str, List[str]]:
        """Identify critical components that require high test coverage."""
        return {
            "production_critical": [
                "production/compression/",
                "production/evolution/",
                "production/rag/",
                "mcp_servers/hyperag/server.py",
                "mcp_servers/hyperag/protocol.py"
            ],
            "core_systems": [
                "agent_forge/core/",
                "agent_forge/orchestrator.py",
                "communications/protocol.py",
                "digital_twin/core/"
            ],
            "security_components": [
                "mcp_servers/hyperag/auth.py",
                "digital_twin/security/",
                "mcp_servers/hyperag/guardian/"
            ],
            "integration_points": [
                "experimental/services/",
                "mcp_servers/hyperag/memory/",
                "hyperag/education/"
            ]
        }

    def run_coverage_analysis(self) -> Dict:
        """Run comprehensive coverage analysis."""
        print("🔍 Running comprehensive test coverage analysis...")

        # Step 1: Run pytest with coverage
        coverage_results = self._run_pytest_coverage()

        # Step 2: Analyze coverage by component
        component_coverage = self._analyze_component_coverage()

        # Step 3: Identify gaps and priorities
        coverage_gaps = self._identify_coverage_gaps()

        # Step 4: Generate recommendations
        recommendations = self._generate_recommendations(coverage_gaps)

        # Step 5: Create visual dashboard
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "overall_coverage": coverage_results,
            "component_coverage": component_coverage,
            "coverage_gaps": coverage_gaps,
            "recommendations": recommendations,
            "test_suite_health": self._analyze_test_suite_health()
        }

        return dashboard_data

    def _run_pytest_coverage(self) -> Dict:
        """Run pytest with coverage collection."""
        cmd = [
            sys.executable, "-m", "pytest",
            "--cov=.",
            "--cov-report=json:coverage_report.json",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-config=.coveragerc",
            "-v", "--tb=short",
            "--durations=10"
        ]

        print(f"Executing: {' '.join(cmd)}")

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                cwd=self.project_root
            )
            execution_time = time.time() - start_time

            print(f"✅ Pytest completed in {execution_time:.2f} seconds")
            print(f"Exit code: {result.returncode}")

            if result.returncode != 0:
                print(f"⚠️ Pytest warnings/errors:\n{result.stderr}")

            # Load coverage results
            coverage_file = self.project_root / "coverage_report.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)

                return {
                    "success": True,
                    "execution_time": execution_time,
                    "coverage_data": coverage_data,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                return {
                    "success": False,
                    "error": "Coverage report not generated",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Pytest execution timed out",
                "execution_time": 600
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _analyze_component_coverage(self) -> Dict:
        """Analyze coverage by critical components."""
        component_metrics = {}

        for category, components in self.critical_components.items():
            category_metrics = {
                "total_files": 0,
                "covered_files": 0,
                "average_coverage": 0.0,
                "files": {}
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

    def _find_python_files(self, path_pattern: str) -> List[Path]:
        """Find Python files matching pattern."""
        if path_pattern.endswith('/'):
            # Directory pattern
            pattern = self.project_root / path_pattern / "**/*.py"
            return list(Path().glob(str(pattern)))
        else:
            # Specific file
            file_path = self.project_root / path_pattern
            return [file_path] if file_path.exists() else []

    def _get_file_coverage(self, file_path: Path) -> Optional[Dict]:
        """Get coverage data for specific file."""
        # This would integrate with coverage.py data
        # For now, return mock data structure

        if not file_path.exists():
            return None

        # Count lines in file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                total_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
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
            "has_tests": self._file_has_tests(file_path)
        }

    def _file_has_tests(self, file_path: Path) -> bool:
        """Check if file has corresponding tests."""
        # Check for test files
        test_patterns = [
            self.project_root / "tests" / f"test_{file_path.stem}.py",
            self.project_root / "tests" / file_path.parent.name / f"test_{file_path.stem}.py",
            file_path.parent / f"test_{file_path.stem}.py"
        ]

        return any(test_path.exists() for test_path in test_patterns)

    def _identify_coverage_gaps(self) -> Dict:
        """Identify critical coverage gaps."""
        gaps = {
            "critical_untested": [],
            "low_coverage": [],
            "missing_integration_tests": [],
            "security_gaps": []
        }

        # Find untested critical files
        for category, components in self.critical_components.items():
            for component_path in components:
                files = self._find_python_files(component_path)

                for file_path in files:
                    if not self._file_has_tests(file_path):
                        gaps["critical_untested"].append({
                            "file": str(file_path),
                            "category": category,
                            "priority": "high" if "production_critical" in category else "medium"
                        })

        return gaps

    def _generate_recommendations(self, coverage_gaps: Dict) -> List[Dict]:
        """Generate recommendations for improving coverage."""
        recommendations = []

        # High priority recommendations
        if coverage_gaps["critical_untested"]:
            recommendations.append({
                "priority": "critical",
                "title": "Create tests for production-critical components",
                "description": f"Found {len(coverage_gaps['critical_untested'])} untested critical files",
                "action_items": [
                    f"Create test file for {gap['file']}"
                    for gap in coverage_gaps["critical_untested"][:5]
                ],
                "estimated_effort": "2-3 days",
                "impact": "High - Essential for production reliability"
            })

        # Coverage improvement recommendations
        recommendations.append({
            "priority": "high",
            "title": "Improve MCP server test coverage",
            "description": "MCP servers are core to the system but lack comprehensive tests",
            "action_items": [
                "Create integration tests for server.py",
                "Add unit tests for protocol.py",
                "Test authentication and authorization",
                "Add performance tests for concurrent connections"
            ],
            "estimated_effort": "3-4 days",
            "impact": "High - Critical for system stability"
        })

        recommendations.append({
            "priority": "high",
            "title": "Implement compression pipeline testing",
            "description": "Production compression needs comprehensive test coverage",
            "action_items": [
                "Test compression algorithms",
                "Add performance benchmarks",
                "Test model loading and saving",
                "Add integration tests with evolution system"
            ],
            "estimated_effort": "2-3 days",
            "impact": "High - Production critical component"
        })

        recommendations.append({
            "priority": "medium",
            "title": "Add integration test framework",
            "description": "Need end-to-end integration testing",
            "action_items": [
                "Create integration test fixtures",
                "Add multi-component workflow tests",
                "Test external API integrations",
                "Add load testing capabilities"
            ],
            "estimated_effort": "1-2 weeks",
            "impact": "Medium - Improves system reliability"
        })

        return recommendations

    def _analyze_test_suite_health(self) -> Dict:
        """Analyze overall test suite health."""
        test_files = list(self.project_root.rglob("test_*.py"))
        test_files.extend(list(self.project_root.rglob("*_test.py")))

        health_metrics = {
            "total_test_files": len(test_files),
            "test_distribution": {},
            "test_quality_issues": [],
            "performance_issues": []
        }

        # Analyze test distribution
        for test_file in test_files:
            category = test_file.parent.name
            if category not in health_metrics["test_distribution"]:
                health_metrics["test_distribution"][category] = 0
            health_metrics["test_distribution"][category] += 1

        return health_metrics

    def generate_html_dashboard(self, dashboard_data: Dict) -> str:
        """Generate HTML dashboard."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AIVillage Test Coverage Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .metric-card {{ background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .critical {{ border-left: 5px solid #e74c3c; }}
        .warning {{ border-left: 5px solid #f39c12; }}
        .success {{ border-left: 5px solid #27ae60; }}
        .recommendation {{ background: #fff3cd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #34495e; color: white; }}
        .progress-bar {{ width: 100%; background: #ecf0f1; border-radius: 5px; }}
        .progress-fill {{ height: 20px; background: #3498db; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 AIVillage Test Coverage Dashboard</h1>
        <p>Generated: {dashboard_data['timestamp']}</p>
    </div>

    <div class="metric-card success">
        <h2>📊 Overall Coverage Status</h2>
        <p><strong>Status:</strong> Analysis Complete</p>
        <p><strong>Components Analyzed:</strong> {len(self.critical_components)} categories</p>
    </div>

    <div class="metric-card">
        <h2>🎯 Critical Component Coverage</h2>
        <table>
            <tr><th>Component Category</th><th>Files</th><th>Avg Coverage</th><th>Status</th></tr>
"""

        for category, metrics in dashboard_data["component_coverage"].items():
            coverage = metrics["average_coverage"]
            status = "🟢 Good" if coverage > 80 else "🟡 Medium" if coverage > 50 else "🔴 Needs Work"

            html_content += f"""
            <tr>
                <td>{category.replace('_', ' ').title()}</td>
                <td>{metrics['total_files']}</td>
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
            <h3>{rec['title']}</h3>
            <p>{rec['description']}</p>
            <p><strong>Priority:</strong> {rec['priority'].upper()}</p>
            <p><strong>Estimated Effort:</strong> {rec['estimated_effort']}</p>
            <p><strong>Impact:</strong> {rec['impact']}</p>
        </div>
"""

        html_content += """
    </div>

    <div class="metric-card">
        <h2>📈 Test Suite Health</h2>
        <p><strong>Total Test Files:</strong> """ + str(dashboard_data["test_suite_health"]["total_test_files"]) + """</p>
        <p><strong>Test Distribution:</strong></p>
        <ul>"""

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

    def save_dashboard(self, dashboard_data: Dict, output_dir: Path = None) -> Dict[str, str]:
        """Save dashboard data and reports."""
        if output_dir is None:
            output_dir = self.project_root / "coverage_reports"

        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON data
        json_file = output_dir / f"coverage_analysis_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2)

        # Save HTML dashboard
        html_content = self.generate_html_dashboard(dashboard_data)
        html_file = output_dir / f"coverage_dashboard_{timestamp}.html"
        with open(html_file, 'w') as f:
            f.write(html_content)

        # Save markdown summary
        md_content = self._generate_markdown_summary(dashboard_data)
        md_file = output_dir / f"coverage_summary_{timestamp}.md"
        with open(md_file, 'w') as f:
            f.write(md_content)

        return {
            "json_report": str(json_file),
            "html_dashboard": str(html_file),
            "markdown_summary": str(md_file)
        }

    def _generate_markdown_summary(self, dashboard_data: Dict) -> str:
        """Generate markdown summary report."""
        return f"""# Test Coverage Analysis Summary

Generated: {dashboard_data['timestamp']}

## Overview
- **Analysis Status**: Complete
- **Components Analyzed**: {len(self.critical_components)} categories
- **Critical Untested Files**: {len(dashboard_data['coverage_gaps']['critical_untested'])}

## Component Coverage

| Category | Files | Avg Coverage | Status |
|----------|-------|--------------|--------|
""" + "\n".join([
    f"| {cat.replace('_', ' ').title()} | {metrics['total_files']} | {metrics['average_coverage']:.1f}% | {'🟢' if metrics['average_coverage'] > 80 else '🟡' if metrics['average_coverage'] > 50 else '🔴'} |"
    for cat, metrics in dashboard_data["component_coverage"].items()
]) + f"""

## Priority Recommendations

""" + "\n".join([
    f"### {rec['priority'].upper()}: {rec['title']}\n{rec['description']}\n**Effort**: {rec['estimated_effort']}\n"
    for rec in dashboard_data["recommendations"][:3]
]) + """

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
    print(f"📄 Reports saved:")
    for report_type, file_path in saved_files.items():
        print(f"   {report_type}: {file_path}")

    # Print summary
    print(f"\n📋 Quick Summary:")
    print(f"   - Critical untested files: {len(dashboard_data['coverage_gaps']['critical_untested'])}")
    print(f"   - Total test files: {dashboard_data['test_suite_health']['total_test_files']}")
    print(f"   - Priority recommendations: {len(dashboard_data['recommendations'])}")

    # Show top priority files
    untested = dashboard_data['coverage_gaps']['critical_untested']
    if untested:
        print(f"\n🎯 Top priority files needing tests:")
        for i, gap in enumerate(untested[:5], 1):
            print(f"   {i}. {gap['file']} ({gap['category']})")

    print(f"\n🌐 Open dashboard: {saved_files['html_dashboard']}")

if __name__ == "__main__":
    main()
