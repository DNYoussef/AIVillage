#!/usr/bin/env python3
"""
CI/CD Integration Script for Architectural Fitness Functions

This script integrates architectural fitness functions into CI/CD pipelines,
providing automated architecture validation and reporting.

Usage:
    # Run fitness functions as part of CI
    python scripts/ci_integration.py --mode ci

    # Generate pre-commit hooks
    python scripts/ci_integration.py --mode pre-commit

    # Generate GitHub Actions workflow
    python scripts/ci_integration.py --mode github-actions

    # Run with specific quality gate thresholds
    python scripts/ci_integration.py --mode ci --fail-on-drift --max-debt-ratio 10
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

PROJECT_ROOT = Path(__file__).parent.parent


class CIIntegration:
    """CI/CD integration for architectural fitness functions"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.scripts_dir = project_root / "scripts"
        self.config_dir = project_root / "config"
        self.tests_dir = project_root / "tests" / "architecture"

    def run_fitness_functions(self, fail_fast: bool = False) -> tuple[bool, dict]:
        """Run all architectural fitness functions"""
        print("Running architectural fitness functions...")

        # Run pytest on architecture tests
        cmd = [sys.executable, "-m", "pytest", str(self.tests_dir / "test_fitness_functions.py"), "-v", "--tb=short"]

        if fail_fast:
            cmd.append("-x")

        # Add JSON report generation
        report_file = tempfile.mktemp(suffix=".json")
        cmd.extend(["--json-report", f"--json-report-file={report_file}"])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

            # Load test results
            test_results = {}
            if Path(report_file).exists():
                with open(report_file) as f:
                    test_results = json.load(f)
                os.unlink(report_file)

            success = result.returncode == 0

            return success, {
                "success": success,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "test_results": test_results,
            }

        except Exception as e:
            return False, {"success": False, "error": str(e), "stdout": "", "stderr": "", "test_results": {}}

    def run_architectural_analysis(self) -> tuple[bool, dict]:
        """Run comprehensive architectural analysis"""
        print("Running comprehensive architectural analysis...")

        cmd = [
            sys.executable,
            str(self.scripts_dir / "architectural_analysis.py"),
            "--output-dir",
            "reports/architecture",
            "--format",
            "json",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

            success = result.returncode == 0

            return success, {
                "success": success,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except Exception as e:
            return False, {"success": False, "error": str(e), "stdout": "", "stderr": ""}

    def run_ci_pipeline(self, config: dict) -> dict:
        """Run complete CI pipeline for architecture validation"""
        print("=== AIVillage Architecture CI Pipeline ===")

        pipeline_results = {"timestamp": datetime.now().isoformat(), "overall_success": True, "stages": {}}

        # Stage 1: Fitness Functions
        print("\n📋 Stage 1: Running Fitness Functions...")
        fitness_success, fitness_results = self.run_fitness_functions(fail_fast=config.get("fail_fast", False))

        pipeline_results["stages"]["fitness_functions"] = fitness_results
        pipeline_results["overall_success"] &= fitness_success

        if not fitness_success and config.get("fail_fast", False):
            print("❌ Fitness functions failed, stopping pipeline")
            return pipeline_results

        # Stage 2: Architectural Analysis
        print("\n🔍 Stage 2: Running Architectural Analysis...")
        analysis_success, analysis_results = self.run_architectural_analysis()

        pipeline_results["stages"]["architectural_analysis"] = analysis_results
        pipeline_results["overall_success"] &= analysis_success

        # Stage 3: Quality Gate Evaluation
        print("\n🚪 Stage 3: Evaluating Quality Gates...")
        gates_success, gates_results = self.evaluate_quality_gates(config)

        pipeline_results["stages"]["quality_gates"] = gates_results
        pipeline_results["overall_success"] &= gates_success

        # Stage 4: Generate Reports
        print("\n📊 Stage 4: Generating Reports...")
        self.generate_ci_reports(pipeline_results)

        # Print summary
        self.print_pipeline_summary(pipeline_results)

        return pipeline_results

    def evaluate_quality_gates(self, config: dict) -> tuple[bool, dict]:
        """Evaluate quality gates based on configuration"""
        gates_config = config.get("quality_gates", {})
        results = {"gates_evaluated": [], "gates_passed": 0, "gates_failed": 0, "overall_pass": True}

        # Load latest architectural analysis report
        reports_dir = self.project_root / "reports" / "architecture"
        if not reports_dir.exists():
            return False, {"error": "No architectural analysis reports found", "overall_pass": False}

        # Find most recent report
        json_reports = list(reports_dir.glob("architecture_report_*.json"))
        if not json_reports:
            return False, {"error": "No JSON reports found", "overall_pass": False}

        latest_report_path = max(json_reports, key=lambda p: p.stat().st_mtime)

        try:
            with open(latest_report_path) as f:
                report_data = json.load(f)
        except Exception as e:
            return False, {"error": f"Could not load report: {e}", "overall_pass": False}

        # Evaluate each quality gate
        quality_gates = report_data.get("quality_gates", {})

        for gate_name, passed in quality_gates.items():
            gate_result = {"name": gate_name, "passed": passed, "required": gates_config.get(gate_name, True)}

            results["gates_evaluated"].append(gate_result)

            if passed:
                results["gates_passed"] += 1
            else:
                results["gates_failed"] += 1
                if gate_result["required"]:
                    results["overall_pass"] = False

        # Additional custom gates
        summary = report_data.get("summary", {})

        # Technical debt ratio gate
        max_debt_ratio = config.get("max_debt_ratio", 10)
        debt_items = len(report_data.get("technical_debt", []))
        debt_gate_passed = debt_items <= max_debt_ratio

        results["gates_evaluated"].append(
            {
                "name": "technical_debt_acceptable",
                "passed": debt_gate_passed,
                "required": True,
                "details": f"{debt_items} debt items (max: {max_debt_ratio})",
            }
        )

        if debt_gate_passed:
            results["gates_passed"] += 1
        else:
            results["gates_failed"] += 1
            results["overall_pass"] = False

        # Critical violations gate
        critical_violations = summary.get("critical_violations", 0)
        max_critical = config.get("max_critical_violations", 0)
        critical_gate_passed = critical_violations <= max_critical

        results["gates_evaluated"].append(
            {
                "name": "no_critical_violations",
                "passed": critical_gate_passed,
                "required": True,
                "details": f"{critical_violations} critical violations (max: {max_critical})",
            }
        )

        if critical_gate_passed:
            results["gates_passed"] += 1
        else:
            results["gates_failed"] += 1
            results["overall_pass"] = False

        return results["overall_pass"], results

    def generate_ci_reports(self, pipeline_results: dict):
        """Generate CI-specific reports"""
        reports_dir = self.project_root / "reports" / "ci"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Generate JSON report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_report_path = reports_dir / f"ci_pipeline_{timestamp}.json"

        with open(json_report_path, "w") as f:
            json.dump(pipeline_results, f, indent=2, default=str)

        # Generate JUnit XML report for CI systems
        self.generate_junit_report(pipeline_results, reports_dir / f"ci_results_{timestamp}.xml")

        # Generate badge data for README
        self.generate_badge_data(pipeline_results, reports_dir / "badges.json")

    def generate_junit_report(self, pipeline_results: dict, output_path: Path):
        """Generate JUnit XML report for CI integration"""
        from xml.dom import minidom
        from xml.etree.ElementTree import Element, SubElement, tostring

        testsuites = Element("testsuites")
        testsuites.set("name", "Architecture Fitness Functions")
        testsuites.set("time", "0")  # Would calculate actual time

        total_tests = 0
        total_failures = 0

        # Add fitness functions results
        if "fitness_functions" in pipeline_results["stages"]:
            fitness_results = pipeline_results["stages"]["fitness_functions"]
            test_results = fitness_results.get("test_results", {})

            if "tests" in test_results:
                testsuite = SubElement(testsuites, "testsuite")
                testsuite.set("name", "Fitness Functions")
                testsuite.set("tests", str(len(test_results["tests"])))

                for test in test_results["tests"]:
                    testcase = SubElement(testsuite, "testcase")
                    testcase.set("classname", test.get("nodeid", "").split("::")[0])
                    testcase.set("name", test.get("nodeid", "").split("::")[-1])

                    if test.get("outcome") == "failed":
                        failure = SubElement(testcase, "failure")
                        failure.set("message", test.get("call", {}).get("longrepr", ""))
                        total_failures += 1

                    total_tests += 1

        # Add quality gates results
        if "quality_gates" in pipeline_results["stages"]:
            gates_results = pipeline_results["stages"]["quality_gates"]

            testsuite = SubElement(testsuites, "testsuite")
            testsuite.set("name", "Quality Gates")

            gates_evaluated = gates_results.get("gates_evaluated", [])
            testsuite.set("tests", str(len(gates_evaluated)))

            for gate in gates_evaluated:
                testcase = SubElement(testsuite, "testcase")
                testcase.set("classname", "QualityGates")
                testcase.set("name", gate["name"])

                if not gate["passed"]:
                    failure = SubElement(testcase, "failure")
                    failure.set("message", f"Quality gate failed: {gate.get('details', '')}")
                    total_failures += 1

                total_tests += 1

        testsuites.set("tests", str(total_tests))
        testsuites.set("failures", str(total_failures))

        # Write XML file
        rough_string = tostring(testsuites, "unicode")
        reparsed = minidom.parseString(rough_string)

        with open(output_path, "w") as f:
            f.write(reparsed.toprettyxml(indent="  "))

    def generate_badge_data(self, pipeline_results: dict, output_path: Path):
        """Generate badge data for GitHub README"""
        overall_success = pipeline_results["overall_success"]

        # Architecture badge
        arch_badge = {
            "schemaVersion": 1,
            "label": "architecture",
            "message": "healthy" if overall_success else "issues",
            "color": "brightgreen" if overall_success else "red",
        }

        # Quality gates badge
        gates_results = pipeline_results.get("stages", {}).get("quality_gates", {})
        gates_passed = gates_results.get("gates_passed", 0)
        gates_total = gates_passed + gates_results.get("gates_failed", 0)

        if gates_total > 0:
            gates_percentage = int((gates_passed / gates_total) * 100)
            gates_badge = {
                "schemaVersion": 1,
                "label": "quality gates",
                "message": f"{gates_passed}/{gates_total} ({gates_percentage}%)",
                "color": "brightgreen" if gates_percentage == 100 else "yellow" if gates_percentage >= 80 else "red",
            }
        else:
            gates_badge = {"schemaVersion": 1, "label": "quality gates", "message": "unknown", "color": "lightgrey"}

        badge_data = {
            "architecture": arch_badge,
            "quality_gates": gates_badge,
            "timestamp": pipeline_results["timestamp"],
        }

        with open(output_path, "w") as f:
            json.dump(badge_data, f, indent=2)

    def print_pipeline_summary(self, pipeline_results: dict):
        """Print pipeline execution summary"""
        print("\n" + "=" * 60)
        print("🏗️  ARCHITECTURE CI PIPELINE SUMMARY")
        print("=" * 60)

        overall_success = pipeline_results["overall_success"]
        status_icon = "✅" if overall_success else "❌"
        status_text = "PASSED" if overall_success else "FAILED"

        print(f"\nOverall Result: {status_icon} {status_text}")
        print(f"Timestamp: {pipeline_results['timestamp']}")

        print("\n📋 Stage Results:")
        for stage_name, stage_results in pipeline_results["stages"].items():
            stage_success = stage_results.get("success", False)
            stage_icon = "✅" if stage_success else "❌"
            print(f"  {stage_icon} {stage_name.replace('_', ' ').title()}")

            # Print specific details for quality gates
            if stage_name == "quality_gates" and "gates_evaluated" in stage_results:
                gates = stage_results["gates_evaluated"]
                passed_count = sum(1 for g in gates if g["passed"])
                total_count = len(gates)
                print(f"    Quality Gates: {passed_count}/{total_count} passed")

                for gate in gates:
                    if not gate["passed"]:
                        gate_icon = "❌"
                        details = f" - {gate.get('details', '')}" if gate.get("details") else ""
                        print(f"    {gate_icon} {gate['name']}{details}")

        print("\n" + "=" * 60)

    def generate_pre_commit_hooks(self):
        """Generate pre-commit hooks configuration"""
        hooks_config = {
            "repos": [
                {
                    "repo": "local",
                    "hooks": [
                        {
                            "id": "architecture-fitness-functions",
                            "name": "Architecture Fitness Functions",
                            "entry": f"{sys.executable} -m pytest tests/architecture/test_fitness_functions.py -x",
                            "language": "system",
                            "types": ["python"],
                            "pass_filenames": False,
                        },
                        {
                            "id": "architecture-analysis",
                            "name": "Architecture Analysis",
                            "entry": f"{sys.executable} scripts/architectural_analysis.py --format json",
                            "language": "system",
                            "types": ["python"],
                            "pass_filenames": False,
                        },
                    ],
                }
            ]
        }

        output_path = self.project_root / ".pre-commit-config.yaml"
        with open(output_path, "w") as f:
            yaml.dump(hooks_config, f, default_flow_style=False)

        print(f"Pre-commit configuration written to: {output_path}")

    def generate_github_actions_workflow(self):
        """Generate GitHub Actions workflow"""
        workflow = {
            "name": "Architecture Quality Assurance",
            "on": {"push": {"branches": ["main", "develop"]}, "pull_request": {"branches": ["main"]}},
            "jobs": {
                "architecture-qa": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"name": "Checkout code", "uses": "actions/checkout@v3"},
                        {"name": "Set up Python", "uses": "actions/setup-python@v4", "with": {"python-version": "3.9"}},
                        {
                            "name": "Install dependencies",
                            "run": "pip install -r requirements.txt pytest pytest-json-report networkx matplotlib seaborn pandas numpy radon pyyaml jinja2",
                        },
                        {
                            "name": "Run Architecture Fitness Functions",
                            "run": "python scripts/ci_integration.py --mode ci",
                        },
                        {
                            "name": "Upload Architecture Reports",
                            "uses": "actions/upload-artifact@v3",
                            "if": "always()",
                            "with": {"name": "architecture-reports", "path": "reports/"},
                        },
                        {
                            "name": "Comment PR with Results",
                            "if": "github.event_name == 'pull_request'",
                            "uses": "actions/github-script@v6",
                            "with": {
                                "script": """
                                  const fs = require('fs');
                                  const path = require('path');

                                  // Read latest CI results
                                  const reportsDir = 'reports/ci';
                                  const files = fs.readdirSync(reportsDir);
                                  const latestReport = files
                                    .filter(f => f.startsWith('ci_pipeline_'))
                                    .sort()
                                    .pop();

                                  if (latestReport) {
                                    const reportPath = path.join(reportsDir, latestReport);
                                    const reportData = JSON.parse(fs.readFileSync(reportPath, 'utf8'));

                                    const success = reportData.overall_success;
                                    const icon = success ? '✅' : '❌';
                                    const status = success ? 'PASSED' : 'FAILED';

                                    let comment = `## ${icon} Architecture Quality Check ${status}\\n\\n`;

                                    // Add quality gates summary
                                    const gates = reportData.stages.quality_gates;
                                    if (gates) {
                                      comment += `### Quality Gates: ${gates.gates_passed}/${gates.gates_passed + gates.gates_failed} passed\\n\\n`;

                                      for (const gate of gates.gates_evaluated) {
                                        const gateIcon = gate.passed ? '✅' : '❌';
                                        comment += `- ${gateIcon} ${gate.name}\\n`;
                                      }
                                    }

                                    github.rest.issues.createComment({
                                      issue_number: context.issue.number,
                                      owner: context.repo.owner,
                                      repo: context.repo.repo,
                                      body: comment
                                    });
                                  }
                                """
                            },
                        },
                    ],
                }
            },
        }

        workflow_dir = self.project_root / ".github" / "workflows"
        workflow_dir.mkdir(parents=True, exist_ok=True)

        output_path = workflow_dir / "architecture-qa.yml"
        with open(output_path, "w") as f:
            yaml.dump(workflow, f, default_flow_style=False, sort_keys=False)

        print(f"GitHub Actions workflow written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CI/CD Integration for Architecture Fitness Functions")
    parser.add_argument(
        "--mode", choices=["ci", "pre-commit", "github-actions"], required=True, help="Integration mode"
    )
    parser.add_argument("--config", help="CI configuration file")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--max-debt-ratio", type=int, default=10, help="Maximum acceptable technical debt ratio")
    parser.add_argument("--max-critical-violations", type=int, default=0, help="Maximum acceptable critical violations")

    args = parser.parse_args()

    ci_integration = CIIntegration(PROJECT_ROOT)

    if args.mode == "ci":
        # Run CI pipeline
        config = {
            "fail_fast": args.fail_fast,
            "max_debt_ratio": args.max_debt_ratio,
            "max_critical_violations": args.max_critical_violations,
            "quality_gates": {
                "no_circular_dependencies": True,
                "coupling_threshold": True,
                "no_critical_connascence": True,
                "technical_debt_acceptable": True,
                "no_critical_drift": True,
            },
        }

        # Load custom config if provided
        if args.config:
            with open(args.config) as f:
                custom_config = yaml.safe_load(f)
                config.update(custom_config)

        pipeline_results = ci_integration.run_ci_pipeline(config)

        # Exit with appropriate code
        exit_code = 0 if pipeline_results["overall_success"] else 1
        sys.exit(exit_code)

    elif args.mode == "pre-commit":
        ci_integration.generate_pre_commit_hooks()

    elif args.mode == "github-actions":
        ci_integration.generate_github_actions_workflow()


if __name__ == "__main__":
    main()
