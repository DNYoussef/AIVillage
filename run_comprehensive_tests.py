#!/usr/bin/env python3
"""
Comprehensive Test Runner for AIVillage
Intelligent test execution with coverage analysis and reporting.
"""

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional
import json

class TestRunner:
    """Intelligent test runner with multiple execution modes."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.test_results = {}
        
    def run_unit_tests(self, coverage: bool = True) -> Dict:
        """Run unit tests with optional coverage."""
        print("🧪 Running unit tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            "-m", "unit or not (integration or performance or slow)",
            "-v", "--tb=short"
        ]
        
        if coverage:
            cmd.extend([
                "--cov=.",
                "--cov-report=term-missing",
                "--cov-report=json:unit_coverage.json"
            ])
        
        return self._execute_tests(cmd, "unit_tests")
    
    def run_integration_tests(self) -> Dict:
        """Run integration tests."""
        print("🔗 Running integration tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            "-m", "integration",
            "-v", "--tb=short",
            "--maxfail=3"
        ]
        
        return self._execute_tests(cmd, "integration_tests")
    
    def run_mcp_server_tests(self, coverage: bool = True) -> Dict:
        """Run MCP server specific tests."""
        print("🏗️ Running MCP server tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/mcp_servers/",
            "-m", "mcp or not slow",
            "-v", "--tb=short"
        ]
        
        if coverage:
            cmd.extend([
                "--cov=mcp_servers",
                "--cov-report=term-missing",
                "--cov-report=json:mcp_coverage.json"
            ])
        
        return self._execute_tests(cmd, "mcp_tests")
    
    def run_production_tests(self, coverage: bool = True) -> Dict:
        """Run production component tests."""
        print("🏭 Running production tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/production/",
            "-m", "not slow",
            "-v", "--tb=short"
        ]
        
        if coverage:
            cmd.extend([
                "--cov=production",
                "--cov-report=term-missing", 
                "--cov-report=json:production_coverage.json"
            ])
        
        return self._execute_tests(cmd, "production_tests")
    
    def run_performance_tests(self) -> Dict:
        """Run performance tests."""
        print("⚡ Running performance tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            "-m", "performance",
            "-v", "--tb=short",
            "--durations=0"
        ]
        
        return self._execute_tests(cmd, "performance_tests")
    
    def run_security_tests(self) -> Dict:
        """Run security tests."""
        print("🛡️ Running security tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            "-m", "security",
            "-v", "--tb=short"
        ]
        
        return self._execute_tests(cmd, "security_tests")
    
    def run_smoke_tests(self) -> Dict:
        """Run smoke tests for quick validation."""
        print("💨 Running smoke tests...")
        
        # Key tests that should always pass
        smoke_tests = [
            "tests/core/",
            "tests/test_*.py",
            "-k", "not (slow or performance)",
            "--maxfail=1"
        ]
        
        cmd = [sys.executable, "-m", "pytest"] + smoke_tests + ["-v", "--tb=line"]
        
        return self._execute_tests(cmd, "smoke_tests")
    
    def run_all_tests(self, coverage: bool = True) -> Dict:
        """Run complete test suite."""
        print("🎯 Running complete test suite...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            "-v", "--tb=short",
            "--durations=10",
            "--maxfail=10"
        ]
        
        if coverage:
            cmd.extend([
                "--cov=.",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=json:complete_coverage.json"
            ])
        
        return self._execute_tests(cmd, "all_tests")
    
    def run_changed_files_tests(self) -> Dict:
        """Run tests for changed files only."""
        print("📝 Running tests for changed files...")
        
        # Get changed files from git
        changed_files = self._get_changed_files()
        
        if not changed_files:
            print("No changed files detected.")
            return {"success": True, "message": "No tests needed"}
        
        # Find corresponding test files
        test_files = self._find_test_files_for_changes(changed_files)
        
        if not test_files:
            print("No test files found for changed files.")
            return {"success": True, "message": "No corresponding tests"}
        
        cmd = [
            sys.executable, "-m", "pytest"
        ] + test_files + [
            "-v", "--tb=short",
            "--cov=" + ",".join(changed_files[:10]),  # Limit coverage to changed files
            "--cov-report=term-missing"
        ]
        
        return self._execute_tests(cmd, "changed_files_tests")
    
    def run_ci_tests(self) -> Dict:
        """Run tests optimized for CI environment."""
        print("🤖 Running CI-optimized tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            "-x",  # Stop on first failure
            "--tb=short",
            "-q",  # Quiet output
            "--durations=5",
            "--cov=.",
            "--cov-report=xml:coverage.xml",
            "--cov-report=term:skip-covered",
            "--junitxml=test-results.xml"
        ]
        
        return self._execute_tests(cmd, "ci_tests")
    
    def _execute_tests(self, cmd: List[str], test_type: str) -> Dict:
        """Execute test command and capture results."""
        start_time = time.time()
        
        try:
            print(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
                cwd=self.project_root
            )
            
            execution_time = time.time() - start_time
            
            # Parse pytest output for metrics
            test_metrics = self._parse_pytest_output(result.stdout)
            
            test_result = {
                "success": result.returncode == 0,
                "execution_time": execution_time,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "metrics": test_metrics
            }
            
            self.test_results[test_type] = test_result
            
            if result.returncode == 0:
                print(f"✅ {test_type} completed successfully in {execution_time:.2f}s")
            else:
                print(f"❌ {test_type} failed in {execution_time:.2f}s")
                if result.stderr:
                    print(f"Errors: {result.stderr[:500]}...")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            print(f"⏰ {test_type} timed out after {execution_time:.2f}s")
            
            return {
                "success": False,
                "execution_time": execution_time,
                "error": "Test execution timed out",
                "timeout": True
            }
        
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"💥 {test_type} failed with error: {e}")
            
            return {
                "success": False,
                "execution_time": execution_time,
                "error": str(e)
            }
    
    def _parse_pytest_output(self, output: str) -> Dict:
        """Parse pytest output to extract metrics."""
        metrics = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0
        }
        
        # Look for pytest summary line
        for line in output.split('\n'):
            if '=====' in line and ('passed' in line or 'failed' in line):
                # Parse summary line like "===== 10 passed, 2 failed, 1 skipped in 5.23s ====="
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.isdigit():
                        count = int(part)
                        if i + 1 < len(parts):
                            result_type = parts[i + 1].replace(',', '')
                            if result_type in metrics:
                                metrics[result_type] = count
                                metrics["total_tests"] += count
                break
        
        return metrics
    
    def _get_changed_files(self) -> List[str]:
        """Get list of changed Python files from git."""
        try:
            # Get changed files from git
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                files = [
                    f for f in result.stdout.strip().split('\n')
                    if f.endswith('.py') and not f.startswith('tests/')
                ]
                return files
            
        except Exception:
            pass
        
        return []
    
    def _find_test_files_for_changes(self, changed_files: List[str]) -> List[str]:
        """Find test files corresponding to changed files."""
        test_files = []
        
        for file_path in changed_files:
            file_path = Path(file_path)
            
            # Look for corresponding test files
            test_patterns = [
                self.project_root / "tests" / f"test_{file_path.stem}.py",
                self.project_root / "tests" / file_path.parent.name / f"test_{file_path.stem}.py",
                file_path.parent / f"test_{file_path.stem}.py"
            ]
            
            for test_pattern in test_patterns:
                if test_pattern.exists():
                    test_files.append(str(test_pattern))
                    break
        
        return test_files
    
    def generate_report(self) -> Dict:
        """Generate comprehensive test report."""
        if not self.test_results:
            return {"error": "No test results available"}
        
        total_time = sum(
            result.get("execution_time", 0) 
            for result in self.test_results.values()
        )
        
        successful_tests = sum(
            1 for result in self.test_results.values() 
            if result.get("success", False)
        )
        
        report = {
            "summary": {
                "total_test_suites": len(self.test_results),
                "successful_suites": successful_tests,
                "total_execution_time": total_time,
                "overall_success": successful_tests == len(self.test_results)
            },
            "detailed_results": self.test_results,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check for failed tests
        failed_suites = [
            suite for suite, result in self.test_results.items()
            if not result.get("success", False)
        ]
        
        if failed_suites:
            recommendations.append(
                f"Address failing test suites: {', '.join(failed_suites)}"
            )
        
        # Check for slow tests
        slow_suites = [
            suite for suite, result in self.test_results.items()
            if result.get("execution_time", 0) > 300  # 5 minutes
        ]
        
        if slow_suites:
            recommendations.append(
                f"Optimize slow test suites: {', '.join(slow_suites)}"
            )
        
        # Check for skipped tests
        for suite, result in self.test_results.items():
            metrics = result.get("metrics", {})
            if metrics.get("skipped", 0) > 0:
                recommendations.append(
                    f"Review {metrics['skipped']} skipped tests in {suite}"
                )
        
        return recommendations

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="AIVillage Comprehensive Test Runner")
    parser.add_argument("--mode", choices=[
        "unit", "integration", "mcp", "production", "performance", 
        "security", "smoke", "all", "changed", "ci"
    ], default="smoke", help="Test execution mode")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage collection")
    parser.add_argument("--output", help="Output file for test report")
    parser.add_argument("--parallel", type=int, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting AIVillage test execution (mode: {args.mode})")
    
    runner = TestRunner()
    
    # Execute tests based on mode
    coverage = not args.no_coverage
    
    if args.mode == "unit":
        runner.run_unit_tests(coverage=coverage)
    elif args.mode == "integration":
        runner.run_integration_tests()
    elif args.mode == "mcp":
        runner.run_mcp_server_tests(coverage=coverage)
    elif args.mode == "production":
        runner.run_production_tests(coverage=coverage)
    elif args.mode == "performance":
        runner.run_performance_tests()
    elif args.mode == "security":
        runner.run_security_tests()
    elif args.mode == "smoke":
        runner.run_smoke_tests()
    elif args.mode == "all":
        runner.run_all_tests(coverage=coverage)
    elif args.mode == "changed":
        runner.run_changed_files_tests()
    elif args.mode == "ci":
        runner.run_ci_tests()
    
    # Generate and save report
    report = runner.generate_report()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Test report saved to {args.output}")
    
    # Print summary
    summary = report["summary"]
    print(f"\n📊 Test Execution Summary:")
    print(f"   Total suites: {summary['total_test_suites']}")
    print(f"   Successful: {summary['successful_suites']}")
    print(f"   Total time: {summary['total_execution_time']:.2f}s")
    print(f"   Overall success: {'✅' if summary['overall_success'] else '❌'}")
    
    if report.get("recommendations"):
        print(f"\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"   - {rec}")
    
    # Exit with appropriate code
    sys.exit(0 if summary["overall_success"] else 1)

if __name__ == "__main__":
    main()