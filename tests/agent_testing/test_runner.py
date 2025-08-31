"""Comprehensive test runner for agent abstract method implementations.

Executes all test suites with proper reporting, coverage analysis, and 
performance benchmarking. Implements TDD London School methodology validation.
"""

import asyncio
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

import pytest
import coverage


class TestRunner:
    """Comprehensive test runner for agent testing."""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path(__file__).parent.parent
        self.results = {
            "summary": {},
            "suites": {},
            "coverage": {},
            "performance": {},
            "timestamp": datetime.now().isoformat()
        }
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites with comprehensive reporting."""
        print("🧪 Starting comprehensive agent testing suite...")
        print("=" * 60)
        
        # Initialize coverage tracking
        cov = coverage.Coverage(source=[str(self.base_path / "agents")])
        cov.start()
        
        try:
            # Run test suites in order
            self._run_unit_tests()
            self._run_integration_tests()
            self._run_behavior_tests() 
            self._run_performance_tests()
            self._run_chaos_tests()
            
            # Stop coverage and generate report
            cov.stop()
            cov.save()
            
            # Generate coverage report
            self._generate_coverage_report(cov)
            
            # Generate final summary
            self._generate_summary()
            
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            self.results["summary"]["status"] = "FAILED"
            self.results["summary"]["error"] = str(e)
        
        return self.results
    
    def _run_unit_tests(self):
        """Run all unit test suites."""
        print("\n📋 Running Unit Tests...")
        print("-" * 40)
        
        unit_test_paths = [
            "tests/agents/test_unified_base_agent.py",
            "tests/agents/test_base_analytics.py", 
            "tests/agents/test_processing_interface.py"
        ]
        
        for test_path in unit_test_paths:
            suite_name = Path(test_path).stem
            print(f"  Running {suite_name}...")
            
            result = self._run_pytest_suite(test_path, ["unit"])
            self.results["suites"][suite_name] = result
            
            if result["status"] == "PASSED":
                print(f"  ✅ {suite_name}: {result['passed']}/{result['total']} tests passed")
            else:
                print(f"  ❌ {suite_name}: {result['failed']} tests failed")
    
    def _run_integration_tests(self):
        """Run integration test suites."""
        print("\n🔗 Running Integration Tests...")
        print("-" * 40)
        
        integration_path = "tests/integration/test_agent_integration.py"
        print("  Running agent integration tests...")
        
        result = self._run_pytest_suite(integration_path, ["integration"])
        self.results["suites"]["integration"] = result
        
        if result["status"] == "PASSED":
            print(f"  ✅ Integration: {result['passed']}/{result['total']} tests passed")
        else:
            print(f"  ❌ Integration: {result['failed']} tests failed")
    
    def _run_behavior_tests(self):
        """Run behavior-driven test suites."""
        print("\n🎭 Running Behavior Tests...")
        print("-" * 40)
        
        behavior_path = "tests/behavior/test_agent_behavior.py"
        print("  Running workflow behavior tests...")
        
        result = self._run_pytest_suite(behavior_path, ["behavior"])
        self.results["suites"]["behavior"] = result
        
        if result["status"] == "PASSED":
            print(f"  ✅ Behavior: {result['passed']}/{result['total']} tests passed")
        else:
            print(f"  ❌ Behavior: {result['failed']} tests failed")
    
    def _run_performance_tests(self):
        """Run performance benchmark tests."""
        print("\n⚡ Running Performance Tests...")
        print("-" * 40)
        
        perf_path = "tests/performance/test_agent_performance.py"
        print("  Running performance benchmarks...")
        
        start_time = time.perf_counter()
        result = self._run_pytest_suite(perf_path, ["performance"])
        end_time = time.perf_counter()
        
        result["benchmark_duration_seconds"] = end_time - start_time
        self.results["suites"]["performance"] = result
        
        if result["status"] == "PASSED":
            print(f"  ✅ Performance: {result['passed']}/{result['total']} benchmarks passed")
            print(f"     Benchmark duration: {result['benchmark_duration_seconds']:.2f}s")
        else:
            print(f"  ❌ Performance: {result['failed']} benchmarks failed")
    
    def _run_chaos_tests(self):
        """Run chaos engineering tests."""
        print("\n🌪️  Running Chaos Engineering Tests...")
        print("-" * 40)
        
        chaos_path = "tests/performance/test_agent_performance.py"
        print("  Running chaos engineering scenarios...")
        
        result = self._run_pytest_suite(chaos_path, ["chaos"])
        self.results["suites"]["chaos"] = result
        
        if result["status"] == "PASSED":
            print(f"  ✅ Chaos: {result['passed']}/{result['total']} scenarios passed")
        else:
            print(f"  ❌ Chaos: {result['failed']} scenarios failed")
    
    def _run_pytest_suite(self, test_path: str, markers: List[str]) -> Dict[str, Any]:
        """Run a specific pytest suite and return results."""
        full_path = self.base_path / test_path
        
        if not full_path.exists():
            return {
                "status": "SKIPPED",
                "reason": f"Test file not found: {test_path}",
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0
            }
        
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            str(full_path),
            "-v",
            "--tb=short",
            "--json-report", 
            "--json-report-file=/tmp/pytest_report.json"
        ]
        
        # Add marker filters
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])
        
        try:
            # Run pytest
            result = subprocess.run(
                cmd,
                cwd=str(self.base_path),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Parse results
            return self._parse_pytest_results(result)
            
        except subprocess.TimeoutExpired:
            return {
                "status": "TIMEOUT", 
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "error": "Test suite timed out after 5 minutes"
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "error": str(e)
            }
    
    def _parse_pytest_results(self, result: subprocess.CompletedProcess) -> Dict[str, Any]:
        """Parse pytest results from subprocess output."""
        try:
            # Try to load JSON report if available
            json_path = Path("/tmp/pytest_report.json")
            if json_path.exists():
                with open(json_path) as f:
                    json_data = json.load(f)
                
                summary = json_data.get("summary", {})
                return {
                    "status": "PASSED" if result.returncode == 0 else "FAILED",
                    "total": summary.get("total", 0),
                    "passed": summary.get("passed", 0),
                    "failed": summary.get("failed", 0),
                    "skipped": summary.get("skipped", 0),
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            
            # Fallback to parsing stdout
            lines = result.stdout.split('\n')
            summary_line = ""
            
            for line in lines:
                if "passed" in line and ("failed" in line or "error" in line or "skipped" in line):
                    summary_line = line
                    break
            
            if summary_line:
                # Parse summary line (e.g., "5 passed, 2 failed, 1 skipped")
                parts = summary_line.lower().split(',')
                passed = failed = skipped = 0
                
                for part in parts:
                    part = part.strip()
                    if 'passed' in part:
                        passed = int(part.split()[0])
                    elif 'failed' in part or 'error' in part:
                        failed = int(part.split()[0])
                    elif 'skipped' in part:
                        skipped = int(part.split()[0])
                
                total = passed + failed + skipped
                
                return {
                    "status": "PASSED" if result.returncode == 0 else "FAILED",
                    "total": total,
                    "passed": passed,
                    "failed": failed,
                    "skipped": skipped,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            
            # Default result if parsing fails
            return {
                "status": "PASSED" if result.returncode == 0 else "FAILED",
                "total": 0,
                "passed": 0,
                "failed": 1 if result.returncode != 0 else 0,
                "skipped": 0,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "total": 0,
                "passed": 0,
                "failed": 1,
                "skipped": 0,
                "error": f"Failed to parse results: {e}",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
    
    def _generate_coverage_report(self, cov: coverage.Coverage):
        """Generate coverage report."""
        print("\n📊 Generating Coverage Report...")
        print("-" * 40)
        
        try:
            # Get coverage data
            cov.load()
            
            # Generate text report
            import io
            coverage_stream = io.StringIO()
            cov.report(file=coverage_stream, show_missing=True)
            coverage_text = coverage_stream.getvalue()
            
            # Parse coverage percentage
            lines = coverage_text.split('\n')
            total_line = [line for line in lines if line.startswith('TOTAL')]
            
            coverage_percentage = 0
            if total_line:
                parts = total_line[0].split()
                if len(parts) >= 4 and parts[-1].endswith('%'):
                    coverage_percentage = float(parts[-1][:-1])
            
            self.results["coverage"] = {
                "percentage": coverage_percentage,
                "report": coverage_text,
                "status": "GOOD" if coverage_percentage >= 80 else "NEEDS_IMPROVEMENT"
            }
            
            print(f"  📈 Overall Coverage: {coverage_percentage:.1f}%")
            
            if coverage_percentage >= 90:
                print("  🎉 Excellent coverage!")
            elif coverage_percentage >= 80:
                print("  ✅ Good coverage")
            else:
                print("  ⚠️  Coverage needs improvement")
                
        except Exception as e:
            print(f"  ❌ Coverage report failed: {e}")
            self.results["coverage"] = {
                "percentage": 0,
                "error": str(e),
                "status": "ERROR"
            }
    
    def _generate_summary(self):
        """Generate final test summary."""
        print("\n📋 Test Summary")
        print("=" * 60)
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        
        suite_statuses = []
        
        for suite_name, suite_result in self.results["suites"].items():
            total_tests += suite_result.get("total", 0)
            total_passed += suite_result.get("passed", 0)
            total_failed += suite_result.get("failed", 0)
            total_skipped += suite_result.get("skipped", 0)
            suite_statuses.append(suite_result.get("status", "UNKNOWN"))
        
        # Calculate overall status
        if all(status in ["PASSED", "SKIPPED"] for status in suite_statuses):
            overall_status = "PASSED"
        elif "ERROR" in suite_statuses or "TIMEOUT" in suite_statuses:
            overall_status = "ERROR"
        else:
            overall_status = "FAILED"
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        self.results["summary"] = {
            "status": overall_status,
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "skipped": total_skipped,
            "success_rate": success_rate,
            "coverage_percentage": self.results["coverage"].get("percentage", 0)
        }
        
        print(f"Overall Status: {'🎉 PASSED' if overall_status == 'PASSED' else '❌ ' + overall_status}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")
        print(f"Skipped: {total_skipped}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Coverage: {self.results['coverage'].get('percentage', 0):.1f}%")
        
        # Print detailed suite results
        print(f"\nSuite Details:")
        for suite_name, suite_result in self.results["suites"].items():
            status_emoji = "✅" if suite_result["status"] == "PASSED" else "❌"
            print(f"  {status_emoji} {suite_name}: {suite_result['passed']}/{suite_result['total']}")
        
        # Quality assessment
        print(f"\n🏆 Quality Assessment:")
        if success_rate >= 95 and self.results["coverage"].get("percentage", 0) >= 90:
            print("  🌟 EXCELLENT - Production ready!")
        elif success_rate >= 90 and self.results["coverage"].get("percentage", 0) >= 80:
            print("  ✅ GOOD - Ready for staging")
        elif success_rate >= 80:
            print("  ⚠️  FAIR - Needs improvement before production")
        else:
            print("  🚫 POOR - Requires significant work")
    
    def save_report(self, filename: Optional[str] = None):
        """Save test report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"agent_test_report_{timestamp}.json"
        
        report_path = self.base_path / "tests" / "reports" / filename
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Report saved to: {report_path}")
        return report_path


def main():
    """Main entry point for test runner."""
    runner = TestRunner()
    
    # Run all tests
    results = runner.run_all_tests()
    
    # Save report
    runner.save_report()
    
    # Exit with appropriate code
    if results["summary"].get("status") == "PASSED":
        print("\n🎉 All tests passed! Agent implementations are ready.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please review the results above.")
        sys.exit(1)


if __name__ == "__main__":
    main()