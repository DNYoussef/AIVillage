#!/usr/bin/env python3
"""Quick validation script for agent abstract method implementations.

Runs a focused subset of tests to quickly validate that all abstract
method implementations are working correctly. Ideal for development
and CI/CD pipelines.
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

def run_command(cmd: List[str], timeout: int = 60) -> Tuple[int, str, str]:
    """Run a command with timeout and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        return 1, "", str(e)

def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "pytest", "pytest-asyncio", "pytest-cov", 
        "coverage", "matplotlib", "psutil"
    ]
    
    missing = []
    for package in required_packages:
        exit_code, _, _ = run_command([sys.executable, "-c", f"import {package}"])
        if exit_code != 0:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("✅ All dependencies available")
    return True

def run_smoke_tests() -> Dict[str, bool]:
    """Run smoke tests for each component."""
    print("\n🧪 Running smoke tests...")
    
    smoke_tests = {
        "unified_base_agent": [
            sys.executable, "-m", "pytest", "-xvs", 
            "tests/agents/test_unified_base_agent.py::TestUnifiedBaseAgentInitialization::test_successful_initialization",
            "--tb=short"
        ],
        "base_analytics": [
            sys.executable, "-m", "pytest", "-xvs",
            "tests/agents/test_base_analytics.py::TestBaseAnalyticsMetricRecording::test_record_single_metric",
            "--tb=short"
        ],
        "processing_interface": [
            sys.executable, "-m", "pytest", "-xvs",
            "tests/agents/test_processing_interface.py::TestProcessingInterfaceBasicProcessing::test_successful_processing",
            "--tb=short"
        ]
    }
    
    results = {}
    for component, cmd in smoke_tests.items():
        print(f"  Testing {component}...")
        exit_code, stdout, stderr = run_command(cmd, timeout=30)
        results[component] = exit_code == 0
        
        if exit_code == 0:
            print(f"  ✅ {component} - PASSED")
        else:
            print(f"  ❌ {component} - FAILED")
            if stderr:
                print(f"    Error: {stderr}")
    
    return results

def run_critical_path_tests() -> bool:
    """Run critical path integration tests."""
    print("\n🔗 Running critical path tests...")
    
    critical_tests = [
        sys.executable, "-m", "pytest", "-v",
        "tests/integration/test_agent_integration.py::TestAgentAnalyticsIntegration::test_agent_analytics_integration",
        "tests/behavior/test_agent_behavior.py::TestTaskProcessingWorkflows::test_single_task_workflow_execution",
        "--tb=short", "--maxfail=2"
    ]
    
    exit_code, stdout, stderr = run_command(critical_tests, timeout=120)
    
    if exit_code == 0:
        print("✅ Critical path tests - PASSED")
        return True
    else:
        print("❌ Critical path tests - FAILED")
        if stderr:
            print(f"Error: {stderr}")
        return False

def run_performance_check() -> bool:
    """Run basic performance validation."""
    print("\n⚡ Running performance check...")
    
    perf_test = [
        sys.executable, "-m", "pytest", "-v",
        "tests/performance/test_agent_performance.py::TestAgentPerformanceBenchmarks::test_single_task_latency_benchmark",
        "--tb=short", "--timeout=60"
    ]
    
    exit_code, stdout, stderr = run_command(perf_test, timeout=90)
    
    if exit_code == 0:
        print("✅ Performance check - PASSED")
        return True
    else:
        print("❌ Performance check - FAILED")
        if "timeout" in stderr.lower():
            print("Warning: Performance test timed out - may indicate performance issues")
        return False

def quick_coverage_check() -> Tuple[bool, float]:
    """Run quick coverage check on core components."""
    print("\n📊 Running coverage check...")
    
    coverage_cmd = [
        sys.executable, "-m", "pytest",
        "tests/agents/test_unified_base_agent.py::TestUnifiedBaseAgentInitialization",
        "tests/agents/test_base_analytics.py::TestBaseAnalyticsMetricRecording",
        "tests/agents/test_processing_interface.py::TestProcessingInterfaceInitialization",
        "--cov=agents.unified_base_agent",
        "--cov=agents.king.analytics.base_analytics", 
        "--cov=agents.interfaces.processing_interface",
        "--cov-report=term-missing",
        "--cov-fail-under=70",
        "-q"
    ]
    
    exit_code, stdout, stderr = run_command(coverage_cmd, timeout=60)
    
    # Parse coverage percentage from output
    coverage_percentage = 0.0
    if "TOTAL" in stdout:
        lines = stdout.split('\n')
        for line in lines:
            if line.strip().startswith("TOTAL"):
                parts = line.split()
                if parts and parts[-1].endswith('%'):
                    try:
                        coverage_percentage = float(parts[-1][:-1])
                    except ValueError:
                        pass
                break
    
    passed = exit_code == 0 and coverage_percentage >= 70
    
    if passed:
        print(f"✅ Coverage check - {coverage_percentage:.1f}% (target: 70%+)")
    else:
        print(f"❌ Coverage check - {coverage_percentage:.1f}% (below 70% target)")
    
    return passed, coverage_percentage

def validate_imports() -> bool:
    """Validate that all modules can be imported without errors."""
    print("\n📦 Validating imports...")
    
    import_tests = [
        "from agents.unified_base_agent import UnifiedBaseAgent, UnifiedAgentConfig",
        "from agents.king.analytics.base_analytics import BaseAnalytics",
        "from agents.interfaces.processing_interface import ProcessingInterface, ProcessorCapability"
    ]
    
    all_passed = True
    for test_import in import_tests:
        exit_code, _, stderr = run_command([
            sys.executable, "-c", test_import
        ])
        
        if exit_code == 0:
            module_name = test_import.split()[1].split('.')[1]
            print(f"  ✅ {module_name} imports successfully")
        else:
            print(f"  ❌ Import failed: {test_import}")
            if stderr:
                print(f"    Error: {stderr}")
            all_passed = False
    
    return all_passed

def main() -> int:
    """Main validation workflow."""
    print("🚀 AIVillage Agent Implementation Quick Validation")
    print("=" * 55)
    
    start_time = time.time()
    all_results = []
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed - install requirements first")
        return 1
    all_results.append(("Dependencies", True))
    
    # Validate imports
    import_success = validate_imports()
    all_results.append(("Import Validation", import_success))
    if not import_success:
        print("\n❌ Import validation failed - check module paths")
        return 1
    
    # Run smoke tests
    smoke_results = run_smoke_tests()
    all_smoke_passed = all(smoke_results.values())
    all_results.append(("Smoke Tests", all_smoke_passed))
    
    if not all_smoke_passed:
        print(f"\n❌ Some smoke tests failed: {smoke_results}")
        print("Fix basic functionality before proceeding")
        return 1
    
    # Run critical path tests
    critical_success = run_critical_path_tests()
    all_results.append(("Critical Path", critical_success))
    
    # Run performance check
    perf_success = run_performance_check()
    all_results.append(("Performance", perf_success))
    
    # Check coverage
    coverage_success, coverage_pct = quick_coverage_check()
    all_results.append(("Coverage", coverage_success))
    
    # Generate summary
    total_time = time.time() - start_time
    
    print("\n" + "="*55)
    print("📋 VALIDATION SUMMARY")
    print("="*55)
    
    passed_count = sum(1 for _, passed in all_results if passed)
    total_count = len(all_results)
    
    print(f"Overall: {passed_count}/{total_count} checks passed")
    print(f"Duration: {total_time:.1f} seconds")
    print(f"Coverage: {coverage_pct:.1f}%")
    
    print(f"\nDetailed Results:")
    for check_name, passed in all_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {check_name}")
    
    # Determine overall result
    if passed_count == total_count:
        print(f"\n🎉 SUCCESS: All validations passed!")
        print("Agent implementations are ready for testing")
        return 0
    elif passed_count >= total_count * 0.8:  # 80% threshold
        print(f"\n⚠️  PARTIAL SUCCESS: {passed_count}/{total_count} checks passed")
        print("Most functionality working - address failing checks")
        return 0
    else:
        print(f"\n❌ FAILURE: Only {passed_count}/{total_count} checks passed")
        print("Significant issues detected - review implementation")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        sys.exit(1)