#!/usr/bin/env python3
"""Simple test dashboard for AIVillage project."""

import subprocess
import time
from datetime import datetime

def run_command(cmd, timeout=60):
    """Run a command and return results."""
    print(f"Running: {' '.join(cmd)}")
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        duration = time.time() - start
        return {
            "duration": duration,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {"duration": timeout, "returncode": -1, "stdout": "", "stderr": "TIMEOUT"}
    except Exception as e:
        return {"duration": 0, "returncode": -1, "stdout": "", "stderr": str(e)}

def main():
    print("=" * 80)
    print("AIVillage Test Dashboard - Comprehensive Results")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    tests_run = []

    # Sprint 4 Tests
    print("\n[1/5] Sprint 4 - Distributed Infrastructure Tests")
    result = run_command(["python", "scripts/create_integration_tests.py"], timeout=120)
    tests_run.append(("Sprint 4 Integration", result))

    if result["returncode"] == 0:
        if "Passed:" in result["stdout"]:
            passed_line = [line for line in result["stdout"].split('\n') if "Passed:" in line]
            if passed_line:
                print(f"Result: {passed_line[0]}")
        print("Status: PASSED")
    else:
        print("Status: FAILED")

    # Core Module Tests
    core_tests = [
        ("Compression Pipeline", ["python", "-m", "pytest", "tests/test_compression_only.py", "-v", "--tb=no"]),
        ("Pipeline Simple", ["python", "-m", "pytest", "tests/test_pipeline_simple.py", "-v", "--tb=no"]),
        ("Evolution System", ["python", "-m", "pytest", "tests/test_corrected_evolution.py", "-v", "--tb=no"]),
    ]

    for i, (name, cmd) in enumerate(core_tests, 2):
        print(f"\n[{i}/5] {name}")
        result = run_command(cmd, timeout=90)
        tests_run.append((name, result))

        if result["returncode"] == 0:
            # Count passed tests from pytest output
            if " passed" in result["stdout"]:
                lines = result["stdout"].split('\n')
                summary = [line for line in lines if " passed" in line and "=" in line]
                if summary:
                    print(f"Result: {summary[-1].strip()}")
            print("Status: PASSED")
        else:
            print("Status: FAILED")

    # System Health Check
    print("\n[5/5] System Health Check")
    health_cmd = ["python", "-c", "import agent_forge; print('All imports successful')"]
    result = run_command(health_cmd, timeout=30)
    tests_run.append(("System Health", result))

    if result["returncode"] == 0:
        print("Status: PASSED")
    else:
        print("Status: FAILED")

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed_count = sum(1 for _, result in tests_run if result["returncode"] == 0)
    total_count = len(tests_run)
    total_duration = sum(result["duration"] for _, result in tests_run)

    print(f"Tests Passed: {passed_count}/{total_count} ({passed_count/total_count*100:.1f}%)")
    print(f"Total Duration: {total_duration:.1f} seconds")
    print()

    for name, result in tests_run:
        status = "PASS" if result["returncode"] == 0 else "FAIL"
        duration = f"({result['duration']:.1f}s)"
        print(f"  {status:4} {name:<30} {duration}")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if passed_count == total_count:
        print("EXCELLENT! All tests are passing.")
        print("- System is in great shape")
        print("- Continue with regular development")
    elif passed_count >= total_count * 0.8:
        print("GOOD! Most tests are passing.")
        print("- Address failing tests for better stability")
        print("- System is mostly functional")
    else:
        print("NEEDS ATTENTION! Multiple test failures detected.")
        print("- Priority: Fix failing tests")
        print("- Review error messages for specific issues")

    # Detailed failure analysis
    failed_tests = [(name, result) for name, result in tests_run if result["returncode"] != 0]
    if failed_tests:
        print("\nFailed Test Details:")
        for name, result in failed_tests:
            print(f"\n  {name}:")
            if result["stderr"]:
                error_preview = result["stderr"][:200] + "..." if len(result["stderr"]) > 200 else result["stderr"]
                print(f"    Error: {error_preview}")

    print("\n" + "=" * 80)

    return {
        "passed": passed_count,
        "total": total_count,
        "pass_rate": passed_count/total_count*100,
        "duration": total_duration,
        "tests": tests_run
    }

if __name__ == "__main__":
    main()
