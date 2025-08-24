#!/usr/bin/env python3
"""
Performance monitoring script for AIVillage CI/CD pipeline.
Generates metrics snapshot for performance tracking.
"""

import json
from pathlib import Path
import platform
import sys
import time

import psutil


def get_system_metrics():
    """Get basic system performance metrics."""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_usage": psutil.virtual_memory()._asdict(),
        "disk_usage": psutil.disk_usage("/")._asdict()
        if platform.system() != "Windows"
        else psutil.disk_usage("C:")._asdict(),
        "load_average": psutil.getloadavg() if hasattr(psutil, "getloadavg") else None,
        "boot_time": psutil.boot_time(),
        "system": platform.system(),
        "python_version": platform.python_version(),
    }


def run_basic_performance_test():
    """Run basic performance tests."""
    start_time = time.time()

    # Simple computational test
    test_result = sum(i * i for i in range(10000))

    end_time = time.time()

    return {
        "computation_test_ms": (end_time - start_time) * 1000,
        "computation_result": test_result,
        "timestamp": int(time.time()),
    }


def check_python_imports():
    """Check if critical imports work quickly."""
    import_tests = {}

    test_imports = ["json", "pathlib", "subprocess", "platform", "asyncio", "concurrent.futures", "multiprocessing"]

    for module in test_imports:
        start_time = time.time()
        try:
            __import__(module)
            import_tests[module] = {"success": True, "time_ms": (time.time() - start_time) * 1000}
        except ImportError as e:
            import_tests[module] = {"success": False, "error": str(e), "time_ms": (time.time() - start_time) * 1000}

    return import_tests


def generate_performance_summary():
    """Generate comprehensive performance summary."""
    print("[INFO] Generating performance metrics snapshot...")

    summary = {
        "timestamp": int(time.time()),
        "system_metrics": get_system_metrics(),
        "performance_test": run_basic_performance_test(),
        "import_tests": check_python_imports(),
        "ci_environment": {
            "github_actions": "GITHUB_ACTIONS" in os.environ if "os" in globals() else False,
            "runner_os": os.environ.get("RUNNER_OS", "unknown") if "os" in globals() else "unknown",
        },
    }

    # Write to expected output file
    output_file = Path("test_performance_summary.json")
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Performance summary written to {output_file}")
    print(f"[METRICS] CPU Usage: {summary['system_metrics']['cpu_percent']:.1f}%")
    print(f"[METRICS] Memory Usage: {summary['system_metrics']['memory_usage']['percent']:.1f}%")
    print(f"[PERF] Computation Test: {summary['performance_test']['computation_test_ms']:.2f}ms")

    return summary


if __name__ == "__main__":
    import os

    try:
        generate_performance_summary()
        print("[SUCCESS] Performance monitoring completed successfully")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Performance monitoring failed: {e}")
        sys.exit(1)
