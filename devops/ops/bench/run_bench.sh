#!/bin/bash
# AIVillage Benchmark Script
# Runs performance benchmarks for the Scion Production CI pipeline

set -e

echo "[INFO] Starting AIVillage benchmark suite..."

# Performance tracking
start_time=$(date +%s)
bench_results_file="bench_results.json"

# Check if benchmark tools are available
check_requirements() {
    echo "[CHECK] Checking benchmark requirements..."

    # Check if hyperfine is available (optional but preferred)
    if command -v hyperfine &> /dev/null; then
        HYPERFINE_AVAILABLE=true
        echo "[OK] hyperfine found - will use for precise timing"
    else
        HYPERFINE_AVAILABLE=false
        echo "[WARN] hyperfine not found - using basic timing"
    fi

    # Check Python availability
    if command -v python3 &> /dev/null; then
        PYTHON_CMD=python3
    elif command -v python &> /dev/null; then
        PYTHON_CMD=python
    else
        echo "[ERROR] Python not found - cannot run benchmarks"
        exit 1
    fi

    echo "[OK] Using Python: $PYTHON_CMD"
}

# Run Python import benchmarks
run_import_benchmarks() {
    echo "[BENCH] Running import benchmarks..."

    cat > import_bench.py << 'EOF'
import time
import json
import sys

def time_import(module_name):
    start = time.time()
    try:
        __import__(module_name)
        end = time.time()
        return {"module": module_name, "time": end - start, "success": True}
    except ImportError as e:
        end = time.time()
        return {"module": module_name, "time": end - start, "success": False, "error": str(e)}

# Core Python modules
core_modules = [
    "asyncio", "json", "pathlib", "subprocess", "multiprocessing",
    "concurrent.futures", "sqlite3", "uuid", "hashlib", "hmac"
]

# Optional modules (might not be available)
optional_modules = [
    "torch", "transformers", "numpy", "scipy", "sklearn",
    "fastapi", "uvicorn", "pydantic", "httpx", "websockets"
]

results = {"core": [], "optional": []}

print("Testing core module imports...")
for module in core_modules:
    result = time_import(module)
    results["core"].append(result)
    status = "[OK]" if result["success"] else "[FAIL]"
    print(f"{status} {module}: {result['time']*1000:.2f}ms")

print("\nTesting optional module imports...")
for module in optional_modules:
    result = time_import(module)
    results["optional"].append(result)
    status = "[OK]" if result["success"] else "[WARN]"
    print(f"{status} {module}: {result['time']*1000:.2f}ms")

with open("import_benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n[SUCCESS] Import benchmarks complete. Results saved to import_benchmark_results.json")
EOF

    $PYTHON_CMD import_bench.py
    rm import_bench.py
}

# Run computation benchmarks
run_computation_benchmarks() {
    echo "[BENCH] Running computation benchmarks..."

    cat > computation_bench.py << 'EOF'
import time
import json
import hashlib
import random

def benchmark_cpu_intensive():
    """CPU-intensive computation benchmark"""
    start = time.time()
    result = sum(i * i for i in range(100000))
    end = time.time()
    return {"test": "cpu_intensive", "time": end - start, "result": result}

def benchmark_memory_allocation():
    """Memory allocation benchmark"""
    start = time.time()
    data = [random.random() for _ in range(100000)]
    checksum = sum(data)
    end = time.time()
    return {"test": "memory_allocation", "time": end - start, "size": len(data)}

def benchmark_hashing():
    """Cryptographic hashing benchmark"""
    start = time.time()
    data = b"benchmark data" * 10000
    hash_result = hashlib.sha256(data).hexdigest()
    end = time.time()
    return {"test": "sha256_hashing", "time": end - start, "data_size": len(data)}

def benchmark_string_operations():
    """String processing benchmark"""
    start = time.time()
    text = "benchmark test string " * 1000
    operations = [
        text.upper(),
        text.lower(),
        text.replace("benchmark", "performance"),
        text.split(" "),
        "".join(text.split())
    ]
    end = time.time()
    return {"test": "string_operations", "time": end - start, "operations": len(operations)}

# Run all benchmarks
benchmarks = [
    benchmark_cpu_intensive,
    benchmark_memory_allocation,
    benchmark_hashing,
    benchmark_string_operations
]

results = []
print("Running computation benchmarks...")

for benchmark in benchmarks:
    result = benchmark()
    results.append(result)
    print(f"[OK] {result['test']}: {result['time']*1000:.2f}ms")

with open("computation_benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n[SUCCESS] Computation benchmarks complete. Results saved to computation_benchmark_results.json")
EOF

    $PYTHON_CMD computation_bench.py
    rm computation_bench.py
}

# Run file I/O benchmarks
run_io_benchmarks() {
    echo "[BENCH] Running file I/O benchmarks..."

    cat > io_bench.py << 'EOF'
import time
import json
import tempfile
import os

def benchmark_file_write():
    """File writing benchmark"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        temp_file = f.name

    start = time.time()
    with open(temp_file, 'w') as f:
        for i in range(10000):
            f.write(f"Line {i}: This is a test line for benchmarking file I/O performance.\n")
    end = time.time()

    file_size = os.path.getsize(temp_file)
    os.unlink(temp_file)

    return {"test": "file_write", "time": end - start, "file_size": file_size}

def benchmark_file_read():
    """File reading benchmark"""
    # Create a test file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        temp_file = f.name
        for i in range(10000):
            f.write(f"Line {i}: This is a test line for benchmarking file I/O performance.\n")

    start = time.time()
    with open(temp_file, 'r') as f:
        lines = f.readlines()
    end = time.time()

    os.unlink(temp_file)

    return {"test": "file_read", "time": end - start, "lines_read": len(lines)}

# Run I/O benchmarks
benchmarks = [benchmark_file_write, benchmark_file_read]
results = []

print("Running I/O benchmarks...")
for benchmark in benchmarks:
    result = benchmark()
    results.append(result)
    print(f"[OK] {result['test']}: {result['time']*1000:.2f}ms")

with open("io_benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n[SUCCESS] I/O benchmarks complete. Results saved to io_benchmark_results.json")
EOF

    $PYTHON_CMD io_bench.py
    rm io_bench.py
}

# Aggregate results
aggregate_results() {
    echo "[BENCH] Aggregating benchmark results..."

    cat > aggregate_results.py << 'EOF'
import json
import time
import os

# Aggregate all benchmark results
aggregate = {
    "timestamp": int(time.time()),
    "benchmarks": {}
}

# Load import benchmarks
if os.path.exists("import_benchmark_results.json"):
    with open("import_benchmark_results.json", "r") as f:
        aggregate["benchmarks"]["imports"] = json.load(f)

# Load computation benchmarks
if os.path.exists("computation_benchmark_results.json"):
    with open("computation_benchmark_results.json", "r") as f:
        aggregate["benchmarks"]["computation"] = json.load(f)

# Load I/O benchmarks
if os.path.exists("io_benchmark_results.json"):
    with open("io_benchmark_results.json", "r") as f:
        aggregate["benchmarks"]["io"] = json.load(f)

# Calculate summary statistics
summary = {
    "total_tests": 0,
    "total_time": 0,
    "fastest_test": {"time": float('inf'), "name": ""},
    "slowest_test": {"time": 0, "name": ""}
}

for category, tests in aggregate["benchmarks"].items():
    if isinstance(tests, list):
        for test in tests:
            if "time" in test:
                summary["total_tests"] += 1
                summary["total_time"] += test["time"]

                test_name = f"{category}.{test.get('test', test.get('module', 'unknown'))}"
                if test["time"] < summary["fastest_test"]["time"]:
                    summary["fastest_test"] = {"time": test["time"], "name": test_name}
                if test["time"] > summary["slowest_test"]["time"]:
                    summary["slowest_test"] = {"time": test["time"], "name": test_name}
    elif isinstance(tests, dict):
        for sub_category, sub_tests in tests.items():
            if isinstance(sub_tests, list):
                for test in sub_tests:
                    if "time" in test:
                        summary["total_tests"] += 1
                        summary["total_time"] += test["time"]

                        test_name = f"{category}.{sub_category}.{test.get('test', test.get('module', 'unknown'))}"
                        if test["time"] < summary["fastest_test"]["time"]:
                            summary["fastest_test"] = {"time": test["time"], "name": test_name}
                        if test["time"] > summary["slowest_test"]["time"]:
                            summary["slowest_test"] = {"time": test["time"], "name": test_name}

aggregate["summary"] = summary

# Save aggregated results
with open("bench.json", "w") as f:
    json.dump(aggregate, f, indent=2)

print(f"[SUMMARY] Benchmark Summary:")
print(f"   Total tests: {summary['total_tests']}")
print(f"   Total time: {summary['total_time']*1000:.2f}ms")
print(f"   Average time: {(summary['total_time']/summary['total_tests'])*1000:.2f}ms")
print(f"   Fastest: {summary['fastest_test']['name']} ({summary['fastest_test']['time']*1000:.2f}ms)")
print(f"   Slowest: {summary['slowest_test']['name']} ({summary['slowest_test']['time']*1000:.2f}ms)")

print(f"\n[SUCCESS] Results aggregated to bench.json")
EOF

    $PYTHON_CMD aggregate_results.py
    rm aggregate_results.py

    # Clean up individual result files
    rm -f import_benchmark_results.json computation_benchmark_results.json io_benchmark_results.json
}

# Main execution
main() {
    check_requirements

    echo ""
    run_import_benchmarks

    echo ""
    run_computation_benchmarks

    echo ""
    run_io_benchmarks

    echo ""
    aggregate_results

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo ""
    echo "[COMPLETE] Benchmark suite completed in ${duration}s"
    echo "[RESULTS] Results available in bench.json"

    # Output for Scion Production CI
    if [[ -f "bench.json" ]]; then
        echo "[SUCCESS] Benchmark results generated successfully"
        # Show brief summary for CI logs
        if command -v jq &> /dev/null; then
            echo "[SUMMARY] Quick Summary:"
            jq -r '.summary | "Tests: \(.total_tests), Total Time: \(.total_time*1000 | round)ms, Avg: \(.total_time/.total_tests*1000 | round)ms"' bench.json
        fi
    else
        echo "[ERROR] Failed to generate benchmark results"
        exit 1
    fi
}

# Run main function
main "$@"
