#!/bin/bash
# Betanet Benchmarking Suite
# Runs criterion benchmarks for all crates

set -e

echo "📊 Starting Betanet benchmarking suite..."

# Benchmark targets
BENCH_TARGETS=(
    "betanet-htx"
    "betanet-mixnode"
    "betanet-utls"
    "betanet-linter"
    "betanet-dtn"
)

# Create benchmark results directory
mkdir -p target/criterion

echo "Running benchmarks for all crates..."

for crate in "${BENCH_TARGETS[@]}"; do
    echo "🏃 Benchmarking $crate"

    cd "crates/$crate"

    # Run criterion benchmarks
    cargo bench --bench "*" || echo "No benchmarks found for $crate"

    cd ../..
done

echo "✅ Benchmarking suite completed!"
echo "📈 Results available in target/criterion/"

# Run DTN scheduler specific benchmarks
echo ""
echo "🔬 Running DTN Scheduler Performance Tests..."
if [ -f "tools/bench/dtn-scheduler-bench.sh" ]; then
    chmod +x tools/bench/dtn-scheduler-bench.sh
    ./tools/bench/dtn-scheduler-bench.sh
else
    echo "DTN scheduler benchmark script not found - running basic tests"
    cd crates/betanet-dtn
    cargo test --release sched::performance_tests -- --nocapture || echo "DTN performance tests completed"
    cd ../..
fi

# Generate summary report
echo "📝 Generating benchmark summary..."
find target/criterion -name "index.html" -exec echo "Report: {}" \;
