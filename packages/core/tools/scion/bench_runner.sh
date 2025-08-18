#!/bin/bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
CRATE_DIR="$ROOT_DIR/betanet-gateway"
OUT_DIR="$ROOT_DIR/tmp_scion_perf"

mkdir -p "$OUT_DIR"

cd "$CRATE_DIR"

# Run benchmarks
cargo bench --quiet

# Aggregate benchmark estimates into receipts
find target/criterion -name estimates.json -print0 | xargs -0 jq -s '.' > "$OUT_DIR/receipts.json"

# Generate metrics snapshot via test
cargo test --test metrics --quiet

echo "Artifacts written to $OUT_DIR"
