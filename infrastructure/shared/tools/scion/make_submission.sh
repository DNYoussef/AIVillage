#!/usr/bin/env bash
set -euo pipefail

COMMIT=$(git rev-parse --short HEAD)
OUTDIR="dist"
BIN="$OUTDIR/sign_receipts"
ARCHIVE="$OUTDIR/betanet_submission_${COMMIT}.tar.gz"

ARTIFACTS=(
    "reports/htx"
    "utls/templates"
    "utls/diffs"
    "linter/report.txt"
    "sbom"
    "benchmarks/mixnode"
    "logs/c_lib"
    "receipts/aead"
    "receipts/replay"
    "receipts/perf"
    "logs/ci"
    "README_SUBMISSION.md"
)

mkdir -p "$OUTDIR"

# Ensure required artifacts exist
for path in "${ARTIFACTS[@]}"; do
    if [ ! -e "$path" ]; then
        echo "Missing required artifact: $path" >&2
        exit 1
    fi
done

# Compile signer
rustc tools/scion/sign_receipts.rs -O -o "$BIN"

# Sign receipts directory
RECEIPTS_DIR="receipts"
$BIN "$RECEIPTS_DIR" "$OUTDIR/SIGNATURE.txt" "$OUTDIR/PUBLIC_KEY.txt"

# Capture toolchain versions
{
    git rev-parse HEAD 2>/dev/null | head -n1
    rustc --version 2>/dev/null || true
    go version 2>/dev/null || true
    python --version 2>/dev/null || true
} > "$OUTDIR/toolchain_versions.txt"

# Create tarball
tar -czf "$ARCHIVE" \
    reports/htx \
    utls/templates \
    utls/diffs \
    linter/report.txt \
    sbom \
    benchmarks/mixnode \
    logs/c_lib \
    receipts/aead \
    receipts/replay \
    receipts/perf \
    logs/ci \
    -C "$OUTDIR" SIGNATURE.txt PUBLIC_KEY.txt toolchain_versions.txt \
    -C "$PWD" README_SUBMISSION.md

echo "Created $ARCHIVE"
