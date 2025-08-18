#!/bin/bash
# Betanet Fuzzing Suite
# Runs cargo-fuzz targets for all crates

set -e

echo "🔍 Starting Betanet fuzzing suite..."

# Check if cargo-fuzz is installed
if ! command -v cargo-fuzz &> /dev/null; then
    echo "Installing cargo-fuzz..."
    cargo install cargo-fuzz
fi

# Fuzz targets
FUZZ_TARGETS=(
    "betanet-htx:frame_parsing"
    "betanet-htx:handshake_processing"
    "betanet-mixnode:packet_processing"
    "betanet-mixnode:sphinx_decryption"
    "betanet-utls:clienthello_generation"
    "betanet-utls:fingerprint_parsing"
    "betanet-linter:rule_engine"
)

# Duration for each fuzz target (in seconds)
FUZZ_DURATION=${FUZZ_DURATION:-60}

echo "Running each fuzz target for ${FUZZ_DURATION} seconds..."

for target in "${FUZZ_TARGETS[@]}"; do
    IFS=':' read -r crate fuzz_name <<< "$target"

    echo "🎯 Fuzzing $crate::$fuzz_name"

    cd "crates/$crate"

    # Initialize fuzz targets if they don't exist
    if [ ! -d "fuzz" ]; then
        cargo fuzz init
    fi

    # Add fuzz target if it doesn't exist
    if [ ! -f "fuzz/fuzz_targets/${fuzz_name}.rs" ]; then
        cargo fuzz add "$fuzz_name"
    fi

    # Run fuzzing for specified duration
    timeout "${FUZZ_DURATION}s" cargo fuzz run "$fuzz_name" || true

    cd ../..
done

echo "✅ Fuzzing suite completed!"
echo "📊 Check fuzz/artifacts/ directories for any crashes found"
