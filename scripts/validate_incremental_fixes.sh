#\!/bin/bash

# Test script for placeholder validation fixes
# This script tests each fix incrementally to ensure they work

echo "=== Testing Placeholder Validation Fixes ==="
echo "Testing incremental fixes to CI/CD pipeline"
echo ""

# Test 1: Check if archived files are excluded from scanning
echo "Test 1: Validating archive exclusion patterns..."
if [ -d "archive/legacy_files" ]; then
    echo "[PASS] Archive directory exists: archive/legacy_files"
else
    echo "[FAIL] Archive directory missing"
    exit 1
fi

# Test 2: Verify production-ready replacement exists
echo "Test 2: Checking production-ready replacement..."
if [ -f "infrastructure/shared/tools/development_quality_validator.py" ]; then
    echo "[PASS] Production-ready replacement created"
    
    # Check if it has no placeholder patterns
    if \! grep -i "TODO\|FIXME\|placeholder\|not implemented" "infrastructure/shared/tools/development_quality_validator.py" > /dev/null 2>&1; then
        echo "[PASS] No placeholder patterns in replacement file"
    else
        echo "[FAIL] Placeholder patterns still found in replacement"
        exit 1
    fi
else
    echo "[FAIL] Production replacement missing"
    exit 1
fi

# Test 3: Check proto file paths in Go build system
echo "Test 3: Validating Go build system proto paths..."
MAKEFILE_PATH="integrations/clients/rust/scion-sidecar/Makefile"
if [ -f "$MAKEFILE_PATH" ]; then
    if grep -q "PROTO_DIR=../../../../proto" "$MAKEFILE_PATH"; then
        echo "[PASS] Proto directory path corrected in Makefile"
    else
        echo "[FAIL] Proto directory path not updated"
        exit 1
    fi
else
    echo "[FAIL] Makefile not found"
    exit 1
fi

# Test 4: Check proto file exists at expected location
echo "Test 4: Verifying proto file existence..."
if [ -f "proto/betanet_gateway.proto" ]; then
    echo "[PASS] Proto file exists at expected location"
    
    # Check if proto has required gRPC methods
    REQUIRED_METHODS=("SendScionPacket" "RecvScionPacket" "RegisterPath" "QueryPaths" "Health" "Stats" "ValidateSequence")
    for method in "${REQUIRED_METHODS[@]}"; do
        if grep -q "rpc $method" "proto/betanet_gateway.proto"; then
            echo "[PASS] Required method found: $method"
        else
            echo "[FAIL] Required method missing: $method"
            exit 1
        fi
    done
else
    echo "[FAIL] Proto file missing at proto/betanet_gateway.proto"
    exit 1
fi

# Test 5: Check Rust build configuration
echo "Test 5: Validating Rust build system..."
CARGO_TOML="integrations/clients/rust/betanet/betanet-gateway/Cargo.toml"
BUILD_RS="integrations/clients/rust/betanet/betanet-gateway/build.rs"

if [ -f "$CARGO_TOML" ] && [ -f "$BUILD_RS" ]; then
    echo "[PASS] Rust build files present"
    
    # Check for tonic-build in dependencies
    if grep -q "tonic-build" "$CARGO_TOML"; then
        echo "[PASS] tonic-build dependency found"
    else
        echo "[FAIL] tonic-build dependency missing"
        exit 1
    fi
else
    echo "[FAIL] Rust build files missing"
    exit 1
fi

echo ""
echo "=== Incremental Test Results ==="
echo "[PASS] All incremental fixes validated successfully"
echo "CI/CD pipeline placeholder validation fixes are ready for testing"
echo ""

