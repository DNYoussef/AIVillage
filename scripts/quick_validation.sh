#!/bin/bash

echo "🔍 Quick validation test for SCION production files..."

VIOLATIONS_FOUND=false

# Check the specific files mentioned in CI for common placeholder patterns
FILES_TO_CHECK="infrastructure/p2p/scion_gateway.py core/agents/infrastructure/navigation/scion_navigator.py"

for file in $FILES_TO_CHECK; do
    if [ -f "$file" ]; then
        echo "Checking $file for placeholder patterns..."
        
        # Check for the patterns that were causing failures
        if grep -i "mock.*implementation" "$file"; then
            echo "[FAIL] Found 'mock implementation' in $file"
            VIOLATIONS_FOUND=true
        fi
        
        if grep -i "TODO:\|FIXME:\|XXX:\|HACK:\|placeholder\|not implemented\|stub\|fake\|dummy\|temporary" "$file"; then
            echo "[FAIL] Found placeholder patterns in $file"
            VIOLATIONS_FOUND=true
        fi
    fi
done

# Check proto file exists
if [ ! -f "proto/betanet_gateway.proto" ]; then
    echo "[FAIL] Proto file not found: proto/betanet_gateway.proto"
    VIOLATIONS_FOUND=true
else
    echo "[PASS] Proto file exists: proto/betanet_gateway.proto"
    
    # Check for required gRPC methods
    REQUIRED_METHODS=("SendScionPacket" "RecvScionPacket" "RegisterPath" "QueryPaths" "Health" "Stats" "ValidateSequence")
    
    for method in "${REQUIRED_METHODS[@]}"; do
        if ! grep -q "rpc $method" "proto/betanet_gateway.proto"; then
            echo "[FAIL] Required gRPC method '$method' not found"
            VIOLATIONS_FOUND=true
        fi
    done
    echo "[PASS] All required gRPC methods found"
fi

if [ "$VIOLATIONS_FOUND" = true ]; then
    echo ""
    echo "[FAIL] VALIDATION FAILED"
    exit 1
else
    echo ""
    echo "[PASS] VALIDATION PASSED - Production files look clean"
fi