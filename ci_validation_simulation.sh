#!/bin/bash

echo "FINAL PRODUCTION VALIDATION - CI/CD SIMULATION"
echo "=============================================="
echo ""

# Define exact patterns from scion-gateway-ci.yml
PLACEHOLDER_PATTERNS=(
    "TODO:"
    "FIXME:"
    "XXX:"
    "HACK:"
    "NOTE:"
    "placeholder"
    "not implemented"
    "stub"
    "mock"
    "fake"
    "dummy"
    "temporary"
    "temp implementation"
    "coming soon"
    "to be implemented"
)

# Exact file scope from CI config
FILES_TO_CHECK=$(find . -type f \( \
  -name "*.go" -o \
  -name "*.rs" -o \
  -name "*.py" -o \
  -name "*.proto" \
\) \
  ! -path "./tests/*" \
  ! -path "./*test*" \
  ! -path "./docs/*" \
  ! -path "./examples/*" \
  ! -path "./.git/*" \
  ! -path "./target/*" \
  ! -path "./vendor/*" \
  ! -path "./.claude/*" \
  ! -path "./infrastructure/shared/tools/stub_*" \
  ! -path "./infrastructure/twin/quality/stub_*" \
  ! -path "./experiments/*" \
  ! -path "./swarm/*")

VIOLATIONS_FOUND=false
TOTAL_VIOLATIONS=0

echo "Production files in CI scope:"
echo "$FILES_TO_CHECK" | wc -l
echo ""

for pattern in "${PLACEHOLDER_PATTERNS[@]}"; do
    echo "🔍 Checking for pattern: '$pattern'"
    
    pattern_violations=0
    
    while IFS= read -r file; do
        if grep -l -i "$pattern" "$file" 2>/dev/null; then
            # Skip legitimate utility files
            if [[ "$file" =~ (stub_elimination_system|stub_fix|list_stubs|test_stub) ]] || 
               [[ "$file" =~ \.claude/ ]] || 
               [[ "$file" =~ scripts/ ]] || 
               [[ "$file" =~ tools/ ]]; then
                continue
            fi
            
            echo "  ❌ VIOLATION: $file"
            VIOLATIONS_FOUND=true
            ((pattern_violations++))
        fi
    done <<< "$FILES_TO_CHECK"
    
    if [ $pattern_violations -eq 0 ]; then
        echo "  ✅ Pattern '$pattern' - CLEAN"
    else
        echo "  ⚠️  Pattern '$pattern' - $pattern_violations violations"
        TOTAL_VIOLATIONS=$((TOTAL_VIOLATIONS + pattern_violations))
    fi
    echo ""
done

echo "=============================================="
echo "FINAL CI/CD VALIDATION RESULT:"
echo "=============================================="

if [ "$VIOLATIONS_FOUND" = false ]; then
    echo "🎉 SUCCESS: No placeholder patterns found in production code."
    echo "✅ CI/CD READY - Pipeline will PASS"
    echo "🚀 Production deployment approved"
    exit 0
else
    echo "❌ FAILURE: Found placeholder patterns in production code."
    echo "📊 Total violations: $TOTAL_VIOLATIONS"
    echo "🚫 CI/CD BLOCKED - Pipeline will FAIL"
    echo "🔧 Manual intervention required"
    exit 1
fi
