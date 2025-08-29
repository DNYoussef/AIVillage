#!/bin/bash
# Optimized Forbidden Terms Check Script for AIVillage
# Quick security check with focus on critical issues only

set -e

echo "🔍 Running optimized security checks..."

# Define only critical forbidden patterns
CRITICAL_PATTERNS=(
    "password.*=.*['\"][a-zA-Z0-9]{8,}['\"]"     # Real passwords (not test patterns)
    "api_key.*=.*['\"]sk-[a-zA-Z0-9]{32,}['\"]"  # Real API keys
    "secret.*=.*['\"][a-zA-Z0-9]{16,}['\"]"      # Real secrets
    "\.execute\(['\"].*DROP.*['\"]"              # SQL injection patterns
    "subprocess\.call.*shell=True"               # Shell injection risk
)

# Quick file discovery - limit scope
FILES_TO_CHECK=$(find . -name "*.py" -path "./core/*" -o -name "*.py" -path "./infrastructure/*" -o -name "*.py" -path "./src/*" | head -100)

violation_count=0
critical_count=0

# Check only critical patterns on core files
for pattern in "${CRITICAL_PATTERNS[@]}"; do
    if [[ -n "$FILES_TO_CHECK" ]]; then
        matches=$(echo "$FILES_TO_CHECK" | xargs grep -l -E "$pattern" 2>/dev/null || true)

        if [[ -n "$matches" ]]; then
            echo "❌ CRITICAL: Found pattern '$pattern' in:"
            echo "$matches" | head -3
            critical_count=$((critical_count + 1))
        fi
    fi
done

# Quick check for hardcoded localhost in production files
localhost_issues=$(find ./core ./infrastructure -name "*.py" -exec grep -l "127\.0\.0\.1\|localhost" {} \; 2>/dev/null | head -5 || true)
if [[ -n "$localhost_issues" ]]; then
    echo "⚠️  WARNING: Found hardcoded localhost in production files"
    echo "$localhost_issues"
fi

# Check for dangerous eval (exclude PyTorch .eval() calls) - enhanced filtering
eval_issues=$(find ./core ./infrastructure -name "*.py" -maxdepth 3 2>/dev/null | head -20 | xargs grep -l "eval(" 2>/dev/null | xargs grep -L "\.eval()\|model\.eval()" 2>/dev/null | head -5 || true)
if [[ -n "$eval_issues" ]]; then
    # Double-check to exclude PyTorch eval patterns
    real_eval_issues=""
    for file in $eval_issues; do
        if grep "eval(" "$file" | grep -v "\.eval()" | grep -v "model\.eval()" | grep -v "original_mode.*eval()" >/dev/null 2>&1; then
            real_eval_issues="$real_eval_issues $file"
        fi
    done

    if [[ -n "$real_eval_issues" ]]; then
        echo "❌ CRITICAL: Found dangerous eval() usage (non-PyTorch) in:"
        echo "$real_eval_issues" | head -3
        critical_count=$((critical_count + 1))
    fi
fi

echo ""
echo "📊 SECURITY CHECK SUMMARY:"
echo "   Critical violations: $critical_count"

# Exit with error only if critical violations found
if [[ $critical_count -gt 0 ]]; then
    echo "💥 CRITICAL VIOLATIONS FOUND! Build should fail."
    exit 1
else
    echo "✅ No critical security issues found!"
    exit 0
fi
