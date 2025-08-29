#!/bin/bash
# Fast security checks for CI/CD
set -e
echo "🔍 Running fast security checks..."

# Just check for the most critical patterns in a few key files
CRITICAL_FILES=$(find ./core -name "*.py" -type f | head -10)

critical_count=0

# Check for real API keys
if echo "$CRITICAL_FILES" | xargs grep -l "sk-[a-zA-Z0-9]{32,}" 2>/dev/null; then
    echo "❌ CRITICAL: Real API keys found!"
    critical_count=$((critical_count + 1))
fi

# Check for SQL injection
if echo "$CRITICAL_FILES" | xargs grep -l "DROP TABLE" 2>/dev/null; then
    echo "❌ CRITICAL: SQL injection patterns found!"
    critical_count=$((critical_count + 1))
fi

echo "📊 SECURITY CHECK SUMMARY:"
echo "   Critical violations: $critical_count"

if [[ $critical_count -gt 0 ]]; then
    echo "💥 CRITICAL VIOLATIONS FOUND!"
    exit 1
else
    echo "✅ No critical security issues found!"
    exit 0
fi
