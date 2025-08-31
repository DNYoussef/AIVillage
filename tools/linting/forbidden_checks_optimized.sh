#!/bin/bash

echo "Running optimized forbidden pattern checks..."

VIOLATIONS=0
CRITICAL_VIOLATIONS=0

# Check for hardcoded secrets
echo "Checking for hardcoded secrets..."
if grep -r "password.*=.*[\"']" --include="*.py" core/ infrastructure/ 2>/dev/null | grep -v test | grep -v "password.*os.environ\|password.*getenv"; then
  echo "Warning: Potential hardcoded passwords found"
  VIOLATIONS=$((VIOLATIONS + 1))
fi

# Check for API keys
if grep -r "api_key.*=.*[\"']" --include="*.py" core/ infrastructure/ 2>/dev/null | grep -v test | grep -v "os.environ\|getenv"; then
  echo "Warning: Potential hardcoded API keys found"
  VIOLATIONS=$((VIOLATIONS + 1))
fi

# Check for critical TODOs/FIXMEs
echo "Checking for critical TODOs/FIXMEs..."
if grep -r "TODO.*CRITICAL\|FIXME.*CRITICAL" --include="*.py" core/ infrastructure/ 2>/dev/null; then
  echo "Error: Critical TODOs/FIXMEs found"
  CRITICAL_VIOLATIONS=$((CRITICAL_VIOLATIONS + 1))
fi

# Check for debug mode in production
if grep -r "DEBUG.*=.*True" --include="*.py" core/ infrastructure/ 2>/dev/null; then
  echo "Warning: Debug mode found in production code"
  VIOLATIONS=$((VIOLATIONS + 1))
fi

# Check for print statements (should use logging)
PRINT_COUNT=$(grep -r "print(" --include="*.py" core/ infrastructure/ 2>/dev/null | grep -v test | wc -l || echo "0")
if [ "$PRINT_COUNT" -gt 10 ]; then
  echo "Warning: $PRINT_COUNT print statements found (consider using logging)"
  VIOLATIONS=$((VIOLATIONS + 1))
fi

echo "Forbidden checks completed:"
echo "  Warnings: $VIOLATIONS"
echo "  Critical: $CRITICAL_VIOLATIONS"

if [ $CRITICAL_VIOLATIONS -gt 0 ]; then
  echo "CRITICAL violations found - failing build"
  exit 1
fi

if [ $VIOLATIONS -gt 5 ]; then
  echo "Too many warnings - consider fixing"
  exit 1
fi

echo "All forbidden checks passed"
