#!/bin/bash
# Forbidden Terms Check Script for AIVillage
# Checks for dangerous patterns, hardcoded secrets, and prohibited terms

set -e

echo "🔍 Checking for forbidden terms and dangerous patterns..."

# Define forbidden patterns
FORBIDDEN_PATTERNS=(
    "password.*=.*['\"][^'\"]*['\"]"     # Hardcoded passwords
    "secret.*=.*['\"][^'\"]*['\"]"      # Hardcoded secrets
    "api_key.*=.*['\"][^'\"]*['\"]"     # Hardcoded API keys
    "token.*=.*['\"][^'\"]*['\"]"       # Hardcoded tokens
    "\.execute\(['\"].*DROP.*['\"]"     # SQL injection patterns
    "eval\("                            # Dangerous eval usage
    "exec\("                            # Dangerous exec usage
    "subprocess\.call.*shell=True"      # Shell injection risk
    "os\.system\("                      # OS command injection
    "127\.0\.0\.1"                      # Hardcoded localhost
    "localhost"                         # Hardcoded localhost
    "TODO.*SECURITY"                    # Security TODOs (should be fixed)
    "FIXME.*CRITICAL"                   # Critical FIXMEs (should be fixed)
    "XXX.*URGENT"                       # Urgent markers (should be fixed)
)

# Files to check
EXTENSIONS=("*.py" "*.rs" "*.go" "*.js" "*.ts" "*.yaml" "*.yml" "*.json" "*.toml")

# Directories to exclude
EXCLUDE_DIRS=("deprecated" "archive" "experimental" "tests" "build" ".git" "node_modules" "__pycache__" ".venv")

# Build find command with exclusions
FIND_CMD="find . -type f \("
for ext in "${EXTENSIONS[@]}"; do
    FIND_CMD="$FIND_CMD -name \"$ext\" -o"
done
FIND_CMD="${FIND_CMD% -o} \)"

for exclude_dir in "${EXCLUDE_DIRS[@]}"; do
    FIND_CMD="$FIND_CMD -not -path \"./$exclude_dir/*\""
done

echo "📁 Scanning files with extensions: ${EXTENSIONS[*]}"
echo "🚫 Excluding directories: ${EXCLUDE_DIRS[*]}"

# Count violations
violation_count=0
critical_count=0

# Check each forbidden pattern
for pattern in "${FORBIDDEN_PATTERNS[@]}"; do
    echo "🔍 Checking pattern: $pattern"

    # Use eval to execute the complex find command with grep
    matches=$(eval "$FIND_CMD" | xargs grep -l -i -E "$pattern" 2>/dev/null || true)

    if [[ -n "$matches" ]]; then
        echo "❌ VIOLATION: Found forbidden pattern '$pattern' in:"
        echo "$matches" | while read -r file; do
            echo "   - $file"
            # Show the actual matching lines
            grep -n -i -E "$pattern" "$file" | head -3
        done
        echo ""
        violation_count=$((violation_count + 1))

        # Check if this is a critical violation
        if [[ "$pattern" =~ (password|secret|api_key|token|DROP|eval|exec) ]]; then
            critical_count=$((critical_count + 1))
        fi
    fi
done

# Check for insecure randomness
echo "🔍 Checking for insecure randomness patterns..."
insecure_random=$(eval "$FIND_CMD" | xargs grep -l "random\." 2>/dev/null | xargs grep -v "secrets\." 2>/dev/null || true)
if [[ -n "$insecure_random" ]]; then
    echo "⚠️  WARNING: Found potentially insecure random usage (consider using secrets module):"
    echo "$insecure_random" | head -5
fi

# Check for debug statements in production code
echo "🔍 Checking for debug statements..."
debug_statements=$(eval "$FIND_CMD -not -path \"./tests/*\"" | xargs grep -l -E "(print\(|console\.log|debugger|pdb\.set_trace)" 2>/dev/null || true)
if [[ -n "$debug_statements" ]]; then
    echo "⚠️  WARNING: Found debug statements in production code:"
    echo "$debug_statements" | head -5
fi

# Summary
echo ""
echo "📊 FORBIDDEN TERMS CHECK SUMMARY:"
echo "   Total violations: $violation_count"
echo "   Critical violations: $critical_count"
echo "   Debug warnings: $(echo "$debug_statements" | wc -l)"

# Exit with error if critical violations found
if [[ $critical_count -gt 0 ]]; then
    echo "💥 CRITICAL VIOLATIONS FOUND! Build should fail."
    echo "   Please fix all critical security issues before proceeding."
    exit 1
elif [[ $violation_count -gt 0 ]]; then
    echo "⚠️  Non-critical violations found. Please review and fix."
    # Don't fail build for non-critical issues, just warn
    exit 0
else
    echo "✅ No forbidden terms found. Security check passed!"
    exit 0
fi
