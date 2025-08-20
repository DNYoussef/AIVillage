#!/bin/bash
# Forbidden Terms and Patterns Check Script
# Used by Scion Production CI to ensure code quality and security

set -e

echo "[CHECK] Running forbidden terms and patterns check..."

# Configuration
EXIT_CODE=0
FORBIDDEN_PATTERNS_FILE="$(dirname "$0")/forbidden_patterns.txt"

# Default forbidden patterns if file doesn't exist
setup_default_patterns() {
    cat > "$FORBIDDEN_PATTERNS_FILE" << 'EOF'
# Security-related forbidden patterns
password=
secret=
api_key=
private_key=
token=
auth_token=
access_token=
refresh_token=

# Debugging patterns that shouldn't be in production
console.log(
print(.*password
print(.*secret
print(.*token
debugger;
alert(
confirm(

# Hardcoded URLs and IPs (examples)
http://localhost
127.0.0.1
192.168.
10.0.0.
172.16.

# Forbidden imports/packages in production
import pdb
from pdb import
import debugpy
import pudb

# Quality issues
TODO:.*URGENT
FIXME:.*URGENT
XXX:.*CRITICAL
HACK:.*PROD

# Insecure patterns
eval(
exec(
system(
shell_exec(
passthru(

# Development-only annotations
@pytest.skip
@unittest.skip
@pytest.mark.skip
EOF
}

# Check if forbidden patterns file exists
if [[ ! -f "$FORBIDDEN_PATTERNS_FILE" ]]; then
    echo "📝 Creating default forbidden patterns file..."
    setup_default_patterns
fi

# Function to check files for forbidden patterns
check_files() {
    local pattern="$1"
    local description="$2"
    local severity="$3"

    echo "🔍 Checking for: $description"

    # Find files and check for pattern
    found_files=()
    while IFS= read -r -d '' file; do
        if grep -l "$pattern" "$file" >/dev/null 2>&1; then
            found_files+=("$file")
        fi
    done < <(find . -type f \( -name "*.py" -o -name "*.js" -o -name "*.ts" -o -name "*.rs" -o -name "*.go" \) \
             ! -path "./deprecated/*" \
             ! -path "./archive/*" \
             ! -path "./.git/*" \
             ! -path "./tmp*" \
             ! -path "./build/*" \
             ! -path "./dist/*" \
             ! -path "./node_modules/*" \
             ! -path "./target/*" \
             ! -path "./venv/*" \
             ! -path "./__pycache__/*" \
             -print0)

    if [[ ${#found_files[@]} -gt 0 ]]; then
        case "$severity" in
            "ERROR")
                echo "[ERROR] FORBIDDEN PATTERN FOUND: $description"
                EXIT_CODE=1
                ;;
            "WARNING")
                echo "[WARN]  WARNING PATTERN FOUND: $description"
                ;;
            "INFO")
                echo "[INFO]  INFO PATTERN FOUND: $description"
                ;;
        esac

        for file in "${found_files[@]}"; do
            echo "   📁 $file"
            # Show the line(s) with context
            grep -n -H --color=never "$pattern" "$file" | head -3 | while read -r line; do
                echo "      $line"
            done
        done
        echo ""
    else
        echo "[OK] No issues found for: $description"
    fi
}

# Function to run all checks
run_forbidden_checks() {
    echo "[CRITICAL] Running security and quality checks..."
    echo ""

    # Critical security issues (BLOCK BUILD)
    check_files "password\s*=" "Hardcoded passwords" "ERROR"
    check_files "secret\s*=" "Hardcoded secrets" "ERROR"
    check_files "api_key\s*=" "Hardcoded API keys" "ERROR"
    check_files "private_key\s*=" "Hardcoded private keys" "ERROR"
    check_files "sk-[a-zA-Z0-9]" "OpenAI API keys" "ERROR"

    # Debugging code (BLOCK BUILD)
    check_files "debugger;" "JavaScript debugger statements" "ERROR"
    check_files "import pdb" "Python debugger imports" "ERROR"
    check_files "breakpoint()" "Python breakpoint calls" "ERROR"

    # Quality issues (WARNING)
    check_files "TODO:.*URGENT" "Urgent TODOs" "WARNING"
    check_files "FIXME:.*CRITICAL" "Critical FIXMEs" "WARNING"
    check_files "XXX:.*PROD" "Production XXX comments" "WARNING"

    # Development patterns (WARNING)
    check_files "@pytest\.skip" "Skipped tests" "WARNING"
    check_files "localhost" "Localhost references" "WARNING"
    check_files "127\.0\.0\.1" "Localhost IP references" "WARNING"

    # Info patterns (INFO)
    check_files "console\.log\(" "Console.log statements" "INFO"
    check_files "print\(" "Print statements" "INFO"

    echo "🔍 Custom patterns from $FORBIDDEN_PATTERNS_FILE:"
    echo ""

    # Process custom patterns
    while IFS= read -r line; do
        # Skip comments and empty lines
        [[ "$line" =~ ^#.*$ ]] && continue
        [[ -z "$line" ]] && continue

        # Determine severity based on pattern content
        severity="WARNING"
        if [[ "$line" =~ password|secret|api_key|private_key|token ]]; then
            severity="ERROR"
        elif [[ "$line" =~ debugger|pdb|breakpoint ]]; then
            severity="ERROR"
        elif [[ "$line" =~ TODO|FIXME|XXX ]]; then
            severity="WARNING"
        else
            severity="INFO"
        fi

        check_files "$line" "Custom pattern: $line" "$severity"
    done < "$FORBIDDEN_PATTERNS_FILE"
}

# Function to check for large files that might accidentally be committed
check_large_files() {
    echo "📊 Checking for large files..."

    large_files=()
    while IFS= read -r -d '' file; do
        if [[ -f "$file" ]]; then
            size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)
            if [[ $size -gt 10485760 ]]; then  # 10MB
                large_files+=("$file:$size")
            fi
        fi
    done < <(find . -type f ! -path "./.git/*" ! -path "./deprecated/*" ! -path "./archive/*" -print0)

    if [[ ${#large_files[@]} -gt 0 ]]; then
        echo "[WARN]  Large files found (>10MB):"
        for file_info in "${large_files[@]}"; do
            file="${file_info%:*}"
            size="${file_info#*:}"
            size_mb=$((size / 1048576))
            echo "   📁 $file (${size_mb}MB)"
        done
        echo ""
    else
        echo "[OK] No large files found"
        echo ""
    fi
}

# Function to check for binary files that might be accidentally committed
check_binary_files() {
    echo "🔒 Checking for unexpected binary files..."

    binary_files=()
    while IFS= read -r -d '' file; do
        if [[ -f "$file" ]] && file "$file" | grep -q "binary"; then
            # Skip known good binary file types
            case "$file" in
                *.png|*.jpg|*.jpeg|*.gif|*.ico|*.woff|*.woff2|*.ttf|*.eot) continue ;;
                *.zip|*.tar.gz|*.tar|*.gz) continue ;;
                *.pdf|*.doc|*.docx) continue ;;
                */.git/*) continue ;;
                */target/*) continue ;;
                */node_modules/*) continue ;;
                */venv/*) continue ;;
                *) binary_files+=("$file") ;;
            esac
        fi
    done < <(find . -type f ! -path "./.git/*" ! -path "./deprecated/*" ! -path "./archive/*" ! -path "./target/*" ! -path "./node_modules/*" ! -path "./venv/*" -print0)

    if [[ ${#binary_files[@]} -gt 0 ]]; then
        echo "[WARN]  Unexpected binary files found:"
        for file in "${binary_files[@]}"; do
            echo "   📁 $file"
        done
        echo ""
    else
        echo "[OK] No unexpected binary files found"
        echo ""
    fi
}

# Main execution
main() {
    echo "[CHECK] AIVillage Forbidden Patterns Check"
    echo "======================================"
    echo ""

    run_forbidden_checks
    check_large_files
    check_binary_files

    echo "🏁 Forbidden checks completed"

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "[OK] All checks passed"
    else
        echo "[ERROR] Some checks failed - see output above"
        echo ""
        echo "💡 To fix:"
        echo "   1. Remove or replace forbidden patterns"
        echo "   2. Move secrets to environment variables"
        echo "   3. Remove debugging code"
        echo "   4. Address urgent TODOs/FIXMEs"
    fi

    exit $EXIT_CODE
}

# Run main function
main "$@"
