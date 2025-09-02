#!/bin/bash

# Enhanced Security Validation Script
# Comprehensive security pattern detection with severity-based deployment gates
# Supports production-ready mode for flexible deployment authorization

echo "===================================================================================="
echo "    ENHANCED SECURITY PATTERN VALIDATION"
echo "===================================================================================="

# Initialize counters for each severity level
CRITICAL_VIOLATIONS=0
ERROR_VIOLATIONS=0
WARNING_VIOLATIONS=0
INFO_VIOLATIONS=0

# Parse command line arguments
PRODUCTION_READY=false
VERBOSE=false
AUDIT_LOG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --production-ready)
      PRODUCTION_READY=true
      shift
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --audit-log)
      AUDIT_LOG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Set up logging
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
if [ -n "$AUDIT_LOG" ]; then
  echo "{\"validation_run\": {\"timestamp\": \"$TIMESTAMP\", \"production_ready_mode\": $PRODUCTION_READY}, \"audit_events\": [" > "$AUDIT_LOG"
fi

log_security_event() {
  local severity=$1
  local check_type=$2
  local description=$3
  local file_pattern=$4
  local count=$5
  
  if [ "$VERBOSE" = true ]; then
    echo "[$severity] $check_type: $description (Files: $file_pattern, Count: $count)"
  fi
  
  if [ -n "$AUDIT_LOG" ]; then
    echo "{\"severity\": \"$severity\", \"check_type\": \"$check_type\", \"description\": \"$description\", \"file_pattern\": \"$file_pattern\", \"count\": $count, \"timestamp\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\"}," >> "$AUDIT_LOG"
  fi
}

echo "Security Validation Mode: $([ "$PRODUCTION_READY" = true ] && echo "PRODUCTION-READY" || echo "STRICT")"
echo ""

# =====================================================================================
# CRITICAL SECURITY CHECKS - ALWAYS BLOCK DEPLOYMENT
# =====================================================================================

echo "🔴 CRITICAL Security Checks (Always Block Deployment):"

# Live API keys and tokens
echo "  Checking for live production credentials..."

# OpenAI API keys
OPENAI_KEYS=$(grep -r "sk-[A-Za-z0-9]\{48,\}" --include="*.py" --include="*.js" --include="*.json" --include="*.yaml" --include="*.yml" . 2>/dev/null | wc -l)
if [ "$OPENAI_KEYS" -gt 0 ]; then
  echo "  ❌ CRITICAL: $OPENAI_KEYS potential OpenAI API keys detected"
  log_security_event "CRITICAL" "openai_api_key" "Live OpenAI API keys detected" "**/*.{py,js,json,yaml,yml}" "$OPENAI_KEYS"
  CRITICAL_VIOLATIONS=$((CRITICAL_VIOLATIONS + 1))
fi

# GitHub Personal Access Tokens
GITHUB_TOKENS=$(grep -r "ghp_[A-Za-z0-9]\{36\}" --include="*.py" --include="*.js" --include="*.json" --include="*.yaml" --include="*.yml" . 2>/dev/null | wc -l)
if [ "$GITHUB_TOKENS" -gt 0 ]; then
  echo "  ❌ CRITICAL: $GITHUB_TOKENS potential GitHub Personal Access Tokens detected"
  log_security_event "CRITICAL" "github_pat" "Live GitHub Personal Access Tokens detected" "**/*.{py,js,json,yaml,yml}" "$GITHUB_TOKENS"
  CRITICAL_VIOLATIONS=$((CRITICAL_VIOLATIONS + 1))
fi

# AWS Access Keys
AWS_KEYS=$(grep -r "AKIA[A-Z0-9]\{16\}" --include="*.py" --include="*.js" --include="*.json" --include="*.yaml" --include="*.yml" . 2>/dev/null | wc -l)
if [ "$AWS_KEYS" -gt 0 ]; then
  echo "  ❌ CRITICAL: $AWS_KEYS potential AWS access keys detected"
  log_security_event "CRITICAL" "aws_access_key" "Live AWS access keys detected" "**/*.{py,js,json,yaml,yml}" "$AWS_KEYS"
  CRITICAL_VIOLATIONS=$((CRITICAL_VIOLATIONS + 1))
fi

# Private key blocks
PRIVATE_KEYS=$(grep -r "-----BEGIN [A-Z ]* PRIVATE KEY-----" --include="*.py" --include="*.pem" --include="*.key" . 2>/dev/null | wc -l)
if [ "$PRIVATE_KEYS" -gt 0 ]; then
  echo "  ❌ CRITICAL: $PRIVATE_KEYS private key blocks detected"
  log_security_event "CRITICAL" "private_key_block" "Private key blocks detected in code" "**/*.{py,pem,key}" "$PRIVATE_KEYS"
  CRITICAL_VIOLATIONS=$((CRITICAL_VIOLATIONS + 1))
fi

# Password hashes in non-test files
PASSWORD_HASHES=$(grep -r "\$[0-9][a-z]\$[^\"']*" --include="*.py" core/ infrastructure/ 2>/dev/null | grep -v test | wc -l)
if [ "$PASSWORD_HASHES" -gt 0 ]; then
  echo "  ❌ CRITICAL: $PASSWORD_HASHES potential password hashes in production code"
  log_security_event "CRITICAL" "password_hash" "Password hashes detected in production code" "core/**/*.py infrastructure/**/*.py" "$PASSWORD_HASHES"
  CRITICAL_VIOLATIONS=$((CRITICAL_VIOLATIONS + 1))
fi

# =====================================================================================
# ERROR SECURITY CHECKS - BLOCK IN STRICT MODE, ACCEPT IN PRODUCTION MODE
# =====================================================================================

echo ""
echo "🟠 ERROR Security Checks (Block in Strict Mode):"

# Hardcoded passwords without pragma allowlist
echo "  Checking for hardcoded credentials without pragma comments..."
HARDCODED_PASSWORDS=$(grep -r "password.*=.*[\"'][^\"']*[A-Za-z0-9!@#$%^&*()]\{8,\}[^\"']*[\"']" --include="*.py" core/ infrastructure/ 2>/dev/null | grep -v "pragma.*allowlist" | grep -v test | wc -l)
if [ "$HARDCODED_PASSWORDS" -gt 0 ]; then
  echo "  🟠 ERROR: $HARDCODED_PASSWORDS hardcoded passwords without pragma allowlist"
  log_security_event "ERROR" "hardcoded_password" "Hardcoded passwords without pragma allowlist" "core/**/*.py infrastructure/**/*.py" "$HARDCODED_PASSWORDS"
  ERROR_VIOLATIONS=$((ERROR_VIOLATIONS + 1))
fi

# Hardcoded API keys without pragma allowlist
HARDCODED_API_KEYS=$(grep -r "api_key.*=.*[\"'][^\"']\{20,\}[\"']" --include="*.py" core/ infrastructure/ 2>/dev/null | grep -v "pragma.*allowlist" | grep -v test | grep -v "os.environ\|getenv" | wc -l)
if [ "$HARDCODED_API_KEYS" -gt 0 ]; then
  echo "  🟠 ERROR: $HARDCODED_API_KEYS hardcoded API keys without pragma allowlist"
  log_security_event "ERROR" "hardcoded_api_key" "Hardcoded API keys without pragma allowlist" "core/**/*.py infrastructure/**/*.py" "$HARDCODED_API_KEYS"
  ERROR_VIOLATIONS=$((ERROR_VIOLATIONS + 1))
fi

# Database connection strings with embedded credentials
DB_CONNECTIONS=$(grep -r "jdbc:.*password=" --include="*.py" --include="*.properties" --include="*.yaml" --include="*.yml" . 2>/dev/null | grep -v test | wc -l)
if [ "$DB_CONNECTIONS" -gt 0 ]; then
  echo "  🟠 ERROR: $DB_CONNECTIONS database connection strings with embedded passwords"
  log_security_event "ERROR" "db_connection_creds" "Database connection strings with embedded credentials" "**/*.{py,properties,yaml,yml}" "$DB_CONNECTIONS"
  ERROR_VIOLATIONS=$((ERROR_VIOLATIONS + 1))
fi

# Critical TODOs/FIXMEs that must be resolved
CRITICAL_TODOS=$(grep -r "TODO.*CRITICAL\|FIXME.*CRITICAL\|TODO.*SECURITY\|FIXME.*SECURITY" --include="*.py" core/ infrastructure/ 2>/dev/null | wc -l)
if [ "$CRITICAL_TODOS" -gt 0 ]; then
  echo "  🟠 ERROR: $CRITICAL_TODOS critical TODOs/FIXMEs requiring resolution"
  log_security_event "ERROR" "critical_todo" "Critical TODOs/FIXMEs requiring resolution" "core/**/*.py infrastructure/**/*.py" "$CRITICAL_TODOS"
  ERROR_VIOLATIONS=$((ERROR_VIOLATIONS + 1))
fi

# =====================================================================================
# WARNING SECURITY CHECKS - ACCEPTED IN PRODUCTION MODE
# =====================================================================================

echo ""
echo "🟡 WARNING Security Checks (Accepted in Production Mode):"

# Debug mode in production code
DEBUG_MODE=$(grep -r "DEBUG.*=.*True" --include="*.py" core/ infrastructure/ 2>/dev/null | wc -l)
if [ "$DEBUG_MODE" -gt 0 ]; then
  echo "  🟡 WARNING: $DEBUG_MODE debug mode settings found in production code"
  log_security_event "WARNING" "debug_mode" "Debug mode settings in production code" "core/**/*.py infrastructure/**/*.py" "$DEBUG_MODE"
  WARNING_VIOLATIONS=$((WARNING_VIOLATIONS + 1))
fi

# Long base64 strings without pragma comments
BASE64_STRINGS=$(grep -r "[\"'][A-Za-z0-9+/]\{40,\}={0,2}[\"']" --include="*.py" core/ infrastructure/ 2>/dev/null | grep -v "pragma.*allowlist" | grep -v test | wc -l)
if [ "$BASE64_STRINGS" -gt 0 ]; then
  echo "  🟡 WARNING: $BASE64_STRINGS long base64 strings without pragma comments"
  log_security_event "WARNING" "base64_string" "Long base64 strings without pragma comments" "core/**/*.py infrastructure/**/*.py" "$BASE64_STRINGS"
  WARNING_VIOLATIONS=$((WARNING_VIOLATIONS + 1))
fi

# Test passwords without pragma comments
TEST_PASSWORDS=$(grep -r "password.*=.*[\"'][^\"']*test[^\"']*[\"']" --include="*.py" . 2>/dev/null | grep -v "pragma.*allowlist" | wc -l)
if [ "$TEST_PASSWORDS" -gt 0 ]; then
  echo "  🟡 WARNING: $TEST_PASSWORDS test passwords without pragma comments"
  log_security_event "WARNING" "test_password_no_pragma" "Test passwords without pragma comments" "**/*.py" "$TEST_PASSWORDS"
  WARNING_VIOLATIONS=$((WARNING_VIOLATIONS + 1))
fi

# Authorization headers
AUTH_HEADERS=$(grep -r "authorization:.*[\"'][^\"']\{20,\}[\"']" --include="*.py" --include="*.js" . 2>/dev/null | grep -v test | wc -l)
if [ "$AUTH_HEADERS" -gt 0 ]; then
  echo "  🟡 WARNING: $AUTH_HEADERS authorization headers with long values"
  log_security_event "WARNING" "auth_header" "Authorization headers with long values" "**/*.{py,js}" "$AUTH_HEADERS"
  WARNING_VIOLATIONS=$((WARNING_VIOLATIONS + 1))
fi

# =====================================================================================
# INFO SECURITY CHECKS - INFORMATIONAL ONLY
# =====================================================================================

echo ""
echo "🔵 INFO Security Checks (Informational Only):"

# Excessive print statements (should use logging)
PRINT_COUNT=$(grep -r "print(" --include="*.py" core/ infrastructure/ 2>/dev/null | grep -v test | wc -l || echo "0")
if [ "$PRINT_COUNT" -gt 10 ]; then
  echo "  🔵 INFO: $PRINT_COUNT print statements found (consider using logging)"
  log_security_event "INFO" "excessive_prints" "Excessive print statements (should use logging)" "core/**/*.py infrastructure/**/*.py" "$PRINT_COUNT"
  INFO_VIOLATIONS=$((INFO_VIOLATIONS + 1))
fi

# Generic token assignments
GENERIC_TOKENS=$(grep -r "token.*=.*[\"'][^\"']\{10,\}[\"']" --include="*.py" core/ infrastructure/ 2>/dev/null | grep -v test | wc -l)
if [ "$GENERIC_TOKENS" -gt 0 ]; then
  echo "  🔵 INFO: $GENERIC_TOKENS generic token assignments found"
  log_security_event "INFO" "generic_token" "Generic token assignments" "core/**/*.py infrastructure/**/*.py" "$GENERIC_TOKENS"
  INFO_VIOLATIONS=$((INFO_VIOLATIONS + 1))
fi

# Long uppercase strings (potential credentials)
UPPERCASE_STRINGS=$(grep -r "[\"'][A-Z0-9]\{10,\}[\"']" --include="*.py" core/ infrastructure/ 2>/dev/null | grep -v test | wc -l)
if [ "$UPPERCASE_STRINGS" -gt 5 ]; then
  echo "  🔵 INFO: $UPPERCASE_STRINGS long uppercase strings (check if credentials)"
  log_security_event "INFO" "uppercase_string" "Long uppercase strings (potential credentials)" "core/**/*.py infrastructure/**/*.py" "$UPPERCASE_STRINGS"
  INFO_VIOLATIONS=$((INFO_VIOLATIONS + 1))
fi

# =====================================================================================
# DEPLOYMENT GATE DECISION LOGIC
# =====================================================================================

echo ""
echo "===================================================================================="
echo "    SECURITY VALIDATION RESULTS"
echo "===================================================================================="

echo "SEVERITY BREAKDOWN:"
echo "  🔴 CRITICAL: $CRITICAL_VIOLATIONS issues"
echo "  🟠 ERROR: $ERROR_VIOLATIONS issues"
echo "  🟡 WARNING: $WARNING_VIOLATIONS issues"
echo "  🔵 INFO: $INFO_VIOLATIONS issues"
echo ""

# Determine deployment authorization
DEPLOYMENT_BLOCKED=false
EXIT_CODE=0

# CRITICAL issues always block deployment
if [ $CRITICAL_VIOLATIONS -gt 0 ]; then
  echo "🔴 DEPLOYMENT BLOCKED - CRITICAL security issues detected"
  echo "   Critical issues MUST be resolved before deployment"
  DEPLOYMENT_BLOCKED=true
  EXIT_CODE=2
elif [ $ERROR_VIOLATIONS -gt 0 ] && [ "$PRODUCTION_READY" != true ]; then
  echo "🟠 DEPLOYMENT BLOCKED - ERROR security issues in STRICT mode"
  echo "   Use --production-ready flag to accept ERROR issues with justification"
  DEPLOYMENT_BLOCKED=true
  EXIT_CODE=1
elif [ $ERROR_VIOLATIONS -gt 0 ] && [ "$PRODUCTION_READY" = true ]; then
  echo "🟡 DEPLOYMENT AUTHORIZED - ERROR issues accepted in PRODUCTION-READY mode"
  echo "   ERROR issues require review and justification"
  EXIT_CODE=0
elif [ $WARNING_VIOLATIONS -gt 0 ]; then
  if [ "$PRODUCTION_READY" = true ]; then
    echo "🟢 DEPLOYMENT AUTHORIZED - WARNING issues accepted in PRODUCTION-READY mode"
  else
    echo "🟡 DEPLOYMENT AUTHORIZED - WARNING issues present but acceptable"
  fi
  EXIT_CODE=0
else
  echo "🟢 DEPLOYMENT AUTHORIZED - No blocking security issues detected"
  EXIT_CODE=0
fi

echo ""
echo "DEPLOYMENT GATE DECISION:"
echo "  Production Ready Mode: $PRODUCTION_READY"
echo "  Blocking Issues: $CRITICAL_VIOLATIONS"
echo "  Review Required: $ERROR_VIOLATIONS"
echo "  Warnings: $WARNING_VIOLATIONS"
echo "  Exit Code: $EXIT_CODE"

# Finalize audit log
if [ -n "$AUDIT_LOG" ]; then
  # Remove trailing comma and close JSON
  sed -i '$s/,$//' "$AUDIT_LOG" 2>/dev/null || true
  echo "], \"summary\": {\"critical\": $CRITICAL_VIOLATIONS, \"error\": $ERROR_VIOLATIONS, \"warning\": $WARNING_VIOLATIONS, \"info\": $INFO_VIOLATIONS, \"deployment_authorized\": $([ "$DEPLOYMENT_BLOCKED" = false ] && echo "true" || echo "false"), \"exit_code\": $EXIT_CODE}}" >> "$AUDIT_LOG"
fi

echo ""
echo "Security validation completed: $(date)"
echo "===================================================================================="

exit $EXIT_CODE
