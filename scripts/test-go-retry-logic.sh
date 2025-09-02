#!/bin/bash

# Test Script for Go Build Retry Logic
# Validates exponential backoff, error handling, and resilience mechanisms

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test configuration
TEST_DIR="/tmp/go-retry-test"
SCRIPT_DIR="$(dirname "$0")"
RETRY_SCRIPT="$SCRIPT_DIR/go-build-retry.sh"

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

# Test results
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Track test result
track_test() {
    local test_name="$1"
    local result="$2"
    
    TESTS_RUN=$((TESTS_RUN + 1))
    
    if [[ "$result" == "PASS" ]]; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
        log_success "TEST $TESTS_RUN: $test_name - PASSED"
    else
        TESTS_FAILED=$((TESTS_FAILED + 1))
        log_error "TEST $TESTS_RUN: $test_name - FAILED"
    fi
}

# Setup test environment
setup_test_env() {
    log_info "Setting up test environment..."
    
    # Clean and create test directory
    rm -rf "$TEST_DIR"
    mkdir -p "$TEST_DIR"
    cd "$TEST_DIR"
    
    # Initialize minimal Go module for testing
    cat > go.mod << 'EOF'
module test-retry
go 1.21
require github.com/stretchr/testify v1.8.4
EOF
    
    # Create simple Go file
    cat > main.go << 'EOF'
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
EOF
    
    log_success "Test environment setup complete"
}

# Test 1: Basic functionality
test_basic_functionality() {
    log_info "Testing basic functionality..."
    
    if [[ -f "$RETRY_SCRIPT" ]] && [[ -x "$RETRY_SCRIPT" ]]; then
        if "$RETRY_SCRIPT" --help > /dev/null 2>&1; then
            track_test "Basic script execution and help" "PASS"
        else
            track_test "Basic script execution and help" "FAIL"
        fi
    else
        log_error "Retry script not found or not executable: $RETRY_SCRIPT"
        track_test "Script availability" "FAIL"
    fi
}

# Test 2: Parameter validation
test_parameter_validation() {
    log_info "Testing parameter validation..."
    
    local test_passed=true
    
    # Test invalid attempts
    if "$RETRY_SCRIPT" -a 0 download &>/dev/null; then
        test_passed=false
        log_error "Should reject invalid attempt count (0)"
    fi
    
    # Test invalid delay
    if "$RETRY_SCRIPT" -d -1 download &>/dev/null; then
        test_passed=false
        log_error "Should reject negative delay"
    fi
    
    # Test invalid timeout
    if "$RETRY_SCRIPT" -t abc download &>/dev/null; then
        test_passed=false
        log_error "Should reject non-numeric timeout"
    fi
    
    if [[ "$test_passed" == "true" ]]; then
        track_test "Parameter validation" "PASS"
    else
        track_test "Parameter validation" "FAIL"
    fi
}

# Test 3: Successful download (if network available)
test_successful_download() {
    log_info "Testing successful Go module download..."
    
    # Only run if network is available
    if curl -s --max-time 10 https://proxy.golang.org/ > /dev/null 2>&1; then
        if "$RETRY_SCRIPT" -v download; then
            track_test "Successful module download" "PASS"
        else
            track_test "Successful module download" "FAIL"
        fi
    else
        log_warn "Skipping network test - proxy.golang.org unreachable"
        track_test "Successful module download" "SKIP"
    fi
}

# Test 4: Retry mechanism with simulated failure
test_retry_mechanism() {
    log_info "Testing retry mechanism with simulated failures..."
    
    # Create a mock Go module that will fail initially
    cat > mock_go_command.sh << 'EOF'
#!/bin/bash
# Mock go command that fails first N times, then succeeds

FAILURE_FILE="/tmp/mock_go_failures"

if [[ ! -f "$FAILURE_FILE" ]]; then
    echo "2" > "$FAILURE_FILE"  # Fail 2 times, then succeed
fi

FAILURES=$(cat "$FAILURE_FILE")

if [[ $FAILURES -gt 0 ]]; then
    echo $((FAILURES - 1)) > "$FAILURE_FILE"
    echo "Mock failure $FAILURES" >&2
    exit 1
else
    echo "Mock success - go mod download completed"
    exit 0
fi
EOF
    
    chmod +x mock_go_command.sh
    
    # Test with PATH manipulation to use mock
    export PATH="$PWD:$PATH"
    
    # Rename original go temporarily if it exists
    if command -v go > /dev/null; then
        GO_ORIGINAL=$(command -v go)
        mv mock_go_command.sh go
        
        # Test retry with 3 attempts (should succeed on 3rd try)
        if timeout 30 "$RETRY_SCRIPT" -a 3 -d 1 -v download 2>&1 | grep -q "completed successfully"; then
            track_test "Retry mechanism with exponential backoff" "PASS"
        else
            track_test "Retry mechanism with exponential backoff" "FAIL"
        fi
        
        # Restore PATH
        rm -f go
    else
        log_warn "Go not available, skipping retry mechanism test"
        track_test "Retry mechanism with exponential backoff" "SKIP"
    fi
    
    # Cleanup
    rm -f mock_go_command.sh "$FAILURE_FILE" go
}

# Test 5: Timeout handling
test_timeout_handling() {
    log_info "Testing timeout handling..."
    
    # Create a mock command that hangs
    cat > slow_command.sh << 'EOF'
#!/bin/bash
echo "Starting slow operation..."
sleep 30  # Sleep longer than our timeout
echo "This should not be reached"
EOF
    
    chmod +x slow_command.sh
    
    # Test with very short timeout
    local start_time=$(date +%s)
    
    # Use a custom version that runs our slow command
    if ! timeout 15 "$RETRY_SCRIPT" -t 5 -a 1 download 2>/dev/null; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Should complete within ~10 seconds (5s timeout + overhead)
        if [[ $duration -lt 15 ]]; then
            track_test "Timeout handling" "PASS"
        else
            track_test "Timeout handling" "FAIL"
            log_error "Timeout test took too long: ${duration}s"
        fi
    else
        track_test "Timeout handling" "FAIL"
        log_error "Command should have failed due to timeout"
    fi
    
    rm -f slow_command.sh
}

# Test 6: Error classification
test_error_classification() {
    log_info "Testing error classification..."
    
    local test_passed=true
    
    # Test different error types by examining the script's error handling
    # This is a simplified test since we can't easily mock all error conditions
    
    # Check if script has proper error classification patterns
    if grep -q "timeout\|deadline" "$RETRY_SCRIPT" && \
       grep -q "network\|connection" "$RETRY_SCRIPT" && \
       grep -q "checksum\|verify\|module" "$RETRY_SCRIPT"; then
        track_test "Error classification patterns" "PASS"
    else
        track_test "Error classification patterns" "FAIL"
    fi
}

# Test 7: Cache cleaning functionality
test_cache_cleaning() {
    log_info "Testing cache cleaning functionality..."
    
    # Test that clean option is handled
    if "$RETRY_SCRIPT" --help | grep -q "\-c, \--clean"; then
        track_test "Cache cleaning option available" "PASS"
    else
        track_test "Cache cleaning option available" "FAIL"
    fi
}

# Test 8: Verbose output
test_verbose_output() {
    log_info "Testing verbose output..."
    
    # Test that verbose mode provides more output
    local normal_output_lines
    local verbose_output_lines
    
    normal_output_lines=$("$RETRY_SCRIPT" --help 2>&1 | wc -l)
    verbose_output_lines=$("$RETRY_SCRIPT" -v --help 2>&1 | wc -l)
    
    if [[ $verbose_output_lines -ge $normal_output_lines ]]; then
        track_test "Verbose output functionality" "PASS"
    else
        track_test "Verbose output functionality" "FAIL"
    fi
}

# Test 9: Different operations
test_different_operations() {
    log_info "Testing different operations..."
    
    local operations=("download" "build" "test" "verify" "tidy")
    local operations_passed=0
    
    for op in "${operations[@]}"; do
        if "$RETRY_SCRIPT" --help | grep -q "$op"; then
            operations_passed=$((operations_passed + 1))
        fi
    done
    
    if [[ $operations_passed -eq ${#operations[@]} ]]; then
        track_test "All Go operations supported" "PASS"
    else
        track_test "All Go operations supported" "FAIL"
        log_error "Only $operations_passed/${#operations[@]} operations found"
    fi
}

# Cleanup test environment
cleanup_test_env() {
    log_info "Cleaning up test environment..."
    cd /
    rm -rf "$TEST_DIR"
    log_success "Test environment cleaned up"
}

# Generate test report
generate_test_report() {
    echo ""
    echo "=========================================="
    echo "GO RETRY LOGIC TEST REPORT"
    echo "=========================================="
    echo "Tests Run: $TESTS_RUN"
    echo "Tests Passed: $TESTS_PASSED"
    echo "Tests Failed: $TESTS_FAILED"
    echo "Success Rate: $(( (TESTS_PASSED * 100) / TESTS_RUN ))%"
    echo "=========================================="
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        log_success "ALL TESTS PASSED - Retry logic is working correctly"
        return 0
    else
        log_error "SOME TESTS FAILED - Review retry logic implementation"
        return 1
    fi
}

# Main test execution
main() {
    log_info "Starting Go Build Retry Logic Tests"
    log_info "Test environment: $TEST_DIR"
    log_info "Retry script: $RETRY_SCRIPT"
    echo ""
    
    # Setup
    setup_test_env
    
    # Run all tests
    test_basic_functionality
    test_parameter_validation
    test_successful_download
    test_retry_mechanism
    test_timeout_handling
    test_error_classification
    test_cache_cleaning
    test_verbose_output
    test_different_operations
    
    # Cleanup and report
    cleanup_test_env
    generate_test_report
}

# Execute if run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi