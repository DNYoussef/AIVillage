#!/bin/bash

# Go Build Retry Script with Exponential Backoff
# Provides resilient Go module operations with comprehensive error handling

set -euo pipefail

# Configuration
DEFAULT_MAX_ATTEMPTS=3
DEFAULT_BASE_DELAY=2
DEFAULT_TIMEOUT=900
DEFAULT_OPERATION="go mod download"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Go Build Retry Script with Exponential Backoff

USAGE:
    $0 [OPTIONS] [COMMAND]

OPTIONS:
    -a, --attempts NUM      Maximum number of retry attempts (default: $DEFAULT_MAX_ATTEMPTS)
    -d, --delay NUM         Base delay in seconds for exponential backoff (default: $DEFAULT_BASE_DELAY)
    -t, --timeout NUM       Timeout in seconds for each attempt (default: $DEFAULT_TIMEOUT)
    -c, --clean             Clean module cache before retrying
    -v, --verbose           Verbose output
    -h, --help              Show this help message

COMMANDS:
    download               Download Go modules (default)
    build                  Build Go project
    test                   Run Go tests
    verify                 Verify Go modules
    tidy                   Tidy Go modules

EXAMPLES:
    $0                                    # Download modules with default settings
    $0 -a 5 -d 3 download                # Download with 5 attempts, 3s base delay
    $0 -t 1200 -c build                  # Build with 20min timeout, clean cache
    $0 -v test                           # Run tests with verbose output

EXIT CODES:
    0                      Success
    1                      General error
    2                      Timeout error
    3                      Network error
    4                      Module error
EOF
}

# Parse command line arguments
MAX_ATTEMPTS=$DEFAULT_MAX_ATTEMPTS
BASE_DELAY=$DEFAULT_BASE_DELAY
TIMEOUT=$DEFAULT_TIMEOUT
OPERATION=""
CLEAN_CACHE=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--attempts)
            MAX_ATTEMPTS="$2"
            shift 2
            ;;
        -d|--delay)
            BASE_DELAY="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_CACHE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        download|build|test|verify|tidy)
            OPERATION="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Set default operation if none specified
if [[ -z "$OPERATION" ]]; then
    OPERATION="download"
fi

# Validate arguments
if ! [[ "$MAX_ATTEMPTS" =~ ^[0-9]+$ ]] || [ "$MAX_ATTEMPTS" -lt 1 ]; then
    log_error "Invalid max attempts: $MAX_ATTEMPTS (must be positive integer)"
    exit 1
fi

if ! [[ "$BASE_DELAY" =~ ^[0-9]+$ ]] || [ "$BASE_DELAY" -lt 1 ]; then
    log_error "Invalid base delay: $BASE_DELAY (must be positive integer)"
    exit 1
fi

if ! [[ "$TIMEOUT" =~ ^[0-9]+$ ]] || [ "$TIMEOUT" -lt 1 ]; then
    log_error "Invalid timeout: $TIMEOUT (must be positive integer)"
    exit 1
fi

# Verbose logging
if [[ "$VERBOSE" == "true" ]]; then
    log_info "Configuration:"
    log_info "  Operation: $OPERATION"
    log_info "  Max attempts: $MAX_ATTEMPTS"
    log_info "  Base delay: ${BASE_DELAY}s"
    log_info "  Timeout: ${TIMEOUT}s"
    log_info "  Clean cache: $CLEAN_CACHE"
fi

# Check if we're in a Go project
check_go_project() {
    if [[ ! -f "go.mod" ]]; then
        log_error "go.mod not found. Please run this script from a Go project root directory."
        exit 1
    fi
    
    if ! command -v go &> /dev/null; then
        log_error "Go is not installed or not in PATH"
        exit 1
    fi
    
    local go_version
    go_version=$(go version)
    log_info "Using $go_version"
    
    if [[ "$VERBOSE" == "true" ]]; then
        log_info "Go environment:"
        go env | grep -E "(GOROOT|GOPATH|GOCACHE|GOMODCACHE|GOPROXY)" | while read -r line; do
            log_info "  $line"
        done
    fi
}

# Clean Go module cache
clean_cache() {
    if [[ "$CLEAN_CACHE" == "true" ]]; then
        log_info "Cleaning Go module cache..."
        if go clean -modcache; then
            log_success "Module cache cleaned"
        else
            log_warn "Failed to clean module cache, continuing..."
        fi
    fi
}

# Get appropriate Go command based on operation
get_go_command() {
    case "$OPERATION" in
        download)
            echo "go mod download -x"
            ;;
        build)
            echo "go build -v ./..."
            ;;
        test)
            echo "go test -v ./..."
            ;;
        verify)
            echo "go mod verify"
            ;;
        tidy)
            echo "go mod tidy -v"
            ;;
        *)
            log_error "Unknown operation: $OPERATION"
            exit 1
            ;;
    esac
}

# Network connectivity check
check_network() {
    log_info "Checking network connectivity..."
    
    # Check Go proxy
    if timeout 10 curl -s -I https://proxy.golang.org/ > /dev/null; then
        log_success "Go proxy reachable"
        return 0
    else
        log_warn "Go proxy unreachable, checking alternatives..."
        
        # Check direct GitHub access
        if timeout 10 curl -s -I https://github.com/ > /dev/null; then
            log_info "GitHub reachable, setting GOPROXY=direct"
            export GOPROXY=direct
            return 0
        else
            log_error "Network connectivity issues detected"
            return 3
        fi
    fi
}

# Enhanced error analysis
analyze_error() {
    local error_output="$1"
    local exit_code="$2"
    
    if [[ "$VERBOSE" == "true" ]]; then
        log_info "Error analysis for exit code $exit_code:"
        log_info "Error output: $error_output"
    fi
    
    # Classify error type
    if echo "$error_output" | grep -qi "timeout\|deadline"; then
        log_error "Timeout error detected"
        return 2
    elif echo "$error_output" | grep -qi "network\|connection\|dns\|proxy"; then
        log_error "Network error detected"
        return 3
    elif echo "$error_output" | grep -qi "checksum\|verify\|module"; then
        log_error "Module integrity error detected"
        return 4
    else
        log_error "General error (exit code: $exit_code)"
        return 1
    fi
}

# Main retry function with exponential backoff
retry_with_backoff() {
    local cmd="$1"
    local attempt=1
    local error_output
    local exit_code
    
    log_info "Starting Go $OPERATION with retry logic..."
    log_info "Command: $cmd"
    
    while [ $attempt -le $MAX_ATTEMPTS ]; do
        log_info "Attempt $attempt of $MAX_ATTEMPTS..."
        
        # Execute command with timeout
        if error_output=$(timeout ${TIMEOUT}s bash -c "$cmd" 2>&1); then
            log_success "Go $OPERATION completed successfully on attempt $attempt"
            
            if [[ "$VERBOSE" == "true" ]]; then
                echo "$error_output"
            fi
            
            # Save metrics for CI/CD reporting
            echo "RETRY_COUNT=$attempt" >> "${GITHUB_OUTPUT:-/dev/null}"
            echo "SUCCESS=true" >> "${GITHUB_OUTPUT:-/dev/null}"
            echo "OPERATION=$OPERATION" >> "${GITHUB_OUTPUT:-/dev/null}"
            
            return 0
        else
            exit_code=$?
            log_error "Attempt $attempt failed (exit code: $exit_code)"
            
            # Analyze error type
            analyze_error "$error_output" "$exit_code"
            error_type=$?
            
            if [ $attempt -eq $MAX_ATTEMPTS ]; then
                log_error "All $MAX_ATTEMPTS attempts failed for Go $OPERATION"
                log_error "Final error output:"
                echo "$error_output"
                
                # Save failure metrics
                echo "RETRY_COUNT=$attempt" >> "${GITHUB_OUTPUT:-/dev/null}"
                echo "SUCCESS=false" >> "${GITHUB_OUTPUT:-/dev/null}"
                echo "ERROR_TYPE=$error_type" >> "${GITHUB_OUTPUT:-/dev/null}"
                echo "OPERATION=$OPERATION" >> "${GITHUB_OUTPUT:-/dev/null}"
                
                return $error_type
            fi
            
            # Calculate exponential backoff delay
            local delay=$((BASE_DELAY * (1 << (attempt - 1))))
            log_warn "Waiting ${delay}s before next attempt..."
            sleep $delay
            
            # Clean cache on network/module errors before retrying
            if [[ $error_type -eq 3 ]] || [[ $error_type -eq 4 ]]; then
                log_info "Cleaning cache before retry due to error type..."
                go clean -modcache || true
            fi
            
            attempt=$((attempt + 1))
        fi
    done
}

# Main execution
main() {
    log_info "Starting Go Build Retry Script"
    log_info "Operation: $OPERATION"
    
    # Pre-flight checks
    check_go_project
    check_network || {
        error_code=$?
        log_error "Pre-flight network check failed"
        exit $error_code
    }
    
    # Clean cache if requested
    clean_cache
    
    # Get command to execute
    local cmd
    cmd=$(get_go_command)
    
    # Execute with retry logic
    retry_with_backoff "$cmd"
    local result=$?
    
    if [ $result -eq 0 ]; then
        log_success "Go $OPERATION completed successfully with retry script"
    else
        log_error "Go $OPERATION failed with retry script (exit code: $result)"
    fi
    
    return $result
}

# Execute main function
main "$@"