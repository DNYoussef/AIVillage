#!/bin/bash
# Claude Code MCP Import Script for AIVillage
# Usage: ./scripts/import_mcp_to_claude.sh [--dry-run] [--help]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CANONICAL_CONFIG="$PROJECT_ROOT/infra/mcp/servers.jsonc"
CLAUDE_CONFIG="$PROJECT_ROOT/.roo/mcp.json"
BACKUP_DIR="$PROJECT_ROOT/.mcp/backups"

# Help function
show_help() {
    echo "AIVillage MCP Configuration Import Script"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --dry-run    Show what would be done without making changes"
    echo "  --help       Show this help message"
    echo
    echo "This script imports the canonical MCP configuration into Claude Code."
    echo "It creates a backup of the current configuration before making changes."
}

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    local deps=("node" "npx")
    local missing=()

    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing+=("$dep")
        fi
    done

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing dependencies: ${missing[*]}"
        log_info "Please install missing dependencies:"
        log_info "  - Node.js: https://nodejs.org/"
        exit 1
    fi
}

# Create backup
create_backup() {
    local config_file=$1
    local backup_file="$BACKUP_DIR/mcp_$(date +%Y%m%d_%H%M%S).json"

    if [ -f "$config_file" ]; then
        mkdir -p "$BACKUP_DIR"
        cp "$config_file" "$backup_file"
        log_success "Backup created: $backup_file"
        echo "$backup_file"
    else
        log_warning "No existing config to backup"
        return 1
    fi
}

# Strip JSONC comments and handle placeholders
strip_jsonc_and_process() {
    local input_file=$1
    local output_file=$2

    # Use node to properly handle JSONC and placeholders
    node -e "
        const fs = require('fs');
        const jsonc = fs.readFileSync('$input_file', 'utf8');

        // Remove single-line comments
        let json = jsonc.replace(/\\/\\/.*$/gm, '');

        // Remove multi-line comments
        json = json.replace(/\\/\\*[\\s\\S]*?\\*\\//g, '');

        // Replace \${input:...} placeholders with empty strings for Claude Code
        json = json.replace(/\\\$\\{input:([^}]+)\\}/g, '');

        // Parse and re-stringify to ensure valid JSON
        try {
            const parsed = JSON.parse(json);
            fs.writeFileSync('$output_file', JSON.stringify(parsed, null, 2));
            console.log('JSONC processed successfully');
        } catch (e) {
            console.error('Invalid JSON:', e.message);
            process.exit(1);
        }
    "
}

# Main function
main() {
    local dry_run=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                dry_run=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    log_info "Starting MCP configuration import..."
    log_info "Project root: $PROJECT_ROOT"

    # Check dependencies
    check_dependencies

    # Check if canonical config exists
    if [ ! -f "$CANONICAL_CONFIG" ]; then
        log_error "Canonical config not found: $CANONICAL_CONFIG"
        exit 1
    fi

    log_success "Canonical config found: $CANONICAL_CONFIG"

    if [ "$dry_run" = true ]; then
        log_info "DRY RUN MODE - No changes will be made"

        # Show what would be done
        log_info "Would create backup of: $CLAUDE_CONFIG"
        log_info "Would process JSONC from: $CANONICAL_CONFIG"
        log_info "Would write to: $CLAUDE_CONFIG"

        # Preview the content
        echo
        log_info "Preview of processed JSON:"
        node -e "
            const fs = require('fs');
            const jsonc = fs.readFileSync('$CANONICAL_CONFIG', 'utf8');
            let json = jsonc.replace(/\\/\\/.*$/gm, '').replace(/\\/\\*[\\s\\S]*?\\*\\//g, '');
            json = json.replace(/\\\$\\{input:([^}]+)\\}/g, '');
            const parsed = JSON.parse(json);
            console.log(JSON.stringify(parsed, null, 2));
        "

        exit 0
    fi

    # Create backup
    create_backup "$CLAUDE_CONFIG"

    # Process JSONC and create Claude config
    log_info "Processing JSONC and placeholders..."
    strip_jsonc_and_process "$CANONICAL_CONFIG" "$CLAUDE_CONFIG"

    log_success "Configuration imported successfully!"
    log_info "Claude Code config updated: $CLAUDE_CONFIG"

    # Test the configuration
    log_info "Testing configuration..."
    if node -e "
        const fs = require('fs');
        const config = JSON.parse(fs.readFileSync('$CLAUDE_CONFIG', 'utf8'));
        console.log('Valid JSON ✓');
        console.log('Servers configured:', Object.keys(config.mcpServers || {}).length);
    "; then
        log_success "Configuration test passed!"
    else
        log_error "Configuration test failed!"
        exit 1
    fi

    log_info "Please restart Claude Code to apply changes"
}

# Run main function
main "$@"
