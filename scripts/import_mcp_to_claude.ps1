<#
.SYNOPSIS
    Import MCP configuration from canonical JSONC to Claude Code format
.DESCRIPTION
    Processes the canonical MCP configuration (infra/mcp/servers.jsonc) and imports it
    into Claude Code's format (.roo/mcp.json) with proper JSONC comment removal
    and placeholder handling.
.PARAMETER DryRun
    Preview changes without making them
.EXAMPLE
    .\import_mcp_to_claude.ps1
    .\import_mcp_to_claude.ps1 -DryRun
#>
param(
    [switch]$DryRun
)

# Color codes for output
$GREEN = "`e[32m"
$YELLOW = "`e[33m"
$RED = "`e[31m"
$RESET = "`e[0m"

function Write-Info {
    param($Message)
    Write-Host "${GREEN}[INFO]${RESET} $Message"
}

function Write-Success {
    param($Message)
    Write-Host "${GREEN}[SUCCESS]${RESET} $Message"
}

function Write-Warning {
    param($Message)
    Write-Host "${YELLOW}[WARNING]${RESET} $Message"
}

function Write-Error {
    param($Message)
    Write-Host "${RED}[ERROR]${RESET} $Message"
}

# Get project root
$projectRoot = Split-Path -Parent $PSScriptRoot
Write-Info "Starting MCP configuration import..."
Write-Info "Project root: $projectRoot"

# Define file paths
$canonicalConfig = Join-Path $projectRoot "infra/mcp/servers.jsonc"
$claudeConfig = Join-Path $projectRoot ".roo/mcp.json"
$backupDir = Join-Path $projectRoot ".mcp/backups"

# Check if canonical config exists
if (-not (Test-Path $canonicalConfig)) {
    Write-Error "Canonical config not found: $canonicalConfig"
    exit 1
}
Write-Success "Canonical config found: $canonicalConfig"

# Create backup directory if it doesn't exist
if (-not (Test-Path $backupDir)) {
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
}

# Create backup if claude config exists
if (Test-Path $claudeConfig) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupFile = Join-Path $backupDir "mcp_${timestamp}.json"

    if (-not $DryRun) {
        Copy-Item $claudeConfig $backupFile
        Write-Success "Backup created: $backupFile"
    }
    else {
        Write-Info "Would create backup of: $claudeConfig"
    }
}

# Read and process JSONC
Write-Info "Processing JSONC and placeholders..."
$content = Get-Content $canonicalConfig -Raw

# Remove JSONC comments
$jsonContent = $content -replace '(?<!\\)//.*$', '' -replace '(?s)/\*.*?\*/', ''

# Handle ${input:...} placeholders by replacing with empty strings
$jsonContent = $jsonContent -replace '\$\{input:[^}]+\}', ''

# Validate JSON
try {
    $json = $jsonContent | ConvertFrom-Json

    # Convert back to clean JSON
    $cleanJson = $json | ConvertTo-Json -Depth 10

    if ($DryRun) {
        Write-Info "DRY RUN MODE - No changes will be made"
        Write-Info "Preview of processed JSON:"
        Write-Host $cleanJson
    }
    else {
        # Write to Claude config
        $cleanJson | Out-File -FilePath $claudeConfig -Encoding UTF8
        Write-Success "Configuration imported successfully!"
        Write-Success "Claude Code config updated: $claudeConfig"

        # Test configuration
        Write-Info "Testing configuration..."
        $testJson = Get-Content $claudeConfig -Raw | ConvertFrom-Json
        $serverCount = ($testJson.mcpServers | Get-Member -MemberType NoteProperty).Count
        Write-Success "Configuration test passed! ($serverCount servers configured)"
        Write-Info "Please restart Claude Code to apply changes"
    }
}
catch {
    Write-Error "Failed to process JSON: $_"
    exit 1
}
