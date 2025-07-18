Param(
    [switch]$DryRun
)

# GitHub Issues Creator Script
#
# Usage:
#   1. Set environment variable: $env:GITHUB_TOKEN = "your_github_token_here"
#   2. Dry run mode: .\create_issues_api.ps1 -DryRun
#   3. Real execution: .\create_issues_api.ps1
#
# This script reads docs/REFRACTORING_TASKS.md, extracts unchecked tasks,
# and creates GitHub issues in the DNYoussef/AIVillage repository.

# Ensure GITHUB_TOKEN is set
if (-not $env:GITHUB_TOKEN) {
    Write-Error "GITHUB_TOKEN environment variable is not set. Please set it before running the script."
    exit 1
}

$repo = "DNYoussef/AIVillage"
$token = $env:GITHUB_TOKEN
$headers = @{
    Authorization = "Bearer $token"
    "User-Agent"  = "AIVillage-Script"
}

# Read the backlog file
$lines = Get-Content docs/REFRACTORING_TASKS.md

# Pattern to detect unchecked tasks
$pattern = '^\s*-\s*\[\s*\]\s*(.+)$'

foreach ($line in $lines) {
    if ($line -match $pattern) {
        $title = $Matches[1].Trim()
        $body  = $title  # Modify to include full context if needed
        $labels = @("refactoring")  # Add additional static labels or map dynamically

        $issuePayload = @{
            title  = $title
            body   = $body
            labels = $labels
        } | ConvertTo-Json

        $uri = "https://api.github.com/repos/$repo/issues"

        if ($DryRun) {
            Write-Host "DRY-RUN: POST $uri -Body $issuePayload"
        } else {
            Invoke-RestMethod -Uri $uri -Method Post -Headers $headers -ContentType "application/json" -Body $issuePayload
            Write-Host "Created issue: $title"
        }
    }
}
