#!/usr/bin/env powershell
# Download the 3 seed models for Agent Forge pipeline
# Automates model download and sets up D:\ structure

Param(
    [string]$Root = "D:\AIVillage",
    [switch]$Force,
    [switch]$SkipExisting
)

# Set up environment
. "$PSScriptRoot\setup_env.ps1" -Root $Root

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Agent Forge Seed Model Downloader" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Target: $Root" -ForegroundColor White
Write-Host ""

# Determine python path
$venv = Join-Path $Root ".venv"
if (Test-Path $venv) {
    $py = Join-Path $venv "Scripts\python.exe"
    Write-Host "Using virtual environment: $venv" -ForegroundColor Green
}
else {
    $py = "python"
    Write-Host "Using system Python (consider running OFFLINE_install.ps1 first)" -ForegroundColor Yellow
}

# Ensure huggingface_hub available
try {
    & $py -c "import huggingface_hub; print('huggingface_hub_ok')" > $null
    Write-Host "huggingface_hub available" -ForegroundColor Green
}
catch {
    Write-Host "huggingface_hub not found. Installing..." -ForegroundColor Yellow
    & $py -m pip install huggingface_hub transformers torch --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install required packages" -ForegroundColor Red
        exit 1
    }
}

$models = @(
    @{
        id        = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        type      = "coding"
        revision  = "main"
        safe_name = "Qwen__Qwen2.5-Coder-1.5B-Instruct"
    },
    @{
        id        = "Qwen/Qwen2.5-Math-1.5B-Instruct"
        type      = "math"
        revision  = "main"
        safe_name = "Qwen__Qwen2.5-Math-1.5B-Instruct"
    },
    @{
        id        = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        type      = "logic"
        revision  = "main"
        safe_name = "deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B"
    }
)

Write-Host "Models to download:" -ForegroundColor Blue
foreach ($m in $models) {
    $td = Join-Path $env:AIV_MODELS_DIR $m.safe_name
    $exists = Test-Path (Join-Path $td "config.json")
    $status = if ($exists) { "[EXISTS]" } else { "[DOWNLOAD]" }
    Write-Host "  $($m.id) ($($m.type)) - $status" -ForegroundColor $(if ($exists) { "Green" } else { "Yellow" })
}
Write-Host ""

$downloaded = 0
$skipped = 0
$failed = 0

foreach ($model in $models) {
    $modelId = $model.id
    $targetDir = Join-Path $env:AIV_MODELS_DIR $model.safe_name
    Write-Host "Processing: $modelId -> $targetDir" -ForegroundColor Yellow

    # Check existing
    if ((Test-Path (Join-Path $targetDir "config.json")) -and -not $Force) {
        if ($SkipExisting) {
            Write-Host "  Skipping existing" -ForegroundColor Green
            $skipped++
            continue
        }
        else {
            $resp = Read-Host "  Model exists. Overwrite? (y/N)"
            if ($resp -notmatch "^[Yy]") {
                Write-Host "  Skipped by user" -ForegroundColor Yellow
                $skipped++
                continue
            }
        }
    }

    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null

    # Write python downloader to temp file
    $pyLines = @(
        '#!/usr/bin/env python3',
        'import sys',
        'from huggingface_hub import snapshot_download',
        'from pathlib import Path',
        '',
        'model_id = sys.argv[1]',
        'target_dir = sys.argv[2]',
        'revision = sys.argv[3]',
        '',
        'print("Downloading {} to {}".format(model_id, target_dir))',
        '',
        'try:',
        '    snapshot_download(',
        '        repo_id=model_id,',
        '        revision=revision,',
        '        local_dir=target_dir,',
        '        local_dir_use_symlinks=False,',
        '        resume_download=True',
        '    )',
        '    config_path = Path(target_dir) / "config.json"',
        '    if not config_path.exists():',
        '        print("[x] Download failed: config.json not found in {}".format(target_dir))',
        '        sys.exit(1)',
        '    print("[v] Download successful: {}".format(config_path))',
        '    sys.exit(0)',
        'except Exception as e:',
        '    print("[x] Download failed: {}".format(e))',
        '    sys.exit(1)'
    )
    $tempPy = Join-Path $env:TEMP "download_model.py"
    $pyLines | Set-Content -Path $tempPy -Encoding UTF8

    try {
        & $py $tempPy $modelId $targetDir $model.revision
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Download completed" -ForegroundColor Green
            $downloaded++
        }
        else {
            Write-Host "  Download failed (python exit code $LASTEXITCODE)" -ForegroundColor Red
            $failed++
        }
    }
    catch {
        Write-Host "  Error running python: $_" -ForegroundColor Red
        $failed++
    }
    finally {
        Remove-Item $tempPy -ErrorAction SilentlyContinue
    }

    Write-Host ""
}

Write-Host "Updating models.yaml" -ForegroundColor Blue
$yamlLines = @(
    "# Agent Forge Seed Models for EvoMerge Pipeline",
    "# Generated by download_seed_models.ps1",
    "",
    "models:"
)
foreach ($model in $models) {
    $targetDir = Join-Path $env:AIV_MODELS_DIR $model.safe_name
    if (Test-Path (Join-Path $targetDir "config.json")) {
        $description = switch ($model.type) {
            'coding' { 'Code generation and programming specialist' }
            'math' { 'Mathematical reasoning and problem solving specialist' }
            'logic' { 'Logical reasoning and general intelligence specialist' }
            default { 'General purpose model' }
        }
        $yamlLines += "  - id: $($model.id)"
        $yamlLines += "    local: `"$targetDir`""
        $yamlLines += "    type: $($model.type)"
        $yamlLines += "    revision: $($model.revision)"
        $yamlLines += "    description: `"$description`""
        $yamlLines += ""
    }
}
$modelsYaml = Join-Path $Root "models\models.yaml"
$yamlLines | Set-Content -Path $modelsYaml -Encoding UTF8
Write-Host "  Updated: $modelsYaml" -ForegroundColor Green

Write-Host "Download Summary" -ForegroundColor Cyan
Write-Host "  Downloaded: $downloaded" -ForegroundColor Green
Write-Host "  Skipped: $skipped" -ForegroundColor Yellow
Write-Host "  Failed: $failed" -ForegroundColor Red

if ($failed -gt 0) {
    Write-Host "Some downloads failed. Check logs and network." -ForegroundColor Yellow
    exit 1
}

$readyModels = 0
foreach ($model in $models) {
    $targetDir = Join-Path $env:AIV_MODELS_DIR $model.safe_name
    if (Test-Path (Join-Path $targetDir "config.json")) {
        $readyModels++
    }
}

Write-Host "Models ready: $readyModels/3" -ForegroundColor White
Write-Host "Storage path: $env:AIV_MODELS_DIR" -ForegroundColor White
Write-Host "Models config: $modelsYaml" -ForegroundColor White

if ($readyModels -eq 3) {
    Write-Host "All models ready." -ForegroundColor Green
    Write-Host "Next steps: Run EvoMerge: ./scripts/run_evomerge.ps1" -ForegroundColor White
}
else {
    Write-Host "Not all models downloaded. Pipeline not ready." -ForegroundColor Yellow
    exit 1
}

Write-Host "Seed model download complete!" -ForegroundColor Green
