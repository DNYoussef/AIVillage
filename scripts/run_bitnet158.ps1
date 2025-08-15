# BitNet 1.58-bit Training Pipeline
# Wrapper script for complete BitNet training with self-generated data

param(
    [Parameter(Mandatory=$true)]
    [string]$ModelPath,

    [string]$OutputDir = "",
    [int]$NumSamples = 500,
    [int]$Steps = 1000,
    [int]$BatchSize = 2,
    [int]$GradAccum = 8,
    [float]$LearningRate = 5e-5,
    [float]$LambdaWarmupFrac = 0.4,
    [int]$RMSNormPostAttn = 1,
    [int]$MaxNewTokens = 512,
    [int]$Seed = 42,
    [string]$Device = "auto"
)

# Set up environment
$ErrorActionPreference = "Stop"

# Set up environment
. "$PSScriptRoot\setup_env.ps1"

# Check for virtual environment
$venv = "$env:AIV_ROOT\.venv"
$py = "$venv\Scripts\python.exe"

if (!(Test-Path $venv)) {
    Write-Host "[ERROR] Virtual environment not found at $venv" -ForegroundColor Red
    Write-Host "Please run OFFLINE_install.ps1 first" -ForegroundColor Yellow
    exit 1
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "BitNet 1.58-bit Training Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Validate model path
if (-not (Test-Path $ModelPath)) {
    Write-Error "Model path does not exist: $ModelPath"
    exit 1
}

$ModelName = Split-Path $ModelPath -Leaf
Write-Host "Model: $ModelName" -ForegroundColor Green

# Set output directory if not provided
if (-not $OutputDir) {
    $OutputDir = "$env:AIV_ARTIFACTS_DIR\bitnet158\$ModelName"
}

# Create artifacts directory
$ArtifactsDir = "$env:AIV_ARTIFACTS_DIR\bitnet158"
New-Item -ItemType Directory -Path $ArtifactsDir -Force | Out-Null

Write-Host "Output directory: $OutputDir" -ForegroundColor Green
Write-Host "Artifacts directory: $ArtifactsDir" -ForegroundColor Green

# Step 1: Generate self-training data
Write-Host "`n🔄 Step 1: Generating self-training data..." -ForegroundColor Yellow

$DatasetPath = "$ArtifactsDir\selfgen_${ModelName}_${NumSamples}.jsonl"

$GenerateArgs = @(
    "--model_path", $ModelPath,
    "--out", $DatasetPath,
    "--num", $NumSamples,
    "--max_new_tokens", $MaxNewTokens,
    "--seed", $Seed,
    "--device", $Device
)

Write-Host "Running: python selfgen/generate.py $($GenerateArgs -join ' ')" -ForegroundColor Gray

try {
    $GenerateResult = & $py "$env:AIV_ROOT\src\production\compression\selfgen\generate.py" @GenerateArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Data generation failed with exit code $LASTEXITCODE"
    }
    Write-Host "✅ Data generation completed" -ForegroundColor Green
}
catch {
    Write-Error "Failed to generate training data: $_"
    exit 1
}

# Verify dataset was created
if (-not (Test-Path $DatasetPath)) {
    Write-Error "Dataset file was not created: $DatasetPath"
    exit 1
}

$DatasetSize = (Get-Content $DatasetPath | Measure-Object).Count
Write-Host "📊 Generated $DatasetSize samples" -ForegroundColor Blue

# Step 2: Train BitNet model with λ warmup
Write-Host "`n🚀 Step 2: Training BitNet model with λ warmup..." -ForegroundColor Yellow

$TrainArgs = @(
    "--base_model", $ModelPath,
    "--dataset", $DatasetPath,
    "--out_dir", $OutputDir,
    "--steps", $Steps,
    "--bsz", $BatchSize,
    "--grad_accum", $GradAccum,
    "--lr", $LearningRate,
    "--lambda_warmup_frac", $LambdaWarmupFrac,
    "--rmsnorm_post_attn", $RMSNormPostAttn
)

Write-Host "Running: python train_bitnet.py $($TrainArgs -join ' ')" -ForegroundColor Gray

try {
    $TrainResult = & $py "$env:AIV_ROOT\src\production\compression\train_bitnet.py" @TrainArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed with exit code $LASTEXITCODE"
    }
    Write-Host "✅ Training completed" -ForegroundColor Green
}
catch {
    Write-Error "Failed to train BitNet model: $_"
    exit 1
}

# Verify training output
if (-not (Test-Path "$OutputDir\training_manifest.json")) {
    Write-Error "Training manifest not found. Training may have failed."
    exit 1
}

# Display results
Write-Host "`n🎉 BitNet 1.58-bit training completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

# Read and display training manifest
$Manifest = Get-Content "$OutputDir\training_manifest.json" | ConvertFrom-Json

Write-Host "📋 Training Summary:" -ForegroundColor Blue
Write-Host "  Base Model: $($Manifest.base_model)" -ForegroundColor Gray
Write-Host "  Dataset: $($Manifest.dataset_path)" -ForegroundColor Gray
Write-Host "  Training Steps: $($Manifest.training_steps)" -ForegroundColor Gray
Write-Host "  Lambda Warmup Fraction: $($Manifest.lambda_warmup_frac)" -ForegroundColor Gray
Write-Host "  RMSNorm Post-Attention: $($Manifest.rmsnorm_post_attn)" -ForegroundColor Gray
Write-Host "  Final Lambda: $($Manifest.final_lambda)" -ForegroundColor Gray
Write-Host "  Compression Ready: $($Manifest.compression_ready)" -ForegroundColor Gray

Write-Host "`n📁 Output Files:" -ForegroundColor Blue
Write-Host "  Model: $OutputDir" -ForegroundColor Gray
Write-Host "  Dataset: $DatasetPath" -ForegroundColor Gray
Write-Host "  Manifest: $OutputDir\training_manifest.json" -ForegroundColor Gray

# Display W&B info
Write-Host "`n📊 Monitoring:" -ForegroundColor Blue
Write-Host "  W&B logs: $env:WANDB_DIR" -ForegroundColor Gray
Write-Host "  W&B mode: $env:WANDB_MODE" -ForegroundColor Gray

if ($env:WANDB_MODE -eq "offline") {
    Write-Host "  💡 To sync W&B logs: wandb sync $env:WANDB_DIR" -ForegroundColor Yellow
}

Write-Host "`n🔧 Next Steps:" -ForegroundColor Blue
Write-Host "  1. The trained model is ready for compression with bitnet.py::compress()" -ForegroundColor Gray
Write-Host "  2. Use the model for inference or further fine-tuning" -ForegroundColor Gray
Write-Host "  3. Evaluate on benchmarks to measure compression impact" -ForegroundColor Gray

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Pipeline completed successfully! 🎯" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
