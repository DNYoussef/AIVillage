Param(
  [string]$Root = "D:\AIVillage"
)

$dirs = @(
  "$Root",
  "$Root\models",
  "$Root\benchmarks\results",
  "$Root\hf_cache\hub",
  "$Root\hf_cache\datasets",
  "$Root\hf_cache\transformers",
  "$Root\bundle",
  "$Root\artifacts",
  "$Root\wandb"
)
foreach ($d in $dirs) { if (!(Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null } }

$envs = @{
  "AIV_ROOT"              = "$Root"
  "AIV_MODELS_DIR"        = "$Root\models"
  "AIV_BENCHMARKS_DIR"    = "$Root\benchmarks\results"
  "AIV_BUNDLE_DIR"        = "$Root\bundle"
  "AIV_ARTIFACTS_DIR"     = "$Root\artifacts"
  "AIV_WANDB_DIR"         = "$Root\wandb"
  "HF_HOME"               = "$Root\hf_cache\hub"
  "HUGGINGFACE_HUB_CACHE" = "$Root\hf_cache\hub"
  "TRANSFORMERS_CACHE"    = "$Root\hf_cache\transformers"
  "HF_DATASETS_CACHE"     = "$Root\hf_cache\datasets"
  # offline-safe defaults; allow overrides via existing environment variables
  "HF_HUB_OFFLINE"        = $(if ($env:HF_HUB_OFFLINE) { $env:HF_HUB_OFFLINE } else { "1" })
  "HF_DATASETS_OFFLINE"   = $(if ($env:HF_DATASETS_OFFLINE) { $env:HF_DATASETS_OFFLINE } else { "1" })
  "TRANSFORMERS_OFFLINE"  = $(if ($env:TRANSFORMERS_OFFLINE) { $env:TRANSFORMERS_OFFLINE } else { "1" })
  "WANDB_MODE"            = $(if ($env:WANDB_MODE) { $env:WANDB_MODE } else { "offline" })
  "WANDB_DIR"             = "$Root\wandb"
}
foreach ($k in $envs.Keys) {
  [Environment]::SetEnvironmentVariable($k, $envs[$k], "User")
  Set-Item -Path ("Env:" + $k) -Value $envs[$k] -Force
}
Write-Host "[OK] Env set. Root=$Root"
