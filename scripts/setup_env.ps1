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
  "AIV_ROOT"             = "$Root"
  "AIV_MODELS_DIR"       = "$Root\models"
  "AIV_BENCHMARKS_DIR"   = "$Root\benchmarks\results"
  "AIV_BUNDLE_DIR"       = "$Root\bundle"
  "AIV_ARTIFACTS_DIR"    = "$Root\artifacts"
  "AIV_WANDB_DIR"        = "$Root\wandb"
  "HF_HOME"              = "$Root\hf_cache\hub"
  "HUGGINGFACE_HUB_CACHE"= "$Root\hf_cache\hub"
  "TRANSFORMERS_CACHE"   = "$Root\hf_cache\transformers"
  "HF_DATASETS_CACHE"    = "$Root\hf_cache\datasets"
  # offline-safe defaults; override on ONLINE box in the online script
  "HF_HUB_OFFLINE"       = "1"
  "HF_DATASETS_OFFLINE"  = "1"
  "TRANSFORMERS_OFFLINE" = "1"
  "WANDB_MODE"           = "offline"
  "WANDB_DIR"            = "$Root\wandb"
}
foreach ($k in $envs.Keys) {
  [Environment]::SetEnvironmentVariable($k, $envs[$k], "User")
  $env:$k = $envs[$k]
}
Write-Host "[OK] Env set. Root=$Root"
