Param(
  [string]$Root = "D:\AIVillage",
  [string]$Py = "python"
)
# Use online-friendly env on this machine
$env:HF_HUB_OFFLINE = "0"
$env:HF_DATASETS_OFFLINE = "0"
$env:TRANSFORMERS_OFFLINE = "0"

. "$PSScriptRoot\setup_env.ps1" -Root $Root

$bundle     = "$env:AIV_BUNDLE_DIR"
$wheelhouse = "$bundle\wheelhouse"
$modelsDir  = "$bundle\models"
$hubCache   = "$bundle\hf_cache\hub"
$dsCache    = "$bundle\hf_cache\datasets"

$mk = @($bundle, $wheelhouse, $modelsDir, $hubCache, $dsCache)
foreach ($d in $mk) { if (!(Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null } }

# 1. Download wheels
Copy-Item "benchmarks\requirements.txt" "$bundle\requirements.txt" -Force
& $Py -m pip download -r "$bundle\requirements.txt" --dest "$wheelhouse"
if ($LASTEXITCODE -ne 0) { throw "pip download failed" }

# 2. Snapshot the three models to bundle\models
& $Py - <<'PY'
import os, yaml, re
from pathlib import Path
from huggingface_hub import snapshot_download

root   = os.environ.get("AIV_ROOT", r"D:\AIVillage")
bundle = Path(root, "bundle")
models_yaml = Path(root, "models", "models.yaml")
outdir = bundle / "models"
outdir.mkdir(parents=True, exist_ok=True)

def safe(rid:str)->str: return rid.replace("/", "__").replace("\\","__")

# Download our 3 seed models
models = [
    {"id": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "type": "coding"},
    {"id": "Qwen/Qwen2.5-Math-1.5B-Instruct", "type": "math"},
    {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "type": "logic"}
]

for m in models:
    rid = m["id"]
    dst = outdir / safe(rid)
    print(f"[HF] {rid} -> {dst}")
    snapshot_download(repo_id=rid, revision="main", local_dir=str(dst), local_dir_use_symlinks=False, allow_patterns=["*"])
print("[OK] model snapshots ready.")
PY
if ($LASTEXITCODE -ne 0) { throw "model snapshot failed" }

# 3. Warm dataset caches into bundle\hf_cache\datasets for our suites
& $Py scripts\ONLINE_prefetch_datasets.py --bundle "$bundle"
if ($LASTEXITCODE -ne 0) { throw "dataset prefetch failed" }

Write-Host "`n[BUNDLE READY] Copy the ENTIRE folder: $bundle -> OFFLINE box at the same path."
