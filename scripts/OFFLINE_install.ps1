Param(
  [string]$Root = "D:\AIVillage",
  [string]$Py = "python"
)
. "$PSScriptRoot\setup_env.ps1" -Root $Root

$bundle     = "$env:AIV_BUNDLE_DIR"
$wheelhouse = "$bundle\wheelhouse"
$reqs       = "$bundle\requirements.txt"
$venv       = "$Root\.venv"

if (!(Test-Path $bundle)) { throw "Bundle not found at $bundle. Copy it from the ONLINE box first." }

# Create venv
if (!(Test-Path $venv)) {
  & $Py -m venv $venv
  if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}
$pip = "$venv\Scripts\pip.exe"
$pyV = "$venv\Scripts\python.exe"

# Offline install from wheelhouse
& $pip install --no-index --find-links "$wheelhouse" -r "$reqs"
if ($LASTEXITCODE -ne 0) { throw "offline pip install failed" }

# Copy models snapshot into AIV_MODELS_DIR
robocopy "$bundle\models" "$env:AIV_MODELS_DIR" /MIR | Out-Null

# Copy HF caches into D: caches
robocopy "$bundle\hf_cache\hub" "$env:HF_HOME" /MIR    | Out-Null
robocopy "$bundle\hf_cache\datasets" "$env:HF_DATASETS_CACHE" /MIR | Out-Null

Write-Host "[OK] Offline install ready. Venv: $venv"
