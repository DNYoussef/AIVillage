Param(
  [string]$Generation = "G0001",
  [string]$EvoEntry = "evolutionary_tournament.py",
  [int]$MaxGens = 50,
  [float]$EarlyDeltaPct = 0.25,
  [int]$WindowGens = 3
)

. "$PSScriptRoot\setup_env.ps1"

# Ensure venv
$venv = "$env:AIV_ROOT\.venv"
$py = "$venv\Scripts\python.exe"

# 1) Seed benchmarks (suite-aware)
& $py evomerge\bench_driver.py

# 2) Start EvoMerge — adjust CLI flags to your tournament driver
$args = @(
  $EvoEntry,
  "--max_gens", $MaxGens.ToString(),
  "--early_delta_pct", $EarlyDeltaPct.ToString(),
  "--window_gens", $WindowGens.ToString()
)
Write-Host "[EVOMERGE] python $($args -join ' ')"
& $py @args
