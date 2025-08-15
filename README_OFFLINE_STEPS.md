# Agent Forge: OFFLINE/ONLINE setup to download 3 models, set benchmarks, enable W&B, and start EvoMerge

## ONLINE machine (has internet)
1) Open PowerShell as admin.
2) From repo root:
   ```powershell
   ./scripts/setup_env.ps1 -Root D:\AIVillage
   ./scripts/ONLINE_bundle.ps1 -Root D:\AIVillage -Py python
   ```
3) Copy the entire folder `D:\AIVillage\bundle` to the OFFLINE box at the same path.

## OFFLINE dev box (no internet; VS Code + Claude Code)
1) In PowerShell:
   ```powershell
   ./scripts/setup_env.ps1 -Root D:\AIVillage
   ./scripts/OFFLINE_install.ps1 -Root D:\AIVillage -Py python
   ```
2) (Optional) Set your W&B key for later cloud sync:
   ```powershell
   $env:WANDB_API_KEY = "<your_wandb_key>"  # keeps WANDB_MODE=offline; you can sync later
   ```
3) Run the initial seed benchmarks and start EvoMerge:
   ```powershell
   ./scripts/run_evomerge.ps1 -Generation G0001
   ```

## Notes
* All caches, models, artifacts, and W&B files live on **D:\AIVillage**.
* W&B is **offline** by default; later, on any machine with internet:
  ```powershell
  ./scripts/wandb_sync.ps1 -Root D:\AIVillage
  ```
* Suites are modular: `benchmarks/suites/{coding,math,logic,general}.yaml`. Model→suite mapping comes from `models/models.yaml (type: ...)`.
