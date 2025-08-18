# EvoMerge Enhanced: Dual-Phase Evolution with Multi-Objective Optimization

## Overview

The enhanced EvoMerge system implements a sophisticated dual-phase evolutionary model merging approach with configurable benchmark suites and early stopping criteria.

## Key Features

### 1. Dual-Phase Generation Loop

Each generation follows this pattern:
1. **Phase Pre**: 3 parents → MERGE → 8 children → BENCHMARK → SELECT top 3
2. **Phase Post**: 3 selected → MERGE+MUTATE → 8 children → BENCHMARK → SELECT top 3

This approach ensures both exploration (pre-phase) and exploitation (post-phase) within each generation.

### 2. Modular Benchmark Suites

Four specialized benchmark suites:
- **General**: `mmlu`, `gsm8k`, `hellaswag`, `arc_easy`, `winogrande`
- **Coding**: `humaneval`, `mbpp`, `boolq`, `winogrande`
- **Math**: `gsm8k`, `mmlu_stem`, `winogrande`, `arc_challenge`
- **Writing**: `mmlu`, `truthfulqa_mc`, `hellaswag`, `boolq`

### 3. Early Stopping with Configurable Windows

- `--early_delta_pct`: Minimum improvement percentage (default: 0.25%)
- `--window_gens`: Window size for improvement calculation (default: 3 generations)
- Stops when fitness improvement < threshold over the specified window

### 4. Multi-Objective Optimization (NSGA-II Preserved)

- Maintains existing NSGA-II selection algorithm
- Supports Pareto front calculation and crowding distance
- Suite-specific objectives defined in YAML configurations

## CLI Usage

```bash
python evolutionary_tournament.py \
    --max_gens 50 \
    --early_delta_pct 0.25 \
    --window_gens 3 \
    --suite coding \
    --phase_both \
    --merge_set linear,slerp,ties,dare,task_arith,dfs
```

## Configuration Files

### Benchmark Suites (`benchmarks/suites/*.yaml`)

```yaml
objectives: [mmlu_score, gsm8k_score, hellaswag_score]
task_groups:
  - name: core
    tasks: [mmlu, gsm8k, hellaswag]
  - name: reasoning
    tasks: [arc_easy, winogrande]
```

### Model Metadata (`models/models.yaml`)

```yaml
models:
  - id: Qwen/Qwen2.5-Coder-1.5B-Instruct
    local: "D:\\AIVillage\\models\\Qwen__Qwen2.5-Coder-1.5B-Instruct"
    type: coding
    description: "Code generation specialist"
```

## Environment Setup

Run the setup script to configure required environment variables:

```powershell
.\scripts\setup_env.ps1 -Permanent
```

This sets:
- `AIV_ROOT=D:\AIVillage`
- `AIV_MODELS_DIR=D:\AIVillage\models`
- `AIV_BENCHMARKS_DIR=D:\AIVillage\benchmarks\results`
- `WANDB_DIR=D:\AIVillage\wandb`
- `WANDB_MODE=offline`

## W&B Integration

- Default offline mode with results in `D:\AIVillage\wandb`
- Tags: `generation:<id>`, `phase:{pre|post}`, `suite:<name>`
- Project: `AIVillage-EvoMerge`
- Group: `<model_dirname>`

## Directory Structure

```
D:\AIVillage\
├── artifacts\gens\G####\phase_{pre|post}\child_##\    # Generated models
├── benchmarks\results\G####\phase_{pre|post}\child_##\  # Benchmark results
├── models\                                              # Source models
├── wandb\                                              # W&B logs
└── checkpoints\                                        # Evolution checkpoints
```

## Advanced Merge Techniques

The system cycles through these merge combinations:
- Linear + TIES
- Linear + DARE
- SLERP + TIES
- SLERP + DARE
- Linear + Task Arithmetic
- SLERP + Task Arithmetic
- Linear + DFS
- SLERP + DFS

## Testing

Run the enhanced test suite:

```bash
python -m pytest tests/test_evomerge_enhanced.py -v
```

Tests cover:
- Benchmark orchestrator functionality
- Model suite auto-detection
- Early stopping logic
- Dual-phase generation workflow
- Integration scenarios

## Performance Considerations

- **Memory**: Models are cleaned up after each phase to manage memory
- **Storage**: Intermediate models are stored in structured generation directories
- **Parallelization**: Benchmarking supports parallel evaluation
- **Checkpointing**: Saves state every 5 generations for recovery

## Monitoring and Logging

- Comprehensive logging at INFO/DEBUG levels
- Real-time progress bars with generation tracking
- W&B metrics and artifact logging
- Detailed evolution reports in `evolution_report.txt`
