# Agent Forge EvoMerge Legacy Code Deprecation

## Deprecation Date: August 20, 2025

This directory contains legacy EvoMerge implementations that have been consolidated into the production-ready Agent Forge pipeline.

## Files Moved to Deprecated

The following legacy implementations have been moved here:

- `legacy_production/` - Old production EvoMerge implementations
- `legacy_production_evomerge/` - Duplicate production implementations  
- `legacy_software/` - Software layer implementations
- `legacy_src/` - Source implementations
- `evomerge/` - Old evomerge directory

## Current Production Implementation

The production-ready EvoMerge implementation is now located at:

- **Core Implementation**: `packages/agent_forge/phases/evomerge.py`
- **Experiments**: `packages/agent_forge/experiments/`
- **Models**: `packages/agent_forge/models/hrrm_models/`
- **Outputs**: `packages/agent_forge/outputs/`
- **Benchmarks**: `packages/agent_forge/benchmarks/evomerge_datasets/`

## Key Features of Production Implementation

✅ **N-2 Generation Cleanup**: Maintains max 16 models on device  
✅ **Real Benchmark Integration**: HumanEval, GSM8K, HellaSwag, ARC  
✅ **HRRM Bootstrap**: Uses trained HRRM models as seed models  
✅ **6 Merge Techniques**: linear, slerp, ties, dare, frankenmerge, dfs  
✅ **Breeding Algorithm**: Top 2 → 6 children, Bottom 6 → 2 children  
✅ **Convergence Detection**: Variance threshold and plateau patience  

## Migration Guide

If you were using any legacy EvoMerge code:

1. **Update imports** to use `packages.agent_forge.phases.evomerge`
2. **Use new config structure** - see `HRRMEvoMergeConfig` in experiments
3. **Update model paths** to `packages/agent_forge/models/hrrm_models/`
4. **Update output paths** to `packages/agent_forge/outputs/`

## Removal Schedule

This deprecated code will be removed permanently on **September 20, 2025** (30 days from deprecation date).