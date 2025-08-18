# Agent Forge Pipeline Implementation

This repository now contains a lightweight model management layer used by the
Agent Forge experiments.  The code lives in `src/agent_forge/models/` and
provides three focused utilities:

1. **Seed download** – `seed_downloader.py` fetches the initial seed models
   from Hugging Face.  By default it downloads:
   - `Qwen/Qwen2.5-Coder-1.5B-Instruct`
   - `Qwen/Qwen2-1.5B`
   - `microsoft/phi-1_5`
2. **Storage cleanup** – `storage.py` prunes old model directories so that no
   more than eight models are kept locally.
3. **Benchmarking** – `benchmark.py` performs a trivial directory scan and
   reports file counts, total bytes and elapsed time.

A small CLI exposes these features:

```bash
python -m agent_forge.models.cli download-seeds
python -m agent_forge.models.cli start-evomerge --generations 3
python -m agent_forge.models.cli run-pipeline
```

The `run-pipeline` command downloads the seed models, cleans up the storage
location and benchmarks each model.  `start-evomerge` delegates to the
production EvoMerge pipeline when the optional dependency is available.

These modules provide the foundation for the broader Agent Forge pipeline while
remaining light-weight enough for testing and local experimentation.
