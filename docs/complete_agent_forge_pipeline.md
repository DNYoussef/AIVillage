# Complete Agent Forge Training Pipeline

This document details the full training and optimization process for building self‑evolving AI agents. It expands upon the overview in `agent_forge_pipeline_overview.md` and complements the repository layout described in `system_overview.md`.

> **Note**
> The pipeline below describes the long‑term vision for Agent Forge. Only a
> subset of these phases has been implemented in the current codebase. The
> implemented pieces are limited to basic RAG retrieval with the FastAPI server,
> a minimal training loop and experimental merging utilities. All advanced stages
> (Quiet‑STaR, expert vectors, ADAS, etc.) remain conceptual and are included here
> as future work.

## Phase 1 – Model Foundation & Merging
1. **Evolution and Merge Pipeline** – Start with three specialized models and generate eight merged candidates using linear, SLERP, TIES, DARE, Frankenmerge and DFS techniques. Evaluate all models, mutate the best performers and recombine weaker ones for 50 generations to select the strongest foundation.
2. **Quiet‑STaR Integration** – Modify the architecture to produce parallel thought tokens using learnable `<|startofthought|>` and `<|endofthought|>` markers. A lightweight implementation is available in `agent_forge/training/quiet_star.py`, though full integration remains a goal for future versions.
3. **Initial Compression** – Apply 1.58‑bit quantization and convert to BitNetModel format before heavy training begins.

## Phase 2 – Core Training Loop
1. **Curriculum Creation** – Automatically produce up to 1000 assessment tasks ranging from very easy to cutting edge. Determine the failure point and build ten curriculum levels mixing organic data, synthetic tasks, RAG queries and multi‑agent scenarios.
2. **Training Cycle** – For each level the model receives tasks with a self‑awareness prefix, generates internal thoughts via Quiet‑STaR, outputs a final answer and receives scores for both answer quality and reasoning. Reinforcement updates are applied and sleep/dream cycles run every 50 rounds. Quiet‑STaR support is experimental and disabled by default.
3. **Self‑Modeling** – At regular intervals the model trains on its own generated texts across five temperature ranges (0‑0.05 to 0.95‑1.0) to refine reasoning and creativity.

## Phase 3 – Self‑Modeling & Expert Vectors
1. **Quiet‑STaR Self‑Reflection** – The model analyses its own thoughts to identify errors and generate improved reasoning traces. *(future work)*
2. **Expert Vector Creation** – Successful reasoning patterns are converted into expert vectors using Singular Value Fine‑Tuning (SVF). A dispatch system selects which experts to use based on task type and confidence. *(future work)*

## Phase 4 – Advanced Integration
1. **Prompt Baking** – Iterate on prompts and bake effective reasoning strategies directly into model weights.
2. **Tool & Memory Integration** – Connect the agent to external tools and the RAG system for persistent knowledge storage and retrieval.
3. **ADAS Optimization** – A meta‑model repeatedly evaluates the agent, proposes architectural improvements and refines the model until performance plateaus. *(future work)*

## Phase 5 – Deployment
1. **Compression & Packaging** – Apply additional compression passes for efficient deployment.
2. **Monitoring & Continuous Learning** – Deployed agents continue to log new experiences which feed back into the training loop.

This pipeline combines evolutionary merging, curriculum learning, self‑modeling and automated architecture search to produce robust agents that improve over time. For the high‑level summary see `agent_forge_pipeline_overview.md`.

Implementation helpers can be found in the codebase:
- `agent_forge/training/self_modeling.py` for the self‑modeling loop
- `agent_forge/phase3/self_modeling_gate.py` for the internal grok gate used during curriculum advancement
- `agent_forge/training/train_level.py` for running a level with the self-modeling gate
- `agent_forge/training/prompt_baking.py` for automated prompt baking
- `agent_forge/training/expert_vectors.py` for creating and applying expert vectors *(planned feature)*
- `agent_forge/training/identity.py` for identity formation and the moral framework baker
- `agent_forge/training/quiet_star.py` for an experimental Quiet‑STaR module
