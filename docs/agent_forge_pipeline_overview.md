# Agent Forge Training Pipeline Overview

This document summarizes the full Agent Forge pipeline used to create self-improving AI agents. It condenses the detailed procedure described in the project documentation and implementation guide.

## Phase 1 – Model Foundation & Merging
1. **Evolution and Merge Pipeline** – Start with three specialized base models. Use multiple merge techniques (linear, SLERP, TIES, DARE, Frankenmerge, DFS) to create an initial population of merged models. Evaluate, select top performers, mutate and recombine over many generations until the best foundation model is obtained.
2. **Quiet‑STaR Integration** – Modify the architecture to generate parallel "thought" tokens. Introduce learnable `<|startofthought|>` and `<|endofthought|>` tokens for internal monologue generation.
3. **Initial Compression** – Apply 1.58‑bit quantization and convert the model to the BitNet format to reduce size before heavy training.

## Phase 2 – Core Training Loop
1. **Curriculum Creation** – Automatically generate questions across difficulty levels to determine the model’s baseline. Build a 10‑level curriculum mixing organic data, synthetic examples, retrieval‑augmented tasks and multi‑agent scenarios.
2. **Training Cycle** – For each level, present tasks with self‑awareness prefixes, generate internal thoughts with Quiet‑STaR, produce final answers, score answers and reasoning, and update the model with reinforcement signals. Sleep and dream cycles consolidate knowledge and encourage creative exploration.
3. **Self‑Modeling** – Every few cycles, the model trains on its own generated texts across a range of temperatures to refine reasoning patterns and creativity.

## Phase 3 – Self‑Modeling & Expert Vectors
1. **Quiet‑STaR Self‑Reflection** – Periodically run deeper self‑modeling where the model analyses its thoughts, identifies errors and generates improved reasoning.
2. **Expert Vector Creation** – Successful reasoning patterns are converted into specialized expert vectors using Singular Value Fine‑tuning (SVF). A dispatch system selects appropriate vectors depending on task type and confidence.

## Phase 4 – Advanced Integration
1. **Prompt Baking** – Incorporate new knowledge and reasoning strategies directly into model weights via an iterative prompt testing and baking process.
2. **Tool & Memory Integration** – Connect the model to external tools and the shared RAG system, enabling persistent knowledge storage and retrieval.
3. **ADAS Optimization** – A meta‑model repeatedly tests, grades and adjusts the agent’s architecture until further improvements plateau.
4. After training completes the code automatically runs `ADASystem.optimize_agent_architecture` to produce the final optimized model saved in `adas_optimized_model`.

## Phase 5 – Deployment
1. **Compression & Packaging** – Apply final compression passes for efficient deployment.
2. **Monitoring & Continuous Learning** – Once deployed, the agent continues to gather data, log behaviour and feed new experiences back into the training loop.

This pipeline combines evolutionary merging, reinforcement learning, self‑reflection and automated architectural search to produce robust agents that improve over time.

Implementation helpers:
- `training/self_modeling.py` – self-modeling trainer used in Phase 3.
- `training/prompt_baking.py` – utilities for baking prompt strategies.
- `training/expert_vectors.py` – simple expert vector system for vector composition.
- `training/identity.py` – basic identity formation and moral framework utilities.

For a step-by-step description of each phase see [complete_agent_forge_pipeline.md](complete_agent_forge_pipeline.md).
