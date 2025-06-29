# Geometry-Aware Training Modules

This document summarises the additional components proposed for the self-adaptive "geometry-aware" training pipeline. Most of these modules are experimental stubs only. Features such as expert vectors and advanced sleep/dream cycles are **not implemented** in the current repository.

## Key External Projects

- **Transformer² (SakanaAI/self-adaptive-llms)** — provides the `ExpertVector` class for Σ-scale expert vectors and prompt baking utilities.
- **SVF (syp2ysy/SVF)** — lightweight singular value fine-tuning kernel used by `apply_svf` in `training/svf_ops.py`.
- **torch-twonn** — Two-NN intrinsic dimension estimator.  `geometry/snapshot.py`
  will automatically fall back to the local implementation in
  `geometry/id_twonn.py` if this package is unavailable.
- **grokfast** — custom optimizer giving slow gradients extra weight, referenced in the training loop.
- **Intelligence_at_the_edge_of_chaos** — computes Langton λ and entropy for the PID loop in `training/pid_edgechaos.py`.
- **unexpected-benefits-of-self-modeling** — hidden-state predictor that regularises weights in the self-modeling task.
- **SleepNet-DreamNet** — encoder/decoder pair used during sleep and dream cycles (`training/sleep_and_dream.py`).
- **1.58-bit BitNet quantizer** — deployed via `foundation/bitnet.py` for ternary compression.

## Integration Points

1. **Expert Vectors**: `training/expert_vectors.py` is a placeholder for SVF-based expert vectors. Use `ExpertVectorSystem.train_expert_vector_from_texts` to build vectors from raw text or curriculum tasks once this feature is implemented. `PromptBakingManager` currently does not load real vectors.
2. **Intrinsic Dimension Monitoring**: `geometry/snapshot.py` wraps the Two‑NN estimator and token entropy probes for a complete geometry snapshot each mini‑batch.
3. **Grokfast Optimizer**: `training/grokfast_opt.py` exposes an AugmentedAdam wrapper with a `slow_power()` probe used whenever `pre_grok` is `True`.
4. **Edge-of-Chaos PID**: `training/pid_edgechaos.py` adjusts the learning rate based on complexity metrics to keep the network near λ ≈ 0.5.
5. **Unexpected Self-Modeling**: `training/self_modeling.py` includes a hidden-state predictor that adds an MSE term to the language modeling loss.
6. **Sleep/Dream Cycle**: `training/sleep_and_dream.py` implements frozen encoder/decoder modules that propose weight deltas during the dream phase.
7. **Self-Model Grok Gate**: `phase3/self_modeling_gate.py` ensures each level is mastered internally before promotion and is called from `training/train_level.py`.
8. **BitNet Compression**: `foundation/bitnet.py` offers a ternary quantization function applied before and after training for efficient deployment.

For usage instructions see `docs/external_modules_roadmap.md` and the in-code docstrings of each module.
