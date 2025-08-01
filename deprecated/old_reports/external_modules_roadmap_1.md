# External Modules Roadmap

This document lists useful third-party projects referenced by the Agent Forge pipeline. They provide building blocks for geometry-aware training and compression.

| Repo / module | Purpose |
| --- | --- |
| `SakanaAI/self-adaptive-llms` | Σ-scale expert vectors and prompt baking utilities |
| `syp2ysy/SVF` | Fast SVD kernel for Singular Value Fine-tuning |
| `torch-twonn` | Intrinsic dimension estimator (Two-NN) |
| `ironjr/grokfast` | Augmented Adam optimizer with slow-gradient boost |
| `Intelligence_at_the_edge_of_chaos` | Complexity metrics (Langton λ, entropy) |
| `unexpected-benefits-of-self-modeling` | Hidden state predictor for self-modeling |
| `SleepNet-DreamNet` | Dreaming encoder/decoder pair |
| Hugging Face BitNet blog code | 1.58‑bit ternary quantizer |

Install the Python packages via `pip` or clone the repositories as needed. These modules are not bundled with the repository but can be integrated where the training stubs reference them.
