# Groks in the Shadows

"Groks in the Shadows" is an internal training discipline used to detect early
emergence of grok-like behaviour in language models. It combines the
**Quiet‑STaR** thought layer and the **Grokfast** optimizer to monitor hidden
states and gradient dynamics long before the validation loss visibly improves.

The technique operates in two phases:

1. **Shadow Grok Monitoring**
   - During each mini-batch, the training loop records a window of recent
gradients using `GrokfastTask`.
   - Intrinsic dimension, entropy and the variance of these gradients are
continuously logged.
   - If gradient variance collapses while perplexity remains high, the model is
said to be *grokking in the shadows*. This indicates that internal structure is
forming even though external metrics have not yet caught up.

2. **Adaptive Hyperparameter Boost**
   - When shadow grokking is detected, the optimizer temporarily increases the
learning rate of the slowest parameters while clamping rapid oscillations.
   - The self-modelling gate is enabled to run deeper reflection passes, using
the Quiet‑STaR tokens to produce hidden reasoning traces.
   - Once validation performance surges, the training loop reverts to the
standard schedule and records the event in the evolution log.

The goal is to accelerate the transition from memorisation to generalisation by
reacting to subtle signs of grokking. Implementation details can be found in the
`agent_forge/training` modules referenced in `docs/geometry_aware_training.md`.
