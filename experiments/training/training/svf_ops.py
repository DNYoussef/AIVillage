"""svf_ops.py.
----------
Batched Singular-Value Fine-tuning helpers (Transformer² style).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from AIVillage.experimental.training.svf.ops import batched_svd  # NeurIPS-22 SVF kernel


def apply_svf(model: nn.Module, z: dict[str, Tensor], clamp: float = 0.05) -> None:
    """In-place Σ scaling for Linear layers listed in ``z``.

    Parameters
    ----------
    model : nn.Module
        The backbone whose weights will be mutated.
    z : dict[str, Tensor]
        Mapping of module names to scale deltas for their singular values.
    clamp : float
        Max |dS|/S to avoid exploding activations.
    """
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or name not in z:
            continue

        with torch.no_grad():
            U, S, Vh = batched_svd(mod.weight, full_matrices=False)
            dS = z[name].to(S.device).clamp_(-clamp, clamp)
            S.mul_(1 + dS)
            mod.weight.copy_(U @ torch.diag(S) @ Vh)


if __name__ == "__main__":
    lin = nn.Linear(4, 4, bias=False)
    delta = {"": torch.full((min(lin.weight.size()),), 0.01)}
    apply_svf(lin, delta)
    print(lin.weight)
