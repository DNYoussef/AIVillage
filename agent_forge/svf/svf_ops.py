"""Apply batched SVF deltas to Linear layers."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .ops import batched_svd


def apply_svf(model: nn.Module, z: dict[str, Tensor], clamp: float = 0.05) -> None:
    """In-place Σ-scaling for Linear layers listed in ``z``."""
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or name not in z:
            continue
        with torch.no_grad():
            U, S, Vh = batched_svd(mod.weight, full_matrices=False)
            dS = z[name].to(S.device).clamp_(-clamp, clamp)
            S.mul_(1 + dS)
            mod.weight.copy_(U @ torch.diag(S) @ Vh)
