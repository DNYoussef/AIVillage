"""
geometry/snapshot.py
--------------------
Fast utilities for sensing representational geometry every mini-batch.
"""

from __future__ import annotations
import torch, math
from torch import Tensor
try:
    from twonn import twonn  # external Two-NN implementation
except Exception:  # pragma: no cover - fallback for minimal installs
    from .id_twonn import twonn  # lightweight local estimator
from torch.linalg import svdvals                  # native batched SVD (PyTorch ≥2.1)

__all__ = ["snapshot", "GeomState"]


class GeomState(dict):
    """Typed wrapper so mypy/autocomplete know the keys."""
    ID_nl: float          # nonlinear intrinsic dimension
    ID_lin: int           # linear PCA dimension (99 % var)
    ratio: float          # compression ratio ID_nl / ID_lin
    entropy: float        # token-wise softmax entropy


@torch.inference_mode()
def snapshot(hidden: Tensor, pca_q: int = 128) -> GeomState:
    """Return intrinsic dimension stats.

    Parameters
    ----------
    hidden : (B, L, D) final-layer hidden states
    pca_q  : int        number of eigvecs for torch.pca_lowrank
    """
    B, L, D = hidden.shape
    flat = hidden.reshape(B * L, D).float().cpu()     # move to RAM for SVD if needed

    # --- 1. nonlinear ID ------------------------------------------------------
    id_nl = float(twonn(flat))                        # Facco et al. estimator

    # --- 2. linear effective dimension ---------------------------------------
    # pca_lowrank is O(N D q);  q=128 is plenty for LM dims 1024-4096
    _, S, _ = torch.pca_lowrank(flat, q=min(pca_q, D // 2))
    id_lin = int((S.cumsum(0) / S.sum()).lt(0.99).sum())  # 99 % energy

    # --- 3. token-distribution entropy ---------------------------------------
    probs = torch.softmax(hidden.to(dtype=torch.float32), dim=-1)
    entropy = float(-(probs * probs.log()).sum(dim=-1).mean())

    return GeomState(
        ID_nl=id_nl,
        ID_lin=id_lin,
        ratio=id_nl / max(id_lin, 1e-6),
        entropy=entropy,
    )

if __name__ == "__main__":
    x = torch.randn(2, 4, 16)
    print(snapshot(x))
