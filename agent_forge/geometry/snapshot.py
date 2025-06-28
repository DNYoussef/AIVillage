import torch
from .id_twonn import twonn


def geom_snapshot(hidden: torch.Tensor) -> dict:
    """Return nonlinear ID, linear PCA dim, ratio, entropy from a sample of hidden states."""
    id_nl = twonn(hidden)                                    # nonlinear
    # linear effective dimension (99 % variance)
    _, S, _ = torch.pca_lowrank(hidden, q=min(256, hidden.size(-1)))
    id_lin = (S.cumsum(0) / S.sum()).lt(.99).sum().item()
    ratio  = id_nl / max(id_lin, 1e-6)
    entropy = -(hidden.softmax(-1) * hidden.log_softmax(-1)).sum(-1).mean().item()
    return dict(ID_nl=id_nl, ID_lin=id_lin, ratio=ratio, entropy=entropy)
