# credit: torch-twonn repo, MIT licence  ➜  https://github.com/VRehnberg/torch-twonn
import torch


def twonn(x: torch.Tensor, k1: int = 2, k2: int = 3) -> float:
    """Facco et al. Two-NN estimator - intrinsic dimensionality of `x` (N×D).
    Returns float ID_nl.
    """
    with torch.no_grad():
        dist = torch.cdist(x, x, p=2)
        d1 = dist.kthvalue(k1, dim=-1).values
        d2 = dist.kthvalue(k2, dim=-1).values
        mu = d1 / d2
        eps = 1e-12
        id_est = (mu.log().mean() / (1 - mu).clamp(min=eps).log().mean()).clamp_min(1e-3)
    return id_est.item()
