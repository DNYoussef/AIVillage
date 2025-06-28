import torch


def apply_svf(layer: torch.nn.Linear, z: torch.Tensor, clamp: float = .05):
    """
    Rescale singular values of weight matrix in-place.
    z: same length as min(in,out), assumed small (≈0.05).
    """
    with torch.no_grad():
        U, S, Vh = torch.linalg.svd(layer.weight, full_matrices=False)
        z = z.clamp(-clamp, clamp)
        S = S * (1 + z)
        layer.weight.copy_(U @ torch.diag(S) @ Vh)
