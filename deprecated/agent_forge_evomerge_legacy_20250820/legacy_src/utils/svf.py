import torch
from torch import nn


class SVFLinear(nn.Module):
    """Linear layer with explicit singular value factors."""

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.U = nn.Parameter(torch.randn(out_features, rank))
        self.s = nn.Parameter(torch.ones(rank))
        self.V = nn.Parameter(torch.randn(rank, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.U @ torch.diag(self.s) @ self.V
        return x @ W.T


def replace_linear_with_svf(model: nn.Module, rank: int) -> None:
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, SVFLinear(module.in_features, module.out_features, rank))
        else:
            replace_linear_with_svf(module, rank)
