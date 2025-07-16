import math

from scipy.optimize import minimize
import torch


def fit_hyperfunc(values: torch.Tensor):
    n = torch.arange(len(values), dtype=torch.float32)

    def fun(p):
        A, B, alpha, C, tau, D = p
        pred = (
            A * torch.sin(2 * math.pi * n * alpha)
            + B * torch.cos(2 * math.pi * n * alpha)
            + C * torch.exp(-n / tau)
            + D
        )
        return torch.norm(pred - values).item()

    init = [0, 0, 0.1, 0, 1.0, float(values.mean())]
    res = minimize(fun, init, method="Nelder-Mead")
    return res.x


def reconstruct(params, length):
    n = torch.arange(length, dtype=torch.float32)
    A, B, alpha, C, tau, D = params
    return (
        A * torch.sin(2 * math.pi * n * alpha)
        + B * torch.cos(2 * math.pi * n * alpha)
        + C * torch.exp(-n / tau)
        + D
    )
