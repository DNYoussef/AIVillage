"""Lightweight Sleep ▬ Dream weight update helpers."""
import torch


def apply_delta(model: torch.nn.Module, delta, scale: float = 0.2) -> None:
    with torch.no_grad():
        for p, d in zip(model.parameters(), delta, strict=False):
            p.add_(d.to(p.dtype) * scale)
