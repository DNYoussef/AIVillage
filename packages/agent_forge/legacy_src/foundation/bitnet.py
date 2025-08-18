"""Minimal BitNet 1.58-bit quantisation helpers."""

import torch


def q_bitnet(t: torch.Tensor, bits: float = 1.58) -> torch.Tensor:
    """Quantise tensor to ternary BitNet format (−1, 0, +1)."""
    s = t.abs().amax() / (2 ** (bits - 1) - 1 + 1e-9)
    return (t / s).round().clamp_(-1, 1) * s
