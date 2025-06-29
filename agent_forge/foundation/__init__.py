"""Foundation utilities for Agent Forge."""

from .quiet_star import START_TOK, END_TOK
from .bitnet import q_bitnet
import torch

__all__ = ["START_TOK", "END_TOK", "q_bitnet", "bootstrap"]


def bootstrap(base_model: torch.nn.Module) -> torch.nn.Module:
    """Attach Quiet-STaR tokens and BitNet quantisation."""
    base_model.config.special_tokens = [START_TOK, END_TOK]
    for p in base_model.parameters():
        p.data = q_bitnet(p.data)
    return base_model
