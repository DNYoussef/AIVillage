"""Foundation utilities for Agent Forge."""

import torch

from .bitnet import q_bitnet
from .quiet_star import END_TOK, START_TOK

__all__ = ["END_TOK", "START_TOK", "bootstrap", "q_bitnet"]


def bootstrap(base_model: torch.nn.Module) -> torch.nn.Module:
    """Attach Quiet-STaR tokens and BitNet quantisation."""
    base_model.config.special_tokens = [START_TOK, END_TOK]
    for p in base_model.parameters():
        p.data = q_bitnet(p.data)
    return base_model
