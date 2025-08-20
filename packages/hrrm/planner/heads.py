"""Planner-specific heads and configuration."""

from dataclasses import dataclass

import torch.nn as nn


@dataclass
class PlannerConfig:
    """Configuration for HRM Planner model."""

    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 16
    n_head: int = 8
    max_seq_len: int = 2048
    rope_base: int = 10000

    # HRM parameters
    max_H: int = 4
    inner_T: int = 4

    # Control tokens
    control_tokens: int = 5  # <PLAN>, <SUBGOAL>, <ACTION>, <CHECK>, <ENDPLAN>
    lambda_ctrl: float = 0.2

    # Training
    dropout: float = 0.0
    tie_embeddings: bool = True


class ControllerHead(nn.Module):
    """Auxiliary head to detect control tokens."""

    def __init__(self, d_model: int, n_ctrl: int):
        super().__init__()
        self.proj = nn.Linear(d_model, n_ctrl)

    def forward(self, h_last):
        """Use last token hidden state for control token prediction."""
        return self.proj(h_last)
