"""Scratchpad supervisor for Start-of-Thought / End-of-Thought instrumentation."""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ReasonerConfig:
    """Configuration for HRM Reasoner model."""

    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 16
    n_head: int = 8
    max_seq_len: int = 2048
    rope_base: int = 10000

    # HRM parameters
    max_H: int = 4
    inner_T: int = 4

    # Reasoning tokens
    start_thought_token: str = "<SoT>"
    end_thought_token: str = "<EoT>"

    # Self-consistency parameters
    self_consistency_k: int = 5

    # Training
    dropout: float = 0.0
    tie_embeddings: bool = True


class ScratchpadSupervisor(nn.Module):
    """Supervisor head for managing scratchpad reasoning spans."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.thought_detector = nn.Linear(d_model, 2)  # Binary: thought or not
        self.thought_gate = nn.Linear(d_model, 1)  # Gating mechanism

    def forward(self, hidden_states):
        """
        Detect and gate reasoning spans.

        Args:
            hidden_states: [B, N, D] hidden states from transformer

        Returns:
            thought_logits: [B, N, 2] thought vs non-thought predictions
            thought_gates: [B, N, 1] gating values for thoughts
        """
        thought_logits = self.thought_detector(hidden_states)
        thought_gates = torch.sigmoid(self.thought_gate(hidden_states))

        return thought_logits, thought_gates
