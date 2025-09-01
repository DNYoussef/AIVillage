"""Cogment model configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CogmentConfig:
    """Configuration for Cogment model with RefinementCore and ACT."""

    # Model dimensions (Option A: ~25M params)
    d_model: int = 320  # Hidden dimension
    n_head: int = 8  # Number of attention heads (d_kv = d_model // n_head = 40)
    n_layers: int = 7  # Number of transformer layers
    d_ff: int = 1280  # Feed-forward dimension (4 * d_model)

    # Vocabulary and sequence
    vocab_size: int = 16000  # Reduced vocab for parameter efficiency
    max_seq_len: int = 2048
    rope_base: int = 10000

    # RefinementCore specific
    memory_fusion_dim: int = 512  # Memory gating dimension
    refinement_steps: int = 8  # Default refinement iterations
    min_refinement_steps: int = 2  # Minimum steps before halting
    max_refinement_steps: int = 16  # Maximum steps before force halt

    # ACT halting parameters
    act_threshold: float = 0.99  # Halting threshold (1 - ε)
    ponder_cost_weight: float = 0.1  # λ for ponder cost in loss
    halt_epsilon: float = 0.01  # Small value to ensure progress

    # Training
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # Memory parameters (for interface compatibility)
    ltm_capacity: int = 1024  # Long-term memory slots
    ltm_dim: int = 512  # Memory embedding dimension

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.d_model % self.n_head == 0, f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"
        assert (
            self.min_refinement_steps <= self.max_refinement_steps
        ), "min_refinement_steps must be <= max_refinement_steps"
        assert 0 < self.act_threshold < 1, "act_threshold must be in (0, 1)"
        assert self.ponder_cost_weight >= 0, "ponder_cost_weight must be non-negative"

    @property
    def d_kv(self) -> int:
        """Key-value dimension per head."""
        return self.d_model // self.n_head
