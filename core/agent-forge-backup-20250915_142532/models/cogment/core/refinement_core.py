"""RefinementCore: The heart of Cogment's iterative reasoning."""

from __future__ import annotations

from hrrm.common.transformer_blocks import RMSNorm, SwiGLU
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryGate(nn.Module):
    """Memory-gated fusion for integrating LTM into refinement."""

    def __init__(self, d_model: int, memory_dim: int):
        super().__init__()
        self.d_model = d_model
        self.memory_dim = memory_dim

        # Projection layers
        self.memory_proj = nn.Linear(memory_dim, d_model, bias=False)
        self.gate_proj = nn.Linear(2 * d_model, d_model)  # Fixed: concat of hidden_states + memory_context
        self.norm = RMSNorm(d_model)

    def forward(self, hidden_states: torch.Tensor, memory: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply memory-gated fusion.

        Args:
            hidden_states: Current hidden states [B, N, d_model]
            memory: Optional memory embeddings [B, M, memory_dim]

        Returns:
            fused_states: Memory-gated hidden states [B, N, d_model]
        """
        if memory is None:
            return hidden_states

        B, N, D = hidden_states.shape
        _, M, mem_D = memory.shape

        # Project memory to model dimension
        memory_proj = self.memory_proj(memory)  # [B, M, d_model]

        # Attention-like fusion with memory
        # Compute attention scores between hidden states and memory
        scores = torch.einsum("bnd,bmd->bnm", hidden_states, memory_proj)  # [B, N, M]
        scores = scores / (D**0.5)
        attn_weights = F.softmax(scores, dim=-1)  # [B, N, M]

        # Weighted memory context
        memory_context = torch.einsum("bnm,bmd->bnd", attn_weights, memory_proj)  # [B, N, d_model]

        # Gating mechanism
        gate_input = torch.cat([hidden_states, memory_context], dim=-1)  # [B, N, 2*d_model]
        gate = torch.sigmoid(self.gate_proj(gate_input))  # [B, N, d_model]

        # Apply gate
        fused = hidden_states + gate * memory_context
        return self.norm(fused)


class RefinementCore(nn.Module):
    """
    Core refinement module with MLP encoder and three prediction heads.

    Architecture:
    - MLP encoder transforms input representations
    - Memory-gated fusion integrates long-term memory
    - Three heads: y (base prediction), Δy (residual), halt_p (ACT halting)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config.d_model

        # MLP encoder for input processing
        self.encoder = nn.Sequential(
            RMSNorm(d_model),
            SwiGLU(d_model, mult=4),  # 4x expansion like in transformer
            nn.Dropout(config.dropout),
            RMSNorm(d_model),
            nn.Linear(d_model, d_model, bias=False),
        )

        # Memory gating for LTM integration
        self.memory_gate = MemoryGate(d_model, config.ltm_dim)

        # Three prediction heads
        self.y_head = nn.Linear(d_model, config.vocab_size, bias=False)  # Base prediction
        self.delta_y_head = nn.Linear(d_model, config.vocab_size, bias=False)  # Residual edit
        self.halt_head = nn.Linear(d_model, 1, bias=True)  # Halting probability

        # Residual scaling for stable training
        self.delta_scale = nn.Parameter(torch.ones(1) * 0.1)

        # Initialize heads
        self._init_heads()

    def _init_heads(self):
        """Initialize prediction heads with appropriate scales."""
        # Initialize y_head and delta_y_head with Xavier
        nn.init.xavier_uniform_(self.y_head.weight)
        nn.init.xavier_uniform_(self.delta_y_head.weight)

        # Initialize halt_head to start with low halt probabilities
        nn.init.constant_(self.halt_head.weight, 0.0)
        nn.init.constant_(self.halt_head.bias, -2.0)  # sigmoid(-2) ≈ 0.12

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, N, d_model]
        memory: torch.Tensor | None = None,  # [B, M, ltm_dim]
        step: int = 0,  # Current refinement step
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through refinement core.

        Args:
            hidden_states: Input hidden states [B, N, d_model]
            memory: Optional long-term memory [B, M, ltm_dim]
            step: Current refinement step (for potential step-dependent behavior)

        Returns:
            y_logits: Base prediction logits [B, N, vocab_size]
            delta_logits: Residual edit logits [B, N, vocab_size]
            halt_prob: Halting probabilities [B, N, 1]
            refined_states: Refined hidden states [B, N, d_model]
        """
        # Encode input representations
        encoded = self.encoder(hidden_states)  # [B, N, d_model]

        # Memory-gated fusion
        fused = self.memory_gate(encoded, memory)  # [B, N, d_model]

        # Compute predictions
        y_logits = self.y_head(fused)  # [B, N, vocab_size]
        delta_logits = self.delta_y_head(fused)  # [B, N, vocab_size]
        halt_logits = self.halt_head(fused)  # [B, N, 1]

        # Apply sigmoid to halt logits for probability
        halt_prob = torch.sigmoid(halt_logits)  # [B, N, 1]

        # Scale delta for stable training
        delta_logits = delta_logits * self.delta_scale

        # Refined states (residual connection)
        refined_states = hidden_states + encoded  # [B, N, d_model]

        return y_logits, delta_logits, halt_prob, refined_states

    def compute_prediction(
        self,
        y_logits: torch.Tensor,  # [B, N, vocab_size]
        delta_logits: torch.Tensor,  # [B, N, vocab_size]
        alpha: float = 1.0,  # Mixing coefficient for delta
    ) -> torch.Tensor:
        """
        Combine base and residual predictions.

        Args:
            y_logits: Base prediction logits
            delta_logits: Residual edit logits
            alpha: Mixing coefficient (1.0 = full residual, 0.0 = base only)

        Returns:
            combined_logits: Final prediction logits [B, N, vocab_size]
        """
        return y_logits + alpha * delta_logits

    def count_parameters(self) -> int:
        """Count trainable parameters in RefinementCore."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RefinementOutput:
    """Output container for refinement step."""

    def __init__(
        self,
        y_logits: torch.Tensor,
        delta_logits: torch.Tensor,
        halt_prob: torch.Tensor,
        refined_states: torch.Tensor,
        combined_logits: torch.Tensor | None = None,
    ):
        self.y_logits = y_logits
        self.delta_logits = delta_logits
        self.halt_prob = halt_prob
        self.refined_states = refined_states
        self.combined_logits = combined_logits if combined_logits is not None else (y_logits + delta_logits)

    def __getitem__(self, key):
        """Allow tuple-like unpacking for backward compatibility."""
        if key == 0:
            return self.y_logits
        elif key == 1:
            return self.delta_logits
        elif key == 2:
            return self.halt_prob
        elif key == 3:
            return self.refined_states
        else:
            raise IndexError(f"Index {key} out of range")
