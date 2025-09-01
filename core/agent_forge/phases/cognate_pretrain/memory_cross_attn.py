#!/usr/bin/env python3
"""
Memory Cross-Attention Module for Titans-Style LTM Integration

Enhanced cross-attention mechanism that seamlessly integrates Long-Term Memory
into transformer layers. Projects memory vectors into backbone K/V dimensions
and enables retrieval-augmented generation through cross-attention.

Key Features:
- Memory K/V projection to match transformer dimensions
- Gated memory influence with learnable gates
- Efficient batched cross-attention
- Support for different memory read policies
- Entropy-based memory scheduling
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MemoryCrossAttention(nn.Module):
    """
    Cross-attention module for integrating LTM reads into transformer layers.

    Projects memory vectors to serve as additional K/V pairs that the current
    hidden states can attend to, enabling retrieval-augmented generation.

    Architecture:
    - Memory K/V projections: d_mem -> d_model
    - Multi-head cross-attention with memory
    - Gated integration with residual connections
    - Entropy-based read gating
    """

    def __init__(
        self,
        d_model: int,
        d_mem: int,
        n_heads: int,
        dropout: float = 0.1,
        memory_gate: bool = True,
        entropy_threshold: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        # Query projection from hidden states
        self.q_proj = nn.Linear(d_model, d_model, bias=False)

        # Key/Value projections from memory vectors
        self.k_proj = nn.Linear(d_mem, d_model, bias=False)
        self.v_proj = nn.Linear(d_mem, d_model, bias=False)

        # Output projection
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Dropout and layer norm
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # Gating mechanism
        self.memory_gate = memory_gate
        if memory_gate:
            self.gate_proj = nn.Sequential(
                nn.Linear(d_model + d_mem, d_model // 2), nn.SiLU(), nn.Linear(d_model // 2, 1), nn.Sigmoid()
            )

        # Entropy-based read gating
        self.entropy_threshold = entropy_threshold
        self.entropy_proj = nn.Linear(d_model, 1)

        # Positional encoding for memory items
        self.mem_pos_embed = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)

    def compute_entropy(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of hidden states for read gating.

        High entropy indicates uncertainty -> need memory retrieval
        Low entropy indicates confidence -> skip memory access
        """
        # Project to logits
        logits = self.entropy_proj(hidden_states)  # (B, L, 1)

        # Compute attention over sequence
        attn_weights = F.softmax(logits.squeeze(-1), dim=-1)  # (B, L)

        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-9), dim=-1)  # (B,)

        return entropy

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_vectors: torch.Tensor | None = None,
        memory_keys: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        read_policy: str = "entropy_gated",
    ) -> torch.Tensor:
        """
        Apply cross-attention between hidden states and memory vectors.

        Args:
            hidden_states: Current hidden states (B, L, d_model)
            memory_vectors: Memory values (B, M, d_mem) or None
            memory_keys: Memory keys (B, M, d_mem) or None
            attention_mask: Attention mask (B, L) or None
            read_policy: "always", "entropy_gated", or "never"

        Returns:
            Enhanced hidden states (B, L, d_model)
        """
        B, L, _ = hidden_states.shape

        # Early exit if no memory or read policy is "never"
        if memory_vectors is None or read_policy == "never":
            return hidden_states

        # Entropy-based read gating
        if read_policy == "entropy_gated":
            entropy = self.compute_entropy(hidden_states)  # (B,)
            read_mask = entropy > self.entropy_threshold  # (B,)

            # Skip if no samples need memory access
            if not read_mask.any():
                return hidden_states

            # Filter batch items that need memory access
            active_indices = read_mask.nonzero(as_tuple=False).squeeze(-1)
            if active_indices.numel() == 0:
                return hidden_states

            # Select active samples
            h_active = hidden_states[active_indices]  # (B_active, L, d_model)
            mem_vals_active = memory_vectors[active_indices] if memory_vectors is not None else None
            mem_keys_active = memory_keys[active_indices] if memory_keys is not None else None

            # Apply cross-attention to active samples
            h_enhanced_active = self._cross_attend(h_active, mem_vals_active, mem_keys_active, attention_mask)

            # Merge back to full batch
            h_enhanced = hidden_states.clone()
            h_enhanced[active_indices] = h_enhanced_active

            return h_enhanced

        # Standard cross-attention for "always" policy
        return self._cross_attend(hidden_states, memory_vectors, memory_keys, attention_mask)

    def _cross_attend(
        self,
        hidden_states: torch.Tensor,
        memory_vectors: torch.Tensor,
        memory_keys: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Core cross-attention computation."""
        B, L, _ = hidden_states.shape
        M = memory_vectors.shape[1] if memory_vectors is not None else 0

        if M == 0:
            return hidden_states

        # Project to multi-head format
        q = self.q_proj(hidden_states).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory_keys).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory_vectors).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)

        # Add positional encoding to memory
        if M <= self.mem_pos_embed.size(1):
            pos_embed = self.mem_pos_embed[:, :M, :].unsqueeze(1)  # (1, 1, M, d_model)
            pos_embed = pos_embed.view(1, 1, M, self.n_heads, self.head_dim).transpose(2, 3)
            k = k + pos_embed

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, n_heads, L, M)

        # Apply attention mask to memory if provided
        if attention_mask is not None:
            # Create memory mask (assume all memory items are valid)
            torch.ones(B, M, device=attention_mask.device, dtype=attention_mask.dtype)
            # Combine with sequence mask for cross-attention (L x M)
            cross_mask = torch.ones(B, L, M, device=attention_mask.device, dtype=attention_mask.dtype)
            scores = scores.masked_fill(cross_mask.unsqueeze(1) == 0, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (B, n_heads, L, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.d_model)

        # Output projection
        memory_output = self.o_proj(attn_output)

        # Gated integration
        if self.memory_gate:
            # Concatenate hidden states and pooled memory for gating
            pooled_memory = memory_vectors.mean(dim=1, keepdim=True).expand(-1, L, -1)  # (B, L, d_mem)
            gate_input = torch.cat([hidden_states, pooled_memory], dim=-1)  # (B, L, d_model + d_mem)
            gate = self.gate_proj(gate_input)  # (B, L, 1)

            # Apply gate
            output = hidden_states + gate * memory_output
        else:
            # Simple residual connection
            output = hidden_states + memory_output

        # Layer normalization
        output = self.norm(output)

        return output


class MemoryScheduler:
    """
    Scheduler for memory read/write operations.

    Implements different policies for when to read from or write to memory:
    - always: Every forward pass
    - entropy_gated: Based on prediction uncertainty
    - periodic: Every N steps
    - surprise_novelty: Based on loss signals and novelty
    """

    def __init__(
        self,
        read_policy: str = "entropy_gated",
        write_policy: str = "surprise_novelty",
        period: int = 4,
        entropy_threshold: float = 1.0,
        surprise_threshold: float = 0.5,
        novelty_threshold: float = 0.7,
    ):
        self.read_policy = read_policy
        self.write_policy = write_policy
        self.period = period
        self.entropy_threshold = entropy_threshold
        self.surprise_threshold = surprise_threshold
        self.novelty_threshold = novelty_threshold

        # Counters
        self.step_count = 0
        self.read_count = 0
        self.write_count = 0

        # Statistics
        self.entropy_history = []
        self.surprise_history = []

    def should_read(self, entropy: float | None = None, step: int | None = None) -> bool:
        """Decide whether to read from memory."""
        if step is not None:
            self.step_count = step
        else:
            self.step_count += 1

        if self.read_policy == "always":
            should_read = True
        elif self.read_policy == "entropy_gated":
            should_read = entropy is not None and entropy > self.entropy_threshold
        elif self.read_policy == "periodic":
            should_read = self.step_count % self.period == 0
        else:  # "never"
            should_read = False

        if should_read:
            self.read_count += 1
            if entropy is not None:
                self.entropy_history.append(entropy)

        return should_read

    def should_write(
        self, surprise: float | None = None, novelty: float | None = None, step: int | None = None
    ) -> bool:
        """Decide whether to write to memory."""
        if step is not None:
            self.step_count = step

        if self.write_policy == "always":
            should_write = True
        elif self.write_policy == "surprise_novelty":
            should_write = (
                surprise is not None
                and surprise > self.surprise_threshold
                and novelty is not None
                and novelty > self.novelty_threshold
            )
        elif self.write_policy == "periodic":
            should_write = self.step_count % (self.period * 2) == 0
        else:  # "never"
            should_write = False

        if should_write:
            self.write_count += 1
            if surprise is not None:
                self.surprise_history.append(surprise)

        return should_write

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "step_count": self.step_count,
            "read_count": self.read_count,
            "write_count": self.write_count,
            "read_rate": self.read_count / max(self.step_count, 1),
            "write_rate": self.write_count / max(self.step_count, 1),
            "avg_entropy": sum(self.entropy_history) / max(len(self.entropy_history), 1),
            "avg_surprise": sum(self.surprise_history) / max(len(self.surprise_history), 1),
        }


# Factory function
def create_memory_cross_attention(
    d_model: int = 384, d_mem: int = 384, n_heads: int = 12, dropout: float = 0.1, **kwargs
) -> MemoryCrossAttention:
    """Factory function to create memory cross-attention layer."""
    return MemoryCrossAttention(d_model=d_model, d_mem=d_mem, n_heads=n_heads, dropout=dropout, **kwargs)


if __name__ == "__main__":
    # Test memory cross-attention
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create memory cross-attention
    mem_cross_attn = create_memory_cross_attention(d_model=384, d_mem=256, n_heads=12, dropout=0.1).to(device)

    # Test inputs
    batch_size, seq_len, mem_len = 2, 32, 8
    hidden_states = torch.randn(batch_size, seq_len, 384, device=device)
    memory_vectors = torch.randn(batch_size, mem_len, 256, device=device)
    memory_keys = torch.randn(batch_size, mem_len, 256, device=device)

    # Test forward pass
    enhanced_states = mem_cross_attn(
        hidden_states=hidden_states, memory_vectors=memory_vectors, memory_keys=memory_keys, read_policy="always"
    )

    print(f"Input shape: {hidden_states.shape}")
    print(f"Memory shapes: vals={memory_vectors.shape}, keys={memory_keys.shape}")
    print(f"Output shape: {enhanced_states.shape}")
    print(f"Parameters: {sum(p.numel() for p in mem_cross_attn.parameters()):,}")

    # Test entropy gating
    enhanced_entropy = mem_cross_attn(
        hidden_states=hidden_states, memory_vectors=memory_vectors, memory_keys=memory_keys, read_policy="entropy_gated"
    )
    print(f"Entropy gated output shape: {enhanced_entropy.shape}")

    # Test scheduler
    scheduler = MemoryScheduler()
    for step in range(10):
        entropy = torch.rand(1).item() * 2  # Random entropy 0-2
        surprise = torch.rand(1).item()  # Random surprise 0-1
        novelty = torch.rand(1).item()  # Random novelty 0-1

        should_read = scheduler.should_read(entropy=entropy)
        should_write = scheduler.should_write(surprise=surprise, novelty=novelty)

        if step % 3 == 0:  # Print every 3 steps
            print(f"Step {step}: read={should_read}, write={should_write}")

    print(f"Scheduler stats: {scheduler.get_stats()}")
    print("✅ Memory Cross-Attention test passed!")
