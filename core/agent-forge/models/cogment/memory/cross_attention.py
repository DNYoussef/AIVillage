"""
Cross-attention mechanisms for memory reads.

Implements efficient cross-attention between query states and memory slots
with top-k preselection for scalable memory access.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionReader(nn.Module):
    """
    Cross-attention memory reader with top-k preselection.

    Efficiently retrieves relevant memory context by:
    1. Top-k preselection based on key similarity
    2. Cross-attention over selected slots
    3. Context aggregation and projection
    """

    def __init__(
        self,
        query_dim: int,
        memory_dim: int,
        output_dim: int,
        n_heads: int = 8,
        topk: int = 32,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.memory_dim = memory_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.topk = topk
        self.temperature = temperature

        assert output_dim % n_heads == 0, f"output_dim ({output_dim}) must be divisible by n_heads ({n_heads})"
        self.head_dim = output_dim // n_heads

        # Projection layers for multi-head attention
        self.q_proj = nn.Linear(query_dim, output_dim, bias=False)
        self.k_proj = nn.Linear(memory_dim, output_dim, bias=False)
        self.v_proj = nn.Linear(memory_dim, output_dim, bias=False)
        self.out_proj = nn.Linear(output_dim, output_dim)

        # Layer normalization and dropout
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        query_states: torch.Tensor,  # [B, N, query_dim]
        memory_keys: torch.Tensor,  # [M, memory_dim]
        memory_values: torch.Tensor,  # [M, memory_dim]
        key_mask: torch.Tensor | None = None,  # [M] mask for valid slots
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform cross-attention memory read.

        Args:
            query_states: Query representations [B, N, query_dim]
            memory_keys: Memory key slots [M, memory_dim]
            memory_values: Memory value slots [M, memory_dim]
            key_mask: Optional mask for valid memory slots [M]

        Returns:
            memory_context: Retrieved memory context [B, N, output_dim]
            attention_weights: Attention weights over selected slots [B, N, topk]
            selected_indices: Indices of selected memory slots [B, N, topk]
        """
        B, N, _ = query_states.shape
        M, _ = memory_keys.shape

        # Step 1: Top-k preselection for each query
        # Use mean query for efficiency, or per-position for accuracy
        if self.training or N == 1:
            # Per-position selection (more accurate but slower)
            selected_indices, selected_keys, selected_values = self._per_position_selection(
                query_states, memory_keys, memory_values, key_mask
            )
        else:
            # Mean query selection (faster for inference)
            selected_indices, selected_keys, selected_values = self._mean_query_selection(
                query_states, memory_keys, memory_values, key_mask
            )

        # Step 2: Multi-head cross-attention over selected slots
        memory_context, attention_weights = self._cross_attention(query_states, selected_keys, selected_values)

        return memory_context, attention_weights, selected_indices

    def _per_position_selection(
        self,
        query_states: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select top-k slots independently for each query position."""
        B, N, _ = query_states.shape
        M, _ = memory_keys.shape
        K = min(self.topk, M)

        # Project query to memory space for similarity computation
        query_for_sim = self.q_proj(query_states)  # [B, N, output_dim]
        query_for_sim = query_for_sim.mean(dim=-1, keepdim=True).expand(-1, -1, self.memory_dim)  # Rough projection

        selected_indices = torch.zeros(B, N, K, dtype=torch.long, device=query_states.device)
        selected_keys = torch.zeros(B, N, K, self.memory_dim, device=query_states.device)
        selected_values = torch.zeros(B, N, K, self.memory_dim, device=query_states.device)

        for b in range(B):
            for n in range(N):
                query_pos = query_for_sim[b, n]  # [memory_dim]

                # Compute similarities
                similarities = F.cosine_similarity(query_pos.unsqueeze(0), memory_keys, dim=-1)  # [M]

                # Apply mask if provided
                if key_mask is not None:
                    similarities = similarities.masked_fill(~key_mask, -float("inf"))

                # Select top-k
                _, top_idx = torch.topk(similarities, K, dim=-1)

                selected_indices[b, n] = top_idx
                selected_keys[b, n] = memory_keys[top_idx]
                selected_values[b, n] = memory_values[top_idx]

        return selected_indices, selected_keys, selected_values

    def _mean_query_selection(
        self,
        query_states: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select top-k slots using mean query (faster for inference)."""
        B, N, _ = query_states.shape
        M, _ = memory_keys.shape
        K = min(self.topk, M)

        # Use mean query across sequence
        mean_query = query_states.mean(dim=1)  # [B, query_dim]

        # Project to memory space for similarity
        query_proj = self.q_proj(mean_query)  # [B, output_dim]

        # Use subset of output_dim that matches memory_dim
        query_for_sim = query_proj[:, : self.memory_dim]  # [B, memory_dim]

        # Compute similarities [B, M]
        similarities = torch.einsum("bd,md->bm", F.normalize(query_for_sim, dim=-1), F.normalize(memory_keys, dim=-1))

        # Apply mask if provided
        if key_mask is not None:
            similarities = similarities.masked_fill(~key_mask.unsqueeze(0), -float("inf"))

        # Select top-k for each batch
        _, top_indices = torch.topk(similarities, K, dim=-1)  # [B, K]

        # Expand to [B, N, K] (same selection for all positions)
        selected_indices = top_indices.unsqueeze(1).expand(B, N, K)

        # Gather selected keys and values
        selected_keys = memory_keys[top_indices]  # [B, K, memory_dim]
        selected_values = memory_values[top_indices]  # [B, K, memory_dim]

        # Expand to [B, N, K, memory_dim]
        selected_keys = selected_keys.unsqueeze(1).expand(B, N, K, self.memory_dim)
        selected_values = selected_values.unsqueeze(1).expand(B, N, K, self.memory_dim)

        return selected_indices, selected_keys, selected_values

    def _cross_attention(
        self,
        query_states: torch.Tensor,  # [B, N, query_dim]
        selected_keys: torch.Tensor,  # [B, N, K, memory_dim]
        selected_values: torch.Tensor,  # [B, N, K, memory_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform multi-head cross-attention over selected memory slots."""
        B, N, _ = query_states.shape
        _, _, K, _ = selected_keys.shape

        # Project to multi-head space
        q = self.q_proj(query_states)  # [B, N, output_dim]
        k = self.k_proj(selected_keys)  # [B, N, K, output_dim]
        v = self.v_proj(selected_values)  # [B, N, K, output_dim]

        # Reshape for multi-head attention
        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, N, head_dim]
        k = k.view(B, N, K, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # [B, n_heads, N, K, head_dim]
        v = v.view(B, N, K, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # [B, n_heads, N, K, head_dim]

        # Compute attention scores
        scores = torch.einsum("bhnd,bhnkd->bhnk", q, k)  # [B, n_heads, N, K]
        scores = scores / (self.head_dim**0.5)

        # Apply temperature
        scores = scores / self.temperature

        # Softmax attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [B, n_heads, N, K]
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.einsum("bhnk,bhnkd->bhnd", attn_weights, v)  # [B, n_heads, N, head_dim]

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.output_dim)  # [B, N, output_dim]
        output = self.out_proj(attn_output)
        output = self.norm(output)

        # Average attention weights across heads for interpretability
        avg_attn_weights = attn_weights.mean(dim=1)  # [B, N, K]

        return output, avg_attn_weights


class MemoryContextProjector(nn.Module):
    """
    Projects memory context to match model dimensions.

    Handles dimension mismatches between memory and model spaces.
    """

    def __init__(self, memory_dim: int, model_dim: int, use_layernorm: bool = True, dropout: float = 0.1):
        super().__init__()
        self.memory_dim = memory_dim
        self.model_dim = model_dim

        # Projection layer
        if memory_dim != model_dim:
            self.proj = nn.Linear(memory_dim, model_dim)
        else:
            self.proj = nn.Identity()

        # Optional normalization and dropout
        self.norm = nn.LayerNorm(model_dim) if use_layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, memory_context: torch.Tensor) -> torch.Tensor:
        """
        Project memory context to model dimensions.

        Args:
            memory_context: Memory context [B, N, memory_dim]

        Returns:
            projected_context: Projected context [B, N, model_dim]
        """
        projected = self.proj(memory_context)
        projected = self.norm(projected)
        projected = self.dropout(projected)
        return projected


class AdaptiveTopK(nn.Module):
    """
    Adaptive top-k selection based on query complexity.

    Dynamically adjusts the number of memory slots retrieved
    based on query characteristics.
    """

    def __init__(self, query_dim: int, min_k: int = 8, max_k: int = 64, complexity_threshold: float = 0.5):
        super().__init__()
        self.min_k = min_k
        self.max_k = max_k
        self.complexity_threshold = complexity_threshold

        # Complexity estimator
        self.complexity_net = nn.Sequential(
            nn.Linear(query_dim, query_dim // 4), nn.ReLU(), nn.Linear(query_dim // 4, 1), nn.Sigmoid()
        )

    def forward(self, query_states: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive k values for each query.

        Args:
            query_states: Query representations [B, N, query_dim]

        Returns:
            k_values: Number of slots to retrieve for each position [B, N]
        """
        # Estimate query complexity
        complexity = self.complexity_net(query_states)  # [B, N, 1]
        complexity = complexity.squeeze(-1)  # [B, N]

        # Map complexity to k values
        k_range = self.max_k - self.min_k
        k_values = self.min_k + (complexity * k_range)
        k_values = torch.round(k_values).long()

        # Ensure valid range
        k_values = torch.clamp(k_values, self.min_k, self.max_k)

        return k_values
