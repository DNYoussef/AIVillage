"""
Memory utilities for GatedLTM system.

Provides decay mechanisms, surprisal calculation, and memory management utilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryDecay(nn.Module):
    """
    Implements exponential decay for memory slots to prevent saturation.

    Titan-style slow forgetting: m_t = (1 - decay_rate) * m_{t-1}
    """

    def __init__(self, decay_rate: float = 1e-3):
        super().__init__()
        self.decay_rate = decay_rate

    def forward(self, memory_slots: torch.Tensor) -> torch.Tensor:
        """
        Apply decay to memory slots.

        Args:
            memory_slots: Memory tensor [n_slots, slot_dim] or [B, n_slots, slot_dim]

        Returns:
            decayed_memory: Memory after applying decay
        """
        return memory_slots * (1.0 - self.decay_rate)


def surprisal_from_loss(predictions: torch.Tensor, targets: torch.Tensor, reduction: str = "none") -> torch.Tensor:
    """
    Calculate surprisal (prediction error) from model predictions.

    Surprisal = -log P(target | context) = cross_entropy loss

    Args:
        predictions: Model predictions [B, N, vocab_size]
        targets: Target tokens [B, N]
        reduction: How to reduce across sequence ('none', 'mean', 'sum')

    Returns:
        surprisal: Surprisal values [B, N] or [B] depending on reduction
    """
    # Reshape for cross entropy
    flat_preds = predictions.view(-1, predictions.size(-1))  # [B*N, vocab_size]
    flat_targets = targets.view(-1)  # [B*N]

    # Calculate per-token surprisal
    surprisal = F.cross_entropy(flat_preds, flat_targets, reduction="none")  # [B*N]
    surprisal = surprisal.view(targets.shape)  # [B, N]

    if reduction == "mean":
        return surprisal.mean(dim=1)  # [B]
    elif reduction == "sum":
        return surprisal.sum(dim=1)  # [B]
    else:
        return surprisal  # [B, N]


def normalize_memory_slots(slots: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    L2 normalize memory slots for stable cosine similarity.

    Args:
        slots: Memory slots tensor
        dim: Dimension to normalize along

    Returns:
        normalized_slots: L2 normalized slots
    """
    return F.normalize(slots, p=2, dim=dim)


def topk_memory_selection(
    query: torch.Tensor, memory_keys: torch.Tensor, k: int, temperature: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-k memory slots based on similarity to query.

    Args:
        query: Query tensor [B, d_model]
        memory_keys: Memory key slots [n_slots, d_model]
        k: Number of top slots to select
        temperature: Temperature for similarity computation

    Returns:
        top_indices: Indices of top-k slots [B, k]
        top_scores: Similarity scores [B, k]
    """
    # Normalize for cosine similarity
    query_norm = normalize_memory_slots(query)  # [B, d_model]
    keys_norm = normalize_memory_slots(memory_keys)  # [n_slots, d_model]

    # Compute similarities
    similarities = torch.einsum("bd,nd->bn", query_norm, keys_norm)  # [B, n_slots]
    similarities = similarities / temperature

    # Get top-k
    top_scores, top_indices = torch.topk(similarities, k=min(k, memory_keys.size(0)), dim=-1)

    return top_indices, top_scores


def compute_slot_utilization(attention_weights: torch.Tensor, window_size: int = 1000) -> torch.Tensor:
    """
    Compute utilization statistics for memory slots.

    Args:
        attention_weights: Attention weights over slots [B, n_slots]
        window_size: Window for computing moving averages

    Returns:
        utilization: Per-slot utilization scores [n_slots]
    """
    # Average attention across batch
    avg_attention = attention_weights.mean(dim=0)  # [n_slots]

    # Simple utilization metric (can be enhanced with moving averages)
    return avg_attention


def memory_consolidation_loss(
    old_memory: torch.Tensor, new_memory: torch.Tensor, consolidation_weight: float = 0.01
) -> torch.Tensor:
    """
    Compute consolidation loss to preserve important memories.

    Args:
        old_memory: Previous memory state
        new_memory: Updated memory state
        consolidation_weight: Weight for consolidation loss

    Returns:
        loss: Consolidation loss scalar
    """
    # L2 distance between old and new memory
    diff = new_memory - old_memory
    consolidation_loss = torch.mean(diff**2)

    return consolidation_weight * consolidation_loss
