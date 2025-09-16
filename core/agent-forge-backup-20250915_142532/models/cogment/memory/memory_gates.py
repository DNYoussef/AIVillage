"""
Memory gating mechanisms for surprise-based writes.

Implements Titan-style surprise gating for selective memory updates
based on prediction error and novelty detection.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .memory_utils import surprisal_from_loss


class SurpriseGate(nn.Module):
    """
    Surprise-based gating mechanism for memory writes.

    Only writes to memory when surprisal exceeds threshold, implementing
    the core Titan principle of selective memory formation.
    """

    def __init__(
        self,
        alpha: float = 0.1,  # Surprise sensitivity
        threshold: float = 1.0,  # Minimum surprisal for writing
        gate_type: str = "sigmoid",  # 'sigmoid', 'relu', or 'threshold'
    ):
        super().__init__()
        self.alpha = alpha
        self.threshold = threshold
        self.gate_type = gate_type

        # Learnable parameters for adaptive gating
        self.gate_bias = nn.Parameter(torch.zeros(1))
        self.gate_scale = nn.Parameter(torch.ones(1))

    def forward(self, surprisal: torch.Tensor) -> torch.Tensor:
        """
        Compute surprise gate values.

        Args:
            surprisal: Surprisal values [B, N] or [B]

        Returns:
            gate_values: Gating weights [same shape as surprisal]
        """
        # Apply learnable scaling and bias
        scaled_surprisal = self.gate_scale * surprisal + self.gate_bias

        if self.gate_type == "sigmoid":
            # Smooth gating: g = σ(α * (surprisal - threshold))
            gate_input = self.alpha * (scaled_surprisal - self.threshold)
            return torch.sigmoid(gate_input)

        elif self.gate_type == "relu":
            # ReLU gating: g = ReLU(α * (surprisal - threshold))
            gate_input = self.alpha * (scaled_surprisal - self.threshold)
            return torch.relu(gate_input)

        elif self.gate_type == "threshold":
            # Hard threshold: g = 1 if surprisal > threshold else 0
            return (scaled_surprisal > self.threshold).float()

        else:
            raise ValueError(f"Unknown gate_type: {self.gate_type}")

    def compute_write_probability(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute write probability from model predictions.

        Args:
            predictions: Model predictions [B, N, vocab_size]
            targets: Target tokens [B, N]

        Returns:
            write_prob: Probability of writing to memory [B]
        """
        # Calculate surprisal
        surprisal = surprisal_from_loss(predictions, targets, reduction="mean")  # [B]

        # Apply gating
        return self.forward(surprisal)


class MemoryWriter(nn.Module):
    """
    Handles memory slot updates with competition and consolidation.

    Implements top-k slot competition where multiple writes compete
    for limited memory slots based on attention scores.
    """

    def __init__(
        self, memory_dim: int, update_rate: float = 0.1, momentum: float = 0.9, competitive_updates: bool = True
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.update_rate = update_rate
        self.momentum = momentum
        self.competitive_updates = competitive_updates

        # Momentum buffers for stable updates
        self.register_buffer("key_momentum", None)
        self.register_buffer("value_momentum", None)

    def initialize_momentum(self, n_slots: int, device: torch.device):
        """Initialize momentum buffers."""
        if self.key_momentum is None:
            self.key_momentum = torch.zeros(n_slots, self.memory_dim, device=device)
            self.value_momentum = torch.zeros(n_slots, self.memory_dim, device=device)

    def forward(
        self,
        memory_keys: torch.Tensor,  # [n_slots, memory_dim]
        memory_values: torch.Tensor,  # [n_slots, memory_dim]
        write_query: torch.Tensor,  # [B, memory_dim]
        write_value: torch.Tensor,  # [B, memory_dim]
        gate_weights: torch.Tensor,  # [B]
        top_indices: torch.Tensor,  # [B, k] indices of slots to potentially update
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update memory slots with gated writes.

        Args:
            memory_keys: Current memory key slots
            memory_values: Current memory value slots
            write_query: Query to write as new key
            write_value: Value to write as new value
            gate_weights: Surprise gate weights for each batch item
            top_indices: Top-k slots for competitive writing

        Returns:
            updated_keys: Updated memory keys
            updated_values: Updated memory values
        """
        device = memory_keys.device
        n_slots = memory_keys.size(0)

        # Initialize momentum if needed
        self.initialize_momentum(n_slots, device)

        # Only update if gate is open (surprisal is high enough)
        active_updates = gate_weights > 0.01  # Small threshold to avoid tiny updates

        if not active_updates.any():
            return memory_keys, memory_values

        # Filter to active updates
        active_queries = write_query[active_updates]  # [A, memory_dim] where A <= B
        active_values = write_value[active_updates]  # [A, memory_dim]
        active_gates = gate_weights[active_updates]  # [A]
        active_indices = top_indices[active_updates]  # [A, k]

        if self.competitive_updates:
            # Competitive updates: multiple writes compete for each slot
            updated_keys, updated_values = self._competitive_update(
                memory_keys, memory_values, active_queries, active_values, active_gates, active_indices
            )
        else:
            # Simple updates: each write updates its top-1 slot
            updated_keys, updated_values = self._simple_update(
                memory_keys, memory_values, active_queries, active_values, active_gates, active_indices
            )

        return updated_keys, updated_values

    def _competitive_update(
        self,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        gates: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Competitive memory updates where multiple writes compete for slots.
        """
        n_slots = memory_keys.size(0)
        A, k = indices.shape

        # Accumulate updates for each slot
        key_updates = torch.zeros_like(memory_keys)
        value_updates = torch.zeros_like(memory_values)
        update_counts = torch.zeros(n_slots, device=memory_keys.device)

        for a in range(A):
            for slot_idx in indices[a]:
                # Weight update by gate strength
                weight = gates[a] * self.update_rate

                # Accumulate weighted updates
                key_updates[slot_idx] += weight * queries[a]
                value_updates[slot_idx] += weight * values[a]
                update_counts[slot_idx] += weight

        # Normalize by update counts to prevent amplitude explosion
        valid_slots = update_counts > 0
        key_updates[valid_slots] /= update_counts[valid_slots].unsqueeze(-1)
        value_updates[valid_slots] /= update_counts[valid_slots].unsqueeze(-1)

        # Apply momentum
        if self.momentum > 0:
            self.key_momentum.mul_(self.momentum).add_(key_updates, alpha=(1 - self.momentum))
            self.value_momentum.mul_(self.momentum).add_(value_updates, alpha=(1 - self.momentum))

            final_key_updates = self.key_momentum.clone()
            final_value_updates = self.value_momentum.clone()
        else:
            final_key_updates = key_updates
            final_value_updates = value_updates

        # Apply updates
        updated_keys = memory_keys + final_key_updates
        updated_values = memory_values + final_value_updates

        return updated_keys, updated_values

    def _simple_update(
        self,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        gates: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Simple memory updates: each write updates its most similar slot.
        """
        updated_keys = memory_keys.clone()
        updated_values = memory_values.clone()

        A = queries.size(0)

        for a in range(A):
            # Update the top-1 slot for this query
            slot_idx = indices[a, 0]  # Top-1 slot
            weight = gates[a] * self.update_rate

            # Exponential moving average update
            updated_keys[slot_idx] = (1 - weight) * updated_keys[slot_idx] + weight * queries[a]
            updated_values[slot_idx] = (1 - weight) * updated_values[slot_idx] + weight * values[a]

        return updated_keys, updated_values


class NoveltyDetector(nn.Module):
    """
    Detects novel patterns for enhanced surprise calculation.

    Combines prediction error with novelty detection for more
    sophisticated memory gating decisions.
    """

    def __init__(self, hidden_dim: int, novelty_weight: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.novelty_weight = novelty_weight

        # Novelty detection network
        self.novelty_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid()
        )

    def forward(
        self, hidden_states: torch.Tensor, memory_context: torch.Tensor  # [B, N, hidden_dim]  # [B, N, hidden_dim]
    ) -> torch.Tensor:
        """
        Compute novelty scores based on hidden states and memory context.

        Args:
            hidden_states: Current hidden representations
            memory_context: Retrieved memory context

        Returns:
            novelty_scores: Novelty scores [B, N]
        """
        # Compute difference between current state and memory
        state_diff = hidden_states - memory_context  # [B, N, hidden_dim]

        # Pass through novelty detection network
        novelty_logits = self.novelty_net(state_diff)  # [B, N, 1]
        novelty_scores = novelty_logits.squeeze(-1)  # [B, N]

        return novelty_scores

    def enhanced_surprisal(
        self, prediction_surprisal: torch.Tensor, novelty_scores: torch.Tensor  # [B, N]  # [B, N]
    ) -> torch.Tensor:
        """
        Combine prediction surprisal with novelty for enhanced surprise.

        Args:
            prediction_surprisal: Standard prediction error surprisal
            novelty_scores: Novelty detection scores

        Returns:
            enhanced_surprisal: Combined surprisal measure
        """
        return prediction_surprisal + self.novelty_weight * novelty_scores
