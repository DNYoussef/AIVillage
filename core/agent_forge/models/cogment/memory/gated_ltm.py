"""
GatedLTMMemory: Main Titan-style Long-Term Memory implementation.

Combines cross-attention reads, surprise-gated writes, and decay mechanisms
for efficient episodic memory with controlled slot competition.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_attention import CrossAttentionReader, MemoryContextProjector
from .memory_gates import MemoryWriter, NoveltyDetector, SurpriseGate
from .memory_utils import MemoryDecay, surprisal_from_loss, topk_memory_selection


class GatedLTMMemory(nn.Module):
    """
    Titan-style Gated Long-Term Memory.

    Features:
    - Cross-attention memory reads with top-k preselection
    - Surprise-gated writes based on prediction error
    - Exponential decay to prevent slot saturation
    - Competitive slot updates with momentum
    - Optional novelty detection for enhanced gating

    Parameter Budget Analysis:
    - Memory slots: 1024 × 512 = 524,288 params (keys + values = 2 × 524k = 1.05M)
    - Cross-attention: ~100k params (projections)
    - Gates and utilities: ~50k params
    - Total: ~1.2M params (within 1.4M budget)
    """

    def __init__(
        self,
        query_dim: int = 320,  # From config.d_model
        memory_dim: int = 512,  # From config.ltm_dim
        n_slots: int = 1024,  # From config.ltm_capacity
        n_heads: int = 8,  # Multi-head attention
        topk: int = 32,  # Top-k preselection
        # Surprise gating parameters
        surprise_alpha: float = 0.1,  # Surprise sensitivity
        surprise_threshold: float = 1.0,  # Minimum surprisal for writing
        gate_type: str = "sigmoid",  # Gate function type
        # Update parameters
        update_rate: float = 0.1,  # Memory update learning rate
        momentum: float = 0.9,  # Momentum for stable updates
        decay_rate: float = 1e-3,  # Exponential decay rate
        # Architecture parameters
        use_novelty_detection: bool = False,  # Enable novelty detector
        competitive_updates: bool = True,  # Competitive slot updates
        normalize_slots: bool = True,  # L2 normalize memory slots
        temperature: float = 1.0,  # Attention temperature
        dropout: float = 0.1,
    ):
        super().__init__()

        # Store configuration
        self.query_dim = query_dim
        self.memory_dim = memory_dim
        self.n_slots = n_slots
        self.n_heads = n_heads
        self.topk = topk
        self.normalize_slots = normalize_slots
        self.use_novelty_detection = use_novelty_detection

        # Memory slots (the core ~1M parameters)
        self.memory_keys = nn.Parameter(
            torch.randn(n_slots, memory_dim) * (2.0 / memory_dim) ** 0.5,
            requires_grad=False,  # Updated via online learning, not backprop
        )
        self.memory_values = nn.Parameter(
            torch.randn(n_slots, memory_dim) * (2.0 / memory_dim) ** 0.5, requires_grad=False
        )

        # Slot utilization tracking
        self.register_buffer("slot_usage", torch.zeros(n_slots))
        self.register_buffer("last_access", torch.zeros(n_slots))
        self.register_buffer("update_count", torch.zeros(1))

        # Core components
        self.cross_attention = CrossAttentionReader(
            query_dim=memory_dim,  # After projection to memory space
            memory_dim=memory_dim,
            output_dim=memory_dim,  # Keep in memory space
            n_heads=n_heads,
            topk=topk,
            temperature=temperature,
            dropout=dropout,
        )

        self.surprise_gate = SurpriseGate(alpha=surprise_alpha, threshold=surprise_threshold, gate_type=gate_type)

        self.memory_writer = MemoryWriter(
            memory_dim=memory_dim, update_rate=update_rate, momentum=momentum, competitive_updates=competitive_updates
        )

        self.memory_decay = MemoryDecay(decay_rate=decay_rate)

        # Optional novelty detection
        if use_novelty_detection:
            self.novelty_detector = NoveltyDetector(hidden_dim=memory_dim, novelty_weight=0.1)
        else:
            self.novelty_detector = None

        # Projection layers for interfacing with different model dimensions
        self.query_projector = nn.Linear(query_dim, memory_dim, bias=False)
        self.output_projector = MemoryContextProjector(
            memory_dim=memory_dim,
            model_dim=query_dim,  # Project back to model dimensions
            use_layernorm=True,
            dropout=dropout,
        )

        # Initialize components
        self._init_memory_slots()
        self._init_projectors()

    def _init_memory_slots(self):
        """Initialize memory slots with diverse, normalized vectors."""
        # Initialize with unit norm random vectors
        with torch.no_grad():
            # Keys: diverse unit vectors
            nn.init.normal_(self.memory_keys, mean=0, std=1.0)
            if self.normalize_slots:
                self.memory_keys.data = F.normalize(self.memory_keys.data, p=2, dim=-1)

            # Values: smaller initialization
            nn.init.normal_(self.memory_values, mean=0, std=0.1)

    def _init_projectors(self):
        """Initialize projection layers."""
        nn.init.xavier_uniform_(self.query_projector.weight)

    def read(
        self,
        query_states: torch.Tensor,  # [B, N, query_dim]
        return_attention: bool = False,  # Return attention weights
        key_mask: torch.Tensor | None = None,  # [n_slots] mask for valid slots
    ) -> torch.Tensor:
        """
        Read from memory using cross-attention.

        Args:
            query_states: Query representations [B, N, query_dim]
            return_attention: Whether to return attention weights
            key_mask: Optional mask for valid memory slots

        Returns:
            memory_context: Retrieved memory context [B, N, query_dim]
            attention_info: Optional attention weights and indices (if return_attention=True)
        """
        # Project query to memory space
        memory_queries = self.query_projector(query_states)  # [B, N, memory_dim]

        # Get current memory state
        current_keys = self.memory_keys
        current_values = self.memory_values

        # Normalize if enabled
        if self.normalize_slots:
            current_keys = F.normalize(current_keys, p=2, dim=-1)
            current_values = F.normalize(current_values, p=2, dim=-1)

        # Cross-attention read
        memory_context, attention_weights, selected_indices = self.cross_attention(
            query_states=memory_queries, memory_keys=current_keys, memory_values=current_values, key_mask=key_mask
        )

        # Update slot usage statistics
        self._update_usage_stats(selected_indices, attention_weights)

        # Project back to model dimensions
        output_context = self.output_projector(memory_context)  # [B, N, query_dim]

        if return_attention:
            attention_info = {
                "attention_weights": attention_weights,
                "selected_indices": selected_indices,
                "raw_memory_context": memory_context,
            }
            return output_context, attention_info
        else:
            return output_context

    def write(
        self,
        query_states: torch.Tensor,  # [B, N, query_dim]
        predictions: torch.Tensor | None = None,  # [B, N, vocab_size] for surprisal
        targets: torch.Tensor | None = None,  # [B, N] target tokens
        surprisal: torch.Tensor | None = None,  # [B] pre-computed surprisal
        force_write: bool = False,  # Force write regardless of gate
        return_gate_info: bool = False,  # Return gating information
    ) -> dict[str, torch.Tensor] | None:
        """
        Write to memory with surprise-based gating.

        Args:
            query_states: Query representations to potentially store
            predictions: Model predictions for surprisal calculation
            targets: Target tokens for surprisal calculation
            surprisal: Pre-computed surprisal values
            force_write: Force write regardless of surprise gate
            return_gate_info: Return gating and update information

        Returns:
            gate_info: Optional information about gating decisions
        """
        B, N, _ = query_states.shape

        # Calculate surprisal if not provided
        if surprisal is None and predictions is not None and targets is not None:
            surprisal = surprisal_from_loss(predictions, targets, reduction="mean")  # [B]
        elif surprisal is None:
            # Default surprisal if no targets (e.g., during inference)
            surprisal = torch.ones(B, device=query_states.device)

        # Compute surprise gate values
        if not force_write:
            gate_weights = self.surprise_gate(surprisal)  # [B]
        else:
            gate_weights = torch.ones_like(surprisal)

        # Early exit if no writes are gated
        if not force_write and gate_weights.max() < 0.01:
            if return_gate_info:
                return {"gate_weights": gate_weights, "num_writes": torch.tensor(0), "surprisal": surprisal}
            return None

        # Project queries to memory space
        memory_queries = self.query_projector(query_states)  # [B, N, memory_dim]

        # Use sequence mean as the representation to store
        write_keys = memory_queries.mean(dim=1)  # [B, memory_dim]
        write_values = memory_queries.mean(dim=1)  # [B, memory_dim]

        # Enhanced surprisal with novelty detection if enabled
        if self.novelty_detector is not None and predictions is not None:
            # Read current memory context for novelty comparison
            current_context, _ = self.read(query_states, return_attention=True)
            novelty_scores = self.novelty_detector(
                query_states.mean(dim=1), current_context.mean(dim=1)  # [B, query_dim]  # [B, query_dim]
            )
            enhanced_surprisal = self.novelty_detector.enhanced_surprisal(
                surprisal, novelty_scores.mean(dim=1) if novelty_scores.dim() > 1 else novelty_scores
            )
            gate_weights = self.surprise_gate(enhanced_surprisal)

        # Find top-k slots for competitive writing
        top_indices, _ = topk_memory_selection(
            query=write_keys,
            memory_keys=self.memory_keys,
            k=min(8, self.n_slots),  # Consider top-8 slots for writing
            temperature=1.0,
        )

        # Perform memory updates
        updated_keys, updated_values = self.memory_writer(
            memory_keys=self.memory_keys,
            memory_values=self.memory_values,
            write_query=write_keys,
            write_value=write_values,
            gate_weights=gate_weights,
            top_indices=top_indices,
        )

        # Update memory slots (in-place, no gradients)
        with torch.no_grad():
            self.memory_keys.data = updated_keys
            self.memory_values.data = updated_values

        # Increment update counter
        self.update_count += 1

        if return_gate_info:
            return {
                "gate_weights": gate_weights,
                "num_writes": (gate_weights > 0.01).sum(),
                "surprisal": surprisal,
                "top_indices": top_indices,
                "write_locations": top_indices[gate_weights > 0.01] if (gate_weights > 0.01).any() else torch.empty(0),
            }

        return None

    def decay_step(self):
        """Apply exponential decay to all memory slots."""
        with torch.no_grad():
            self.memory_keys.data = self.memory_decay(self.memory_keys.data)
            self.memory_values.data = self.memory_decay(self.memory_values.data)

            # Decay usage statistics
            self.slot_usage.mul_(0.99)
            self.last_access.add_(1)  # Age all slots

    def consolidate_memory(self, threshold: float = 0.01):
        """
        Consolidate memory by removing unused slots and compacting.

        Args:
            threshold: Usage threshold below which slots are candidates for removal
        """
        with torch.no_grad():
            # Find underutilized slots
            low_usage = self.slot_usage < threshold
            old_access = self.last_access > 1000  # Haven't been accessed in 1000 steps

            # Reset candidates (low usage AND old)
            reset_candidates = low_usage & old_access

            if reset_candidates.any():
                # Reinitialize these slots
                n_reset = reset_candidates.sum().item()
                self.memory_keys.data[reset_candidates] = (
                    torch.randn(n_reset, self.memory_dim, device=self.memory_keys.device)
                    * (2.0 / self.memory_dim) ** 0.5
                )

                self.memory_values.data[reset_candidates] = (
                    torch.randn(n_reset, self.memory_dim, device=self.memory_values.device) * 0.1
                )

                # Reset statistics
                self.slot_usage[reset_candidates] = 0
                self.last_access[reset_candidates] = 0

                return n_reset

        return 0

    def _update_usage_stats(self, selected_indices: torch.Tensor, attention_weights: torch.Tensor):
        """Update slot usage statistics."""
        with torch.no_grad():
            B, N, K = selected_indices.shape

            # Flatten and accumulate usage
            flat_indices = selected_indices.view(-1)  # [B*N*K]
            flat_weights = attention_weights.view(-1)  # [B*N*K]

            # Update usage counts
            self.slot_usage.index_add_(0, flat_indices, flat_weights)

            # Update last access time
            unique_indices = torch.unique(flat_indices)
            self.last_access[unique_indices] = 0  # Reset access time for used slots

    def get_memory_stats(self) -> dict[str, torch.Tensor]:
        """Get memory utilization and health statistics."""
        with torch.no_grad():
            stats = {
                "total_slots": torch.tensor(self.n_slots),
                "avg_usage": self.slot_usage.mean(),
                "max_usage": self.slot_usage.max(),
                "min_usage": self.slot_usage.min(),
                "unused_slots": (self.slot_usage < 0.001).sum(),
                "overused_slots": (self.slot_usage > 1.0).sum(),
                "avg_age": self.last_access.float().mean(),
                "max_age": self.last_access.max(),
                "update_count": self.update_count,
                "memory_norm": {
                    "keys_mean": self.memory_keys.norm(dim=-1).mean(),
                    "keys_std": self.memory_keys.norm(dim=-1).std(),
                    "values_mean": self.memory_values.norm(dim=-1).mean(),
                    "values_std": self.memory_values.norm(dim=-1).std(),
                },
            }

        return stats

    def reset_memory(self, keep_fraction: float = 0.0):
        """
        Reset memory slots, optionally keeping top-utilized slots.

        Args:
            keep_fraction: Fraction of top-utilized slots to preserve
        """
        with torch.no_grad():
            if keep_fraction > 0:
                # Keep top-utilized slots
                n_keep = int(self.n_slots * keep_fraction)
                _, top_indices = torch.topk(self.slot_usage, n_keep)

                # Reset non-top slots
                reset_mask = torch.ones(self.n_slots, dtype=torch.bool, device=self.memory_keys.device)
                reset_mask[top_indices] = False

                n_reset = reset_mask.sum().item()
                if n_reset > 0:
                    self.memory_keys.data[reset_mask] = (
                        torch.randn(n_reset, self.memory_dim, device=self.memory_keys.device)
                        * (2.0 / self.memory_dim) ** 0.5
                    )

                    self.memory_values.data[reset_mask] = (
                        torch.randn(n_reset, self.memory_dim, device=self.memory_values.device) * 0.1
                    )

                    self.slot_usage[reset_mask] = 0
                    self.last_access[reset_mask] = 0
            else:
                # Full reset
                self._init_memory_slots()
                self.slot_usage.zero_()
                self.last_access.zero_()
                self.update_count.zero_()

    def forward(
        self,
        query_states: torch.Tensor,  # [B, N, query_dim]
        predictions: torch.Tensor | None = None,  # [B, N, vocab_size]
        targets: torch.Tensor | None = None,  # [B, N]
        mode: str = "read",  # 'read', 'write', or 'read_write'
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through memory system.

        Args:
            query_states: Query representations
            predictions: Model predictions (for write mode)
            targets: Target tokens (for write mode)
            mode: Operation mode ('read', 'write', 'read_write')

        Returns:
            memory_context: Retrieved memory context [B, N, query_dim]
        """
        if mode == "read":
            return self.read(query_states, **kwargs)

        elif mode == "write":
            self.write(query_states, predictions, targets, **kwargs)
            # Return zero context for write-only mode
            return torch.zeros_like(query_states)

        elif mode == "read_write":
            # Read first, then write
            memory_context = self.read(query_states, **kwargs)
            self.write(query_states, predictions, targets, **kwargs)
            return memory_context

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'read', 'write', or 'read_write'")

    def count_parameters(self) -> int:
        """Count trainable parameters (excluding memory slots)."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        memory_slots = self.memory_keys.numel() + self.memory_values.numel()

        return {"trainable": trainable_params, "memory_slots": memory_slots, "total": trainable_params + memory_slots}
