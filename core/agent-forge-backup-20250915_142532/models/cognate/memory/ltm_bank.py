#!/usr/bin/env python3
"""
Cognate Long-Term Memory (LTM) Bank

This module implements a Titans-style Long-Term Memory system with:
- Surprise × Novelty gating for selective writing  
- Cosine similarity-based retrieval
- Memory persistence and serialization
- Usage tracking and decay
- Efficient memory management
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """Individual memory item in the LTM bank."""

    key: torch.Tensor  # Memory key for retrieval (d_mem,)
    value: torch.Tensor  # Memory content (d_mem,)
    timestamp: float  # When this memory was created
    usage_count: int  # How many times this memory has been retrieved
    last_access: float  # Last time this memory was accessed
    surprise: float  # Surprise value when written
    novelty: float  # Novelty value when written
    metadata: dict[str, Any]  # Additional metadata


@dataclass
class MemoryConfig:
    """Configuration for the LTM memory system."""

    capacity: int = 4096  # Maximum number of memories
    d_mem: int = 216  # Memory dimension
    topk: int = 4  # Top-k retrieval

    # Surprise × Novelty gating (Titans-style)
    surprise_alpha: float = 4.0  # Surprise scaling factor
    novelty_beta: float = 0.9  # Momentum for novelty detection
    memory_eta: float = 0.01  # Memory update learning rate
    memory_decay: float = 0.0001  # Memory decay rate

    # Retrieval parameters
    similarity_threshold: float = 0.1  # Minimum similarity for retrieval
    usage_decay: float = 0.99  # Usage count decay per step

    # Memory management
    replacement_policy: str = "lru"  # "lru", "random", "least_used"
    enable_compression: bool = False  # Enable memory compression
    compression_threshold: float = 0.95  # Similarity threshold for compression


class CognateLTMBank(nn.Module):
    """
    Long-Term Memory Bank with Titans-style gating.

    This implements a sophisticated memory system that:
    1. Uses surprise × novelty gating to decide what to store
    2. Maintains usage statistics for intelligent replacement
    3. Supports efficient similarity-based retrieval
    4. Provides persistence and serialization capabilities
    """

    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config

        # Memory storage (registered as buffers for device movement)
        self.register_buffer("memory_keys", torch.zeros(config.capacity, config.d_mem))
        self.register_buffer("memory_values", torch.zeros(config.capacity, config.d_mem))
        self.register_buffer("memory_usage", torch.zeros(config.capacity))
        self.register_buffer("memory_timestamps", torch.zeros(config.capacity))

        # Memory metadata (not tensors, stored separately)
        self.memory_metadata: list[dict[str, Any] | None] = [None] * config.capacity

        # State tracking
        self.current_size = 0
        self.global_step = 0
        self.write_count = 0
        self.read_count = 0

        # Novelty detection state
        self.novelty_ema = None  # Running average for novelty detection

        logger.info(f"Initialized LTM bank: capacity={config.capacity}, d_mem={config.d_mem}")

    def forward(self, *args, **kwargs):
        """Placeholder forward method (not used in typical training)."""
        # LTM bank operates via read/write methods, not forward pass
        # Return empty tensor for compatibility with PyTorch modules
        import torch

        return torch.empty(0, device=next(self.parameters()).device if len(list(self.parameters())) > 0 else "cpu")

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        surprise: float,
        novelty: float,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[bool, float, int]:
        """
        Write a memory to the bank using surprise × novelty gating.

        Args:
            key: Memory key for retrieval (d_mem,)
            value: Memory content (d_mem,)
            surprise: Surprise value (0-1, higher = more surprising)
            novelty: Novelty value (0-1, higher = more novel)
            metadata: Optional metadata dictionary

        Returns:
            Tuple of (written: bool, gate_prob: float, slot: int)
        """
        # Titans-style gating: P(write) = surprise × novelty
        gate_prob = surprise * novelty

        # Apply gating decision
        if torch.rand(1, device=key.device).item() > gate_prob:
            return False, gate_prob, -1

        # Find slot to write to
        slot = self._find_write_slot()

        # Write to memory
        self.memory_keys[slot] = key.clone().detach()
        self.memory_values[slot] = value.clone().detach()
        self.memory_usage[slot] = 1.0  # Reset usage
        self.memory_timestamps[slot] = float(self.global_step)

        # Store metadata
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "surprise": surprise,
                "novelty": novelty,
                "write_step": self.global_step,
                "write_time": time.time(),
            }
        )

        self.memory_metadata[slot] = metadata

        # Update counters
        if slot >= self.current_size:
            self.current_size = slot + 1

        self.write_count += 1

        # Update novelty EMA for future novelty computation
        self._update_novelty_ema(value)

        logger.debug(f"Wrote memory to slot {slot} (gate_prob={gate_prob:.3f})")
        return True, gate_prob, slot

    def read(
        self, query: torch.Tensor, topk: int | None = None, return_similarities: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor | None]:
        """
        Read top-k memories based on cosine similarity.

        Args:
            query: Query vector (d_mem,)
            topk: Number of memories to retrieve (default: config.topk)
            return_similarities: Whether to return similarity scores

        Returns:
            Tuple of (values, keys, indices, similarities)
            - values: Retrieved memory values (topk, d_mem)
            - keys: Retrieved memory keys (topk, d_mem)
            - indices: Retrieved memory indices
            - similarities: Similarity scores (if requested)
        """
        if topk is None:
            topk = self.config.topk

        if self.current_size == 0:
            # No memories stored yet
            device = query.device
            empty_values = torch.zeros(topk, self.config.d_mem, device=device)
            empty_keys = torch.zeros(topk, self.config.d_mem, device=device)
            empty_sims = torch.zeros(topk, device=device) if return_similarities else None
            return empty_values, empty_keys, [], empty_sims

        # Compute similarities with active memories
        active_keys = self.memory_keys[: self.current_size]  # (current_size, d_mem)
        similarities = F.cosine_similarity(
            query.unsqueeze(0), active_keys, dim=1  # (1, d_mem)  # (current_size, d_mem)
        )  # (current_size,)

        # Filter by similarity threshold
        valid_mask = similarities >= self.config.similarity_threshold
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

        if len(valid_indices) == 0:
            # No memories above threshold
            device = query.device
            empty_values = torch.zeros(topk, self.config.d_mem, device=device)
            empty_keys = torch.zeros(topk, self.config.d_mem, device=device)
            empty_sims = torch.zeros(topk, device=device) if return_similarities else None
            return empty_values, empty_keys, [], empty_sims

        # Get top-k from valid memories
        valid_similarities = similarities[valid_indices]
        actual_topk = min(topk, len(valid_indices))

        _, top_indices_in_valid = torch.topk(valid_similarities, actual_topk)
        top_indices = valid_indices[top_indices_in_valid]

        # Retrieve memories
        retrieved_values = self.memory_values[top_indices]  # (actual_topk, d_mem)
        retrieved_keys = self.memory_keys[top_indices]  # (actual_topk, d_mem)
        retrieved_sims = similarities[top_indices] if return_similarities else None

        # Update usage statistics
        self.memory_usage[top_indices] = torch.clamp(self.memory_usage[top_indices] * 1.1, max=2.0)

        # Pad to requested topk if needed
        if actual_topk < topk:
            pad_size = topk - actual_topk
            device = query.device

            pad_values = torch.zeros(pad_size, self.config.d_mem, device=device)
            pad_keys = torch.zeros(pad_size, self.config.d_mem, device=device)

            retrieved_values = torch.cat([retrieved_values, pad_values], dim=0)
            retrieved_keys = torch.cat([retrieved_keys, pad_keys], dim=0)

            if return_similarities:
                pad_sims = torch.zeros(pad_size, device=device)
                retrieved_sims = torch.cat([retrieved_sims, pad_sims], dim=0)

        self.read_count += 1

        logger.debug(f"Read {actual_topk} memories (requested {topk})")

        return (retrieved_values, retrieved_keys, top_indices.tolist()[:actual_topk], retrieved_sims)

    def _find_write_slot(self) -> int:
        """Find the best slot to write a new memory."""
        if self.current_size < self.config.capacity:
            # Simple case: use next available slot
            return self.current_size

        # Memory is full, need replacement
        if self.config.replacement_policy == "lru":
            # Least Recently Used: find oldest timestamp
            slot = torch.argmin(self.memory_timestamps).item()

        elif self.config.replacement_policy == "least_used":
            # Least frequently used: find lowest usage count
            slot = torch.argmin(self.memory_usage).item()

        elif self.config.replacement_policy == "random":
            # Random replacement
            slot = torch.randint(0, self.config.capacity, (1,)).item()

        else:
            raise ValueError(f"Unknown replacement policy: {self.config.replacement_policy}")

        logger.debug(f"Replacing memory in slot {slot} (policy={self.config.replacement_policy})")
        return slot

    def _update_novelty_ema(self, value: torch.Tensor):
        """Update the novelty EMA for future novelty detection."""
        if self.novelty_ema is None:
            self.novelty_ema = value.clone().detach()
        else:
            beta = self.config.novelty_beta
            self.novelty_ema = beta * self.novelty_ema + (1 - beta) * value.detach()

    def compute_novelty(self, value: torch.Tensor) -> float:
        """
        Compute novelty of a value relative to stored memories.

        Args:
            value: Value to compute novelty for (d_mem,)

        Returns:
            Novelty score (0-1, higher = more novel)
        """
        if self.novelty_ema is None or self.current_size == 0:
            return 1.0  # Everything is novel when memory is empty

        # Compute similarity to running average
        similarity = F.cosine_similarity(value.unsqueeze(0), self.novelty_ema.unsqueeze(0), dim=1).item()

        # Novelty is inverse of similarity
        novelty = 1.0 - max(0.0, similarity)
        return novelty

    def decay_usage(self):
        """Decay usage statistics (called periodically during training)."""
        self.memory_usage *= self.config.usage_decay
        self.global_step += 1

    def compress_memories(self) -> int:
        """
        Compress similar memories to save space.

        Returns:
            Number of memories compressed
        """
        if not self.config.enable_compression or self.current_size < 2:
            return 0

        compressed_count = 0
        threshold = self.config.compression_threshold

        # Find pairs of very similar memories
        similarities = torch.mm(
            F.normalize(self.memory_values[: self.current_size], dim=1),
            F.normalize(self.memory_values[: self.current_size], dim=1).t(),
        )

        # Mask diagonal (self-similarities)
        similarities.fill_diagonal_(0.0)

        # Find highly similar pairs
        high_sim_pairs = torch.nonzero(similarities > threshold, as_tuple=False)

        compressed_indices = set()

        for i, j in high_sim_pairs:
            i, j = i.item(), j.item()

            if i in compressed_indices or j in compressed_indices:
                continue

            # Keep the more frequently used memory
            if self.memory_usage[i] > self.memory_usage[j]:
                keep_idx, remove_idx = i, j
            else:
                keep_idx, remove_idx = j, i

            # Average the memories
            self.memory_values[keep_idx] = (self.memory_values[keep_idx] + self.memory_values[remove_idx]) / 2.0
            self.memory_keys[keep_idx] = (self.memory_keys[keep_idx] + self.memory_keys[remove_idx]) / 2.0

            # Update usage
            self.memory_usage[keep_idx] += self.memory_usage[remove_idx]

            # Mark removed index
            compressed_indices.add(remove_idx)
            compressed_count += 1

        # Compact memory if any compressions occurred
        if compressed_count > 0:
            self._compact_memory(compressed_indices)

        logger.info(f"Compressed {compressed_count} similar memories")
        return compressed_count

    def _compact_memory(self, removed_indices: set):
        """Compact memory by removing gaps from removed indices."""
        # Create mapping from old to new indices
        new_memories = []
        new_metadata = []

        for i in range(self.current_size):
            if i not in removed_indices:
                new_memories.append(i)
                new_metadata.append(self.memory_metadata[i])

        # Update current size
        new_size = len(new_memories)

        if new_size < self.current_size:
            # Copy memories to compact layout
            for new_idx, old_idx in enumerate(new_memories):
                if new_idx != old_idx:  # Only copy if positions differ
                    self.memory_keys[new_idx] = self.memory_keys[old_idx]
                    self.memory_values[new_idx] = self.memory_values[old_idx]
                    self.memory_usage[new_idx] = self.memory_usage[old_idx]
                    self.memory_timestamps[new_idx] = self.memory_timestamps[old_idx]

            # Update metadata
            self.memory_metadata = new_metadata + [None] * (self.config.capacity - new_size)
            self.current_size = new_size

    def get_stats(self) -> dict[str, Any]:
        """Get memory bank statistics."""
        stats = {
            "current_size": self.current_size,
            "capacity": self.config.capacity,
            "utilization": self.current_size / self.config.capacity,
            "write_count": self.write_count,
            "read_count": self.read_count,
            "global_step": self.global_step,
        }

        if self.current_size > 0:
            active_usage = self.memory_usage[: self.current_size]
            stats.update(
                {
                    "avg_usage": float(active_usage.mean()),
                    "max_usage": float(active_usage.max()),
                    "min_usage": float(active_usage.min()),
                    "usage_std": float(active_usage.std()),
                }
            )

        return stats

    def save(self, path: str | Path):
        """Save memory bank state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save tensor data
        tensor_data = {
            "memory_keys": self.memory_keys.cpu(),
            "memory_values": self.memory_values.cpu(),
            "memory_usage": self.memory_usage.cpu(),
            "memory_timestamps": self.memory_timestamps.cpu(),
        }

        torch.save(tensor_data, path.with_suffix(".pt"))

        # Save metadata and state
        metadata = {
            "config": {
                "capacity": self.config.capacity,
                "d_mem": self.config.d_mem,
                "topk": self.config.topk,
                "surprise_alpha": self.config.surprise_alpha,
                "novelty_beta": self.config.novelty_beta,
                "memory_eta": self.config.memory_eta,
                "memory_decay": self.config.memory_decay,
                "similarity_threshold": self.config.similarity_threshold,
                "usage_decay": self.config.usage_decay,
                "replacement_policy": self.config.replacement_policy,
            },
            "state": {
                "current_size": self.current_size,
                "global_step": self.global_step,
                "write_count": self.write_count,
                "read_count": self.read_count,
            },
            "memory_metadata": self.memory_metadata[: self.current_size],
            "novelty_ema": self.novelty_ema.tolist() if self.novelty_ema is not None else None,
        }

        with open(path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved memory bank to {path} ({self.current_size} memories)")

    def load(self, path: str | Path):
        """Load memory bank state from disk."""
        path = Path(path)

        # Load tensor data
        tensor_path = path.with_suffix(".pt")
        if tensor_path.exists():
            tensor_data = torch.load(tensor_path, map_location="cpu")

            self.memory_keys = tensor_data["memory_keys"].to(self.memory_keys.device)
            self.memory_values = tensor_data["memory_values"].to(self.memory_values.device)
            self.memory_usage = tensor_data["memory_usage"].to(self.memory_usage.device)
            self.memory_timestamps = tensor_data["memory_timestamps"].to(self.memory_timestamps.device)

        # Load metadata and state
        metadata_path = path.with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Restore state
            state = metadata.get("state", {})
            self.current_size = state.get("current_size", 0)
            self.global_step = state.get("global_step", 0)
            self.write_count = state.get("write_count", 0)
            self.read_count = state.get("read_count", 0)

            # Restore metadata
            self.memory_metadata = metadata.get("memory_metadata", [])
            self.memory_metadata.extend([None] * (self.config.capacity - len(self.memory_metadata)))

            # Restore novelty EMA
            novelty_ema = metadata.get("novelty_ema")
            if novelty_ema is not None:
                self.novelty_ema = torch.tensor(novelty_ema, device=self.memory_keys.device)

        logger.info(f"Loaded memory bank from {path} ({self.current_size} memories)")

    def clear(self):
        """Clear all memories."""
        self.memory_keys.zero_()
        self.memory_values.zero_()
        self.memory_usage.zero_()
        self.memory_timestamps.zero_()
        self.memory_metadata = [None] * self.config.capacity
        self.current_size = 0
        self.novelty_ema = None
        logger.info("Cleared all memories")


def create_memory_bank(
    capacity: int, d_mem: int, device: torch.device | None = None, **config_kwargs
) -> CognateLTMBank:
    """
    Factory function to create a memory bank.

    Args:
        capacity: Memory bank capacity
        d_mem: Memory dimension
        device: Device to place memory on
        **config_kwargs: Additional configuration parameters

    Returns:
        CognateLTMBank: Initialized memory bank
    """
    config = MemoryConfig(capacity=capacity, d_mem=d_mem, **config_kwargs)

    memory_bank = CognateLTMBank(config)

    if device is not None:
        memory_bank = memory_bank.to(device)

    return memory_bank


if __name__ == "__main__":
    # Test the LTM bank
    logging.basicConfig(level=logging.INFO)

    print("Testing Cognate LTM Bank...")

    # Create memory bank
    config = MemoryConfig(capacity=100, d_mem=64, topk=5)
    memory_bank = create_memory_bank(config.capacity, config.d_mem)

    print(f"Created memory bank: {memory_bank.get_stats()}")

    # Test writing memories
    for i in range(20):
        key = torch.randn(64)
        value = torch.randn(64)
        surprise = torch.rand(1).item()
        novelty = memory_bank.compute_novelty(value)

        written, gate_prob, slot = memory_bank.write(key, value, surprise, novelty)
        if written:
            print(f"Wrote memory {i} to slot {slot} (gate_prob={gate_prob:.3f})")

    print(f"After writing: {memory_bank.get_stats()}")

    # Test reading memories
    query = torch.randn(64)
    values, keys, indices, sims = memory_bank.read(query, return_similarities=True)

    print(f"Retrieved {len(indices)} memories with similarities: {sims[:len(indices)]}")

    # Test persistence
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        memory_bank.save(f.name)

        # Create new bank and load
        new_bank = create_memory_bank(config.capacity, config.d_mem)
        new_bank.load(f.name)

        print(f"Loaded memory bank: {new_bank.get_stats()}")
        assert new_bank.current_size == memory_bank.current_size

    print("✅ LTM bank test completed!")
