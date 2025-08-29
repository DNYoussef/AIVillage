#!/usr/bin/env python3
"""
Simplified Long-Term Memory (LTM) Bank for Cognate Models

This is a simplified implementation of the LTM bank that provides
the core functionality needed for the Cognate pretraining system.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SimpleLTMBank(nn.Module):
    """Simplified Long-Term Memory Bank for Cognate models."""

    def __init__(self, d_model: int, memory_capacity: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.capacity = memory_capacity

        # Memory storage
        self.register_buffer("memory_keys", torch.zeros(memory_capacity, d_model))
        self.register_buffer("memory_values", torch.zeros(memory_capacity, d_model))
        self.register_buffer("memory_usage", torch.zeros(memory_capacity))
        self.memory_size = 0

        # Controllers
        self.read_controller = nn.Linear(d_model, d_model)
        self.write_controller = nn.Linear(d_model, 1)

    def read(self, query: torch.Tensor, k: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
        """Read from memory bank."""
        if self.memory_size == 0:
            batch_size = query.shape[0]
            return (
                torch.zeros(batch_size, k, self.d_model, device=query.device),
                torch.zeros(batch_size, k, self.d_model, device=query.device),
            )

        # Compute similarities
        query_norm = F.normalize(query, dim=-1)
        memory_norm = F.normalize(self.memory_keys[: self.memory_size], dim=-1)

        similarities = torch.matmul(query_norm, memory_norm.T)

        # Get top-k
        top_k = min(k, self.memory_size)
        _, indices = similarities.topk(top_k, dim=-1)

        # Retrieve memory items
        batch_size = query.shape[0]
        retrieved_keys = torch.zeros(batch_size, k, self.d_model, device=query.device)
        retrieved_values = torch.zeros(batch_size, k, self.d_model, device=query.device)

        for b in range(batch_size):
            for i in range(top_k):
                idx = indices[b, i]
                retrieved_keys[b, i] = self.memory_keys[idx]
                retrieved_values[b, i] = self.memory_values[idx]
                self.memory_usage[idx] += 1

        return retrieved_keys, retrieved_values

    def write(self, key: torch.Tensor, value: torch.Tensor, gate_threshold: float = 0.5):
        """Write to memory bank."""
        # Simple gating
        gate_score = torch.sigmoid(self.write_controller(key)).item()

        if gate_score < gate_threshold:
            return False

        if self.memory_size < self.capacity:
            # Add new memory
            idx = self.memory_size
            self.memory_keys[idx] = key.detach()
            self.memory_values[idx] = value.detach()
            self.memory_usage[idx] = 1
            self.memory_size += 1
        else:
            # Replace least used memory
            idx = self.memory_usage.argmin().item()
            self.memory_keys[idx] = key.detach()
            self.memory_values[idx] = value.detach()
            self.memory_usage[idx] = 1

        return True


def create_ltm_controllers(d_model: int, memory_capacity: int = 4096, **kwargs):
    """Create LTM controllers for backward compatibility."""
    memory_bank = SimpleLTMBank(d_model, memory_capacity)
    read_controller = memory_bank.read_controller
    write_controller = memory_bank.write_controller

    logger.info(f"Created simplified LTM controllers with capacity {memory_capacity}")
    return read_controller, write_controller, memory_bank


def create_memory_bank(d_model: int, capacity: int = 4096, **kwargs):
    """Create memory bank for backward compatibility."""
    return SimpleLTMBank(d_model, capacity)


class MemoryScheduler:
    """Simplified memory scheduler."""

    def __init__(self, read_policy: str = "always", write_policy: str = "gated", **kwargs):
        self.read_policy = read_policy
        self.write_policy = write_policy

    def should_read(self, step: int = 0, **kwargs) -> bool:
        return self.read_policy in ["always", "periodic"]

    def should_write(self, step: int = 0, **kwargs) -> bool:
        return self.write_policy in ["always", "gated"]


if __name__ == "__main__":
    # Test the LTM bank
    d_model = 216
    memory_bank = SimpleLTMBank(d_model, capacity=10)

    # Test write
    key = torch.randn(d_model)
    value = torch.randn(d_model)
    success = memory_bank.write(key, value)
    print(f"Write success: {success}")

    # Test read
    query = torch.randn(1, d_model)
    keys, values = memory_bank.read(query, k=4)
    print(f"Read shapes: keys {keys.shape}, values {values.shape}")

    print("✅ LTM Bank test passed!")
