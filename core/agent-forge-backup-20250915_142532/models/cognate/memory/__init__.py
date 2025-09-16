"""
Cognate Memory System

This module implements the Titans-style Long-Term Memory (LTM) system with
cross-attention integration for the Cognate model. Features:

- Surprise × Novelty gating for selective memory writing
- Entropy-based gating for memory reading
- Cross-attention memory integration
- Memory bank persistence and serialization
"""

from .cross_attention import (
    MemoryCrossAttention,
    create_memory_cross_attention,
)
from .ltm_bank import (
    CognateLTMBank,
    MemoryConfig,
    MemoryItem,
    create_memory_bank,
)
from .memory_scheduler import (
    MemoryScheduler,
    ReadPolicy,
    WritePolicy,
)

__all__ = [
    "CognateLTMBank",
    "MemoryConfig",
    "MemoryItem",
    "create_memory_bank",
    "MemoryCrossAttention",
    "create_memory_cross_attention",
    "MemoryScheduler",
    "ReadPolicy",
    "WritePolicy",
]
