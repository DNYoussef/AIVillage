"""
HyperRAG Memory Subsystems

Neurobiologically-inspired memory systems:
- HippoRAG: Episodic memory with temporal decay
- Memory interfaces and utilities
"""

try:
    from .hippo_index import ConfidenceType, EpisodicDocument, HippoIndex, MemoryType, create_hippo_node
except ImportError as e:
    # Graceful fallback for missing dependencies
    HippoIndex = None
    EpisodicDocument = None
    MemoryType = None
    ConfidenceType = None
    create_hippo_node = None

__all__ = ["HippoIndex", "EpisodicDocument", "MemoryType", "ConfidenceType", "create_hippo_node"]
