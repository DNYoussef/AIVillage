"""
HypeRAG Dual-Memory System

Brain-inspired dual-memory architecture combining:
- HippoIndex: Fast episodic memory (hippocampus-like)
- HypergraphKG: Deep semantic memory (neocortex-like)
"""

from .hippo_index import HippoIndex, EpisodicDocument, HippoNode
from .hypergraph_kg import HypergraphKG, SemanticNode, Hyperedge, Subgraph
from .consolidator import MemoryConsolidator, ConsolidationConfig
from .schemas import HippoSchema, HypergraphSchema
from .base import Document, Node, Edge, MemoryBackend

__all__ = [
    "HippoIndex",
    "HypergraphKG",
    "MemoryConsolidator",
    "EpisodicDocument",
    "HippoNode",
    "SemanticNode",
    "Hyperedge",
    "Subgraph",
    "ConsolidationConfig",
    "HippoSchema",
    "HypergraphSchema",
    "Document",
    "Node",
    "Edge",
    "MemoryBackend"
]
