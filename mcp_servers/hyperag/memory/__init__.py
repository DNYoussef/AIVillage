"""HypeRAG Dual-Memory System

Brain-inspired dual-memory architecture combining:
- HippoIndex: Fast episodic memory (hippocampus-like)
- HypergraphKG: Deep semantic memory (neocortex-like)
"""

from .base import Document, Edge, MemoryBackend, Node
from .consolidator import ConsolidationConfig, MemoryConsolidator
from .hippo_index import EpisodicDocument, HippoIndex, HippoNode
from .hypergraph_kg import Hyperedge, HypergraphKG, SemanticNode, Subgraph
from .schemas import HippoSchema, HypergraphSchema

__all__ = [
    "ConsolidationConfig",
    "Document",
    "Edge",
    "EpisodicDocument",
    "HippoIndex",
    "HippoNode",
    "HippoSchema",
    "Hyperedge",
    "HypergraphKG",
    "HypergraphSchema",
    "MemoryBackend",
    "MemoryConsolidator",
    "Node",
    "SemanticNode",
    "Subgraph"
]
