"""
HyperRAG Retrieval Engines

High-performance retrieval systems:
- VectorEngine: Contextual similarity search
- GraphEngine: Bayesian trust networks
"""

try:
    from .vector_engine import ChunkingStrategy, ContextTag, ContextualVectorEngine, SimilarityMetric

    # Alias for backward compatibility
    VectorEngine = ContextualVectorEngine
except ImportError:
    ContextualVectorEngine = None
    ChunkingStrategy = None
    SimilarityMetric = None
    ContextTag = None
    VectorEngine = None

try:
    from .graph_engine import BayesianTrustGraph, GraphNode, RelationType, TrustLevel

    # Alias for backward compatibility
    GraphEngine = BayesianTrustGraph
except ImportError:
    BayesianTrustGraph = None
    RelationType = None
    TrustLevel = None
    GraphNode = None
    GraphEngine = None

__all__ = [
    "ContextualVectorEngine",
    "VectorEngine",
    "ChunkingStrategy",
    "SimilarityMetric",
    "ContextTag",
    "BayesianTrustGraph",
    "GraphEngine",
    "RelationType",
    "TrustLevel",
    "GraphNode",
]
