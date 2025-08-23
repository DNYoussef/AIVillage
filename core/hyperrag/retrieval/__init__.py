"""
HyperRAG Retrieval Engines

High-performance retrieval systems:
- VectorEngine: Contextual similarity search
- GraphEngine: Bayesian trust networks
"""

try:
    from .vector_engine import (
        ContextualVectorEngine,
        ChunkingStrategy,
        SimilarityMetric,
        ContextTag
    )
    # Alias for backward compatibility
    VectorEngine = ContextualVectorEngine
except ImportError:
    ContextualVectorEngine = None
    ChunkingStrategy = None
    SimilarityMetric = None
    ContextTag = None
    VectorEngine = None

try:
    from .graph_engine import (
        BayesianTrustGraph,
        RelationType,
        TrustLevel,
        GraphNode
    )
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
    "GraphNode"
]