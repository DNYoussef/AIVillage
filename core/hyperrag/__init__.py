"""
Consolidated HyperRAG System - Main Exports

Unified access to all HyperRAG components:
- Main orchestrator (HyperRAG)
- Memory systems (HippoRAG, GraphRAG, VectorRAG)
- Cognitive systems (CognitiveNexus, InsightEngine, GraphFixer)
- Integration bridges (Edge, P2P, Fog)

This consolidates 53+ scattered files into a single, clean API.
"""

# Main orchestrator
from .hyperrag import (
    HyperRAG,
    HyperRAGSystem,
    HyperRAGConfig,
    QueryMode,
    MemoryType,
    RetrievedInformation,
    SynthesizedAnswer
)

# Memory subsystems
try:
    from .memory.hippo_index import HippoIndex, EpisodicDocument
except ImportError:
    HippoIndex = None
    EpisodicDocument = None

# Retrieval engines
try:
    from .retrieval.graph_engine import BayesianTrustGraph
    from .retrieval.vector_engine import ContextualVectorEngine
except ImportError:
    BayesianTrustGraph = None
    ContextualVectorEngine = None

# Cognitive systems
try:
    from .cognitive.cognitive_nexus import (
        CognitiveNexus,
        AnalysisType,
        ReasoningStrategy,
        ConfidenceLevel
    )
    from .cognitive.insight_engine import CreativityEngine
    from .cognitive.graph_fixer import GraphFixer
except ImportError:
    CognitiveNexus = None
    AnalysisType = None
    ReasoningStrategy = None
    ConfidenceLevel = None
    CreativityEngine = None
    GraphFixer = None

# Integration bridges
try:
    from .integration.edge_device_bridge import EdgeDeviceRAGBridge
    from .integration.p2p_network_bridge import P2PNetworkRAGBridge
    from .integration.fog_compute_bridge import FogComputeBridge
except ImportError:
    EdgeDeviceRAGBridge = None
    P2PNetworkRAGBridge = None
    FogComputeBridge = None

# Main exports - guaranteed to be available
__all__ = [
    # Core system (always available)
    "HyperRAG",
    "HyperRAGSystem", 
    "HyperRAGConfig",
    "QueryMode",
    "MemoryType",
    "RetrievedInformation",
    "SynthesizedAnswer",
    
    # Subsystems (may be None if not available)
    "HippoIndex",
    "EpisodicDocument",
    "BayesianTrustGraph", 
    "ContextualVectorEngine",
    "CognitiveNexus",
    "AnalysisType",
    "ReasoningStrategy",
    "ConfidenceLevel",
    "CreativityEngine",
    "GraphFixer",
    "EdgeDeviceRAGBridge",
    "P2PNetworkRAGBridge", 
    "FogComputeBridge"
]

# Version info
__version__ = "1.0.0"
__author__ = "AIVillage HyperRAG Consolidation"
__description__ = "Unified HyperRAG system consolidating 53+ scattered implementations"