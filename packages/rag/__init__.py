"""
Unified RAG System - Hippocampus + Graph + Vector + Cognitive Nexus

This module provides a comprehensive RAG (Retrieval-Augmented Generation) system
that integrates multiple advanced retrieval paradigms:

1. **HippoRAG**: Neurobiologically-inspired episodic memory with time-based decay
2. **GraphRAG**: Knowledge graph with Bayesian trust propagation and semantic relationships
3. **VectorRAG**: High-performance vector similarity search with contextual embeddings
4. **Cognitive Nexus**: Multi-perspective reasoning, analysis, and synthesis system
5. **Database Integration**: Distributed fog computing with edge device coordination

Key Features:
- Multihop caching with hippocampus-style rapid storage and retrieval
- Bayesian probabilistic reasoning with trust propagation across knowledge graphs
- Double context tags (book/chapter summaries) for enhanced retrieval accuracy
- Creativity system for discovering non-obvious paths and connections
- Graph gap detection and automated node proposal system
- Edge device integration with mobile-optimized resource management
- P2P network integration for distributed knowledge sharing

Components:
- HyperRAG: Main orchestrator integrating all subsystems
- CognitiveNexus: Complex analysis and reasoning engine
- HippoIndex: Fast episodic memory storage with neurobiological patterns
- BayesianTrustGraph: Knowledge graph with probabilistic reasoning
- VectorEngine: High-performance similarity search with contextual awareness
- GraphFixer: Automated gap detection and knowledge completion
- CreativityEngine: Non-obvious path discovery and insight generation
"""

from .analysis.graph_fixer import GraphFixer
from .core.cognitive_nexus import AnalysisType, CognitiveNexus, ReasoningStrategy
from .core.hyper_rag import HyperRAG, MemoryType, QueryMode
from .creativity.insight_engine import CreativityEngine
from .graph.bayesian_trust_graph import BayesianTrustGraph, Relationship, RelationshipType, create_graph_node
from .integration.edge_device_bridge import EdgeDeviceRAGBridge
from .integration.fog_compute_bridge import FogComputeBridge
from .integration.p2p_network_bridge import P2PNetworkRAGBridge
from .memory.hippo_index import HippoIndex, create_episodic_document, create_hippo_node
from .vector.contextual_vector_engine import ContextualChunk, ContextualVectorEngine, VectorDocument

__all__ = [
    # Core system
    "HyperRAG",
    "CognitiveNexus",
    "QueryMode",
    "MemoryType",
    "ReasoningStrategy",
    "AnalysisType",
    # Memory subsystems
    "HippoIndex",
    "create_hippo_node",
    "create_episodic_document",
    # Knowledge graph
    "BayesianTrustGraph",
    "create_graph_node",
    "Relationship",
    "RelationshipType",
    # Vector search
    "ContextualVectorEngine",
    "VectorDocument",
    "ContextualChunk",
    # Analysis and enhancement
    "GraphFixer",
    "CreativityEngine",
    # Integration bridges
    "EdgeDeviceRAGBridge",
    "P2PNetworkRAGBridge",
    "FogComputeBridge",
]

__version__ = "2.0.0"
__description__ = "Unified RAG System with Hippocampus, Graph, Vector, and Cognitive Nexus Integration"
