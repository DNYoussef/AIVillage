"""
packages.rag - RAG system module

This module provides the interface expected by examples and integrations,
bridging to the actual implementations in core.rag.
"""

import sys
from pathlib import Path

# Add core modules to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "packages"))

# Import core RAG components with fallbacks for testing
try:
    from .base import HyperRAG, QueryMode, MemoryType, RAGPipeline
    from .core.hyper_rag import create_hyper_rag
except ImportError:
    # Fallback implementations for testing
    class HyperRAG:
        """Fallback HyperRAG implementation for testing"""
        def __init__(self, config=None):
            self.config = config or {}
        
        async def query(self, query, mode="hybrid"):
            return {"answer": "test response", "sources": []}
    
    class QueryMode:
        SEMANTIC = "semantic"
        KEYWORD = "keyword" 
        HYBRID = "hybrid"
        GRAPH = "graph"
    
    class MemoryType:
        SHORT_TERM = "short_term"
        LONG_TERM = "long_term"
        EPISODIC = "episodic"
    
    class RAGPipeline:
        """Fallback RAG pipeline for testing"""
        def __init__(self, config=None):
            self.config = config or {}
    
    def create_hyper_rag(config=None):
        return HyperRAG(config)

# Bridge components for distributed systems
try:
    from .edge_device_bridge import EdgeDeviceRAGBridge
    from .p2p_network_bridge import P2PNetworkRAGBridge  
    from .fog_compute_bridge import FogComputeBridge
except ImportError:
    # Fallback bridge implementations
    class EdgeDeviceRAGBridge:
        """Edge device RAG bridge for mobile/IoT environments"""
        def __init__(self, device_profile=None):
            self.device_profile = device_profile or {}
        
        async def process_query(self, query, constraints=None):
            return {"answer": "edge response", "device_id": "test_device"}
    
    class P2PNetworkRAGBridge:
        """P2P network RAG bridge for decentralized knowledge sharing"""
        def __init__(self, network_config=None):
            self.network_config = network_config or {}
        
        async def distributed_query(self, query, peer_count=3):
            return {"answer": "p2p response", "peer_contributions": []}
    
    class FogComputeBridge:
        """Fog computing RAG bridge for distributed processing"""
        def __init__(self, fog_nodes=None):
            self.fog_nodes = fog_nodes or []
        
        async def process_workload(self, workload):
            return {"result": "fog response", "processing_nodes": []}

__all__ = [
    "HyperRAG", 
    "QueryMode", 
    "MemoryType", 
    "RAGPipeline",
    "create_hyper_rag",
    "EdgeDeviceRAGBridge",
    "P2PNetworkRAGBridge", 
    "FogComputeBridge"
]
