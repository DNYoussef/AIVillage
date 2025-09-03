"""
Unified RAG System - The Ultimate RAG Implementation

Combines the best of all RAG approaches with strategic MCP server integration:
- Advanced Ingestion System (Markitdown MCP)
- HippoRAG Memory Architecture (Memory MCP + HuggingFace MCP)
- Dual Context Vector RAG (HuggingFace MCP)
- Bayesian Knowledge Graph RAG (DeepWiki MCP + Sequential Thinking MCP)
- Cognitive Nexus Integration (Sequential Thinking MCP + Memory MCP)
- Creative Graph Search (Sequential Thinking MCP)
- Missing Node Detection (Memory MCP + DeepWiki MCP)
"""

from .core.unified_rag_system import UnifiedRAGSystem
from .core.mcp_coordinator import MCPCoordinator
from .ingestion.advanced_ingestion_engine import AdvancedIngestionEngine
from .memory.hippo_memory_system import HippoMemorySystem
from .vector.dual_context_vector import DualContextVectorRAG
from .graph.bayesian_knowledge_graph import BayesianKnowledgeGraphRAG
from .cognitive.cognitive_nexus import CognitiveNexusIntegration
from .graph.creative_graph_search import CreativeGraphSearch
from .graph.missing_node_detector import MissingNodeDetector

__version__ = "1.0.0"
__author__ = "AI Village"

# Export main components
__all__ = [
    "UnifiedRAGSystem",
    "MCPCoordinator", 
    "AdvancedIngestionEngine",
    "HippoMemorySystem",
    "DualContextVectorRAG",
    "BayesianKnowledgeGraphRAG",
    "CognitiveNexusIntegration",
    "CreativeGraphSearch",
    "MissingNodeDetector",
]