"""
Dual Context Vector RAG with HuggingFace MCP Integration

Advanced vector similarity search with contextual embeddings,
dual context tagging, and hierarchical retrieval strategies.
"""

from .dual_context_vector import DualContextVectorRAG
from .contextual_embeddings import ContextualEmbeddingEngine
from .hierarchical_search import HierarchicalSearchEngine

__all__ = ["DualContextVectorRAG", "ContextualEmbeddingEngine", "HierarchicalSearchEngine"]