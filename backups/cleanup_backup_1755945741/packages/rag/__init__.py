"""
packages.rag - RAG system module

This module provides the interface expected by examples and integrations,
bridging to the actual implementations in core.rag.
"""

# Re-export key components for backward compatibility
try:
    __all__ = ["HyperRAG", "QueryMode", "MemoryType", "create_hyper_rag", "RAGPipeline"]
except ImportError:
    # Components not yet fully implemented
    __all__ = []
