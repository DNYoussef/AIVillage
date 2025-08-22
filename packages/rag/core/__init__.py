"""
packages.rag.core - Core RAG functionality

Provides the main RAG pipeline and orchestration components.
Uses consolidated implementations.
"""

from .pipeline import RAGPipeline

try:
    __all__ = ["RAGPipeline", "HyperRAG", "HyperRAGConfig", "QueryMode", "MemoryType"]
except ImportError:
    __all__ = ["RAGPipeline"]
