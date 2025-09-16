"""
Base RAG System Module

Provides the foundational abstract classes and interfaces for all RAG implementations.
This module establishes the hierarchy: base -> hyperrag -> minirag specialization chain.
"""

from .base_rag_system import BaseRAGSystem, RAGConfiguration, RAGResult
from .factory import RAGFactory

__all__ = ["BaseRAGSystem", "RAGConfiguration", "RAGResult", "RAGFactory"]