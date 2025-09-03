"""
Vector subsystem for HyperRAG - High-performance contextual vector search.

This module provides vector similarity search with dual context tags
and semantic chunking capabilities.
"""

from unified_rag.vector.dual_context_vector import (
    ChunkingStrategy,
    ContextTag,
    ContextualVectorEngine,
    SimilarityMetric,
    VectorDocument,
    VectorSearchResult,
    create_book_chapter_contexts,
    create_context_tag,
)

__all__ = [
    "ContextualVectorEngine",
    "VectorDocument",
    "ContextTag",
    "VectorSearchResult",
    "ChunkingStrategy",
    "SimilarityMetric",
    "create_context_tag",
    "create_book_chapter_contexts",
]
