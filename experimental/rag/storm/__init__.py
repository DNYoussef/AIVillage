"""Experimental STORM-based RAG system.

Formerly located at the repository root as ``rag_system``.
This module provides a lightweight pipeline and related utilities for
educational mobile scenarios. It remains experimental and is not
intended for production use.
"""

import warnings

warnings.warn(
    "experimental.rag.storm is a stub implementation. " "Replace with actual implementation before production use.",
    UserWarning,
    stacklevel=2,
)


class Ragsystem:
    """Placeholder class for testing."""

    def __init__(self) -> None:
        self.initialized = True

    def process(self, *args, **kwargs):
        """Stub processing method."""
        return {"status": "stub", "module": "rag_system"}


from .wikipedia_storm_pipeline import (
    ContentDatabase,
    EducationalContentGenerator,
    Lesson,
    OfflineOptimizedRAG,
    Perspective,
    WikipediaSTORMPipeline,
)

# Module-level exports
__all__ = [
    "ContentDatabase",
    "EducationalContentGenerator",
    "Lesson",
    "OfflineOptimizedRAG",
    "Perspective",
    "Ragsystem",
    "WikipediaSTORMPipeline",
]
