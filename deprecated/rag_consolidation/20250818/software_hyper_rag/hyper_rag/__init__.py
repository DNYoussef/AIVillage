"""
Hyper RAG System - Advanced Retrieval Augmented Generation

Managed by Sage Agent, combines:
- Vector stores for semantic similarity
- Graph RAG for relationship mapping
- BayesNet-like probability ratings
- Dual context tags (book/chapter summaries)
- Cognitive Nexus for analysis
- Hippo RAG for frequent idea caching
- Read-only access for all agents
"""

from .bayes_engine import BayesianBeliefEngine
from .cognitive_nexus import CognitiveNexus
from .hyper_rag_pipeline import (
    ContextTag,
    HyperRAGPipeline,
    KnowledgeItem,
    RAGType,
    RetrievalResult,
)

__all__ = [
    "HyperRAGPipeline",
    "RAGType",
    "KnowledgeItem",
    "RetrievalResult",
    "ContextTag",
    "BayesianBeliefEngine",
    "CognitiveNexus",
]
