"""
Hypergraph Knowledge System

Dual-memory architecture for AI Village RAG system:
- Hippo-Index: Fast episodic memory for recent interactions
- Hypergraph-KG: Semantic knowledge graph with n-ary relationships

This module provides the foundation for Sprint R-2 implementation.
"""

from .models import Hyperedge, HippoNode
from .migrations import run_cypher_migrations

__all__ = ['Hyperedge', 'HippoNode', 'run_cypher_migrations']
