"""Hypergraph Knowledge System

Dual-memory architecture for AI Village RAG system:
- Hippo-Index: Fast episodic memory for recent interactions
- Hypergraph-KG: Semantic knowledge graph with n-ary relationships

This module provides the foundation for Sprint R-2 implementation.
"""

from .migrations import run_cypher_migrations
from .models import HippoNode, Hyperedge

__all__ = ["HippoNode", "Hyperedge", "run_cypher_migrations"]
