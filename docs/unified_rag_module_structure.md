# Unified RAG Module Structure

This document summarizes the consolidation of legacy `core/rag` modules into the production `src/unified_rag` package.  The table below highlights the main feature sets in both versions and shows where the unified implementation resides.

| Domain | Legacy Module (core/rag) | Unified Module (src/unified_rag) | Key Features from Legacy | Key Features from Unified | Merged Approach |
|-------|--------------------------|----------------------------------|--------------------------|---------------------------|----------------|
| Cognitive | `core/rag/cognitive_nexus.py` | `cognitive/cognitive_nexus.py` | Bayesian belief networks, advanced analysis enums | Integration layer with existing reasoning engines | Unified module now includes legacy enums and dataclasses with the integration interface |
| Graph | `core/rag/graph/bayesian_trust_graph.py` | `graph/bayesian_knowledge_graph.py` | Trust propagation, relationship types, graph node helpers | Persistent knowledge graph with probabilistic inference | Unified module incorporates trust enums, relationship helpers, and aliases |
| Memory | `core/rag/memory/hippo_index.py` | `memory/hippo_memory_system.py` | Episodic documents, hippocampal nodes, confidence types | Consolidation workflows and forgetting curves | Unified memory system exports legacy structures and helper creators |
| Vector | `core/rag/vector/contextual_vector_engine.py` | `vector/dual_context_vector.py` | Chunking strategies, similarity metrics, context tag creators | Hierarchical search with dual context embeddings | Unified vector module exposes legacy strategies and helper functions with compatibility alias |
| System | `core/rag/unified_rag_system.py` | `core/unified_rag_system.py` | Query modes and context tags | Comprehensive orchestration of all RAG components | Unified system retains modern orchestration; legacy enumerations preserved where applicable |

## Deprecation

The legacy files in `core/rag` have been removed or reduced to thin wrappers that import from `unified_rag`.  New development should target the `unified_rag` package exclusively.


## Related Documentation

- [Creative Search and Gap Detection](rag/creative_search_and_gap_detection.md)
