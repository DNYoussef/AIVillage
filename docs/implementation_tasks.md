# Implementation Tasks

This document summarizes missing features and placeholders throughout the
repository based on the various documentation files.

## Core Areas
- **Agent Forge Training** – advanced phases such as Quiet-STaR, expert vectors
  and ADAS optimisation are not implemented.  The current pipeline only supports
  basic model merging and a simplified training loop.
- **Self‑Evolving System** – `SelfEvolvingSystem` in `agents/unified_base_agent.py`
  remains a stub meant for demonstrations.  Real integration of quality
  assurance, decision‑making modules and capability evolution is future work.
- **King Agent Evolution** – the evolution manager trains models using randomly
  generated data.  Replace this with real datasets and evaluation metrics.
- **Knowledge Tracking** – rollback and long‑term persistence are incomplete in
  `KnowledgeTracker` and `KnowledgeEvolutionTracker`.
- **RAG Pipeline** – the graph store and user intent interpreter contain
  placeholder logic.  Confidence estimation and creative exploration scores are
  rudimentary.

## How to Proceed
1. **Data and Tokenizers** – populate `rag_system/utils/token_data` with the
   actual `cl100k_base.tiktoken` file from the `tiktoken` project.
2. **GraphStore Implementation** – replace the temporary edge creation in
   `GraphStore.add_documents` with real similarity metrics (e.g., cosine
   similarity between embeddings).
3. **Embedding Fallback** – when the Transformers library cannot load models,
   `BERTEmbeddingModel` returns random embeddings.  Provide a deterministic
   fallback (e.g., hashing tokens) so repeated runs are reproducible.
4. **Intent Interpretation** – expand `UserIntentInterpreterAgent` with the
   keyword patterns described in `docs/system_overview.md`.
5. **Exploration Scoring** – implement a more robust creativity score in
   `ExplorationMode._calculate_creativity_score` by combining novelty and
   relevance of edges.
6. **Confidence Estimation** – `ConfidenceEstimator` should consider retrieval
   scores and response length instead of returning a constant.
7. **Rollback Support** – `KnowledgeTracker.rollback_change` should reverse the
   recorded modification in the knowledge graph.
8. **Latent Space Activation** – flesh out activation and evolution methods to
   update `latent_space` with embeddings and perform simple averaging when
   evolving.
9. **Cognitive Nexus** – implement simple query and update operations that call
   the reasoning engine and self‑referential processor.
10. **Groks in the Shadows** – detect early grokking events by monitoring
    gradient variance collapse and trigger adaptive hyperparameter boosts. See
    `docs/groks_in_the_shadows.md` for an overview.

These steps bring the repository closer to the roadmap in
`docs/complete_agent_forge_pipeline.md` and
`docs/geometry_aware_training.md`.
