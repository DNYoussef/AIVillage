"""
Hyper RAG Pipeline - Advanced Retrieval Augmented Generation System

Integrates multiple RAG approaches with Bayesian probability ratings:
- Vector RAG: Semantic similarity search using embeddings
- Graph RAG: Relationship-based knowledge graph traversal
- Bayesian Belief Engine: Dynamic probability ratings for all beliefs/ideas
- Cognitive Nexus: Multi-perspective analysis and synthesis
- Hippo Cache: Frequent idea caching system
- Dual Context Tags: Book/chapter summary context hierarchy

Managed by Sage Agent with read-only access for all other agents.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from .bayes_engine import BayesianBeliefEngine
from .cognitive_nexus import CognitiveNexus


class RAGType(Enum):
    """Types of RAG retrieval methods."""

    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"
    BAYESIAN = "bayesian"


@dataclass
class ContextTag:
    """Dual context tags for hierarchical knowledge organization."""

    book_summary: str  # High-level topic/domain summary
    chapter_summary: str  # Specific section/subtopic summary
    tag_id: str
    probability_weight: float = 1.0


@dataclass
class KnowledgeItem:
    """Individual knowledge item with Bayesian probability."""

    content: str
    item_id: str
    context_tags: list[ContextTag]
    embedding: np.ndarray | None = None
    belief_probability: float = 0.5  # Initial neutral probability
    source_confidence: float = 0.8
    last_updated: datetime = None
    access_count: int = 0
    semantic_connections: list[str] = None  # Connected item IDs


@dataclass
class RetrievalResult:
    """Result from RAG retrieval with probability scoring."""

    items: list[KnowledgeItem]
    retrieval_method: RAGType
    confidence_score: float
    bayesian_scores: dict[str, float]  # Item ID -> Bayesian probability
    semantic_coherence: float
    context_relevance: float
    total_items_considered: int


class HyperRAGPipeline:
    """
    Advanced RAG pipeline with Bayesian probability integration.

    Combines vector similarity, graph relationships, and Bayesian belief
    propagation to provide probabilistically-weighted knowledge retrieval.
    """

    def __init__(self, sage_agent_id: str = "sage"):
        self.sage_agent_id = sage_agent_id

        # Initialize core components
        self.belief_engine = BayesianBeliefEngine("hyper_rag_beliefs")
        self.cognitive_nexus = CognitiveNexus()

        # Knowledge storage
        self.knowledge_items: dict[str, KnowledgeItem] = {}
        self.context_hierarchy: dict[str, list[str]] = {}  # Book -> Chapter IDs
        self.semantic_graph: dict[str, list[str]] = {}  # Item ID -> Connected Item IDs

        # Caching system (Hippo Cache)
        self.frequent_items_cache: dict[str, KnowledgeItem] = {}
        self.cache_access_threshold = 5  # Items accessed 5+ times get cached

        # Performance metrics
        self.retrieval_stats = {
            "total_queries": 0,
            "vector_retrievals": 0,
            "graph_retrievals": 0,
            "hybrid_retrievals": 0,
            "bayesian_retrievals": 0,
            "cache_hits": 0,
            "avg_response_time": 0.0,
        }

        logger = logging.getLogger(__name__)
        logger.info(f"Hyper RAG Pipeline initialized for Sage Agent: {sage_agent_id}")

    async def ingest_knowledge(
        self,
        content: str,
        book_summary: str,
        chapter_summary: str,
        source_confidence: float = 0.8,
    ) -> str:
        """
        Ingest new knowledge with dual context tags and Bayesian belief.

        Args:
            content: The knowledge content to store
            book_summary: High-level topic/domain context
            chapter_summary: Specific section/subtopic context
            source_confidence: Confidence in the source (0.0 to 1.0)

        Returns:
            str: Generated knowledge item ID
        """
        import hashlib

        # Generate unique item ID
        item_id = hashlib.md5(f"{content}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]

        # Create context tags
        tag_id = hashlib.md5(f"{book_summary}_{chapter_summary}".encode()).hexdigest()[:12]
        context_tag = ContextTag(
            book_summary=book_summary,
            chapter_summary=chapter_summary,
            tag_id=tag_id,
            probability_weight=source_confidence,
        )

        # Generate embedding (simplified - in production would use actual embedding model)
        embedding = self._generate_embedding(content)

        # Create knowledge item
        knowledge_item = KnowledgeItem(
            content=content,
            item_id=item_id,
            context_tags=[context_tag],
            embedding=embedding,
            belief_probability=source_confidence,  # Initial probability based on source confidence
            source_confidence=source_confidence,
            last_updated=datetime.now(),
            access_count=0,
            semantic_connections=[],
        )

        # Store knowledge item
        self.knowledge_items[item_id] = knowledge_item

        # Update context hierarchy
        if book_summary not in self.context_hierarchy:
            self.context_hierarchy[book_summary] = []
        if tag_id not in self.context_hierarchy[book_summary]:
            self.context_hierarchy[book_summary].append(tag_id)

        # Add belief to Bayesian engine
        belief_description = f"Knowledge from {book_summary}/{chapter_summary}: {content[:100]}..."
        self.belief_engine.add_belief(item_id, belief_description, source_confidence)

        # Update semantic connections
        await self._update_semantic_connections(knowledge_item)

        logger = logging.getLogger(__name__)
        logger.info(f"Ingested knowledge item {item_id} with {source_confidence:.2f} confidence")

        return item_id

    def _generate_embedding(self, content: str) -> np.ndarray:
        """Generate simplified embedding for content (placeholder implementation)."""
        # In production, this would use a real embedding model like sentence-transformers
        # For now, create a simple hash-based embedding
        import hashlib

        hash_bytes = hashlib.sha256(content.encode()).digest()
        embedding = np.frombuffer(hash_bytes, dtype=np.uint8)[:384].astype(np.float32)
        # Normalize to unit vector
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    async def _update_semantic_connections(self, new_item: KnowledgeItem):
        """Update semantic connections between knowledge items."""
        # Find semantically similar items based on embedding similarity
        similarities = {}

        for item_id, existing_item in self.knowledge_items.items():
            if item_id == new_item.item_id:
                continue

            # Calculate cosine similarity
            if existing_item.embedding is not None and new_item.embedding is not None:
                similarity = np.dot(new_item.embedding, existing_item.embedding)
                if similarity > 0.7:  # High similarity threshold
                    similarities[item_id] = similarity

        # Create bidirectional connections for highly similar items
        for connected_id, similarity in similarities.items():
            # Add connection from new item to existing item
            if connected_id not in new_item.semantic_connections:
                new_item.semantic_connections.append(connected_id)

            # Add connection from existing item to new item
            existing_item = self.knowledge_items[connected_id]
            if new_item.item_id not in existing_item.semantic_connections:
                existing_item.semantic_connections.append(new_item.item_id)

            # Update semantic graph
            if new_item.item_id not in self.semantic_graph:
                self.semantic_graph[new_item.item_id] = []
            if connected_id not in self.semantic_graph[new_item.item_id]:
                self.semantic_graph[new_item.item_id].append(connected_id)

    async def retrieve_knowledge(
        self,
        query: str,
        retrieval_type: RAGType = RAGType.HYBRID,
        max_results: int = 10,
        context_filter: dict[str, str] | None = None,
    ) -> RetrievalResult:
        """
        Retrieve knowledge using specified RAG method with Bayesian scoring.

        Args:
            query: Query string to search for
            retrieval_type: Type of RAG retrieval to use
            max_results: Maximum number of results to return
            context_filter: Optional filter for context tags {"book": "...", "chapter": "..."}

        Returns:
            RetrievalResult with probabilistically scored items
        """
        start_time = datetime.now()
        self.retrieval_stats["total_queries"] += 1

        # Check frequent items cache first
        cache_results = self._check_cache(query)
        if cache_results:
            self.retrieval_stats["cache_hits"] += 1
            return cache_results

        # Filter items by context if specified
        candidate_items = (
            self._apply_context_filter(context_filter) if context_filter else list(self.knowledge_items.values())
        )

        # Perform retrieval based on type
        if retrieval_type == RAGType.VECTOR:
            results = await self._vector_retrieval(query, candidate_items, max_results)
            self.retrieval_stats["vector_retrievals"] += 1

        elif retrieval_type == RAGType.GRAPH:
            results = await self._graph_retrieval(query, candidate_items, max_results)
            self.retrieval_stats["graph_retrievals"] += 1

        elif retrieval_type == RAGType.BAYESIAN:
            results = await self._bayesian_retrieval(query, candidate_items, max_results)
            self.retrieval_stats["bayesian_retrievals"] += 1

        else:  # HYBRID
            results = await self._hybrid_retrieval(query, candidate_items, max_results)
            self.retrieval_stats["hybrid_retrievals"] += 1

        # Update access counts and cache frequently accessed items
        for item in results.items:
            item.access_count += 1
            if item.access_count >= self.cache_access_threshold:
                self.frequent_items_cache[item.item_id] = item

        # Update performance stats
        response_time = (datetime.now() - start_time).total_seconds()
        self.retrieval_stats["avg_response_time"] = (
            self.retrieval_stats["avg_response_time"] * (self.retrieval_stats["total_queries"] - 1) + response_time
        ) / self.retrieval_stats["total_queries"]

        return results

    def _check_cache(self, query: str) -> RetrievalResult | None:
        """Check frequent items cache for query matches."""
        # Simple cache check - in production would use more sophisticated matching
        query_lower = query.lower()
        matching_items = []

        for item in self.frequent_items_cache.values():
            if any(word in item.content.lower() for word in query_lower.split()):
                matching_items.append(item)

        if matching_items:
            # Score cached items
            bayesian_scores = {}
            for item in matching_items:
                belief = self.belief_engine.beliefs.get(item.item_id)
                bayesian_scores[item.item_id] = belief.probability if belief else item.belief_probability

            return RetrievalResult(
                items=matching_items[:5],  # Top 5 cached results
                retrieval_method=RAGType.HYBRID,
                confidence_score=0.9,  # High confidence for cached items
                bayesian_scores=bayesian_scores,
                semantic_coherence=0.8,
                context_relevance=0.8,
                total_items_considered=len(matching_items),
            )

        return None

    def _apply_context_filter(self, context_filter: dict[str, str]) -> list[KnowledgeItem]:
        """Filter knowledge items by context tags."""
        filtered_items = []

        for item in self.knowledge_items.values():
            for tag in item.context_tags:
                book_match = context_filter.get("book", "").lower() in tag.book_summary.lower()
                chapter_match = context_filter.get("chapter", "").lower() in tag.chapter_summary.lower()

                if book_match or chapter_match:
                    filtered_items.append(item)
                    break

        return filtered_items

    async def _vector_retrieval(self, query: str, candidates: list[KnowledgeItem], max_results: int) -> RetrievalResult:
        """Perform vector-based semantic similarity retrieval."""
        query_embedding = self._generate_embedding(query)

        # Calculate similarities
        similarities = []
        for item in candidates:
            if item.embedding is not None:
                similarity = np.dot(query_embedding, item.embedding)
                similarities.append((item, similarity))

        # Sort by similarity and take top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_items = [item for item, sim in similarities[:max_results]]

        # Generate Bayesian scores
        bayesian_scores = {}
        for item in top_items:
            belief = self.belief_engine.beliefs.get(item.item_id)
            bayesian_scores[item.item_id] = belief.probability if belief else item.belief_probability

        return RetrievalResult(
            items=top_items,
            retrieval_method=RAGType.VECTOR,
            confidence_score=np.mean([sim for _, sim in similarities[:max_results]]) if similarities else 0.0,
            bayesian_scores=bayesian_scores,
            semantic_coherence=self._calculate_semantic_coherence(top_items),
            context_relevance=0.8,  # Vector retrieval generally has good context relevance
            total_items_considered=len(candidates),
        )

    async def _graph_retrieval(self, query: str, candidates: list[KnowledgeItem], max_results: int) -> RetrievalResult:
        """Perform graph-based relationship traversal retrieval."""
        # Find initial seed items that match query
        query_lower = query.lower()
        seed_items = []

        for item in candidates:
            if any(word in item.content.lower() for word in query_lower.split()):
                seed_items.append(item)

        # Expand through semantic connections
        expanded_items = set(seed_items)
        for seed_item in seed_items[:3]:  # Limit expansion to top 3 seeds
            connected_ids = seed_item.semantic_connections or []
            for connected_id in connected_ids[:5]:  # Limit connections per seed
                if connected_id in self.knowledge_items:
                    expanded_items.add(self.knowledge_items[connected_id])

        # Score items by connection strength and belief probability
        scored_items = []
        for item in expanded_items:
            connection_score = 1.0 if item in seed_items else 0.7  # Direct matches score higher
            belief = self.belief_engine.beliefs.get(item.item_id)
            belief_score = belief.probability if belief else item.belief_probability

            combined_score = connection_score * belief_score
            scored_items.append((item, combined_score))

        # Sort by combined score and take top results
        scored_items.sort(key=lambda x: x[1], reverse=True)
        top_items = [item for item, score in scored_items[:max_results]]

        # Generate Bayesian scores
        bayesian_scores = {}
        for item in top_items:
            belief = self.belief_engine.beliefs.get(item.item_id)
            bayesian_scores[item.item_id] = belief.probability if belief else item.belief_probability

        return RetrievalResult(
            items=top_items,
            retrieval_method=RAGType.GRAPH,
            confidence_score=np.mean([score for _, score in scored_items[:max_results]]) if scored_items else 0.0,
            bayesian_scores=bayesian_scores,
            semantic_coherence=self._calculate_semantic_coherence(top_items),
            context_relevance=0.9,  # Graph retrieval has excellent context relevance
            total_items_considered=len(candidates),
        )

    async def _bayesian_retrieval(
        self, query: str, candidates: list[KnowledgeItem], max_results: int
    ) -> RetrievalResult:
        """Perform Bayesian belief-based retrieval with probability propagation."""
        # Update beliefs based on query relevance
        query_beliefs = []
        for item in candidates:
            if any(word in item.content.lower() for word in query.lower().split()):
                # Increase belief probability for query-relevant items
                belief = self.belief_engine.beliefs.get(item.item_id)
                if belief:
                    old_prob = belief.probability
                    new_prob = min(0.95, old_prob + 0.1)  # Boost by 0.1, cap at 0.95
                    self.belief_engine.update_belief_probability(item.item_id, new_prob)
                    query_beliefs.append((item, new_prob))

        # Propagate belief updates through semantic connections
        for item, updated_prob in query_beliefs:
            for connected_id in item.semantic_connections or []:
                if connected_id in self.belief_engine.beliefs:
                    connected_belief = self.belief_engine.beliefs[connected_id]
                    # Propagate 50% of the belief increase
                    propagated_increase = (updated_prob - item.belief_probability) * 0.5
                    new_connected_prob = min(0.95, connected_belief.probability + propagated_increase)
                    self.belief_engine.update_belief_probability(connected_id, new_connected_prob)

        # Rank items by updated belief probabilities
        belief_ranked = []
        for item in candidates:
            belief = self.belief_engine.beliefs.get(item.item_id)
            probability = belief.probability if belief else item.belief_probability
            belief_ranked.append((item, probability))

        # Sort by belief probability and take top results
        belief_ranked.sort(key=lambda x: x[1], reverse=True)
        top_items = [item for item, prob in belief_ranked[:max_results]]

        # Generate Bayesian scores
        bayesian_scores = {}
        for item in top_items:
            belief = self.belief_engine.beliefs.get(item.item_id)
            bayesian_scores[item.item_id] = belief.probability if belief else item.belief_probability

        return RetrievalResult(
            items=top_items,
            retrieval_method=RAGType.BAYESIAN,
            confidence_score=np.mean([prob for _, prob in belief_ranked[:max_results]]) if belief_ranked else 0.0,
            bayesian_scores=bayesian_scores,
            semantic_coherence=self._calculate_semantic_coherence(top_items),
            context_relevance=0.7,  # Bayesian retrieval may be less context-specific
            total_items_considered=len(candidates),
        )

    async def _hybrid_retrieval(self, query: str, candidates: list[KnowledgeItem], max_results: int) -> RetrievalResult:
        """Perform hybrid retrieval combining vector, graph, and Bayesian approaches."""
        # Get results from all methods
        vector_results = await self._vector_retrieval(query, candidates, max_results * 2)
        graph_results = await self._graph_retrieval(query, candidates, max_results * 2)
        bayesian_results = await self._bayesian_retrieval(query, candidates, max_results * 2)

        # Combine and score results
        item_scores = {}

        # Weight different retrieval methods
        weights = {RAGType.VECTOR: 0.4, RAGType.GRAPH: 0.4, RAGType.BAYESIAN: 0.2}

        # Score vector results
        for i, item in enumerate(vector_results.items):
            position_score = 1.0 - (i / len(vector_results.items))  # Higher score for earlier positions
            item_scores[item.item_id] = item_scores.get(item.item_id, 0) + weights[RAGType.VECTOR] * position_score

        # Score graph results
        for i, item in enumerate(graph_results.items):
            position_score = 1.0 - (i / len(graph_results.items))
            item_scores[item.item_id] = item_scores.get(item.item_id, 0) + weights[RAGType.GRAPH] * position_score

        # Score Bayesian results
        for i, item in enumerate(bayesian_results.items):
            position_score = 1.0 - (i / len(bayesian_results.items))
            item_scores[item.item_id] = item_scores.get(item.item_id, 0) + weights[RAGType.BAYESIAN] * position_score

        # Get top items by combined score
        all_items = {item.item_id: item for item in vector_results.items + graph_results.items + bayesian_results.items}
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = [all_items[item_id] for item_id, score in sorted_items[:max_results] if item_id in all_items]

        # Generate combined Bayesian scores
        bayesian_scores = {}
        for item in top_items:
            belief = self.belief_engine.beliefs.get(item.item_id)
            bayesian_scores[item.item_id] = belief.probability if belief else item.belief_probability

        # Calculate combined confidence
        combined_confidence = (
            vector_results.confidence_score * weights[RAGType.VECTOR]
            + graph_results.confidence_score * weights[RAGType.GRAPH]
            + bayesian_results.confidence_score * weights[RAGType.BAYESIAN]
        )

        return RetrievalResult(
            items=top_items,
            retrieval_method=RAGType.HYBRID,
            confidence_score=combined_confidence,
            bayesian_scores=bayesian_scores,
            semantic_coherence=self._calculate_semantic_coherence(top_items),
            context_relevance=(vector_results.context_relevance + graph_results.context_relevance) / 2,
            total_items_considered=len(candidates),
        )

    def _calculate_semantic_coherence(self, items: list[KnowledgeItem]) -> float:
        """Calculate semantic coherence score for a set of items."""
        if len(items) < 2:
            return 1.0

        # Calculate average pairwise similarity between items
        similarities = []
        for i, item1 in enumerate(items):
            for item2 in items[i + 1 :]:
                if item1.embedding is not None and item2.embedding is not None:
                    similarity = np.dot(item1.embedding, item2.embedding)
                    similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.5

    async def analyze_with_cognitive_nexus(self, retrieval_result: RetrievalResult, query: str) -> dict[str, Any]:
        """
        Analyze retrieval results using the Cognitive Nexus for multi-perspective reasoning.

        Args:
            retrieval_result: Results from knowledge retrieval
            query: Original query for context

        Returns:
            Dict with cognitive analysis including synthesis and insights
        """
        # Prepare content for cognitive analysis
        content_items = []
        for item in retrieval_result.items:
            content_items.append(
                {
                    "content": item.content,
                    "item_id": item.item_id,
                    "belief_probability": retrieval_result.bayesian_scores.get(item.item_id, 0.5),
                    "context_tags": [
                        {"book": tag.book_summary, "chapter": tag.chapter_summary} for tag in item.context_tags
                    ],
                }
            )

        # Use Cognitive Nexus for analysis
        analysis_result = await self.cognitive_nexus.analyze_information(
            content_items=content_items, query_context=query, dual_context_tags=True
        )

        return analysis_result

    def get_system_stats(self) -> dict[str, Any]:
        """Get comprehensive system statistics."""
        # Calculate belief distribution
        belief_probs = [belief.probability for belief in self.belief_engine.beliefs.values()]
        belief_stats = {
            "total_beliefs": len(belief_probs),
            "avg_probability": np.mean(belief_probs) if belief_probs else 0.0,
            "high_confidence_beliefs": sum(1 for p in belief_probs if p > 0.8),
            "low_confidence_beliefs": sum(1 for p in belief_probs if p < 0.3),
        }

        # Calculate context distribution
        context_stats = {
            "total_context_books": len(self.context_hierarchy),
            "total_context_chapters": sum(len(chapters) for chapters in self.context_hierarchy.values()),
            "avg_chapters_per_book": np.mean([len(chapters) for chapters in self.context_hierarchy.values()])
            if self.context_hierarchy
            else 0.0,
        }

        # Calculate semantic graph stats
        graph_stats = {
            "total_semantic_connections": sum(len(connections) for connections in self.semantic_graph.values()),
            "highly_connected_items": sum(1 for connections in self.semantic_graph.values() if len(connections) > 5),
            "avg_connections_per_item": np.mean([len(connections) for connections in self.semantic_graph.values()])
            if self.semantic_graph
            else 0.0,
        }

        return {
            "knowledge_items": len(self.knowledge_items),
            "cached_items": len(self.frequent_items_cache),
            "belief_statistics": belief_stats,
            "context_statistics": context_stats,
            "semantic_graph_statistics": graph_stats,
            "retrieval_statistics": self.retrieval_stats,
            "system_health": {
                "operational": True,
                "last_updated": datetime.now().isoformat(),
                "managed_by": self.sage_agent_id,
            },
        }

    def export_knowledge_graph(self) -> dict[str, Any]:
        """Export the complete knowledge graph for visualization or backup."""
        return {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "sage_agent": self.sage_agent_id,
                "total_items": len(self.knowledge_items),
                "total_beliefs": len(self.belief_engine.beliefs),
            },
            "knowledge_items": {
                item_id: {
                    "content": item.content,
                    "context_tags": [
                        {
                            "book": tag.book_summary,
                            "chapter": tag.chapter_summary,
                            "probability_weight": tag.probability_weight,
                        }
                        for tag in item.context_tags
                    ],
                    "belief_probability": item.belief_probability,
                    "source_confidence": item.source_confidence,
                    "access_count": item.access_count,
                    "semantic_connections": item.semantic_connections or [],
                }
                for item_id, item in self.knowledge_items.items()
            },
            "belief_network": {
                belief_id: {
                    "description": belief.description,
                    "probability": belief.probability,
                    "connected_beliefs": belief.connected_beliefs,
                }
                for belief_id, belief in self.belief_engine.beliefs.items()
            },
            "context_hierarchy": self.context_hierarchy,
            "semantic_graph": self.semantic_graph,
        }
