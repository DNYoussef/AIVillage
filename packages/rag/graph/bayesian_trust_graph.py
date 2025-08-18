"""
BayesianTrustGraph - Knowledge Graph with Probabilistic Reasoning

Advanced knowledge graph system with Bayesian trust propagation,
probabilistic reasoning, and semantic relationship modeling.
Designed for long-term semantic memory storage with trust-based retrieval.

This module provides the graph component of the unified HyperRAG system.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """Types of relationships in the knowledge graph."""

    SEMANTIC = "semantic"  # Conceptual similarity
    CAUSAL = "causal"  # Cause-effect relationship
    TEMPORAL = "temporal"  # Time-based relationship
    HIERARCHICAL = "hierarchical"  # Parent-child relationship
    ASSOCIATIVE = "associative"  # General association
    CONTRADICTION = "contradiction"  # Conflicting information
    EVIDENCE = "evidence"  # Supporting evidence
    INFERENCE = "inference"  # Inferred relationship


class TrustLevel(Enum):
    """Trust levels for information in the graph."""

    VERY_HIGH = 0.9
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    VERY_LOW = 0.2
    UNKNOWN = 0.5


@dataclass
class GraphNode:
    """Node in the Bayesian trust graph."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    concept: str = ""  # Main concept this node represents

    # Trust and probability
    trust_score: float = 0.5
    confidence: float = 0.5
    belief_strength: float = 0.5  # How strongly we believe this information

    # Evidence tracking
    evidence_count: int = 0
    source_reliability: float = 0.5
    verification_status: str = "unverified"  # verified, disputed, unverified

    # Temporal aspects
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    temporal_decay: float = 0.95  # How trust decays over time

    # Connectivity
    incoming_edges: set[str] = field(default_factory=set)
    outgoing_edges: set[str] = field(default_factory=set)

    # Vector representation
    embedding: np.ndarray | None = None

    # Metadata
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def update_trust_from_evidence(self, new_evidence_score: float, evidence_weight: float = 1.0):
        """Update trust score based on new evidence using Bayesian updating."""
        # Bayesian update: P(H|E) = P(E|H) * P(H) / P(E)
        prior = self.trust_score
        likelihood = new_evidence_score

        # Simple Bayesian update formula
        posterior = (likelihood * evidence_weight * prior) / (
            likelihood * evidence_weight * prior
            + (1 - likelihood) * evidence_weight * (1 - prior)
            + (1 - evidence_weight)
        )

        self.trust_score = min(1.0, max(0.0, posterior))
        self.evidence_count += 1
        self.last_updated = datetime.now()

    def apply_temporal_decay(self):
        """Apply time-based decay to trust score."""
        time_since_update = (datetime.now() - self.last_updated).total_seconds() / 3600  # hours
        decay_factor = self.temporal_decay ** (time_since_update / 24)  # Daily decay
        self.trust_score *= decay_factor


@dataclass
class GraphEdge:
    """Edge representing a relationship between nodes."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""

    # Relationship properties
    relation_type: RelationType = RelationType.ASSOCIATIVE
    relation_strength: float = 0.5  # How strong is this relationship
    trust_score: float = 0.5  # How much we trust this relationship

    # Evidence for this relationship
    evidence_count: int = 0
    supporting_docs: list[str] = field(default_factory=list)

    # Probabilistic properties
    conditional_probability: float = 0.5  # P(target|source)
    mutual_information: float = 0.0  # Information shared between nodes

    # Temporal aspects
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def calculate_trust_propagation(self, source_trust: float) -> float:
        """Calculate how much trust propagates through this edge."""
        # Trust propagation with relationship strength consideration
        propagated_trust = source_trust * self.trust_score * self.relation_strength

        # Apply conditional probability weighting
        propagated_trust *= self.conditional_probability

        return min(1.0, max(0.0, propagated_trust))


@dataclass
class BayesianQueryResult:
    """Result from Bayesian graph query with probabilistic reasoning."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    # Trust propagation results
    trust_scores: dict[str, float] = field(default_factory=dict)
    belief_network: dict[str, dict[str, float]] = field(default_factory=dict)

    # Query metadata
    query_confidence: float = 0.0
    reasoning_path: list[str] = field(default_factory=list)
    query_time_ms: float = 0.0

    # Probabilistic analysis
    uncertainty_measures: dict[str, float] = field(default_factory=dict)
    conflicting_information: list[dict[str, Any]] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)


class BayesianTrustGraph:
    """
    Knowledge Graph with Bayesian Trust Propagation

    Advanced graph-based knowledge system with probabilistic reasoning,
    trust propagation, and semantic relationship modeling. Designed for
    long-term semantic memory with evidence-based belief updating.

    Features:
    - Bayesian trust propagation across relationships
    - Probabilistic reasoning and belief networks
    - Evidence-based trust updating
    - Temporal decay and information aging
    - Conflict detection and resolution
    - Multi-hop semantic traversal
    """

    def __init__(
        self,
        similarity_threshold: float = 0.4,
        trust_decay_factor: float = 0.85,
        max_propagation_hops: int = 3,
        belief_update_rate: float = 0.1,
    ):
        self.similarity_threshold = similarity_threshold
        self.trust_decay_factor = trust_decay_factor
        self.max_propagation_hops = max_propagation_hops
        self.belief_update_rate = belief_update_rate

        # Graph storage
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, GraphEdge] = {}

        # Indexing for efficient queries
        self.concept_index: dict[str, set[str]] = {}  # concept -> node_ids
        self.relationship_index: dict[RelationType, set[str]] = {}  # relation_type -> edge_ids
        self.trust_index: dict[float, set[str]] = {}  # trust_level -> node_ids

        # Caching
        self.query_cache: dict[str, BayesianQueryResult] = {}
        self.propagation_cache: dict[str, dict[str, float]] = {}

        # Statistics
        self.stats = {
            "nodes_added": 0,
            "edges_added": 0,
            "trust_updates": 0,
            "queries_processed": 0,
            "cache_hits": 0,
            "propagation_calculations": 0,
        }

        self.initialized = False

    async def initialize(self):
        """Initialize the Bayesian trust graph system."""
        logger.info("Initializing BayesianTrustGraph...")

        # Initialize relationship type indexes
        for relation_type in RelationType:
            self.relationship_index[relation_type] = set()

        # Set up periodic maintenance tasks
        asyncio.create_task(self._periodic_maintenance())

        self.initialized = True
        logger.info("🕸️ BayesianTrustGraph knowledge system ready")

    async def add_node(
        self,
        content: str,
        concept: str,
        trust_score: float = 0.5,
        evidence_count: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> GraphNode:
        """Add a new node to the knowledge graph."""
        node = GraphNode(
            content=content,
            concept=concept,
            trust_score=trust_score,
            confidence=trust_score,  # Initialize confidence with trust
            evidence_count=evidence_count,
            metadata=metadata or {},
        )

        # Store node
        self.nodes[node.id] = node

        # Update concept index
        if concept not in self.concept_index:
            self.concept_index[concept] = set()
        self.concept_index[concept].add(node.id)

        # Update trust index
        trust_bucket = round(trust_score, 1)
        if trust_bucket not in self.trust_index:
            self.trust_index[trust_bucket] = set()
        self.trust_index[trust_bucket].add(node.id)

        self.stats["nodes_added"] += 1
        logger.debug(f"Added node {node.id} with concept '{concept}'")

        return node

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        relation_strength: float = 0.5,
        trust_score: float = 0.5,
        conditional_probability: float = 0.5,
        supporting_docs: list[str] | None = None,
    ) -> GraphEdge:
        """Add a relationship edge between nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Source and target nodes must exist before adding edge")

        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            relation_strength=relation_strength,
            trust_score=trust_score,
            conditional_probability=conditional_probability,
            supporting_docs=supporting_docs or [],
        )

        # Store edge
        self.edges[edge.id] = edge

        # Update node connectivity
        self.nodes[source_id].outgoing_edges.add(edge.id)
        self.nodes[target_id].incoming_edges.add(edge.id)

        # Update relationship index
        self.relationship_index[relation_type].add(edge.id)

        # Clear propagation cache (relationships changed)
        self.propagation_cache.clear()

        self.stats["edges_added"] += 1
        logger.debug(f"Added edge {edge.id} ({relation_type.value}): {source_id} -> {target_id}")

        return edge

    async def add_document_with_relationships(
        self, content: str, doc_id: str, metadata: dict[str, Any] | None = None
    ) -> bool:
        """Add document and automatically extract relationships."""
        try:
            # Extract main concepts from content (simplified)
            concepts = await self._extract_concepts(content)

            # Create nodes for each concept
            created_nodes = []
            for concept in concepts:
                # Check if concept already exists
                existing_nodes = self.concept_index.get(concept, set())

                if existing_nodes:
                    # Update existing node with new evidence
                    node_id = next(iter(existing_nodes))
                    node = self.nodes[node_id]
                    node.update_trust_from_evidence(0.7, evidence_weight=0.5)
                    created_nodes.append(node)
                else:
                    # Create new node
                    node = await self.add_node(
                        content=f"Concept: {concept}",
                        concept=concept,
                        trust_score=0.6,
                        metadata={**(metadata or {}), "source_doc": doc_id},
                    )
                    created_nodes.append(node)

            # Create relationships between concepts (simplified)
            await self._create_concept_relationships(created_nodes, doc_id)

            return True

        except Exception as e:
            logger.exception(f"Failed to add document {doc_id}: {e}")
            return False

    async def retrieve_with_trust_propagation(
        self, query: str, k: int = 10, min_trust_score: float = 0.3
    ) -> list[GraphNode]:
        """Retrieve nodes using trust propagation and semantic matching."""
        start_time = time.time()

        try:
            # Check cache
            cache_key = f"retrieve:{hash(query)}:{k}:{min_trust_score}"
            if cache_key in self.query_cache:
                self.stats["cache_hits"] += 1
                return self.query_cache[cache_key].nodes

            # Find initial matching nodes
            initial_nodes = await self._find_matching_nodes(query, min_trust_score)

            if not initial_nodes:
                return []

            # Perform trust propagation
            propagated_scores = await self._propagate_trust(initial_nodes)

            # Rank nodes by propagated trust scores
            all_candidates = []
            for node_id, trust_score in propagated_scores.items():
                if trust_score >= min_trust_score:
                    node = self.nodes[node_id]
                    # Boost score for query relevance
                    relevance_boost = await self._calculate_query_relevance(node, query)
                    final_score = trust_score * 0.7 + relevance_boost * 0.3
                    all_candidates.append((node, final_score))

            # Sort by score and return top k
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            result_nodes = [node for node, score in all_candidates[:k]]

            # Cache result
            query_time = (time.time() - start_time) * 1000
            result = BayesianQueryResult(
                nodes=result_nodes,
                trust_scores=propagated_scores,
                query_time_ms=query_time,
                query_confidence=np.mean([score for _, score in all_candidates[:k]]) if all_candidates else 0.0,
            )
            self.query_cache[cache_key] = result

            self.stats["queries_processed"] += 1
            return result_nodes

        except Exception as e:
            logger.exception(f"Retrieval failed: {e}")
            return []

    async def detect_conflicts(self, concept: str) -> list[dict[str, Any]]:
        """Detect conflicting information about a concept."""
        conflicts = []

        try:
            # Get all nodes for this concept
            concept_node_ids = self.concept_index.get(concept, set())

            if len(concept_node_ids) < 2:
                return conflicts  # Need at least 2 nodes to have conflicts

            concept_nodes = [self.nodes[nid] for nid in concept_node_ids]

            # Look for contradiction relationships
            for edge_id in self.relationship_index.get(RelationType.CONTRADICTION, set()):
                edge = self.edges[edge_id]

                if edge.source_id in concept_node_ids and edge.target_id in concept_node_ids:
                    source_node = self.nodes[edge.source_id]
                    target_node = self.nodes[edge.target_id]

                    conflict_info = {
                        "type": "explicit_contradiction",
                        "source_node": source_node,
                        "target_node": target_node,
                        "edge": edge,
                        "confidence": edge.trust_score,
                    }
                    conflicts.append(conflict_info)

            # Look for trust score disagreements
            for i, node1 in enumerate(concept_nodes):
                for _j, node2 in enumerate(concept_nodes[i + 1 :], i + 1):
                    trust_diff = abs(node1.trust_score - node2.trust_score)

                    if trust_diff > 0.4:  # Significant disagreement
                        conflict_info = {
                            "type": "trust_disagreement",
                            "source_node": node1,
                            "target_node": node2,
                            "trust_difference": trust_diff,
                            "confidence": 1.0 - trust_diff,  # Lower confidence for higher disagreement
                        }
                        conflicts.append(conflict_info)

            return conflicts

        except Exception as e:
            logger.exception(f"Conflict detection failed: {e}")
            return []

    async def get_health_status(self) -> dict[str, Any]:
        """Get health status of the graph system."""
        try:
            # Calculate graph statistics
            total_nodes = len(self.nodes)
            total_edges = len(self.edges)

            # Trust distribution
            trust_scores = [node.trust_score for node in self.nodes.values()]
            avg_trust = np.mean(trust_scores) if trust_scores else 0.0
            trust_std = np.std(trust_scores) if trust_scores else 0.0

            # Connectivity analysis
            isolated_nodes = sum(
                1 for node in self.nodes.values() if len(node.incoming_edges) == 0 and len(node.outgoing_edges) == 0
            )

            # Cache performance
            cache_hit_rate = self.stats["cache_hits"] / max(1, self.stats["queries_processed"])

            return {
                "status": "healthy",
                "graph_size": {
                    "nodes": total_nodes,
                    "edges": total_edges,
                    "concepts": len(self.concept_index),
                    "isolated_nodes": isolated_nodes,
                },
                "trust_metrics": {
                    "average_trust": avg_trust,
                    "trust_std_dev": trust_std,
                    "high_trust_nodes": sum(1 for score in trust_scores if score > 0.8),
                    "low_trust_nodes": sum(1 for score in trust_scores if score < 0.3),
                },
                "performance": {"cache_hit_rate": cache_hit_rate, "avg_query_time": "N/A"},  # Would need to track this
                "statistics": self.stats.copy(),
            }

        except Exception as e:
            logger.exception(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}

    async def shutdown(self):
        """Shutdown the graph system and clean up resources."""
        logger.info("Shutting down BayesianTrustGraph...")

        # Clear all data structures
        self.nodes.clear()
        self.edges.clear()
        self.concept_index.clear()
        self.relationship_index.clear()
        self.trust_index.clear()
        self.query_cache.clear()
        self.propagation_cache.clear()

        logger.info("BayesianTrustGraph shutdown complete")

    # Private implementation methods

    async def _extract_concepts(self, content: str) -> list[str]:
        """Extract key concepts from content (simplified implementation)."""
        # This is a simplified concept extraction
        # In a real implementation, you'd use NLP libraries like spaCy or NLTK

        # Simple keyword extraction based on content analysis
        words = content.lower().split()

        # Filter out common words and focus on potential concepts
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
        }

        concepts = []
        for word in words:
            if len(word) > 3 and word not in stop_words and word.isalpha():
                concepts.append(word)

        # Return unique concepts, limited to avoid too many
        return list(set(concepts))[:5]

    async def _create_concept_relationships(self, nodes: list[GraphNode], doc_id: str):
        """Create relationships between concepts from the same document."""
        # Create associative relationships between concepts from same document
        for i, node1 in enumerate(nodes):
            for _j, node2 in enumerate(nodes[i + 1 :], i + 1):
                # Create bidirectional associative relationships
                await self.add_edge(
                    source_id=node1.id,
                    target_id=node2.id,
                    relation_type=RelationType.ASSOCIATIVE,
                    relation_strength=0.6,
                    trust_score=0.7,
                    conditional_probability=0.5,
                    supporting_docs=[doc_id],
                )

    async def _find_matching_nodes(self, query: str, min_trust_score: float) -> list[GraphNode]:
        """Find nodes that match the query."""
        matching_nodes = []

        query_lower = query.lower()

        # Simple text matching (would be replaced with semantic similarity in production)
        for node in self.nodes.values():
            if node.trust_score >= min_trust_score:
                # Check content and concept matching
                if (
                    query_lower in node.content.lower()
                    or query_lower in node.concept.lower()
                    or any(query_lower in tag.lower() for tag in node.tags)
                ):
                    matching_nodes.append(node)

        return matching_nodes[:20]  # Limit initial matches

    async def _propagate_trust(self, initial_nodes: list[GraphNode]) -> dict[str, float]:
        """Propagate trust scores through the graph."""
        # Check cache
        cache_key = ":".join(sorted([node.id for node in initial_nodes]))
        if cache_key in self.propagation_cache:
            return self.propagation_cache[cache_key]

        # Initialize trust scores
        trust_scores = {}
        for node in initial_nodes:
            trust_scores[node.id] = node.trust_score

        # Propagate trust through multiple hops
        for hop in range(self.max_propagation_hops):
            new_scores = trust_scores.copy()

            for node_id in list(trust_scores.keys()):
                node = self.nodes[node_id]
                current_trust = trust_scores[node_id]

                # Propagate through outgoing edges
                for edge_id in node.outgoing_edges:
                    edge = self.edges[edge_id]
                    self.nodes[edge.target_id]

                    # Calculate propagated trust
                    propagated = edge.calculate_trust_propagation(current_trust)
                    propagated *= self.trust_decay_factor**hop  # Decay with distance

                    # Update target node trust (take maximum)
                    if edge.target_id not in new_scores:
                        new_scores[edge.target_id] = propagated
                    else:
                        new_scores[edge.target_id] = max(new_scores[edge.target_id], propagated)

            trust_scores = new_scores
            self.stats["propagation_calculations"] += len(new_scores)

        # Cache result
        self.propagation_cache[cache_key] = trust_scores
        return trust_scores

    async def _calculate_query_relevance(self, node: GraphNode, query: str) -> float:
        """Calculate how relevant a node is to the query."""
        query_lower = query.lower()
        content_lower = node.content.lower()
        concept_lower = node.concept.lower()

        # Simple relevance scoring
        relevance = 0.0

        # Exact concept match
        if query_lower == concept_lower:
            relevance += 0.5

        # Concept containment
        elif query_lower in concept_lower or concept_lower in query_lower:
            relevance += 0.3

        # Content matching
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())

        if query_words and content_words:
            overlap = len(query_words.intersection(content_words))
            relevance += 0.2 * (overlap / len(query_words))

        # Tag matching
        for tag in node.tags:
            if query_lower in tag.lower():
                relevance += 0.1
                break

        return min(1.0, relevance)

    async def _periodic_maintenance(self):
        """Periodic maintenance tasks."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                # Apply temporal decay to all nodes
                for node in self.nodes.values():
                    node.apply_temporal_decay()

                # Clear old cache entries
                if len(self.query_cache) > 1000:
                    # Keep only recent 500 entries
                    cache_items = list(self.query_cache.items())
                    self.query_cache = dict(cache_items[-500:])

                if len(self.propagation_cache) > 500:
                    # Clear propagation cache periodically
                    self.propagation_cache.clear()

                logger.debug("Performed periodic maintenance on BayesianTrustGraph")

            except Exception as e:
                logger.exception(f"Periodic maintenance failed: {e}")
                await asyncio.sleep(3600)


if __name__ == "__main__":

    async def test_bayesian_trust_graph():
        """Test BayesianTrustGraph functionality."""
        # Create system
        graph = BayesianTrustGraph(similarity_threshold=0.4, trust_decay_factor=0.85, max_propagation_hops=3)
        await graph.initialize()

        # Add some test nodes
        node1 = await graph.add_node(
            content="Machine learning is a subset of artificial intelligence",
            concept="machine_learning",
            trust_score=0.8,
        )

        node2 = await graph.add_node(
            content="Deep learning uses neural networks with multiple layers", concept="deep_learning", trust_score=0.9
        )

        node3 = await graph.add_node(
            content="Neural networks are inspired by biological neurons", concept="neural_networks", trust_score=0.7
        )

        # Add relationships
        await graph.add_edge(
            source_id=node2.id,
            target_id=node1.id,
            relation_type=RelationType.HIERARCHICAL,
            relation_strength=0.8,
            trust_score=0.9,
            conditional_probability=0.8,
        )

        await graph.add_edge(
            source_id=node2.id,
            target_id=node3.id,
            relation_type=RelationType.SEMANTIC,
            relation_strength=0.7,
            trust_score=0.8,
            conditional_probability=0.6,
        )

        # Test document addition
        success = await graph.add_document_with_relationships(
            content="Artificial intelligence and machine learning are revolutionizing technology", doc_id="test_doc_1"
        )
        print(f"Document added: {success}")

        # Test retrieval with trust propagation
        results = await graph.retrieve_with_trust_propagation(query="machine learning", k=5, min_trust_score=0.3)
        print(f"Retrieved {len(results)} nodes for 'machine learning'")

        # Test conflict detection
        conflicts = await graph.detect_conflicts("machine_learning")
        print(f"Found {len(conflicts)} conflicts")

        # Health check
        health = await graph.get_health_status()
        print(f"Graph health: {health['status']}")
        print(f"Graph size: {health['graph_size']}")
        print(f"Trust metrics: {health['trust_metrics']}")

        await graph.shutdown()

    import asyncio

    asyncio.run(test_bayesian_trust_graph())


# Additional classes and types for compatibility


class RelationshipType(Enum):
    """Types of relationships (alias for RelationType)."""

    SEMANTIC = "semantic"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    HIERARCHICAL = "hierarchical"
    ASSOCIATIVE = "associative"
    CONTRADICTION = "contradiction"
    EVIDENCE = "evidence"
    INFERENCE = "inference"
    RELATES_TO = "relates_to"
    ENABLES = "enables"
    DEPENDS_ON = "depends_on"
    CONTAINS = "contains"


@dataclass
class Relationship:
    """Relationship between graph nodes."""

    subject_id: str
    predicate: RelationshipType
    object_id: str
    confidence: float = 0.5
    trust_score: float = 0.5
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def create_graph_node(
    content: str,
    node_id: str | None = None,
    concept: str | None = None,
    concepts: list[str] | None = None,
    trust_score: float = 0.5,
    confidence: float = 0.5,
    metadata: dict[str, Any] | None = None,
) -> GraphNode:
    """Create a new graph node with the given parameters."""

    node = GraphNode(
        id=node_id or str(uuid.uuid4()),
        content=content,
        concept=concept or (concepts[0] if concepts else ""),
        trust_score=trust_score,
        confidence=confidence,
        metadata=metadata or {},
    )

    if concepts:
        node.tags.extend(concepts)

    return node
