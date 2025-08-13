"""Bayesian Trust Graph for Semantic Chunk Knowledge Representation.

This module implements a comprehensive knowledge graph system where semantic chunks
are represented as nodes and conceptual relationships as edges, with Bayesian trust
propagation for enhanced retrieval quality.

Features:
- NetworkX-based knowledge graph structure
- Semantic relationship detection between chunks
- Bayesian trust propagation from source credibility
- Graph-based contextual retrieval with trust weighting
- Multiple relationship types (CONTINUES, ELABORATES, CONTRASTS, etc.)
"""

from dataclasses import asdict, dataclass
from enum import Enum
import logging
import time
from typing import Any

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    """Types of semantic relationships between chunks."""

    CONTINUES = "continues"  # Sequential continuation of ideas
    ELABORATES = "elaborates"  # Expansion or detailed explanation
    CONTRASTS = "contrasts"  # Opposing or alternative viewpoints
    REFERENCES = "references"  # Citation or reference relationship
    SUPPORTS = "supports"  # Evidence supporting another claim
    QUESTIONS = "questions"  # Raises questions about another chunk
    DEFINES = "defines"  # Provides definition for concepts
    EXEMPLIFIES = "exemplifies"  # Provides examples for concepts
    SUMMARIZES = "summarizes"  # Summarizes content from other chunks
    CONTEXTUALIZES = "contextualizes"  # Provides context for understanding


@dataclass
class SemanticChunkNode:
    """Represents a semantic chunk node in the knowledge graph."""

    chunk_id: str
    document_id: str
    text: str
    position: int
    start_idx: int
    end_idx: int

    # Contextual metadata
    chunk_type: str
    local_summary: str
    section_hierarchy: list[str]
    key_entities: list[dict[str, Any]]
    local_keywords: list[str]

    # Trust and quality metrics
    base_credibility: float  # From document source
    quality_score: float  # Intrinsic quality of the chunk
    coherence_score: float  # Internal coherence
    completeness_score: float  # Content completeness

    # Graph-specific attributes
    trust_score: float = 0.0  # Propagated trust score
    centrality_score: float = 0.0  # Graph centrality measure
    embedding: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.embedding is not None:
            data["embedding"] = self.embedding.tolist()
        return data


@dataclass
class SemanticRelationship:
    """Represents a semantic relationship edge in the knowledge graph."""

    source_chunk_id: str
    target_chunk_id: str
    relationship_type: RelationshipType
    confidence: float  # Confidence in the relationship
    weight: float  # Relationship strength

    # Context information
    semantic_similarity: float  # Embedding similarity
    contextual_proximity: float  # Document/section proximity
    discourse_markers: list[str]  # Linguistic indicators

    # Trust propagation attributes
    trust_transfer_rate: float  # How much trust transfers through this edge
    bidirectional: bool = False  # Whether trust flows both ways


class BayesianTrustGraph:
    """Bayesian trust propagation system for semantic chunk knowledge graph.

    Implements a knowledge graph where:
    - Nodes represent semantic chunks with contextual metadata
    - Edges represent semantic relationships with trust weights
    - Bayesian trust propagation enhances retrieval quality
    - Graph structure enables contextual traversal
    """

    def __init__(
        self,
        embedding_model: str = "paraphrase-MiniLM-L3-v2",
        similarity_threshold: float = 0.3,
        trust_decay_factor: float = 0.85,
        max_propagation_hops: int = 3,
        relationship_confidence_threshold: float = 0.6,
    ) -> None:
        """Initialize Bayesian trust graph."""
        self.embedding_model_name = embedding_model
        self.similarity_threshold = similarity_threshold
        self.trust_decay_factor = trust_decay_factor
        self.max_propagation_hops = max_propagation_hops
        self.relationship_confidence_threshold = relationship_confidence_threshold

        # Initialize graph and embedding model
        self.graph = nx.DiGraph()  # Directed graph for relationships
        self.embedding_model = SentenceTransformer(embedding_model)

        # Node and relationship storage
        self.chunk_nodes: dict[str, SemanticChunkNode] = {}
        self.relationships: dict[tuple[str, str], SemanticRelationship] = {}

        # Performance tracking
        self.stats = {
            "nodes_added": 0,
            "relationships_detected": 0,
            "trust_propagations": 0,
            "graph_retrievals": 0,
            "avg_trust_propagation_time": 0.0,
            "avg_retrieval_time": 0.0,
        }

        logger.info(f"Initialized Bayesian trust graph with model: {embedding_model}")

    def add_semantic_chunk(
        self,
        chunk_id: str,
        document_id: str,
        text: str,
        position: int,
        start_idx: int,
        end_idx: int,
        contextual_metadata: dict[str, Any],
        base_credibility: float = 0.7,
    ) -> SemanticChunkNode:
        """Add a semantic chunk node to the knowledge graph."""
        try:
            # Extract contextual information
            chunk_context = contextual_metadata.get("chunk_context", {})
            quality_metrics = contextual_metadata.get("quality_metrics", {})

            # Create semantic chunk node
            chunk_node = SemanticChunkNode(
                chunk_id=chunk_id,
                document_id=document_id,
                text=text,
                position=position,
                start_idx=start_idx,
                end_idx=end_idx,
                # Contextual metadata
                chunk_type=chunk_context.get("chunk_type", "body"),
                local_summary=chunk_context.get("local_summary", ""),
                section_hierarchy=chunk_context.get("section_hierarchy", []),
                key_entities=chunk_context.get("key_entities", []),
                local_keywords=chunk_context.get("local_keywords", []),
                # Trust and quality metrics
                base_credibility=base_credibility,
                quality_score=quality_metrics.get("overall_quality", 0.7),
                coherence_score=quality_metrics.get("coherence_score", 0.7),
                completeness_score=quality_metrics.get("completeness_score", 0.7),
                # Initialize trust score with base credibility
                trust_score=base_credibility,
            )

            # Generate embedding
            chunk_node.embedding = self.embedding_model.encode(text)

            # Add to graph and storage
            self.graph.add_node(chunk_id, **chunk_node.to_dict())
            self.chunk_nodes[chunk_id] = chunk_node

            self.stats["nodes_added"] += 1

            logger.debug(f"Added semantic chunk node: {chunk_id}")
            return chunk_node

        except Exception as e:
            logger.exception(f"Failed to add semantic chunk {chunk_id}: {e}")
            raise

    def detect_semantic_relationships(
        self, chunk_node: SemanticChunkNode, context_window: int = 5
    ) -> list[SemanticRelationship]:
        """Detect semantic relationships between chunks."""
        relationships = []

        try:
            # Get existing chunk nodes for comparison
            existing_chunks = list(self.chunk_nodes.values())

            for other_chunk in existing_chunks:
                if other_chunk.chunk_id == chunk_node.chunk_id:
                    continue

                # Calculate semantic similarity
                if (
                    chunk_node.embedding is not None
                    and other_chunk.embedding is not None
                ):
                    similarity = np.dot(chunk_node.embedding, other_chunk.embedding) / (
                        np.linalg.norm(chunk_node.embedding)
                        * np.linalg.norm(other_chunk.embedding)
                    )
                else:
                    continue

                # Skip low similarity relationships
                if similarity < self.similarity_threshold:
                    continue

                # Determine relationship type and properties
                (
                    relationship_type,
                    confidence,
                    bidirectional,
                ) = self._classify_relationship(chunk_node, other_chunk, similarity)

                if confidence < self.relationship_confidence_threshold:
                    continue

                # Calculate contextual proximity
                contextual_proximity = self._calculate_contextual_proximity(
                    chunk_node, other_chunk
                )

                # Calculate trust transfer rate
                trust_transfer_rate = self._calculate_trust_transfer_rate(
                    relationship_type, confidence, similarity
                )

                # Create relationship
                relationship = SemanticRelationship(
                    source_chunk_id=chunk_node.chunk_id,
                    target_chunk_id=other_chunk.chunk_id,
                    relationship_type=relationship_type,
                    confidence=confidence,
                    weight=similarity * confidence,
                    semantic_similarity=similarity,
                    contextual_proximity=contextual_proximity,
                    discourse_markers=self._extract_discourse_markers(
                        chunk_node.text, other_chunk.text
                    ),
                    trust_transfer_rate=trust_transfer_rate,
                    bidirectional=bidirectional,
                )

                relationships.append(relationship)

                # Add edge to graph
                self.graph.add_edge(
                    chunk_node.chunk_id, other_chunk.chunk_id, **asdict(relationship)
                )

                # Store relationship
                self.relationships[(chunk_node.chunk_id, other_chunk.chunk_id)] = (
                    relationship
                )

                # Add reverse relationship if bidirectional
                if bidirectional:
                    reverse_relationship = SemanticRelationship(
                        source_chunk_id=other_chunk.chunk_id,
                        target_chunk_id=chunk_node.chunk_id,
                        relationship_type=relationship_type,
                        confidence=confidence,
                        weight=similarity * confidence,
                        semantic_similarity=similarity,
                        contextual_proximity=contextual_proximity,
                        discourse_markers=relationship.discourse_markers,
                        trust_transfer_rate=trust_transfer_rate,
                        bidirectional=True,
                    )

                    self.graph.add_edge(
                        other_chunk.chunk_id,
                        chunk_node.chunk_id,
                        **asdict(reverse_relationship),
                    )

                    self.relationships[(other_chunk.chunk_id, chunk_node.chunk_id)] = (
                        reverse_relationship
                    )

            self.stats["relationships_detected"] += len(relationships)
            logger.debug(
                f"Detected {len(relationships)} relationships for chunk {chunk_node.chunk_id}"
            )

            return relationships

        except Exception as e:
            logger.exception(
                f"Failed to detect relationships for {chunk_node.chunk_id}: {e}"
            )
            return []

    def _classify_relationship(
        self, chunk1: SemanticChunkNode, chunk2: SemanticChunkNode, similarity: float
    ) -> tuple[RelationshipType, float, bool]:
        """Classify the type of relationship between two chunks."""
        # Check sequential continuation (same document, adjacent positions)
        if (
            chunk1.document_id == chunk2.document_id
            and abs(chunk1.position - chunk2.position) == 1
        ):
            return RelationshipType.CONTINUES, 0.9, True

        # Check for elaboration patterns (hierarchical sections)
        if (
            chunk1.document_id == chunk2.document_id
            and self._is_hierarchical_elaboration(chunk1, chunk2)
        ):
            return RelationshipType.ELABORATES, 0.8, False

        # Check for contrasting language
        if self._has_contrasting_markers(chunk1.text, chunk2.text):
            return RelationshipType.CONTRASTS, 0.7, True

        # Check for supporting evidence patterns
        if self._has_supporting_markers(chunk1.text, chunk2.text):
            return RelationshipType.SUPPORTS, 0.75, False

        # Check for definition patterns
        if self._has_definition_markers(chunk1.text, chunk2.text):
            return RelationshipType.DEFINES, 0.8, False

        # Check for example patterns
        if self._has_example_markers(chunk1.text, chunk2.text):
            return RelationshipType.EXEMPLIFIES, 0.75, False

        # Default to reference relationship for high similarity
        if similarity > 0.7:
            return RelationshipType.REFERENCES, similarity, True
        return RelationshipType.CONTEXTUALIZES, similarity * 0.8, False

    def _is_hierarchical_elaboration(
        self, chunk1: SemanticChunkNode, chunk2: SemanticChunkNode
    ) -> bool:
        """Check if chunks have hierarchical elaboration relationship."""
        h1 = chunk1.section_hierarchy
        h2 = chunk2.section_hierarchy

        if not h1 or not h2:
            return False

        # Check if one hierarchy is a subset of the other (elaboration)
        return (len(h2) > len(h1) and h2[: len(h1)] == h1) or (
            len(h1) > len(h2) and h1[: len(h2)] == h2
        )

    def _has_contrasting_markers(self, text1: str, text2: str) -> bool:
        """Check for linguistic markers of contrasting relationships."""
        contrast_markers = [
            "however",
            "but",
            "nevertheless",
            "on the other hand",
            "in contrast",
            "conversely",
            "alternatively",
            "whereas",
            "unlike",
            "different from",
            "rather than",
        ]

        combined_text = (text1 + " " + text2).lower()
        return any(marker in combined_text for marker in contrast_markers)

    def _has_supporting_markers(self, text1: str, text2: str) -> bool:
        """Check for linguistic markers of supporting relationships."""
        support_markers = [
            "furthermore",
            "moreover",
            "additionally",
            "in addition",
            "similarly",
            "likewise",
            "for example",
            "for instance",
            "evidence suggests",
            "studies show",
            "research indicates",
        ]

        combined_text = (text1 + " " + text2).lower()
        return any(marker in combined_text for marker in support_markers)

    def _has_definition_markers(self, text1: str, text2: str) -> bool:
        """Check for linguistic markers of definition relationships."""
        definition_markers = [
            "is defined as",
            "refers to",
            "means",
            "is the process of",
            "can be understood as",
            "is characterized by",
            "denotes",
            "is a type of",
            "is a form of",
        ]

        combined_text = (text1 + " " + text2).lower()
        return any(marker in combined_text for marker in definition_markers)

    def _has_example_markers(self, text1: str, text2: str) -> bool:
        """Check for linguistic markers of exemplification relationships."""
        example_markers = [
            "for example",
            "for instance",
            "such as",
            "including",
            "specifically",
            "namely",
            "in particular",
            "like",
            "consider the case",
            "take the example",
        ]

        combined_text = (text1 + " " + text2).lower()
        return any(marker in combined_text for marker in example_markers)

    def _calculate_contextual_proximity(
        self, chunk1: SemanticChunkNode, chunk2: SemanticChunkNode
    ) -> float:
        """Calculate contextual proximity between chunks."""
        proximity = 0.0

        # Same document bonus
        if chunk1.document_id == chunk2.document_id:
            proximity += 0.4

            # Position proximity within document
            pos_diff = abs(chunk1.position - chunk2.position)
            if pos_diff <= 1:
                proximity += 0.3
            elif pos_diff <= 3:
                proximity += 0.2
            elif pos_diff <= 5:
                proximity += 0.1

        # Section hierarchy similarity
        h1 = chunk1.section_hierarchy
        h2 = chunk2.section_hierarchy
        if h1 and h2:
            common_levels = 0
            for i in range(min(len(h1), len(h2))):
                if h1[i] == h2[i]:
                    common_levels += 1
                else:
                    break
            proximity += 0.3 * (common_levels / max(len(h1), len(h2)))

        return min(proximity, 1.0)

    def _calculate_trust_transfer_rate(
        self, relationship_type: RelationshipType, confidence: float, similarity: float
    ) -> float:
        """Calculate trust transfer rate for relationship."""
        # Base rates by relationship type
        base_rates = {
            RelationshipType.CONTINUES: 0.9,
            RelationshipType.ELABORATES: 0.8,
            RelationshipType.SUPPORTS: 0.85,
            RelationshipType.DEFINES: 0.7,
            RelationshipType.EXEMPLIFIES: 0.75,
            RelationshipType.REFERENCES: 0.6,
            RelationshipType.CONTEXTUALIZES: 0.5,
            RelationshipType.CONTRASTS: 0.3,
            RelationshipType.QUESTIONS: 0.2,
            RelationshipType.SUMMARIZES: 0.8,
        }

        base_rate = base_rates.get(relationship_type, 0.5)

        # Adjust by confidence and similarity
        adjusted_rate = base_rate * confidence * (0.7 + 0.3 * similarity)

        return min(adjusted_rate, 0.95)  # Cap at 95%

    def _extract_discourse_markers(self, text1: str, text2: str) -> list[str]:
        """Extract discourse markers from text pair."""
        markers = []
        combined_text = (text1 + " " + text2).lower()

        discourse_patterns = {
            "contrast": ["however", "but", "nevertheless", "in contrast"],
            "addition": ["furthermore", "moreover", "additionally", "also"],
            "example": ["for example", "for instance", "such as", "including"],
            "causation": ["because", "therefore", "thus", "consequently"],
            "sequence": ["first", "second", "then", "next", "finally"],
        }

        for category, pattern_list in discourse_patterns.items():
            for pattern in pattern_list:
                if pattern in combined_text:
                    markers.append(f"{category}:{pattern}")

        return markers

    def propagate_trust(
        self,
        source_chunk_ids: list[str] | None = None,
        max_iterations: int = 10,
        convergence_threshold: float = 0.001,
    ) -> dict[str, float]:
        """Propagate trust scores through the knowledge graph using Bayesian inference.

        Args:
            source_chunk_ids: Specific chunks to start propagation from
            max_iterations: Maximum propagation iterations
            convergence_threshold: Convergence threshold for iterative algorithm

        Returns:
            Dictionary mapping chunk IDs to final trust scores
        """
        start_time = time.perf_counter()

        try:
            # Initialize trust scores
            if source_chunk_ids is None:
                source_chunk_ids = list(self.chunk_nodes.keys())

            # Reset trust scores to base credibility
            for chunk_id, chunk_node in self.chunk_nodes.items():
                chunk_node.trust_score = chunk_node.base_credibility
                self.graph.nodes[chunk_id]["trust_score"] = chunk_node.trust_score

            # Iterative trust propagation
            for iteration in range(max_iterations):
                previous_scores = {
                    chunk_id: node.trust_score
                    for chunk_id, node in self.chunk_nodes.items()
                }

                # Update trust scores based on incoming relationships
                for chunk_id, chunk_node in self.chunk_nodes.items():
                    incoming_trust = 0.0
                    total_weight = 0.0

                    # Aggregate trust from predecessors
                    for pred_id in self.graph.predecessors(chunk_id):
                        if pred_id in self.chunk_nodes:
                            pred_node = self.chunk_nodes[pred_id]
                            edge_data = self.graph[pred_id][chunk_id]

                            trust_transfer_rate = edge_data.get(
                                "trust_transfer_rate", 0.5
                            )
                            weight = edge_data.get("weight", 0.5)

                            # Bayesian trust transfer
                            transferred_trust = (
                                pred_node.trust_score
                                * trust_transfer_rate
                                * self.trust_decay_factor
                            )

                            incoming_trust += transferred_trust * weight
                            total_weight += weight

                    # Update trust score with weighted combination
                    if total_weight > 0:
                        aggregated_trust = incoming_trust / total_weight

                        # Bayesian combination of base credibility and propagated trust
                        prior = chunk_node.base_credibility
                        evidence = aggregated_trust

                        # Weighted average based on confidence in evidence
                        confidence_in_evidence = min(total_weight, 1.0)

                        chunk_node.trust_score = (
                            prior * (1 - confidence_in_evidence)
                            + evidence * confidence_in_evidence
                        )

                    # Update graph node
                    self.graph.nodes[chunk_id]["trust_score"] = chunk_node.trust_score

                # Check convergence
                max_change = max(
                    abs(chunk_node.trust_score - previous_scores[chunk_id])
                    for chunk_id, chunk_node in self.chunk_nodes.items()
                )

                if max_change < convergence_threshold:
                    logger.debug(
                        f"Trust propagation converged after {iteration + 1} iterations"
                    )
                    break

            # Calculate centrality scores
            self._update_centrality_scores()

            # Update statistics
            propagation_time = (time.perf_counter() - start_time) * 1000
            self.stats["trust_propagations"] += 1
            self.stats["avg_trust_propagation_time"] = (
                self.stats["avg_trust_propagation_time"]
                * (self.stats["trust_propagations"] - 1)
                + propagation_time
            ) / self.stats["trust_propagations"]

            logger.info(f"Trust propagation completed in {propagation_time:.1f}ms")

            return {
                chunk_id: node.trust_score
                for chunk_id, node in self.chunk_nodes.items()
            }

        except Exception as e:
            logger.exception(f"Trust propagation failed: {e}")
            return {}

    def _update_centrality_scores(self) -> None:
        """Update centrality scores for graph nodes."""
        try:
            if len(self.graph.nodes) == 0:
                return

            # Calculate PageRank centrality
            pagerank_scores = nx.pagerank(
                self.graph, weight="weight", max_iter=100, tol=1e-06
            )

            # Update chunk nodes with centrality scores
            for chunk_id, centrality_score in pagerank_scores.items():
                if chunk_id in self.chunk_nodes:
                    self.chunk_nodes[chunk_id].centrality_score = centrality_score
                    self.graph.nodes[chunk_id]["centrality_score"] = centrality_score

            logger.debug("Updated centrality scores")

        except Exception as e:
            logger.warning(f"Failed to update centrality scores: {e}")

    def retrieve_with_graph_traversal(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        trust_weight: float = 0.3,
        centrality_weight: float = 0.2,
        similarity_weight: float = 0.5,
        min_trust_score: float = 0.4,
        traversal_depth: int = 2,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Retrieve chunks using graph traversal with trust-weighted scoring.

        Args:
            query_embedding: Query vector embedding
            k: Number of results to return
            trust_weight: Weight for trust score in final ranking
            centrality_weight: Weight for centrality score
            similarity_weight: Weight for semantic similarity
            min_trust_score: Minimum trust score threshold
            traversal_depth: Maximum traversal depth from initial matches

        Returns:
            List of (chunk_id, score, metadata) tuples
        """
        start_time = time.perf_counter()

        try:
            # Calculate similarity scores for all chunks
            chunk_scores = []

            for chunk_id, chunk_node in self.chunk_nodes.items():
                if chunk_node.embedding is None:
                    continue

                # Skip low-trust chunks
                if chunk_node.trust_score < min_trust_score:
                    continue

                # Calculate semantic similarity
                similarity = np.dot(query_embedding, chunk_node.embedding) / (
                    np.linalg.norm(query_embedding)
                    * np.linalg.norm(chunk_node.embedding)
                )

                # Calculate composite score
                composite_score = (
                    similarity_weight * similarity
                    + trust_weight * chunk_node.trust_score
                    + centrality_weight * chunk_node.centrality_score
                )

                chunk_scores.append(
                    (
                        chunk_id,
                        composite_score,
                        {
                            "semantic_similarity": similarity,
                            "trust_score": chunk_node.trust_score,
                            "centrality_score": chunk_node.centrality_score,
                            "base_credibility": chunk_node.base_credibility,
                            "quality_score": chunk_node.quality_score,
                            "chunk_type": chunk_node.chunk_type,
                            "document_id": chunk_node.document_id,
                            "position": chunk_node.position,
                            "text": chunk_node.text,
                        },
                    )
                )

            # Sort by composite score
            chunk_scores.sort(key=lambda x: x[1], reverse=True)

            # Get initial top-k results
            initial_results = chunk_scores[: k * 2]  # Get more for graph expansion

            # Expand results through graph traversal
            expanded_results = self._expand_through_graph_traversal(
                initial_results, traversal_depth, query_embedding
            )

            # Re-rank and limit to k results
            final_results = sorted(expanded_results, key=lambda x: x[1], reverse=True)[
                :k
            ]

            # Update statistics
            retrieval_time = (time.perf_counter() - start_time) * 1000
            self.stats["graph_retrievals"] += 1
            self.stats["avg_retrieval_time"] = (
                self.stats["avg_retrieval_time"] * (self.stats["graph_retrievals"] - 1)
                + retrieval_time
            ) / self.stats["graph_retrievals"]

            logger.debug(f"Graph retrieval completed in {retrieval_time:.1f}ms")

            return final_results

        except Exception as e:
            logger.exception(f"Graph retrieval failed: {e}")
            return []

    def _expand_through_graph_traversal(
        self,
        initial_results: list[tuple[str, float, dict[str, Any]]],
        traversal_depth: int,
        query_embedding: np.ndarray,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Expand results through graph traversal."""
        expanded_chunks = set()
        result_map = {}

        # Add initial results
        for chunk_id, score, metadata in initial_results:
            expanded_chunks.add(chunk_id)
            result_map[chunk_id] = (score, metadata)

        # Traverse graph from initial results
        for chunk_id, initial_score, _ in initial_results[:5]:  # Limit expansion seeds
            self._traverse_from_chunk(
                chunk_id,
                traversal_depth,
                expanded_chunks,
                result_map,
                query_embedding,
                initial_score,
            )

        # Convert back to list format
        return [
            (chunk_id, score, metadata)
            for chunk_id, (score, metadata) in result_map.items()
        ]

    def _traverse_from_chunk(
        self,
        chunk_id: str,
        remaining_depth: int,
        expanded_chunks: set[str],
        result_map: dict[str, tuple[float, dict[str, Any]]],
        query_embedding: np.ndarray,
        parent_score: float,
    ) -> None:
        """Recursively traverse graph from a chunk."""
        if remaining_depth <= 0:
            return

        # Get neighbors
        for neighbor_id in self.graph.neighbors(chunk_id):
            if neighbor_id in expanded_chunks or neighbor_id not in self.chunk_nodes:
                continue

            neighbor_node = self.chunk_nodes[neighbor_id]
            edge_data = self.graph[chunk_id][neighbor_id]

            # Calculate traversal score
            if neighbor_node.embedding is not None:
                similarity = np.dot(query_embedding, neighbor_node.embedding) / (
                    np.linalg.norm(query_embedding)
                    * np.linalg.norm(neighbor_node.embedding)
                )

                # Combine parent score with relationship strength
                relationship_strength = edge_data.get("weight", 0.5)
                traversal_bonus = parent_score * relationship_strength * 0.3

                neighbor_score = (
                    0.5 * similarity
                    + 0.3 * neighbor_node.trust_score
                    + 0.2 * neighbor_node.centrality_score
                    + traversal_bonus
                )

                # Add to results if score is reasonable
                if neighbor_score > 0.3:
                    expanded_chunks.add(neighbor_id)
                    result_map[neighbor_id] = (
                        neighbor_score,
                        {
                            "semantic_similarity": similarity,
                            "trust_score": neighbor_node.trust_score,
                            "centrality_score": neighbor_node.centrality_score,
                            "base_credibility": neighbor_node.base_credibility,
                            "quality_score": neighbor_node.quality_score,
                            "chunk_type": neighbor_node.chunk_type,
                            "document_id": neighbor_node.document_id,
                            "position": neighbor_node.position,
                            "text": neighbor_node.text,
                            "traversal_depth": 3 - remaining_depth,
                            "parent_chunk": chunk_id,
                            "relationship_type": edge_data.get(
                                "relationship_type", "unknown"
                            ),
                        },
                    )

                    # Continue traversal
                    self._traverse_from_chunk(
                        neighbor_id,
                        remaining_depth - 1,
                        expanded_chunks,
                        result_map,
                        query_embedding,
                        neighbor_score,
                    )

    def get_graph_statistics(self) -> dict[str, Any]:
        """Get comprehensive graph statistics."""
        try:
            # Basic graph metrics
            num_nodes = len(self.graph.nodes)
            num_edges = len(self.graph.edges)

            # Trust score distribution
            trust_scores = [node.trust_score for node in self.chunk_nodes.values()]
            avg_trust = np.mean(trust_scores) if trust_scores else 0.0

            # Centrality distribution
            centrality_scores = [
                node.centrality_score for node in self.chunk_nodes.values()
            ]
            avg_centrality = np.mean(centrality_scores) if centrality_scores else 0.0

            # Relationship type distribution
            relationship_types = {}
            for relationship in self.relationships.values():
                rel_type = relationship.relationship_type.value
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1

            # Graph connectivity
            if num_nodes > 0:
                density = nx.density(self.graph)
                is_connected = nx.is_weakly_connected(self.graph)
                num_components = nx.number_weakly_connected_components(self.graph)
            else:
                density = 0.0
                is_connected = False
                num_components = 0

            return {
                "graph_structure": {
                    "nodes": num_nodes,
                    "edges": num_edges,
                    "density": density,
                    "is_connected": is_connected,
                    "connected_components": num_components,
                },
                "trust_metrics": {
                    "avg_trust_score": avg_trust,
                    "trust_score_std": np.std(trust_scores) if trust_scores else 0.0,
                    "min_trust_score": min(trust_scores) if trust_scores else 0.0,
                    "max_trust_score": max(trust_scores) if trust_scores else 0.0,
                },
                "centrality_metrics": {
                    "avg_centrality": avg_centrality,
                    "centrality_std": (
                        np.std(centrality_scores) if centrality_scores else 0.0
                    ),
                },
                "relationship_distribution": relationship_types,
                "performance_stats": self.stats.copy(),
                "quality_metrics": {
                    "avg_quality_score": (
                        np.mean(
                            [node.quality_score for node in self.chunk_nodes.values()]
                        )
                        if self.chunk_nodes
                        else 0.0
                    ),
                    "avg_coherence_score": (
                        np.mean(
                            [node.coherence_score for node in self.chunk_nodes.values()]
                        )
                        if self.chunk_nodes
                        else 0.0
                    ),
                },
            }

        except Exception as e:
            logger.exception(f"Failed to generate graph statistics: {e}")
            return {"error": str(e)}

    def export_graph_data(self, include_embeddings: bool = False) -> dict[str, Any]:
        """Export graph data for analysis or storage."""
        # Export nodes
        nodes_data = []
        for chunk_node in self.chunk_nodes.values():
            node_data = chunk_node.to_dict()
            if not include_embeddings and "embedding" in node_data:
                del node_data["embedding"]
            nodes_data.append(node_data)

        # Export edges
        edges_data = []
        for relationship in self.relationships.values():
            edges_data.append(asdict(relationship))

        return {
            "nodes": nodes_data,
            "edges": edges_data,
            "statistics": self.get_graph_statistics(),
            "config": {
                "embedding_model": self.embedding_model_name,
                "similarity_threshold": self.similarity_threshold,
                "trust_decay_factor": self.trust_decay_factor,
                "max_propagation_hops": self.max_propagation_hops,
            },
        }
