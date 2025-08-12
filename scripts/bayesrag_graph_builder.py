#!/usr/bin/env python3
"""Graph RAG builder with Bayesian trust networks for Wikipedia knowledge.
Creates semantic relationship graphs with trust-weighted connections.
"""

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SemanticRelationship:
    """Semantic relationship between knowledge chunks."""

    source_chunk: str
    target_chunk: str
    relationship_type: str
    semantic_similarity: float
    contextual_relevance: float
    temporal_alignment: float
    geographic_alignment: float
    trust_weight: float
    evidence_sources: list[str]


class BayesianTrustCalculator:
    """Calculate Bayesian trust scores for relationships and nodes."""

    def __init__(self, prior_trust: float = 0.5) -> None:
        self.prior_trust = prior_trust

    def calculate_relationship_trust(
        self,
        semantic_sim: float,
        context_overlap: float,
        source_trust: float,
        target_trust: float,
        evidence_count: int,
        temporal_consistency: float = 1.0,
    ) -> float:
        """Calculate Bayesian trust score for relationship between chunks."""
        # Evidence strength based on similarity and context
        evidence_strength = (
            semantic_sim * 0.4 + context_overlap * 0.3 + temporal_consistency * 0.3
        )

        # Node trustworthiness (average of source and target)
        node_trust = (source_trust + target_trust) / 2.0

        # Evidence count factor (more evidence = higher trust, with diminishing returns)
        evidence_factor = 1 - np.exp(-evidence_count / 3.0)

        # Bayesian update
        likelihood = evidence_strength * node_trust * evidence_factor

        # Posterior trust using Bayesian formula
        posterior_trust = (likelihood * self.prior_trust) / (
            likelihood * self.prior_trust + (1 - likelihood) * (1 - self.prior_trust)
        )

        return max(0.01, min(0.99, posterior_trust))

    def update_node_trust(
        self,
        current_trust: float,
        successful_retrievals: int,
        total_retrievals: int,
        connection_quality: float,
    ) -> float:
        """Update node trust based on retrieval performance."""
        if total_retrievals == 0:
            return current_trust

        success_rate = successful_retrievals / total_retrievals

        # Bayesian update based on performance
        likelihood = success_rate * connection_quality

        posterior_trust = (likelihood * current_trust) / (
            likelihood * current_trust + (1 - likelihood) * (1 - current_trust)
        )

        return max(0.01, min(0.99, posterior_trust))


class GraphRAGBuilder:
    """Build knowledge graph with Bayesian trust networks."""

    def __init__(self, data_dir: Path = Path("data")) -> None:
        self.data_dir = data_dir
        self.embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        self.trust_calculator = BayesianTrustCalculator()

        # Database paths
        self.local_db_path = self.data_dir / "wikipedia_local_context.db"
        self.global_db_path = self.data_dir / "wikipedia_global_context.db"
        self.graph_db_path = self.data_dir / "wikipedia_graph.db"

        # Knowledge graph
        self.knowledge_graph = nx.DiGraph()

        # Cache for embeddings and chunks
        self.chunk_cache = {}
        self.embedding_cache = {}

    def load_chunks_and_embeddings(self) -> dict[str, Any]:
        """Load all chunks and their embeddings from database."""
        chunks = {}

        with sqlite3.connect(self.local_db_path) as conn:
            cursor = conn.execute(
                """
                SELECT chunk_id, parent_title, section_title, content, local_summary,
                       local_tags, temporal_context, geographic_context,
                       cross_references, embedding
                FROM local_contexts
            """
            )

            for row in cursor.fetchall():
                chunk_id = row[0]
                embedding_bytes = row[9]

                # Reconstruct embedding from bytes
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

                chunk_data = {
                    "chunk_id": chunk_id,
                    "parent_title": row[1],
                    "section_title": row[2],
                    "content": row[3],
                    "local_summary": row[4],
                    "local_tags": json.loads(row[5]) if row[5] else [],
                    "temporal_context": row[6],
                    "geographic_context": row[7],
                    "cross_references": json.loads(row[8]) if row[8] else [],
                    "embedding": embedding,
                }

                chunks[chunk_id] = chunk_data

        logger.info(f"Loaded {len(chunks)} chunks with embeddings")
        return chunks

    def calculate_semantic_relationships(
        self, chunks: dict[str, Any], similarity_threshold: float = 0.3
    ) -> list[SemanticRelationship]:
        """Calculate semantic relationships between chunks."""
        relationships = []
        chunk_ids = list(chunks.keys())

        logger.info(f"Calculating semantic relationships for {len(chunk_ids)} chunks")

        # Create embedding matrix for batch similarity calculation
        embeddings = np.array([chunks[chunk_id]["embedding"] for chunk_id in chunk_ids])

        # Calculate pairwise similarities
        similarity_matrix = cosine_similarity(embeddings)

        for i, source_id in enumerate(chunk_ids):
            for j, target_id in enumerate(chunk_ids):
                if i >= j:  # Skip self and duplicate pairs
                    continue

                semantic_sim = similarity_matrix[i][j]

                if semantic_sim < similarity_threshold:
                    continue

                source_chunk = chunks[source_id]
                target_chunk = chunks[target_id]

                # Calculate contextual factors
                context_relevance = self._calculate_contextual_relevance(
                    source_chunk, target_chunk
                )

                temporal_alignment = self._calculate_temporal_alignment(
                    source_chunk, target_chunk
                )

                geographic_alignment = self._calculate_geographic_alignment(
                    source_chunk, target_chunk
                )

                # Determine relationship type
                rel_type = self._determine_relationship_type(
                    source_chunk, target_chunk, semantic_sim
                )

                # Calculate trust weight
                source_trust = self._get_chunk_trust(source_id)
                target_trust = self._get_chunk_trust(target_id)

                trust_weight = self.trust_calculator.calculate_relationship_trust(
                    semantic_sim=semantic_sim,
                    context_overlap=context_relevance,
                    source_trust=source_trust,
                    target_trust=target_trust,
                    evidence_count=len(source_chunk["cross_references"])
                    + len(target_chunk["cross_references"]),
                    temporal_consistency=temporal_alignment,
                )

                # Create relationship
                relationship = SemanticRelationship(
                    source_chunk=source_id,
                    target_chunk=target_id,
                    relationship_type=rel_type,
                    semantic_similarity=semantic_sim,
                    contextual_relevance=context_relevance,
                    temporal_alignment=temporal_alignment,
                    geographic_alignment=geographic_alignment,
                    trust_weight=trust_weight,
                    evidence_sources=source_chunk["cross_references"]
                    + target_chunk["cross_references"],
                )

                relationships.append(relationship)

        logger.info(f"Found {len(relationships)} semantic relationships")
        return relationships

    def _calculate_contextual_relevance(self, chunk1: dict, chunk2: dict) -> float:
        """Calculate contextual relevance between two chunks."""
        # Tag overlap
        tags1 = set(chunk1["local_tags"])
        tags2 = set(chunk2["local_tags"])

        if (not tags1 and not tags2) or not tags1 or not tags2:
            tag_overlap = 0.0
        else:
            tag_overlap = len(tags1 & tags2) / len(tags1 | tags2)

        # Parent article relationship
        same_article = chunk1["parent_title"] == chunk2["parent_title"]
        article_factor = 1.0 if same_article else 0.5

        # Cross-reference overlap
        refs1 = set(chunk1["cross_references"])
        refs2 = set(chunk2["cross_references"])

        if refs1 and refs2:
            ref_overlap = len(refs1 & refs2) / len(refs1 | refs2)
        else:
            ref_overlap = 0.0

        # Weighted combination
        relevance = tag_overlap * 0.4 + article_factor * 0.3 + ref_overlap * 0.3
        return min(1.0, relevance)

    def _calculate_temporal_alignment(self, chunk1: dict, chunk2: dict) -> float:
        """Calculate temporal alignment between chunks."""
        temp1 = chunk1.get("temporal_context")
        temp2 = chunk2.get("temporal_context")

        if not temp1 or not temp2:
            return 0.7  # Neutral score for missing temporal data

        # Simple year-based alignment
        try:
            if temp1 == temp2:
                return 1.0

            # Extract years from temporal context
            years1 = self._extract_years(temp1)
            years2 = self._extract_years(temp2)

            if not years1 or not years2:
                return 0.7

            # Calculate temporal distance
            min_distance = min(abs(y1 - y2) for y1 in years1 for y2 in years2)

            # Convert distance to similarity (closer = more similar)
            if min_distance == 0:
                return 1.0
            if min_distance <= 10:
                return 0.9
            if min_distance <= 50:
                return 0.7
            if min_distance <= 100:
                return 0.5
            return 0.3

        except:
            return 0.7

    def _extract_years(self, temporal_context: str) -> list[int]:
        """Extract years from temporal context string."""
        import re

        year_pattern = r"\b(1[0-9]{3}|20[0-2][0-9])\b"
        matches = re.findall(year_pattern, temporal_context)
        return [int(year) for year in matches]

    def _calculate_geographic_alignment(self, chunk1: dict, chunk2: dict) -> float:
        """Calculate geographic alignment between chunks."""
        geo1 = chunk1.get("geographic_context")
        geo2 = chunk2.get("geographic_context")

        if not geo1 or not geo2:
            return 0.7  # Neutral score

        if geo1 == geo2:
            return 1.0

        # Simple hierarchical matching (e.g., "Germany" and "Europe")
        geo_hierarchy = {
            "Germany": "Europe",
            "France": "Europe",
            "England": "Europe",
            "Italy": "Europe",
            "Spain": "Europe",
        }

        # Check if one is parent of other
        if geo_hierarchy.get(geo1) == geo2 or geo_hierarchy.get(geo2) == geo1:
            return 0.8

        # Same continent/region
        if (
            geo_hierarchy.get(geo1) == geo_hierarchy.get(geo2)
            and geo_hierarchy.get(geo1) is not None
        ):
            return 0.6

        return 0.3  # Different regions

    def _determine_relationship_type(
        self, chunk1: dict, chunk2: dict, semantic_sim: float
    ) -> str:
        """Determine the type of relationship between chunks."""
        # Same article = hierarchical relationship
        if chunk1["parent_title"] == chunk2["parent_title"]:
            return "hierarchical"

        # High semantic similarity = semantic relationship
        if semantic_sim > 0.7:
            return "semantic"

        # Temporal context suggests causal/temporal relationship
        temp1 = chunk1.get("temporal_context")
        temp2 = chunk2.get("temporal_context")

        if temp1 and temp2:
            years1 = self._extract_years(temp1)
            years2 = self._extract_years(temp2)

            if years1 and years2:
                # Check for temporal progression
                max_year1 = max(years1)
                min_year2 = min(years2)

                if max_year1 < min_year2 and min_year2 - max_year1 <= 50:
                    return "temporal"

        # Cross-references suggest reference relationship
        refs1 = set(chunk1["cross_references"])
        refs2 = set(chunk2["cross_references"])

        if chunk1["parent_title"] in refs2 or chunk2["parent_title"] in refs1:
            return "reference"

        # Default to semantic
        return "semantic"

    def _get_chunk_trust(self, chunk_id: str) -> float:
        """Get trust score for a chunk (from parent article)."""
        with sqlite3.connect(self.local_db_path) as conn:
            cursor = conn.execute(
                "SELECT parent_title FROM local_contexts WHERE chunk_id = ?",
                (chunk_id,),
            )
            result = cursor.fetchone()

            if not result:
                return 0.5

            parent_title = result[0]

        # Get parent article trust score
        with sqlite3.connect(self.global_db_path) as conn:
            cursor = conn.execute(
                "SELECT trust_score FROM global_contexts WHERE title = ?",
                (parent_title,),
            )
            result = cursor.fetchone()

            if result:
                return result[0]
            return 0.5

    def build_knowledge_graph(self, relationships: list[SemanticRelationship]) -> None:
        """Build NetworkX knowledge graph from relationships."""
        logger.info("Building NetworkX knowledge graph")

        # Add nodes (chunks)
        for rel in relationships:
            if not self.knowledge_graph.has_node(rel.source_chunk):
                self.knowledge_graph.add_node(
                    rel.source_chunk,
                    trust_score=self._get_chunk_trust(rel.source_chunk),
                    node_type="chunk",
                )

            if not self.knowledge_graph.has_node(rel.target_chunk):
                self.knowledge_graph.add_node(
                    rel.target_chunk,
                    trust_score=self._get_chunk_trust(rel.target_chunk),
                    node_type="chunk",
                )

        # Add edges (relationships)
        for rel in relationships:
            self.knowledge_graph.add_edge(
                rel.source_chunk,
                rel.target_chunk,
                relationship_type=rel.relationship_type,
                semantic_similarity=rel.semantic_similarity,
                contextual_relevance=rel.contextual_relevance,
                trust_weight=rel.trust_weight,
                temporal_alignment=rel.temporal_alignment,
                geographic_alignment=rel.geographic_alignment,
            )

        logger.info(
            f"Knowledge graph: {self.knowledge_graph.number_of_nodes()} nodes, "
            f"{self.knowledge_graph.number_of_edges()} edges"
        )

    def store_graph_in_database(
        self, relationships: list[SemanticRelationship]
    ) -> None:
        """Store graph relationships in database."""
        with sqlite3.connect(self.graph_db_path) as conn:
            # Store edges
            for rel in relationships:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO graph_edges
                    (source_node, target_node, relationship_type, trust_weight,
                     evidence_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        rel.source_chunk,
                        rel.target_chunk,
                        rel.relationship_type,
                        rel.trust_weight,
                        len(rel.evidence_sources),
                        json.dumps(
                            {
                                "semantic_similarity": float(rel.semantic_similarity),
                                "contextual_relevance": float(rel.contextual_relevance),
                                "temporal_alignment": float(rel.temporal_alignment),
                                "geographic_alignment": float(rel.geographic_alignment),
                                "evidence_sources": rel.evidence_sources,
                            }
                        ),
                    ),
                )

        logger.info(f"Stored {len(relationships)} relationships in graph database")

    def find_trust_paths(
        self, source_chunk: str, target_chunk: str, max_path_length: int = 4
    ) -> list[tuple[list[str], float]]:
        """Find trust-weighted paths between chunks."""
        if not self.knowledge_graph.has_node(
            source_chunk
        ) or not self.knowledge_graph.has_node(target_chunk):
            return []

        try:
            # Find all simple paths
            paths = list(
                nx.all_simple_paths(
                    self.knowledge_graph,
                    source_chunk,
                    target_chunk,
                    cutoff=max_path_length,
                )
            )

            # Calculate trust score for each path
            trust_paths = []

            for path in paths:
                path_trust = 1.0

                for i in range(len(path) - 1):
                    edge_data = self.knowledge_graph.get_edge_data(path[i], path[i + 1])
                    edge_trust = edge_data.get("trust_weight", 0.5)
                    path_trust *= edge_trust

                trust_paths.append((path, path_trust))

            # Sort by trust score
            trust_paths.sort(key=lambda x: x[1], reverse=True)

            return trust_paths[:10]  # Return top 10 paths

        except nx.NetworkXNoPath:
            return []


async def main() -> None:
    """Test the Graph RAG builder."""
    builder = GraphRAGBuilder()

    logger.info("Loading chunks and building knowledge graph...")

    # Load chunks
    chunks = builder.load_chunks_and_embeddings()

    if not chunks:
        logger.error("No chunks found. Please run Wikipedia ingestion first.")
        return

    # Calculate relationships
    relationships = builder.calculate_semantic_relationships(chunks)

    # Build graph
    builder.build_knowledge_graph(relationships)

    # Store in database
    builder.store_graph_in_database(relationships)

    # Test graph queries
    chunk_ids = list(chunks.keys())
    if len(chunk_ids) >= 2:
        source = chunk_ids[0]
        target = chunk_ids[1]

        trust_paths = builder.find_trust_paths(source, target)

        print(f"\n=== Trust Paths from {source} to {target} ===")
        for i, (path, trust) in enumerate(trust_paths[:3]):
            print(f"Path {i + 1} (Trust: {trust:.3f}): {' -> '.join(path)}")

    # Graph statistics
    print("\n=== Graph Statistics ===")
    print(f"Nodes: {builder.knowledge_graph.number_of_nodes()}")
    print(f"Edges: {builder.knowledge_graph.number_of_edges()}")
    print(
        f"Average degree: {sum(dict(builder.knowledge_graph.degree()).values()) / builder.knowledge_graph.number_of_nodes():.2f}"
    )

    # Trust distribution
    trust_weights = [
        data["trust_weight"] for _, _, data in builder.knowledge_graph.edges(data=True)
    ]
    if trust_weights:
        print(f"Average trust weight: {np.mean(trust_weights):.3f}")
        print(f"Trust weight std: {np.std(trust_weights):.3f}")

    logger.info("Graph RAG building completed!")


if __name__ == "__main__":
    asyncio.run(main())
