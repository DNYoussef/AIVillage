"""HypergraphKG: Deep semantic memory storage.

Brain-inspired neocortical memory system for consolidated semantic knowledge
with hypergraph relationships, personalized PageRank, and community detection.
"""

from datetime import datetime
import json
import logging
from typing import Any
import uuid

from neo4j import AsyncGraphDatabase
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import redis.asyncio as redis

from .base import (
    ConfidenceType,
    Edge,
    EmbeddingManager,
    MemoryBackend,
    MemoryStats,
    MemoryType,
    Node,
)
from .schemas import HypergraphSchema, QdrantSchema, RedisSchema

logger = logging.getLogger(__name__)


class SemanticNode(Node):
    """Node optimized for semantic/neocortical storage."""

    def __init__(self, content: str, node_type: str = "concept", **kwargs):
        super().__init__(
            id=str(uuid.uuid4()),
            content=content,
            node_type=node_type,
            memory_type=MemoryType.SEMANTIC,
            **kwargs,
        )
        # Semantic-specific defaults
        if self.confidence == 1.0:  # Base class default
            self.confidence = 0.8  # More conservative for semantic
        if self.importance_score == 0.5:  # Base class default
            self.importance_score = 0.8  # Higher for semantic
        if self.decay_rate == 0.1:  # Base class default
            self.decay_rate = 0.01  # Much slower decay
        self.confidence_type = ConfidenceType.BAYESIAN  # Use Bayesian for semantic

        # Semantic-specific properties
        self.community_id: str | None = None
        self.pagerank_score: float = 0.0
        self.consolidation_count: int = 0


class Hyperedge(Edge):
    """Enhanced edge supporting true hypergraph relationships."""

    def __init__(self, participants: list[str], relation: str, **kwargs):
        # Use first two participants as source/target for compatibility
        source_id = participants[0] if participants else str(uuid.uuid4())
        target_id = participants[1] if len(participants) > 1 else str(uuid.uuid4())

        super().__init__(
            id=str(uuid.uuid4()),
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            participants=participants.copy(),
            memory_type=MemoryType.SEMANTIC,
            **kwargs,
        )

        # Hypergraph-specific properties
        self.hyperedge_type: str = "n-ary" if len(participants) > 2 else "binary"
        self.semantic_role: str | None = None  # Subject, predicate, object, etc.
        self.consolidation_source: list[
            str
        ] = []  # IDs of episodic edges that formed this


class Subgraph:
    """A connected subgraph for retrieval and reasoning."""

    def __init__(
        self,
        nodes: list[SemanticNode],
        edges: list[Hyperedge],
        center_node_id: str | None = None,
        confidence: float = 1.0,
    ):
        self.id = str(uuid.uuid4())
        self.nodes = nodes
        self.edges = edges
        self.center_node_id = center_node_id
        self.confidence = confidence
        self.created_at = datetime.now()

        # Computed properties
        self.node_ids = {node.id for node in nodes}
        self.edge_relations = {edge.relation for edge in edges}
        self.avg_confidence = np.mean([e.confidence for e in edges]) if edges else 0.0

    def get_neighbors(self, node_id: str) -> set[str]:
        """Get all nodes connected to the given node."""
        neighbors = set()
        for edge in self.edges:
            if node_id in edge.participants:
                neighbors.update(edge.participants)
                neighbors.discard(node_id)  # Remove self
        return neighbors

    def contains_path(self, start_id: str, end_id: str, max_hops: int = 3) -> bool:
        """Check if there's a path between two nodes within max_hops."""
        if start_id == end_id:
            return True

        visited = {start_id}
        current_level = {start_id}

        for _ in range(max_hops):
            next_level = set()
            for node_id in current_level:
                neighbors = self.get_neighbors(node_id)
                if end_id in neighbors:
                    return True
                next_level.update(neighbors - visited)

            if not next_level:
                break

            visited.update(next_level)
            current_level = next_level

        return False


class HypergraphKG(MemoryBackend):
    """Deep semantic memory - like neocortex.

    Features:
    - Consolidated semantic knowledge
    - True hypergraph relationships (n-ary)
    - Personalized PageRank (PPR)
    - Community detection
    - Neo4j for graph operations
    - Qdrant for semantic vectors
    - Redis for caching
    """

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        qdrant_url: str = "http://localhost:6333",
        redis_url: str = "redis://localhost:6379",
        embedding_dim: int = 768,
    ):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.qdrant_url = qdrant_url
        self.redis_url = redis_url
        self.embedding_dim = embedding_dim

        # Connections
        self.neo4j_driver = None
        self.qdrant_client: QdrantClient | None = None
        self.redis_client: redis.Redis | None = None
        self.embedding_manager = EmbeddingManager(embedding_dim)

        # Schemas
        self.schema = HypergraphSchema()
        self.qdrant_schema = QdrantSchema()
        self.redis_schema = RedisSchema()

        # Cache settings
        self.cache_ttl = self.redis_schema.get_ttl_configs()

        logger.info("HypergraphKG initialized with neo4j_uri=%s", neo4j_uri)

    async def initialize(self) -> None:
        """Initialize all backend connections and schemas."""
        try:
            # Initialize Neo4j
            self.neo4j_driver = AsyncGraphDatabase.driver(
                self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
            )
            await self._setup_neo4j_schema()

            # Initialize Qdrant
            self.qdrant_client = QdrantClient(url=self.qdrant_url)
            await self._setup_qdrant_collections()

            # Initialize Redis
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()

            logger.info("HypergraphKG initialization complete")

        except Exception as e:
            logger.error("Failed to initialize HypergraphKG: %s", e)
            raise

    async def close(self) -> None:
        """Close all connections."""
        try:
            if self.neo4j_driver:
                await self.neo4j_driver.close()
            if self.qdrant_client:
                self.qdrant_client.close()
            if self.redis_client:
                await self.redis_client.close()
            logger.info("HypergraphKG connections closed")
        except Exception as e:
            logger.error("Error closing HypergraphKG: %s", e)

    async def health_check(self) -> dict[str, Any]:
        """Check health of all backend systems."""
        health = {"status": "healthy", "backends": {}}

        try:
            # Neo4j health
            async with self.neo4j_driver.session() as session:
                result = await session.run("RETURN 1 as health")
                record = await result.single()
                health["backends"]["neo4j"] = "healthy" if record else "unhealthy"
        except Exception as e:
            health["backends"]["neo4j"] = f"error: {e!s}"
            health["status"] = "degraded"

        try:
            # Qdrant health
            collections = self.qdrant_client.get_collections()
            health["backends"]["qdrant"] = "healthy"
        except Exception as e:
            health["backends"]["qdrant"] = f"error: {e!s}"
            health["status"] = "degraded"

        try:
            # Redis health
            await self.redis_client.ping()
            health["backends"]["redis"] = "healthy"
        except Exception as e:
            health["backends"]["redis"] = f"error: {e!s}"
            health["status"] = "degraded"

        return health

    async def store_semantic_node(
        self, node: SemanticNode, generate_embedding: bool = True
    ) -> bool:
        """Store a semantic node in the knowledge graph."""
        try:
            # Generate embedding if needed
            if generate_embedding and node.embedding is None:
                node.embedding = self.embedding_manager.create_embedding(node.content)

            # Store in Neo4j
            async with self.neo4j_driver.session() as session:
                await session.run(
                    """
                    MERGE (n:SemanticNode {id: $id})
                    SET n.content = $content,
                        n.node_type = $node_type,
                        n.confidence = $confidence,
                        n.created_at = datetime($created_at),
                        n.importance_score = $importance_score,
                        n.popularity_rank = $popularity_rank,
                        n.gdc_flags = $gdc_flags,
                        n.uncertainty = $uncertainty,
                        n.user_id = $user_id,
                        n.community_id = $community_id,
                        n.pagerank_score = $pagerank_score,
                        n.consolidation_count = $consolidation_count,
                        n.metadata = $metadata
                """,
                    {
                        "id": node.id,
                        "content": node.content,
                        "node_type": node.node_type,
                        "confidence": node.confidence,
                        "created_at": node.created_at.isoformat(),
                        "importance_score": node.importance_score,
                        "popularity_rank": node.popularity_rank,
                        "gdc_flags": node.gdc_flags,
                        "uncertainty": node.uncertainty,
                        "user_id": node.user_id,
                        "community_id": getattr(node, "community_id", None),
                        "pagerank_score": getattr(node, "pagerank_score", 0.0),
                        "consolidation_count": getattr(node, "consolidation_count", 0),
                        "metadata": json.dumps(node.metadata),
                    },
                )

            # Store embedding in Qdrant
            if node.embedding is not None:
                point = PointStruct(
                    id=node.id,
                    vector=node.embedding.tolist(),
                    payload={
                        "node_id": node.id,
                        "content": node.content,
                        "confidence": node.confidence,
                        "created_at": node.created_at.isoformat(),
                        "node_type": node.node_type,
                        "popularity_rank": node.popularity_rank,
                        "community_id": getattr(node, "community_id", None),
                    },
                )

                self.qdrant_client.upsert(
                    collection_name="semantic_embeddings", points=[point]
                )

            # Cache in Redis
            await self._cache_semantic_node(node)

            logger.debug("Stored semantic node %s", node.id)
            return True

        except Exception as e:
            logger.error("Failed to store semantic node %s: %s", node.id, e)
            return False

    async def store_hyperedge(self, hyperedge: Hyperedge) -> bool:
        """Store a hyperedge in the knowledge graph."""
        try:
            async with self.neo4j_driver.session() as session:
                # Create the hyperedge node
                await session.run(
                    """
                    MERGE (h:Hyperedge {id: $id})
                    SET h.relation = $relation,
                        h.confidence = $confidence,
                        h.created_at = datetime($created_at),
                        h.participants = $participants,
                        h.popularity_rank = $popularity_rank,
                        h.evidence_count = $evidence_count,
                        h.gdc_flags = $gdc_flags,
                        h.uncertainty = $uncertainty,
                        h.alpha_weight = $alpha_weight,
                        h.user_id = $user_id,
                        h.hyperedge_type = $hyperedge_type,
                        h.semantic_role = $semantic_role,
                        h.consolidation_source = $consolidation_source,
                        h.metadata = $metadata
                """,
                    {
                        "id": hyperedge.id,
                        "relation": hyperedge.relation,
                        "confidence": hyperedge.confidence,
                        "created_at": hyperedge.created_at.isoformat(),
                        "participants": hyperedge.participants,
                        "popularity_rank": hyperedge.popularity_rank,
                        "evidence_count": hyperedge.evidence_count,
                        "gdc_flags": hyperedge.gdc_flags,
                        "uncertainty": hyperedge.uncertainty,
                        "alpha_weight": hyperedge.alpha_weight,
                        "user_id": hyperedge.user_id,
                        "hyperedge_type": getattr(
                            hyperedge, "hyperedge_type", "binary"
                        ),
                        "semantic_role": getattr(hyperedge, "semantic_role", None),
                        "consolidation_source": getattr(
                            hyperedge, "consolidation_source", []
                        ),
                        "metadata": json.dumps(hyperedge.metadata),
                    },
                )

                # Connect all participants to the hyperedge
                for i, participant_id in enumerate(hyperedge.participants):
                    await session.run(
                        """
                        MATCH (n:SemanticNode {id: $participant_id})
                        MATCH (h:Hyperedge {id: $hyperedge_id})
                        MERGE (n)-[r:PARTICIPATES]->(h)
                        SET r.role = $role,
                            r.confidence = $confidence,
                            r.position = $position
                    """,
                        {
                            "participant_id": participant_id,
                            "hyperedge_id": hyperedge.id,
                            "role": "participant",
                            "confidence": hyperedge.confidence,
                            "position": i,
                        },
                    )

            # Cache the hyperedge
            await self._cache_hyperedge(hyperedge)

            logger.debug("Stored hyperedge %s", hyperedge.id)
            return True

        except Exception as e:
            logger.error("Failed to store hyperedge %s: %s", hyperedge.id, e)
            return False

    async def personalized_pagerank(
        self,
        start_nodes: list[str],
        user_id: str | None = None,
        alpha: float = 0.15,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
    ) -> dict[str, float]:
        """Compute Personalized PageRank from start nodes.

        Args:
            start_nodes: List of node IDs to start from
            user_id: User ID for personalized α-weights
            alpha: Damping factor (restart probability)
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Dict mapping node_id to PageRank score
        """
        try:
            async with self.neo4j_driver.session() as session:
                # Get user's α-profile for relation weighting
                alpha_profile = {}
                if user_id:
                    alpha_result = await session.run(
                        """
                        MATCH (u:User {id: $user_id})
                        RETURN u.alpha_profile as profile
                    """,
                        {"user_id": user_id},
                    )
                    record = await alpha_result.single()
                    if record and record["profile"]:
                        alpha_profile = record["profile"]

                # Get all nodes and edges in the graph
                graph_result = await session.run("""
                    MATCH (n:SemanticNode)
                    OPTIONAL MATCH (n)-[:PARTICIPATES]->(h:Hyperedge)<-[:PARTICIPATES]-(m:SemanticNode)
                    WHERE n.id <> m.id
                    RETURN DISTINCT n.id as node_id,
                           collect(DISTINCT {
                               target: m.id,
                               relation: h.relation,
                               confidence: h.confidence,
                               alpha_weight: COALESCE(h.alpha_weight, 1.0)
                           }) as edges
                """)

                # Build adjacency information
                nodes = {}
                adjacency = {}

                async for record in graph_result:
                    node_id = record["node_id"]
                    edges = record["edges"] or []

                    nodes[node_id] = (
                        1.0 / len(start_nodes) if node_id in start_nodes else 0.0
                    )
                    adjacency[node_id] = []

                    for edge in edges:
                        if edge["target"]:
                            # Apply user's α-profile weighting
                            relation_weight = alpha_profile.get(edge["relation"], 1.0)
                            edge_weight = (
                                edge["confidence"]
                                * edge["alpha_weight"]
                                * relation_weight
                            )
                            adjacency[node_id].append((edge["target"], edge_weight))

                # Normalize adjacency weights
                for node_id in adjacency:
                    total_weight = sum(weight for _, weight in adjacency[node_id])
                    if total_weight > 0:
                        adjacency[node_id] = [
                            (target, weight / total_weight)
                            for target, weight in adjacency[node_id]
                        ]

                # Initialize PageRank scores
                current_scores = {node_id: 1.0 / len(nodes) for node_id in nodes}
                personalization = {
                    node_id: 1.0 / len(start_nodes) if node_id in start_nodes else 0.0
                    for node_id in nodes
                }

                # Iterative computation
                for iteration in range(max_iterations):
                    new_scores = {}

                    for node_id in nodes:
                        # Restart probability component
                        score = alpha * personalization[node_id]

                        # Random walk component
                        for source_id, edges in adjacency.items():
                            for target_id, weight in edges:
                                if target_id == node_id:
                                    score += (
                                        (1 - alpha) * current_scores[source_id] * weight
                                    )

                        new_scores[node_id] = score

                    # Check convergence
                    max_diff = max(
                        abs(new_scores[node_id] - current_scores[node_id])
                        for node_id in nodes
                    )

                    current_scores = new_scores

                    if max_diff < tolerance:
                        logger.debug("PPR converged after %d iterations", iteration + 1)
                        break

                return current_scores

        except Exception as e:
            logger.error("Failed to compute Personalized PageRank: %s", e)
            return {}

    async def query_subgraph(
        self,
        center_nodes: list[str],
        max_depth: int = 2,
        min_confidence: float = 0.5,
        max_nodes: int = 50,
    ) -> Subgraph | None:
        """Extract a subgraph around center nodes for reasoning.

        Args:
            center_nodes: Starting nodes for subgraph extraction
            max_depth: Maximum hops from center nodes
            min_confidence: Minimum confidence threshold
            max_nodes: Maximum nodes to include

        Returns:
            Subgraph object or None if extraction fails
        """
        try:
            async with self.neo4j_driver.session() as session:
                # Extract subgraph using Cypher
                subgraph_result = await session.run(
                    """
                    MATCH (center:SemanticNode)
                    WHERE center.id IN $center_nodes

                    CALL apoc.path.subgraphNodes(center, {
                        relationshipFilter: "PARTICIPATES",
                        minLevel: 0,
                        maxLevel: $max_depth,
                        limit: $max_nodes
                    }) YIELD node as n

                    WITH collect(DISTINCT n) as nodes

                    UNWIND nodes as n1
                    UNWIND nodes as n2

                    MATCH (n1)-[:PARTICIPATES]->(h:Hyperedge)<-[:PARTICIPATES]-(n2)
                    WHERE n1.id <> n2.id
                    AND h.confidence >= $min_confidence

                    RETURN
                        collect(DISTINCT {
                            id: n1.id,
                            content: n1.content,
                            node_type: n1.node_type,
                            confidence: n1.confidence,
                            created_at: n1.created_at,
                            importance_score: n1.importance_score,
                            popularity_rank: n1.popularity_rank,
                            community_id: n1.community_id,
                            pagerank_score: n1.pagerank_score,
                            metadata: n1.metadata
                        }) as nodes,
                        collect(DISTINCT {
                            id: h.id,
                            relation: h.relation,
                            confidence: h.confidence,
                            participants: h.participants,
                            created_at: h.created_at,
                            popularity_rank: h.popularity_rank,
                            evidence_count: h.evidence_count,
                            hyperedge_type: h.hyperedge_type,
                            metadata: h.metadata
                        }) as edges
                """,
                    {
                        "center_nodes": center_nodes,
                        "max_depth": max_depth,
                        "min_confidence": min_confidence,
                        "max_nodes": max_nodes,
                    },
                )

                record = await subgraph_result.single()
                if not record:
                    return None

                # Convert to objects
                nodes = []
                for node_data in record["nodes"]:
                    node = SemanticNode(
                        content=node_data["content"], node_type=node_data["node_type"]
                    )
                    node.id = node_data["id"]
                    node.confidence = node_data["confidence"]
                    node.created_at = datetime.fromisoformat(node_data["created_at"])
                    node.importance_score = node_data["importance_score"]
                    node.popularity_rank = node_data["popularity_rank"]
                    node.community_id = node_data.get("community_id")
                    node.pagerank_score = node_data.get("pagerank_score", 0.0)
                    node.metadata = (
                        json.loads(node_data["metadata"])
                        if node_data["metadata"]
                        else {}
                    )
                    nodes.append(node)

                edges = []
                for edge_data in record["edges"]:
                    edge = Hyperedge(
                        participants=edge_data["participants"],
                        relation=edge_data["relation"],
                    )
                    edge.id = edge_data["id"]
                    edge.confidence = edge_data["confidence"]
                    edge.created_at = datetime.fromisoformat(edge_data["created_at"])
                    edge.popularity_rank = edge_data["popularity_rank"]
                    edge.evidence_count = edge_data["evidence_count"]
                    edge.hyperedge_type = edge_data.get("hyperedge_type", "binary")
                    edge.metadata = (
                        json.loads(edge_data["metadata"])
                        if edge_data["metadata"]
                        else {}
                    )
                    edges.append(edge)

                # Create subgraph
                center_node_id = center_nodes[0] if center_nodes else None
                avg_confidence = (
                    np.mean([e.confidence for e in edges]) if edges else 0.0
                )

                subgraph = Subgraph(
                    nodes=nodes,
                    edges=edges,
                    center_node_id=center_node_id,
                    confidence=avg_confidence,
                )

                return subgraph

        except Exception as e:
            logger.error("Failed to query subgraph: %s", e)
            return None

    async def semantic_similarity_search(
        self,
        query_text: str,
        limit: int = 10,
        score_threshold: float = 0.7,
        community_filter: str | None = None,
    ) -> list[tuple[SemanticNode, float]]:
        """Perform semantic similarity search using vector embeddings."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.create_embedding(query_text)

            # Build Qdrant filter
            filter_conditions = {}
            if community_filter:
                filter_conditions["community_id"] = community_filter

            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name="semantic_embeddings",
                query_vector=query_embedding.tolist(),
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_conditions if filter_conditions else None,
            )

            # Retrieve full node data from Neo4j
            results = []
            for result in search_results:
                node_id = result.id

                async with self.neo4j_driver.session() as session:
                    node_result = await session.run(
                        """
                        MATCH (n:SemanticNode {id: $node_id})
                        RETURN n.id as id, n.content as content, n.node_type as node_type,
                               n.confidence as confidence, n.created_at as created_at,
                               n.importance_score as importance_score,
                               n.popularity_rank as popularity_rank,
                               n.community_id as community_id,
                               n.pagerank_score as pagerank_score,
                               n.metadata as metadata
                    """,
                        {"node_id": node_id},
                    )

                    record = await node_result.single()
                    if record:
                        node = SemanticNode(
                            content=record["content"], node_type=record["node_type"]
                        )
                        node.id = record["id"]
                        node.confidence = record["confidence"]
                        node.created_at = datetime.fromisoformat(record["created_at"])
                        node.importance_score = record["importance_score"]
                        node.popularity_rank = record["popularity_rank"]
                        node.community_id = record.get("community_id")
                        node.pagerank_score = record.get("pagerank_score", 0.0)
                        node.metadata = (
                            json.loads(record["metadata"]) if record["metadata"] else {}
                        )

                        results.append((node, result.score))

            return results

        except Exception as e:
            logger.error("Semantic similarity search failed: %s", e)
            return []

    async def detect_communities(self, algorithm: str = "louvain") -> dict[str, str]:
        """Detect communities in the semantic graph.

        Args:
            algorithm: Community detection algorithm ('louvain', 'label_propagation')

        Returns:
            Dict mapping node_id to community_id
        """
        try:
            async with self.neo4j_driver.session() as session:
                if algorithm == "louvain":
                    result = await session.run("""
                        CALL gds.louvain.stream('semantic-graph')
                        YIELD nodeId, communityId
                        MATCH (n:SemanticNode) WHERE id(n) = nodeId
                        RETURN n.id as node_id, communityId as community_id
                    """)
                elif algorithm == "label_propagation":
                    result = await session.run("""
                        CALL gds.labelPropagation.stream('semantic-graph')
                        YIELD nodeId, communityId
                        MATCH (n:SemanticNode) WHERE id(n) = nodeId
                        RETURN n.id as node_id, communityId as community_id
                    """)
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm}")

                communities = {}
                async for record in result:
                    communities[record["node_id"]] = str(record["community_id"])

                # Update nodes with community IDs
                for node_id, community_id in communities.items():
                    await session.run(
                        """
                        MATCH (n:SemanticNode {id: $node_id})
                        SET n.community_id = $community_id
                    """,
                        {"node_id": node_id, "community_id": community_id},
                    )

                logger.info(
                    "Detected %d communities using %s", len(set(communities.values())), algorithm
                )
                return communities

        except Exception as e:
            logger.error("Community detection failed: %s", e)
            return {}

    async def get_memory_stats(self) -> MemoryStats:
        """Get statistics about semantic memory usage."""
        try:
            async with self.neo4j_driver.session() as session:
                stats_result = await session.run("""
                    MATCH (n:SemanticNode)
                    OPTIONAL MATCH (h:Hyperedge)
                    RETURN
                        count(DISTINCT n) as total_nodes,
                        count(DISTINCT h) as total_edges,
                        avg(n.confidence) as avg_confidence
                """)

                record = await stats_result.single()

                return MemoryStats(
                    total_nodes=record["total_nodes"] or 0,
                    total_edges=record["total_edges"] or 0,
                    episodic_nodes=0,  # HypergraphKG only handles semantic
                    semantic_nodes=record["total_nodes"] or 0,
                    avg_confidence=record["avg_confidence"] or 0.0,
                    memory_usage_mb=0.0,  # TODO: Calculate actual usage
                    last_consolidation=None,  # TODO: Track consolidation
                    pending_consolidations=0,
                )

        except Exception as e:
            logger.error("Failed to get memory stats: %s", e)
            return MemoryStats(
                total_nodes=0,
                total_edges=0,
                episodic_nodes=0,
                semantic_nodes=0,
                avg_confidence=0.0,
                memory_usage_mb=0.0,
                last_consolidation=None,
                pending_consolidations=0,
            )

    # Private helper methods

    async def _setup_neo4j_schema(self) -> None:
        """Set up Neo4j constraints and indexes."""
        async with self.neo4j_driver.session() as session:
            # Create constraints
            for constraint in self.schema.get_node_constraints():
                try:
                    await session.run(constraint)
                except Exception as e:
                    logger.warning("Failed to create constraint: %s", e)

            for constraint in self.schema.get_relationship_constraints():
                try:
                    await session.run(constraint)
                except Exception as e:
                    logger.warning("Failed to create relationship constraint: %s", e)

            # Create indexes
            for index in self.schema.get_indexes():
                try:
                    await session.run(index)
                except Exception as e:
                    logger.warning("Failed to create index: %s", e)

            # Create sample data if needed
            sample_data = self.schema.get_sample_data_cypher()
            for cypher in sample_data:
                try:
                    await session.run(cypher)
                except Exception as e:
                    logger.debug(
                        "Sample data creation (expected to fail if data exists): %s", e
                    )

        logger.info("Neo4j schema setup complete")

    async def _setup_qdrant_collections(self) -> None:
        """Set up Qdrant collections for semantic vectors."""
        collections_config = self.qdrant_schema.get_collection_configs()

        for collection_name, config in collections_config.items():
            if collection_name in ["semantic_embeddings", "user_profiles"]:
                try:
                    existing_collections = self.qdrant_client.get_collections()
                    if collection_name not in [
                        c.name for c in existing_collections.collections
                    ]:
                        self.qdrant_client.create_collection(
                            collection_name=collection_name,
                            vectors_config=VectorParams(
                                size=config["vectors"]["size"], distance=Distance.COSINE
                            ),
                        )
                        logger.info("Created Qdrant collection: %s", collection_name)

                except Exception as e:
                    logger.error(
                        "Failed to setup Qdrant collection %s: %s", collection_name, e
                    )

        logger.info("Qdrant semantic collections setup complete")

    async def _cache_semantic_node(self, node: SemanticNode) -> None:
        """Cache semantic node in Redis."""
        try:
            cache_key = self.redis_schema.get_key_patterns()["node"].format(
                node_id=node.id
            )

            node_data = {
                "id": node.id,
                "content": node.content,
                "confidence": node.confidence,
                "created_at": node.created_at.isoformat(),
                "node_type": node.node_type,
                "memory_type": "semantic",
                "community_id": getattr(node, "community_id", None),
                "pagerank_score": getattr(node, "pagerank_score", 0.0),
            }

            await self.redis_client.setex(
                cache_key,
                self.cache_ttl.get(
                    "user_profile", 86400
                ),  # Cache semantic nodes longer
                json.dumps(node_data),
            )

        except Exception as e:
            logger.warning("Failed to cache semantic node %s: %s", node.id, e)

    async def _cache_hyperedge(self, edge: Hyperedge) -> None:
        """Cache hyperedge in Redis."""
        try:
            cache_key = self.redis_schema.get_key_patterns()["edge"].format(
                edge_id=edge.id
            )

            edge_data = {
                "id": edge.id,
                "relation": edge.relation,
                "confidence": edge.confidence,
                "participants": edge.participants,
                "hyperedge_type": getattr(edge, "hyperedge_type", "binary"),
                "evidence_count": edge.evidence_count,
            }

            await self.redis_client.setex(
                cache_key,
                self.cache_ttl.get("popular_edges", 21600),
                json.dumps(edge_data),
            )

        except Exception as e:
            logger.warning("Failed to cache hyperedge %s: %s", edge.id, e)


# Factory functions for creating semantic memory components


def create_semantic_node(
    content: str, node_type: str = "concept", confidence: float = 0.8
) -> SemanticNode:
    """Create a semantic node for consolidated knowledge."""
    return SemanticNode(
        content=content,
        node_type=node_type,
        confidence=confidence,
        importance_score=0.8,
        decay_rate=0.01,  # Very slow decay for semantic
        confidence_type=ConfidenceType.BAYESIAN,
    )


def create_hyperedge(
    participants: list[str],
    relation: str,
    confidence: float = 0.8,
    evidence_count: int = 1,
) -> Hyperedge:
    """Create a hyperedge for n-ary semantic relationships."""
    edge = Hyperedge(
        participants=participants,
        relation=relation,
        confidence=confidence,
        evidence_count=evidence_count,
    )
    edge.hyperedge_type = "n-ary" if len(participants) > 2 else "binary"
    return edge
