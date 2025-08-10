"""HippoIndex: Fast episodic memory storage.

Brain-inspired hippocampal memory system for rapid storage and retrieval
of recent episodic information with time-based decay and PPR access patterns.
"""

from datetime import datetime, timedelta
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any
import uuid

import duckdb
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import redis.asyncio as redis

from .base import (
    ConfidenceType,
    Document,
    Edge,
    EmbeddingManager,
    MemoryBackend,
    MemoryStats,
    MemoryType,
    Node,
    QueryResult,
)
from .schemas import HippoSchema, QdrantSchema, RedisSchema

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .consolidator import MemoryConsolidator

logger = logging.getLogger(__name__)


class EpisodicDocument(Document):
    """Document optimized for episodic storage."""

    def __init__(
        self, content: str, doc_type: str, user_id: str | None = None, **kwargs
    ) -> None:
        super().__init__(
            id=str(uuid.uuid4()),
            content=content,
            doc_type=doc_type,
            created_at=datetime.now(),
            user_id=user_id,
            **kwargs,
        )
        # Episodic-specific properties
        self.access_pattern = "recent"
        self.decay_applied = False


class HippoNode(Node):
    """Node optimized for hippocampal-style storage."""

    def __init__(self, content: str, user_id: str | None = None, **kwargs) -> None:
        super().__init__(
            id=str(uuid.uuid4()),
            content=content,
            node_type="episodic",
            memory_type=MemoryType.EPISODIC,
            user_id=user_id,
            **kwargs,
        )
        # Default episodic properties
        if self.ttl is None:
            self.ttl = 7 * 24 * 3600  # 7 days default
        if self.decay_rate == 0.1:  # Base class default
            self.decay_rate = 0.2  # Faster decay for episodic
        if self.importance_score == 0.5:  # Base class default
            self.importance_score = 0.3  # Lower importance for episodic


class HippoIndex(MemoryBackend):
    """Fast episodic memory - like hippocampus.

    Features:
    - Rapid storage of new information
    - Lightweight hyperedges
    - Time-based decay
    - PPR-ready access patterns
    - DuckDB for structured queries
    - Redis for caching
    - Qdrant for vector similarity
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        redis_url: str = "redis://localhost:6379",
        qdrant_url: str = "http://localhost:6333",
        embedding_dim: int = 768,
        consolidator: "MemoryConsolidator | None" = None,
    ) -> None:
        self.db_path = db_path
        self.redis_url = redis_url
        self.qdrant_url = qdrant_url
        self.embedding_dim = embedding_dim

        # Connections
        self.duckdb_conn: duckdb.DuckDBPyConnection | None = None
        self.redis_client: redis.Redis | None = None
        self.qdrant_client: QdrantClient | None = None
        self.embedding_manager = EmbeddingManager(embedding_dim)
        self.consolidator: MemoryConsolidator | None = consolidator

        # Schemas
        self.schema = HippoSchema()
        self.qdrant_schema = QdrantSchema()
        self.redis_schema = RedisSchema()

        # Cache settings
        self.cache_ttl = self.redis_schema.get_ttl_configs()

        logger.info("HippoIndex initialized with db_path=%s", db_path)

    def set_consolidator(self, consolidator: "MemoryConsolidator") -> None:
        """Attach a consolidator instance for tracking consolidation stats."""
        self.consolidator = consolidator

    async def initialize(self) -> None:
        """Initialize all backend connections and schemas."""
        try:
            # Initialize DuckDB
            self.duckdb_conn = duckdb.connect(self.db_path)
            await self._setup_duckdb_schema()

            # Initialize Redis
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()

            # Initialize Qdrant
            self.qdrant_client = QdrantClient(url=self.qdrant_url)
            await self._setup_qdrant_collections()

            logger.info("HippoIndex initialization complete")

        except Exception as e:
            logger.exception("Failed to initialize HippoIndex: %s", e)
            raise

    async def close(self) -> None:
        """Close all connections."""
        try:
            if self.duckdb_conn:
                self.duckdb_conn.close()
            if self.redis_client:
                await self.redis_client.close()
            if self.qdrant_client:
                self.qdrant_client.close()
            logger.info("HippoIndex connections closed")
        except Exception as e:
            logger.exception("Error closing HippoIndex: %s", e)

    async def health_check(self) -> dict[str, Any]:
        """Check health of all backend systems."""
        health = {"status": "healthy", "backends": {}}

        try:
            # DuckDB health
            result = self.duckdb_conn.execute("SELECT 1").fetchone()
            health["backends"]["duckdb"] = "healthy" if result else "unhealthy"
        except Exception as e:
            health["backends"]["duckdb"] = f"error: {e!s}"
            health["status"] = "degraded"

        try:
            # Redis health
            await self.redis_client.ping()
            health["backends"]["redis"] = "healthy"
        except Exception as e:
            health["backends"]["redis"] = f"error: {e!s}"
            health["status"] = "degraded"

        try:
            # Qdrant health
            self.qdrant_client.get_collections()
            health["backends"]["qdrant"] = "healthy"
        except Exception as e:
            health["backends"]["qdrant"] = f"error: {e!s}"
            health["status"] = "degraded"

        return health

    async def store_node(self, node: Node, generate_embedding: bool = True) -> bool:
        """Store a node in episodic memory."""
        try:
            # Generate embedding if needed
            if generate_embedding and node.embedding is None:
                node.embedding = self.embedding_manager.create_embedding(node.content)

            # Store in DuckDB
            embedding_list = (
                node.embedding.tolist() if node.embedding is not None else None
            )

            self.duckdb_conn.execute(
                """
                INSERT INTO hippo_nodes (
                    id, content, node_type, memory_type, confidence, embedding,
                    created_at, user_id, gdc_flags, popularity_rank,
                    importance_score, decay_rate, ttl, uncertainty,
                    confidence_type, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    node.id,
                    node.content,
                    node.node_type,
                    node.memory_type.value,
                    node.confidence,
                    embedding_list,
                    node.created_at,
                    node.user_id,
                    node.gdc_flags,
                    node.popularity_rank,
                    node.importance_score,
                    node.decay_rate,
                    node.ttl,
                    node.uncertainty,
                    node.confidence_type.value,
                    json.dumps(node.metadata),
                ],
            )

            # Store embedding in Qdrant if available
            if node.embedding is not None:
                point = PointStruct(
                    id=node.id,
                    vector=node.embedding.tolist(),
                    payload={
                        "node_id": node.id,
                        "content": node.content,
                        "user_id": node.user_id,
                        "confidence": node.confidence,
                        "created_at": node.created_at.isoformat(),
                        "memory_type": node.memory_type.value,
                        "gdc_flags": node.gdc_flags,
                        "importance_score": node.importance_score,
                    },
                )

                self.qdrant_client.upsert(
                    collection_name="hippo_embeddings", points=[point]
                )

            # Cache in Redis
            await self._cache_node(node)

            logger.debug("Stored episodic node %s", node.id)
            return True

        except Exception as e:
            logger.exception("Failed to store node %s: %s", node.id, e)
            return False

    async def store_edge(self, edge: Edge) -> bool:
        """Store an edge in episodic memory."""
        try:
            self.duckdb_conn.execute(
                """
                INSERT INTO hippo_edges (
                    id, source_id, target_id, relation, confidence, participants,
                    memory_type, created_at, gdc_flags, popularity_rank,
                    alpha_weight, user_id, uncertainty, evidence_count,
                    source_docs, tags, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    edge.id,
                    edge.source_id,
                    edge.target_id,
                    edge.relation,
                    edge.confidence,
                    edge.participants,
                    edge.memory_type.value,
                    edge.created_at,
                    edge.gdc_flags,
                    edge.popularity_rank,
                    edge.alpha_weight,
                    edge.user_id,
                    edge.uncertainty,
                    edge.evidence_count,
                    edge.source_docs,
                    edge.tags,
                    json.dumps(edge.metadata),
                ],
            )

            # Cache edge relationships
            await self._cache_edge(edge)

            logger.debug("Stored episodic edge %s", edge.id)
            return True

        except Exception as e:
            logger.exception("Failed to store edge %s: %s", edge.id, e)
            return False

    async def store_document(self, document: Document) -> bool:
        """Store a document in episodic memory."""
        try:
            # Generate embedding if needed
            if document.embedding is None:
                document.embedding = self.embedding_manager.create_embedding(
                    document.content
                )

            embedding_list = (
                document.embedding.tolist() if document.embedding is not None else None
            )

            self.duckdb_conn.execute(
                """
                INSERT INTO hippo_documents (
                    id, content, doc_type, created_at, user_id,
                    embedding, confidence, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    document.id,
                    document.content,
                    document.doc_type,
                    document.created_at,
                    document.user_id,
                    embedding_list,
                    document.confidence,
                    json.dumps(document.metadata),
                ],
            )

            logger.debug("Stored episodic document %s", document.id)
            return True

        except Exception as e:
            logger.exception("Failed to store document %s: %s", document.id, e)
            return False

    async def query_nodes(
        self,
        query: str,
        limit: int = 20,
        user_id: str | None = None,
        confidence_threshold: float = 0.0,
        max_age_hours: int | None = None,
    ) -> QueryResult:
        """Query episodic nodes with optional filters."""
        start_time = time.time()

        try:
            # Build query conditions
            conditions = ["confidence >= ?"]
            params = [confidence_threshold]

            if user_id:
                conditions.append("user_id = ?")
                params.append(user_id)

            if max_age_hours:
                cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
                conditions.append("created_at >= ?")
                params.append(cutoff_time)

            # Add text search if query provided
            if query.strip():
                conditions.append("content LIKE ?")
                params.append(f"%{query}%")

            where_clause = " AND ".join(conditions)
            params.append(limit)

            # Execute query
            results = self.duckdb_conn.execute(
                f"""
                SELECT id, content, node_type, memory_type, confidence,
                       embedding, created_at, last_accessed, access_count,
                       user_id, gdc_flags, popularity_rank, importance_score,
                       decay_rate, ttl, uncertainty, confidence_type, metadata
                FROM hippo_nodes
                WHERE {where_clause}
                ORDER BY importance_score DESC, created_at DESC
                LIMIT ?
            """,
                params,
            ).fetchall()

            # Convert to Node objects
            nodes = []
            for row in results:
                node = Node(
                    id=row[0],
                    content=row[1],
                    node_type=row[2],
                    memory_type=MemoryType(row[3]),
                    confidence=row[4],
                    created_at=row[6],
                    last_accessed=row[7],
                    access_count=row[8],
                    user_id=row[9],
                    gdc_flags=row[10] or [],
                    popularity_rank=row[11],
                    importance_score=row[12],
                    decay_rate=row[13],
                    ttl=row[14],
                    uncertainty=row[15],
                    confidence_type=ConfidenceType(row[16]),
                    metadata=json.loads(row[17]) if row[17] else {},
                )

                # Restore embedding
                if row[5]:
                    node.embedding = np.array(row[5], dtype=np.float32)

                nodes.append(node)

            query_time = (time.time() - start_time) * 1000

            return QueryResult(
                nodes=nodes,
                edges=[],  # Node query doesn't return edges
                total_count=len(nodes),
                query_time_ms=query_time,
                confidence=np.mean([n.confidence for n in nodes]) if nodes else 0.0,
                metadata={"query_type": "episodic_nodes"},
            )

        except Exception as e:
            logger.exception("Failed to query nodes: %s", e)
            query_time = (time.time() - start_time) * 1000
            return QueryResult(
                nodes=[],
                edges=[],
                total_count=0,
                query_time_ms=query_time,
                confidence=0.0,
                metadata={"error": str(e)},
            )

    async def vector_similarity_search(
        self, query_text: str, limit: int = 10, score_threshold: float = 0.7
    ) -> list[tuple[Node, float]]:
        """Perform vector similarity search using Qdrant."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.create_embedding(query_text)

            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name="hippo_embeddings",
                query_vector=query_embedding.tolist(),
                limit=limit,
                score_threshold=score_threshold,
            )

            # Retrieve full node data from DuckDB
            results = []
            for result in search_results:
                node_id = result.id
                node_data = self.duckdb_conn.execute(
                    """
                    SELECT id, content, node_type, memory_type, confidence,
                           embedding, created_at, last_accessed, access_count,
                           user_id, gdc_flags, popularity_rank, importance_score,
                           decay_rate, ttl, uncertainty, confidence_type, metadata
                    FROM hippo_nodes WHERE id = ?
                """,
                    [node_id],
                ).fetchone()

                if node_data:
                    node = Node(
                        id=node_data[0],
                        content=node_data[1],
                        node_type=node_data[2],
                        memory_type=MemoryType(node_data[3]),
                        confidence=node_data[4],
                        created_at=node_data[6],
                        last_accessed=node_data[7],
                        access_count=node_data[8],
                        user_id=node_data[9],
                        gdc_flags=node_data[10] or [],
                        popularity_rank=node_data[11],
                        importance_score=node_data[12],
                        decay_rate=node_data[13],
                        ttl=node_data[14],
                        uncertainty=node_data[15],
                        confidence_type=ConfidenceType(node_data[16]),
                        metadata=json.loads(node_data[17]) if node_data[17] else {},
                    )

                    if node_data[5]:
                        node.embedding = np.array(node_data[5], dtype=np.float32)

                    results.append((node, result.score))

            return results

        except Exception as e:
            logger.exception("Vector similarity search failed: %s", e)
            return []

    async def get_recent_nodes(
        self, hours: int = 24, user_id: str | None = None, limit: int = 50
    ) -> list[Node]:
        """Get recent episodic nodes."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        conditions = ["created_at >= ?"]
        params = [cutoff_time]

        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)

        params.append(limit)
        where_clause = " AND ".join(conditions)

        try:
            results = self.duckdb_conn.execute(
                f"""
                SELECT id, content, node_type, memory_type, confidence,
                       embedding, created_at, last_accessed, access_count,
                       user_id, gdc_flags, popularity_rank, importance_score,
                       decay_rate, ttl, uncertainty, confidence_type, metadata
                FROM hippo_nodes
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ?
            """,
                params,
            ).fetchall()

            nodes = []
            for row in results:
                node = Node(
                    id=row[0],
                    content=row[1],
                    node_type=row[2],
                    memory_type=MemoryType(row[3]),
                    confidence=row[4],
                    created_at=row[6],
                    last_accessed=row[7],
                    access_count=row[8],
                    user_id=row[9],
                    gdc_flags=row[10] or [],
                    popularity_rank=row[11],
                    importance_score=row[12],
                    decay_rate=row[13],
                    ttl=row[14],
                    uncertainty=row[15],
                    confidence_type=ConfidenceType(row[16]),
                    metadata=json.loads(row[17]) if row[17] else {},
                )

                if row[5]:
                    node.embedding = np.array(row[5], dtype=np.float32)

                nodes.append(node)

            return nodes

        except Exception as e:
            logger.exception("Failed to get recent nodes: %s", e)
            return []

    async def cleanup_expired_nodes(self) -> int:
        """Remove expired nodes based on TTL."""
        try:
            # Find expired nodes
            current_time = datetime.now()

            expired_nodes = self.duckdb_conn.execute(
                """
                SELECT id FROM hippo_nodes
                WHERE ttl IS NOT NULL
                AND (EXTRACT(EPOCH FROM ? - created_at)) > ttl
            """,
                [current_time],
            ).fetchall()

            expired_count = len(expired_nodes)

            if expired_count > 0:
                # Delete from DuckDB
                expired_ids = [row[0] for row in expired_nodes]
                placeholders = ",".join(["?" for _ in expired_ids])

                self.duckdb_conn.execute(
                    f"""
                    DELETE FROM hippo_nodes WHERE id IN ({placeholders})
                """,
                    expired_ids,
                )

                # Delete from Qdrant
                self.qdrant_client.delete(
                    collection_name="hippo_embeddings", points_selector=expired_ids
                )

                # Clear from Redis cache
                for node_id in expired_ids:
                    cache_key = self.redis_schema.get_key_patterns()["node"].format(
                        node_id=node_id
                    )
                    await self.redis_client.delete(cache_key)

                logger.info("Cleaned up %d expired episodic nodes", expired_count)

            return expired_count

        except Exception as e:
            logger.exception("Failed to cleanup expired nodes: %s", e)
            return 0

    async def get_memory_stats(self) -> MemoryStats:
        """Get statistics about episodic memory usage."""
        try:
            # Get node counts
            node_stats = self.duckdb_conn.execute(
                """
                SELECT
                    COUNT(*) as total_nodes,
                    COUNT(CASE WHEN memory_type = 'episodic' THEN 1 END) as episodic_nodes,
                    AVG(confidence) as avg_confidence
                FROM hippo_nodes
            """
            ).fetchone()

            # Get edge count
            edge_count = self.duckdb_conn.execute(
                """
                SELECT COUNT(*) FROM hippo_edges
            """
            ).fetchone()[0]
            memory_usage_mb = self._calculate_memory_usage_mb()
            last_consolidation = (
                self.consolidator.last_consolidation
                if self.consolidator is not None
                else None
            )
            pending_consolidations = (
                self.consolidator.pending_consolidations
                if self.consolidator is not None
                else 0
            )

            return MemoryStats(
                total_nodes=node_stats[0] or 0,
                total_edges=edge_count or 0,
                episodic_nodes=node_stats[1] or 0,
                semantic_nodes=0,  # HippoIndex only handles episodic
                avg_confidence=node_stats[2] or 0.0,
                memory_usage_mb=memory_usage_mb,
                last_consolidation=last_consolidation,
                pending_consolidations=pending_consolidations,
            )

        except Exception as e:
            logger.exception("Failed to get memory stats: %s", e)
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

    def _calculate_memory_usage_mb(self) -> float:
        """Calculate DuckDB database size in megabytes."""
        try:
            if self.duckdb_conn is None:
                return 0.0
            if self.db_path != ":memory:" and os.path.exists(self.db_path):
                return os.path.getsize(self.db_path) / (1024 * 1024)
            size_info = self.duckdb_conn.execute("PRAGMA database_size").fetchone()
            if size_info and len(size_info) > 7:
                return self._parse_size_to_mb(size_info[7])
        except Exception as e:
            logger.warning("Failed to calculate memory usage: %s", e)
        return 0.0

    @staticmethod
    def _parse_size_to_mb(size_str: str) -> float:
        """Convert DuckDB size string (e.g., '10.0 KiB') to megabytes."""
        try:
            value, unit = size_str.split()
            factors = {
                "bytes": 1,
                "KiB": 1024,
                "MiB": 1024**2,
                "GiB": 1024**3,
                "TiB": 1024**4,
            }
            size_bytes = float(value) * factors.get(unit, 1)
            return size_bytes / (1024**2)
        except Exception:
            return 0.0

    async def _setup_duckdb_schema(self) -> None:
        """Set up DuckDB tables and indexes."""
        # Create tables
        for sql in self.schema.get_create_tables_sql():
            self.duckdb_conn.execute(sql)

        # Create indexes
        for sql in self.schema.get_create_indexes_sql():
            try:
                self.duckdb_conn.execute(sql)
            except Exception as e:
                # Some indexes might not be supported in all DuckDB versions
                logger.warning("Failed to create index: %s", e)

        # Create materialized views
        for sql in self.schema.get_materialized_views_sql():
            try:
                self.duckdb_conn.execute(sql)
            except Exception as e:
                logger.warning("Failed to create materialized view: %s", e)

        logger.info("DuckDB schema setup complete")

    async def _setup_qdrant_collections(self) -> None:
        """Set up Qdrant collections for vector storage."""
        collections_config = self.qdrant_schema.get_collection_configs()
        self.qdrant_schema.get_hnsw_configs()

        for collection_name, config in collections_config.items():
            try:
                # Check if collection exists
                existing_collections = self.qdrant_client.get_collections()
                if collection_name not in [
                    c.name for c in existing_collections.collections
                ]:
                    # Create collection
                    self.qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=config["vectors"]["size"], distance=Distance.COSINE
                        ),
                    )
                    logger.info("Created Qdrant collection: %s", collection_name)

            except Exception as e:
                logger.exception(
                    "Failed to setup Qdrant collection %s: %s", collection_name, e
                )

        logger.info("Qdrant collections setup complete")

    async def _cache_node(self, node: Node) -> None:
        """Cache node in Redis."""
        try:
            cache_key = self.redis_schema.get_key_patterns()["node"].format(
                node_id=node.id
            )

            node_data = {
                "id": node.id,
                "content": node.content,
                "confidence": node.confidence,
                "created_at": node.created_at.isoformat(),
                "user_id": node.user_id,
                "memory_type": node.memory_type.value,
            }

            await self.redis_client.setex(
                cache_key,
                self.cache_ttl.get("query_result", 3600),
                json.dumps(node_data),
            )

        except Exception as e:
            logger.warning("Failed to cache node %s: %s", node.id, e)

    async def _cache_edge(self, edge: Edge) -> None:
        """Cache edge in Redis."""
        try:
            cache_key = self.redis_schema.get_key_patterns()["edge"].format(
                edge_id=edge.id
            )

            edge_data = {
                "id": edge.id,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relation": edge.relation,
                "confidence": edge.confidence,
                "participants": edge.participants,
            }

            await self.redis_client.setex(
                cache_key,
                self.cache_ttl.get("query_result", 3600),
                json.dumps(edge_data),
            )

        except Exception as e:
            logger.warning("Failed to cache edge %s: %s", edge.id, e)


# Factory functions for creating episodic memory components


def create_episodic_document(
    content: str, doc_type: str = "episodic", user_id: str | None = None
) -> EpisodicDocument:
    """Create an episodic document."""
    return EpisodicDocument(
        content=content,
        doc_type=doc_type,
        user_id=user_id,
        confidence=0.8,  # Default episodic confidence
        metadata={"memory_system": "hippo", "access_pattern": "recent"},
    )


def create_hippo_node(
    content: str, user_id: str | None = None, ttl_hours: int = 168
) -> HippoNode:  # 7 days default
    """Create a hippocampal episodic node."""
    return HippoNode(
        content=content,
        user_id=user_id,
        ttl=ttl_hours * 3600,
        confidence=0.7,  # Lower than semantic
        importance_score=0.3,  # Lower for episodic
        decay_rate=0.2,  # Faster decay
        confidence_type=ConfidenceType.TEMPORAL,
    )
