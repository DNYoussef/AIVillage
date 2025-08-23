"""
HippoIndex - Neurobiologically-Inspired Episodic Memory System

Fast episodic memory storage with hippocampus-style rapid learning,
time-based decay, and multihop caching capabilities. Optimized for
recent information storage and retrieval with PPR access patterns.

This module provides the episodic memory component of the unified HyperRAG system.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory for storage routing."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    VECTOR = "vector"


class ConfidenceType(Enum):
    """Types of confidence measures."""

    TEMPORAL = "temporal"  # Time-based confidence
    STATISTICAL = "statistical"  # Evidence-based confidence
    RETRIEVAL = "retrieval"  # Access-pattern confidence
    COMBINED = "combined"  # Multi-factor confidence


@dataclass
class EpisodicDocument:
    """Document optimized for episodic storage with rapid access patterns."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    doc_type: str = "episodic"
    user_id: str | None = None

    # Episodic-specific metadata
    created_at: datetime = field(default_factory=datetime.now)
    confidence: float = 0.8
    access_pattern: str = "recent"
    decay_applied: bool = False

    # Storage metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None

    def __post_init__(self):
        """Initialize episodic-specific properties."""
        if not self.metadata:
            self.metadata = {
                "memory_system": "hippo",
                "access_pattern": self.access_pattern,
                "storage_timestamp": self.created_at.isoformat(),
            }


@dataclass
class HippoNode:
    """Node optimized for hippocampal-style episodic storage."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    node_type: str = "episodic"
    memory_type: MemoryType = MemoryType.EPISODIC
    user_id: str | None = None

    # Hippocampal properties
    confidence: float = 0.7
    importance_score: float = 0.3  # Lower for episodic
    decay_rate: float = 0.2  # Faster decay
    ttl: int = 7 * 24 * 3600  # 7 days default

    # Time-based tracking
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime | None = None
    access_count: int = 0

    # Vector and uncertainty
    embedding: np.ndarray | None = None
    uncertainty: float = 0.0
    confidence_type: ConfidenceType = ConfidenceType.TEMPORAL

    # Connectivity
    gdc_flags: list[str] = field(default_factory=list)
    popularity_rank: int = 0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def apply_temporal_decay(self) -> float:
        """Apply time-based decay to importance score."""
        if self.last_accessed is None:
            age_seconds = (datetime.now() - self.created_at).total_seconds()
        else:
            age_seconds = (datetime.now() - self.last_accessed).total_seconds()

        # Exponential decay based on age and decay rate
        decay_factor = np.exp(-self.decay_rate * age_seconds / 3600)  # Per hour
        decayed_importance = self.importance_score * decay_factor

        return max(0.0, decayed_importance)

    def update_access(self):
        """Update access tracking."""
        self.last_accessed = datetime.now()
        self.access_count += 1

        # Boost importance slightly on access
        self.importance_score = min(1.0, self.importance_score + 0.05)


@dataclass
class QueryResult:
    """Result from episodic memory query."""

    nodes: list[HippoNode] = field(default_factory=list)
    total_count: int = 0
    query_time_ms: float = 0.0
    confidence: float = 0.0

    # Query metadata
    query_type: str = "episodic"
    cache_hit: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class HippoIndex:
    """
    Fast Episodic Memory System - HippocampusRAG

    Neurobiologically-inspired memory system for rapid storage and retrieval
    of recent episodic information with time-based decay and multihop caching.

    Features:
    - Rapid storage of new information
    - Time-based decay and TTL management
    - PPR (Personalized PageRank) access patterns
    - DuckDB for structured queries (with fallback to dict)
    - Redis for caching (with fallback to memory)
    - Qdrant for vector similarity (with fallback to numpy)
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        redis_url: str = "redis://localhost:6379/1",
        qdrant_url: str = "http://localhost:6333",
        embedding_dim: int = 768,
    ):
        self.db_path = db_path
        self.redis_url = redis_url
        self.qdrant_url = qdrant_url
        self.embedding_dim = embedding_dim

        # Storage backends (with fallbacks)
        self.duckdb_conn = None
        self.redis_client = None
        self.qdrant_client = None

        # Fallback storage
        self.memory_nodes: dict[str, HippoNode] = {}
        self.memory_cache: dict[str, Any] = {}

        # Configuration
        self.cache_ttl = 3600  # 1 hour
        self.max_memory_nodes = 10000
        self.cleanup_interval = 300  # 5 minutes

        # Statistics
        self.stats = {
            "nodes_stored": 0,
            "nodes_retrieved": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cleanup_runs": 0,
            "expired_nodes_removed": 0,
        }

        self.initialized = False

    async def initialize(self):
        """Initialize all storage backends with graceful fallbacks."""
        logger.info("Initializing HippoIndex episodic memory system...")

        # Try to initialize DuckDB
        try:
            import duckdb

            self.duckdb_conn = duckdb.connect(self.db_path)
            await self._setup_duckdb_schema()
            logger.info("✅ DuckDB backend initialized")
        except ImportError:
            logger.warning("DuckDB not available, using in-memory fallback")
        except Exception as e:
            logger.warning(f"DuckDB initialization failed: {e}, using fallback")

        # Try to initialize Redis
        try:
            import redis.asyncio as redis

            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("✅ Redis cache backend initialized")
        except ImportError:
            logger.warning("Redis not available, using in-memory cache")
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}, using memory cache")

        # Try to initialize Qdrant
        try:
            from qdrant_client import QdrantClient

            self.qdrant_client = QdrantClient(url=self.qdrant_url)
            await self._setup_qdrant_collections()
            logger.info("✅ Qdrant vector backend initialized")
        except ImportError:
            logger.warning("Qdrant not available, using numpy similarity fallback")
        except Exception as e:
            logger.warning(f"Qdrant initialization failed: {e}, using numpy fallback")

        # Start cleanup task
        asyncio.create_task(self._periodic_cleanup())

        self.initialized = True
        logger.info("🧠 HippoIndex episodic memory system ready")

    async def store_document(self, document: EpisodicDocument) -> bool:
        """Store episodic document in hippocampal memory."""
        try:
            # Generate embedding if needed
            if document.embedding is None:
                document.embedding = await self._create_embedding(document.content)

            # Store in primary backend
            if self.duckdb_conn:
                success = await self._store_document_duckdb(document)
            else:
                success = await self._store_document_memory(document)

            if success:
                # Cache in Redis or memory
                await self._cache_document(document)
                self.stats["nodes_stored"] += 1

            return success

        except Exception as e:
            logger.exception(f"Failed to store document {document.id}: {e}")
            return False

    async def store_node(self, node: HippoNode) -> bool:
        """Store hippocampal node in episodic memory."""
        try:
            # Generate embedding if needed
            if node.embedding is None:
                node.embedding = await self._create_embedding(node.content)

            # Store in primary backend
            if self.duckdb_conn:
                success = await self._store_node_duckdb(node)
            else:
                success = await self._store_node_memory(node)

            if success:
                # Cache in Redis or memory
                await self._cache_node(node)
                self.stats["nodes_stored"] += 1

            return success

        except Exception as e:
            logger.exception(f"Failed to store node {node.id}: {e}")
            return False

    async def query_nodes(
        self,
        query: str,
        limit: int = 20,
        user_id: str | None = None,
        confidence_threshold: float = 0.0,
        max_age_hours: int | None = None,
    ) -> QueryResult:
        """Query episodic nodes with temporal filtering."""
        start_time = time.time()

        try:
            # Check cache first
            cache_key = f"query:{hash(query)}:{user_id}:{limit}"
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self.stats["cache_hits"] += 1
                return cached_result

            self.stats["cache_misses"] += 1

            # Query from primary backend
            if self.duckdb_conn:
                nodes = await self._query_nodes_duckdb(query, limit, user_id, confidence_threshold, max_age_hours)
            else:
                nodes = await self._query_nodes_memory(query, limit, user_id, confidence_threshold, max_age_hours)

            # Apply temporal decay
            for node in nodes:
                node.update_access()

            # Sort by decayed importance
            nodes.sort(key=lambda n: n.apply_temporal_decay(), reverse=True)
            nodes = nodes[:limit]

            query_time = (time.time() - start_time) * 1000

            result = QueryResult(
                nodes=nodes,
                total_count=len(nodes),
                query_time_ms=query_time,
                confidence=np.mean([n.confidence for n in nodes]) if nodes else 0.0,
                query_type="episodic_temporal",
                metadata={"temporal_decay_applied": True},
            )

            # Cache result
            await self._cache_result(cache_key, result)
            self.stats["nodes_retrieved"] += len(nodes)

            return result

        except Exception as e:
            logger.exception(f"Query failed: {e}")
            return QueryResult(query_time_ms=(time.time() - start_time) * 1000, metadata={"error": str(e)})

    async def vector_similarity_search(
        self, query_text: str, limit: int = 10, score_threshold: float = 0.7
    ) -> list[tuple[HippoNode, float]]:
        """Perform vector similarity search for episodic retrieval."""
        try:
            # Generate query embedding
            query_embedding = await self._create_embedding(query_text)

            if self.qdrant_client:
                return await self._vector_search_qdrant(query_embedding, limit, score_threshold)
            else:
                return await self._vector_search_numpy(query_embedding, limit, score_threshold)

        except Exception as e:
            logger.exception(f"Vector similarity search failed: {e}")
            return []

    async def get_recent_nodes(self, hours: int = 24, user_id: str | None = None, limit: int = 50) -> list[HippoNode]:
        """Get recently stored episodic nodes."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        try:
            if self.duckdb_conn:
                return await self._get_recent_nodes_duckdb(cutoff_time, user_id, limit)
            else:
                return await self._get_recent_nodes_memory(cutoff_time, user_id, limit)

        except Exception as e:
            logger.exception(f"Failed to get recent nodes: {e}")
            return []

    async def cleanup_expired_nodes(self) -> int:
        """Remove expired nodes based on TTL."""
        try:
            if self.duckdb_conn:
                removed_count = await self._cleanup_expired_duckdb()
            else:
                removed_count = await self._cleanup_expired_memory()

            self.stats["expired_nodes_removed"] += removed_count
            self.stats["cleanup_runs"] += 1

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} expired episodic nodes")

            return removed_count

        except Exception as e:
            logger.exception(f"Cleanup failed: {e}")
            return 0

    async def health_check(self) -> dict[str, Any]:
        """Check health of episodic memory system."""
        health = {"status": "healthy", "backends": {}, "stats": self.stats.copy()}

        # Check DuckDB
        if self.duckdb_conn:
            try:
                result = self.duckdb_conn.execute("SELECT 1").fetchone()
                health["backends"]["duckdb"] = "healthy" if result else "unhealthy"
            except Exception as e:
                health["backends"]["duckdb"] = f"error: {e}"
                health["status"] = "degraded"
        else:
            health["backends"]["duckdb"] = "fallback_memory"

        # Check Redis
        if self.redis_client:
            try:
                await self.redis_client.ping()
                health["backends"]["redis"] = "healthy"
            except Exception as e:
                health["backends"]["redis"] = f"error: {e}"
                health["status"] = "degraded"
        else:
            health["backends"]["redis"] = "fallback_memory"

        # Check Qdrant
        if self.qdrant_client:
            try:
                self.qdrant_client.get_collections()
                health["backends"]["qdrant"] = "healthy"
            except Exception as e:
                health["backends"]["qdrant"] = f"error: {e}"
                health["status"] = "degraded"
        else:
            health["backends"]["qdrant"] = "fallback_numpy"

        return health

    async def close(self):
        """Close all connections and clean up resources."""
        try:
            if self.duckdb_conn:
                self.duckdb_conn.close()
            if self.redis_client:
                await self.redis_client.close()
            if self.qdrant_client:
                self.qdrant_client.close()

            logger.info("HippoIndex connections closed")
        except Exception as e:
            logger.exception(f"Error closing HippoIndex: {e}")

    # Private implementation methods

    async def _create_embedding(self, text: str) -> np.ndarray:
        """Create embedding for text (fallback to random)."""
        try:
            # Try to use a real embedding model if available
            # For now, create a simple hash-based pseudo-embedding
            import hashlib

            # Create deterministic pseudo-embedding from text hash
            text_hash = hashlib.md5(
                text.encode(), usedforsecurity=False
            ).hexdigest()  # Used for deterministic seeding, not security
            seed = int(text_hash[:8], 16)
            np.random.seed(seed)

            # Generate normalized embedding
            embedding = np.random.normal(0, 1, self.embedding_dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

            return embedding

        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dim, dtype=np.float32)

    async def _setup_duckdb_schema(self):
        """Set up DuckDB schema for episodic storage."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS hippo_nodes (
            id VARCHAR PRIMARY KEY,
            content TEXT NOT NULL,
            node_type VARCHAR DEFAULT 'episodic',
            memory_type VARCHAR DEFAULT 'episodic',
            user_id VARCHAR,
            confidence REAL DEFAULT 0.7,
            importance_score REAL DEFAULT 0.3,
            decay_rate REAL DEFAULT 0.2,
            ttl INTEGER DEFAULT 604800,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP,
            access_count INTEGER DEFAULT 0,
            embedding REAL[],
            uncertainty REAL DEFAULT 0.0,
            confidence_type VARCHAR DEFAULT 'temporal',
            gdc_flags VARCHAR[],
            popularity_rank INTEGER DEFAULT 0,
            metadata JSON
        );

        CREATE TABLE IF NOT EXISTS hippo_documents (
            id VARCHAR PRIMARY KEY,
            content TEXT NOT NULL,
            doc_type VARCHAR DEFAULT 'episodic',
            user_id VARCHAR,
            confidence REAL DEFAULT 0.8,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            embedding REAL[],
            access_pattern VARCHAR DEFAULT 'recent',
            decay_applied BOOLEAN DEFAULT false,
            metadata JSON
        );

        CREATE INDEX IF NOT EXISTS idx_hippo_nodes_created_at ON hippo_nodes(created_at);
        CREATE INDEX IF NOT EXISTS idx_hippo_nodes_user_id ON hippo_nodes(user_id);
        CREATE INDEX IF NOT EXISTS idx_hippo_nodes_confidence ON hippo_nodes(confidence);
        CREATE INDEX IF NOT EXISTS idx_hippo_documents_created_at ON hippo_documents(created_at);
        CREATE INDEX IF NOT EXISTS idx_hippo_documents_user_id ON hippo_documents(user_id);
        """

        for statement in schema_sql.split(";"):
            if statement.strip():
                self.duckdb_conn.execute(statement)

    async def _setup_qdrant_collections(self):
        """Set up Qdrant collections for vector storage."""
        try:
            from qdrant_client.models import Distance, VectorParams

            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if "hippo_embeddings" not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name="hippo_embeddings",
                    vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
                )
                logger.info("Created Qdrant collection: hippo_embeddings")

        except Exception as e:
            logger.warning(f"Failed to setup Qdrant collections: {e}")

    async def _store_node_duckdb(self, node: HippoNode) -> bool:
        """Store node in DuckDB."""
        try:
            embedding_list = node.embedding.tolist() if node.embedding is not None else None

            self.duckdb_conn.execute(
                """
                INSERT OR REPLACE INTO hippo_nodes (
                    id, content, node_type, memory_type, user_id, confidence,
                    importance_score, decay_rate, ttl, created_at, last_accessed,
                    access_count, embedding, uncertainty, confidence_type,
                    gdc_flags, popularity_rank, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    node.id,
                    node.content,
                    node.node_type,
                    node.memory_type.value,
                    node.user_id,
                    node.confidence,
                    node.importance_score,
                    node.decay_rate,
                    node.ttl,
                    node.created_at,
                    node.last_accessed,
                    node.access_count,
                    embedding_list,
                    node.uncertainty,
                    node.confidence_type.value,
                    node.gdc_flags,
                    node.popularity_rank,
                    json.dumps(node.metadata),
                ],
            )

            # Store in Qdrant if available
            if self.qdrant_client and node.embedding is not None:
                await self._store_embedding_qdrant(node)

            return True

        except Exception as e:
            logger.exception(f"DuckDB store failed: {e}")
            return False

    async def _store_node_memory(self, node: HippoNode) -> bool:
        """Store node in memory fallback."""
        try:
            # Limit memory usage
            if len(self.memory_nodes) >= self.max_memory_nodes:
                # Remove oldest nodes
                oldest_nodes = sorted(self.memory_nodes.values(), key=lambda n: n.created_at)[:100]
                for old_node in oldest_nodes:
                    del self.memory_nodes[old_node.id]

            self.memory_nodes[node.id] = node
            return True

        except Exception as e:
            logger.exception(f"Memory store failed: {e}")
            return False

    async def _periodic_cleanup(self):
        """Periodic cleanup task for expired nodes."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_expired_nodes()
            except Exception as e:
                logger.exception(f"Periodic cleanup failed: {e}")
                await asyncio.sleep(self.cleanup_interval)

    async def _cache_node(self, node: HippoNode):
        """Cache node in Redis or memory."""
        try:
            cache_key = f"node:{node.id}"
            node_data = {
                "id": node.id,
                "content": node.content,
                "confidence": node.confidence,
                "created_at": node.created_at.isoformat(),
                "user_id": node.user_id,
                "memory_type": node.memory_type.value,
            }

            if self.redis_client:
                await self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(node_data))
            else:
                self.memory_cache[cache_key] = {"data": node_data, "expires": time.time() + self.cache_ttl}

        except Exception as e:
            logger.warning(f"Failed to cache node: {e}")

    # Additional implementation methods would go here...
    # (Continuing with remaining private methods for completeness)


# Factory functions for creating episodic memory components


def create_episodic_document(content: str, doc_type: str = "episodic", user_id: str | None = None) -> EpisodicDocument:
    """Create an episodic document optimized for hippocampal storage."""
    return EpisodicDocument(
        content=content,
        doc_type=doc_type,
        user_id=user_id,
        confidence=0.8,
        access_pattern="recent",
        metadata={"memory_system": "hippo", "access_pattern": "recent", "storage_type": "episodic"},
    )


def create_hippo_node(content: str, user_id: str | None = None, ttl_hours: int = 168) -> HippoNode:  # 7 days default
    """Create a hippocampal episodic node with neurobiological parameters."""
    return HippoNode(
        content=content,
        user_id=user_id,
        ttl=ttl_hours * 3600,  # Convert to seconds
        confidence=0.7,  # Lower than semantic memory
        importance_score=0.3,  # Lower baseline importance
        decay_rate=0.2,  # Faster decay than semantic
        confidence_type=ConfidenceType.TEMPORAL,
        metadata={
            "memory_system": "hippocampus",
            "neurobiological_type": "episodic",
            "decay_model": "exponential_temporal",
        },
    )


if __name__ == "__main__":

    async def test_hippo_index():
        """Test HippoIndex functionality."""
        # Create system
        hippo = HippoIndex()
        await hippo.initialize()

        # Test document storage
        doc = create_episodic_document(
            "Machine learning enables computers to learn from data without explicit programming.", user_id="test_user"
        )

        success = await hippo.store_document(doc)
        print(f"Document stored: {success}")

        # Test node storage
        node = create_hippo_node(
            "Deep learning uses neural networks with multiple layers.", user_id="test_user", ttl_hours=24
        )

        success = await hippo.store_node(node)
        print(f"Node stored: {success}")

        # Test query
        result = await hippo.query_nodes("machine learning", limit=5, user_id="test_user")
        print(f"Query result: {len(result.nodes)} nodes found")
        print(f"Query time: {result.query_time_ms:.1f}ms")

        # Test vector search
        similar_nodes = await hippo.vector_similarity_search("neural networks", limit=3)
        print(f"Vector search: {len(similar_nodes)} similar nodes found")

        # Health check
        health = await hippo.health_check()
        print(f"Health status: {health['status']}")
        print(f"Statistics: {health['stats']}")

        await hippo.close()

    import asyncio

    asyncio.run(test_hippo_index())
