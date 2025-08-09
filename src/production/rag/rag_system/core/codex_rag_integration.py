"""CODEX-compliant RAG Integration Module.

This module implements the RAG pipeline according to CODEX Integration Requirements,
with exact configuration values, models, and performance targets.
"""

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

from diskcache import Cache as DiskCache
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import redis
from sentence_transformers import CrossEncoder, SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CODEX-compliant environment variables
RAG_CACHE_ENABLED = os.getenv("RAG_CACHE_ENABLED", "true").lower() == "true"
RAG_L1_CACHE_SIZE = int(os.getenv("RAG_L1_CACHE_SIZE", "128"))
RAG_REDIS_URL = os.getenv("RAG_REDIS_URL", "redis://localhost:6379/1")
RAG_DISK_CACHE_DIR = os.getenv("RAG_DISK_CACHE_DIR", "/tmp/rag_disk_cache")

RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "paraphrase-MiniLM-L3-v2")
RAG_CROSS_ENCODER_MODEL = os.getenv("RAG_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-2-v2")

RAG_VECTOR_DIM = int(os.getenv("RAG_VECTOR_DIM", "384"))
RAG_FAISS_INDEX_PATH = os.getenv("RAG_FAISS_INDEX_PATH", "./data/faiss_index")
RAG_BM25_CORPUS_PATH = os.getenv("RAG_BM25_CORPUS_PATH", "./data/bm25_corpus")

RAG_DEFAULT_K = int(os.getenv("RAG_DEFAULT_K", "10"))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "512"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))


@dataclass
class Document:
    """Document container with CODEX-compliant metadata."""
    id: str
    title: str
    content: str
    source_type: str = "wikipedia"  # wikipedia, educational, etc.
    metadata: dict[str, Any] | None = None


@dataclass
class Chunk:
    """Chunk with position and metadata."""
    id: str
    document_id: str
    text: str
    position: int
    start_idx: int
    end_idx: int
    metadata: dict[str, Any] | None = None


@dataclass
class RetrievalResult:
    """CODEX-compliant retrieval result."""
    chunk_id: str
    document_id: str
    text: str
    score: float
    retrieval_method: str  # "vector", "keyword", "hybrid"
    metadata: dict[str, Any] | None = None


class CODEXCompliantCache:
    """Three-tier cache meeting CODEX specifications.
    
    L1: In-memory LRU cache (128 entries)
    L2: Redis cache (optional, falls back gracefully)  
    L3: Disk cache for persistence
    """

    def __init__(self):
        self.enabled = RAG_CACHE_ENABLED
        self.l1_cache = OrderedDict()
        self.l1_capacity = RAG_L1_CACHE_SIZE

        # L2 Redis cache (optional)
        self.l2_cache = None
        if self.enabled:
            try:
                redis_url = RAG_REDIS_URL
                # Parse Redis URL
                if redis_url.startswith("redis://"):
                    parts = redis_url.replace("redis://", "").split("/")
                    host_port = parts[0].split(":")
                    host = host_port[0]
                    port = int(host_port[1]) if len(host_port) > 1 else 6379
                    db = int(parts[1]) if len(parts) > 1 else 1

                    self.l2_cache = redis.Redis(
                        host=host, port=port, db=db,
                        decode_responses=True,
                        socket_connect_timeout=1,
                        socket_timeout=1
                    )
                    # Test connection
                    self.l2_cache.ping()
                    logger.info(f"Connected to Redis at {host}:{port}/{db}")
            except Exception as e:
                logger.warning(f"Redis unavailable, falling back to disk cache: {e}")
                self.l2_cache = None

        # L3 Disk cache
        self.l3_cache = None
        if self.enabled:
            cache_dir = Path(RAG_DISK_CACHE_DIR)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.l3_cache = DiskCache(str(cache_dir))
            logger.info(f"Disk cache initialized at {cache_dir}")

        # Metrics
        self.hits = {"l1": 0, "l2": 0, "l3": 0}
        self.misses = 0
        self.latencies = []

    def _make_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.encode()).hexdigest()

    async def get(self, query: str) -> list[RetrievalResult] | None:
        """Get cached results with latency tracking."""
        if not self.enabled:
            return None

        start_time = time.perf_counter()
        key = self._make_key(query)

        # L1 check
        if key in self.l1_cache:
            self.hits["l1"] += 1
            value = self.l1_cache.pop(key)
            self.l1_cache[key] = value  # Move to end (LRU)
            latency = (time.perf_counter() - start_time) * 1000
            self.latencies.append(latency)
            logger.debug(f"L1 cache hit for query (latency: {latency:.2f}ms)")
            return value

        # L2 Redis check
        if self.l2_cache:
            try:
                value_json = self.l2_cache.get(key)
                if value_json:
                    self.hits["l2"] += 1
                    value = self._deserialize_results(value_json)
                    # Promote to L1
                    self._add_to_l1(key, value)
                    latency = (time.perf_counter() - start_time) * 1000
                    self.latencies.append(latency)
                    logger.debug(f"L2 cache hit for query (latency: {latency:.2f}ms)")
                    return value
            except Exception as e:
                logger.warning(f"L2 cache error: {e}")

        # L3 disk check
        if self.l3_cache and key in self.l3_cache:
            self.hits["l3"] += 1
            value = self.l3_cache[key]
            # Promote to L2 and L1
            if self.l2_cache:
                try:
                    self.l2_cache.setex(key, 3600, self._serialize_results(value))
                except Exception:
                    pass
            self._add_to_l1(key, value)
            latency = (time.perf_counter() - start_time) * 1000
            self.latencies.append(latency)
            logger.debug(f"L3 cache hit for query (latency: {latency:.2f}ms)")
            return value

        self.misses += 1
        return None

    async def set(self, query: str, results: list[RetrievalResult]) -> None:
        """Store results in all cache layers."""
        if not self.enabled:
            return

        key = self._make_key(query)

        # L1
        self._add_to_l1(key, results)

        # L2 Redis
        if self.l2_cache:
            try:
                self.l2_cache.setex(key, 3600, self._serialize_results(results))
            except Exception as e:
                logger.warning(f"Failed to set L2 cache: {e}")

        # L3 disk
        if self.l3_cache:
            self.l3_cache[key] = results

    def _add_to_l1(self, key: str, value: Any) -> None:
        """Add to L1 cache with LRU eviction."""
        self.l1_cache[key] = value
        while len(self.l1_cache) > self.l1_capacity:
            self.l1_cache.popitem(last=False)

    def _serialize_results(self, results: list[RetrievalResult]) -> str:
        """Serialize retrieval results to JSON."""
        data = [
            {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "text": r.text,
                "score": r.score,
                "retrieval_method": r.retrieval_method,
                "metadata": r.metadata
            }
            for r in results
        ]
        return json.dumps(data)

    def _deserialize_results(self, data: str) -> list[RetrievalResult]:
        """Deserialize retrieval results from JSON."""
        items = json.loads(data)
        return [RetrievalResult(**item) for item in items]

    def get_metrics(self) -> dict[str, Any]:
        """Get cache performance metrics."""
        total_hits = sum(self.hits.values())
        total_requests = total_hits + self.misses
        hit_rate = total_hits / total_requests if total_requests > 0 else 0
        avg_latency = np.mean(self.latencies) if self.latencies else 0

        return {
            "hit_rate": hit_rate,
            "hits": self.hits,
            "misses": self.misses,
            "avg_latency_ms": avg_latency,
            "total_requests": total_requests
        }


class CODEXRAGPipeline:
    """CODEX-compliant RAG pipeline implementation.
    
    Features:
    - Exact embedding model: paraphrase-MiniLM-L3-v2 (384 dims)
    - FAISS vector index with ID mapping
    - BM25 keyword search
    - Hybrid retrieval with reciprocal rank fusion
    - Optional cross-encoder reranking
    - Three-tier caching system
    - <100ms retrieval target
    """

    def __init__(self):
        logger.info("Initializing CODEX-compliant RAG pipeline...")

        # Load embedding model (CODEX-specified)
        self.embedder = SentenceTransformer(RAG_EMBEDDING_MODEL)
        self.vector_dim = self.embedder.get_sentence_embedding_dimension()

        # Validate vector dimensions
        if self.vector_dim != RAG_VECTOR_DIM:
            logger.warning(
                f"Model dimension {self.vector_dim} != configured {RAG_VECTOR_DIM}"
            )

        # Optional cross-encoder for reranking
        self.cross_encoder = None
        if RAG_CROSS_ENCODER_MODEL:
            try:
                self.cross_encoder = CrossEncoder(RAG_CROSS_ENCODER_MODEL)
                logger.info(f"Loaded cross-encoder: {RAG_CROSS_ENCODER_MODEL}")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}")

        # Initialize or load FAISS index
        self.index_path = Path(RAG_FAISS_INDEX_PATH)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            logger.info(f"Loaded FAISS index from {self.index_path}")
        else:
            # Create new index with ID mapping
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.vector_dim))
            logger.info(f"Created new FAISS index (dim={self.vector_dim})")

        # BM25 corpus
        self.bm25_path = Path(RAG_BM25_CORPUS_PATH)
        self.bm25_path.parent.mkdir(parents=True, exist_ok=True)

        self.keyword_corpus = []
        self.keyword_ids = []
        self.bm25_index = None

        if self.bm25_path.exists():
            self._load_bm25_corpus()

        # Chunk storage
        self.chunk_store = {}

        # Cache system
        self.cache = CODEXCompliantCache()

        # Performance metrics
        self.retrieval_times = []

        logger.info("RAG pipeline initialization complete")

    def _load_bm25_corpus(self) -> None:
        """Load BM25 corpus from disk."""
        try:
            with open(self.bm25_path, encoding="utf-8") as f:
                data = json.load(f)
                self.keyword_corpus = data["corpus"]
                self.keyword_ids = data["ids"]
                if self.keyword_corpus:
                    self.bm25_index = BM25Okapi(self.keyword_corpus)
                logger.info(f"Loaded BM25 corpus with {len(self.keyword_corpus)} chunks")
        except Exception as e:
            logger.warning(f"Failed to load BM25 corpus: {e}")

    def _save_bm25_corpus(self) -> None:
        """Save BM25 corpus to disk."""
        try:
            data = {
                "corpus": self.keyword_corpus,
                "ids": self.keyword_ids
            }
            with open(self.bm25_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            logger.info(f"Saved BM25 corpus to {self.bm25_path}")
        except Exception as e:
            logger.error(f"Failed to save BM25 corpus: {e}")

    def chunk_document(
        self,
        document: Document,
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> list[Chunk]:
        """Chunk document according to CODEX specifications."""
        chunk_size = chunk_size or RAG_CHUNK_SIZE
        chunk_overlap = chunk_overlap or RAG_CHUNK_OVERLAP

        text = document.content
        words = text.split()
        chunks = []

        start_idx = 0
        position = 0

        while start_idx < len(words):
            end_idx = min(start_idx + chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)

            # Calculate character positions
            char_start = len(" ".join(words[:start_idx])) + (1 if start_idx > 0 else 0)
            char_end = char_start + len(chunk_text)

            chunk = Chunk(
                id=f"{document.id}_chunk_{position}",
                document_id=document.id,
                text=chunk_text,
                position=position,
                start_idx=char_start,
                end_idx=char_end,
                metadata=document.metadata
            )
            chunks.append(chunk)

            position += 1
            start_idx += (chunk_size - chunk_overlap)

        return chunks

    def index_documents(self, documents: list[Document]) -> dict[str, Any]:
        """Index documents with CODEX-compliant processing."""
        start_time = time.perf_counter()
        stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "vectors_indexed": 0,
            "processing_time_ms": 0
        }

        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            stats["documents_processed"] += 1

        stats["chunks_created"] = len(all_chunks)

        if not all_chunks:
            return stats

        # Batch encode chunks
        texts = [chunk.text for chunk in all_chunks]
        embeddings = self.embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Add to FAISS index
        chunk_ids = []
        for chunk, embedding in zip(all_chunks, embeddings, strict=False):
            # Generate unique chunk ID
            chunk_id = hash(chunk.id) & 0x7FFFFFFFFFFFFFFF  # Ensure positive int64
            chunk_ids.append(chunk_id)

            # Store chunk metadata
            self.chunk_store[chunk_id] = {
                "chunk": chunk,
                "embedding": embedding
            }

            # Add to keyword corpus
            tokens = chunk.text.lower().split()
            self.keyword_corpus.append(tokens)
            self.keyword_ids.append(chunk_id)

        # Batch add to FAISS
        embeddings_array = np.array(embeddings).astype("float32")
        ids_array = np.array(chunk_ids, dtype="int64")
        self.index.add_with_ids(embeddings_array, ids_array)
        stats["vectors_indexed"] = len(embeddings)

        # Rebuild BM25 index
        if self.keyword_corpus:
            self.bm25_index = BM25Okapi(self.keyword_corpus)

        # Save indices
        faiss.write_index(self.index, str(self.index_path))
        self._save_bm25_corpus()

        stats["processing_time_ms"] = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"Indexed {stats['documents_processed']} documents, "
            f"{stats['chunks_created']} chunks in {stats['processing_time_ms']:.2f}ms"
        )

        return stats

    async def retrieve(
        self,
        query: str,
        k: int = None,
        use_cache: bool = True
    ) -> tuple[list[RetrievalResult], dict[str, Any]]:
        """Retrieve relevant chunks with <100ms target latency."""
        k = k or RAG_DEFAULT_K
        start_time = time.perf_counter()

        # Check cache
        if use_cache:
            cached = await self.cache.get(query)
            if cached:
                latency = (time.perf_counter() - start_time) * 1000
                self.retrieval_times.append(latency)
                return cached, {"cache_hit": True, "latency_ms": latency}

        # Encode query
        query_embedding = self.embedder.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Vector search
        vector_results = []
        if self.index.ntotal > 0:
            scores, ids = self.index.search(
                np.array([query_embedding]).astype("float32"),
                min(k * 2, self.index.ntotal)
            )
            vector_results = list(zip(ids[0], scores[0], strict=False))

        # Keyword search
        keyword_results = []
        if self.bm25_index and self.keyword_corpus:
            query_tokens = query.lower().split()
            scores = self.bm25_index.get_scores(query_tokens)
            keyword_results = list(zip(self.keyword_ids, scores, strict=False))
            keyword_results.sort(key=lambda x: x[1], reverse=True)
            keyword_results = keyword_results[:k * 2]

        # Reciprocal rank fusion
        combined_scores = {}
        for rank, (chunk_id, score) in enumerate(vector_results):
            if chunk_id != -1:  # FAISS returns -1 for empty results
                combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + 1.0 / (60 + rank)

        for rank, (chunk_id, score) in enumerate(keyword_results):
            combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + 1.0 / (60 + rank)

        # Sort by combined score
        ranked_ids = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        # Build retrieval results
        results = []
        for chunk_id, score in ranked_ids:
            if chunk_id in self.chunk_store:
                chunk_data = self.chunk_store[chunk_id]
                chunk = chunk_data["chunk"]

                result = RetrievalResult(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=float(score),
                    retrieval_method="hybrid",
                    metadata=chunk.metadata
                )
                results.append(result)

        # Optional reranking with cross-encoder
        if self.cross_encoder and results:
            pairs = [(query, r.text) for r in results]
            ce_scores = self.cross_encoder.predict(pairs)
            for result, ce_score in zip(results, ce_scores, strict=False):
                result.score = float(ce_score)
            results.sort(key=lambda x: x.score, reverse=True)

        # Cache results
        if use_cache:
            await self.cache.set(query, results)

        # Calculate metrics
        latency = (time.perf_counter() - start_time) * 1000
        self.retrieval_times.append(latency)

        metrics = {
            "cache_hit": False,
            "latency_ms": latency,
            "num_results": len(results),
            "vector_search": len(vector_results) > 0,
            "keyword_search": len(keyword_results) > 0,
            "reranked": self.cross_encoder is not None
        }

        # Log performance warning if >100ms
        if latency > 100:
            logger.warning(f"Retrieval latency {latency:.2f}ms exceeds 100ms target")

        return results, metrics

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get pipeline performance metrics."""
        if not self.retrieval_times:
            return {
                "avg_latency_ms": 0,
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "meets_target": True
            }

        latencies = np.array(self.retrieval_times)

        return {
            "avg_latency_ms": float(np.mean(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "meets_target": float(np.mean(latencies)) < 100,
            "cache_metrics": self.cache.get_metrics(),
            "index_size": self.index.ntotal,
            "corpus_size": len(self.keyword_corpus)
        }


async def test_integration():
    """Test the CODEX-compliant RAG integration."""
    logger.info("Testing CODEX RAG integration...")

    # Initialize pipeline
    pipeline = CODEXRAGPipeline()

    # Create test documents (Wikipedia-style educational content)
    test_docs = [
        Document(
            id="wiki_1",
            title="Machine Learning",
            content="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future.",
            source_type="wikipedia",
            metadata={"category": "AI", "difficulty": "beginner"}
        ),
        Document(
            id="wiki_2",
            title="Neural Networks",
            content="Neural networks are computing systems inspired by the biological neural networks that constitute animal brains. An artificial neural network is based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain. Each connection, like the synapses in a biological brain, can transmit a signal to other neurons. Deep learning architectures such as deep neural networks, recurrent neural networks, and convolutional neural networks have been applied to fields including computer vision, machine translation, and speech recognition.",
            source_type="wikipedia",
            metadata={"category": "AI", "difficulty": "intermediate"}
        ),
        Document(
            id="wiki_3",
            title="Natural Language Processing",
            content="Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. In particular, it focuses on how to program computers to process and analyze large amounts of natural language data. The goal is to enable computers to understand, interpret, and generate human language in a valuable way. Common NLP tasks include text classification, named entity recognition, machine translation, question answering, and sentiment analysis.",
            source_type="wikipedia",
            metadata={"category": "AI", "difficulty": "intermediate"}
        )
    ]

    # Index documents
    stats = pipeline.index_documents(test_docs)
    logger.info(f"Indexing stats: {stats}")

    # Test queries
    test_queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "What are the applications of NLP?",
        "Explain deep learning architectures",
        "What is artificial intelligence?"
    ]

    # Run retrieval tests
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        results, metrics = await pipeline.retrieve(query, k=5)

        logger.info(f"Retrieved {len(results)} results in {metrics['latency_ms']:.2f}ms")
        logger.info(f"Cache hit: {metrics['cache_hit']}")

        if results:
            logger.info(f"Top result: {results[0].text[:100]}...")
            logger.info(f"Score: {results[0].score:.4f}")

    # Run same queries again to test cache
    logger.info("\n--- Testing cache performance ---")
    for query in test_queries[:2]:
        results, metrics = await pipeline.retrieve(query, k=5)
        logger.info(f"Query: {query[:30]}... - Latency: {metrics['latency_ms']:.2f}ms (cache: {metrics['cache_hit']})")

    # Get final performance metrics
    perf_metrics = pipeline.get_performance_metrics()
    logger.info("\n--- Performance Metrics ---")
    logger.info(f"Average latency: {perf_metrics['avg_latency_ms']:.2f}ms")
    logger.info(f"P95 latency: {perf_metrics['p95_latency_ms']:.2f}ms")
    logger.info(f"Meets <100ms target: {perf_metrics['meets_target']}")
    logger.info(f"Cache hit rate: {perf_metrics['cache_metrics']['hit_rate']:.2%}")

    return perf_metrics


if __name__ == "__main__":
    # Run integration test
    asyncio.run(test_integration())
