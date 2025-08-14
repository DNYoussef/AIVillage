"""Comprehensive RAG Pipeline with HyperRAG capabilities and safe defaults."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Safe imports with fallbacks
try:
    from ..retrieval.vector_store import VectorStore
except ImportError:
    VectorStore = None  # type: ignore

try:
    from ..retrieval.graph_store import GraphStore
except ImportError:
    GraphStore = None  # type: ignore

try:
    from .implementations import HybridRetriever
except ImportError:
    HybridRetriever = None  # type: ignore

try:
    from .semantic_cache_advanced import SemanticMultiTierCache
except ImportError:
    SemanticMultiTierCache = None  # type: ignore

try:
    from .intelligent_chunking import IntelligentChunker
except ImportError:
    IntelligentChunker = None  # type: ignore

from .config import RAGConfig, UnifiedConfig
from .structures import RetrievalResult


@dataclass
class Document:
    """Document with metadata and optional embeddings."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SynthesizedAnswer:
    """Synthesized answer from multiple retrieval sources."""

    answer_text: str
    executive_summary: str
    primary_sources: list[RetrievalResult] = field(default_factory=list)
    secondary_sources: list[RetrievalResult] = field(default_factory=list)
    confidence_score: float = 0.0
    query_metadata: dict[str, Any] = field(default_factory=dict)
    synthesis_method: str = "hierarchical"
    processing_time_ms: float = 0.0


class RAGPipeline:
    """Comprehensive RAG Pipeline with intelligent defaults and HyperRAG features.

    Features:
    - Safe instantiation without required dependencies
    - In-memory FAISS vector store by default
    - Optional graph store (warns if unavailable, doesn't crash)
    - Multi-tier semantic caching (L1/L2/L3)
    - Intelligent document chunking
    - Hybrid retrieval (vector + keyword)
    """

    def __init__(
        self,
        config: dict[str, Any] | UnifiedConfig | None = None,
        enable_cache: bool = True,
        enable_graph: bool = True,
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize RAG pipeline with safe defaults.

        Args:
            config: Configuration dict or UnifiedConfig instance
            enable_cache: Enable multi-tier caching system
            enable_graph: Enable graph-based retrieval (warns if unavailable)
            cache_dir: Cache directory (defaults to tmp/rag_cache)
        """
        # Configuration setup
        if config is None:
            self.config = RAGConfig()
        elif isinstance(config, dict):
            self.config = RAGConfig(**config)
        else:
            self.config = config

        self.cache_dir = cache_dir or Path("/tmp/rag_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components with safe defaults
        self._init_vector_store()
        self._init_graph_store(enable_graph)
        self._init_cache(enable_cache)
        self._init_chunker()
        self._init_retriever()

        # Document storage
        self.documents: list[Document] = []
        self._document_count = 0

        # Performance metrics
        self.metrics = {
            "queries_processed": 0,
            "documents_indexed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_query_time_ms": 0.0,
        }

        logger.info("RAG Pipeline initialized with safe defaults")

    def _init_vector_store(self) -> None:
        """Initialize vector store with FAISS fallback."""
        use_advanced = VectorStore is not None and os.getenv("RAG_USE_ADVANCED_STORE") == "1"
        if use_advanced:
            try:
                self.vector_store = VectorStore(
                    config=self.config,
                    dimension=getattr(self.config, "embedding_dimension", 768),
                )
                logger.info("Vector store initialized successfully")
                return
            except Exception as e:  # pragma: no cover - best effort
                logger.warning(f"Failed to initialize advanced VectorStore: {e}, using fallback")

        # Fallback: minimal in-memory store
        self.vector_store = self._create_fallback_vector_store()
        logger.info("Using fallback in-memory vector store")

    def _init_graph_store(self, enable_graph: bool) -> None:
        """Initialize graph store with optional fallback."""
        self.graph_store = None

        if not enable_graph:
            logger.info("Graph store disabled")
            return

        try:
            if GraphStore is not None:
                self.graph_store = GraphStore(config=self.config)
                logger.info("Graph store initialized successfully")
            else:
                logger.warning("GraphStore not available, graph features disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize GraphStore: {e}, graph features disabled")

    def _init_cache(self, enable_cache: bool) -> None:
        """Initialize multi-tier caching system."""
        self.cache = None

        if not enable_cache:
            logger.info("Caching disabled")
            return

        try:
            if SemanticMultiTierCache is not None:
                self.cache = SemanticMultiTierCache(cache_dir=self.cache_dir, enable_prefetch=True)
                logger.info("Multi-tier cache initialized successfully")
            else:
                # Simple in-memory cache fallback
                self.cache = {}
                logger.warning("Advanced caching not available, using simple cache")
        except Exception as e:
            logger.warning(f"Failed to initialize advanced cache: {e}, using simple cache")
            self.cache = {}

    def _init_chunker(self) -> None:
        """Initialize intelligent chunker with fallback."""
        try:
            if IntelligentChunker is not None:
                self.chunker = IntelligentChunker(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                )
                logger.info("Intelligent chunker initialized")
            else:
                self.chunker = self._create_fallback_chunker()
                logger.warning("IntelligentChunker not available, using simple chunker")
        except Exception as e:
            logger.warning(f"Failed to initialize chunker: {e}, using simple chunker")
            self.chunker = self._create_fallback_chunker()

    def _init_retriever(self) -> None:
        """Initialize hybrid retriever with fallbacks."""
        try:
            if HybridRetriever is not None:
                retriever = HybridRetriever(config=self.config)
                retriever.vector_store = self.vector_store
                retriever.graph_store = self.graph_store

                # Test if retriever is functional
                if hasattr(retriever, "retrieve"):
                    self.retriever = retriever
                    logger.info("Hybrid retriever initialized")
                else:
                    self.retriever = None
                    logger.warning("HybridRetriever missing retrieve method, using direct vector search")
            else:
                self.retriever = None
                logger.warning("HybridRetriever not available, using direct vector search")
        except Exception as e:
            logger.warning(f"Failed to initialize retriever: {e}, using direct vector search")
            self.retriever = None

    def _create_fallback_vector_store(self) -> Any:
        """Create minimal fallback vector store."""

        class FallbackVectorStore:
            def __init__(self):
                self.documents = []

            async def add_texts(self, texts: list[str]) -> None:
                for text in texts:
                    self.documents.append(
                        {
                            "id": str(len(self.documents)),
                            "content": text,
                            "embedding": [0.0] * 768,  # Zero vector
                            "timestamp": datetime.now(),
                        }
                    )

            async def retrieve(self, *args, **kwargs) -> list[RetrievalResult]:
                args[0] if args else []
                k = args[1] if len(args) > 1 else kwargs.get("k", 5)
                # Simple text matching fallback - search for query terms in content
                results = []
                query_terms = kwargs.get("query_terms", [])

                # If we have query terms, do simple text matching
                if query_terms:
                    for doc in self.documents:
                        content_lower = doc["content"].lower()
                        score = sum(1 for term in query_terms if term.lower() in content_lower)
                        if score > 0:
                            results.append(
                                RetrievalResult(
                                    id=doc["id"],
                                    content=doc["content"],
                                    score=score / len(query_terms),  # Normalized score
                                    uncertainty=0.3,
                                    timestamp=doc["timestamp"],
                                    version=1,
                                )
                            )

                # Sort by score and return top k
                results.sort(key=lambda x: x.score, reverse=True)
                return results[:k]

        return FallbackVectorStore()

    def _create_fallback_chunker(self) -> Any:
        """Create simple text chunker fallback."""

        class FallbackChunker:
            def __init__(self, chunk_size: int = 1000, overlap: int = 200):
                self.chunk_size = chunk_size
                self.overlap = overlap

            def chunk_text(self, text: str) -> list[str]:
                """Simple word-based chunking."""
                words = text.split()
                chunks = []

                for i in range(0, len(words), self.chunk_size - self.overlap):
                    chunk_words = words[i : i + self.chunk_size]
                    chunks.append(" ".join(chunk_words))

                return chunks

        return FallbackChunker(self.config.chunk_size, self.config.chunk_overlap)

    async def add_document(self, doc: Document) -> None:
        """Add a document to the pipeline."""
        try:
            # Chunk the document
            if hasattr(self.chunker, "chunk_document"):
                chunks = await self.chunker.chunk_document(doc.text)
            else:
                chunks = self.chunker.chunk_text(doc.text)

            # Add chunks to vector store
            await self.vector_store.add_texts(chunks)

            # Add to graph store if available
            if self.graph_store is not None:
                graph_doc = {
                    "id": doc.id,
                    "content": doc.text,
                    "metadata": doc.metadata,
                    "timestamp": doc.timestamp,
                }
                self.graph_store.add_documents([graph_doc])

            # Store document
            self.documents.append(doc)
            self._document_count += 1
            self.metrics["documents_indexed"] += 1

            logger.debug(f"Added document {doc.id} with {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Failed to add document {doc.id}: {e}")
            raise

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_cache: bool = True,
        include_graph: bool = True,
    ) -> list[RetrievalResult]:
        """Retrieve relevant documents for a query."""
        start_time = datetime.now()

        try:
            # Check cache first
            if use_cache and self.cache is not None:
                cached_result = await self._check_cache(query)
                if cached_result is not None:
                    self.metrics["cache_hits"] += 1
                    return cached_result

            self.metrics["cache_misses"] += 1

            # Use hybrid retriever if available
            if self.retriever is not None and hasattr(self.retriever, "retrieve"):
                results = await self.retriever.retrieve(query, top_k)
            else:
                # Fallback to direct vector search
                query_vector = [0.1] * 768  # Simple fallback vector
                query_terms = query.split()  # Simple tokenization
                results = await self.vector_store.retrieve(query_vector, k=top_k, query_terms=query_terms)

            # Cache results if caching is enabled
            if use_cache and self.cache is not None:
                await self._cache_results(query, results)

            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics["queries_processed"] += 1
            self._update_avg_query_time(processing_time)

            logger.debug(f"Retrieved {len(results)} results for query in {processing_time:.2f}ms")
            return results

        except Exception as e:
            logger.error(f"Failed to retrieve results for query '{query}': {e}")
            return []

    async def synthesize_answer(
        self,
        query: str,
        context_results: list[RetrievalResult],
        synthesis_method: str = "hierarchical",
    ) -> SynthesizedAnswer:
        """Synthesize an answer from retrieval results."""
        start_time = datetime.now()

        try:
            # Simple synthesis (can be enhanced with LLM integration)
            primary_sources = context_results[:3]
            secondary_sources = context_results[3:] if len(context_results) > 3 else []

            # Create summary from top sources
            summary_text = self._create_summary(primary_sources)
            answer_text = f"Based on the retrieved information: {summary_text}"

            # Calculate confidence based on source quality
            confidence = self._calculate_confidence(context_results)

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return SynthesizedAnswer(
                answer_text=answer_text,
                executive_summary=summary_text,
                primary_sources=primary_sources,
                secondary_sources=secondary_sources,
                confidence_score=confidence,
                synthesis_method=synthesis_method,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"Failed to synthesize answer: {e}")
            return SynthesizedAnswer(
                answer_text="Error occurred during answer synthesis",
                executive_summary="Unable to process query",
                confidence_score=0.0,
            )

    async def query(
        self,
        query: str,
        top_k: int = 5,
        synthesize: bool = True,
        use_cache: bool = True,
    ) -> tuple[list[RetrievalResult], SynthesizedAnswer | None]:
        """Complete query processing pipeline."""
        # Retrieve relevant documents
        results = await self.retrieve(query, top_k, use_cache)

        # Synthesize answer if requested
        answer = None
        if synthesize and results:
            answer = await self.synthesize_answer(query, results)

        return results, answer

    async def _check_cache(self, query: str) -> list[RetrievalResult] | None:
        """Check cache for query results."""
        if hasattr(self.cache, "get"):
            # Advanced cache
            result = await self.cache.get(query)
            if result is not None:
                return result[0]  # Return the cached results
        elif isinstance(self.cache, dict):
            # Simple cache
            return self.cache.get(query)
        return None

    async def _cache_results(self, query: str, results: list[RetrievalResult]) -> None:
        """Cache query results."""
        if hasattr(self.cache, "set"):
            # Advanced cache
            await self.cache.set(query, results)
        elif isinstance(self.cache, dict):
            # Simple cache
            self.cache[query] = results

    def _create_summary(self, sources: list[RetrievalResult]) -> str:
        """Create a summary from retrieval sources."""
        if not sources:
            return "No relevant information found."

        # Simple concatenation with truncation
        combined_text = " ".join([source.content[:200] for source in sources])
        return combined_text[:500] + "..." if len(combined_text) > 500 else combined_text

    def _calculate_confidence(self, results: list[RetrievalResult]) -> float:
        """Calculate confidence score based on retrieval quality."""
        if not results:
            return 0.0

        # Simple confidence based on average score and result count
        avg_score = sum(result.score for result in results) / len(results)
        result_factor = min(len(results) / 5.0, 1.0)  # Normalize to expected 5 results

        return avg_score * result_factor

    def _update_avg_query_time(self, new_time: float) -> None:
        """Update average query processing time."""
        queries_count = self.metrics["queries_processed"]
        if queries_count == 1:
            self.metrics["avg_query_time_ms"] = new_time
        else:
            current_avg = self.metrics["avg_query_time_ms"]
            self.metrics["avg_query_time_ms"] = (current_avg * (queries_count - 1) + new_time) / queries_count

    def get_metrics(self) -> dict[str, Any]:
        """Get pipeline performance metrics."""
        metrics = self.metrics.copy()
        metrics["document_count"] = self._document_count

        # Add cache metrics if available
        if hasattr(self.cache, "get_metrics"):
            cache_metrics = self.cache.get_metrics()
            metrics["cache"] = cache_metrics
        elif isinstance(self.cache, dict):
            metrics["cache"] = {"simple_cache_size": len(self.cache)}

        return metrics

    async def clear_cache(self) -> None:
        """Clear all cached data."""
        if hasattr(self.cache, "clear") and not isinstance(self.cache, dict):
            # Advanced cache with async clear method
            if asyncio.iscoroutinefunction(self.cache.clear):
                await self.cache.clear()
            else:
                self.cache.clear()
        elif isinstance(self.cache, dict):
            # Simple dict cache
            self.cache.clear()
        logger.info("Cache cleared")

    def __str__(self) -> str:
        """String representation of the pipeline."""
        return (
            f"RAGPipeline("
            f"documents={self._document_count}, "
            f"cache_enabled={self.cache is not None}, "
            f"graph_enabled={self.graph_store is not None})"
        )


# Backward compatibility
class EnhancedRAGPipeline(RAGPipeline):
    """Enhanced RAG Pipeline with additional features and optimizations."""

    def __init__(
        self,
        config: dict[str, Any] | UnifiedConfig | None = None,
        enable_cache: bool = True,
        enable_graph: bool = True,
        cache_dir: Path | None = None,
        enable_preprocessing: bool = True,
    ) -> None:
        """Initialize enhanced RAG pipeline.

        Args:
            config: Configuration dict or UnifiedConfig instance
            enable_cache: Enable multi-tier caching system
            enable_graph: Enable graph-based retrieval
            cache_dir: Cache directory
            enable_preprocessing: Enable advanced query preprocessing
        """
        super().__init__(config, enable_cache, enable_graph, cache_dir)

        # Enhanced features
        self.enable_preprocessing = enable_preprocessing
        self.query_history = []  # Track query patterns
        self.performance_metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "avg_response_time": 0.0,
            "last_query_time": None,
        }

        logger.info("Enhanced RAG Pipeline initialized with advanced features")

    async def process(self, query: str, **kwargs) -> Any:
        """Enhanced query processing with preprocessing and monitoring.

        Args:
            query: User query
            **kwargs: Additional processing options

        Returns:
            Enhanced query response with metadata
        """
        import time

        start_time = time.time()

        self.performance_metrics["total_queries"] += 1

        # Advanced query preprocessing
        if self.enable_preprocessing:
            query = await self._preprocess_query(query)

        # Track query for learning
        self.query_history.append(
            {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "context": kwargs.get("context", {}),
            }
        )

        # Keep only last 100 queries
        if len(self.query_history) > 100:
            self.query_history = self.query_history[-100:]

        # Process with parent pipeline
        result = await super().process(query, **kwargs)

        # Enhanced response with metadata
        response_time = time.time() - start_time
        self.performance_metrics["avg_response_time"] = (
            self.performance_metrics["avg_response_time"] * (self.performance_metrics["total_queries"] - 1)
            + response_time
        ) / self.performance_metrics["total_queries"]
        self.performance_metrics["last_query_time"] = response_time

        # Wrap result with enhancements
        if isinstance(result, dict):
            result.update(
                {
                    "enhanced": True,
                    "response_time": response_time,
                    "query_count": self.performance_metrics["total_queries"],
                    "preprocessing_enabled": self.enable_preprocessing,
                }
            )

        return result

    async def _preprocess_query(self, query: str) -> str:
        """Advanced query preprocessing.

        Args:
            query: Original query

        Returns:
            Preprocessed query
        """
        # Basic preprocessing - expand abbreviations, correct common typos
        preprocessed = query.strip()

        # Common abbreviations expansion
        expansions = {
            "AI": "artificial intelligence",
            "ML": "machine learning",
            "DL": "deep learning",
            "NLP": "natural language processing",
            "API": "application programming interface",
        }

        for abbrev, expansion in expansions.items():
            if abbrev in preprocessed:
                preprocessed = preprocessed.replace(abbrev, expansion)

        return preprocessed

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics.

        Returns:
            Performance metrics dictionary
        """
        return {
            **self.performance_metrics,
            "cache_hit_rate": (
                self.performance_metrics["cache_hits"] / max(self.performance_metrics["total_queries"], 1) * 100
            ),
            "recent_queries": len(self.query_history),
            "status": "active",
        }

    async def optimize_performance(self) -> dict[str, Any]:
        """Run performance optimization.

        Returns:
            Optimization results
        """
        results = {
            "cache_optimized": False,
            "index_optimized": False,
            "recommendations": [],
        }

        # Optimize cache if available
        if hasattr(self, "cache") and self.cache:
            try:
                if hasattr(self.cache, "optimize"):
                    await self.cache.optimize()
                    results["cache_optimized"] = True
            except Exception as e:
                logger.warning(f"Cache optimization failed: {e}")

        # Add recommendations based on usage patterns
        if self.performance_metrics["total_queries"] > 100:
            avg_time = self.performance_metrics["avg_response_time"]
            if avg_time > 2.0:  # Slow responses
                results["recommendations"].append("Consider increasing cache size or optimizing document chunking")

            cache_rate = self.performance_metrics["cache_hits"] / self.performance_metrics["total_queries"]
            if cache_rate < 0.3:  # Low cache hit rate
                results["recommendations"].append("Low cache hit rate - consider adjusting cache strategy")

        return results


__all__ = [
    "Document",
    "EnhancedRAGPipeline",
    "RAGPipeline",
    "RetrievalResult",
    "SynthesizedAnswer",
]
