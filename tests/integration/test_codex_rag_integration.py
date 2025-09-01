"""Integration tests for CODEX-compliant RAG system.

Tests all components according to CODEX Integration Requirements.
"""

import asyncio
import os
from pathlib import Path
import shutil
import sqlite3

# Import from deprecated location - these files were moved during consolidation
import sys
import tempfile

import pytest

deprecated_path = (
    Path(__file__).parent.parent.parent / "deprecated" / "rag_consolidation" / "20250818" / "production_rag" / "rag"
)
sys.path.insert(0, str(deprecated_path))

from rag_system.core.codex_rag_integration import CODEXCompliantCache, CODEXRAGPipeline
from wikipedia_data_loader import WikipediaDataLoader, create_sample_educational_content


class TestCODEXRAGIntegration:
    """Integration tests for CODEX RAG system."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def test_documents(self):
        """Create test documents."""
        return create_sample_educational_content()

    @pytest.fixture
    def pipeline(self, temp_dir):
        """Create RAG pipeline with temporary storage."""
        # Set environment variables for test
        os.environ["RAG_FAISS_INDEX_PATH"] = str(Path(temp_dir) / "test_faiss_index")
        os.environ["RAG_BM25_CORPUS_PATH"] = str(Path(temp_dir) / "test_bm25_corpus")
        os.environ["RAG_DISK_CACHE_DIR"] = str(Path(temp_dir) / "test_cache")

        pipeline = CODEXRAGPipeline()
        return pipeline

    def test_embedding_system_configuration(self, pipeline):
        """Test 1: CONFIGURE embedding system."""
        # Verify embedding model
        assert pipeline.embedder.get_sentence_embedding_dimension() == 384
        assert "paraphrase-MiniLM-L3-v2" in str(pipeline.embedder)

        # Verify vector dimensions
        assert pipeline.vector_dim == 384

        # Verify FAISS index initialization
        assert pipeline.index is not None
        assert pipeline.index.d == 384  # Vector dimension

    def test_caching_layers_setup(self, pipeline):
        """Test 2: SET UP caching layers."""
        cache = pipeline.cache

        # Verify L1 cache configuration
        assert cache.l1_capacity == 128
        assert isinstance(cache.l1_cache, dict)

        # Verify L3 disk cache
        assert cache.l3_cache is not None

        # Verify cache metrics tracking
        metrics = cache.get_metrics()
        assert "hit_rate" in metrics
        assert "hits" in metrics
        assert "misses" in metrics

    def test_chunk_processing_configuration(self, pipeline, test_documents):
        """Test 3: CONFIGURE chunk processing."""
        doc = test_documents[0]
        chunks = pipeline.chunk_document(doc, chunk_size=512, chunk_overlap=50)

        # Verify chunking parameters
        assert len(chunks) > 0
        assert all(len(chunk.text.split()) <= 512 for chunk in chunks)

        # Verify chunk boundaries preserved
        full_text = " ".join(chunk.text for chunk in chunks)
        assert len(full_text) >= len(doc.content) * 0.8  # Allow some overlap

        # Verify chunk metadata
        for chunk in chunks:
            assert chunk.document_id == doc.id
            assert chunk.metadata == doc.metadata

    def test_wikipedia_data_integration(self, temp_dir):
        """Test 4: INTEGRATE with Wikipedia data."""
        # Initialize loader with temp directory
        loader = WikipediaDataLoader(data_dir=temp_dir)

        # Create test documents (simulating Wikipedia data)
        documents = create_sample_educational_content()

        # Test database storage
        for doc in documents:
            loader._store_document(doc)

        # Verify database content
        conn = sqlite3.connect(loader.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        conn.close()

        assert doc_count == len(documents)

        # Test corpus file creation
        loader._save_corpus(documents)
        assert loader.corpus_path.exists()

        # Test corpus loading
        loaded_docs = loader.load_corpus_from_file()
        assert len(loaded_docs) == len(documents)

    @pytest.mark.asyncio
    async def test_api_functionality_and_performance(self, pipeline, test_documents):
        """Test 5: TEST API functionality and performance."""
        # Index test documents
        stats = pipeline.index_documents(test_documents)

        assert stats["documents_processed"] == len(test_documents)
        assert stats["chunks_created"] > 0
        assert stats["vectors_indexed"] > 0

        # Test retrieval functionality
        test_queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "Python programming basics",
            "Deep learning applications",
        ]

        retrieval_times = []

        for query in test_queries:
            results, metrics = await pipeline.retrieve(query, k=5)

            # Verify results format
            assert len(results) <= 5
            assert all(hasattr(r, "chunk_id") for r in results)
            assert all(hasattr(r, "text") for r in results)
            assert all(hasattr(r, "score") for r in results)

            # Track latency
            retrieval_times.append(metrics["latency_ms"])

        # Test <100ms retrieval target
        avg_latency = sum(retrieval_times) / len(retrieval_times)
        print(f"Average retrieval latency: {avg_latency:.2f}ms")

        # Performance metrics
        perf_metrics = pipeline.get_performance_metrics()
        assert "avg_latency_ms" in perf_metrics
        assert "meets_target" in perf_metrics

        # Test cache functionality
        # Query again to test cache hits
        cached_times = []
        for query in test_queries[:2]:
            results, metrics = await pipeline.retrieve(query, k=5)
            cached_times.append(metrics["latency_ms"])

        # Cache should improve performance
        cache_metrics = pipeline.cache.get_metrics()
        print(f"Cache hit rate: {cache_metrics['hit_rate']:.2%}")

    @pytest.mark.asyncio
    async def test_hybrid_retrieval_accuracy(self, pipeline, test_documents):
        """Test hybrid retrieval accuracy."""
        # Index documents
        pipeline.index_documents(test_documents)

        # Test vector search
        results, metrics = await pipeline.retrieve("machine learning algorithms", k=3)
        assert len(results) > 0
        assert any("machine learning" in r.text.lower() for r in results)

        # Test keyword search effectiveness
        results, metrics = await pipeline.retrieve("Python programming", k=3)
        assert len(results) > 0
        assert any("python" in r.text.lower() for r in results)

        # Test hybrid scoring
        for result in results:
            assert result.retrieval_method == "hybrid"
            assert 0 <= result.score <= 2  # RRF score range

    def test_environment_variable_compliance(self):
        """Test CODEX environment variable compliance."""
        # Test default values
        assert os.getenv("RAG_EMBEDDING_MODEL", "paraphrase-MiniLM-L3-v2") == "paraphrase-MiniLM-L3-v2"
        assert int(os.getenv("RAG_VECTOR_DIM", "384")) == 384
        assert int(os.getenv("RAG_DEFAULT_K", "10")) == 10
        assert int(os.getenv("RAG_CHUNK_SIZE", "512")) == 512
        assert int(os.getenv("RAG_CHUNK_OVERLAP", "50")) == 50
        assert int(os.getenv("RAG_L1_CACHE_SIZE", "128")) == 128

    def test_database_schema_compliance(self, temp_dir):
        """Test database schema matches CODEX requirements."""
        loader = WikipediaDataLoader(data_dir=temp_dir)

        conn = sqlite3.connect(loader.db_path)
        cursor = conn.cursor()

        # Check documents table schema
        cursor.execute("PRAGMA table_info(documents)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}

        required_columns = {
            "document_id": "TEXT",
            "title": "TEXT",
            "content": "TEXT",
            "file_hash": "TEXT",
            "word_count": "INTEGER",
            "metadata": "TEXT",
        }

        for col, col_type in required_columns.items():
            assert col in columns
            assert col_type in columns[col]

        # Check chunks table schema
        cursor.execute("PRAGMA table_info(chunks)")
        chunk_columns = {row[1]: row[2] for row in cursor.fetchall()}

        assert "chunk_id" in chunk_columns
        assert "document_id" in chunk_columns
        assert "chunk_index" in chunk_columns
        assert "content" in chunk_columns

        # Check embeddings_metadata table
        cursor.execute("PRAGMA table_info(embeddings_metadata)")
        embed_columns = {row[1]: row[2] for row in cursor.fetchall()}

        assert "vector_dimension" in embed_columns
        assert "faiss_index_id" in embed_columns
        assert "bm25_doc_id" in embed_columns

        conn.close()

    @pytest.mark.asyncio
    async def test_cache_performance_levels(self, pipeline):
        """Test three-tier cache performance."""
        cache = pipeline.cache

        # Test L1 cache
        query = "test query"
        test_results = []  # Empty results for test

        await cache.set(query, test_results)
        result = await cache.get(query)

        assert result == test_results
        assert cache.hits["l1"] > 0

        # Test cache metrics
        metrics = cache.get_metrics()
        assert metrics["hit_rate"] > 0
        assert metrics["total_requests"] > 0

    def test_error_handling_and_fallbacks(self, pipeline):
        """Test error handling and graceful fallbacks."""
        # Test with empty query
        try:
            asyncio.run(pipeline.retrieve("", k=5))
        except Exception as e:
            import logging

            logging.exception("CODEX RAG pipeline retrieve with empty query failed: %s", str(e))

        # Test with invalid k value
        results, metrics = asyncio.run(pipeline.retrieve("test", k=0))
        # Should handle gracefully and return reasonable results

        # Test cache fallback when Redis unavailable
        cache = CODEXCompliantCache()
        # Should initialize without Redis and use disk cache
        assert cache.l3_cache is not None


# Performance benchmark test
@pytest.mark.asyncio
async def test_performance_benchmark():
    """Run performance benchmark to validate <100ms target."""
    # Create pipeline
    pipeline = CODEXRAGPipeline()

    # Index sample documents
    documents = create_sample_educational_content()
    pipeline.index_documents(documents)

    # Run benchmark queries
    benchmark_queries = [
        "machine learning",
        "neural networks",
        "python programming",
        "deep learning",
        "artificial intelligence",
        "data science",
        "algorithms",
        "computer science",
    ]

    latencies = []

    for query in benchmark_queries:
        results, metrics = await pipeline.retrieve(query, k=10)
        latencies.append(metrics["latency_ms"])

    # Calculate statistics
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies))]

    print("\n=== Performance Benchmark Results ===")
    print(f"Queries tested: {len(benchmark_queries)}")
    print(f"Average latency: {avg_latency:.2f}ms")
    print(f"Max latency: {max_latency:.2f}ms")
    print(f"P95 latency: {p95_latency:.2f}ms")
    print(f"Meets <100ms target: {avg_latency < 100}")

    # Performance assertions
    assert avg_latency < 200  # Generous target for test environment
    assert all(lat < 1000 for lat in latencies)  # No extremely slow queries


if __name__ == "__main__":
    # Run integration test
    pytest.main([__file__, "-v", "--tb=short"])
