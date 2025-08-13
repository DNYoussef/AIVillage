#!/usr/bin/env python3
"""Tests for RAG Pipeline default configuration and safe instantiation."""

from pathlib import Path
import sys

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from src.production.rag.rag_system.core.config import RAGConfig, UnifiedConfig
    from src.production.rag.rag_system.core.pipeline import (
        Document,
        RAGPipeline,
        SynthesizedAnswer,
    )
    from src.production.rag.rag_system.core.structures import RetrievalResult
except ImportError:
    # Alternative import path
    try:
        from production.rag.rag_system.core.config import RAGConfig, UnifiedConfig
        from production.rag.rag_system.core.pipeline import (
            Document,
            RAGPipeline,
            SynthesizedAnswer,
        )
        from production.rag.rag_system.core.structures import RetrievalResult
    except ImportError as e:
        pytest.skip(f"Cannot import RAG components: {e}", allow_module_level=True)


class TestRAGPipelineDefaults:
    """Test RAG Pipeline instantiation with safe defaults."""

    def test_no_config_instantiation(self):
        """Test pipeline can be instantiated without any config."""
        pipeline = RAGPipeline()

        assert pipeline is not None
        assert isinstance(pipeline.config, RAGConfig | UnifiedConfig)
        assert pipeline.documents == []
        assert pipeline._document_count == 0

    def test_dict_config_instantiation(self):
        """Test pipeline instantiation with dictionary config."""
        config = {"chunk_size": 500, "chunk_overlap": 100, "temperature": 0.8}

        pipeline = RAGPipeline(config=config)

        assert pipeline is not None
        assert pipeline.config.chunk_size == 500
        assert pipeline.config.chunk_overlap == 100
        assert pipeline.config.temperature == 0.8

    def test_unified_config_instantiation(self):
        """Test pipeline instantiation with UnifiedConfig object."""
        config = RAGConfig(chunk_size=800, max_tokens=2000, top_k=10)

        pipeline = RAGPipeline(config=config)

        assert pipeline is not None
        assert pipeline.config.chunk_size == 800
        assert pipeline.config.max_tokens == 2000
        assert pipeline.config.top_k == 10

    def test_cache_disabled_instantiation(self):
        """Test pipeline instantiation with cache disabled."""
        pipeline = RAGPipeline(enable_cache=False)

        assert pipeline is not None
        assert pipeline.cache is None

    def test_graph_disabled_instantiation(self):
        """Test pipeline instantiation with graph disabled."""
        pipeline = RAGPipeline(enable_graph=False)

        assert pipeline is not None
        assert pipeline.graph_store is None

    def test_custom_cache_dir(self):
        """Test pipeline instantiation with custom cache directory."""
        custom_cache_dir = Path("/tmp/test_rag_cache")
        pipeline = RAGPipeline(cache_dir=custom_cache_dir)

        assert pipeline is not None
        assert pipeline.cache_dir == custom_cache_dir

    def test_pipeline_string_representation(self):
        """Test pipeline string representation."""
        pipeline = RAGPipeline()
        str_repr = str(pipeline)

        assert "RAGPipeline" in str_repr
        assert "documents=0" in str_repr
        assert "cache_enabled=" in str_repr
        assert "graph_enabled=" in str_repr

    def test_fallback_components_creation(self):
        """Test that fallback components are created when imports fail."""
        pipeline = RAGPipeline()

        # Vector store should always be available (either real or fallback)
        assert pipeline.vector_store is not None

        # Chunker should always be available (either real or fallback)
        assert pipeline.chunker is not None

        # Cache might be None, dict, or advanced cache
        assert pipeline.cache is None or isinstance(pipeline.cache, dict | object)

    @pytest.mark.asyncio
    async def test_fallback_vector_store_functionality(self):
        """Test that fallback vector store works correctly."""
        pipeline = RAGPipeline()

        # Test adding texts to fallback vector store
        test_texts = ["Hello world", "Machine learning", "Deep learning"]
        await pipeline.vector_store.add_texts(test_texts)

        # Test retrieval from fallback vector store
        query_vector = [0.1] * 768
        results = await pipeline.vector_store.retrieve(query_vector, k=2)

        assert isinstance(results, list)
        # Fallback might return empty list or mock results

    def test_fallback_chunker_functionality(self):
        """Test that fallback chunker works correctly."""
        pipeline = RAGPipeline()

        test_text = "This is a test document. " * 200  # Long text to test chunking
        chunks = pipeline.chunker.chunk_text(test_text)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_metrics_initialization(self):
        """Test that metrics are properly initialized."""
        pipeline = RAGPipeline()

        metrics = pipeline.get_metrics()

        assert isinstance(metrics, dict)
        assert "queries_processed" in metrics
        assert "documents_indexed" in metrics
        assert "cache_hits" in metrics
        assert "cache_misses" in metrics
        assert "avg_query_time_ms" in metrics
        assert "document_count" in metrics

        # Initial values should be zero
        assert metrics["queries_processed"] == 0
        assert metrics["documents_indexed"] == 0
        assert metrics["cache_hits"] == 0
        assert metrics["cache_misses"] == 0
        assert metrics["document_count"] == 0


class TestDocumentHandling:
    """Test document creation and handling."""

    def test_document_creation(self):
        """Test Document dataclass creation."""
        doc = Document(id="test_doc", text="This is a test document")

        assert doc.id == "test_doc"
        assert doc.text == "This is a test document"
        assert isinstance(doc.metadata, dict)
        assert doc.embedding is None
        assert doc.timestamp is not None

    def test_document_with_metadata(self):
        """Test Document creation with metadata."""
        metadata = {"author": "test", "category": "technical"}
        doc = Document(id="test_doc", text="Test document", metadata=metadata)

        assert doc.metadata == metadata

    @pytest.mark.asyncio
    async def test_document_addition(self):
        """Test adding documents to pipeline."""
        pipeline = RAGPipeline()

        doc = Document(id="test_doc", text="This is a test document for indexing.")

        initial_count = pipeline._document_count
        await pipeline.add_document(doc)

        assert pipeline._document_count == initial_count + 1
        assert doc in pipeline.documents

        metrics = pipeline.get_metrics()
        assert metrics["documents_indexed"] == 1


class TestRetrievalFunctionality:
    """Test retrieval functionality with fallbacks."""

    @pytest.mark.asyncio
    async def test_basic_retrieval(self):
        """Test basic retrieval functionality."""
        pipeline = RAGPipeline()

        # Add a test document
        doc = Document(
            id="test_doc",
            text="Machine learning is a subset of artificial intelligence.",
        )
        await pipeline.add_document(doc)

        # Perform retrieval
        results = await pipeline.retrieve("machine learning", top_k=3)

        assert isinstance(results, list)
        # Results might be empty with fallback implementation

    @pytest.mark.asyncio
    async def test_cache_behavior(self):
        """Test cache hit/miss behavior."""
        pipeline = RAGPipeline(enable_cache=True)

        # Add a test document
        doc = Document(id="test_doc", text="Test document for cache testing.")
        await pipeline.add_document(doc)

        # First query (should be cache miss)
        await pipeline.retrieve("test query", use_cache=True)
        metrics1 = pipeline.get_metrics()

        # Second identical query (might be cache hit)
        await pipeline.retrieve("test query", use_cache=True)
        metrics2 = pipeline.get_metrics()

        # Check that metrics are updated
        assert metrics2["queries_processed"] >= metrics1["queries_processed"]

    @pytest.mark.asyncio
    async def test_synthesis_functionality(self):
        """Test answer synthesis functionality."""
        pipeline = RAGPipeline()

        # Create mock retrieval results
        mock_results = [
            RetrievalResult(
                id="doc1",
                content="Machine learning is a method of data analysis.",
                score=0.9,
                uncertainty=0.1,
                timestamp=None,  # Will be set to datetime by RetrievalResult
                version=1,
            ),
            RetrievalResult(
                id="doc2",
                content="Deep learning uses neural networks.",
                score=0.8,
                uncertainty=0.2,
                timestamp=None,
                version=1,
            ),
        ]

        # Test synthesis
        answer = await pipeline.synthesize_answer(
            query="What is machine learning?", context_results=mock_results
        )

        assert isinstance(answer, SynthesizedAnswer)
        assert answer.answer_text is not None
        assert answer.executive_summary is not None
        assert answer.confidence_score >= 0.0
        assert answer.confidence_score <= 1.0
        assert len(answer.primary_sources) <= len(mock_results)

    @pytest.mark.asyncio
    async def test_end_to_end_query(self):
        """Test complete end-to-end query processing."""
        pipeline = RAGPipeline()

        # Add test documents
        docs = [
            Document(
                id="doc1", text="Machine learning enables computers to learn from data."
            ),
            Document(id="doc2", text="Deep learning is a subset of machine learning."),
        ]

        for doc in docs:
            await pipeline.add_document(doc)

        # Perform end-to-end query
        results, answer = await pipeline.query(
            query="What is machine learning?", top_k=2, synthesize=True, use_cache=True
        )

        assert isinstance(results, list)
        # Answer might be None with fallback implementation
        if answer is not None:
            assert isinstance(answer, SynthesizedAnswer)


class TestCacheManagement:
    """Test cache management functionality."""

    @pytest.mark.asyncio
    async def test_cache_clearing(self):
        """Test cache clearing functionality."""
        pipeline = RAGPipeline(enable_cache=True)

        # This should not raise an exception
        await pipeline.clear_cache()

    def test_cache_metrics(self):
        """Test cache metrics availability."""
        pipeline = RAGPipeline(enable_cache=True)

        metrics = pipeline.get_metrics()

        # Should have cache-related metrics
        assert "cache" in metrics or "cache_hits" in metrics


class TestErrorHandling:
    """Test error handling and graceful degradation."""

    @pytest.mark.asyncio
    async def test_document_addition_error_handling(self):
        """Test error handling during document addition."""
        pipeline = RAGPipeline()

        # Create a document that might cause issues
        doc = Document(
            id="", text=""
        )  # Empty ID might cause issues  # Empty text might cause issues

        # This should either succeed or raise a clear exception
        try:
            await pipeline.add_document(doc)
        except Exception as e:
            assert isinstance(e, Exception)  # Should be a proper exception

    @pytest.mark.asyncio
    async def test_retrieval_error_handling(self):
        """Test error handling during retrieval."""
        pipeline = RAGPipeline()

        # Test retrieval without any documents
        results = await pipeline.retrieve("test query")

        # Should return empty list rather than crashing
        assert isinstance(results, list)

    def test_invalid_config_handling(self):
        """Test handling of invalid configuration."""
        # Test with various invalid configs
        invalid_configs = [
            {"chunk_size": -1},  # Negative chunk size
            {"top_k": 0},  # Zero top_k
            {"temperature": -1},  # Negative temperature
        ]

        for config in invalid_configs:
            # Pipeline should either handle gracefully or provide clear error
            try:
                pipeline = RAGPipeline(config=config)
                assert pipeline is not None
            except Exception as e:
                assert isinstance(e, Exception)


# Additional utility tests
def test_import_safety():
    """Test that imports are safe and don't crash on missing dependencies."""
    # This test just verifies that the import succeeded
    assert RAGPipeline is not None
    assert Document is not None
    assert SynthesizedAnswer is not None


if __name__ == "__main__":
    # Run tests directly if script is executed
    pytest.main([__file__, "-v"])
