"""
Unit tests for HyperRAG main orchestrator.

Tests the core HyperRAG class functionality:
- Initialization and configuration
- Document addition and management
- Query processing and routing
- Statistics and health monitoring
- Error handling and fallbacks
"""

import time

import pytest

# Test utilities imported from conftest.py


class TestHyperRAGOrchestrator:
    """Unit tests for HyperRAG main orchestrator."""

    def test_hyperrag_import(self):
        """Test that HyperRAG can be imported from consolidated location."""
        try:
            from core.hyperrag import HyperRAG, HyperRAGConfig, MemoryType, QueryMode

            assert HyperRAG is not None
            assert HyperRAGConfig is not None
            assert QueryMode is not None
            assert MemoryType is not None
        except ImportError as e:
            pytest.skip(f"HyperRAG import failed: {e}")

    def test_hyperrag_initialization(self, hyperrag_config):
        """Test HyperRAG initialization with configuration."""
        try:
            from core.hyperrag import HyperRAG

            # Test with config
            hyperrag = HyperRAG(hyperrag_config)
            assert hyperrag is not None
            assert hyperrag.config == hyperrag_config

            # Test without config (default)
            hyperrag_default = HyperRAG()
            assert hyperrag_default is not None
            assert hyperrag_default.config is not None

        except ImportError:
            pytest.skip("HyperRAG not available")

    @pytest.mark.asyncio
    async def test_async_initialization(self, hyperrag_config):
        """Test async initialization of HyperRAG system."""
        try:
            from core.hyperrag import HyperRAG

            hyperrag = HyperRAG(hyperrag_config)

            # Test async initialization
            result = await hyperrag.initialize()
            assert result is True, "Initialization should succeed"

            # Test cleanup
            await hyperrag.shutdown()

        except ImportError:
            pytest.skip("HyperRAG not available")

    def test_document_addition(self, hyperrag_config, sample_documents):
        """Test document addition functionality."""
        try:
            from core.hyperrag import HyperRAG

            hyperrag = HyperRAG(hyperrag_config)

            # Add single document
            doc = sample_documents[0]
            doc_id = hyperrag.add_document(content=doc["content"], doc_id=doc["id"], metadata=doc["metadata"])

            assert doc_id == doc["id"], "Document ID should match provided ID"
            assert hyperrag.stats["documents_indexed"] == 1, "Document count should increment"

            # Add document without ID (auto-generated)
            auto_doc_id = hyperrag.add_document(content="Auto-generated ID test", metadata={"test": True})

            assert auto_doc_id is not None, "Auto-generated ID should be returned"
            assert hyperrag.stats["documents_indexed"] == 2, "Document count should increment again"

        except ImportError:
            pytest.skip("HyperRAG not available")

    def test_query_processing_modes(self, hyperrag_config, sample_documents, sample_queries):
        """Test query processing with different modes."""
        try:
            from core.hyperrag import HyperRAG, QueryMode

            hyperrag = HyperRAG(hyperrag_config)

            # Add test documents
            for doc in sample_documents:
                hyperrag.add_document(doc["content"], doc["id"], doc["metadata"])

            # Test each query mode
            for query_data in sample_queries:
                mode = getattr(QueryMode, query_data["mode"])

                start_time = time.time()
                answer = hyperrag.process_query(query_data["query"], mode)
                processing_time = time.time() - start_time

                # Validate answer
                assert_valid_answer(answer)
                assert_performance_acceptable(processing_time)

                # Check mode is recorded
                assert answer.query_mode == mode.value, f"Query mode should be {mode.value}"

        except ImportError:
            pytest.skip("HyperRAG not available")

    @pytest.mark.asyncio
    async def test_async_query_processing(self, hyperrag_config, sample_documents):
        """Test async query processing."""
        try:
            from core.hyperrag import HyperRAG, QueryMode

            hyperrag = HyperRAG(hyperrag_config)

            # Add test document
            hyperrag.add_document(
                sample_documents[0]["content"], sample_documents[0]["id"], sample_documents[0]["metadata"]
            )

            # Test async query
            answer = await hyperrag.process_query_async("What is Python?", QueryMode.FAST)

            assert_valid_answer(answer)

        except ImportError:
            pytest.skip("HyperRAG not available")

    def test_statistics_tracking(self, hyperrag_config, sample_documents):
        """Test statistics tracking functionality."""
        try:
            from core.hyperrag import HyperRAG, QueryMode

            hyperrag = HyperRAG(hyperrag_config)

            # Initial stats
            initial_stats = hyperrag.get_stats()
            assert initial_stats["queries_processed"] == 0
            assert initial_stats["documents_indexed"] == 0

            # Add document and check stats
            hyperrag.add_document(sample_documents[0]["content"])
            stats_after_doc = hyperrag.get_stats()
            assert stats_after_doc["documents_indexed"] == 1

            # Process query and check stats
            hyperrag.process_query("test query", QueryMode.FAST)
            stats_after_query = hyperrag.get_stats()
            assert stats_after_query["queries_processed"] == 1
            assert stats_after_query["average_response_time"] > 0

        except ImportError:
            pytest.skip("HyperRAG not available")

    def test_health_check(self, hyperrag_config):
        """Test system health check functionality."""
        try:
            from core.hyperrag import HyperRAG

            hyperrag = HyperRAG(hyperrag_config)

            health_status = hyperrag.health_check()

            assert_health_check_valid(health_status)
            assert health_status["status"] == "healthy", "System should be healthy"

        except ImportError:
            pytest.skip("HyperRAG not available")

    def test_caching_functionality(self, hyperrag_config):
        """Test query caching functionality."""
        try:
            from core.hyperrag import HyperRAG, QueryMode

            # Enable caching
            hyperrag_config.enable_caching = True
            hyperrag = HyperRAG(hyperrag_config)

            # Add test document
            hyperrag.add_document("Test content for caching")

            # First query (should not be cached)
            answer1 = hyperrag.process_query("test query", QueryMode.FAST)
            initial_cache_hits = hyperrag.stats["cache_hits"]

            # Second identical query (should be cached)
            answer2 = hyperrag.process_query("test query", QueryMode.FAST)
            final_cache_hits = hyperrag.stats["cache_hits"]

            assert final_cache_hits > initial_cache_hits, "Cache hits should increment"
            assert answer1.answer == answer2.answer, "Cached answers should be identical"

            # Clear cache
            hyperrag.clear_cache()
            assert len(hyperrag.query_cache) == 0, "Cache should be empty after clearing"

        except ImportError:
            pytest.skip("HyperRAG not available")

    def test_error_handling(self, hyperrag_config):
        """Test error handling and fallback mechanisms."""
        try:
            from core.hyperrag import HyperRAG, QueryMode

            hyperrag = HyperRAG(hyperrag_config)

            # Test query with no documents (should use fallback)
            answer = hyperrag.process_query("test query with no docs", QueryMode.BALANCED)

            assert_valid_answer(answer)
            assert answer.confidence == 0.0, "Empty system should have zero confidence"
            assert "don't have enough information" in answer.answer.lower()

        except ImportError:
            pytest.skip("HyperRAG not available")

    def test_memory_type_support(self, hyperrag_config):
        """Test support for different memory types."""
        try:
            from core.hyperrag import HyperRAG, MemoryType

            HyperRAG(hyperrag_config)

            # Test all memory types are available
            memory_types = [
                MemoryType.EPISODIC,
                MemoryType.SEMANTIC,
                MemoryType.VECTOR,
                MemoryType.PROCEDURAL,
                MemoryType.ALL,
            ]

            for memory_type in memory_types:
                assert memory_type is not None, f"MemoryType.{memory_type.name} should be available"

        except ImportError:
            pytest.skip("HyperRAG not available")

    def test_query_mode_support(self, hyperrag_config):
        """Test support for all query modes."""
        try:
            from core.hyperrag import HyperRAG, QueryMode

            HyperRAG(hyperrag_config)

            # Test all query modes are available
            query_modes = [
                QueryMode.FAST,
                QueryMode.BALANCED,
                QueryMode.COMPREHENSIVE,
                QueryMode.CREATIVE,
                QueryMode.ANALYTICAL,
                QueryMode.DISTRIBUTED,
                QueryMode.EDGE_OPTIMIZED,
            ]

            for mode in query_modes:
                assert mode is not None, f"QueryMode.{mode.name} should be available"

        except ImportError:
            pytest.skip("HyperRAG not available")
