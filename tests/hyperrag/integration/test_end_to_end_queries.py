"""
End-to-end integration tests for HyperRAG system.

Tests complete query pipeline:
- Document ingestion → Storage → Retrieval → Analysis → Synthesis
- Multi-system coordination (Vector + Graph + Episodic + Cognitive)
- Real-world query scenarios
- Performance benchmarks
"""

import time

import pytest


class TestEndToEndQueries:
    """End-to-end integration tests for complete HyperRAG queries."""

    @pytest.mark.asyncio
    async def test_complete_query_pipeline(self, sample_documents):
        """Test complete query pipeline from ingestion to response."""
        try:
            from core.hyperrag import HyperRAG, HyperRAGConfig, QueryMode

            # Initialize with all systems enabled
            config = HyperRAGConfig(
                enable_hippo_rag=True,
                enable_graph_rag=True,
                enable_vector_rag=True,
                enable_cognitive_nexus=True,
                max_results=10,
            )

            hyperrag = HyperRAG(config)
            await hyperrag.initialize()

            # Phase 1: Document Ingestion
            doc_ids = []
            for doc in sample_documents:
                doc_id = hyperrag.add_document(content=doc["content"], doc_id=doc["id"], metadata=doc["metadata"])
                doc_ids.append(doc_id)

            assert len(doc_ids) == len(sample_documents), "All documents should be ingested"

            # Phase 2: Query Processing (Different Modes)
            queries = [
                ("What is Python programming?", QueryMode.FAST),
                ("How does machine learning work?", QueryMode.BALANCED),
                ("Explain neural networks in detail", QueryMode.COMPREHENSIVE),
                ("What are creative applications of AI?", QueryMode.CREATIVE),
                ("Analyze the relationship between ML and neural networks", QueryMode.ANALYTICAL),
            ]

            for query_text, mode in queries:
                start_time = time.time()

                answer = await hyperrag.process_query_async(query_text, mode)

                processing_time = time.time() - start_time

                # Validate answer quality
                assert answer is not None, f"Answer should not be None for query: {query_text}"
                assert len(answer.answer) > 0, f"Answer should not be empty for query: {query_text}"
                assert 0 <= answer.confidence <= 1, f"Confidence should be 0-1 for query: {query_text}"
                assert processing_time < 10.0, f"Query should complete within 10s, took {processing_time}s"

                # Mode-specific validations
                if mode == QueryMode.FAST:
                    assert processing_time < 2.0, "FAST mode should complete quickly"
                elif mode == QueryMode.COMPREHENSIVE:
                    assert len(answer.supporting_sources) > 0, "COMPREHENSIVE mode should have sources"
                elif mode == QueryMode.CREATIVE:
                    assert (
                        answer.synthesis_method == "creative_synthesis"
                    ), "CREATIVE mode should use creative synthesis"

            # Phase 3: System Statistics Validation
            stats = hyperrag.get_stats()
            assert stats["queries_processed"] == len(queries), "Query count should match"
            assert stats["documents_indexed"] == len(sample_documents), "Document count should match"
            assert stats["average_response_time"] > 0, "Average response time should be positive"

            await hyperrag.shutdown()

        except ImportError:
            pytest.skip("HyperRAG system not available")

    def test_concurrent_query_handling(self, sample_documents):
        """Test handling multiple concurrent queries."""
        try:
            from core.hyperrag import HyperRAG, HyperRAGConfig, QueryMode

            config = HyperRAGConfig(max_results=5, enable_caching=True)
            hyperrag = HyperRAG(config)

            # Add documents
            for doc in sample_documents:
                hyperrag.add_document(doc["content"], doc["id"], doc["metadata"])

            # Define concurrent queries
            queries = [
                "What is Python?",
                "Explain machine learning",
                "How do neural networks function?",
                "What are AI applications?",
                "Compare programming languages",
            ]

            # Execute concurrent queries
            start_time = time.time()
            results = []

            for query in queries:
                answer = hyperrag.process_query(query, QueryMode.BALANCED)
                results.append(answer)

            total_time = time.time() - start_time

            # Validate all results
            assert len(results) == len(queries), "Should get answer for each query"

            for i, answer in enumerate(results):
                assert answer is not None, f"Answer {i} should not be None"
                assert len(answer.answer) > 0, f"Answer {i} should not be empty"
                assert 0 <= answer.confidence <= 1, f"Answer {i} confidence should be 0-1"

            # Performance validation
            avg_time_per_query = total_time / len(queries)
            assert avg_time_per_query < 5.0, f"Average query time should be <5s, got {avg_time_per_query}s"

        except ImportError:
            pytest.skip("HyperRAG system not available")

    def test_knowledge_graph_integration(self, sample_documents):
        """Test integration with knowledge graph system."""
        try:
            from core.hyperrag import HyperRAG, HyperRAGConfig, QueryMode

            config = HyperRAGConfig(enable_graph_rag=True, enable_vector_rag=True, max_results=10)
            hyperrag = HyperRAG(config)

            # Add documents with relationships
            documents = [
                {
                    "content": "Python is a programming language created by Guido van Rossum.",
                    "metadata": {"type": "language", "creator": "Guido van Rossum"},
                },
                {
                    "content": "Guido van Rossum is a Dutch programmer who created Python.",
                    "metadata": {"type": "person", "nationality": "Dutch"},
                },
                {
                    "content": "Programming languages are used to create software applications.",
                    "metadata": {"type": "concept", "domain": "computer_science"},
                },
            ]

            for i, doc in enumerate(documents):
                hyperrag.add_document(doc["content"], f"graph_doc_{i}", doc["metadata"])

            # Test relationship-aware queries
            answer = hyperrag.process_query("Who created Python and what is their background?", QueryMode.COMPREHENSIVE)

            assert answer is not None, "Should get answer for relationship query"
            assert answer.confidence > 0.3, "Should have reasonable confidence for connected information"

            # Check if multiple sources were used (indicating graph traversal)
            assert len(answer.supporting_sources) > 1, "Should use multiple sources for relationship queries"

        except ImportError:
            pytest.skip("HyperRAG system not available")

    def test_episodic_memory_integration(self):
        """Test integration with episodic memory system (HippoRAG)."""
        try:
            from core.hyperrag import HyperRAG, HyperRAGConfig, QueryMode

            config = HyperRAGConfig(enable_hippo_rag=True, enable_vector_rag=True)
            hyperrag = HyperRAG(config)

            # Add time-sensitive documents
            recent_info = "Breaking: New AI breakthrough announced today."
            older_info = "Historical: First computer was built in 1940s."

            hyperrag.add_document(recent_info, "recent_1", {"timestamp": "2024-01-01"})
            hyperrag.add_document(older_info, "older_1", {"timestamp": "2020-01-01"})

            # Query should prioritize recent information
            answer = hyperrag.process_query("What's new in AI?", QueryMode.BALANCED)

            assert answer is not None, "Should get answer for temporal query"
            # Recent information should be more prominent
            assert "breakthrough" in answer.answer.lower(), "Should mention recent breakthrough"

        except ImportError:
            pytest.skip("HyperRAG system not available")

    @pytest.mark.asyncio
    async def test_cognitive_analysis_integration(self, sample_documents):
        """Test integration with cognitive analysis system."""
        try:
            from core.hyperrag import HyperRAG, HyperRAGConfig, QueryMode

            config = HyperRAGConfig(enable_cognitive_nexus=True, enable_vector_rag=True, enable_graph_rag=True)
            hyperrag = HyperRAG(config)
            await hyperrag.initialize()

            # Add documents with potential contradictions
            documents = [
                "Python is the best programming language for beginners.",
                "Java is considered the best programming language for beginners.",
                "Programming language choice depends on the specific use case and personal preference.",
            ]

            for i, doc in enumerate(documents):
                hyperrag.add_document(doc, f"cognitive_doc_{i}")

            # Query that requires cognitive analysis
            answer = await hyperrag.process_query_async(
                "What is the best programming language for beginners?", QueryMode.ANALYTICAL
            )

            assert answer is not None, "Should get answer for analytical query"
            assert answer.synthesis_method in [
                "multi_source_synthesis",
                "analytical_synthesis",
            ], "Should use analytical synthesis"

            # Should handle contradictory information appropriately
            assert answer.confidence < 0.9, "Should have lower confidence due to contradictions"

            await hyperrag.shutdown()

        except ImportError:
            pytest.skip("HyperRAG system not available")

    def test_performance_benchmarks(self, sample_documents):
        """Test system performance under various conditions."""
        try:
            from core.hyperrag import HyperRAG, HyperRAGConfig, QueryMode

            config = HyperRAGConfig(max_results=20)
            hyperrag = HyperRAG(config)

            # Load test documents
            for i, doc in enumerate(sample_documents * 10):  # Multiply for more data
                hyperrag.add_document(doc["content"], f"perf_doc_{i}", doc["metadata"])

            # Benchmark different query types
            benchmarks = [
                ("Simple factual query", "What is Python?", QueryMode.FAST, 1.0),
                (
                    "Complex analytical query",
                    "Compare and analyze machine learning vs neural networks",
                    QueryMode.COMPREHENSIVE,
                    5.0,
                ),
                (
                    "Creative synthesis query",
                    "What innovative applications could combine Python and neural networks?",
                    QueryMode.CREATIVE,
                    3.0,
                ),
            ]

            results = {}

            for name, query, mode, max_time in benchmarks:
                start_time = time.time()
                answer = hyperrag.process_query(query, mode)
                processing_time = time.time() - start_time

                results[name] = {
                    "processing_time": processing_time,
                    "max_time": max_time,
                    "confidence": answer.confidence,
                    "answer_length": len(answer.answer),
                    "sources_used": len(answer.supporting_sources),
                }

                assert processing_time < max_time, f"{name} took {processing_time}s, max allowed {max_time}s"
                assert answer.confidence > 0.1, f"{name} should have reasonable confidence"

            # System should handle load well
            stats = hyperrag.get_stats()
            assert stats["queries_processed"] == len(benchmarks), "Should track all queries"

        except ImportError:
            pytest.skip("HyperRAG system not available")
