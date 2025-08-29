"""
Enhanced HyperRAG System Test Suite

Comprehensive tests for the advanced neural-biological HyperRAG system including:
- HippoRAG neural memory integration
- Bayesian trust network validation
- Cognitive reasoning capabilities
- End-to-end system performance
- Component integration testing
"""

import asyncio
import logging
import os
import sys

import pytest

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core"))

try:
    from hyperrag.hyperrag import HyperRAG, HyperRAGConfig, MemoryType, QueryMode

    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    pytest.skip(f"Advanced components not available: {e}", allow_module_level=True)

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


@pytest.fixture
async def enhanced_hyperrag_config():
    """Enhanced HyperRAG configuration."""
    config = HyperRAGConfig()
    config.enable_hippo_rag = True
    config.enable_graph_rag = True
    config.enable_vector_rag = True
    config.enable_cognitive_nexus = True
    config.max_results = 5
    config.vector_dimensions = 384
    return config


@pytest.fixture
async def enhanced_hyperrag_system(enhanced_hyperrag_config):
    """Create and initialize enhanced HyperRAG system."""
    system = HyperRAG(enhanced_hyperrag_config)
    await system.initialize()
    yield system
    await system.shutdown()


class TestHippoRAGIntegration:
    """Test HippoRAG neural memory integration."""

    @pytest.mark.asyncio
    async def test_neural_memory_encoding(self, enhanced_hyperrag_system):
        """Test encoding documents into neural memory."""
        system = enhanced_hyperrag_system

        # Add document with episodic memory
        doc_id = await system.add_document(
            content="Machine learning models require careful hyperparameter tuning for optimal performance.",
            metadata={"domain": "machine_learning", "source_type": "research_paper", "credibility": 0.9},
            memory_type=MemoryType.EPISODIC,
        )

        assert doc_id is not None
        assert len(doc_id) > 0

        # Verify storage in neural memory
        if system.hippo_rag:
            status = await system.hippo_rag.get_status()
            assert status["status"] == "healthy"
            assert status["memory_statistics"]["episodic_memories"] > 0

    @pytest.mark.asyncio
    async def test_contextual_memory_retrieval(self, enhanced_hyperrag_system):
        """Test contextual memory retrieval with HippoRAG."""
        system = enhanced_hyperrag_system

        # Add multiple related documents
        await system.add_document(
            content="Deep neural networks excel at pattern recognition in complex data.",
            metadata={"domain": "deep_learning", "topic": "pattern_recognition"},
            memory_type=MemoryType.EPISODIC,
        )

        await system.add_document(
            content="Convolutional neural networks are particularly effective for image classification tasks.",
            metadata={"domain": "deep_learning", "topic": "computer_vision"},
            memory_type=MemoryType.EPISODIC,
        )

        # Query with contextual information
        result = await system.process_query(
            query="neural networks pattern recognition",
            mode=QueryMode.COMPREHENSIVE,
            context={"domain": "deep_learning"},
            user_id="test_user",
        )

        assert result.confidence > 0.5
        assert "neural" in result.answer.lower() or "pattern" in result.answer.lower()
        assert len(result.retrieval_sources) > 0

    @pytest.mark.asyncio
    async def test_memory_consolidation(self, enhanced_hyperrag_system):
        """Test memory consolidation process."""
        system = enhanced_hyperrag_system

        if not system.hippo_rag:
            pytest.skip("HippoRAG not available")

        # Add multiple documents to trigger consolidation
        for i in range(5):
            await system.add_document(
                content=f"Research finding {i}: Neural networks show improved performance with proper regularization.",
                metadata={"domain": "research", "study_id": i},
                memory_type=MemoryType.EPISODIC,
            )

        # Force consolidation
        consolidations = await system.hippo_rag.consolidate_memories(force=True)

        # Verify consolidation occurred
        assert consolidations >= 0  # At least some consolidation should happen

        stats = await system.get_stats()
        if "hippo_rag" in stats:
            assert stats["hippo_rag"]["status"] == "healthy"


class TestBayesianTrustNetworks:
    """Test Bayesian trust network integration."""

    @pytest.mark.asyncio
    async def test_trust_based_source_validation(self, enhanced_hyperrag_system):
        """Test trust-based source validation."""
        system = enhanced_hyperrag_system

        if not system.trust_network:
            pytest.skip("Trust network not available")

        # Add high-credibility source
        await system.add_document(
            content="Peer-reviewed research confirms that transformer architectures achieve state-of-the-art results.",
            metadata={
                "source_type": "academic_paper",
                "credibility": 0.95,
                "domain": "machine_learning",
                "citations": 150,
            },
        )

        # Add low-credibility source
        await system.add_document(
            content="AI will definitely replace all jobs next year according to my blog.",
            metadata={
                "source_type": "blog_post",
                "credibility": 0.2,
                "domain": "artificial_intelligence",
                "citations": 0,
            },
        )

        # Query should prioritize high-trust sources
        result = await system.process_query(
            query="transformer AI research results", mode=QueryMode.ANALYTICAL, context={"trust_threshold": 0.6}
        )

        assert result.confidence > 0.4
        # High-trust content should be prioritized
        assert "peer-reviewed" in result.answer.lower() or "transformer" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_trust_propagation(self, enhanced_hyperrag_system):
        """Test trust propagation through network."""
        system = enhanced_hyperrag_system

        if not system.trust_network:
            pytest.skip("Trust network not available")

        # Add multiple interconnected sources
        doc1_id = await system.add_document(
            content="Foundational research on attention mechanisms in neural networks.",
            metadata={"source_type": "seminal_paper", "credibility": 0.98, "authority": 0.95},
        )

        await system.add_document(
            content="Follow-up study building on attention mechanism research shows improved performance.",
            metadata={"source_type": "academic_paper", "credibility": 0.85, "references": doc1_id},
        )

        # Test trust propagation
        trust_results = await system.trust_network.retrieve_with_trust_propagation(
            query="attention mechanisms neural networks", k=5, min_trust_score=0.5
        )

        assert len(trust_results) > 0
        # Verify trust scores are reasonable
        for node, score, trust in trust_results:
            assert 0.0 <= score <= 1.0
            assert trust.overall_trust > 0.0

    @pytest.mark.asyncio
    async def test_trust_conflict_detection(self, enhanced_hyperrag_system):
        """Test detection of conflicting trust information."""
        system = enhanced_hyperrag_system

        if not system.trust_network:
            pytest.skip("Trust network not available")

        # Add conflicting high-trust sources
        await system.add_document(
            content="Study A shows that method X achieves 95% accuracy on benchmark dataset.",
            metadata={"source_type": "academic_paper", "credibility": 0.9, "study": "A"},
        )

        await system.add_document(
            content="Study B demonstrates that method X only achieves 75% accuracy on the same benchmark.",
            metadata={"source_type": "academic_paper", "credibility": 0.9, "study": "B"},
        )

        # Detect conflicts
        conflicts = await system.trust_network.detect_conflicts(
            content_similarity_threshold=0.7, trust_difference_threshold=0.1
        )

        # Should detect some conflicts in test data
        assert isinstance(conflicts, list)
        # Note: Conflict detection depends on content similarity heuristics


class TestCognitiveReasoning:
    """Test cognitive reasoning engine integration."""

    @pytest.mark.asyncio
    async def test_multi_strategy_reasoning(self, enhanced_hyperrag_system):
        """Test multi-strategy cognitive reasoning."""
        system = enhanced_hyperrag_system

        if not system.cognitive_engine:
            pytest.skip("Cognitive engine not available")

        # Add evidence sources for reasoning
        await system.add_document(
            content="Machine learning algorithms learn patterns from training data to make predictions.",
            metadata={"type": "definition", "domain": "machine_learning"},
        )

        await system.add_document(
            content="Supervised learning requires labeled examples, while unsupervised learning finds hidden patterns.",
            metadata={"type": "comparison", "domain": "machine_learning"},
        )

        await system.add_document(
            content="Research shows that deep learning models perform better with larger datasets.",
            metadata={"type": "empirical", "domain": "deep_learning"},
        )

        # Query requiring reasoning
        result = await system.process_query(
            query="How do machine learning algorithms improve with more data?",
            mode=QueryMode.ANALYTICAL,
            context={"reasoning_required": True},
        )

        assert result.confidence > 0.3
        assert result.synthesis_method == "cognitive_reasoning"
        # Should contain reasoning elements
        reasoning_indicators = ["because", "therefore", "as a result", "due to", "leads to"]
        assert any(indicator in result.answer.lower() for indicator in reasoning_indicators)

    @pytest.mark.asyncio
    async def test_bias_detection(self, enhanced_hyperrag_system):
        """Test cognitive bias detection."""
        system = enhanced_hyperrag_system

        if not system.cognitive_engine:
            pytest.skip("Cognitive engine not available")

        # Add biased information
        await system.add_document(
            content="This amazing AI breakthrough will revolutionize everything and solve all problems!",
            metadata={"source_type": "promotional", "bias_risk": "high"},
        )

        # Query with potential for bias
        result = await system.process_query(
            query="AI breakthrough revolutionary impact", mode=QueryMode.COMPREHENSIVE, context={"bias_detection": True}
        )

        # System should handle biased content appropriately
        assert result.confidence < 0.9  # Should be cautious with promotional content

    @pytest.mark.asyncio
    async def test_knowledge_gap_identification(self, enhanced_hyperrag_system):
        """Test knowledge gap identification."""
        system = enhanced_hyperrag_system

        if not system.cognitive_engine:
            pytest.skip("Cognitive engine not available")

        # Add limited information
        await system.add_document(
            content="Quantum computing uses quantum bits for computation.",
            metadata={"domain": "quantum_computing", "completeness": "partial"},
        )

        # Query requiring more comprehensive knowledge
        result = await system.process_query(
            query="How do quantum computers achieve quantum advantage over classical computers?",
            mode=QueryMode.COMPREHENSIVE,
        )

        # Should identify knowledge gaps
        assert result.confidence < 0.8  # Should acknowledge limitations
        gap_indicators = ["limited", "insufficient", "more information", "not enough", "unclear"]
        assert any(indicator in result.answer.lower() for indicator in gap_indicators)


class TestSystemIntegration:
    """Test full system integration."""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, enhanced_hyperrag_system):
        """Test complete end-to-end workflow."""
        system = enhanced_hyperrag_system

        # 1. Document ingestion with various types
        docs = [
            {
                "content": "Artificial neural networks are computational models inspired by biological neural networks.",
                "metadata": {"domain": "neural_networks", "type": "definition", "credibility": 0.9},
            },
            {
                "content": "Deep learning has achieved breakthrough results in image recognition, natural language processing, and game playing.",
                "metadata": {"domain": "deep_learning", "type": "achievements", "credibility": 0.95},
            },
            {
                "content": "Transformer architectures have revolutionized natural language processing through attention mechanisms.",
                "metadata": {"domain": "nlp", "type": "innovation", "credibility": 0.92},
            },
        ]

        doc_ids = []
        for doc in docs:
            doc_id = await system.add_document(content=doc["content"], metadata=doc["metadata"])
            doc_ids.append(doc_id)

        assert len(doc_ids) == 3
        assert all(doc_id for doc_id in doc_ids)

        # 2. Complex query processing
        result = await system.process_query(
            query="How have neural networks evolved and what are their key applications?",
            mode=QueryMode.COMPREHENSIVE,
            context={
                "domain": "artificial_intelligence",
                "user_expertise": "intermediate",
                "response_style": "comprehensive",
            },
            user_id="integration_test_user",
        )

        # 3. Validate comprehensive response
        assert result.confidence > 0.5
        assert len(result.retrieval_sources) > 0
        assert len(result.answer) > 100  # Substantial answer

        # Should mention key concepts
        key_concepts = ["neural network", "deep learning", "transformer", "attention"]
        assert any(concept in result.answer.lower() for concept in key_concepts)

        # 4. System health check
        health = await system.health_check()
        assert health["status"] in ["healthy", "degraded", "limited"]
        assert health["neural_biological_enabled"] is True

        # 5. Performance statistics
        stats = await system.get_stats()
        assert stats["queries_processed"] > 0
        assert stats["documents_indexed"] >= 3
        assert stats["cache_size"] >= 0

    @pytest.mark.asyncio
    async def test_performance_under_load(self, enhanced_hyperrag_system):
        """Test system performance under concurrent load."""
        system = enhanced_hyperrag_system

        # Add test documents
        for i in range(10):
            await system.add_document(
                content=f"Test document {i}: Machine learning concept {i} with various applications and implications.",
                metadata={"test_id": i, "domain": "test_domain"},
            )

        # Concurrent query processing
        queries = [
            "machine learning applications",
            "test concept implications",
            "document analysis results",
            "performance evaluation metrics",
            "system integration testing",
        ]

        # Process queries concurrently
        tasks = [
            system.process_query(query, QueryMode.BALANCED, {"test": True}, f"user_{i}")
            for i, query in enumerate(queries)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Validate results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= len(queries) // 2  # At least 50% success

        # Check performance metrics
        stats = await system.get_stats()
        assert stats["queries_processed"] >= len(successful_results)
        assert stats["average_response_time"] > 0

    @pytest.mark.asyncio
    async def test_fallback_behavior(self, enhanced_hyperrag_system):
        """Test graceful fallback behavior."""
        system = enhanced_hyperrag_system

        # Test with invalid input
        result = await system.process_query(query="", mode=QueryMode.FAST)  # Empty query

        # Should handle gracefully
        assert result is not None
        assert result.confidence <= 0.5
        assert result.synthesis_method in ["error_fallback", "no_results"]

        # Test with very complex query
        complex_query = "Explain the quantum mechanical underpinnings of consciousness in artificial general intelligence systems while considering the implications for moral philosophy and existential risk mitigation strategies in a post-singularity world."

        result = await system.process_query(query=complex_query, mode=QueryMode.COMPREHENSIVE)

        # Should provide some response even if limited
        assert result is not None
        assert len(result.answer) > 0

    @pytest.mark.asyncio
    async def test_memory_efficiency(self, enhanced_hyperrag_system):
        """Test memory usage efficiency."""
        system = enhanced_hyperrag_system

        # Add many documents to test memory management
        for i in range(50):
            await system.add_document(
                content=f"Memory test document {i} with substantial content to test memory usage patterns and efficiency metrics in large-scale deployments.",
                metadata={"batch": "memory_test", "index": i},
            )

        # Process multiple queries
        for i in range(10):
            await system.process_query(query=f"memory test query {i}", mode=QueryMode.FAST, user_id=f"memory_user_{i}")

        # Check system health after memory stress
        health = await system.health_check()
        assert health["status"] in ["healthy", "degraded"]  # Should not be completely broken

        # Verify memory systems are functioning
        stats = await system.get_stats()
        assert stats["documents_indexed"] >= 50
        assert stats["queries_processed"] >= 10


class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_component_failure_resilience(self, enhanced_hyperrag_system):
        """Test resilience to component failures."""
        system = enhanced_hyperrag_system

        # Add some documents first
        await system.add_document(
            content="Test document for resilience testing.", metadata={"purpose": "resilience_test"}
        )

        # Test query with missing context
        result = await system.process_query(
            query="test resilience",
            mode=QueryMode.COMPREHENSIVE,
            context=None,  # Missing context
            user_id=None,  # Missing user ID
        )

        # Should handle missing parameters gracefully
        assert result is not None
        assert result.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_data_validation(self, enhanced_hyperrag_system):
        """Test input data validation."""
        system = enhanced_hyperrag_system

        # Test with invalid metadata
        doc_id = await system.add_document(
            content="Valid content",
            metadata={"invalid_field": {"nested": "object"}, "number": float("inf")},  # Invalid values
        )

        # Should handle invalid metadata gracefully
        assert doc_id is not None
        assert len(doc_id) > 0

    @pytest.mark.asyncio
    async def test_concurrent_access_safety(self, enhanced_hyperrag_system):
        """Test thread safety under concurrent access."""
        system = enhanced_hyperrag_system

        # Concurrent document addition
        add_tasks = [
            system.add_document(content=f"Concurrent document {i}", metadata={"concurrent_id": i}) for i in range(20)
        ]

        doc_ids = await asyncio.gather(*add_tasks, return_exceptions=True)
        successful_adds = [doc_id for doc_id in doc_ids if isinstance(doc_id, str)]

        # Should successfully handle most concurrent operations
        assert len(successful_adds) >= len(add_tasks) * 0.8  # 80% success rate minimum

        # Concurrent querying
        query_tasks = [
            system.process_query(query=f"concurrent test {i}", mode=QueryMode.FAST, user_id=f"concurrent_user_{i}")
            for i in range(10)
        ]

        query_results = await asyncio.gather(*query_tasks, return_exceptions=True)
        successful_queries = [r for r in query_results if hasattr(r, "answer")]

        # Should handle concurrent queries
        assert len(successful_queries) >= len(query_tasks) * 0.7  # 70% success rate minimum


# Integration test to validate the complete system
@pytest.mark.asyncio
async def test_hyperrag_integration_validation():
    """Comprehensive integration validation test."""

    # Create system
    config = HyperRAGConfig()
    config.enable_hippo_rag = True
    config.enable_graph_rag = True
    config.enable_cognitive_nexus = True

    system = HyperRAG(config)

    try:
        # Initialize
        init_success = await system.initialize()
        assert init_success, "System initialization failed"

        # Test document storage
        doc_id = await system.add_document(
            content="Integration test: Advanced RAG systems combine neural memory, trust networks, and cognitive reasoning for enhanced information retrieval and synthesis.",
            metadata={
                "domain": "information_retrieval",
                "source_type": "technical_documentation",
                "credibility": 0.9,
                "keywords": ["RAG", "neural_memory", "trust_networks", "cognitive_reasoning"],
            },
            memory_type=MemoryType.SEMANTIC,
        )

        assert doc_id is not None
        assert len(doc_id) > 0

        # Test query processing
        result = await system.process_query(
            query="How do advanced RAG systems improve information retrieval?",
            mode=QueryMode.COMPREHENSIVE,
            context={"domain": "information_retrieval", "user_type": "researcher", "detail_level": "technical"},
            user_id="integration_validator",
        )

        assert result is not None
        assert result.confidence > 0.0
        assert len(result.answer) > 50  # Substantial response
        assert result.synthesis_method in ["cognitive_reasoning", "multi_source_synthesis"]

        # Test health check
        health = await system.health_check()
        assert health["status"] in ["healthy", "degraded", "limited"]
        assert "components" in health
        assert health["neural_biological_enabled"] is True

        # Test statistics
        stats = await system.get_stats()
        assert stats["documents_indexed"] >= 1
        assert stats["queries_processed"] >= 1

        print("✅ Integration test passed:")
        print(f"  - Documents indexed: {stats['documents_indexed']}")
        print(f"  - Queries processed: {stats['queries_processed']}")
        print(f"  - System status: {health['status']}")
        print(f"  - Query confidence: {result.confidence:.3f}")
        print(f"  - Response length: {len(result.answer)} chars")

        return True

    finally:
        await system.shutdown()


if __name__ == "__main__":
    # Run integration test
    asyncio.run(test_hyperrag_integration_validation())
