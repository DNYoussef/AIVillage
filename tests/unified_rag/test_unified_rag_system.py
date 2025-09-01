"""
Comprehensive Test Suite for Unified RAG System

Tests all components and their integration with MCP coordination,
performance validation, and end-to-end functionality verification.
"""

import asyncio
import pytest
import time
from typing import Any, Dict, List, Optional
import logging
import os
import sys

# Add source directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from unified_rag import UnifiedRAGSystem, MCPCoordinator
from unified_rag.core.unified_rag_system import QueryType, RetrievalMode, QueryContext

logger = logging.getLogger(__name__)


class TestUnifiedRAGSystem:
    """Comprehensive test suite for the unified RAG system."""
    
    @pytest.fixture
    async def unified_system(self):
        """Create and initialize unified RAG system for testing."""
        system = UnifiedRAGSystem(
            enable_all_components=True,
            mcp_integration=True,
            performance_mode="balanced"
        )
        
        success = await system.initialize()
        assert success, "UnifiedRAGSystem initialization should succeed"
        
        yield system
        
        # Cleanup
        await system.shutdown()
    
    @pytest.fixture
    async def sample_documents(self):
        """Sample documents for testing."""
        return [
            {
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses statistical techniques to give computers the ability to progressively improve performance on specific tasks.",
                "doc_id": "ml_intro_001",
                "metadata": {"topic": "machine_learning", "difficulty": "beginner", "source": "educational"}
            },
            {
                "content": "Deep learning is a machine learning technique that teaches computers to do what comes naturally to humans: learn by example. Deep learning is a key technology behind driverless cars, speech recognition, image recognition, and natural language processing.",
                "doc_id": "dl_intro_002", 
                "metadata": {"topic": "deep_learning", "difficulty": "intermediate", "source": "educational"}
            },
            {
                "content": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information using a connectionist approach. Neural networks can adapt and learn from data through training.",
                "doc_id": "nn_intro_003",
                "metadata": {"topic": "neural_networks", "difficulty": "intermediate", "source": "educational"}
            },
            {
                "content": "Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines including computer science and computational linguistics.",
                "doc_id": "nlp_intro_004",
                "metadata": {"topic": "nlp", "difficulty": "beginner", "source": "educational"}
            },
            {
                "content": "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward. It's used in game playing, robotics, and autonomous systems.",
                "doc_id": "rl_intro_005",
                "metadata": {"topic": "reinforcement_learning", "difficulty": "advanced", "source": "educational"}
            }
        ]
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, unified_system):
        """Test that the unified RAG system initializes correctly."""
        assert unified_system.initialized, "System should be initialized"
        assert len(unified_system.initialized_components) > 0, "Should have initialized components"
        
        # Check MCP coordinator
        if unified_system.mcp_coordinator:
            status = await unified_system.mcp_coordinator.get_coordinator_status()
            assert status["initialized"], "MCP coordinator should be initialized"
    
    @pytest.mark.asyncio
    async def test_document_ingestion(self, unified_system, sample_documents):
        """Test document ingestion across all components."""
        ingestion_results = []
        
        for doc in sample_documents:
            success = await unified_system.ingest_document(
                content=doc["content"],
                doc_id=doc["doc_id"],
                metadata=doc["metadata"]
            )
            ingestion_results.append(success)
            assert success, f"Document {doc['doc_id']} should ingest successfully"
        
        # Verify all documents were ingested
        assert all(ingestion_results), "All documents should be ingested successfully"
    
    @pytest.mark.asyncio
    async def test_query_types(self, unified_system, sample_documents):
        """Test different query types and retrieval modes."""
        # Ingest test documents
        for doc in sample_documents:
            await unified_system.ingest_document(
                content=doc["content"],
                doc_id=doc["doc_id"], 
                metadata=doc["metadata"]
            )
        
        # Test different query types
        test_queries = [
            ("What is machine learning?", QueryType.FACTUAL),
            ("How does deep learning differ from traditional machine learning?", QueryType.ANALYTICAL),
            ("What are some creative applications of neural networks?", QueryType.CREATIVE),
            ("Compare reinforcement learning with supervised learning", QueryType.COMPARATIVE),
            ("Explore the connections between NLP and machine learning", QueryType.EXPLORATORY)
        ]
        
        for query, query_type in test_queries:
            context = QueryContext(max_results=5, confidence_threshold=0.5)
            
            response = await unified_system.query(
                question=query,
                context=context,
                query_type=query_type
            )
            
            assert response is not None, f"Query should return a response for: {query}"
            assert response.answer, f"Response should have an answer for: {query}"
            assert response.confidence >= 0.0, "Confidence should be non-negative"
            assert response.query_type == query_type, "Query type should match"
            assert response.processing_time_ms > 0, "Processing time should be recorded"
            
            logger.info(f"Query: '{query}' | Type: {query_type.value} | "
                       f"Confidence: {response.confidence:.3f} | "
                       f"Time: {response.processing_time_ms:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_retrieval_modes(self, unified_system, sample_documents):
        """Test different retrieval modes."""
        # Ingest test documents
        for doc in sample_documents:
            await unified_system.ingest_document(
                content=doc["content"],
                doc_id=doc["doc_id"],
                metadata=doc["metadata"]
            )
        
        query = "Explain neural networks and deep learning"
        
        # Test different retrieval modes
        retrieval_modes = [
            RetrievalMode.PRECISION,
            RetrievalMode.RECALL,
            RetrievalMode.BALANCED,
            RetrievalMode.COMPREHENSIVE
        ]
        
        for mode in retrieval_modes:
            context = QueryContext(max_results=5)
            
            response = await unified_system.query(
                question=query,
                context=context,
                retrieval_mode=mode
            )
            
            assert response is not None, f"Query should return response for mode: {mode.value}"
            assert response.retrieval_mode == mode, "Retrieval mode should match"
            assert len(response.sources) > 0, f"Should have sources for mode: {mode.value}"
            
            logger.info(f"Mode: {mode.value} | Sources: {len(response.sources)} | "
                       f"Confidence: {response.confidence:.3f}")
    
    @pytest.mark.asyncio
    async def test_component_integration(self, unified_system, sample_documents):
        """Test integration between different RAG components."""
        # Ingest documents
        for doc in sample_documents:
            await unified_system.ingest_document(
                content=doc["content"],
                doc_id=doc["doc_id"],
                metadata=doc["metadata"]
            )
        
        # Test comprehensive retrieval that uses all components
        context = QueryContext(
            max_results=10,
            enable_creative_search=True,
            include_reasoning=True
        )
        
        response = await unified_system.query(
            question="What are the key concepts in artificial intelligence and how do they relate?",
            context=context,
            retrieval_mode=RetrievalMode.COMPREHENSIVE
        )
        
        # Verify multi-component results
        assert response.vector_results or response.graph_results or response.memory_results, \
            "Should have results from multiple components"
        
        assert response.cognitive_analysis, "Should have cognitive analysis"
        assert response.reasoning_trace, "Should have reasoning trace"
        
        # Check component performance
        assert response.component_times, "Should have component timing information"
        
        total_component_time = sum(response.component_times.values())
        assert total_component_time <= response.processing_time_ms, \
            "Component times should not exceed total processing time"
    
    @pytest.mark.asyncio
    async def test_mcp_integration(self, unified_system):
        """Test MCP server integration and coordination."""
        if not unified_system.mcp_coordinator:
            pytest.skip("MCP integration not available")
        
        # Test MCP coordinator status
        status = await unified_system.mcp_coordinator.get_coordinator_status()
        
        assert status["initialized"], "MCP coordinator should be initialized"
        assert status["servers_configured"] > 0, "Should have configured servers"
        
        # Test embedding generation
        embeddings = await unified_system.mcp_coordinator.generate_embeddings([
            "Test text for embedding generation"
        ])
        
        assert embeddings is not None, "Should generate embeddings"
        assert len(embeddings) > 0, "Should return embeddings"
        
        # Test systematic breakdown
        breakdown = await unified_system.mcp_coordinator.systematic_breakdown(
            "Analyze the components of machine learning systems"
        )
        
        assert breakdown is not None, "Should provide systematic breakdown"
        assert isinstance(breakdown, dict), "Breakdown should be a dictionary"
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, unified_system, sample_documents):
        """Test performance benchmarks and optimization."""
        # Ingest documents
        for doc in sample_documents:
            await unified_system.ingest_document(
                content=doc["content"],
                doc_id=doc["doc_id"],
                metadata=doc["metadata"]
            )
        
        # Performance benchmarks
        query = "What is the relationship between machine learning and artificial intelligence?"
        num_queries = 10
        
        # Measure query performance
        start_time = time.time()
        query_times = []
        
        for i in range(num_queries):
            query_start = time.time()
            
            response = await unified_system.query(
                question=f"{query} (iteration {i})",
                context=QueryContext(max_results=5)
            )
            
            query_time = (time.time() - query_start) * 1000
            query_times.append(query_time)
            
            assert response.processing_time_ms > 0, "Should record processing time"
            
        total_time = (time.time() - start_time) * 1000
        avg_query_time = sum(query_times) / len(query_times)
        
        # Performance assertions
        assert avg_query_time < 5000, "Average query time should be under 5 seconds"
        assert max(query_times) < 10000, "No single query should take over 10 seconds"
        
        logger.info(f"Performance Benchmark Results:")
        logger.info(f"  Total queries: {num_queries}")
        logger.info(f"  Total time: {total_time:.1f}ms")
        logger.info(f"  Average query time: {avg_query_time:.1f}ms")
        logger.info(f"  Min query time: {min(query_times):.1f}ms")
        logger.info(f"  Max query time: {max(query_times):.1f}ms")
    
    @pytest.mark.asyncio
    async def test_system_status_and_metrics(self, unified_system):
        """Test system status reporting and metrics collection."""
        status = await unified_system.get_system_status()
        
        # Verify status structure
        assert "initialized" in status, "Status should include initialization state"
        assert "components" in status, "Status should include component information"
        assert "performance" in status, "Status should include performance metrics"
        assert "configuration" in status, "Status should include configuration"
        
        # Verify component details
        components = status["components"]
        assert "total_components" in components, "Should report total components"
        assert "initialized_components" in components, "Should report initialized components"
        assert "active_components" in components, "Should report active components"
        
        # Verify performance metrics
        performance = status["performance"]
        assert "total_queries" in performance, "Should track total queries"
        assert "avg_response_time_ms" in performance, "Should track average response time"
        
        logger.info(f"System Status Summary:")
        logger.info(f"  Initialized: {status['initialized']}")
        logger.info(f"  Components: {components['initialized_components']}/{components['total_components']}")
        logger.info(f"  Performance: {performance}")
    
    @pytest.mark.asyncio
    async def test_error_handling(self, unified_system):
        """Test error handling and graceful degradation."""
        # Test with invalid query
        response = await unified_system.query(
            question="",  # Empty query
            context=QueryContext(max_results=5)
        )
        
        # System should handle gracefully
        assert response is not None, "Should return response even for empty query"
        assert response.confidence == 0.0, "Confidence should be zero for invalid query"
        
        # Test with malformed context
        response = await unified_system.query(
            question="What is machine learning?",
            context=QueryContext(max_results=0)  # Invalid max_results
        )
        
        assert response is not None, "Should handle invalid context gracefully"
    
    @pytest.mark.asyncio
    async def test_concurrent_queries(self, unified_system, sample_documents):
        """Test system performance under concurrent load."""
        # Ingest documents
        for doc in sample_documents:
            await unified_system.ingest_document(
                content=doc["content"],
                doc_id=doc["doc_id"],
                metadata=doc["metadata"]
            )
        
        # Prepare concurrent queries
        queries = [
            "What is machine learning?",
            "Explain deep learning concepts",
            "How do neural networks work?",
            "What is natural language processing?",
            "Define reinforcement learning"
        ]
        
        # Execute queries concurrently
        start_time = time.time()
        
        tasks = []
        for query in queries:
            task = unified_system.query(
                question=query,
                context=QueryContext(max_results=3)
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        concurrent_time = (time.time() - start_time) * 1000
        
        # Verify all responses
        assert len(responses) == len(queries), "Should get response for each query"
        
        for i, response in enumerate(responses):
            assert response is not None, f"Response {i} should not be None"
            assert response.answer, f"Response {i} should have an answer"
            assert response.confidence >= 0.0, f"Response {i} should have valid confidence"
        
        # Performance check
        avg_concurrent_time = concurrent_time / len(queries)
        
        logger.info(f"Concurrent Query Performance:")
        logger.info(f"  Queries: {len(queries)}")
        logger.info(f"  Total time: {concurrent_time:.1f}ms")
        logger.info(f"  Average per query: {avg_concurrent_time:.1f}ms")
        
        # Concurrent execution should be more efficient than sequential
        assert avg_concurrent_time < 3000, "Concurrent queries should be reasonably fast"
    
    @pytest.mark.asyncio
    async def test_memory_and_learning(self, unified_system, sample_documents):
        """Test memory persistence and learning capabilities."""
        # Ingest documents
        for doc in sample_documents:
            await unified_system.ingest_document(
                content=doc["content"],
                doc_id=doc["doc_id"],
                metadata=doc["metadata"]
            )
        
        query = "What is machine learning?"
        
        # First query
        response1 = await unified_system.query(
            question=query,
            context=QueryContext(max_results=5)
        )
        
        # Second identical query (should benefit from caching/learning)
        response2 = await unified_system.query(
            question=query,
            context=QueryContext(max_results=5)
        )
        
        # Verify responses
        assert response1.answer, "First response should have answer"
        assert response2.answer, "Second response should have answer"
        
        # Second query might be faster due to caching
        logger.info(f"Query Performance Comparison:")
        logger.info(f"  First query: {response1.processing_time_ms:.1f}ms")
        logger.info(f"  Second query: {response2.processing_time_ms:.1f}ms")
        
        # Verify memory system status if available
        if unified_system.memory_system:
            memory_status = await unified_system.memory_system.get_memory_status()
            assert memory_status["memory_counts"]["episodic"] > 0, \
                "Should have episodic memories stored"


# Additional test utilities
def create_test_documents(count: int = 10) -> List[Dict[str, Any]]:
    """Create test documents for benchmarking."""
    documents = []
    
    topics = [
        "machine_learning", "deep_learning", "neural_networks", 
        "natural_language_processing", "computer_vision",
        "reinforcement_learning", "artificial_intelligence",
        "data_science", "statistics", "algorithms"
    ]
    
    for i in range(count):
        topic = topics[i % len(topics)]
        
        doc = {
            "content": f"This is a test document about {topic}. " * 10 + 
                      f"It contains information relevant to {topic} research and applications. " +
                      f"The document discusses various aspects of {topic} including theory, " +
                      f"practical applications, and future directions in the field.",
            "doc_id": f"test_doc_{i:03d}",
            "metadata": {
                "topic": topic,
                "test_document": True,
                "index": i,
                "difficulty": ["beginner", "intermediate", "advanced"][i % 3]
            }
        }
        documents.append(doc)
    
    return documents


@pytest.mark.asyncio
async def test_large_scale_ingestion():
    """Test large-scale document ingestion performance."""
    system = UnifiedRAGSystem(enable_all_components=True)
    
    try:
        await system.initialize()
        
        # Create large number of test documents
        documents = create_test_documents(100)
        
        start_time = time.time()
        success_count = 0
        
        for doc in documents:
            success = await system.ingest_document(
                content=doc["content"],
                doc_id=doc["doc_id"],
                metadata=doc["metadata"]
            )
            if success:
                success_count += 1
        
        ingestion_time = (time.time() - start_time) * 1000
        
        logger.info(f"Large Scale Ingestion Results:")
        logger.info(f"  Documents: {len(documents)}")
        logger.info(f"  Successful: {success_count}")
        logger.info(f"  Success rate: {success_count/len(documents):.1%}")
        logger.info(f"  Total time: {ingestion_time:.1f}ms")
        logger.info(f"  Average per document: {ingestion_time/len(documents):.1f}ms")
        
        assert success_count >= len(documents) * 0.9, "At least 90% of documents should ingest successfully"
        
    finally:
        await system.shutdown()


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])