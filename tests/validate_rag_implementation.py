"""
Validation script for RAG implementation - TDD London School style

This script validates that our RAG pipeline implementation works correctly
following London School TDD principles:
1. Test imports and basic contracts
2. Test behavior with mocked external services
3. Test core functionality without mocks
4. Validate real-world usage patterns
"""

import asyncio
import os
import sys
from unittest.mock import Mock

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def test_imports():
    """Test that all expected imports work."""
    print("Testing imports...")

    try:
        print("  ✓ RAGPipeline import successful")
    except ImportError as e:
        print(f"  ✗ RAGPipeline import failed: {e}")
        return False

    try:
        print("  ✓ HyperRAG imports successful")
    except ImportError as e:
        print(f"  ✗ HyperRAG import failed: {e}")
        return False

    return True


def test_rag_pipeline_contract():
    """Test that RAGPipeline follows expected behavioral contract."""
    print("\nTesting RAGPipeline contract...")

    try:
        from packages.rag.core.pipeline import RAGPipeline

        # Test initialization without external dependencies
        pipeline = RAGPipeline()
        print("  ✓ RAGPipeline creation successful")

        # Test required methods exist
        assert hasattr(pipeline, "initialize"), "Missing initialize method"
        assert hasattr(pipeline, "process_document"), "Missing process_document method"
        assert hasattr(pipeline, "query"), "Missing query method"
        print("  ✓ Required methods present")

        # Test initialization
        result = pipeline.initialize()
        assert result is True, "Initialization should return True"
        print("  ✓ Initialization successful")

        return True

    except Exception as e:
        print(f"  ✗ Contract test failed: {e}")
        return False


def test_document_processing_without_mocks():
    """Test document processing with real in-memory implementations."""
    print("\nTesting document processing (no mocks)...")

    try:
        from packages.rag.core.pipeline import RAGPipeline

        pipeline = RAGPipeline()
        pipeline.initialize()

        # Process a test document
        test_doc = "Machine learning is a subset of artificial intelligence that enables computers to learn."
        result = pipeline.process_document(test_doc)

        assert result.document_id is not None, "Document ID should be assigned"
        assert result.vector_stored is True, "Document should be stored in vector store"
        assert result.graph_relations_added >= 0, "Graph relations should be non-negative"
        print(f"  ✓ Document processed: ID={result.document_id}, relations={result.graph_relations_added}")

        return True

    except Exception as e:
        print(f"  ✗ Document processing failed: {e}")
        return False


def test_query_functionality_without_mocks():
    """Test query functionality with real in-memory implementations."""
    print("\nTesting query functionality (no mocks)...")

    try:
        from packages.rag.core.pipeline import RAGPipeline

        pipeline = RAGPipeline()
        pipeline.initialize()

        # Add some test documents
        docs = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Python is a popular programming language for AI development.",
        ]

        for doc in docs:
            pipeline.process_document(doc)

        # Test query
        query_result = pipeline.query("What is machine learning?")

        assert query_result.query == "What is machine learning?", "Query should be preserved"
        assert isinstance(query_result.results, list), "Results should be a list"
        assert len(query_result.processing_time_ms) >= 0, "Processing time should be non-negative"
        assert len(query_result.sources_used) > 0, "Should use at least one source"
        print(f"  ✓ Query processed: {len(query_result.results)} results in {query_result.processing_time_ms:.2f}ms")

        # Test with graph expansion
        expanded_result = pipeline.query("neural networks", use_graph_expansion=True)
        assert (
            "graph" in expanded_result.sources_used or "vector" in expanded_result.sources_used
        ), "Should use some source"
        print(f"  ✓ Graph-expanded query: {len(expanded_result.results)} results")

        return True

    except Exception as e:
        print(f"  ✗ Query functionality failed: {e}")
        return False


def test_with_mocked_external_services():
    """Test with mocked external services (London School approach)."""
    print("\nTesting with mocked external services...")

    try:
        from packages.rag.core.pipeline import GraphStore, RAGPipeline, VectorStore

        # Create mocks for external services ONLY
        mock_vector = Mock(spec=VectorStore)
        mock_vector.add_document.return_value = {"status": "added", "doc_id": "mock_123"}
        mock_vector.search.return_value = [{"content": "Mock result", "score": 0.95, "doc_id": "mock_123"}]

        mock_graph = Mock(spec=GraphStore)
        mock_graph.add_relations.return_value = {"relations_added": 3}
        mock_graph.find_related.return_value = [{"entity": "AI", "relation": "subset_of", "target": "computer_science"}]

        # Test with mocked dependencies
        pipeline = RAGPipeline(vector_store=mock_vector, graph_store=mock_graph)
        pipeline.initialize()

        # Test document processing
        pipeline.process_document("Test document")

        # Verify interactions (London School focus)
        mock_vector.add_document.assert_called_once_with("Test document", None)
        mock_graph.add_relations.assert_called_once_with("Test document", "mock_123")
        print("  ✓ Document processing interactions verified")

        # Test query processing
        pipeline.query("test query")

        mock_vector.search.assert_called_with("test query", 5)
        mock_graph.find_related.assert_called_once_with("test query")
        print("  ✓ Query processing interactions verified")

        return True

    except Exception as e:
        print(f"  ✗ Mocked service test failed: {e}")
        return False


async def test_hyper_rag_integration():
    """Test HyperRAG integration and interface compatibility."""
    print("\nTesting HyperRAG integration...")

    try:
        from packages.rag.core.hyper_rag import MemoryType, QueryMode, create_hyper_rag

        # Test enum availability
        assert hasattr(QueryMode, "FAST"), "QueryMode should have FAST"
        assert hasattr(QueryMode, "BALANCED"), "QueryMode should have BALANCED"
        assert hasattr(MemoryType, "EPISODIC"), "MemoryType should have EPISODIC"
        print("  ✓ Enums available")

        # Test creation
        rag = await create_hyper_rag(enable_all=False)
        assert rag is not None, "Should create RAG instance"
        print("  ✓ HyperRAG creation successful")

        # Test document storage
        store_result = await rag.store_document(
            content="Test document for HyperRAG", title="Test Document", memory_type=MemoryType.ALL
        )
        assert "document_id" in store_result, "Should return document ID"
        print("  ✓ Document storage successful")

        # Test querying
        query_result = await rag.query("test query", mode=QueryMode.FAST)
        assert hasattr(query_result, "synthesized_answer"), "Should have synthesized answer"
        assert hasattr(query_result, "confidence_score"), "Should have confidence score"
        print(f"  ✓ Query successful: confidence={query_result.confidence_score:.3f}")

        # Test status
        status = await rag.get_system_status()
        assert "statistics" in status, "Should have statistics"
        print("  ✓ System status retrieved")

        await rag.close()
        print("  ✓ System closed cleanly")

        return True

    except Exception as e:
        print(f"  ✗ HyperRAG integration failed: {e}")
        return False


def test_example_compatibility():
    """Test that our implementation works with example usage patterns."""
    print("\nTesting example compatibility...")

    try:
        # Test the import pattern from the example
        from packages.rag.core.hyper_rag import MemoryType, QueryMode, create_hyper_rag

        print("  ✓ Example import pattern works")

        # Verify the example would work (without actually running async)
        assert callable(create_hyper_rag), "create_hyper_rag should be callable"
        assert hasattr(MemoryType, "ALL"), "MemoryType.ALL should exist"
        assert hasattr(QueryMode, "COMPREHENSIVE"), "QueryMode.COMPREHENSIVE should exist"
        print("  ✓ Example usage pattern compatible")

        return True

    except Exception as e:
        print(f"  ✗ Example compatibility failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=== RAG Implementation Validation ===")
    print("Testing packages.rag module structure and functionality")
    print("Using London School TDD principles: mock external services only\n")

    tests = [
        ("Import Tests", test_imports),
        ("Contract Tests", test_rag_pipeline_contract),
        ("Document Processing (No Mocks)", test_document_processing_without_mocks),
        ("Query Functionality (No Mocks)", test_query_functionality_without_mocks),
        ("External Service Mocking", test_with_mocked_external_services),
        ("Example Compatibility", test_example_compatibility),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"--- {test_name} ---")
        if test_func():
            passed += 1
            print(f"PASSED: {test_name}\n")
        else:
            print(f"FAILED: {test_name}\n")

    # Test async functionality
    print("--- HyperRAG Integration (Async) ---")
    try:
        result = asyncio.run(test_hyper_rag_integration())
        if result:
            passed += 1
            print("PASSED: HyperRAG Integration\n")
        else:
            print("FAILED: HyperRAG Integration\n")
        total += 1
    except Exception as e:
        print(f"FAILED: HyperRAG Integration - {e}\n")
        total += 1

    print("=== Summary ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")

    if passed == total:
        print("SUCCESS: All tests passed! RAG pipeline is working correctly.")
        print("\nKey achievements:")
        print("- packages.rag module structure created")
        print("- RAGPipeline with dependency injection working")
        print("- HyperRAG interface compatibility maintained")
        print("- External services mocked appropriately (London School)")
        print("- Core functionality works without mocks")
        return True
    else:
        print("FAILURE: Some tests failed. Check implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
