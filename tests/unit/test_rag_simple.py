#!/usr/bin/env python3
"""
Simple RAG System Test

Basic test to validate the consolidated RAG system works correctly.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "packages"))

try:
    from rag import HyperRAG, MemoryType, QueryMode
    from rag.core.hyper_rag import RAGConfig

    print("SUCCESS: Import of RAG system successful")
except ImportError as e:
    print(f"ERROR: Failed to import RAG system: {e}")
    sys.exit(1)


async def test_basic_rag_functionality():
    """Test basic RAG functionality."""
    print("Testing Consolidated RAG System...")

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db = f.name

    try:
        # Initialize RAG system
        print("Initializing HyperRAG system...")
        config = RAGConfig(
            enable_hippo_rag=True,
            enable_graph_rag=True,
            enable_vector_rag=True,
            enable_cognitive_nexus=True,
            enable_creativity_engine=True,
            enable_graph_fixer=True,
            enable_fog_computing=False,  # Disable for basic test
            enable_edge_devices=False,
            enable_p2p_network=False,
        )

        rag = HyperRAG(config)
        await rag.initialize()
        print("SUCCESS: HyperRAG initialized")

        # Test document ingestion
        print("Testing document ingestion...")

        test_doc = {
            "content": "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes that process information through weighted connections.",
            "metadata": {"topic": "neural_networks", "chapter": "1", "book": "AI Fundamentals"},
            "doc_id": "nn_intro",
        }

        result = await rag.store_document(
            content=test_doc["content"],
            title=test_doc["doc_id"],
            metadata=test_doc["metadata"],
            memory_type=MemoryType.ALL,
        )

        if "error" in result:
            print(f"ERROR: Document storage failed: {result['error']}")
            return False
        else:
            print("SUCCESS: Document stored successfully")
            print(f"  Storage results: {result}")
            stored_in_systems = sum(1 for v in result.values() if v)
            print(f"  Stored in {stored_in_systems}/{len(result)} systems")

        # Test basic query
        print("Testing basic query...")

        query_result = await rag.query("neural networks", mode=QueryMode.FAST)

        if query_result.success:
            print(
                f"SUCCESS: Query succeeded - {len(query_result.results)} results in {query_result.execution_time_ms:.1f}ms"
            )
        else:
            print(f"ERROR: Query failed: {query_result.error}")
            return False

        # Test system status
        print("Checking system status...")
        status = await rag.get_system_status()
        print(f"System status: {status}")
        print(f"Systems active: {len([k for k, v in status.items() if v])}")

        # Close system
        await rag.close()
        print("SUCCESS: All basic tests completed!")

        return True

    except Exception as e:
        print(f"ERROR: Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup
        Path(temp_db).unlink(missing_ok=True)


if __name__ == "__main__":
    success = asyncio.run(test_basic_rag_functionality())
    if success:
        print("RESULT: Basic RAG system test PASSED")
        sys.exit(0)
    else:
        print("RESULT: Basic RAG system test FAILED")
        sys.exit(1)
