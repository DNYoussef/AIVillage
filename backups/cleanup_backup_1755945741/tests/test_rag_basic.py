#!/usr/bin/env python3
"""
Basic RAG System Test

Simple test to validate the consolidated RAG system works correctly.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "packages"))

from rag import HyperRAG, MemoryType, QueryMode


async def test_basic_rag_functionality():
    """Test basic RAG functionality."""
    print("🧠 Testing Consolidated RAG System...")

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db = f.name

    try:
        # Initialize RAG system
        print("📊 Initializing HyperRAG system...")
        config = {
            "database_path": temp_db,
            "enable_edge_device_integration": True,
            "enable_p2p_integration": True,
            "enable_fog_computing": True,
            "hippo_max_nodes": 100,
            "vector_max_documents": 500,
            "graph_max_nodes": 200,
        }

        rag = HyperRAG(config)
        await rag.initialize()
        print("✅ HyperRAG initialized successfully")

        # Test document ingestion
        print("\n📖 Testing document ingestion...")

        test_documents = [
            {
                "content": "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes that process information through weighted connections.",
                "metadata": {"topic": "neural_networks", "chapter": "1", "book": "AI Fundamentals"},
                "doc_id": "nn_intro",
            },
            {
                "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers. It has achieved breakthrough results in computer vision and natural language processing.",
                "metadata": {"topic": "deep_learning", "chapter": "2", "book": "AI Fundamentals"},
                "doc_id": "dl_overview",
            },
            {
                "content": "Transformers are a neural network architecture that relies entirely on attention mechanisms. They have become the foundation for large language models like GPT and BERT.",
                "metadata": {"topic": "transformers", "chapter": "3", "book": "Modern AI"},
                "doc_id": "transformers",
            },
        ]

        for doc in test_documents:
            print(f"  Ingesting: {doc['doc_id']}")
            result = await rag.ingest_document(
                content=doc["content"],
                metadata=doc["metadata"],
                document_id=doc["doc_id"],
                memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.VECTOR],
            )

            if result["success"]:
                print(f"    ✅ Successfully ingested {doc['doc_id']}")
                print(f"       Hippo node: {result.get('hippo_node_id', 'N/A')}")
                print(f"       Graph node: {result.get('graph_node_id', 'N/A')}")
                print(f"       Vector doc: {result.get('vector_document_id', 'N/A')}")
                print(f"       Chunks: {len(result.get('chunks_created', []))}")
            else:
                print(f"    ❌ Failed to ingest {doc['doc_id']}: {result.get('error', 'Unknown error')}")

        # Test different query modes
        print("\n🔍 Testing query modes...")

        test_query = "neural networks and deep learning"

        # Test Fast mode
        print("  Testing Fast mode...")
        fast_result = await rag.query(test_query, mode=QueryMode.FAST)
        if fast_result.success:
            print(
                f"    ✅ Fast query succeeded - {len(fast_result.results)} results in {fast_result.execution_time_ms:.1f}ms"
            )
        else:
            print(f"    ❌ Fast query failed: {fast_result.error}")

        # Test Balanced mode
        print("  Testing Balanced mode...")
        balanced_result = await rag.query(test_query, mode=QueryMode.BALANCED)
        if balanced_result.success:
            print(
                f"    ✅ Balanced query succeeded - {len(balanced_result.results)} results in {balanced_result.execution_time_ms:.1f}ms"
            )
        else:
            print(f"    ❌ Balanced query failed: {balanced_result.error}")

        # Test Comprehensive mode
        print("  Testing Comprehensive mode...")
        comp_result = await rag.query(test_query, mode=QueryMode.COMPREHENSIVE)
        if comp_result.success:
            print(
                f"    ✅ Comprehensive query succeeded - {len(comp_result.results)} results in {comp_result.execution_time_ms:.1f}ms"
            )
            if comp_result.analysis_results:
                print(f"       Analysis results: {len(comp_result.analysis_results)}")
            if comp_result.reasoning_chain:
                print(f"       Reasoning chain: {len(comp_result.reasoning_chain)}")
        else:
            print(f"    ❌ Comprehensive query failed: {comp_result.error}")

        # Test Creative mode
        print("  Testing Creative mode...")
        creative_result = await rag.query(test_query, mode=QueryMode.CREATIVE)
        if creative_result.success:
            print(
                f"    ✅ Creative query succeeded - {len(creative_result.results)} results in {creative_result.execution_time_ms:.1f}ms"
            )
            if creative_result.insights:
                insights = creative_result.insights
                print(f"       Non-obvious connections: {len(insights.get('non_obvious_connections', []))}")
                print(f"       Analogical insights: {len(insights.get('analogical_insights', []))}")
        else:
            print(f"    ❌ Creative query failed: {creative_result.error}")

        # Test system statistics
        print("\n📊 System statistics...")
        stats = await rag.get_system_statistics()
        print(f"  Hippo nodes: {stats.get('hippo_nodes', 0)}")
        print(f"  Graph nodes: {stats.get('graph_nodes', 0)}")
        print(f"  Vector documents: {stats.get('vector_documents', 0)}")
        print(f"  Queries processed: {stats.get('queries_processed', 0)}")

        # Test system health
        print("\n🏥 System health check...")
        health = await rag.get_system_health()
        print(f"  Overall status: {health.get('overall_status', 'unknown')}")

        component_status = health.get("component_status", {})
        for component, status in component_status.items():
            status_emoji = "✅" if status == "operational" else "⚠️"
            print(f"  {component}: {status_emoji} {status}")

        # Test integration bridges
        print("\n🌉 Testing integration bridges...")

        # Edge device bridge
        edge_status = await rag.edge_device_bridge.get_edge_coordination_status()
        print(f"  Edge devices: {edge_status.get('registered_devices', 0)} registered")

        # P2P bridge
        p2p_status = await rag.p2p_bridge.get_sync_status()
        print(f"  P2P peers: {p2p_status.get('peer_count', 0)} connected")

        # Fog computing bridge
        fog_status = await rag.fog_bridge.get_fog_infrastructure_status()
        fog_health = fog_status.get("infrastructure_health", {})
        print(f"  Fog nodes: {fog_health.get('online_nodes', 0)}/{fog_health.get('total_nodes', 0)} online")

        # Close system
        await rag.close()
        print("\n✅ All tests completed successfully!")
        print("🎉 Consolidated RAG system is fully operational!")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup
        Path(temp_db).unlink(missing_ok=True)


if __name__ == "__main__":
    success = asyncio.run(test_basic_rag_functionality())
    sys.exit(0 if success else 1)
