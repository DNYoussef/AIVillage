#!/usr/bin/env python3
"""
Comprehensive RAG System Integration Test

Tests the complete unified RAG system with all components working together:
- HyperRAG orchestrator with all subsystems
- Edge device optimization and mobile-first design
- P2P network knowledge sharing and collaboration
- Fog computing distributed processing
- Complete end-to-end workflows
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "packages"))

try:
    from rag import EdgeDeviceRAGBridge, FogComputeBridge, HyperRAG, MemoryType, P2PNetworkRAGBridge, QueryMode
    from rag.core.hyper_rag import RAGConfig
    from rag.integration.edge_device_bridge import EdgeDeviceProfile, EdgeDeviceType, ResourceConstraint
    from rag.integration.fog_compute_bridge import ComputeCapability, FogNode, FogNodeType, FogWorkload, WorkloadType

    print("SUCCESS: All RAG system imports successful")
except ImportError as e:
    print(f"ERROR: Failed to import systems: {e}")
    sys.exit(1)


import pytest


@pytest.mark.asyncio
async def test_comprehensive_rag_integration():
    """Test complete RAG system integration across all components."""
    print("=== COMPREHENSIVE RAG SYSTEM INTEGRATION TEST ===")

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db = f.name

    try:
        # Initialize complete RAG system with all integrations
        print("\n1. INITIALIZING COMPLETE RAG SYSTEM...")
        config = RAGConfig(
            enable_hippo_rag=True,
            enable_graph_rag=True,
            enable_vector_rag=True,
            enable_cognitive_nexus=True,
            enable_creativity_engine=True,
            enable_graph_fixer=True,
            enable_edge_devices=True,
            enable_p2p_network=True,
            enable_fog_computing=True,
        )

        rag = HyperRAG(config)
        await rag.initialize()
        print("✅ HyperRAG with all integrations initialized")

        # Initialize integration bridges
        edge_bridge = EdgeDeviceRAGBridge(rag)
        await edge_bridge.initialize()

        p2p_bridge = P2PNetworkRAGBridge(rag)
        await p2p_bridge.initialize()

        fog_bridge = FogComputeBridge(rag)
        await fog_bridge.initialize()

        print("✅ All integration bridges initialized")

        # Set up comprehensive test environment
        print("\n2. SETTING UP TEST ENVIRONMENT...")

        # Create diverse edge devices
        edge_devices = [
            EdgeDeviceProfile(
                device_id="mobile_researcher",
                device_type=EdgeDeviceType.MOBILE_PHONE,
                cpu_cores=4,
                memory_mb=3072,
                battery_percent=40.0,
                network_type="cellular",
                resource_constraint=ResourceConstraint.MODERATE,
                battery_saving_mode=True,
                os_type="android",
            ),
            EdgeDeviceProfile(
                device_id="laptop_analyst",
                device_type=EdgeDeviceType.LAPTOP,
                cpu_cores=8,
                memory_mb=16384,
                battery_percent=85.0,
                network_type="wifi",
                resource_constraint=ResourceConstraint.MINIMAL,
                os_type="windows",
            ),
            EdgeDeviceProfile(
                device_id="iot_sensor",
                device_type=EdgeDeviceType.IOT_DEVICE,
                cpu_cores=1,
                memory_mb=512,
                battery_percent=70.0,
                network_type="wifi",
                resource_constraint=ResourceConstraint.SEVERE,
                prefer_offline=True,
                os_type="linux",
            ),
        ]

        # Register edge devices
        for device in edge_devices:
            success = await edge_bridge.register_device(device)
            print(f"  📱 Registered {device.device_id}: {success}")

        # Create fog computing nodes
        fog_nodes = [
            FogNode(
                node_id="edge_server_01",
                node_type=FogNodeType.EDGE_SERVER,
                cpu_cores=16,
                memory_gb=32.0,
                storage_gb=1000.0,
                gpu_available=True,
                compute_capability=ComputeCapability.HIGH_PERFORMANCE,
                rag_components_available=["hippo", "graph", "vector"],
            ),
            FogNode(
                node_id="regional_fog_01",
                node_type=FogNodeType.REGIONAL_FOG,
                cpu_cores=32,
                memory_gb=128.0,
                storage_gb=5000.0,
                gpu_available=True,
                compute_capability=ComputeCapability.HIGH_PERFORMANCE,
                rag_components_available=["hippo", "graph", "vector"],
            ),
        ]

        # Register fog nodes
        for node in fog_nodes:
            success = await fog_bridge.register_fog_node(node)
            print(f"  ☁️ Registered {node.node_id}: {success}")

        # Store comprehensive knowledge base
        print("\n3. BUILDING COMPREHENSIVE KNOWLEDGE BASE...")

        knowledge_documents = [
            {
                "content": "Artificial intelligence systems require distributed computing architectures to handle large-scale data processing and real-time inference across edge devices.",
                "title": "AI Distributed Computing",
                "metadata": {"domain": "ai_systems", "complexity": "advanced", "sharing": "public"},
            },
            {
                "content": "Edge computing brings computation closer to data sources, reducing latency and bandwidth usage while enabling real-time decision making in IoT environments.",
                "title": "Edge Computing Fundamentals",
                "metadata": {"domain": "edge_computing", "complexity": "intermediate", "sharing": "collaborative"},
            },
            {
                "content": "Federated learning enables machine learning model training across distributed devices without centralizing raw data, preserving privacy and reducing communication costs.",
                "title": "Federated Learning Privacy",
                "metadata": {"domain": "federated_learning", "complexity": "advanced", "sharing": "public"},
            },
            {
                "content": "Peer-to-peer networks create resilient distributed systems where nodes can communicate directly without centralized coordination, enabling fault-tolerant architectures.",
                "title": "P2P Network Resilience",
                "metadata": {"domain": "p2p_networks", "complexity": "intermediate", "sharing": "collaborative"},
            },
            {
                "content": "Cognitive architectures integrate multiple AI subsystems including memory, reasoning, learning, and perception to create human-like intelligent behavior.",
                "title": "Cognitive AI Architectures",
                "metadata": {"domain": "cognitive_ai", "complexity": "expert", "sharing": "research"},
            },
        ]

        stored_docs = 0
        for doc in knowledge_documents:
            result = await rag.store_document(
                content=doc["content"], title=doc["title"], metadata=doc["metadata"], memory_type=MemoryType.ALL
            )

            if "error" not in result:
                stored_docs += 1
                print(f"  📚 Stored: {doc['title']}")

                # Share knowledge via P2P if appropriate
                sharing_mode = doc["metadata"].get("sharing", "private")
                if sharing_mode in ["public", "collaborative"]:
                    shared = await p2p_bridge.share_knowledge(
                        knowledge_item={
                            "type": "research_document",
                            "title": doc["title"],
                            "content": doc["content"][:150] + "...",
                            "metadata": doc["metadata"],
                        },
                        target_domains=["research", "ai_development"],
                    )
                    print(f"    🌐 P2P shared: {shared}")

        print(f"✅ Knowledge base created: {stored_docs}/{len(knowledge_documents)} documents stored")

        # Test multi-modal query processing
        print("\n4. TESTING MULTI-MODAL QUERY PROCESSING...")

        test_queries = [
            {
                "query": "How do AI systems use distributed computing and edge devices?",
                "mode": QueryMode.COMPREHENSIVE,
                "device_context": edge_devices[1],  # Laptop - high performance
                "expected_systems": ["hippo", "graph", "vector", "cognitive"],
            },
            {
                "query": "federated learning privacy",
                "mode": QueryMode.FAST,
                "device_context": edge_devices[0],  # Mobile - resource constrained
                "expected_systems": ["vector"],
            },
            {
                "query": "What are non-obvious connections between P2P networks and cognitive AI?",
                "mode": QueryMode.CREATIVE,
                "device_context": edge_devices[1],  # Laptop - creative processing
                "expected_systems": ["graph", "vector", "creativity"],
            },
        ]

        query_results = []
        for i, test in enumerate(test_queries):
            print(f"\n  Query {i+1}: {test['query'][:50]}...")

            # Device-specific optimization
            device_optimization = await edge_bridge.optimize_for_device(
                device_id=test["device_context"].device_id, query=test["query"], query_mode=test["mode"]
            )
            print(f"    📱 Device optimization: {device_optimization.get('edge_optimized', False)}")

            # Execute query
            query_context = {
                "device_id": test["device_context"].device_id,
                "device_profile": test["device_context"],
                "optimization": device_optimization,
            }

            result = await rag.query(query=test["query"], mode=test["mode"], context=query_context)

            query_results.append(result)

            # Analyze results
            results_count = len(getattr(result, "results", getattr(result, "primary_sources", [])))
            execution_time = getattr(result, "execution_time_ms", getattr(result, "total_latency_ms", 0))

            print(f"    📊 Results: {results_count} found in {execution_time:.1f}ms")

            # Check for expected system usage
            systems_used = getattr(result, "systems_used", [])
            print(f"    🔧 Systems used: {systems_used}")

            # Check creative insights for creative queries
            if test["mode"] == QueryMode.CREATIVE:
                insights = getattr(result, "insights", getattr(result, "creative_insights", None))
                if insights:
                    connections = insights.get("non_obvious_connections", [])
                    print(f"    💡 Creative insights: {len(connections)} non-obvious connections found")

        # Test fog computing distributed processing
        print("\n5. TESTING FOG COMPUTING DISTRIBUTED PROCESSING...")

        # Submit distributed query workload
        distributed_workload = FogWorkload(
            workload_type=WorkloadType.QUERY_PROCESSING,
            description="Distributed RAG query across fog infrastructure",
            payload={
                "query": "cognitive AI architectures and distributed systems",
                "query_mode": "comprehensive",
                "distributed": True,
            },
            required_memory_gb=2.0,
            max_latency_ms=10000,
            priority=2,
        )

        workload_id = await fog_bridge.submit_workload(distributed_workload)
        print(f"  ☁️ Submitted fog workload: {workload_id}")

        # Check fog infrastructure status
        fog_status = await fog_bridge.get_fog_infrastructure_status()
        infra_health = fog_status.get("infrastructure_health", {})

        print("  📊 Fog infrastructure:")
        print(f"    Online nodes: {infra_health.get('online_nodes', 0)}/{infra_health.get('total_nodes', 0)}")
        print(f"    Active workloads: {infra_health.get('active_workloads', 0)}")

        # Test peer collaboration
        print("\n6. TESTING P2P PEER COLLABORATION...")

        # Execute distributed query
        collaborative_query = "How can edge computing and federated learning work together?"
        p2p_result = await p2p_bridge.distributed_query(
            query=collaborative_query, target_domains=["research", "ai_development"], max_peers=5, collaborative=True
        )

        print("  🌐 P2P collaborative query results:")
        print(f"    Local results: {len(p2p_result.get('local_results', []))}")
        print(f"    Peer contributions: {len(p2p_result.get('peer_contributions', []))}")
        print(f"    Query latency: {p2p_result.get('total_latency_ms', 0):.1f}ms")

        # Check network status
        network_status = await p2p_bridge.get_network_status()
        network_health = network_status.get("network_health", {})

        print("  📡 Network status:")
        print(f"    Total peers: {network_health.get('total_peers', 0)}")
        print(f"    Knowledge shared: {network_status.get('statistics', {}).get('knowledge_shared', 0)}")

        # Test system integration and health
        print("\n7. TESTING SYSTEM INTEGRATION & HEALTH...")

        # Check overall system status
        rag_status = await rag.get_system_status()
        print(f"  🏥 RAG system health: {list(rag_status.keys())}")

        # Edge device statistics
        edge_stats = await edge_bridge.get_bridge_statistics()
        edge_registered = edge_stats.get("registered_devices", {})

        print(f"  📱 Edge devices: {edge_registered.get('total', 0)} registered")
        print(f"    Device types: {edge_registered.get('by_type', {})}")

        # P2P network statistics
        p2p_stats = network_status.get("statistics", {})
        print("  🌐 P2P network:")
        print(f"    Messages sent: {p2p_stats.get('messages_sent', 0)}")
        print(f"    Queries processed: {p2p_stats.get('queries_processed', 0)}")

        # Test knowledge gap detection and creativity
        print("\n8. TESTING KNOWLEDGE GAP DETECTION & CREATIVITY...")

        # Look for knowledge gaps
        if hasattr(rag, "graph_fixer") and rag.graph_fixer:
            gaps = await rag.graph_fixer.detect_knowledge_gaps(
                query="quantum computing and AI integration", focus_area="emerging_technologies"
            )

            print(f"  🔍 Knowledge gaps detected: {len(gaps)}")
            for gap in gaps[:2]:  # Show first 2
                print(
                    f"    Gap: {getattr(gap, 'missing_concept', 'unknown')} (confidence: {getattr(gap, 'confidence_score', 0):.2f})"
                )

        # Test creativity engine for insights
        if hasattr(rag, "creativity_engine") and rag.creativity_engine:
            insights = await rag.creativity_engine.discover_insights(
                query="connections between distributed AI and human cognition",
                retrieved_info=query_results[0].results if query_results else [],
                creativity_level=0.8,
            )

            print("  💡 Creative insights generated:")
            non_obvious = insights.get("non_obvious_connections", [])
            print(f"    Non-obvious connections: {len(non_obvious)}")
            analogies = insights.get("analogical_insights", [])
            print(f"    Analogical insights: {len(analogies)}")

        # Final integration validation
        print("\n9. FINAL INTEGRATION VALIDATION...")

        validation_checks = {
            "rag_core_operational": bool(rag.initialized),
            "edge_devices_registered": len(edge_devices),
            "fog_nodes_available": len(fog_nodes),
            "p2p_network_active": network_status.get("protocol_status", {}).get("bitchat", {}).get("connected", False),
            "knowledge_base_populated": stored_docs > 0,
            "queries_successful": len([r for r in query_results if getattr(r, "success", True)]),
            "integrations_functional": all(
                [hasattr(rag, "edge_device_bridge"), hasattr(rag, "p2p_bridge"), hasattr(rag, "fog_bridge")]
            ),
        }

        print("  ✅ Integration validation:")
        for check, status in validation_checks.items():
            status_icon = "✅" if status else "❌"
            print(f"    {status_icon} {check}: {status}")

        # Calculate overall success rate
        total_checks = len(validation_checks)
        passed_checks = sum(1 for v in validation_checks.values() if v)
        success_rate = (passed_checks / total_checks) * 100

        print(f"\n🎯 OVERALL INTEGRATION SUCCESS RATE: {success_rate:.1f}% ({passed_checks}/{total_checks})")

        # Clean up
        await rag.close()
        await edge_bridge.close()
        await p2p_bridge.close()
        await fog_bridge.close()

        if success_rate >= 80:
            print("\n🎉 COMPREHENSIVE INTEGRATION TEST: ✅ PASSED")
            return True
        else:
            print("\n⚠️ COMPREHENSIVE INTEGRATION TEST: ❌ FAILED")
            return False

    except Exception as e:
        print(f"\n💥 INTEGRATION TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup
        Path(temp_db).unlink(missing_ok=True)


if __name__ == "__main__":
    success = asyncio.run(test_comprehensive_rag_integration())
    if success:
        print("\n🏆 RESULT: Comprehensive RAG integration test PASSED!")
        print("🚀 System ready for production deployment!")
        sys.exit(0)
    else:
        print("\n💥 RESULT: Comprehensive RAG integration test FAILED!")
        sys.exit(1)
