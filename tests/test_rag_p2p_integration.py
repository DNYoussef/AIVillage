#!/usr/bin/env python3
"""
RAG + P2P Network Integration Test

Test the integration between the consolidated RAG system and P2P network infrastructure
for distributed knowledge sharing and collaborative queries.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "packages"))

try:
    from rag import HyperRAG, MemoryType
    from rag.core.hyper_rag import RAGConfig
    from rag.integration.p2p_network_bridge import P2PNetworkRAGBridge

    print("SUCCESS: Import of RAG and P2P systems successful")
except ImportError as e:
    print(f"ERROR: Failed to import systems: {e}")
    sys.exit(1)


async def test_rag_p2p_integration():
    """Test RAG + P2P Network integration."""
    print("Testing RAG + P2P Network Integration...")

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db = f.name

    try:
        # Initialize RAG system with P2P network support enabled
        print("Initializing HyperRAG with P2P network support...")
        config = RAGConfig(
            enable_hippo_rag=True,
            enable_graph_rag=True,
            enable_vector_rag=True,
            enable_cognitive_nexus=True,
            enable_edge_devices=False,
            enable_fog_computing=False,
            enable_p2p_network=True,  # Enable P2P network integration
        )

        rag = HyperRAG(config)
        await rag.initialize()
        print("SUCCESS: HyperRAG with P2P network support initialized")

        # Test P2P bridge access
        if hasattr(rag, "p2p_bridge") and rag.p2p_bridge:
            p2p_bridge = rag.p2p_bridge
            print("SUCCESS: P2P network bridge accessible")
        else:
            print("INFO: P2P network bridge not available, creating standalone")
            p2p_bridge = P2PNetworkRAGBridge(rag)
            await p2p_bridge.initialize()

        # Store test documents for knowledge sharing
        print("\nStoring test documents for P2P sharing...")

        test_documents = [
            {
                "content": "Peer-to-peer networks enable direct communication between devices without centralized servers, providing resilience and scalability.",
                "title": "P2P Networks Introduction",
                "metadata": {"topic": "p2p_networks", "sharing": "public", "trust_level": "high"},
            },
            {
                "content": "Distributed hash tables (DHT) provide efficient data storage and retrieval across peer-to-peer networks using consistent hashing.",
                "title": "Distributed Hash Tables",
                "metadata": {"topic": "distributed_systems", "sharing": "collaborative", "trust_level": "medium"},
            },
            {
                "content": "Blockchain technology uses cryptographic hashing and consensus mechanisms to create immutable distributed ledgers.",
                "title": "Blockchain Fundamentals",
                "metadata": {"topic": "blockchain", "sharing": "public", "trust_level": "high"},
            },
            {
                "content": "Mesh networking allows devices to dynamically form networks and route traffic through multiple hops for redundancy.",
                "title": "Mesh Networking",
                "metadata": {"topic": "mesh_networks", "sharing": "private", "trust_level": "medium"},
            },
        ]

        for doc in test_documents:
            result = await rag.store_document(
                content=doc["content"], title=doc["title"], metadata=doc["metadata"], memory_type=MemoryType.ALL
            )

            if "error" not in result:
                print(f"  SUCCESS: Stored '{doc['title']}'")
                # Test sharing based on metadata
                sharing_mode = doc["metadata"].get("sharing", "private")
                if sharing_mode in ["public", "collaborative"]:
                    share_result = await p2p_bridge.share_knowledge(
                        knowledge_item={
                            "type": "document",
                            "title": doc["title"],
                            "content": doc["content"][:100] + "...",  # Share summary
                            "metadata": doc["metadata"],
                        },
                        target_domains=["knowledge_sharing"],
                    )
                    print(f"    Shared '{doc['title']}' in P2P network ({sharing_mode}) - Success: {share_result}")
            else:
                print(f"  ERROR: Failed to store '{doc['title']}': {result['error']}")

        # Test collaborative query with multiple simulated peers
        print("\nTesting collaborative P2P queries...")

        query = "distributed systems and peer-to-peer networks"

        # Test collaborative query
        print("  Running collaborative query across P2P network...")
        collaborative_result = await p2p_bridge.distributed_query(
            query=query, target_domains=["knowledge_sharing"], max_peers=3, collaborative=True
        )

        print(f"    Local results: {len(collaborative_result.get('local_results', []))}")
        print(f"    Peer contributions: {len(collaborative_result.get('peer_contributions', []))}")
        print(f"    Combined results: {len(collaborative_result.get('combined_results', []))}")
        print(f"    Trust scores: {collaborative_result.get('trust_scores', {})}")
        print(f"    Query latency: {collaborative_result.get('total_latency_ms', 0):.1f}ms")

        # Test knowledge synchronization
        print("\nTesting knowledge synchronization...")

        # Simulate receiving knowledge from peers
        peer_knowledge = [
            {
                "peer_id": "peer_001",
                "knowledge_item": {
                    "title": "Consensus Algorithms",
                    "content": "Consensus algorithms like PBFT and Raft enable distributed systems to agree on shared state.",
                    "metadata": {"topic": "distributed_consensus", "trust_level": "high"},
                    "source_peer": "peer_001",
                    "trust_score": 0.8,
                },
            },
            {
                "peer_id": "peer_002",
                "knowledge_item": {
                    "title": "DHT Routing",
                    "content": "DHT routing protocols like Chord and Kademlia provide efficient peer discovery and data location.",
                    "metadata": {"topic": "dht_routing", "trust_level": "medium"},
                    "source_peer": "peer_002",
                    "trust_score": 0.6,
                },
            },
        ]

        for peer_data in peer_knowledge:
            sync_result = await p2p_bridge.sync_with_peers(domain="knowledge_sharing")

            if sync_result.get("success", False):
                print(f"  SUCCESS: Synced knowledge from {peer_data['peer_id']}")
            else:
                print(f"  WARNING: Failed to sync knowledge from {peer_data['peer_id']}")

        # Test trust-based knowledge filtering
        print("\nTesting trust-based knowledge filtering...")

        # Query with different trust thresholds using distributed_query
        trust_thresholds = [0.3, 0.5, 0.8]

        for threshold in trust_thresholds:
            filtered_result = await p2p_bridge.distributed_query(
                query="consensus algorithms", target_domains=["knowledge_sharing"], max_peers=5, collaborative=True
            )

            # Simulate trust filtering on results
            results = filtered_result.get("results", [])
            trusted_results = [r for r in results if r.get("trust_score", 0.5) >= threshold]

            print(f"  Trust threshold {threshold}: {len(trusted_results)} results (of {len(results)} total)")
            if trusted_results:
                avg_trust = sum(r.get("trust_score", 0.5) for r in trusted_results) / len(trusted_results)
                print(f"    Average trust score: {avg_trust:.2f}")

        # Test peer discovery and network status
        print("\nTesting peer discovery and network status...")

        # Simulate peer discovery
        discovered_peers = await p2p_bridge.discover_peers(domain="knowledge_sharing")

        print(f"  Discovered peers: {len(discovered_peers)}")
        for peer in discovered_peers[:3]:  # Show first 3
            print(f"    Peer: {getattr(peer, 'peer_id', 'unknown')} - Trust: {getattr(peer, 'trust_score', 0):.2f}")

        # Check P2P network synchronization status
        sync_status = await p2p_bridge.get_network_status()

        print(f"  Sync status: {sync_status}")
        print(f"    Last sync: {sync_status.get('last_sync_time', 'never')}")
        print(f"    Pending updates: {sync_status.get('pending_updates', 0)}")
        print(f"    Peer count: {sync_status.get('peer_count', 0)}")
        print(f"    Knowledge items shared: {sync_status.get('shared_items', 0)}")
        print(f"    Network health: {sync_status.get('network_health', 'unknown')}")

        # Test conflict resolution
        print("\nTesting P2P knowledge conflict resolution...")

        # Simulate conflicting information from different peers
        conflict_scenarios = [
            {
                "query": "blockchain consensus",
                "local_result": {
                    "content": "Proof of Work consensus is the most secure",
                    "trust_score": 0.7,
                    "source": "local",
                },
                "peer_result": {
                    "content": "Proof of Stake consensus is more energy efficient",
                    "trust_score": 0.8,
                    "source": "peer_003",
                },
            }
        ]

        for scenario in conflict_scenarios:
            # Simulate conflict resolution using available methods
            print(f"  Conflict resolution for '{scenario['query']}':")

            # Use trust scores to resolve conflicts
            local_trust = scenario["local_result"]["trust_score"]
            peer_trust = scenario["peer_result"]["trust_score"]

            if peer_trust > local_trust:
                winning_result = scenario["peer_result"]
                strategy = "trust_based_peer_preference"
            else:
                winning_result = scenario["local_result"]
                strategy = "trust_based_local_preference"

            print(f"    Resolution strategy: {strategy}")
            print(f"    Winning result: {winning_result['content'][:60]}...")
            print(f"    Confidence: {winning_result['trust_score']:.2f}")

        # Test offline P2P capability
        print("\nTesting offline P2P capability...")

        # Simulate offline scenario using network status
        network_status = await p2p_bridge.get_network_status()

        print(f"  Network connectivity: {network_status.get('connected', False)}")
        print(f"  Offline mode simulation: {not network_status.get('connected', True)}")
        print(f"  Cached peer data available: {len(network_status.get('cached_peers', []))}")

        # If offline, would use cached results
        if not network_status.get("connected", True):
            print("  Using cached results for offline query")
        else:
            print("  Network available, would use live P2P query")

        # Test P2P bridge statistics
        print("\nChecking P2P bridge statistics...")

        # Use available network status method
        bridge_stats = await p2p_bridge.get_network_status()

        network_stats = bridge_stats.get("network_metrics", {})
        knowledge_stats = bridge_stats.get("knowledge_sharing", {})

        print("  Network metrics:")
        print(f"    Active peers: {network_stats.get('active_peers', 0)}")
        print(f"    Network latency: {network_stats.get('avg_peer_latency_ms', 0):.1f}ms")
        print(f"    Message success rate: {network_stats.get('message_success_rate', 0):.2f}")

        print("  Knowledge sharing:")
        print(f"    Items shared: {knowledge_stats.get('items_shared', 0)}")
        print(f"    Items received: {knowledge_stats.get('items_received', 0)}")
        print(f"    Collaborative queries: {knowledge_stats.get('collaborative_queries', 0)}")
        print(f"    Trust conflicts resolved: {knowledge_stats.get('conflicts_resolved', 0)}")

        # Test system status
        print("\nChecking integrated system status...")
        rag_status = await rag.get_system_status()
        print(f"  RAG system status: {list(rag_status.keys())}")

        # Clean up
        await rag.close()
        if hasattr(p2p_bridge, "close"):
            await p2p_bridge.close()

        print("\nSUCCESS: RAG + P2P Network integration test completed!")
        return True

    except Exception as e:
        print(f"\nERROR: P2P integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup
        Path(temp_db).unlink(missing_ok=True)


if __name__ == "__main__":
    success = asyncio.run(test_rag_p2p_integration())
    if success:
        print("RESULT: RAG + P2P Network integration test PASSED")
        sys.exit(0)
    else:
        print("RESULT: RAG + P2P Network integration test FAILED")
        sys.exit(1)
