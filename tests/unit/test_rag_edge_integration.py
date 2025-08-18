#!/usr/bin/env python3
"""
RAG + Edge Device Integration Test

Test the integration between the consolidated RAG system and edge device infrastructure.
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
    from rag.integration.edge_device_bridge import (
        EdgeDeviceProfile,
        EdgeDeviceRAGBridge,
        EdgeDeviceType,
        ResourceConstraint,
    )

    print("SUCCESS: Import of RAG and Edge systems successful")
except ImportError as e:
    print(f"ERROR: Failed to import systems: {e}")
    sys.exit(1)


async def test_rag_edge_integration():
    """Test RAG + Edge Device integration."""
    print("Testing RAG + Edge Device Integration...")

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db = f.name

    try:
        # Initialize RAG system with edge device support enabled
        print("Initializing HyperRAG with edge device support...")
        config = RAGConfig(
            enable_hippo_rag=True,
            enable_graph_rag=True,
            enable_vector_rag=True,
            enable_cognitive_nexus=True,
            enable_edge_devices=True,  # Enable edge device integration
            enable_fog_computing=False,
            enable_p2p_network=False,
        )

        rag = HyperRAG(config)
        await rag.initialize()
        print("SUCCESS: HyperRAG with edge device support initialized")

        # Test edge device bridge access
        if hasattr(rag, "edge_device_bridge") and rag.edge_device_bridge:
            edge_bridge = rag.edge_device_bridge
            print("SUCCESS: Edge device bridge accessible")
        else:
            print("INFO: Edge device bridge not available, creating standalone")
            edge_bridge = EdgeDeviceRAGBridge(rag)
            await edge_bridge.initialize()

        # Create test edge device profiles
        print("\nCreating test edge device profiles...")

        # Mobile phone with battery constraints
        mobile_device = EdgeDeviceProfile(
            device_id="mobile_001",
            device_type=EdgeDeviceType.MOBILE_PHONE,
            cpu_cores=4,
            memory_mb=2048,
            battery_percent=30.0,  # Low battery
            network_type="cellular",
            resource_constraint=ResourceConstraint.MODERATE,
            battery_saving_mode=True,
            data_saving_mode=True,
            os_type="android",
        )

        # High-end laptop
        laptop_device = EdgeDeviceProfile(
            device_id="laptop_001",
            device_type=EdgeDeviceType.LAPTOP,
            cpu_cores=8,
            memory_mb=8192,
            battery_percent=80.0,
            network_type="wifi",
            resource_constraint=ResourceConstraint.MINIMAL,
            battery_saving_mode=False,
            data_saving_mode=False,
            os_type="windows",
        )

        # IoT device with severe constraints
        iot_device = EdgeDeviceProfile(
            device_id="iot_001",
            device_type=EdgeDeviceType.IOT_DEVICE,
            cpu_cores=1,
            memory_mb=512,
            battery_percent=60.0,
            network_type="wifi",
            resource_constraint=ResourceConstraint.SEVERE,
            prefer_offline=True,
            os_type="linux",
        )

        # Register edge devices
        print("Registering edge devices...")
        devices = [mobile_device, laptop_device, iot_device]

        for device in devices:
            success = await edge_bridge.register_device(device)
            if success:
                print(f"  SUCCESS: Registered {device.device_id} ({device.device_type.value})")
            else:
                print(f"  WARNING: Failed to register {device.device_id}")

        # Store test documents in RAG system
        print("\nStoring test documents...")

        test_documents = [
            {
                "content": "Edge computing brings computation and data storage closer to the location where it is needed, to improve response times and save bandwidth.",
                "title": "Edge Computing Introduction",
                "metadata": {"topic": "edge_computing", "difficulty": "basic"},
            },
            {
                "content": "Mobile devices have limited battery life and processing power, requiring optimized algorithms for efficient computation.",
                "title": "Mobile Computing Constraints",
                "metadata": {"topic": "mobile_computing", "difficulty": "intermediate"},
            },
            {
                "content": "Distributed systems enable multiple computers to work together as a single system, providing scalability and fault tolerance.",
                "title": "Distributed Systems Overview",
                "metadata": {"topic": "distributed_systems", "difficulty": "advanced"},
            },
        ]

        for doc in test_documents:
            result = await rag.store_document(
                content=doc["content"], title=doc["title"], metadata=doc["metadata"], memory_type=MemoryType.ALL
            )

            if "error" not in result:
                print(f"  SUCCESS: Stored '{doc['title']}'")
            else:
                print(f"  ERROR: Failed to store '{doc['title']}': {result['error']}")

        # Test device-optimized queries
        print("\nTesting device-optimized queries...")

        query = "edge computing and mobile devices"

        # Test query with mobile device context (low battery, cellular)
        print("  Mobile device query (low battery, cellular)...")
        mobile_context = {
            "device_id": "mobile_001",
            "device_profile": mobile_device,
            "optimize_for_battery": True,
            "optimize_for_data": True,
        }

        mobile_result = await rag.query(
            query=query, mode=QueryMode.FAST, context=mobile_context  # Use fast mode for mobile
        )

        print(
            f"    Mobile query results: {len(getattr(mobile_result, 'results', getattr(mobile_result, 'primary_sources', [])))}"
        )
        print(
            f"    Mobile execution time: {getattr(mobile_result, 'execution_time_ms', getattr(mobile_result, 'total_latency_ms', 0)):.1f}ms"
        )

        # Test query with laptop context (high performance)
        print("  Laptop device query (high performance)...")
        laptop_context = {
            "device_id": "laptop_001",
            "device_profile": laptop_device,
            "optimize_for_battery": False,
            "optimize_for_data": False,
        }

        laptop_result = await rag.query(
            query=query, mode=QueryMode.COMPREHENSIVE, context=laptop_context  # Use comprehensive mode for laptop
        )

        print(
            f"    Laptop query results: {len(getattr(laptop_result, 'results', getattr(laptop_result, 'primary_sources', [])))}"
        )
        print(
            f"    Laptop execution time: {getattr(laptop_result, 'execution_time_ms', getattr(laptop_result, 'total_latency_ms', 0)):.1f}ms"
        )

        # Test query with IoT device context (offline preferred)
        print("  IoT device query (offline preferred)...")
        iot_context = {
            "device_id": "iot_001",
            "device_profile": iot_device,
            "prefer_offline": True,
            "max_memory_mb": 100,
        }

        iot_result = await rag.query(query=query, mode=QueryMode.FAST, context=iot_context)  # Use fast mode for IoT

        print(
            f"    IoT query results: {len(getattr(iot_result, 'results', getattr(iot_result, 'primary_sources', [])))}"
        )
        print(
            f"    IoT execution time: {getattr(iot_result, 'execution_time_ms', getattr(iot_result, 'total_latency_ms', 0)):.1f}ms"
        )

        # Test resource optimization
        print("\nTesting resource optimization...")

        for device in devices:
            # Choose appropriate query mode based on device constraints
            if device.resource_constraint == ResourceConstraint.SEVERE:
                query_mode = QueryMode.FAST
            elif device.resource_constraint == ResourceConstraint.MODERATE:
                query_mode = QueryMode.BALANCED
            else:
                query_mode = QueryMode.COMPREHENSIVE

            optimization = await edge_bridge.optimize_for_device(
                device_id=device.device_id, query=query, query_mode=query_mode
            )

            print(f"  {device.device_id} optimizations:")
            # The result might be a dict instead of an object
            if isinstance(optimization, dict):
                print(f"    Optimization result: {optimization}")
            else:
                print(f"    Chunk size: {getattr(optimization, 'optimized_chunk_size', 'N/A')}")
                print(f"    Max results: {getattr(optimization, 'optimized_max_results', 'N/A')}")
                print(f"    Preferred systems: {getattr(optimization, 'preferred_systems', [])}")
                print(f"    Max memory: {getattr(optimization, 'max_memory_mb', 'N/A')}MB")
                print(f"    Use local cache: {getattr(optimization, 'use_local_cache', 'N/A')}")
                print(f"    Offline fallback: {getattr(optimization, 'offline_fallback_enabled', 'N/A')}")

        # Test device coordination status
        print("\nChecking device coordination status...")
        bridge_stats = await edge_bridge.get_bridge_statistics()

        registered_info = bridge_stats.get("registered_devices", {})
        optimization_info = bridge_stats.get("optimization_metrics", {})

        print(f"  Registered devices: {registered_info.get('total', 0)}")
        print(f"  Device types: {registered_info.get('by_type', {})}")
        print(f"  Resource constraints: {registered_info.get('by_constraint', {})}")
        print(f"  Queries optimized: {optimization_info.get('queries_optimized', 0)}")
        print(f"  Cache hit rate: {optimization_info.get('cache_hit_rate', 0.0):.2f}")

        # Test offline capability simulation
        print("\nTesting offline capability simulation...")

        # Simulate offline scenario for mobile device
        offline_mobile_dict = mobile_device.__dict__.copy()
        offline_mobile_dict["network_type"] = "offline"
        offline_mobile_dict["prefer_offline"] = True
        offline_mobile = EdgeDeviceProfile(**offline_mobile_dict)

        offline_context = {"device_profile": offline_mobile, "offline_mode": True, "use_cached_only": True}

        offline_result = await rag.query(query="mobile computing", mode=QueryMode.FAST, context=offline_context)

        print(
            f"  Offline query results: {len(getattr(offline_result, 'results', getattr(offline_result, 'primary_sources', [])))}"
        )
        print(
            f"  Offline execution time: {getattr(offline_result, 'execution_time_ms', getattr(offline_result, 'total_latency_ms', 0)):.1f}ms"
        )

        # Test system status
        print("\nChecking integrated system status...")
        rag_status = await rag.get_system_status()
        print(f"  RAG system status: {list(rag_status.keys())}")

        # Clean up
        await rag.close()
        if hasattr(edge_bridge, "close"):
            await edge_bridge.close()

        print("\nSUCCESS: RAG + Edge Device integration test completed!")
        return True

    except Exception as e:
        print(f"\nERROR: Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup
        Path(temp_db).unlink(missing_ok=True)


if __name__ == "__main__":
    success = asyncio.run(test_rag_edge_integration())
    if success:
        print("RESULT: RAG + Edge Device integration test PASSED")
        sys.exit(0)
    else:
        print("RESULT: RAG + Edge Device integration test FAILED")
        sys.exit(1)
