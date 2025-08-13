"""Final validation summary of completed components."""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_completed_components():
    """Test all completed components."""
    results = {}

    print("FINAL VALIDATION OF COMPLETED COMPONENTS")
    print("=" * 50)

    # 1. Test Enhanced BitNet Compression
    print("\n1. Enhanced BitNet Compression...")
    try:
        from src.agent_forge.compression.bitnet_enhanced import EnhancedBitNetCompressor

        compressor = EnhancedBitNetCompressor()

        # Create simple model and test
        simple_model = compressor._create_simple_model()
        result = compressor.compress_model(simple_model, "test")

        if "error" not in result:
            ratio = result.get("compression_ratio", 0)
            print(f"   PASS: Compression ratio {ratio:.2f}x")
            results["bitnet_compression"] = "PASS"
        else:
            print(f"   FAIL: {result['error'][:50]}")
            results["bitnet_compression"] = "FAIL"

    except Exception as e:
        print(f"   FAIL: {str(e)[:50]}")
        results["bitnet_compression"] = "FAIL"

    # 2. Test Enhanced Transport Manager
    print("\n2. Enhanced Transport Manager...")
    try:
        from src.core.p2p.transport_manager_enhanced import EnhancedTransportManager

        config = {
            "websocket": {"port": 8773},
            "tcp": {"port": 8774},
            "udp": {"port": 8775},
        }
        manager = EnhancedTransportManager("test_final", config)

        started = await manager.start()
        if started:
            status = manager.get_transport_status()
            active_count = len(status["active_transports"])
            await manager.stop()

            print(f"   PASS: {active_count}/3 transports active")
            results["transport_manager"] = "PASS"
        else:
            print("   FAIL: Could not start transport manager")
            results["transport_manager"] = "FAIL"

    except Exception as e:
        print(f"   FAIL: {str(e)[:50]}")
        results["transport_manager"] = "FAIL"

    # 3. Test Complete LibP2P Mesh
    print("\n3. Complete LibP2P Mesh...")
    try:
        from src.core.p2p.libp2p_mesh import (
            LibP2PMeshNetwork,
            MeshConfiguration,
        )

        config = MeshConfiguration("test_final_mesh", 4004)
        mesh = LibP2PMeshNetwork(config)

        started = await mesh.start()
        if started:
            await asyncio.sleep(1)
            discovered = mesh.get_discovered_peers()
            connected = mesh.get_connected_peers()
            mesh.get_network_stats()
            await mesh.stop()

            print(
                f"   PASS: {len(discovered)} peers discovered, {len(connected)} connected"
            )
            results["libp2p_mesh"] = "PASS"
        else:
            print("   FAIL: Could not start mesh network")
            results["libp2p_mesh"] = "FAIL"

    except Exception as e:
        print(f"   FAIL: {str(e)[:50]}")
        results["libp2p_mesh"] = "FAIL"

    # 4. Test Fixed Agent Factory
    print("\n4. Fixed Agent Factory...")
    try:
        from src.production.agent_forge.agent_factory import AgentFactory

        factory = AgentFactory()
        available = factory.list_available_agents()

        if available:
            # Try to create an agent
            agent = factory.create_agent(available[0]["id"])
            agent_type = type(agent).__name__

            print(f"   PASS: Created {agent_type}, {len(available)} types available")
            results["agent_factory"] = "PASS"
        else:
            print("   PARTIAL: Factory works but no agent templates")
            results["agent_factory"] = "PARTIAL"

    except Exception as e:
        print(f"   FAIL: {str(e)[:50]}")
        results["agent_factory"] = "FAIL"

    # 5. Test Fixed ADAS System
    print("\n5. Fixed ADAS System...")
    try:
        from src.agent_forge.adas.adas import ADASTask

        task = ADASTask(
            task_type="final_test", task_content="Test ADAS system functionality"
        )

        # Test prompt generation
        prompt = task.generate_prompt([])
        if prompt and len(prompt) > 0:
            print(f"   PASS: Task created, prompt generated ({len(prompt)} chars)")
            results["adas_system"] = "PASS"
        else:
            print("   PARTIAL: Task created but prompt generation failed")
            results["adas_system"] = "PARTIAL"

    except Exception as e:
        print(f"   FAIL: {str(e)[:50]}")
        results["adas_system"] = "FAIL"

    return results


def main():
    """Main validation."""
    print("Starting Final Component Validation...")

    results = asyncio.run(test_completed_components())

    # Calculate overall results
    total_tests = len(results)
    passed = sum(1 for r in results.values() if r == "PASS")
    partial = sum(1 for r in results.values() if r == "PARTIAL")
    failed = sum(1 for r in results.values() if r == "FAIL")

    success_rate = (passed + partial * 0.5) / total_tests

    print("\n" + "=" * 50)
    print("FINAL VALIDATION RESULTS")
    print("=" * 50)

    print(f"Total Components Tested: {total_tests}")
    print(f"Fully Working (PASS): {passed}")
    print(f"Partially Working (PARTIAL): {partial}")
    print(f"Not Working (FAIL): {failed}")
    print(f"Overall Success Rate: {success_rate:.1%}")

    if success_rate >= 0.8:
        print("\nSTATUS: EXCELLENT - Components ready for production")
    elif success_rate >= 0.6:
        print("\nSTATUS: GOOD - Most components operational")
    elif success_rate >= 0.4:
        print("\nSTATUS: FAIR - Some components need work")
    else:
        print("\nSTATUS: POOR - Major fixes needed")

    # Show individual results
    print("\nDETAILED RESULTS:")
    print("-" * 30)
    for component, status in results.items():
        status_icon = {"PASS": "[PASS]", "PARTIAL": "[WARN]", "FAIL": "[FAIL]"}
        print(f"{status_icon[status]} {component}")

    return success_rate >= 0.6


if __name__ == "__main__":
    main()
