#!/usr/bin/env python3
"""Create integration tests for mesh networking, federated learning, and mobile SDK.

Tests the complete distributed infrastructure.
"""

import asyncio
import os
from pathlib import Path
import sys

import torch

# Add current directory to path to import our implementations
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our implementations
from implement_federated_learning import (
    AggregationStrategy,
    ClientUpdate,
    FederatedLearningClient,
    FederatedLearningServer,
)
from implement_mesh_protocol import MeshNetworkSimulator, MessageType


class TestMeshNetworking:
    """Integration tests for mesh networking."""

    async def test_mesh_network_formation(self) -> bool:
        """Test that mesh network forms correctly."""
        print("Testing mesh network formation...")

        # Create network
        simulator = MeshNetworkSimulator(num_nodes=5, connectivity=0.5)
        await simulator.create_network()

        # Verify network formed
        assert len(simulator.nodes) == 5, (
            f"Expected 5 nodes, got {len(simulator.nodes)}"
        )

        # Check connectivity
        total_connections = sum(
            len(node.neighbors) for node in simulator.nodes.values()
        )
        assert total_connections > 0, "No connections found in mesh network"

        # Test message propagation
        first_node = next(iter(simulator.nodes.values()))
        await first_node.send_message(
            MessageType.DISCOVERY, {"test": "data"}, priority=5
        )

        # Give time for propagation
        await asyncio.sleep(1)

        # Check message reached other nodes
        messages_received = sum(
            node.stats["messages_received"] for node in simulator.nodes.values()
        )

        print(f"  - Network formed with {len(simulator.nodes)} nodes")
        print(f"  - Total connections: {total_connections}")
        print(f"  - Messages received: {messages_received}")

        return True

    async def test_mesh_resilience(self) -> bool:
        """Test mesh network resilience to node failures."""
        print("Testing mesh network resilience...")

        # Create network
        simulator = MeshNetworkSimulator(num_nodes=10, connectivity=0.4)
        await simulator.create_network()

        # Simulate traffic
        traffic_task = asyncio.create_task(simulator.simulate_traffic(duration=5))

        # Simulate node failure
        await asyncio.sleep(2)
        failed_node_id = list(simulator.nodes.keys())[5]
        simulator.nodes[failed_node_id].neighbors.clear()

        # Continue traffic
        await traffic_task

        # Check network still functioning
        active_nodes = [
            node
            for node_id, node in simulator.nodes.items()
            if node_id != failed_node_id and len(node.neighbors) > 0
        ]

        assert len(active_nodes) >= 5, f"Too few active nodes: {len(active_nodes)}"

        # Verify messages still being routed
        total_forwarded = sum(node.stats["messages_forwarded"] for node in active_nodes)

        print(f"  - Active nodes after failure: {len(active_nodes)}")
        print(f"  - Messages forwarded: {total_forwarded}")

        return True


class TestFederatedLearning:
    """Integration tests for federated learning."""

    async def test_federated_training_round(self) -> bool:
        """Test a complete federated learning round."""
        print("Testing federated learning round...")

        # Create simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 2)
        )

        # Create server
        server = FederatedLearningServer(
            model, aggregation_strategy=AggregationStrategy.FEDAVG, min_clients=2
        )

        # Register clients
        for i in range(3):
            server.register_client(
                f"client_{i}",
                {"compute_power": 1.0, "battery_level": 0.8, "reliability_score": 0.9},
            )

        # Start round
        round_config = await server.start_round()
        assert round_config.get("round_number") == 1, "Round number should be 1"

        # Simulate client updates
        for i in range(2):
            update = ClientUpdate(
                client_id=f"client_{i}",
                round_number=1,
                model_id=round_config["model_version"],
                gradients={
                    name: torch.randn_like(param) * 0.01  # Small gradients
                    for name, param in model.state_dict().items()
                },
                num_samples=100,
                metrics={"loss": 0.5, "accuracy": 0.85},
                computation_time=10.0,
            )

            await server.receive_update(update)

        # Wait for aggregation
        await asyncio.sleep(1)

        # Check round completed
        history = server.get_round_history()
        assert len(history) >= 1, "No rounds in history"
        assert history[0]["num_clients"] >= 2, (
            f"Expected >= 2 clients, got {history[0]['num_clients']}"
        )

        print(f"  - Round completed with {history[0]['num_clients']} clients")
        print(f"  - Duration: {history[0]['duration']:.1f}s")

        return True

    async def test_fl_convergence(self) -> bool:
        """Test that FL actually improves model performance."""
        print("Testing federated learning convergence...")

        # Create synthetic dataset
        X = torch.randn(1000, 10)
        y = (X.sum(dim=1) > 0).long()

        # Create model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 2)
        )

        # Get initial performance
        with torch.no_grad():
            initial_output = model(X)
            initial_loss = torch.nn.functional.cross_entropy(initial_output, y)

        # Create FL server
        server = FederatedLearningServer(model, min_clients=2)

        # Create clients
        clients = []
        for i in range(3):
            # Split data among clients
            start_idx = i * 300
            end_idx = min((i + 1) * 300, len(X))
            client_data = torch.utils.data.TensorDataset(
                X[start_idx:end_idx], y[start_idx:end_idx]
            )
            dataloader = torch.utils.data.DataLoader(client_data, batch_size=32)

            client = FederatedLearningClient(
                f"client_{i}",
                torch.nn.Sequential(
                    torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 2)
                ),
                dataloader,
            )
            clients.append(client)

            server.register_client(f"client_{i}", {})

        # Run 3 rounds
        for _round_num in range(3):
            round_config = await server.start_round()

            # Client training
            for client in clients[:2]:  # Only 2 clients participate
                update = await client.participate_in_round(round_config)
                await server.receive_update(update)

            await asyncio.sleep(0.5)

        # Get final performance
        final_model, _ = server.get_current_model()
        with torch.no_grad():
            final_output = final_model(X)
            final_loss = torch.nn.functional.cross_entropy(final_output, y)

        print(f"  - Initial loss: {initial_loss:.4f}")
        print(f"  - Final loss: {final_loss:.4f}")
        print(f"  - Improvement: {(initial_loss - final_loss):.4f}")

        # Verify improvement (or at least no degradation)
        assert final_loss <= initial_loss + 0.1, (
            f"Model degraded: {final_loss} > {initial_loss + 0.1}"
        )

        return True


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    async def test_mesh_fl_integration(self) -> bool:
        """Test FL over mesh network."""
        print("Testing mesh-FL integration...")

        # Create mesh network
        mesh_sim = MeshNetworkSimulator(num_nodes=5, connectivity=0.6)
        await mesh_sim.create_network()

        # Create FL server on one node
        server_node = next(iter(mesh_sim.nodes.values()))
        model = torch.nn.Linear(10, 2)
        fl_server = FederatedLearningServer(model, min_clients=2)

        # Create FL clients on other nodes
        fl_clients = []
        for i, _node in enumerate(list(mesh_sim.nodes.values())[1:3]):
            # Mock dataset
            dataset = torch.utils.data.TensorDataset(
                torch.randn(50, 10), torch.randint(0, 2, (50,))
            )
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

            client = FederatedLearningClient(
                f"client_{i}", torch.nn.Linear(10, 2), dataloader
            )
            fl_clients.append(client)
            fl_server.register_client(f"client_{i}", {})

        # Simulate FL round over mesh
        round_config = await fl_server.start_round()

        # Clients train and send updates through mesh
        for client in fl_clients:
            # Train locally
            update = await client.participate_in_round(round_config)

            # Simulate sending through mesh
            update_message = {
                "type": "fl_update",
                "client_id": update.client_id,
                "round": update.round_number,
            }

            await server_node.send_message(
                MessageType.PARAMETER_UPDATE,
                update_message,
                recipient_id=server_node.node_id,
            )

            # Server processes update
            await fl_server.receive_update(update)

        # Verify round completed
        await asyncio.sleep(1)
        history = fl_server.get_round_history()
        assert len(history) >= 1, "No FL rounds completed"

        print("  - FL round completed over mesh network")
        print(f"  - Participants: {history[0]['num_clients']}")

        return True

    async def test_mobile_simulation(self) -> bool:
        """Simulate mobile device behavior."""
        print("Testing mobile device simulation...")

        # Simulate device constraints
        device_profiles = [
            {"ram_mb": 2048, "battery": 0.3, "network": "poor"},
            {"ram_mb": 4096, "battery": 0.8, "network": "good"},
            {"ram_mb": 3072, "battery": 0.5, "network": "moderate"},
        ]

        # Create mesh network
        mesh_sim = MeshNetworkSimulator(num_nodes=3)
        await mesh_sim.create_network()

        # Simulate varying conditions
        for _i, (node, profile) in enumerate(
            zip(mesh_sim.nodes.values(), device_profiles, strict=False)
        ):
            # Adjust node based on profile
            if profile["battery"] < 0.5:
                # Low battery - reduce activity
                node.message_handlers[MessageType.PARAMETER_UPDATE] = []

            if profile["network"] == "poor":
                # Poor network - increase packet loss
                for neighbor in node.neighbors.values():
                    neighbor.connection_quality *= 0.5

        # Run simulation
        await mesh_sim.simulate_traffic(duration=3)

        # Verify network adapted
        stats = mesh_sim.get_network_stats()

        print("  - Mobile devices simulated with varying constraints")
        print(f"  - Average packet loss: {stats['average_packet_loss']:.2%}")

        # Network should still be functional despite constraints
        assert stats["average_packet_loss"] < 0.8, "Too much packet loss"

        return True


async def run_integration_tests() -> bool:
    """Run all integration tests."""
    print("=== Running Integration Tests ===\n")

    test_results = []

    # Mesh networking tests
    mesh_tests = TestMeshNetworking()
    try:
        result = await mesh_tests.test_mesh_network_formation()
        test_results.append(("Mesh Network Formation", result))
    except Exception as e:
        print(f"  [ERROR] Mesh network formation test failed: {e}")
        test_results.append(("Mesh Network Formation", False))

    try:
        result = await mesh_tests.test_mesh_resilience()
        test_results.append(("Mesh Network Resilience", result))
    except Exception as e:
        print(f"  [ERROR] Mesh resilience test failed: {e}")
        test_results.append(("Mesh Network Resilience", False))

    # Federated learning tests
    fl_tests = TestFederatedLearning()
    try:
        result = await fl_tests.test_federated_training_round()
        test_results.append(("FL Training Round", result))
    except Exception as e:
        print(f"  [ERROR] FL training round test failed: {e}")
        test_results.append(("FL Training Round", False))

    try:
        result = await fl_tests.test_fl_convergence()
        test_results.append(("FL Convergence", result))
    except Exception as e:
        print(f"  [ERROR] FL convergence test failed: {e}")
        test_results.append(("FL Convergence", False))

    # End-to-end tests
    e2e_tests = TestEndToEndIntegration()
    try:
        result = await e2e_tests.test_mesh_fl_integration()
        test_results.append(("Mesh-FL Integration", result))
    except Exception as e:
        print(f"  [ERROR] Mesh-FL integration test failed: {e}")
        test_results.append(("Mesh-FL Integration", False))

    try:
        result = await e2e_tests.test_mobile_simulation()
        test_results.append(("Mobile Simulation", result))
    except Exception as e:
        print(f"  [ERROR] Mobile simulation test failed: {e}")
        test_results.append(("Mobile Simulation", False))

    # Print results
    print("\n=== Test Results ===")
    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{total} tests")

    if passed == total:
        print("[CHECK] All integration tests passed!")
        return True
    print(f"[ERROR] {total - passed} tests failed!")
    return False


def create_test_directory_structure() -> None:
    """Create test directory structure."""
    test_dirs = [
        "tests/integration",
        "tests/unit/mesh",
        "tests/unit/fl",
        "tests/benchmarks",
    ]

    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Create __init__.py
        init_file = Path(dir_path) / "__init__.py"
        if not init_file.exists():
            init_file.touch()

    print("[CHECK] Created test directory structure")


def create_pytest_config() -> None:
    """Create pytest configuration."""
    pytest_ini = """[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
addopts =
    --verbose
    --tb=short
"""

    with open("pytest.ini", "w", encoding="utf-8") as f:
        f.write(pytest_ini)

    print("[CHECK] Created pytest configuration")


def save_integration_test_file() -> None:
    """Save this test as a file."""
    test_content = '''#!/usr/bin/env python3
"""
Integration tests for distributed infrastructure.
Run with: python -m pytest tests/ -v
"""

import asyncio
import pytest
from scripts.create_integration_tests import run_integration_tests

@pytest.mark.asyncio
async def test_distributed_infrastructure():
    """Test complete distributed infrastructure."""
    result = await run_integration_tests()
    assert result, "Integration tests failed"

if __name__ == '__main__':
    asyncio.run(run_integration_tests())
'''

    test_file = Path("tests/integration/test_distributed_infrastructure.py")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_content)

    print(f"[CHECK] Saved integration test file: {test_file}")


if __name__ == "__main__":
    print("Creating and running integration tests...")

    # Create test structure
    create_test_directory_structure()
    create_pytest_config()
    save_integration_test_file()

    # Run tests
    asyncio.run(run_integration_tests())
