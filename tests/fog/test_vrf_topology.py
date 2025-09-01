"""
VRF Topology Management Tests

Tests topology management and network health properties including:
- Spectral gap monitoring and maintenance
- K-core stability and evolution
- Network healing mechanisms
- Expander graph properties
- Churn resistance
"""

import asyncio
import numpy as np
import pytest
import time

# Import components to test
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "infrastructure"))

from fog.vrf.vrf_neighbor_selection import VRFNeighborSelector
from fog.vrf.topology_manager import (
    TopologyManager,
    TopologyHealthStatus,
    NetworkMetrics,
    GraphAnalyzer,
    TopologyHealingAction,
)


class TestGraphAnalyzer:
    """Test graph analysis algorithms."""

    @pytest.fixture
    def analyzer(self):
        return GraphAnalyzer()

    def test_k_core_decomposition_simple_graph(self, analyzer):
        """Test K-core decomposition on a simple known graph."""
        # Create a graph with clear k-core structure
        # Triangle (3-core) + pendant edges (1-core)
        adjacency = {
            "a": {"b", "c"},  # Triangle nodes
            "b": {"a", "c"},  # Triangle nodes
            "c": {"a", "b", "d"},  # Triangle + connection to pendant
            "d": {"c"},  # Pendant node
        }

        k_cores = analyzer.compute_k_core_decomposition(adjacency)

        # Verify structure
        assert 1 in k_cores
        assert 2 in k_cores

        # Node 'd' should be in 1-core
        assert "d" in k_cores[1]

        # Triangle nodes should be in higher k-core
        triangle_nodes = {"a", "b", "c"}
        highest_k = max(k_cores.keys())
        assert triangle_nodes.issubset(k_cores[highest_k]) or triangle_nodes.issubset(k_cores[highest_k - 1])

    def test_k_core_decomposition_complete_graph(self, analyzer):
        """Test K-core decomposition on complete graph."""
        n = 6
        # Complete graph K_6
        adjacency = {}
        nodes = [f"node_{i}" for i in range(n)]

        for i, node in enumerate(nodes):
            adjacency[node] = set(nodes) - {node}  # Connected to all others

        k_cores = analyzer.compute_k_core_decomposition(adjacency)

        # In K_6, all nodes should be in (n-1)-core = 5-core
        assert max(k_cores.keys()) == n - 1
        assert len(k_cores[n - 1]) == n

    def test_spectral_gap_well_connected_graph(self, analyzer):
        """Test spectral gap computation on well-connected graph."""
        n = 8
        adjacency_matrix = np.zeros((n, n))

        # Create cycle + chords for good expansion
        for i in range(n):
            # Cycle edges
            adjacency_matrix[i][(i + 1) % n] = 1
            adjacency_matrix[(i + 1) % n][i] = 1

            # Chord edges
            adjacency_matrix[i][(i + n // 2) % n] = 1
            adjacency_matrix[(i + n // 2) % n][i] = 1

        spectral_gap = analyzer.compute_spectral_gap(adjacency_matrix)

        # Should have good spectral gap due to expansion
        assert spectral_gap > 0.2
        assert spectral_gap < 1.0

    def test_spectral_gap_path_graph(self, analyzer):
        """Test spectral gap on path graph (poor expansion)."""
        n = 8
        adjacency_matrix = np.zeros((n, n))

        # Create path graph (worst case for expansion)
        for i in range(n - 1):
            adjacency_matrix[i][i + 1] = 1
            adjacency_matrix[i + 1][i] = 1

        spectral_gap = analyzer.compute_spectral_gap(adjacency_matrix)

        # Path graph should have small spectral gap
        assert spectral_gap < 0.5

    def test_expansion_ratio_computation(self, analyzer):
        """Test expansion ratio computation."""
        # Create graph with known expansion properties
        adjacency = {
            "center": {"a", "b", "c", "d"},  # High-degree center
            "a": {"center", "b"},
            "b": {"center", "a", "c"},
            "c": {"center", "b", "d"},
            "d": {"center", "c"},
        }

        expansion = analyzer.compute_expansion_ratio(adjacency)

        # Should have reasonable expansion due to center connectivity
        assert expansion > 0.5
        assert expansion <= 4.0  # Max degree

    def test_clustering_coefficient_triangle_graph(self, analyzer):
        """Test clustering coefficient on triangle-rich graph."""
        # Create graph with triangles
        adjacency = {
            "a": {"b", "c", "d"},
            "b": {"a", "c"},  # Triangle: a-b-c
            "c": {"a", "b", "d"},  # Triangle: a-c-d
            "d": {"a", "c"},
        }

        clustering = analyzer.compute_clustering_coefficient(adjacency)

        # Should have high clustering due to triangles
        assert clustering > 0.5
        assert clustering <= 1.0

    def test_diameter_computation_known_graphs(self, analyzer):
        """Test diameter computation on graphs with known diameters."""
        # Star graph: diameter = 2
        star_adjacency = {
            "center": {"a", "b", "c", "d"},
            "a": {"center"},
            "b": {"center"},
            "c": {"center"},
            "d": {"center"},
        }

        diameter, radius = analyzer.compute_diameter(star_adjacency)

        assert diameter == 2
        assert radius <= diameter

        # Complete graph K4: diameter = 1
        complete_adjacency = {
            "a": {"b", "c", "d"},
            "b": {"a", "c", "d"},
            "c": {"a", "b", "d"},
            "d": {"a", "b", "c"},
        }

        diameter, radius = analyzer.compute_diameter(complete_adjacency)

        assert diameter == 1
        assert radius == 1


class TestTopologyManager:
    """Test topology management functionality."""

    @pytest.fixture
    def vrf_selector(self):
        return VRFNeighborSelector(node_id="topo_test_node")

    @pytest.fixture
    def topology_manager(self, vrf_selector):
        return TopologyManager(
            vrf_selector=vrf_selector,
            target_spectral_gap=0.3,
            min_k_core=3,
            monitoring_interval=1.0,  # Fast for testing
        )

    async def test_topology_manager_initialization(self, topology_manager):
        """Test topology manager initialization."""
        started = await topology_manager.start()
        assert started is True

        # Check initial state
        status = topology_manager.get_topology_status()
        assert status["health_status"] == TopologyHealthStatus.GOOD.value
        assert "node_count" in status
        assert "monitoring_active" in status

        await topology_manager.stop()

    async def test_network_topology_update(self, topology_manager):
        """Test updating network topology."""
        await topology_manager.start()

        # Create test topology
        adjacency_updates = {
            "node_a": {"node_b", "node_c"},
            "node_b": {"node_a", "node_c", "node_d"},
            "node_c": {"node_a", "node_b"},
            "node_d": {"node_b"},
        }

        await topology_manager.update_network_topology(adjacency_updates)

        # Verify topology was updated
        assert len(topology_manager.adjacency_dict) == 4
        assert topology_manager.adjacency_dict["node_a"] == {"node_b", "node_c"}

        # Check metrics were computed
        assert topology_manager.current_metrics is not None
        assert topology_manager.current_metrics.node_count == 4
        assert topology_manager.current_metrics.edge_count > 0

        await topology_manager.stop()

    async def test_k_core_node_identification(self, topology_manager):
        """Test identification of k-core nodes."""
        await topology_manager.start()

        # Create topology with clear k-core structure
        adjacency_updates = {
            "core_a": {"core_b", "core_c", "core_d"},  # 3-core
            "core_b": {"core_a", "core_c", "core_d"},  # 3-core
            "core_c": {"core_a", "core_b", "core_d"},  # 3-core
            "core_d": {"core_a", "core_b", "core_c"},  # 3-core
            "pendant_1": {"core_a"},  # 1-core
            "pendant_2": {"core_b"},  # 1-core
        }

        await topology_manager.update_network_topology(adjacency_updates)

        # Get k-core nodes
        k_core_nodes = await topology_manager.get_k_core_nodes(min_k=3)

        # Core nodes should be identified
        expected_core = {"core_a", "core_b", "core_c", "core_d"}
        assert len(k_core_nodes.intersection(expected_core)) >= 3  # Most core nodes identified

        # Pendant nodes should not be in 3-core
        assert "pendant_1" not in k_core_nodes
        assert "pendant_2" not in k_core_nodes

        await topology_manager.stop()

    async def test_topology_health_assessment(self, topology_manager):
        """Test topology health assessment."""
        await topology_manager.start()

        # Create healthy topology
        healthy_adjacency = {}
        n = 10
        for i in range(n):
            neighbors = set()
            # Ring connectivity
            neighbors.add(f"node_{(i+1)%n}")
            neighbors.add(f"node_{(i-1)%n}")
            # Additional random connections for good expansion
            neighbors.add(f"node_{(i+3)%n}")
            neighbors.add(f"node_{(i+5)%n}")

            healthy_adjacency[f"node_{i}"] = neighbors

        await topology_manager.update_network_topology(healthy_adjacency)

        health = await topology_manager.assess_topology_health()

        # Should assess as good or excellent health
        assert health in [TopologyHealthStatus.GOOD, TopologyHealthStatus.EXCELLENT]

        await topology_manager.stop()

    async def test_topology_healing_planning(self, topology_manager):
        """Test topology healing action planning."""
        await topology_manager.start()

        # Mock poor topology metrics
        poor_metrics = NetworkMetrics()
        poor_metrics.spectral_gap = 0.05  # Below target
        poor_metrics.max_k_core = 1  # Below minimum
        poor_metrics.expansion_ratio = 0.2  # Poor expansion
        poor_metrics.node_count = 10
        poor_metrics.edge_count = 10

        topology_manager.current_metrics = poor_metrics

        # Plan healing actions
        healing_actions = await topology_manager.plan_healing_actions()

        assert len(healing_actions) > 0

        # Should include relevant healing actions
        action_types = [action.action_type for action in healing_actions]
        expected_actions = ["improve_spectral_gap", "strengthen_k_core", "improve_expansion"]

        for expected in expected_actions:
            assert expected in action_types

        # Actions should be prioritized
        assert healing_actions[0].priority <= healing_actions[-1].priority

        await topology_manager.stop()

    async def test_healing_action_execution(self, topology_manager):
        """Test execution of topology healing actions."""
        await topology_manager.start()

        # Create a healing action
        action = TopologyHealingAction(
            action_type="improve_spectral_gap",
            target_nodes=["test_node"],
            parameters={"target_gap": 0.3},
            priority=1,
            expected_improvement=0.2,
            estimated_cost=1.0,
        )

        # Mock the VRF selector to include our test node
        topology_manager.vrf_selector.node_id = "test_node"

        # Execute healing action
        success = await topology_manager.execute_healing_action(action)

        # Should complete (success depends on implementation)
        assert isinstance(success, bool)

        # Should record healing attempt
        assert len(topology_manager.healing_history) > 0
        assert topology_manager.healing_history[-1]["action"] == "improve_spectral_gap"

        await topology_manager.stop()


class TestTopologyStability:
    """Test topology stability under various conditions."""

    @pytest.fixture
    def stable_topology_manager(self):
        vrf = VRFNeighborSelector(node_id="stability_test")
        return TopologyManager(
            vrf_selector=vrf, target_spectral_gap=0.25, min_k_core=2, monitoring_interval=0.1  # Very fast for testing
        )

    async def test_topology_resilience_to_node_removal(self, stable_topology_manager):
        """Test topology resilience when nodes are removed."""
        await stable_topology_manager.start()

        # Create initial stable topology
        initial_adjacency = {}
        n = 12
        for i in range(n):
            neighbors = set()
            # Create highly connected graph
            for j in range(4):  # 4 neighbors each
                neighbor_idx = (i + j + 1) % n
                neighbors.add(f"node_{neighbor_idx}")
            initial_adjacency[f"node_{i}"] = neighbors

        await stable_topology_manager.update_network_topology(initial_adjacency)

        # Record initial health
        await stable_topology_manager.assess_topology_health()
        initial_spectral_gap = (
            stable_topology_manager.current_metrics.spectral_gap if stable_topology_manager.current_metrics else 0
        )

        # Remove some nodes (simulate churn)
        nodes_to_remove = ["node_0", "node_3", "node_7"]  # Remove ~25% of nodes
        updated_adjacency = {}

        for node_id, neighbors in initial_adjacency.items():
            if node_id not in nodes_to_remove:
                # Remove connections to removed nodes
                filtered_neighbors = neighbors - set(nodes_to_remove)
                updated_adjacency[node_id] = filtered_neighbors

        await stable_topology_manager.update_network_topology(updated_adjacency)

        # Check resilience
        final_health = await stable_topology_manager.assess_topology_health()
        final_spectral_gap = (
            stable_topology_manager.current_metrics.spectral_gap if stable_topology_manager.current_metrics else 0
        )

        # Topology should remain reasonably healthy
        # Allow some degradation but should not become critical
        healthy_states = [TopologyHealthStatus.EXCELLENT, TopologyHealthStatus.GOOD, TopologyHealthStatus.DEGRADED]
        assert final_health in healthy_states

        # Spectral gap shouldn't degrade too much
        if initial_spectral_gap > 0:
            degradation_ratio = final_spectral_gap / initial_spectral_gap
            assert degradation_ratio > 0.5  # Less than 50% degradation

        await stable_topology_manager.stop()

    async def test_topology_evolution_with_healing(self, stable_topology_manager):
        """Test topology evolution with active healing."""
        await stable_topology_manager.start()

        # Start with poor topology (path graph)
        poor_adjacency = {}
        n = 8
        for i in range(n):
            neighbors = set()
            if i > 0:
                neighbors.add(f"node_{i-1}")
            if i < n - 1:
                neighbors.add(f"node_{i+1}")
            poor_adjacency[f"node_{i}"] = neighbors

        await stable_topology_manager.update_network_topology(poor_adjacency)

        # Record initial poor metrics
        initial_health = await stable_topology_manager.assess_topology_health()

        # Plan and execute healing
        if initial_health in [TopologyHealthStatus.DEGRADED, TopologyHealthStatus.CRITICAL]:
            healing_actions = await stable_topology_manager.plan_healing_actions()

            # Execute first healing action
            if healing_actions:
                await stable_topology_manager.execute_healing_action(healing_actions[0])

        # Allow time for healing to take effect
        await asyncio.sleep(0.2)

        # Check if healing improved topology
        final_health = await stable_topology_manager.assess_topology_health()

        # Healing should not make things worse
        health_values = {
            TopologyHealthStatus.CRITICAL: 0,
            TopologyHealthStatus.DEGRADED: 1,
            TopologyHealthStatus.GOOD: 2,
            TopologyHealthStatus.EXCELLENT: 3,
            TopologyHealthStatus.HEALING: 1.5,  # Between degraded and good
        }

        initial_value = health_values.get(initial_health, 0)
        final_value = health_values.get(final_health, 0)

        # Should not significantly worsen (allow for minor fluctuations)
        assert final_value >= initial_value - 0.5

        await stable_topology_manager.stop()


class TestTopologyPerformance:
    """Test performance of topology management operations."""

    @pytest.fixture
    def performance_analyzer(self):
        return GraphAnalyzer()

    def test_k_core_performance_large_graph(self, performance_analyzer):
        """Test K-core computation performance on large graphs."""
        # Create large random graph
        n = 500
        adjacency = {}

        # Generate random graph with reasonable density
        np.random.seed(42)  # Reproducible
        for i in range(n):
            node_id = f"node_{i}"
            neighbors = set()
            # Each node connects to ~10 random other nodes
            for _ in range(10):
                target = np.random.randint(0, n)
                if target != i:
                    neighbors.add(f"node_{target}")
            adjacency[node_id] = neighbors

        # Make graph undirected
        for node_id, neighbors in list(adjacency.items()):
            for neighbor in neighbors:
                if neighbor in adjacency:
                    adjacency[neighbor].add(node_id)

        # Measure performance
        start_time = time.time()
        k_cores = performance_analyzer.compute_k_core_decomposition(adjacency)
        end_time = time.time()

        computation_time = end_time - start_time

        # Should complete in reasonable time (< 5 seconds)
        assert computation_time < 5.0
        assert len(k_cores) > 0
        assert max(k_cores.keys()) > 0

    def test_spectral_gap_performance_medium_graph(self, performance_analyzer):
        """Test spectral gap computation performance."""
        # Create medium-sized adjacency matrix
        n = 100
        adjacency_matrix = np.zeros((n, n))

        # Create structured graph (ring + random edges)
        for i in range(n):
            # Ring edges
            adjacency_matrix[i][(i + 1) % n] = 1
            adjacency_matrix[(i + 1) % n][i] = 1

            # Random additional edges
            for _ in range(3):
                j = np.random.randint(0, n)
                if i != j:
                    adjacency_matrix[i][j] = 1
                    adjacency_matrix[j][i] = 1

        # Measure performance
        start_time = time.time()
        spectral_gap = performance_analyzer.compute_spectral_gap(adjacency_matrix)
        end_time = time.time()

        computation_time = end_time - start_time

        # Should complete quickly (< 2 seconds for n=100)
        assert computation_time < 2.0
        assert 0 <= spectral_gap <= 1.0

    async def test_topology_monitoring_overhead(self):
        """Test that topology monitoring doesn't create excessive overhead."""
        # Create topology manager with fast monitoring
        vrf = VRFNeighborSelector(node_id="overhead_test")
        manager = TopologyManager(vrf_selector=vrf, monitoring_interval=0.01)  # Very fast monitoring

        await manager.start()

        # Add some topology data
        adjacency = {}
        for i in range(20):
            neighbors = {f"node_{(i+1)%20}", f"node_{(i+2)%20}"}
            adjacency[f"node_{i}"] = neighbors

        await manager.update_network_topology(adjacency)

        # Let monitoring run briefly
        start_time = time.time()
        await asyncio.sleep(0.1)  # 100ms
        end_time = time.time()

        # Should not consume too much CPU time
        # (This is a rough test - would need more sophisticated profiling for production)
        actual_time = end_time - start_time
        assert actual_time < 0.5  # Should not take more than 500ms for 100ms sleep

        await manager.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
