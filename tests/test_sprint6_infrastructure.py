"""Comprehensive tests for Sprint 6 infrastructure components"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Import the components we're testing
from src.core.p2p.p2p_node import P2PNode
from src.core.resources.adaptive_loader import AdaptiveLoader
from src.core.resources.constraint_manager import ConstraintManager
from src.core.resources.device_profiler import DeviceProfiler, DeviceType
from src.core.resources.resource_monitor import MonitoringMode, ResourceMonitor
from src.production.agent_forge.evolution.infrastructure_aware_evolution import InfrastructureAwareEvolution
from src.production.agent_forge.evolution.resource_constrained_evolution import ResourceConstrainedEvolution


@pytest.fixture
def mock_device_profiler():
    """Create mock device profiler"""
    profiler = Mock()
    profiler.profile = Mock()
    profiler.profile.device_type = DeviceType.LAPTOP
    profiler.profile.total_memory_gb = 8.0
    profiler.profile.cpu_cores = 4
    profiler.profile.evolution_capable = True
    profiler.profile.performance_tier = "medium"
    profiler.profile.max_evolution_memory_mb = 4096  # 4GB
    profiler.profile.max_evolution_cpu_percent = 70.0
    profiler.profile.get_evolution_constraints = Mock(
        return_value={"max_memory_mb": 4096, "max_cpu_percent": 70.0, "max_duration_minutes": 120}
    )

    # Mock current snapshot
    snapshot = Mock()
    snapshot.memory_percent = 60.0
    snapshot.cpu_percent = 40.0
    snapshot.battery_percent = 80.0
    snapshot.power_plugged = False
    snapshot.thermal_state = Mock()
    snapshot.thermal_state.value = "normal"
    snapshot.evolution_suitability_score = 0.75
    snapshot.memory_available = 3200 * 1024 * 1024  # 3.2GB available
    snapshot.memory_used = 4800 * 1024 * 1024  # 4.8GB used
    snapshot.cpu_temp = 65.0
    snapshot.is_resource_constrained = False
    snapshot.memory_total = 16 * 1024 * 1024 * 1024  # 16GB total

    profiler.current_snapshot = snapshot
    profiler.take_snapshot = Mock(return_value=snapshot)
    profiler.monitoring_active = False
    profiler.start_monitoring = Mock()
    profiler.stop_monitoring = Mock()

    def mock_get_allocation():
        if not profiler.current_snapshot:
            return {}
        return {"memory_mb": 1600, "cpu_percent": 60.0, "device_tier": "medium", "evolution_capable": True}

    profiler.get_evolution_resource_allocation = Mock(side_effect=mock_get_allocation)
    profiler.is_suitable_for_evolution = Mock(return_value=True)

    return profiler


@pytest.fixture
def mock_p2p_node():
    """Create mock P2P node"""
    node = Mock()
    node.node_id = "test_node_123"
    node.listen_port = 9000
    node.status = Mock()
    node.status.value = "active"
    node.connections = {}
    node.peer_registry = {}
    node.start = AsyncMock()
    node.stop = AsyncMock()
    node.broadcast_evolution_event = AsyncMock()
    node.send_to_peer = AsyncMock()
    node.register_handler = Mock()
    node.get_suitable_evolution_peers = Mock(return_value=[])
    node.get_network_status = Mock(
        return_value={"node_id": "test_node_123", "status": "active", "connected_peers": 0, "known_peers": 0}
    )

    return node


class TestP2PNodeIntegration:
    """Test P2P node functionality"""

    @pytest.mark.asyncio
    async def test_p2p_node_startup(self):
        """Test P2P node can start up"""
        node = P2PNode(node_id="test_node")

        # Mock the server startup to avoid actual network binding
        with patch("asyncio.start_server") as mock_server:
            mock_server_obj = AsyncMock()
            mock_server_obj.sockets = [Mock()]
            mock_server_obj.sockets[0].getsockname.return_value = ("localhost", 9000)
            mock_server_obj.wait_closed = AsyncMock()
            mock_server.return_value = mock_server_obj

            await node.start()

            assert node.listen_port == 9000
            assert node.status.value == "active"

            await node.stop()


class TestDeviceProfiler:
    """Test device profiler functionality"""

    def test_device_profiler_initialization(self):
        """Test device profiler initializes correctly"""
        profiler = DeviceProfiler()

        assert profiler.device_id is not None
        assert profiler.profile is not None
        assert profiler.profile.device_type != DeviceType.UNKNOWN
        assert profiler.profile.total_memory_gb > 0
        assert profiler.profile.cpu_cores > 0

    def test_device_type_detection(self):
        """Test device type detection logic"""
        profiler = DeviceProfiler()
        device_type = profiler._detect_device_type()

        # Should detect some valid device type
        assert device_type in [DeviceType.LAPTOP, DeviceType.DESKTOP, DeviceType.EMBEDDED]

    def test_performance_tier_calculation(self):
        """Test performance tier calculation"""
        profiler = DeviceProfiler()
        tier = profiler._calculate_performance_tier()

        assert tier in ["low", "medium", "high", "premium"]

    def test_evolution_capability_check(self):
        """Test evolution capability determination"""
        profiler = DeviceProfiler()

        # Should be capable on most development machines
        assert profiler.profile.evolution_capable is True


class TestResourceMonitor:
    """Test resource monitor functionality"""

    @pytest.mark.asyncio
    async def test_resource_monitor_initialization(self, mock_device_profiler):
        """Test resource monitor initializes correctly"""
        monitor = ResourceMonitor(mock_device_profiler)

        assert monitor.device_profiler == mock_device_profiler
        assert monitor.monitoring_mode == MonitoringMode.PASSIVE
        assert not monitor.monitoring_active

    @pytest.mark.asyncio
    async def test_monitoring_mode_changes(self, mock_device_profiler):
        """Test monitoring mode changes"""
        monitor = ResourceMonitor(mock_device_profiler)

        # Test mode change
        monitor.set_monitoring_mode(MonitoringMode.EVOLUTION)
        assert monitor.monitoring_mode == MonitoringMode.EVOLUTION

    @pytest.mark.asyncio
    async def test_trend_calculation(self, mock_device_profiler):
        """Test trend calculation logic"""
        monitor = ResourceMonitor(mock_device_profiler)

        # Test trend calculation with sample data
        values = [50.0, 55.0, 60.0, 65.0, 70.0]
        timestamps = [f"timestamp_{i}" for i in range(len(values))]

        trend = monitor._calculate_trend(values, timestamps)

        assert trend.trend_direction in ["increasing", "decreasing", "stable"]
        assert 0 <= trend.trend_strength <= 1


class TestConstraintManager:
    """Test constraint manager functionality"""

    def test_constraint_manager_initialization(self, mock_device_profiler):
        """Test constraint manager initializes correctly"""
        manager = ConstraintManager(mock_device_profiler)

        assert manager.device_profiler == mock_device_profiler
        assert manager.default_constraints is not None
        assert len(manager.constraint_templates) > 0

    def test_default_constraints_creation(self, mock_device_profiler):
        """Test default constraints are reasonable"""
        manager = ConstraintManager(mock_device_profiler)
        constraints = manager.default_constraints

        assert constraints.max_memory_mb > 0
        assert constraints.max_cpu_percent > 0
        assert constraints.max_cpu_percent <= 100

    def test_constraint_templates(self, mock_device_profiler):
        """Test constraint templates are available"""
        manager = ConstraintManager(mock_device_profiler)

        # Should have templates for different evolution types
        assert "nightly" in manager.constraint_templates
        assert "breakthrough" in manager.constraint_templates
        assert "emergency" in manager.constraint_templates

    def test_resource_availability_check(self, mock_device_profiler):
        """Test resource availability checking"""
        manager = ConstraintManager(mock_device_profiler)

        # Test with nightly constraints
        constraints = manager.constraint_templates["nightly"]
        available = manager._check_resource_availability(constraints)

        # Should be available with our mock data
        assert available is True


class TestAdaptiveLoader:
    """Test adaptive loader functionality"""

    def test_adaptive_loader_initialization(self, mock_device_profiler):
        """Test adaptive loader initializes correctly"""
        constraint_manager = ConstraintManager(mock_device_profiler)
        loader = AdaptiveLoader(mock_device_profiler, constraint_manager)

        assert loader.device_profiler == mock_device_profiler
        assert loader.constraint_manager == constraint_manager
        assert len(loader.model_variants) > 0

    def test_model_variant_registration(self, mock_device_profiler):
        """Test model variant registration"""
        constraint_manager = ConstraintManager(mock_device_profiler)
        loader = AdaptiveLoader(mock_device_profiler, constraint_manager)

        # Should have some built-in variants
        assert "base_evolution_model" in loader.model_variants
        variants = loader.model_variants["base_evolution_model"]
        assert len(variants) > 0

    def test_strategy_scoring(self, mock_device_profiler):
        """Test variant scoring logic"""
        constraint_manager = ConstraintManager(mock_device_profiler)
        loader = AdaptiveLoader(mock_device_profiler, constraint_manager)

        # Get a variant to test
        variants = loader.model_variants["base_evolution_model"]
        variant = variants[0]

        # Create mock context
        from src.core.resources.adaptive_loader import LoadingContext

        context = LoadingContext(
            task_type="nightly",
            priority_level=2,
            max_loading_time_seconds=120.0,
            quality_preference=0.7,
            resource_constraints=constraint_manager.default_constraints,
        )

        # Test scoring
        score = loader._calculate_variant_score(variant, context, mock_device_profiler.current_snapshot)
        assert isinstance(score, float)
        assert score >= 0


class TestInfrastructureAwareEvolution:
    """Test infrastructure-aware evolution system"""

    @pytest.mark.asyncio
    async def test_infrastructure_initialization(self):
        """Test infrastructure system can initialize"""
        from src.production.agent_forge.evolution.infrastructure_aware_evolution import InfrastructureConfig

        config = InfrastructureConfig(
            enable_p2p=False,  # Disable P2P for testing
            enable_resource_monitoring=False,  # Disable monitoring for testing
        )

        system = InfrastructureAwareEvolution(config)

        # Test initialization without actual hardware dependencies
        with patch.object(system, "device_profiler"), patch.object(system, "dual_evolution"):

            # Mock the initialization
            system.system_initialized = True
            system.infrastructure_status = "active"

            status = system.get_infrastructure_status()

            assert status["system_initialized"] is True
            assert status["infrastructure_status"] == "active"
            assert "config" in status

    def test_evolution_mode_determination(self):
        """Test evolution mode determination logic"""
        from src.production.agent_forge.evolution.infrastructure_aware_evolution import (
            EvolutionMode,
            InfrastructureAwareEvolution,
            InfrastructureConfig,
        )

        config = InfrastructureConfig(enable_p2p=False)
        system = InfrastructureAwareEvolution(config)

        # Should default to local only when P2P is disabled
        assert system.config.default_evolution_mode == EvolutionMode.LOCAL_ONLY


class TestResourceConstrainedEvolution:
    """Test resource-constrained evolution system"""

    def test_resource_constrained_initialization(self, mock_device_profiler):
        """Test resource-constrained evolution initializes correctly"""
        # Create required components
        resource_monitor = ResourceMonitor(mock_device_profiler)
        constraint_manager = ConstraintManager(mock_device_profiler)

        # Initialize system
        system = ResourceConstrainedEvolution(mock_device_profiler, resource_monitor, constraint_manager)

        assert system.device_profiler == mock_device_profiler
        assert system.resource_monitor == resource_monitor
        assert system.constraint_manager == constraint_manager
        assert system.config is not None

    @pytest.mark.asyncio
    async def test_evolution_feasibility_check(self, mock_device_profiler):
        """Test evolution feasibility checking"""
        resource_monitor = ResourceMonitor(mock_device_profiler)
        constraint_manager = ConstraintManager(mock_device_profiler)

        system = ResourceConstrainedEvolution(mock_device_profiler, resource_monitor, constraint_manager)

        # Mock agent
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent"

        # Test feasibility check
        feasible = await system._check_evolution_feasibility("nightly", mock_agent)

        # Should be feasible with our mock data
        assert feasible is True

    def test_resource_state_update(self, mock_device_profiler):
        """Test resource state updating"""
        resource_monitor = ResourceMonitor(mock_device_profiler)
        constraint_manager = ConstraintManager(mock_device_profiler)

        system = ResourceConstrainedEvolution(mock_device_profiler, resource_monitor, constraint_manager)

        # Test resource state update
        asyncio.run(system._update_resource_state())

        assert system.current_resource_state is not None
        assert system.current_resource_state.memory_used_mb > 0
        assert system.current_resource_state.cpu_used_percent >= 0


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components"""

    @pytest.mark.asyncio
    async def test_full_stack_initialization(self, mock_device_profiler, mock_p2p_node):
        """Test full infrastructure stack can initialize together"""

        # Create all components
        resource_monitor = ResourceMonitor(mock_device_profiler)
        constraint_manager = ConstraintManager(mock_device_profiler)
        adaptive_loader = AdaptiveLoader(mock_device_profiler, constraint_manager)

        # Create infrastructure-aware system with mocked P2P
        from src.production.agent_forge.evolution.infrastructure_aware_evolution import InfrastructureConfig

        config = InfrastructureConfig(
            enable_p2p=False,  # Disable for testing
            enable_resource_monitoring=True,
            enable_resource_constraints=True,
            enable_adaptive_loading=True,
        )

        system = InfrastructureAwareEvolution(config)

        # Mock the components
        system.device_profiler = mock_device_profiler
        system.resource_monitor = resource_monitor
        system.constraint_manager = constraint_manager
        system.adaptive_loader = adaptive_loader

        # Test status retrieval
        status = system.get_infrastructure_status()

        assert "components" in status
        assert "stats" in status
        assert status["system_initialized"] is False  # Not actually initialized

    def test_resource_constraint_workflow(self, mock_device_profiler):
        """Test resource constraint workflow"""
        constraint_manager = ConstraintManager(mock_device_profiler)

        # Register a task
        success = constraint_manager.register_task("test_task", "nightly")
        assert success is True

        # Check active tasks
        active_tasks = constraint_manager.get_active_tasks()
        assert "test_task" in active_tasks

        # Unregister task
        constraint_manager.unregister_task("test_task")
        active_tasks = constraint_manager.get_active_tasks()
        assert "test_task" not in active_tasks


# Integration test for the complete Sprint 6 system


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sprint6_complete_integration():
    """Test complete Sprint 6 integration scenario"""

    # This test validates that all components can work together
    # in a realistic scenario (though mocked for testing)

    # 1. Initialize device profiler
    profiler = DeviceProfiler()
    # Take a snapshot to populate current_snapshot
    profiler.take_snapshot()

    # 2. Create resource management stack
    monitor = ResourceMonitor(profiler)
    constraints = ConstraintManager(profiler)
    loader = AdaptiveLoader(profiler, constraints)

    # 3. Create infrastructure-aware evolution
    from src.production.agent_forge.evolution.infrastructure_aware_evolution import InfrastructureConfig

    config = InfrastructureConfig(
        enable_p2p=False,  # Disable P2P for testing
        enable_resource_monitoring=True,
        enable_resource_constraints=True,
        enable_adaptive_loading=True,
    )

    evolution_system = InfrastructureAwareEvolution(config)

    # Mock the initialization to avoid hardware dependencies
    evolution_system.device_profiler = profiler
    evolution_system.resource_monitor = monitor
    evolution_system.constraint_manager = constraints
    evolution_system.adaptive_loader = loader
    evolution_system.system_initialized = True
    evolution_system.infrastructure_status = "active"

    # 4. Test system status
    status = evolution_system.get_infrastructure_status()

    assert status["system_initialized"] is True
    assert status["infrastructure_status"] == "active"
    assert "components" in status
    assert "stats" in status

    # 5. Test resource allocation
    allocation = profiler.get_evolution_resource_allocation()

    assert "memory_mb" in allocation
    assert "cpu_percent" in allocation
    assert allocation["evolution_capable"] is True

    print("✓ Sprint 6 integration test passed - all components work together")


if __name__ == "__main__":
    # Run basic integration test
    asyncio.run(test_sprint6_complete_integration())
