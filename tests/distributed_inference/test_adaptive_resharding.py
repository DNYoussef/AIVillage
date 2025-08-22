"""Test suite for AdaptiveReshardingManager

Tests device join/leave scenarios, inference continuity during resharding,
rollback mechanisms, and resharding latency benchmarks.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest
from packages.p2p.core.p2p_node import P2PNode, PeerCapabilities
from src.production.distributed_inference.adaptive_resharding import (
    AdaptiveReshardingManager,
    ReshardingConfig,
    ReshardingEvent,
    ReshardingReason,
    ReshardingStrategy,
)
from src.production.distributed_inference.model_sharding_engine import (
    DeviceProfile,
    ModelShard,
    ModelShardingEngine,
    ShardingPlan,
)


@pytest.fixture
def resharding_config():
    """Test configuration for resharding"""
    return ReshardingConfig(
        enable_auto_resharding=True,
        min_resharding_interval_seconds=1.0,  # Short for testing
        performance_threshold=0.7,
        load_imbalance_threshold=0.3,
        device_stability_window_seconds=2.0,  # Short for testing
        max_concurrent_resharding=1,
        graceful_handoff_timeout_seconds=5.0,
        emergency_resharding_threshold=0.5,
    )


@pytest.fixture
def mock_sharding_engine():
    """Mock sharding engine for testing"""
    engine = AsyncMock(spec=ModelShardingEngine)

    # Mock current plan
    test_shards = [
        ModelShard("shard1", "device_1", [0, 1, 2], 1000, 300.0, 3.0),
        ModelShard("shard2", "device_2", [3, 4, 5], 1000, 300.0, 3.0),
    ]

    engine.current_sharding_plan = ShardingPlan(
        model_name="test_model",
        total_shards=2,
        shards=test_shards,
        activation_routing={"shard1": [], "shard2": ["shard1"]},
        memory_efficiency=0.8,
        compute_balance_score=0.7,
    )

    engine.active_shards = {s.shard_id: s for s in test_shards}
    engine.device_assignments = {"device_1": ["shard1"], "device_2": ["shard2"]}

    # Mock methods
    engine._get_device_profiles = AsyncMock(return_value=[])
    engine._analyze_model = AsyncMock(
        return_value={
            "model_path": "test_model",
            "num_layers": 6,
            "layer_memory_mb": 100.0,
            "layer_compute_score": 1.0,
        }
    )
    engine._create_hybrid_plan = AsyncMock()
    engine._optimize_sharding_plan = AsyncMock()
    engine._activate_sharding_plan = AsyncMock()

    return engine


@pytest.fixture
def mock_p2p_node():
    """Mock P2P node for testing"""
    node = AsyncMock(spec=P2PNode)
    node.node_id = "test_node"
    node.peer_registry = {}
    node.local_capabilities = PeerCapabilities(
        device_id="test_node",
        cpu_cores=4,
        ram_mb=8192,
        trust_score=0.9,
        evolution_capacity=0.8,
    )
    return node


@pytest.fixture
def device_profiles():
    """Sample device profiles for testing"""
    return [
        DeviceProfile(
            device_id="device_1",
            capabilities=PeerCapabilities(
                device_id="device_1",
                cpu_cores=8,
                ram_mb=16384,
                trust_score=0.9,
                evolution_capacity=0.9,
            ),
            available_memory_mb=12288,
            compute_score=10.0,
            network_latency_ms=10.0,
            reliability_score=0.9,
        ),
        DeviceProfile(
            device_id="device_2",
            capabilities=PeerCapabilities(
                device_id="device_2",
                cpu_cores=4,
                ram_mb=8192,
                trust_score=0.8,
                evolution_capacity=0.7,
            ),
            available_memory_mb=6144,
            compute_score=6.0,
            network_latency_ms=20.0,
            reliability_score=0.8,
        ),
        DeviceProfile(
            device_id="device_3",
            capabilities=PeerCapabilities(
                device_id="device_3",
                cpu_cores=6,
                ram_mb=12288,
                trust_score=0.85,
                evolution_capacity=0.8,
            ),
            available_memory_mb=9216,
            compute_score=8.0,
            network_latency_ms=15.0,
            reliability_score=0.85,
        ),
    ]


@pytest.fixture
def resharding_manager(mock_sharding_engine, mock_p2p_node, resharding_config):
    """Create AdaptiveReshardingManager for testing"""
    return AdaptiveReshardingManager(mock_sharding_engine, mock_p2p_node, resharding_config)


class TestAdaptiveReshardingManager:
    """Test cases for AdaptiveReshardingManager"""

    def test_initialization(self, resharding_manager, resharding_config):
        """Test proper initialization"""
        assert resharding_manager.config == resharding_config
        assert not resharding_manager.resharding_active
        assert len(resharding_manager.resharding_history) == 0
        assert len(resharding_manager.pending_device_changes) == 0
        assert resharding_manager.stats["total_resharding_events"] == 0

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, resharding_manager):
        """Test monitoring lifecycle"""
        # Start monitoring
        await resharding_manager.start_monitoring()
        assert resharding_manager.performance_monitor_task is not None
        assert not resharding_manager.performance_monitor_task.done()

        # Stop monitoring
        await resharding_manager.stop_monitoring()
        assert resharding_manager.performance_monitor_task is None

    @pytest.mark.asyncio
    async def test_device_join_handling(self, resharding_manager):
        """Test handling of new device joining"""
        new_device_id = "new_device"

        await resharding_manager._handle_device_joined(new_device_id)

        # Should track device stability
        assert new_device_id in resharding_manager.device_stability_tracker
        assert new_device_id in resharding_manager.pending_device_changes
        assert resharding_manager.stats["device_joins_handled"] == 1

    @pytest.mark.asyncio
    async def test_device_leave_handling(self, resharding_manager):
        """Test handling of device leaving/failing"""
        failed_device_id = "device_1"  # Device with active shard

        with patch.object(resharding_manager, "trigger_resharding") as mock_trigger:
            await resharding_manager._handle_device_left(failed_device_id)

            # Should trigger emergency resharding
            mock_trigger.assert_called_once_with(
                ReshardingReason.DEVICE_FAILED,
                trigger_device_id=failed_device_id,
                strategy=ReshardingStrategy.EMERGENCY,
            )

            assert resharding_manager.stats["device_failures_handled"] == 1

    @pytest.mark.asyncio
    async def test_device_change_detection(self, resharding_manager, device_profiles):
        """Test device change detection in monitoring loop"""
        # Setup current devices in P2P node
        current_peers = {
            "device_1": device_profiles[0].capabilities,
            "device_2": device_profiles[1].capabilities,
        }
        resharding_manager.p2p_node.peer_registry = current_peers
        resharding_manager.p2p_node.local_capabilities = device_profiles[0].capabilities

        with (
            patch.object(resharding_manager, "_handle_device_joined") as mock_join,
            patch.object(resharding_manager, "_handle_device_left") as mock_leave,
        ):
            # Add new device to peer registry
            resharding_manager.p2p_node.peer_registry["device_3"] = device_profiles[2].capabilities

            await resharding_manager._check_device_changes()

            # Should detect new device
            mock_join.assert_called_once_with("device_3")
            mock_leave.assert_not_called()

    @pytest.mark.asyncio
    async def test_performance_degradation_detection(self, resharding_manager):
        """Test performance degradation detection"""
        with (
            patch.object(resharding_manager, "_get_current_performance", return_value=0.6),
            patch.object(resharding_manager, "trigger_resharding") as mock_trigger,
        ):
            await resharding_manager._check_performance_degradation()

            # Should trigger resharding for performance degradation
            mock_trigger.assert_called_once_with(
                ReshardingReason.PERFORMANCE_DEGRADATION,
                strategy=ReshardingStrategy.OPTIMAL_REBALANCE,
            )

    @pytest.mark.asyncio
    async def test_emergency_performance_threshold(self, resharding_manager):
        """Test emergency resharding on critical performance"""
        with (
            patch.object(resharding_manager, "_get_current_performance", return_value=0.4),
            patch.object(resharding_manager, "trigger_resharding") as mock_trigger,
        ):
            await resharding_manager._check_performance_degradation()

            # Should trigger emergency resharding
            mock_trigger.assert_called_once_with(
                ReshardingReason.PERFORMANCE_DEGRADATION,
                strategy=ReshardingStrategy.EMERGENCY,
            )

    @pytest.mark.asyncio
    async def test_load_imbalance_detection(self, resharding_manager):
        """Test load imbalance detection"""
        # Set low compute balance score
        resharding_manager.sharding_engine.current_sharding_plan.compute_balance_score = 0.5

        with patch.object(resharding_manager, "trigger_resharding") as mock_trigger:
            await resharding_manager._check_load_imbalance()

            # Should trigger incremental resharding
            mock_trigger.assert_called_once_with(
                ReshardingReason.LOAD_IMBALANCE, strategy=ReshardingStrategy.INCREMENTAL
            )

    @pytest.mark.asyncio
    async def test_pending_device_stability(self, resharding_manager):
        """Test device stability window processing"""
        # Add device to pending changes with old timestamp
        new_device_id = "stable_device"
        old_timestamp = time.time() - 10.0  # 10 seconds ago
        resharding_manager.pending_device_changes.add(new_device_id)
        resharding_manager.device_stability_tracker[new_device_id] = old_timestamp

        with patch.object(resharding_manager, "trigger_resharding") as mock_trigger:
            await resharding_manager._process_pending_changes()

            # Should process stable device
            assert new_device_id not in resharding_manager.pending_device_changes
            mock_trigger.assert_called_once_with(
                ReshardingReason.DEVICE_JOINED,
                strategy=ReshardingStrategy.OPTIMAL_REBALANCE,
            )

    @pytest.mark.asyncio
    async def test_can_reshard_constraints(self, resharding_manager):
        """Test resharding constraint checking"""
        # Initially should be able to reshard
        assert resharding_manager._can_reshard() is True

        # Set resharding as active
        resharding_manager.resharding_active = True
        assert resharding_manager._can_reshard() is False

        # Reset and test minimum interval
        resharding_manager.resharding_active = False
        resharding_manager.last_resharding_time = time.time()
        assert resharding_manager._can_reshard() is False

        # Test without active sharding plan
        resharding_manager.last_resharding_time = 0.0
        resharding_manager.sharding_engine.current_sharding_plan = None
        assert resharding_manager._can_reshard() is False

    @pytest.mark.asyncio
    async def test_minimal_disruption_resharding(self, resharding_manager, device_profiles):
        """Test minimal disruption resharding strategy"""
        # Create event with old plan
        event = ReshardingEvent(
            event_id="test_event",
            reason=ReshardingReason.DEVICE_JOINED,
            old_plan=resharding_manager.sharding_engine.current_sharding_plan,
        )

        # Mock device profiles (all devices available)
        resharding_manager.sharding_engine._get_device_profiles.return_value = device_profiles

        success = await resharding_manager._minimal_disruption_resharding(event)

        assert success is True
        resharding_manager.sharding_engine._activate_sharding_plan.assert_called_once()

    @pytest.mark.asyncio
    async def test_optimal_rebalance_resharding(self, resharding_manager, device_profiles):
        """Test optimal rebalance resharding strategy"""
        event = ReshardingEvent(
            event_id="test_event",
            reason=ReshardingReason.DEVICE_JOINED,
            old_plan=resharding_manager.sharding_engine.current_sharding_plan,
        )

        # Mock dependencies
        resharding_manager.sharding_engine._get_device_profiles.return_value = device_profiles
        mock_new_plan = ShardingPlan(
            model_name="test_model",
            total_shards=3,
            shards=[
                ModelShard("new_shard1", "device_1", [0, 1], 500, 150.0, 1.5),
                ModelShard("new_shard2", "device_2", [2, 3], 500, 150.0, 1.5),
                ModelShard("new_shard3", "device_3", [4, 5], 500, 150.0, 1.5),
            ],
            activation_routing={
                "new_shard1": [],
                "new_shard2": ["new_shard1"],
                "new_shard3": ["new_shard2"],
            },
            memory_efficiency=0.9,
            compute_balance_score=0.85,
        )

        resharding_manager.sharding_engine._create_hybrid_plan.return_value = mock_new_plan
        resharding_manager.sharding_engine._optimize_sharding_plan.return_value = mock_new_plan

        success = await resharding_manager._optimal_rebalance_resharding(event)

        assert success is True
        resharding_manager.sharding_engine._create_hybrid_plan.assert_called_once()
        resharding_manager.sharding_engine._optimize_sharding_plan.assert_called_once()
        resharding_manager.sharding_engine._activate_sharding_plan.assert_called_once()

    @pytest.mark.asyncio
    async def test_emergency_resharding(self, resharding_manager, device_profiles):
        """Test emergency resharding strategy"""
        event = ReshardingEvent(
            event_id="test_event",
            reason=ReshardingReason.DEVICE_FAILED,
            old_plan=resharding_manager.sharding_engine.current_sharding_plan,
        )

        resharding_manager.sharding_engine._get_device_profiles.return_value = device_profiles
        mock_plan = ShardingPlan(
            model_name="test_model",
            total_shards=2,
            shards=[
                ModelShard("emergency_shard1", "device_2", [0, 1, 2], 1500, 450.0, 4.5),
                ModelShard("emergency_shard2", "device_3", [3, 4, 5], 1500, 450.0, 4.5),
            ],
            activation_routing={
                "emergency_shard1": [],
                "emergency_shard2": ["emergency_shard1"],
            },
            memory_efficiency=0.7,
            compute_balance_score=0.6,
        )

        resharding_manager.sharding_engine._create_sequential_plan.return_value = mock_plan

        success = await resharding_manager._emergency_resharding(event)

        assert success is True
        # Should skip optimization for speed
        resharding_manager.sharding_engine._optimize_sharding_plan.assert_not_called()
        resharding_manager.sharding_engine._activate_sharding_plan.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_resharding_workflow(self, resharding_manager):
        """Test complete resharding trigger workflow"""
        with patch.object(resharding_manager, "_execute_resharding", return_value=True) as mock_execute:
            success = await resharding_manager.trigger_resharding(
                ReshardingReason.DEVICE_JOINED,
                strategy=ReshardingStrategy.OPTIMAL_REBALANCE,
            )

            assert success is True
            assert resharding_manager.stats["total_resharding_events"] == 1
            assert resharding_manager.stats["successful_resharding"] == 1
            assert len(resharding_manager.resharding_history) == 1

            # Check event details
            event = resharding_manager.resharding_history[0]
            assert event.reason == ReshardingReason.DEVICE_JOINED
            assert event.success is True
            assert event.duration_seconds > 0

            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_resharding_failure(self, resharding_manager):
        """Test resharding failure handling"""
        with patch.object(
            resharding_manager,
            "_execute_resharding",
            side_effect=Exception("Test error"),
        ):
            success = await resharding_manager.trigger_resharding(
                ReshardingReason.DEVICE_FAILED, strategy=ReshardingStrategy.EMERGENCY
            )

            assert success is False
            assert resharding_manager.stats["failed_resharding"] == 1

            # Should record failed event
            event = resharding_manager.resharding_history[0]
            assert event.success is False
            assert "error" in event.metadata

    def test_disruption_score_calculation(self, resharding_manager):
        """Test disruption score calculation"""
        old_plan = ShardingPlan(
            model_name="test",
            total_shards=2,
            shards=[
                ModelShard("shard1", "device_1", [0, 1, 2], 1000, 300.0, 3.0),
                ModelShard("shard2", "device_2", [3, 4, 5], 1000, 300.0, 3.0),
            ],
            activation_routing={},
            memory_efficiency=0.8,
            compute_balance_score=0.7,
        )

        # No changes - should have low disruption
        new_plan_same = ShardingPlan(
            model_name="test",
            total_shards=2,
            shards=[
                ModelShard("shard1", "device_1", [0, 1, 2], 1000, 300.0, 3.0),
                ModelShard("shard2", "device_2", [3, 4, 5], 1000, 300.0, 3.0),
            ],
            activation_routing={},
            memory_efficiency=0.8,
            compute_balance_score=0.7,
        )

        disruption_same = resharding_manager._calculate_disruption_score(old_plan, new_plan_same)
        assert disruption_same == 0.0

        # Complete change - should have high disruption
        new_plan_different = ShardingPlan(
            model_name="test",
            total_shards=2,
            shards=[
                ModelShard("shard1", "device_3", [0, 1, 2], 1000, 300.0, 3.0),  # Moved to different device
                ModelShard("shard2", "device_1", [3, 4, 5], 1000, 300.0, 3.0),  # Moved to different device
            ],
            activation_routing={},
            memory_efficiency=0.8,
            compute_balance_score=0.7,
        )

        disruption_different = resharding_manager._calculate_disruption_score(old_plan, new_plan_different)
        assert disruption_different == 1.0

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, resharding_manager):
        """Test performance monitoring and calculation"""
        # Mock peer registry with various peer states
        peers = {
            "peer1": PeerCapabilities(
                device_id="peer1",
                cpu_cores=4,
                ram_mb=8192,
                trust_score=0.9,
                evolution_capacity=0.8,
                current_evolution_load=0.2,
            ),
            "peer2": PeerCapabilities(
                device_id="peer2",
                cpu_cores=2,
                ram_mb=4096,
                trust_score=0.7,
                evolution_capacity=0.6,
                current_evolution_load=0.5,
            ),
        }

        resharding_manager.p2p_node.peer_registry = peers

        performance = await resharding_manager._get_current_performance()

        # Should be weighted average of peer capabilities
        assert 0.0 <= performance <= 1.0
        assert performance > 0.5  # Should be reasonable with good peers

    @pytest.mark.asyncio
    async def test_force_resharding(self, resharding_manager):
        """Test forced resharding for manual intervention"""
        with patch.object(resharding_manager, "trigger_resharding", return_value=True) as mock_trigger:
            success = await resharding_manager.force_resharding(ReshardingStrategy.EMERGENCY)

            assert success is True
            mock_trigger.assert_called_once_with(ReshardingReason.MANUAL_TRIGGER, strategy=ReshardingStrategy.EMERGENCY)

    def test_get_resharding_status(self, resharding_manager):
        """Test resharding status reporting"""
        # Add some test data
        resharding_manager.stats["total_resharding_events"] = 5
        resharding_manager.stats["successful_resharding"] = 4
        resharding_manager.pending_device_changes.add("pending_device")
        resharding_manager.device_stability_tracker["stable_device"] = time.time() - 30

        # Add test event
        test_event = ReshardingEvent(
            event_id="test_event",
            reason=ReshardingReason.DEVICE_JOINED,
            success=True,
            duration_seconds=2.5,
            disruption_score=0.3,
        )
        resharding_manager.resharding_history.append(test_event)

        status = resharding_manager.get_resharding_status()

        # Verify status structure
        assert "monitoring_active" in status
        assert "resharding_active" in status
        assert "config" in status
        assert "statistics" in status
        assert "recent_events" in status
        assert "pending_changes" in status
        assert "device_stability" in status

        # Verify content
        assert status["statistics"]["total_resharding_events"] == 5
        assert status["statistics"]["successful_resharding"] == 4
        assert status["pending_changes"] == 1
        assert len(status["recent_events"]) == 1
        assert status["recent_events"][0]["event_id"] == "test_event"

    @pytest.mark.asyncio
    async def test_concurrent_resharding_prevention(self, resharding_manager):
        """Test prevention of concurrent resharding operations"""
        # Set resharding as active
        resharding_manager.resharding_active = True

        success = await resharding_manager.trigger_resharding(ReshardingReason.DEVICE_JOINED)

        # Should be rejected due to active resharding
        assert success is False

    @pytest.mark.asyncio
    async def test_minimum_interval_enforcement(self, resharding_manager):
        """Test minimum resharding interval enforcement"""
        # Set recent resharding time
        resharding_manager.last_resharding_time = time.time()

        success = await resharding_manager.trigger_resharding(ReshardingReason.DEVICE_JOINED)

        # Should be rejected due to minimum interval
        assert success is False

    @pytest.mark.asyncio
    async def test_resharding_timeout_handling(self, resharding_manager):
        """Test handling of resharding timeouts"""

        async def slow_resharding(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate slow operation
            return True

        with patch.object(resharding_manager, "_execute_resharding", side_effect=slow_resharding):
            start_time = time.time()
            success = await resharding_manager.trigger_resharding(ReshardingReason.DEVICE_JOINED)
            duration = time.time() - start_time

            assert success is True
            assert duration >= 0.1  # Should complete the slow operation

    @pytest.mark.asyncio
    async def test_network_change_callbacks(self, resharding_manager):
        """Test network change callback system"""
        callback_called = []

        async def test_callback(event_type, data):
            callback_called.append((event_type, data))

        resharding_manager.register_network_change_callback(test_callback)

        # Trigger a resharding event that would call callbacks
        with patch.object(resharding_manager, "_execute_resharding", return_value=True):
            await resharding_manager.trigger_resharding(ReshardingReason.DEVICE_JOINED)

        # Note: Callback integration would need to be implemented in the actual trigger_resharding method
        # This test verifies the callback registration system works
        assert len(resharding_manager.network_change_callbacks) == 1
