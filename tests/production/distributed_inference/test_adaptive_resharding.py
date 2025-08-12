import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from src.core.p2p.p2p_node import P2PNode, PeerCapabilities
from src.production.distributed_inference.adaptive_resharding import (
    AdaptiveReshardingManager,
    ReshardingConfig,
    ReshardingEvent,
    ReshardingReason,
    ReshardingStrategy,
)
from src.production.distributed_inference.model_sharding_engine import (
    ModelShard,
    ModelShardingEngine,
    ShardingPlan,
)


@pytest.fixture
def resharding_config():
    return ReshardingConfig(
        enable_auto_resharding=True,
        min_resharding_interval_seconds=0.0,
        performance_threshold=0.7,
        load_imbalance_threshold=0.3,
        device_stability_window_seconds=0.0,
        max_concurrent_resharding=1,
        graceful_handoff_timeout_seconds=5.0,
        emergency_resharding_threshold=0.5,
    )


@pytest.fixture
def mock_sharding_engine():
    engine = AsyncMock(spec=ModelShardingEngine)
    shard = ModelShard("shard1", "device_1", [0], 1000, 300.0, 3.0)
    engine.current_sharding_plan = ShardingPlan(
        model_name="test",
        total_shards=1,
        shards=[shard],
        activation_routing={},
        memory_efficiency=1.0,
        compute_balance_score=1.0,
    )
    return engine


@pytest.fixture
def mock_p2p_node():
    node = AsyncMock(spec=P2PNode)
    node.node_id = "node"
    node.peer_registry = {}
    node.local_capabilities = PeerCapabilities(
        device_id="node", cpu_cores=4, ram_mb=4096, trust_score=1.0, evolution_capacity=1.0
    )
    return node


@pytest.mark.asyncio
async def test_event_serialized_after_success(tmp_path, mock_sharding_engine, mock_p2p_node, resharding_config):
    state_file = tmp_path / "reshard.json"
    manager = AdaptiveReshardingManager(
        mock_sharding_engine, mock_p2p_node, resharding_config, state_file=str(state_file)
    )
    manager._execute_resharding = AsyncMock(return_value=True)
    await manager.trigger_resharding(
        ReshardingReason.MANUAL_TRIGGER, strategy=ReshardingStrategy.OPTIMAL_REBALANCE
    )
    assert state_file.exists()
    with state_file.open() as f:
        data = json.load(f)
    assert data[-1]["success"] is True


@pytest.mark.asyncio
async def test_incomplete_event_replayed_on_restart(tmp_path, mock_sharding_engine, mock_p2p_node, resharding_config):
    state_file = tmp_path / "reshard.json"
    manager = AdaptiveReshardingManager(
        mock_sharding_engine, mock_p2p_node, resharding_config, state_file=str(state_file)
    )
    # Simulate pending event on disk
    event = ReshardingEvent(event_id="1", reason=ReshardingReason.MANUAL_TRIGGER)
    manager.resharding_history.append(event)
    manager._save_history_to_disk()

    # Restart manager and ensure pending event is replayed
    manager2 = AdaptiveReshardingManager(
        mock_sharding_engine, mock_p2p_node, resharding_config, state_file=str(state_file)
    )
    manager2._execute_resharding = AsyncMock(return_value=True)
    await manager2.start_monitoring()
    manager2._execute_resharding.assert_called_once()
    await manager2.stop_monitoring()
