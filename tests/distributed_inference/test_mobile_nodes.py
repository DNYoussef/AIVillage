from unittest.mock import AsyncMock, patch

import pytest
from src.production.distributed_inference.model_sharding_engine import ModelShardingEngine, ShardingStrategy

from packages.core.resources.device_profiler import DeviceProfiler
from packages.core.resources.resource_monitor import ResourceMonitor
from packages.p2p.core.p2p_node import P2PNode, PeerCapabilities


@pytest.fixture
def mock_p2p_node():
    node = AsyncMock(spec=P2PNode)
    node.node_id = "test_node_mobile"
    node.peer_registry = {}
    node.get_suitable_evolution_peers.return_value = []
    node.broadcast_to_peers = AsyncMock(return_value=1)
    return node


@pytest.fixture
def mock_resource_monitor():
    return AsyncMock(spec=ResourceMonitor)


@pytest.fixture
def mock_device_profiler():
    profiler = AsyncMock(spec=DeviceProfiler)
    profiler.current_snapshot = None
    return profiler


@pytest.mark.asyncio
async def test_skip_low_battery_devices(mock_p2p_node, mock_resource_monitor, mock_device_profiler):
    engine = ModelShardingEngine(mock_p2p_node, mock_resource_monitor, mock_device_profiler)

    high_peer = PeerCapabilities(
        device_id="high",
        cpu_cores=4,
        ram_mb=8192,
        battery_percent=80,
        trust_score=0.9,
        evolution_capacity=0.9,
    )
    low_peer = PeerCapabilities(
        device_id="low",
        cpu_cores=4,
        ram_mb=8192,
        battery_percent=10,
        trust_score=0.9,
        evolution_capacity=0.9,
    )

    mock_p2p_node.local_capabilities = high_peer
    mock_p2p_node.get_suitable_evolution_peers.return_value = [high_peer, low_peer]
    mock_p2p_node.peer_registry = {"high": high_peer, "low": low_peer}

    analysis = {
        "model_path": "dummy",
        "num_layers": 4,
        "layer_memory_mb": 10.0,
        "layer_compute_score": 1.0,
    }

    with (
        patch.object(ModelShardingEngine, "_analyze_model", AsyncMock(return_value=analysis)),
        patch.object(ModelShardingEngine, "_activate_sharding_plan", AsyncMock()),
    ):
        plan = await engine.shard_model("dummy", ShardingStrategy.MEMORY_AWARE)

    used_devices = {shard.device_id for shard in plan.shards}
    assert "low" not in used_devices
    assert "high" in used_devices
