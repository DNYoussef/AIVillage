from dataclasses import dataclass
import time
from typing import Any

import pytest

from src.production.distributed_agents.distributed_agent_orchestrator import (
    AgentInstance,
    DistributedAgentOrchestrator,
)


@dataclass
class DeviceProfile:
    """Minimal device profile for orchestrator tests."""

    device_id: str
    capabilities: Any = None
    available_memory_mb: float = 0.0
    compute_score: float = 0.0
    network_latency_ms: float = 0.0
    battery_level: float | None = None
    reliability_score: float = 1.0


class DummyShardingEngine:
    """Stub sharding engine returning predefined device profiles."""

    def __init__(self, devices: list[DeviceProfile]):
        self.devices = devices

    async def _get_device_profiles(self, target_devices=None):  # - matches orchestrator expectations
        return self.devices


class DummyP2PNode:
    """Stub P2P node capturing network calls."""

    def __init__(self, node_id: str = "dev1"):
        self.node_id = node_id
        self.broadcasts = []
        self.sent_messages = []

    async def send_to_peer(self, device_id, message):  # - simple passthrough
        self.sent_messages.append((device_id, message))
        return True

    async def broadcast_to_peers(self, event, message):  # - simple passthrough
        self.broadcasts.append((event, message))
        return True


class DummyResourceMonitor:
    """Placeholder resource monitor."""


async def _fast_deploy(self, instance):
    """Instant local deployment used to speed up tests."""
    instance.status = "running"
    instance.last_heartbeat = time.time()
    instance.resource_usage = {
        "memory_mb": instance.agent_spec.memory_requirement_mb,
        "compute": instance.agent_spec.compute_requirement,
    }
    return True


@pytest.fixture
def orchestrator_setup(monkeypatch):
    """Provide a configured orchestrator and its P2P stub."""
    devices = [
        DeviceProfile(
            device_id="dev1",
            available_memory_mb=8000,
            compute_score=40,
            network_latency_ms=10,
            battery_level=100,
        ),
        DeviceProfile(
            device_id="dev2",
            available_memory_mb=8000,
            compute_score=40,
            network_latency_ms=20,
            battery_level=100,
        ),
    ]
    sharding = DummyShardingEngine(devices)
    p2p = DummyP2PNode("dev1")
    monitor = DummyResourceMonitor()
    orch = DistributedAgentOrchestrator(p2p, monitor, sharding)

    # Speed up local deployments
    monkeypatch.setattr(DistributedAgentOrchestrator, "_deploy_agent_locally", _fast_deploy)
    monkeypatch.setattr(AgentInstance, "__hash__", lambda self: hash(self.instance_id))

    return orch, p2p
