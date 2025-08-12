from unittest.mock import patch

import msgpack
import pytest

from src.core.p2p.p2p_node import PeerCapabilities
from src.production.distributed_agents.agent_migration_manager import (
    AgentCheckpoint,
    AgentMigrationManager,
)
from src.production.distributed_agents.distributed_agent_orchestrator import AgentType


class DummyAgent:
    def __init__(self, instance_id: str) -> None:
        self.instance_id = instance_id
        self.status = "stopped"
        self.device_id = "origin"


class DummyOrchestrator:
    def __init__(self) -> None:
        self.active_agents: dict[str, DummyAgent] = {}
        self.device_agent_assignments: dict[str, list[str]] = {}


class DummyP2P:
    def __init__(self) -> None:
        self.node_id = "local"
        self.peer_registry = {
            "trusted": PeerCapabilities(
                device_id="trusted", cpu_cores=4, ram_mb=4096, trust_score=0.9
            ),
            "evil": PeerCapabilities(
                device_id="evil", cpu_cores=4, ram_mb=4096, trust_score=0.1
            ),
        }

    async def send_to_peer(self, _peer_id: str, _message: dict) -> bool:
        return True


@pytest.mark.asyncio
async def test_untrusted_checkpoint_rejected() -> None:
    p2p = DummyP2P()
    orchestrator = DummyOrchestrator()
    orchestrator.active_agents["agent1"] = DummyAgent("agent1")
    with patch.object(
        AgentMigrationManager, "_start_background_tasks", lambda _self: None
    ):
        manager = AgentMigrationManager(p2p, orchestrator)

    payload = msgpack.dumps(
        {
            "instance_id": "agent1",
            "agent_type": AgentType.KING.value,
            "configuration": {},
            "runtime_state": {},
            "checkpoint_comprehensive": False,
        }
    )
    checkpoint = AgentCheckpoint(
        instance_id="agent1",
        agent_type=AgentType.KING,
        state_data=payload,
        source_device_id="evil",
    )

    with pytest.raises(PermissionError):
        await manager._start_agent_locally_from_checkpoint(checkpoint)  # noqa: SLF001


@pytest.mark.asyncio
async def test_malicious_payload_raises_value_error() -> None:
    p2p = DummyP2P()
    orchestrator = DummyOrchestrator()
    orchestrator.active_agents["agent1"] = DummyAgent("agent1")
    with patch.object(
        AgentMigrationManager, "_start_background_tasks", lambda _self: None
    ):
        manager = AgentMigrationManager(p2p, orchestrator)

    bad_payload = msgpack.dumps({"foo": "bar"})
    checkpoint = AgentCheckpoint(
        instance_id="agent1",
        agent_type=AgentType.KING,
        state_data=bad_payload,
        source_device_id="trusted",
    )

    with pytest.raises(ValueError):  # noqa: PT011
        await manager._start_agent_locally_from_checkpoint(checkpoint)  # noqa: SLF001
