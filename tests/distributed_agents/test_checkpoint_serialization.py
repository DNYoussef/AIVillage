from unittest.mock import patch

import msgpack
import pickle
import pytest

from src.production.distributed_agents.agent_migration_manager import (
    AgentCheckpoint,
    AgentMigrationManager,
)
from src.production.distributed_agents.distributed_agent_orchestrator import (
    AgentInstance,
    AgentPriority,
    AgentSpec,
    AgentType,
)


class DummyP2P:
    node_id = "test-node"


class DummyOrchestrator:
    def __init__(self, agent_instance: AgentInstance):
        self.active_agents = {agent_instance.instance_id: agent_instance}


@pytest.mark.asyncio
async def test_checkpoint_serialization_backward_compatible():
    spec = AgentSpec(
        agent_type=AgentType.SAGE,
        priority=AgentPriority.HIGH,
        memory_requirement_mb=1.0,
        compute_requirement=1.0,
        specialization="test",
    )
    agent = AgentInstance(instance_id="agent1", agent_spec=spec, device_id="dev")

    with patch.object(AgentMigrationManager, "_start_background_tasks", lambda self: None):
        manager = AgentMigrationManager(DummyP2P(), DummyOrchestrator(agent))

    checkpoint = await manager._create_agent_checkpoint(agent)

    state = msgpack.unpackb(checkpoint.state_data, raw=False)
    assert state["instance_id"] == agent.instance_id

    agent.status = "stopped"
    assert await manager._start_agent_locally_from_checkpoint(checkpoint)
    assert agent.status == "running"

    legacy_state = pickle.dumps({"legacy": True})
    legacy_checkpoint = AgentCheckpoint(
        instance_id=agent.instance_id,
        agent_type=agent.agent_spec.agent_type,
        state_data=legacy_state,
        configuration={},
        size_mb=len(legacy_state) / (1024 * 1024),
    )

    agent.status = "stopped"
    assert await manager._start_agent_locally_from_checkpoint(legacy_checkpoint)
    assert agent.status == "running"

