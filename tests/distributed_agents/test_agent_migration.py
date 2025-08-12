from unittest.mock import patch

import pytest

from src.production.distributed_agents.agent_migration_manager import (
    AgentMigrationManager,
    MigrationReason,
)


@pytest.mark.asyncio
async def test_agent_migration_stress(orchestrator_setup):
    orchestrator, p2p = orchestrator_setup
    await orchestrator.deploy_agent_constellation()

    # Collect a subset of agents that are migratable
    migratable = [
        agent_id
        for agent_id, inst in orchestrator.active_agents.items()
        if inst.agent_spec.can_migrate
    ][:5]
    assert migratable, "No migratable agents available for testing"

    with patch.object(
        AgentMigrationManager, "_start_background_tasks", lambda self: None
    ):
        manager = AgentMigrationManager(p2p, orchestrator)

    for agent_id in migratable:
        await manager.request_migration(agent_id, MigrationReason.LOAD_BALANCING)

    assert manager.stats["migrations_requested"] == len(migratable)
    assert len(manager.pending_migrations) == len(migratable)
