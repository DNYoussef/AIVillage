import pytest

from src.production.distributed_agents.agent_registry import AgentLocation, DistributedAgentRegistry


@pytest.mark.asyncio
async def test_basic_registration_and_lookup():
    registry = DistributedAgentRegistry()
    await registry.register("agent1", "dev1")
    loc = await registry.get("agent1")
    assert isinstance(loc, AgentLocation)
    assert loc.device_id == "dev1"

    # update location
    updated = await registry.update("agent1", "dev2")
    assert updated
    loc2 = await registry.get("agent1")
    assert loc2.device_id == "dev2"

    # remove
    removed = await registry.remove("agent1")
    assert removed
    assert await registry.get("agent1") is None


@pytest.mark.asyncio
async def test_list_agents_snapshot_is_isolated():
    registry = DistributedAgentRegistry()
    await registry.register("a1", "dev1")
    agents = await registry.list_agents()
    assert len(agents) == 1
    # mutating the snapshot shouldn't affect registry
    agents[0].device_id = "changed"
    loc = await registry.get("a1")
    assert loc.device_id == "dev1"
