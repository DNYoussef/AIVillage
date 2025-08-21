import pytest
from src.production.distributed_agents.distributed_agent_orchestrator import DistributedAgentOrchestrator


@pytest.mark.asyncio
async def test_agent_collaboration_protocol(orchestrator_setup, monkeypatch):
    orchestrator, p2p = orchestrator_setup
    await orchestrator.deploy_agent_constellation()

    # Avoid long running monitor task
    async def _instant_monitor(self):
        return

    monkeypatch.setattr(DistributedAgentOrchestrator, "_monitor_agent_collaboration", _instant_monitor)

    await orchestrator.enable_cross_device_collaboration()

    # Verify broadcast of collaboration configuration
    assert p2p.broadcasts, "No collaboration broadcast was sent"
    event, payload = p2p.broadcasts[-1]
    assert event == "ENABLE_AGENT_COLLABORATION"
    assert payload["type"] == "ENABLE_AGENT_COLLABORATION"
    assert payload["agent_registry"]
    assert payload["device_assignments"]
