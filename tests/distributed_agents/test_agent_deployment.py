import pytest


@pytest.mark.asyncio
async def test_cross_device_deployment(orchestrator_setup):
    orchestrator, _ = orchestrator_setup
    plan = await orchestrator.deploy_agent_constellation()
    assert plan is not None

    # Ensure agents were distributed across multiple devices
    used_devices = [d for d, ids in plan.device_assignments.items() if ids]
    assert len(used_devices) >= 2

    status = orchestrator.get_deployment_status()
    assert status["deployed"] is True
    assert status["devices_used"] >= 2
