import asyncio

import pytest

from packages.agent_forge.integration.fog_burst import FogBurstOrchestrator


@pytest.mark.asyncio
async def test_shutdown_completes_cleanly_under_load():
    orchestrator = FogBurstOrchestrator(fog_gateway_url="http://test-gateway")

    dummy_networks = []

    async def mock_discover(self):
        class DummyNetwork:
            def __init__(self):
                self.closed = False

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                self.closed = True

        network = DummyNetwork()
        dummy_networks.append(network)
        async with network:
            await asyncio.sleep(0.2)

    async def mock_monitor(self):
        while True:
            await self._discover_fog_nodes()
            await asyncio.sleep(0.05)

    orchestrator._discover_fog_nodes = mock_discover.__get__(orchestrator, FogBurstOrchestrator)
    orchestrator._monitor_fog_nodes = mock_monitor.__get__(orchestrator, FogBurstOrchestrator)

    await orchestrator.initialize()

    load_tasks = [asyncio.create_task(orchestrator._discover_fog_nodes()) for _ in range(20)]
    await asyncio.sleep(0.05)

    await orchestrator.shutdown()

    await asyncio.gather(*load_tasks, return_exceptions=True)

    assert orchestrator._node_discovery_task is None or orchestrator._node_discovery_task.done()
    assert all(network.closed for network in dummy_networks)
