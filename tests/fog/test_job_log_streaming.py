import asyncio
from importlib import util
from pathlib import Path
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp import web

# Dynamically import the target FogClient from integrations path
module_path = Path(__file__).resolve().parents[2] / "integrations/clients/fog-sdk/python/fog_client_original.py"
spec = util.spec_from_file_location("fog_client_original", module_path)
fog_module = util.module_from_spec(spec)
spec.loader.exec_module(fog_module)
FogClient = fog_module.FogClient


@pytest.mark.asyncio
async def test_stream_logs_via_websocket():
    """Verify that log lines are received over WebSocket."""
    log_lines = ["first", "second", "third"]

    async def websocket_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        for line in log_lines:
            await ws.send_str(line)
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get('/v1/fog/jobs/{job_id}/logs/stream', websocket_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    base_url = f'http://localhost:{port}'

    async with FogClient(base_url=base_url, stream_logs_enabled=True) as client:
        received = []
        async for line in client.stream_logs('job123'):
            received.append(line)

    await runner.cleanup()

    assert received == log_lines


@pytest.mark.asyncio
async def test_stream_logs_disabled_falls_back_to_http():
    """Ensure HTTP log fetch is used when streaming disabled."""
    async with FogClient(base_url='http://localhost:0', stream_logs_enabled=False) as client:
        client.get_job_logs = AsyncMock(return_value='a\nb')
        collected = []
        async for line in client.stream_logs('job123'):
            collected.append(line)

    client.get_job_logs.assert_awaited_once_with('job123')
    assert collected == ['a', 'b']
