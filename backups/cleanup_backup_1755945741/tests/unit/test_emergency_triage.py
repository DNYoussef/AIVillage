"""Emergency triage tests for critical components."""

import asyncio
import time

import pytest

from packages.core.resources.resource_monitor import ResourceMonitor
from packages.p2p.core.device_mesh import DeviceMesh
from src.communications.protocol import CommunicationsProtocol
from src.twin_runtime.guard import risk_gate


def test_security_gate_blocks_attacks() -> None:
    """Security gate should block or flag dangerous inputs."""
    assert risk_gate({"content": "rm -rf /"}) == "deny"
    assert risk_gate({"content": "DROP TABLE users"}) == "deny"
    assert risk_gate({"content": "<script>alert(1)</script>"}) == "deny"
    # Sensitive data should at least require confirmation
    assert risk_gate({"content": "4111-1111-1111-1111"}) in {"ask", "deny"}
    assert risk_gate({"content": "hello world"}) == "allow"


def test_resource_monitor_reports_metrics() -> None:
    """Resource monitor should return real metrics."""
    monitor = ResourceMonitor()
    m1 = monitor.get_all_metrics()
    time.sleep(1)
    m2 = monitor.get_all_metrics()

    assert m1["cpu_percent"] >= 0
    assert m1["memory"]["total_gb"] > 0
    assert m2["timestamp"] > m1["timestamp"]
    # can_run_model should return a boolean
    assert isinstance(monitor.can_run_model(10), bool)


def test_device_mesh_discovers_peer() -> None:
    """Device mesh should discover another running peer."""
    mesh1 = DeviceMesh(port=9050)
    mesh2 = DeviceMesh(port=9050)

    mesh2.start_discovery_service()
    time.sleep(0.5)
    peers = mesh1.discover_network_peers()
    mesh2.stop()

    hostnames = {p["hostname"] for p in peers}
    assert mesh2.local_info["hostname"] in hostnames


@pytest.mark.asyncio
async def test_websocket_communication() -> None:
    """WebSocket protocol should send and receive messages."""
    proto1 = CommunicationsProtocol("agent1", port=9101)
    proto2 = CommunicationsProtocol("agent2", port=9102)

    await proto1.start_server()
    await proto2.start_server()

    await proto1.connect("ws://localhost:9102", "agent2")
    await proto2.connect("ws://localhost:9101", "agent1")

    await proto1.send_message("agent2", {"type": "test", "content": "Hello"})
    await asyncio.sleep(0.5)

    history = proto2.get_message_history("agent1")

    await proto1.stop_server()
    await proto2.stop_server()

    assert any(m["message"]["content"] == "Hello" for m in history)
