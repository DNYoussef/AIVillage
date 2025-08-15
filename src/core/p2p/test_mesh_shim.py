import asyncio
import time

import pytest

from .mesh import LocalMeshNetwork


@pytest.mark.asyncio
async def test_local_mesh_broadcast_backpressure_replay() -> None:
    mesh = LocalMeshNetwork(queue_size=1)
    await mesh.join("demo")
    await mesh.publish("demo", b"early")

    messages_a: list[bytes] = []
    messages_b: list[bytes] = []

    async def handler_a(data: bytes) -> None:
        messages_a.append(data)

    async def handler_b(data: bytes) -> None:
        await asyncio.sleep(0.05)
        messages_b.append(data)

    await mesh.subscribe("demo", handler_a)
    await mesh.subscribe("demo", handler_b)

    start = time.monotonic()
    await mesh.publish("demo", b"one")
    await mesh.publish("demo", b"two")
    elapsed = time.monotonic() - start

    assert elapsed >= 0.05
    assert messages_a == [b"early", b"one", b"two"]
    assert messages_b == [b"early", b"one", b"two"]
