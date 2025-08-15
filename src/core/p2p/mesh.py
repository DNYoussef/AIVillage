"""Minimal mesh networking interface and local shim.

Provides a small, in-process pub/sub implementation that mirrors the
expected interface of the future LibP2P mesh.  When the optional betanet
process is available (signaled by ``BETANET_ENABLED=1`` and an open local
port), an RPC based mesh client is returned instead.  This keeps local
agents functional while allowing drop-in replacement with the real mesh
later.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
from typing import Awaitable, Callable, Dict, List, Protocol


Handler = Callable[[bytes], Awaitable[None]]


class MeshNetwork(Protocol):
    """Basic mesh network interface."""

    async def join(self, topic: str) -> None:
        """Join a topic so it can be published or subscribed to."""

    async def publish(self, topic: str, payload: bytes) -> None:
        """Publish a message to a topic."""

    async def subscribe(self, topic: str, handler: Handler) -> None:
        """Subscribe to a topic with an async message handler."""

    def peers(self) -> List[str]:
        """Return known peer identifiers."""


class LocalMeshNetwork(MeshNetwork):
    """In-process pub/sub mesh implementation.

    Maintains per-topic queues for subscribers.  Message history is stored
    so late subscribers receive previously published messages.  ``publish``
    will apply backpressure by awaiting on subscriber queues when they are
    full.
    """

    def __init__(self, queue_size: int = 10) -> None:
        self._queues: Dict[str, List[asyncio.Queue[bytes]]] = {}
        self._history: Dict[str, List[bytes]] = {}
        self._queue_size = queue_size

    async def join(self, topic: str) -> None:  # pragma: no cover - trivial
        self._queues.setdefault(topic, [])
        self._history.setdefault(topic, [])

    async def publish(self, topic: str, payload: bytes) -> None:
        queues = self._queues.get(topic, [])
        for q in queues:
            await q.put(payload)  # apply backpressure
        self._history.setdefault(topic, []).append(payload)

        # Wait for subscribers to process the message before returning
        for q in queues:
            await q.join()

    async def subscribe(self, topic: str, handler: Handler) -> None:
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=self._queue_size)
        self._queues.setdefault(topic, []).append(queue)

        # Replay history for offline messages
        for msg in self._history.get(topic, []):
            await handler(msg)

        async def _consumer() -> None:
            while True:
                msg = await queue.get()
                try:
                    await handler(msg)
                finally:
                    queue.task_done()

        asyncio.create_task(_consumer())

    def peers(self) -> List[str]:  # pragma: no cover - constant
        return ["local"]


class RPCMeshNetwork(MeshNetwork):
    """Very small JSON-line RPC client for a betanet process.

    This is intentionally minimal; it simply encodes operations as JSON
    objects and sends them over a TCP socket.
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._lock = asyncio.Lock()

    async def join(self, topic: str) -> None:
        await self._send({"cmd": "join", "topic": topic})

    async def publish(self, topic: str, payload: bytes) -> None:
        await self._send({"cmd": "publish", "topic": topic, "data": payload.decode("latin1")})

    async def subscribe(self, topic: str, handler: Handler) -> None:  # pragma: no cover - stub
        await self.join(topic)
        # Incoming messages handled by external process

    def peers(self) -> List[str]:  # pragma: no cover - placeholder
        return []

    async def _send(self, msg: dict) -> None:
        data = json.dumps(msg).encode() + b"\n"
        async with self._lock:
            self._sock.sendall(data)


def get_mesh_network() -> MeshNetwork:
    """Return an appropriate mesh network implementation.

    If ``BETANET_ENABLED=1`` and the betanet port is open, an RPC client is
    returned.  Otherwise a local in-process mesh is used.
    """

    if os.getenv("BETANET_ENABLED") == "1":
        port = int(os.getenv("BETANET_PORT", "8777"))
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=0.5)
            sock.setblocking(False)
            return RPCMeshNetwork(sock)
        except OSError:
            pass
    return LocalMeshNetwork()
