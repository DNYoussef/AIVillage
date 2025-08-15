"""Android-facing LibP2P mesh bindings.

This module provides a minimal bridge to the ``py-libp2p`` implementation so
that Android components can interact with a real LibP2P node via the Python
runtime.  The class exposes a small API that mirrors the Kotlin/Java layer used
on Android devices and integrates with the existing :class:`MeshMessage`
dataclass defined in :mod:`src.core.p2p.libp2p_mesh`.

The implementation is intentionally lightweight – it spins up a LibP2P host,
allows explicit peer connections and supports sending and receiving messages
using the ``MeshMessage`` format.  It is designed primarily for integration
tests and Android instrumentation where a full featured mesh network is
unnecessary.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from libp2p import new_host
from libp2p.network.stream.net_stream import INetStream
from libp2p.peer.peerinfo import info_from_p2p_addr
from multiaddr import Multiaddr

from src.core.p2p.libp2p_mesh import (  # Re-use existing message structures
    MeshConfiguration,
    MeshMessage,
    MeshMessageType,
)

logger = logging.getLogger(__name__)


@dataclass
class _Status:
    """Simple status container mirroring Android enums."""

    value: str = "inactive"


class LibP2PMeshNetwork:
    """Minimal LibP2P network wrapper for Android.

    The class exposes three coroutine methods – ``connect``, ``send_message``
    and ``listen`` – which provide enough functionality for peer discovery and
    message exchange tests on Android emulators.
    """

    def __init__(self, config: MeshConfiguration | None = None) -> None:
        self.config = config or MeshConfiguration()
        self.node_id = self.config.node_id or ""
        self.status = _Status()
        self._host = None
        self._message_queue: asyncio.Queue[MeshMessage] = asyncio.Queue()

    async def connect(self, bootstrap_peers: list[str] | None = None) -> None:
        """Start the LibP2P host and connect to optional bootstrap peers.

        Parameters
        ----------
        bootstrap_peers:
            A list of multiaddress strings with peer ids. When provided the
            host will dial each address after starting.
        """

        logger.debug("Starting LibP2P host on port %s", self.config.listen_port)
        listen_addr = Multiaddr(f"/ip4/0.0.0.0/tcp/{self.config.listen_port}")
        self._host = new_host()
        # start listening on the configured address
        await self._host.get_network().listen(listen_addr)

        self.node_id = str(self._host.get_id())
        self.status.value = "active"

        # register stream handler for direct messaging
        self._host.set_stream_handler("/aivillage/mesh/1.0.0", self._stream_handler)

        if bootstrap_peers:
            for addr in bootstrap_peers:
                try:
                    info = info_from_p2p_addr(Multiaddr(addr))
                    await self._host.connect(info)
                    logger.info("Connected to bootstrap peer %s", addr)
                except Exception as exc:  # pragma: no cover - network failures
                    logger.error("Failed to connect to %s: %s", addr, exc)

    async def _stream_handler(self, stream: INetStream) -> None:
        """Handle an incoming LibP2P stream and enqueue messages."""

        try:
            data = await stream.read()
            await stream.close()
            payload = json.loads(data.decode())
            message = MeshMessage.from_dict(payload)
            await self._message_queue.put(message)
            logger.debug("Received message %s", message.id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Stream handler error: %s", exc)

    async def send_message(self, peer_addr: str, message: MeshMessage) -> bool:
        """Send ``message`` to ``peer_addr``.

        Parameters
        ----------
        peer_addr:
            Multiaddress string including the peer id of the recipient.
        message:
            :class:`MeshMessage` instance to send.
        """

        if not self._host:
            msg = "Mesh network not started; call connect() first"
            raise RuntimeError(msg)

        info = info_from_p2p_addr(Multiaddr(peer_addr))
        stream = await self._host.new_stream(info.peer_id, ["/aivillage/mesh/1.0.0"])
        await stream.write(json.dumps(message.to_dict()).encode())
        await stream.close()
        logger.debug("Sent message %s to %s", message.id, peer_addr)
        return True

    async def listen(self) -> AsyncIterator[MeshMessage]:
        """Yield incoming :class:`MeshMessage` instances as they arrive."""

        while True:
            message = await self._message_queue.get()
            yield message

    async def close(self) -> None:
        """Shut down the LibP2P host."""

        if self._host:
            await self._host.close()
            self._host = None
        self.status.value = "inactive"


__all__ = [
    "LibP2PMeshNetwork",
    "MeshMessage",
    "MeshMessageType",
    "MeshConfiguration",
]

