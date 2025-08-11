"""Lightweight WebSocket handler used for communications tests."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import websockets

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Awaitable, Callable

    from websockets import WebSocketServerProtocol

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """Minimal WebSocket server wrapper."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Store configuration for the WebSocket server."""
        self.host = config.get("host", "127.0.0.1")
        self.port = config.get("port", 8765)
        self._server: websockets.WebSocketServer | None = None
        self._handler: (
            Callable[[WebSocketServerProtocol], Awaitable[None]] | None
        ) = None

    async def start(
        self,
        handler: Callable[[WebSocketServerProtocol], Awaitable[None]] | None = None
    ) -> None:
        self._handler = handler or (lambda ws: ws.wait_closed())
        self._server = await websockets.serve(self._handler, self.host, self.port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
