"""Route messages via a communication protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .standard_protocol import StandardCommunicationProtocol


class MessageRouter:
    """Send messages to a destination using the provided protocol."""

    def __init__(self, protocol: StandardCommunicationProtocol) -> None:
        self.protocol = protocol

    def route(self, message: Any, destination: str) -> None:
        """Forward ``message`` tagged with ``destination``."""
        self.protocol.send({"to": destination, "message": message})
