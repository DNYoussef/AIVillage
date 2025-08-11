"""Generic protocol handler utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .message import Message
    from .standard_protocol import StandardCommunicationProtocol


class ProtocolHandler:
    """Binds a protocol to a simple message handler."""

    def __init__(self, protocol: StandardCommunicationProtocol) -> None:
        """Store the protocol instance."""
        self.protocol = protocol

    async def handle(self, message: Message) -> None:
        await self.protocol.send_message(message.receiver, message)
