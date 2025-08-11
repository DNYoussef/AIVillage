"""Simple message router for directing messages via a protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .message import Message
    from .standard_protocol import StandardCommunicationProtocol


class MessageRouter:
    """Routes messages using a :class:`StandardCommunicationProtocol`."""

    def __init__(self, protocol: StandardCommunicationProtocol) -> None:
        """Store the protocol used for routing."""
        self.protocol = protocol

    async def route(self, message: Message) -> None:
        await self.protocol.send_message(message.receiver, message)
