"""Handle messages using a communication protocol."""

from __future__ import annotations

from typing import Any


class ProtocolHandler:
    """Pass messages to a protocol instance."""

    def handle(self, protocol: Any, message: Any) -> None:
        """Use ``protocol`` to send ``message``."""
        protocol.send(message)
