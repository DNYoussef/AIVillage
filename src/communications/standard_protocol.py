"""Minimal communication protocol used for tests."""

from __future__ import annotations

from typing import Any, List


class StandardCommunicationProtocol:
    """Send and receive messages using an in-memory list."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._messages: List[Any] = []

    def send(self, message: Any) -> None:
        """Store ``message`` for later retrieval."""
        self._messages.append(message)

    def send_message(self, message: Any) -> None:
        """Alias for :meth:`send` for compatibility."""
        self.send(message)

    def receive(self) -> Any | None:
        """Return the oldest message if available."""
        return self._messages.pop(0) if self._messages else None

    def receive_message(self) -> Any | None:
        """Alias for :meth:`receive` for compatibility."""
        return self.receive()


__all__ = ["StandardCommunicationProtocol"]
