"""Minimal communication protocol used for tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import logging


class StandardCommunicationProtocol:
    """Send and receive messages using an in-memory message routing system."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._messages: list[Any] = []
        self._subscribers: dict[str, Callable] = {}
        self._message_queue: dict[str, list[Any]] = {}

    def send(self, message: Any) -> None:
        """Store ``message`` for later retrieval."""
        self._messages.append(message)

    async def send_message(self, message: Any) -> None:
        """Send message and trigger appropriate handlers."""
        self._messages.append(message)

        # Route message to subscribers
        receiver = getattr(message, "receiver", None)
        if receiver and receiver in self._subscribers:
            handler = self._subscribers[receiver]
            try:
                await handler(message)
            except Exception as exc:  # pragma: no cover - test helper
                logging.getLogger(__name__).exception(
                    "Handler error for %s: %s", receiver, exc
                )

        # Also queue for manual retrieval
        if receiver:
            if receiver not in self._message_queue:
                self._message_queue[receiver] = []
            self._message_queue[receiver].append(message)

    def receive(self) -> Any | None:
        """Return the oldest message if available."""
        return self._messages.pop(0) if self._messages else None

    def receive_message(self) -> Any | None:
        """Alias for :meth:`receive` for compatibility."""
        return self.receive()

    def subscribe(self, agent_id: str, handler: Callable) -> None:
        """Subscribe to messages for a specific agent."""
        self._subscribers[agent_id] = handler

    def unsubscribe(self, agent_id: str) -> None:
        """Unsubscribe from messages for an agent."""
        if agent_id in self._subscribers:
            del self._subscribers[agent_id]

    def get_messages_for(self, agent_id: str) -> list[Any]:
        """Get all queued messages for an agent."""
        return self._message_queue.get(agent_id, [])

    def clear_messages(self) -> None:
        """Clear all messages."""
        self._messages.clear()
        self._message_queue.clear()


__all__ = ["StandardCommunicationProtocol"]
