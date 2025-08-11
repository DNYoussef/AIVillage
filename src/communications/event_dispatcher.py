"""Minimal event dispatcher."""

from __future__ import annotations

from typing import Any, List


class EventDispatcher:
    """Record dispatched events for later inspection."""

    def __init__(self) -> None:
        self.events: List[Any] = []

    def dispatch(self, event: Any) -> None:
        """Store ``event`` in the internal list."""
        self.events.append(event)
