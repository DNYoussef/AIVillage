"""Tiny event dispatcher used by communication components."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable  # noqa: UP035


class EventDispatcher:
    """Minimal publish/subscribe implementation."""

    def __init__(self) -> None:
        """Create a dispatcher with no registered handlers."""
        self._handlers: dict[str, list[Callable[..., Any]]] = defaultdict(list)

    def register(self, event: str, handler: Callable[..., Any]) -> None:
        self._handlers[event].append(handler)

    def dispatch(self, event: str, *args: Any, **kwargs: Any) -> None:
        for handler in self._handlers.get(event, []):
            handler(*args, **kwargs)
