"""Lightweight registry for tracking agent locations.

This module provides a minimal implementation to support tests
involving distributed agents. It tracks which device currently hosts
an agent instance and exposes simple async helpers for updating
locations. The real production system would likely back this registry
with a database or distributed key-value store. For the purposes of the
unit tests we keep everything in memory and focus on race-free
operations using :class:`asyncio.Lock`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import asyncio
import time
from copy import deepcopy


@dataclass
class AgentLocation:
    """Record of where an agent instance is running."""

    agent_id: str
    device_id: str
    last_seen: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class DistributedAgentRegistry:
    """In-memory registry of agent locations.

    The implementation is intentionally small – just enough for unit tests
    and local experimentation.  All operations are protected by an
    ``asyncio.Lock`` to avoid race conditions when accessed from concurrent
    coroutines.
    """

    def __init__(self) -> None:
        self._locations: Dict[str, AgentLocation] = {}
        self._lock = asyncio.Lock()

    async def register(
        self, agent_id: str, device_id: str, metadata: Optional[dict[str, Any]] = None
    ) -> AgentLocation:
        """Register a new agent location.

        Returns the :class:`AgentLocation` that was stored.
        """

        async with self._lock:
            loc = AgentLocation(agent_id=agent_id, device_id=device_id, metadata=metadata or {})
            self._locations[agent_id] = loc
            return loc

    async def get(self, agent_id: str) -> Optional[AgentLocation]:
        """Return the location for ``agent_id`` if known."""

        async with self._lock:
            loc = self._locations.get(agent_id)
            return deepcopy(loc) if loc else None

    async def update(self, agent_id: str, device_id: str) -> bool:
        """Update the device hosting ``agent_id``.

        Returns ``True`` if the agent was present, ``False`` otherwise.
        """

        async with self._lock:
            loc = self._locations.get(agent_id)
            if not loc:
                return False
            loc.device_id = device_id
            loc.last_seen = time.time()
            return True

    async def remove(self, agent_id: str) -> bool:
        """Remove ``agent_id`` from the registry."""

        async with self._lock:
            return self._locations.pop(agent_id, None) is not None

    async def list_agents(self) -> list[AgentLocation]:
        """Return a snapshot list of all registered agents."""

        async with self._lock:
            return [deepcopy(loc) for loc in self._locations.values()]
