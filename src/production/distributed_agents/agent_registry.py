"""Distributed agent registry with local/remote persistence.

This module tracks locations of distributed agents and persists registry
information to ``.cache/agent_registry.json``.  The registry distinguishes
between agents available locally and those reachable remotely.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional


REGISTRY_FILE = Path(".cache/agent_registry.json")


@dataclass
class AgentLocation:
    """Represents a registered agent."""

    name: str
    endpoint: str
    local: bool = True


class DistributedAgentRegistry:
    """Registry for distributed agents."""

    def __init__(self, cache_path: Path | str = REGISTRY_FILE) -> None:
        self.cache_path = Path(cache_path)
        self.local_agents: Dict[str, str] = {}
        self.remote_agents: Dict[str, str] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load registry data from cache."""
        if not self.cache_path.exists():
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            return

        try:
            data = json.loads(self.cache_path.read_text())
            self.local_agents = data.get("local", {})
            self.remote_agents = data.get("remote", {})
        except Exception:
            # Corrupt cache – start with empty registry
            self.local_agents = {}
            self.remote_agents = {}

    def _save(self) -> None:
        """Persist registry data to cache."""
        data = {"local": self.local_agents, "remote": self.remote_agents}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Registry operations
    # ------------------------------------------------------------------
    def register(self, name: str, endpoint: str, *, local: bool = True) -> AgentLocation:
        """Register a new agent.

        Parameters
        ----------
        name:
            Agent identifier.
        endpoint:
            Network endpoint for the agent.
        local:
            If ``True`` store in the local registry, otherwise remote.
        """

        registry = self.local_agents if local else self.remote_agents
        registry[name] = endpoint
        self._save()
        return AgentLocation(name=name, endpoint=endpoint, local=local)

    def resolve(self, name: str) -> Optional[AgentLocation]:
        """Resolve an agent by name."""
        if name in self.local_agents:
            return AgentLocation(name, self.local_agents[name], True)
        if name in self.remote_agents:
            return AgentLocation(name, self.remote_agents[name], False)
        return None

    def list(self) -> List[AgentLocation]:
        """List all registered agents."""
        agents: List[AgentLocation] = []
        for n, e in self.local_agents.items():
            agents.append(AgentLocation(n, e, True))
        for n, e in self.remote_agents.items():
            agents.append(AgentLocation(n, e, False))
        return agents
