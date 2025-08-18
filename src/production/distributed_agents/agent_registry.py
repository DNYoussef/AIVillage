from __future__ import annotations

import builtins
import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["AgentLocation", "DistributedAgentRegistry"]


REGISTRY_PATH = Path(".cache/agent_registry.json")


@dataclass
class AgentLocation:
    """Represents where an agent can be reached."""

    agent_id: str
    endpoint: str
    location: str = "local"

    @property
    def url(self) -> str:  # Backwards compatibility with older imports
        return self.endpoint


class DistributedAgentRegistry:
    """Simple file-backed registry for distributed agents.

    Agents can be registered as either ``local`` or ``remote``. The registry is
    persisted on disk in :mod:`.cache/agent_registry.json` so that subsequent
    runs can resolve previously registered agents.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else REGISTRY_PATH
        # mapping of {"local": {name: endpoint}, "remote": {...}}
        self._agents: dict[str, dict[str, str]] = {"local": {}, "remote": {}}
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except Exception:
            return
        for location in ("local", "remote"):
            section = data.get(location, {}) if isinstance(data, dict) else {}
            if isinstance(section, dict):
                self._agents[location].update({str(name): str(endpoint) for name, endpoint in section.items()})

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._agents))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register(self, name: str, endpoint: str, location: str = "local") -> None:
        """Register an agent endpoint under ``name``.

        Parameters
        ----------
        name:
            Identifier for the agent.
        endpoint:
            Connection information for reaching the agent.
        location:
            Either ``"local"`` or ``"remote"``. Defaults to ``"local"``.
        """

        loc = "remote" if location == "remote" else "local"
        self._agents[loc][name] = endpoint
        self._save()

    def resolve(self, name: str) -> AgentLocation | None:
        """Resolve an agent by name.

        Returns ``None`` if the agent is unknown."""

        for loc in ("local", "remote"):
            endpoint = self._agents[loc].get(name)
            if endpoint is not None:
                return AgentLocation(name, endpoint, loc)
        return None

    # Backwards compatibility with older code
    def lookup(self, name: str) -> AgentLocation | None:
        return self.resolve(name)

    def list(self, location: str | None = None) -> builtins.list[AgentLocation]:
        """List registered agents.

        Parameters
        ----------
        location:
            Optional filter of ``"local"`` or ``"remote"``. If omitted all
            agents are returned.
        """

        if location:
            loc = "remote" if location == "remote" else "local"
            return [AgentLocation(name, ep, loc) for name, ep in self._agents.get(loc, {}).items()]
        agents: list[AgentLocation] = []
        for loc in ("local", "remote"):
            agents.extend(AgentLocation(name, ep, loc) for name, ep in self._agents[loc].items())
        return agents
