from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AgentLocation:
    agent_id: str
    url: str


class DistributedAgentRegistry:
    """In-memory registry for distributed agents."""

    def __init__(self) -> None:
        self._agents: Dict[str, AgentLocation] = {}

    def register(self, agent_id: str, url: str) -> None:
        self._agents[agent_id] = AgentLocation(agent_id, url)

    def lookup(self, agent_id: str) -> Optional[AgentLocation]:
        return self._agents.get(agent_id)
