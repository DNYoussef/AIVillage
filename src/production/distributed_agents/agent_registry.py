"""Lightweight agent registry stubs for testing."""

from dataclasses import dataclass, field
from typing import Any

# Known agents in the AIVillage ecosystem. The registry recognises
# all 18 specialised agents including those generated at runtime.
ALL_AGENT_TYPES = [
    "King",
    "Sage",
    "Magi",
    "Sword",
    "Shield",
    "Logger",
    "Profiler",
    "Builder",
    "Scribe",
    "Herald",
    "Curator",
    "Navigator",
    "Alchemist",
    "Guardian",
    "Chronicler",
    "Artificer",
    "Emissary",
    "Steward",
]


@dataclass
class AgentLocation:
    """Represents where agents are running."""

    agent_type: Any
    device_id: str
    instance_ids: list[str] = field(default_factory=list)


class DistributedAgentRegistry:
    """Minimal registry tracking agent locations."""

    def __init__(self) -> None:
        # Pre-populate registry with all known agent types
        self.registry: dict[Any, AgentLocation] = {
            name: AgentLocation(agent_type=name, device_id="") for name in ALL_AGENT_TYPES
        }
