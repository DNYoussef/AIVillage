"""Lightweight agent registry stubs for testing."""
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentLocation:
    """Represents where agents are running."""
    agent_type: Any
    device_id: str
    instance_ids: List[str] = field(default_factory=list)


class DistributedAgentRegistry:
    """Minimal registry tracking agent locations."""

    def __init__(self):
        self.registry: Dict[Any, AgentLocation] = {}
