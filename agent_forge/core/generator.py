"""Utility to programmatically generate standardized agents.

The generated agents provide a minimal interface used in tests:
- unique ``agent_id`` and ``metadata``
- basic message handling via an in-memory registry
- simple state management hooks
- heartbeat metric emission
- optional sandbox mode for restricted execution
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Type

# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseGeneratedAgent:
    """Lightweight base class for generated agents."""

    # registry of live agent instances for message passing
    instances: Dict[str, "BaseGeneratedAgent"] = {}

    def __init__(self, name: str, sandboxed: bool = False) -> None:
        self.name = name
        self.role = getattr(self, "ROLE", name)
        self.agent_id = str(uuid.uuid4())
        self.sandboxed = sandboxed
        self.state: Dict[str, Any] = {"status": "initialized"}
        self.metrics: Dict[str, Any] = {"heartbeats": 0}
        self.messages: list[tuple[str, Any]] = []
        # register instance for simple message passing
        BaseGeneratedAgent.instances[name] = self

    # ------------------------------------------------------------------
    # lifecycle commands
    # ------------------------------------------------------------------
    def start(self) -> None:
        self.state["status"] = "running"
        self.emit_heartbeat()

    def stop(self) -> None:
        self.state["status"] = "stopped"

    def status(self) -> str:
        return self.state["status"]

    # ------------------------------------------------------------------
    # messaging utilities
    # ------------------------------------------------------------------
    def send_message(self, recipient: str, content: Any) -> None:
        """Send a message to another agent."""
        target = BaseGeneratedAgent.instances.get(recipient)
        if target:
            target.handle_message(self.name, content)

    def handle_message(self, sender: str, content: Any) -> None:
        """Default message handler simply stores the message."""
        self.messages.append((sender, content))

    # ------------------------------------------------------------------
    # metrics
    # ------------------------------------------------------------------
    def emit_heartbeat(self) -> None:
        self.metrics["heartbeats"] += 1
        self.state["last_heartbeat"] = time.time()


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def create_agent_class(agent_name: str, role: str) -> Type[BaseGeneratedAgent]:
    """Dynamically create a new agent class with the given role."""

    class GeneratedAgent(BaseGeneratedAgent):
        ROLE = role

        def __init__(self, sandboxed: bool = False) -> None:
            super().__init__(agent_name, sandboxed=sandboxed)

    GeneratedAgent.__name__ = f"{agent_name}Agent"
    AGENT_REGISTRY[agent_name] = GeneratedAgent
    return GeneratedAgent


# registry mapping agent names to classes
AGENT_REGISTRY: Dict[str, Type[BaseGeneratedAgent]] = {}

# ---------------------------------------------------------------------------
# Generate the ten missing agents
# ---------------------------------------------------------------------------
AGENT_SPECS = {
    "Scribe": "Documentation generation and updates",
    "Herald": "Event notifications and alerts",
    "Curator": "Content moderation and quality control",
    "Navigator": "Request routing and load balancing",
    "Alchemist": "Model mixing and ensemble coordination",
    "Guardian": "Backup and recovery operations",
    "Chronicler": "History tracking and audit logs",
    "Artificer": "Tool creation and integration",
    "Emissary": "External API gateway",
    "Steward": "Resource allocation and scheduling",
}

for _name, _role in AGENT_SPECS.items():
    globals()[f"{_name}Agent"] = create_agent_class(_name, _role)

__all__ = ["BaseGeneratedAgent", "create_agent_class", "AGENT_REGISTRY"] + [
    f"{name}Agent" for name in AGENT_SPECS
]
