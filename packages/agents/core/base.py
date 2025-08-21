"""Shared base class for specialized agents.

This module centralizes common functionality previously duplicated across
specialized agent implementations. Domain specific agents should subclass
``BaseAgent`` and extend its behavior only where necessary.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from packages.agents.core.agent_interface import AgentInterface

logger = logging.getLogger(__name__)


class BaseAgent(AgentInterface):
    """Minimal agent base class with shared utilities."""

    def __init__(self, agent_id: str, agent_type: str, capabilities: list[str]) -> None:
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.initialized = False

    async def get_embedding(self, text: str) -> list[float]:
        """Generate a simple hash based embedding for *text*."""
        hash_value = int(hashlib.md5(text.encode(), usedforsecurity=False).hexdigest(), 16)  # nosec B324
        return [(hash_value % 1000) / 1000.0] * 384

    async def communicate(self, message: str, recipient: AgentInterface) -> str:
        """Default inter‑agent communication helper."""
        response = await recipient.generate(f"{self.agent_type} Agent says: {message}")
        return f"Received response: {response}"

    async def introspect(self) -> dict[str, Any]:
        """Basic agent status information."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "initialized": self.initialized,
        }

    async def activate_latent_space(self, query: str) -> tuple[str, str]:
        """Return a generic latent space representation for *query*."""
        return "general", f"LATENT[general:{query[:50]}]"
