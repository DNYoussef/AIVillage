"""Core utilities for generating standardized agents.

This package now re-exports the :class:`AgentForge` facade located in the
production path (``src/production/agent_forge``).  The facade wraps existing
evolution and compression engines and provides a minimal public API used by the
tests in this repository.
"""

from .generator import AGENT_REGISTRY, BaseGeneratedAgent, create_agent_class

try:  # pragma: no cover - import happens for convenience
    from src.production.agent_forge.core.forge import AgentForge  # type: ignore
except Exception:  # pragma: no cover - keep import failure silent
    AgentForge = None  # type: ignore

__all__ = [
    "AGENT_REGISTRY",
    "BaseGeneratedAgent",
    "create_agent_class",
    "AgentForge",
]
