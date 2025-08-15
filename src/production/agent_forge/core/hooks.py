"""Hook interfaces for Agent Forge lifecycle events.

The functions defined here act as extension points.  Production deployments
may monkeypatch these hooks with custom behaviour.  The default
implementations are intentionally lightweight and side-effect free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Patch:
    """Represents a mutation suggested during an evolution step."""

    description: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Artifact:
    """Artifact produced by a compression step."""

    engine: str
    data: dict[str, Any] = field(default_factory=dict)


def on_agent_created(agent: Any) -> None:
    """Called whenever a new agent is created."""
    # Default implementation is a no-op.
    return None


def evolution_step(agent: Any, kpis: dict[str, Any]) -> Patch | None:
    """Called after KPIs are evaluated for an agent.

    Returning a :class:`Patch` signals that the agent should be modified.
    """
    return None


def apply_compression(agent: Any, engine_name: str) -> Artifact:
    """Called to apply compression to an agent using the given engine."""
    return Artifact(engine=engine_name)
