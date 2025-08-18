"""AgentForge core facade re-export.

This package exists under ``src/agent_forge`` and simply re-exports the
``AgentForge`` class from the production implementation so that
``from agent_forge.core import AgentForge`` works consistently across test
environments that place ``src`` ahead of the repository root on ``sys.path``.
"""

try:  # pragma: no cover - import convenience
    from packages.agent_forge.legacy_production.core.forge import AgentForge  # type: ignore
except Exception:  # pragma: no cover - keep import failure silent
    AgentForge = None  # type: ignore

__all__ = ["AgentForge"]
