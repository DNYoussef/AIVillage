"""Compatibility layer for Agent Forge.

This package provides a thin wrapper that exposes the production
``AgentForge`` implementation at ``production.agent_forge.core`` under the
``agent_forge`` namespace. Existing imports like ``from agent_forge.core
import AgentForge`` continue to work without modification.
"""

import sys
from importlib import import_module

_core_pkg = import_module("src.production.agent_forge.core")
# Re-export ``AgentForge`` at the package level for convenience
AgentForge = _core_pkg.AgentForge

# Make ``agent_forge.core`` resolve to the production implementation
sys.modules.setdefault(__name__ + ".core", _core_pkg)

__all__ = ["AgentForge"]
