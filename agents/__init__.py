# ruff: noqa: N999
"""Expose experimental agent implementations via the core package."""
from pathlib import Path
from pkgutil import extend_path

# Extend this package path to include experimental agent implementations
__path__ = extend_path(__path__, __name__)
experimental_agents_dir = (
    Path(__file__).resolve().parent.parent / "experimental" / "agents" / "agents"
)
if experimental_agents_dir.exists():
    __path__.append(str(experimental_agents_dir))
