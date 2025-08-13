"""Expose experimental agent implementations via the core package."""

from pathlib import Path
from pkgutil import extend_path

# Extend this package path to include experimental agent implementations
__path__ = extend_path(__path__, __name__)
experimental_agents_dir = Path(__file__).resolve().parent.parent / "experimental" / "agents" / "agents"
if experimental_agents_dir.is_dir():
    experimental_path = str(experimental_agents_dir)
    if experimental_path not in __path__:
        __path__.insert(0, experimental_path)
