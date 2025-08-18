"""Deprecated interface location.

This module remains for backward compatibility.  The canonical
implementation now lives in :mod:`agents.core.agent_interface`.
"""

from warnings import warn

# Re-export everything from the new unified interface
from agents.core.agent_interface import *  # noqa: F403

warn(
    "src.core.agent_interface is deprecated; use agents.core.agent_interface",
    DeprecationWarning,
    stacklevel=2,
)
