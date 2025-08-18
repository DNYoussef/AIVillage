"""
ADAS (Automatic Discovery of Agentic Space) - Expert configuration search system.
Includes both new ADAS×Transformer² implementation and legacy components.
"""

# New ADAS×Transformer² implementation
from .archive import ADASArchive, DispatchSpec, ExperimentResult, ExpertSpec
from .proposer import ADASProposer
from .runner import ADASRunner

# Legacy components (may have dependency issues)
try:
    from .adas import ADASTask
    from .system import ADASystem, adaptive_search

    legacy_available = True
except ImportError:
    # Legacy components have dependency issues
    ADASTask = None
    ADASystem = None
    adaptive_available = False
    legacy_available = False

__all__ = [
    # New implementation
    "ADASArchive",
    "ExperimentResult",
    "ExpertSpec",
    "DispatchSpec",
    "ADASProposer",
    "ADASRunner",
    # Legacy (if available)
    "ADASTask",
    "ADASystem",
    "adaptive_search",
]
