"""AIVillage Comprehensive Debug System

Provides debugging, monitoring, and troubleshooting tools for the entire AIVillage system.
Follows CODEX Integration Requirements for debug mode implementation.
"""

from .dashboard import DebugDashboard
from .debug_manager import DebugManager, get_debug_manager
from .logger_config import get_debug_logger, setup_debug_logging
from .profiler import PerformanceProfiler
from .troubleshooting import TroubleshootingTools

__all__ = [
    "DebugDashboard",
    "DebugManager",
    "PerformanceProfiler",
    "TroubleshootingTools",
    "get_debug_logger",
    "get_debug_manager",
    "setup_debug_logging"
]

__version__ = "1.0.0"
