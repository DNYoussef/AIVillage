"""
Agent Forge Core Module

Core components for the Agent Forge pipeline including phase controllers
and orchestration systems.
"""

from .phase_controller import PhaseController, PhaseResult
from .unified_pipeline import UnifiedConfig, UnifiedPipeline

__version__ = "1.0.0"
__all__ = [
    "PhaseController",
    "PhaseResult",
    "UnifiedConfig",
    "UnifiedPipeline",
]
