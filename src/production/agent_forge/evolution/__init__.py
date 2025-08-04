"""Evolution system for agent self-improvement.

This module provides the core evolution system for AIVillage agents,
including nightly incremental improvements and breakthrough discoveries.
"""

# Core classes - import order matters to avoid circular imports
from .base import EvolvableAgent

# Evolution engines
from .dual_evolution_system import DualEvolutionSystem
from .evolution_metrics import EvolutionMetrics
from .evolution_scheduler import EvolutionScheduler
from .magi_architectural_evolution import MagiArchitecturalEvolution
from .nightly_evolution_orchestrator import NightlyEvolutionOrchestrator

__all__ = [
    "DualEvolutionSystem",
    "EvolutionMetrics",
    "EvolutionScheduler",
    "EvolvableAgent",
    "MagiArchitecturalEvolution",
    "NightlyEvolutionOrchestrator",
]

__version__ = "1.0.0"
