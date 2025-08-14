"""Evolution system for agent self-improvement.

This module provides the core evolution system for AIVillage agents,
including nightly incremental improvements and breakthrough discoveries.
"""

# Core classes - import order matters to avoid circular imports
from .base import EvolvableAgent

# Evolution engines
from .dual_evolution_system import DualEvolutionSystem
from .evolution_scheduler import EvolutionScheduler
from .kpi_evolution_engine import KPIEvolutionEngine
from .magi_architectural_evolution import MagiArchitecturalEvolution
from .metrics import EvolutionMetricsRecorder
from .nightly_evolution_orchestrator import NightlyEvolutionOrchestrator
from .resource_constrained_evolution import ResourceConstrainedEvolution

__all__ = [
    "DualEvolutionSystem",
    "EvolutionScheduler",
    "EvolvableAgent",
    "KPIEvolutionEngine",
    "MagiArchitecturalEvolution",
    "EvolutionMetricsRecorder",
    "NightlyEvolutionOrchestrator",
    "ResourceConstrainedEvolution",
]

__version__ = "1.0.0"
