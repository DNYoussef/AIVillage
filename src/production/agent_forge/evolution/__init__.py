"""Evolution system for agent self-improvement

This module provides the core evolution system for AIVillage agents,
including nightly incremental improvements and breakthrough discoveries.
"""

# Core classes - import order matters to avoid circular imports
from .base import EvolvableAgent
from .evolution_metrics import EvolutionMetrics
# from .evolution_scheduler import EvolutionScheduler  # TODO: Implement

# Evolution engines
from .dual_evolution_system import DualEvolutionSystem
from .nightly_evolution_orchestrator import NightlyEvolutionOrchestrator
from .magi_architectural_evolution import MagiArchitecturalEvolution

__all__ = [
    'EvolvableAgent',
    'EvolutionMetrics',
    # 'EvolutionScheduler',  # TODO: Implement
    'DualEvolutionSystem', 
    'NightlyEvolutionOrchestrator',
    'MagiArchitecturalEvolution'
]

__version__ = "1.0.0"
