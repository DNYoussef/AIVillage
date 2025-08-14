"""Production Agent Forge System.

Provides production-ready agent creation, management, and evolution capabilities.
"""

__version__ = "1.0.0"

# Core components
try:
    from .agent_factory import AgentFactory
    from .base import AgentRole, AgentSpecialization, BaseMetaAgent
    from .validate_all_agents import validate_all_agents

    __all__ = [
        "AgentFactory",
        "validate_all_agents",
        "AgentRole",
        "AgentSpecialization",
        "BaseMetaAgent",
    ]
except ImportError:
    # Handle missing dependencies gracefully
    __all__ = []

# Evolution system
try:
    from .evolution.base import EvolvableAgent
    from .evolution.dual_evolution_system import DualEvolutionSystem
    from .evolution.evolution_scheduler import EvolutionScheduler
    from .evolution.kpi_evolution_engine import KPIEvolutionEngine
    from .evolution.metrics import EvolutionMetricsRecorder
    from .evolution.resource_constrained_evolution import ResourceConstrainedEvolution

    __all__.extend(
        [
            "EvolutionScheduler",
            "KPIEvolutionEngine",
            "ResourceConstrainedEvolution",
            "DualEvolutionSystem",
            "EvolvableAgent",
            "EvolutionMetricsRecorder",
        ]
    )
except ImportError as e:
    # Evolution system optional
    import logging

    logging.getLogger(__name__).warning(f"Evolution system not available: {e}")
