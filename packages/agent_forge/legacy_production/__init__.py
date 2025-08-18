"""Production Agent Forge System.

Provides production-ready agent creation, management, and evolution capabilities.
"""

__version__ = "1.0.0"

# Core components
try:
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
