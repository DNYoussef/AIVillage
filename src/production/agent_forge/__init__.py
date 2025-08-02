"""Production Agent Forge System.

Provides production-ready agent creation, management, and evolution capabilities.
"""

__version__ = "1.0.0"

# Core components
try:
    from .agent_factory import AgentFactory
    from .validate_all_agents import validate_all_agents
    
    __all__ = [
        'AgentFactory',
        'validate_all_agents',
    ]
except ImportError:
    # Handle missing dependencies gracefully
    __all__ = []

# Evolution system
try:
    from .evolution import (
        evolution_scheduler,
        resource_constrained_evolution,
        kpi_evolution_engine
    )
    __all__.extend([
        'evolution_scheduler',
        'resource_constrained_evolution',
        'kpi_evolution_engine'
    ])
except ImportError:
    # Evolution system optional
    pass

