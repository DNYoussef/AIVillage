"""Agent Evolution System - Self-Evolving 18-Agent Ecosystem.

The core differentiator of the Atlantis vision - a fully autonomous, self-improving
agent ecosystem that evolves through genetic algorithms, meta-learning, and safe
code modification.

Key Components:
- AgentEvolutionEngine: Core genetic algorithm implementation
- EvolutionOrchestrator: Main coordination system
- SafeCodeModifier: Autonomous code improvement
- MetaLearningEngine: Learning optimization strategies
- EvolutionDashboard: Real-time monitoring and visualization

This system implements:
1. Performance-based selection of the 18-agent ecosystem
2. Genetic crossover and mutation of agent configurations
3. Safe autonomous code modification capabilities
4. Meta-learning for strategy optimization
5. Real-time monitoring and emergency rollback
6. Comprehensive evolution analytics and reporting
"""

import logging

logger = logging.getLogger(__name__)

# Enable evolution system imports
try:
    from .agent_evolution_engine import (
        AgentEvolutionEngine,
        AgentGenome,
        AgentKPIs,
        GeneticOptimizer,
    )
    from .evolution_dashboard import (
        EvolutionDashboard,
        PerformanceAnalyzer,
    )
    from .evolution_orchestrator import (
        EvolutionOrchestrator,
        OrchestrationConfig,
        OrchestrationState,
    )
    from .meta_learning_engine import (
        MetaLearningEngine,
    )
    from .safe_code_modifier import (
        CodeModification,
        SafeCodeModifier,
    )

    # Set availability flag
    EVOLUTION_AVAILABLE = True

except ImportError as e:
    logger.warning(f"Some evolution modules could not be imported: {e}")
    # Create dummy classes for missing components
    EVOLUTION_AVAILABLE = False

logger = logging.getLogger(__name__)

__version__ = "1.0.0"

if EVOLUTION_AVAILABLE:
    __all__ = [
        # Core Evolution Engine
        "AgentEvolutionEngine",
        "AgentKPIs",
        "AgentGenome",
        "GeneticOptimizer",
        # Orchestration System
        "EvolutionOrchestrator",
        "OrchestrationConfig",
        "OrchestrationState",
        # Safe Code Modification
        "SafeCodeModifier",
        "CodeModification",
        # Meta-Learning
        "MetaLearningEngine",
        # Dashboard and Analytics
        "EvolutionDashboard",
        "PerformanceAnalyzer",
        # Initialization function
        "initialize_evolution_system",
    ]
else:
    __all__ = []


# Evolution system functions
if EVOLUTION_AVAILABLE:

    async def initialize_evolution_system(
        evolution_data_path: str = "evolution_data",
        population_size: int = 18,
        auto_evolution: bool = True,
        safety_mode: bool = True,
        dashboard_port: int = 5000,
    ) -> EvolutionOrchestrator:
        """Quick initialization of the complete evolution system

        Args:
            evolution_data_path: Path for storing evolution data
            population_size: Size of agent population (default: 18)
            auto_evolution: Enable automatic evolution cycles
            safety_mode: Enable safety restrictions for code modifications
            dashboard_port: Port for evolution dashboard

        Returns:
            Configured EvolutionOrchestrator ready to start
        """
        # Create configuration
        config = OrchestrationConfig(
            evolution_interval_hours=24,
            monitoring_interval_minutes=15,
            auto_evolution_enabled=auto_evolution,
            safety_mode=safety_mode,
            max_population_size=population_size,
        )

        # Initialize orchestrator
        orchestrator = EvolutionOrchestrator(
            config=config, storage_path=evolution_data_path
        )

        logger.info(f"Evolution system initialized with {population_size} agents")
        return orchestrator
else:

    async def initialize_evolution_system(*args, **kwargs):
        """Dummy function when evolution system not available."""
        logger.error("Evolution system not available - missing dependencies")
        raise ImportError("Evolution system modules not available")


# Example usage
if __name__ == "__main__":
    import asyncio
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def main():
        if EVOLUTION_AVAILABLE:
            print("Initializing evolution system...")
            try:
                orchestrator = await initialize_evolution_system()
                print(f"Evolution system ready: {orchestrator}")
            except Exception as e:
                print(f"Error initializing evolution system: {e}")
        else:
            print("Evolution system functionality disabled - dependencies missing")

    asyncio.run(main())
