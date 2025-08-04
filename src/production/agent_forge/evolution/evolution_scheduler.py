from dataclasses import dataclass

from .base import EvolvableAgent


@dataclass
class SchedulerConfig:
    """Configuration for :class:`EvolutionScheduler`."""

    retirement_threshold: float = 0.4
    evolution_threshold: float = 0.6


class EvolutionScheduler:
    """Decides when an agent should evolve or retire based on KPIs."""

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()

    def get_action(self, agent: EvolvableAgent) -> str:
        """Return the recommended action for the agent.

        Possible values are ``"retire"``, ``"evolve"``, or ``"none"``.
        """
        kpis = agent.evaluate_kpi()
        performance = kpis.get("performance", 0.5)

        if performance < self.config.retirement_threshold or agent.should_retire():
            return "retire"

        if performance < self.config.evolution_threshold or agent.needs_evolution():
            return "evolve"

        return "none"
