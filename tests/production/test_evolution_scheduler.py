from uuid import uuid4

import pytest
from src.production.agent_forge.evolution import DualEvolutionSystem, EvolutionScheduler, EvolvableAgent


class DummyAgent(EvolvableAgent):
    def __init__(self, performance: float):
        super().__init__({"agent_id": str(uuid4())})
        self._performance = performance

    def evaluate_kpi(self):
        return {"performance": self._performance, "reliability": 0.5}

    def should_retire(self) -> bool:
        return self._performance < self.retirement_threshold

    def needs_evolution(self) -> bool:
        return self.retirement_threshold <= self._performance < self.evolution_threshold


def test_scheduler_basic_actions():
    scheduler = EvolutionScheduler()
    agent_retire = DummyAgent(0.3)
    agent_evolve = DummyAgent(0.5)
    agent_none = DummyAgent(0.8)

    assert scheduler.get_action(agent_retire) == "retire"
    assert scheduler.get_action(agent_evolve) == "evolve"
    assert scheduler.get_action(agent_none) == "none"


@pytest.mark.asyncio
async def test_dual_evolution_system_triggers_scheduler_actions():
    system = DualEvolutionSystem()
    agent_retire = DummyAgent(0.2)
    agent_evolve = DummyAgent(0.5)
    system.register_agent(agent_retire, {})
    system.register_agent(agent_evolve, {})

    evolved = {}

    async def fake_evolve(agent):
        evolved["id"] = agent.agent_id
        return True

    system._evolve_agent_nightly = fake_evolve

    await system._monitor_agent_performance()

    assert agent_retire.agent_id not in system.registered_agents
    assert evolved["id"] == agent_evolve.agent_id
