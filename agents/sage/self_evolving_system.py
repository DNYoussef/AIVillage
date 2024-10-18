import random
from typing import Dict, Any, List

class SelfEvolvingSystem:
    def __init__(self, agent):
        self.agent = agent
        self.evolution_rate = 0.1
        self.mutation_rate = 0.01
        self.learning_rate = 0.001
        self.performance_history = []

    async def evolve(self):
        if random.random() < self.evolution_rate:
            await self._mutate()
        await self._adapt()

    async def _mutate(self):
        if random.random() < self.mutation_rate:
            # Mutate a random research capability
            if self.agent.research_capabilities:
                capability = random.choice(self.agent.research_capabilities)
                new_capability = await self.agent.generate(
                    f"Suggest an improvement or variation of the research capability: {capability}"
                )
                self.agent.research_capabilities.append(new_capability)

    async def _adapt(self):
        if len(self.performance_history) > 10:
            avg_performance = sum(self.performance_history[-10:]) / 10
            if avg_performance > 0.8:
                self.evolution_rate *= 0.9
                self.mutation_rate *= 0.9
            else:
                self.evolution_rate *= 1.1
                self.mutation_rate *= 1.1

    async def update_hyperparameters(self, new_evolution_rate: float, new_mutation_rate: float, new_learning_rate: float):
        self.evolution_rate = new_evolution_rate
        self.mutation_rate = new_mutation_rate
        self.learning_rate = new_learning_rate

    async def process_task(self, task) -> Dict[str, Any]:
        result = await self.agent.execute_task(task)
        performance = result.get('performance', 0.5)  # Default performance metric
        self.performance_history.append(performance)
        return result
