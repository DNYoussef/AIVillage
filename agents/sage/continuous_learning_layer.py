from collections import deque
from typing import Any

from rag_system.retrieval.vector_store import VectorStore


class ContinuousLearningLayer:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.learning_rate = 0.01
        self.recent_learnings = deque(maxlen=100)
        self.performance_history = deque(maxlen=1000)

    async def update(self, task, result):
        learned_info = self.extract_learning(task, result)
        await self.vector_store.add_texts([learned_info])
        self.recent_learnings.append(learned_info)
        self.performance_history.append(result.get("performance", 0.5))

    def extract_learning(self, task, result) -> str:
        return f"Task: {task['content']}\nResult: {result}\nLearned: {self._extract_key_insights(task, result)}"

    def _extract_key_insights(self, task, result) -> str:
        task_text = task.get("content") if isinstance(task, dict) else str(task)
        summary = str(result)
        return f"{task_text} => {summary[:100]}"

    async def evolve(self):
        if len(self.performance_history) > 100:
            recent_performance = sum(self.performance_history[-100:]) / 100
            if recent_performance > 0.8:
                self.learning_rate *= 0.95  # Slow down learning if performing well
            else:
                self.learning_rate *= 1.05  # Speed up learning if not performing well

        await self._consolidate_learnings()

    async def _consolidate_learnings(self):
        if len(self.recent_learnings) > 50:
            consolidated = self._synthesize_learnings(list(self.recent_learnings))
            await self.vector_store.add_texts([consolidated])
            self.recent_learnings.clear()

    def _synthesize_learnings(self, learnings: list[str]) -> str:
        return "\n".join(learnings)

    async def retrieve_relevant_learnings(self, task: dict[str, Any]) -> list[str]:
        return list(self.recent_learnings)
