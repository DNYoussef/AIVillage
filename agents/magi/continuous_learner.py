import numpy as np
from typing import Dict, Any, List
from .analytics.base_analytics import BaseAnalytics
from .quality_assurance_layer import QualityAssuranceLayer
from agents.utils.task import Task as LangroidTask
import logging

logger = logging.getLogger(__name__)

class ContinuousLearner(BaseAnalytics):
    def __init__(self, quality_assurance_layer: QualityAssuranceLayer, learning_rate: float = 0.01):
        super().__init__()
        self.quality_assurance_layer = quality_assurance_layer
        self.learning_rate = learning_rate

    async def update_embeddings(self, task: LangroidTask, result: Dict[str, Any]):
        task_embedding = self.quality_assurance_layer.eudaimonia_triangulator.get_embedding(task.content)
        performance = result.get('performance', 0.5)
        
        # Update empathy vector
        empathy_gradient = self.calculate_gradient(task_embedding, self.quality_assurance_layer.eudaimonia_triangulator.empathy_vector, performance)
        self.quality_assurance_layer.eudaimonia_triangulator.empathy_vector += self.learning_rate * empathy_gradient

        # Update harmony vector
        harmony_gradient = self.calculate_gradient(task_embedding, self.quality_assurance_layer.eudaimonia_triangulator.harmony_vector, performance)
        self.quality_assurance_layer.eudaimonia_triangulator.harmony_vector += self.learning_rate * harmony_gradient

        # Update self-awareness vector
        self_awareness_gradient = self.calculate_gradient(task_embedding, self.quality_assurance_layer.eudaimonia_triangulator.self_awareness_vector, performance)
        self.quality_assurance_layer.eudaimonia_triangulator.self_awareness_vector += self.learning_rate * self_awareness_gradient

        # Update rule embeddings
        for i, rule_embedding in enumerate(self.quality_assurance_layer.rule_embeddings):
            rule_gradient = self.calculate_gradient(task_embedding, rule_embedding, performance)
            self.quality_assurance_layer.rule_embeddings[i] += self.learning_rate * rule_gradient

        logger.info(f"Updated embeddings based on task: {task.content[:50]}...")

    def calculate_gradient(self, task_embedding: np.ndarray, target_embedding: np.ndarray, performance: float) -> np.ndarray:
        # Simple gradient calculation: move target embedding towards task embedding if performance is good, away if bad
        direction = task_embedding - target_embedding
        return direction * (performance - 0.5)  # Center performance around 0

    async def learn_from_feedback(self, feedback: List[Dict[str, Any]]):
        for item in feedback:
            task = LangroidTask(None, item['task_content'])
            result = {'performance': item['performance']}
            await self.update_embeddings(task, result)
        
        logger.info(f"Learned from {len(feedback)} feedback items")

    def adjust_learning_rate(self, performance_history: List[float]):
        # Adjust learning rate based on recent performance
        recent_performance = np.mean(performance_history[-10:])
        if recent_performance > 0.8:
            self.learning_rate *= 0.9  # Decrease learning rate if performing well
        elif recent_performance < 0.6:
            self.learning_rate *= 1.1  # Increase learning rate if performing poorly
        self.learning_rate = max(0.001, min(0.1, self.learning_rate))  # Keep learning rate within reasonable bounds
        
        logger.info(f"Adjusted learning rate to {self.learning_rate}")

    def get_info(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
        }
