import numpy as np
from typing import Dict, Any, List
from agents.quality_assurance_layer import QualityAssuranceLayer
from agents.utils.task import Task as LangroidTask
from agents.magi.evolution_manager import EvolutionManager
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class ContinuousLearner:
    def __init__(self, quality_assurance_layer: QualityAssuranceLayer, evolution_manager: EvolutionManager, learning_rate: float = 0.01):
        self.quality_assurance_layer = quality_assurance_layer
        self.evolution_manager = evolution_manager
        self.learning_rate = learning_rate
        self.tool_creation_history: List[Dict[str, Any]] = []
        self.task_execution_history: List[Dict[str, Any]] = []
        self.knowledge_base: Dict[str, Any] = {}
        self.performance_history: List[float] = []

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
        direction = task_embedding - target_embedding
        return direction * (performance - 0.5)  # Center performance around 0

    async def learn_from_feedback(self, feedback: List[Dict[str, Any]]):
        for item in feedback:
            task = LangroidTask(None, item['task_content'])
            result = {'performance': item['performance']}
            await self.update_embeddings(task, result)
        
        logger.info(f"Learned from {len(feedback)} feedback items")

    def adjust_learning_rate(self):
        recent_performance = np.mean(self.performance_history[-10:])
        if recent_performance > 0.8:
            self.learning_rate *= 0.9  # Decrease learning rate if performing well
        elif recent_performance < 0.6:
            self.learning_rate *= 1.1  # Increase learning rate if performing poorly
        self.learning_rate = max(0.001, min(0.1, self.learning_rate))  # Keep learning rate within reasonable bounds
        
        logger.info(f"Adjusted learning rate to {self.learning_rate}")

    async def learn_from_tool_creation(self, tool_name: str, tool_code: str, tool_description: str, tool_parameters: Dict[str, Any]):
        self.tool_creation_history.append({
            "name": tool_name,
            "code": tool_code,
            "description": tool_description,
            "parameters": tool_parameters
        })
        logger.info(f"Learned from tool creation: {tool_name}")

    async def learn_from_task_execution(self, task: LangroidTask, result: Dict[str, Any], tools_used: List[str]):
        self.task_execution_history.append({
            "task": task.content,
            "result": result,
            "tools_used": tools_used
        })
        performance = result.get('performance', 0.5)
        self.performance_history.append(performance)
        
        # Trigger evolution process if needed
        if len(self.performance_history) % 10 == 0:  # Every 10 tasks
            await self.trigger_evolution()
        
        logger.info(f"Learned from task execution: {task.content[:50]}...")

    async def trigger_evolution(self):
        # Use the last 100 performance scores (or all if less than 100) for evolution
        recent_performance = self.performance_history[-100:]
        await self.evolution_manager.evolve(recent_performance)

    def extract_tool_creation_insights(self) -> List[str]:
        insights = [
            "Frequently created tools: " + ", ".join(self.get_frequent_tools()),
            "Common tool parameters: " + ", ".join(self.get_common_parameters()),
            "Tool complexity trend: " + self.analyze_tool_complexity()
        ]
        return insights

    def extract_task_execution_insights(self) -> List[str]:
        insights = [
            "Frequently used tools: " + ", ".join(self.get_frequent_tools_used()),
            "Common task types: " + ", ".join(self.get_common_task_types()),
            "Performance trend: " + self.analyze_performance_trend()
        ]
        return insights

    def get_frequent_tools(self) -> List[str]:
        tool_counts = {}
        for entry in self.tool_creation_history:
            tool_counts[entry['name']] = tool_counts.get(entry['name'], 0) + 1
        return sorted(tool_counts, key=tool_counts.get, reverse=True)[:5]

    def get_common_parameters(self) -> List[str]:
        parameter_counts = {}
        for entry in self.tool_creation_history:
            for param in entry['parameters']:
                parameter_counts[param] = parameter_counts.get(param, 0) + 1
        return sorted(parameter_counts, key=parameter_counts.get, reverse=True)[:5]

    def analyze_tool_complexity(self) -> str:
        complexities = [len(entry['code'].split('\n')) for entry in self.tool_creation_history]
        if len(complexities) < 2:
            return "Not enough data"
        slope, _, _, _, _ = stats.linregress(range(len(complexities)), complexities)
        return "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Stable"

    def get_frequent_tools_used(self) -> List[str]:
        tool_counts = {}
        for entry in self.task_execution_history:
            for tool in entry['tools_used']:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
        return sorted(tool_counts, key=tool_counts.get, reverse=True)[:5]

    def get_common_task_types(self) -> List[str]:
        task_type_counts = {}
        for entry in self.task_execution_history:
            task_type = entry['task'].split()[0]  # Assume first word is task type
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
        return sorted(task_type_counts, key=task_type_counts.get, reverse=True)[:5]

    def analyze_performance_trend(self) -> str:
        performances = [entry['result'].get('performance', 0.5) for entry in self.task_execution_history]
        if len(performances) < 2:
            return "Not enough data"
        slope, _, _, _, _ = stats.linregress(range(len(performances)), performances)
        return "Improving" if slope > 0 else "Declining" if slope < 0 else "Stable"

    async def update_knowledge_base(self, topic: str, content: str):
        self.knowledge_base[topic] = content
        logger.info(f"Updated knowledge base with topic: {topic}")

    def get_relevant_knowledge(self, query: str) -> List[str]:
        relevant_topics = [topic for topic in self.knowledge_base if topic.lower() in query.lower()]
        return [self.knowledge_base[topic] for topic in relevant_topics]

    async def get_insights(self) -> Dict[str, Any]:
        return {
            "tool_creation_insights": self.extract_tool_creation_insights(),
            "task_execution_insights": self.extract_task_execution_insights(),
            "knowledge_base_size": len(self.knowledge_base),
            "learning_rate": self.learning_rate,
            "performance_trend": self.analyze_performance_trend(),
            "recent_performance": np.mean(self.performance_history[-10:]) if self.performance_history else None
        }
