import logging
from collections import deque
from typing import Any

import numpy as np
import torch
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class IncentiveModel:
    def __init__(
        self,
        num_agents: int,
        num_actions: int,
        learning_rate: float = 0.01,
        history_length: int = 1000,
    ) -> None:
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.incentive_matrix = np.zeros((num_agents, num_actions))
        self.performance_history = {i: deque(maxlen=history_length) for i in range(num_agents)}
        self.task_difficulty_history = deque(maxlen=history_length)
        self.long_term_performance = np.zeros(num_agents)
        self.agent_specialization = np.zeros((num_agents, num_actions))
        self.collaboration_score = np.zeros((num_agents, num_agents))
        self.innovation_score = np.zeros(num_agents)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=5)  # Adjust the number of components as needed

    def calculate_incentive(self, task: dict[str, Any], agent_performance: dict[str, float]) -> dict[str, float]:
        agent_id = self._get_agent_id(task["assigned_agent"])
        action_id = self._map_task_to_action(task)
        base_incentive = self.incentive_matrix[agent_id, action_id]

        # Adjust incentive based on agent's past performance
        performance_factor = agent_performance.get(task["assigned_agent"], 1.0)

        # Adjust incentive based on task difficulty
        task_difficulty = self._calculate_task_difficulty(task)
        difficulty_factor = 1 + task_difficulty

        # Adjust incentive based on long-term performance trend
        trend_factor = self._calculate_performance_trend(agent_id)

        # Adjust incentive based on agent specialization
        specialization_factor = 1 + self.agent_specialization[agent_id, action_id]

        # Adjust incentive based on collaboration score
        collaboration_factor = 1 + np.mean(self.collaboration_score[agent_id])

        # Adjust incentive based on innovation score
        innovation_factor = 1 + self.innovation_score[agent_id]

        adjusted_incentive = (
            base_incentive
            * performance_factor
            * difficulty_factor
            * trend_factor
            * specialization_factor
            * collaboration_factor
            * innovation_factor
        )

        return {
            "agent_id": task["assigned_agent"],
            "incentive": float(adjusted_incentive),
        }

    def update(self, task: dict[str, Any], result: dict[str, Any]) -> None:
        agent_id = self._get_agent_id(task["assigned_agent"])
        action_id = self._map_task_to_action(task)
        reward = self._calculate_reward(result)

        # Update incentive matrix using gradient descent
        gradient = reward - self.incentive_matrix[agent_id, action_id]
        self.incentive_matrix[agent_id, action_id] += self.learning_rate * gradient

        # Update performance history
        self.performance_history[agent_id].append(reward)

        # Update task difficulty history
        task_difficulty = self._calculate_task_difficulty(task)
        self.task_difficulty_history.append(task_difficulty)

        # Update long-term performance
        self.long_term_performance[agent_id] = np.mean(self.performance_history[agent_id])

        # Update agent specialization
        self.agent_specialization[agent_id, action_id] += 0.1 * reward

        # Update collaboration score
        if "collaborators" in task:
            for collaborator in task["collaborators"]:
                collaborator_id = self._get_agent_id(collaborator)
                self.collaboration_score[agent_id, collaborator_id] += 0.1 * reward

        # Update innovation score
        if result.get("innovative_solution", False):
            self.innovation_score[agent_id] += 0.1 * reward

    def update_agent_performance(
        self,
        agent_performance: dict[str, float],
        agent: str,
        result: Any,
        analytics: Any | None = None,
    ) -> float:
        """Update performance tracking for an agent."""
        success = result.get("success", False)
        current = agent_performance.get(agent, 1.0)
        if success:
            agent_performance[agent] = min(current * 1.1, 2.0)
        else:
            agent_performance[agent] = max(current * 0.9, 0.5)
        if analytics is not None:
            try:
                analytics.update_performance_history(agent_performance[agent])
            except Exception as e:  # pragma: no cover - analytics failures shouldn't break
                logger.exception(f"Analytics update failed: {e}")
        logger.info(f"Updated performance for agent {agent}: {agent_performance[agent]}")
        return agent_performance[agent]

    def _get_agent_id(self, agent_name: str) -> int:
        # This method should be implemented to map agent names to their corresponding IDs
        # For now, we'll use a simple hash function
        return hash(agent_name) % self.num_agents

    def _map_task_to_action(self, task: dict[str, Any]) -> int:
        # Implement a more sophisticated mapping of tasks to actions
        # This could involve analyzing the task description, type, or other properties
        task_type = task.get("type", "default")
        task_priority = task.get("priority", 1)
        task_complexity = task.get("complexity", 1)

        # Example mapping logic
        if task_type == "critical":
            return 0
        if task_type == "routine" and task_priority > 5:
            return 1
        if "analysis" in task.get("description", "").lower():
            return 2
        if task_complexity > 7:
            return 3
        return 4 % self.num_actions

    def _calculate_reward(self, result: dict[str, Any]) -> float:
        # Enhanced reward calculation
        base_reward = result.get("success", 0) * 10
        time_factor = max(0, 1 - result.get("time_taken", 0) / result.get("expected_time", 1))
        quality_factor = result.get("quality", 0.5)
        cost_factor = max(0, 1 - result.get("cost", 0) / result.get("budget", 1))
        innovation_factor = 1 + (0.5 if result.get("innovative_solution", False) else 0)
        collaboration_factor = 1 + (0.3 * len(result.get("collaborators", [])))

        return base_reward * (time_factor + quality_factor + cost_factor) / 3 * innovation_factor * collaboration_factor

    def _calculate_task_difficulty(self, task: dict[str, Any]) -> float:
        # Implement logic to calculate task difficulty
        # This could be based on task complexity, estimated time, required skills, etc.
        complexity = task.get("complexity", 1)
        estimated_time = task.get("estimated_time", 1)
        required_skills = len(task.get("required_skills", []))
        priority = task.get("priority", 1)

        difficulty = (complexity * estimated_time * (1 + required_skills * 0.1) * priority) / 100
        return min(max(difficulty, 0), 1)  # Normalize to [0, 1]

    def _calculate_performance_trend(self, agent_id: int) -> float:
        if len(self.performance_history[agent_id]) < 2:
            return 1.0

        x = np.arange(len(self.performance_history[agent_id]))
        y = np.array(self.performance_history[agent_id])
        slope, _, _, _, _ = linregress(x, y)

        # Normalize the slope to a factor around 1
        trend_factor = 1 + slope

        return max(0.5, min(1.5, trend_factor))  # Limit the factor between 0.5 and 1.5

    def analyze_agent_performance(self, agent_id: int) -> dict[str, Any]:
        if len(self.performance_history[agent_id]) == 0:
            return {
                "average": 0,
                "trend": 0,
                "long_term": 0,
                "specialization": [],
                "collaboration": 0,
                "innovation": 0,
            }

        performance_data = np.array(self.performance_history[agent_id]).reshape(-1, 1)
        normalized_data = self.scaler.fit_transform(performance_data)
        pca_result = self.pca.fit_transform(normalized_data)

        return {
            "average": np.mean(self.performance_history[agent_id]),
            "trend": self._calculate_performance_trend(agent_id),
            "long_term": self.long_term_performance[agent_id],
            "specialization": self.agent_specialization[agent_id].tolist(),
            "collaboration": np.mean(self.collaboration_score[agent_id]),
            "innovation": self.innovation_score[agent_id],
            "pca_components": pca_result.flatten().tolist(),
        }

    def get_task_difficulty_summary(self) -> dict[str, float]:
        if len(self.task_difficulty_history) == 0:
            return {"average": 0, "trend": 0}

        average = np.mean(self.task_difficulty_history)
        x = np.arange(len(self.task_difficulty_history))
        y = np.array(self.task_difficulty_history)
        slope, _, _, _, _ = linregress(x, y)

        return {"average": average, "trend": slope}

    def save(self, path: str) -> None:
        torch.save(
            {
                "incentive_matrix": self.incentive_matrix,
                "num_agents": self.num_agents,
                "num_actions": self.num_actions,
                "learning_rate": self.learning_rate,
                "performance_history": self.performance_history,
                "task_difficulty_history": self.task_difficulty_history,
                "long_term_performance": self.long_term_performance,
                "agent_specialization": self.agent_specialization,
                "collaboration_score": self.collaboration_score,
                "innovation_score": self.innovation_score,
            },
            path,
        )
        logger.info(f"Incentive model saved to {path}")

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, weights_only=True)
        self.incentive_matrix = checkpoint["incentive_matrix"]
        self.num_agents = checkpoint["num_agents"]
        self.num_actions = checkpoint["num_actions"]
        self.learning_rate = checkpoint["learning_rate"]
        self.performance_history = checkpoint["performance_history"]
        self.task_difficulty_history = checkpoint["task_difficulty_history"]
        self.long_term_performance = checkpoint["long_term_performance"]
        self.agent_specialization = checkpoint["agent_specialization"]
        self.collaboration_score = checkpoint["collaboration_score"]
        self.innovation_score = checkpoint["innovation_score"]
        logger.info(f"Incentive model loaded from {path}")

    def get_incentive_matrix(self) -> np.ndarray:
        return self.incentive_matrix.copy()

    def update_learning_rate(self, new_learning_rate: float) -> None:
        self.learning_rate = new_learning_rate
        logger.info(f"Learning rate updated to {new_learning_rate}")

    def reset(self) -> None:
        self.incentive_matrix = np.zeros((self.num_agents, self.num_actions))
        self.performance_history = {i: deque(maxlen=self.performance_history[0].maxlen) for i in range(self.num_agents)}
        self.task_difficulty_history.clear()
        self.long_term_performance = np.zeros(self.num_agents)
        self.agent_specialization = np.zeros((self.num_agents, self.num_actions))
        self.collaboration_score = np.zeros((self.num_agents, self.num_agents))
        self.innovation_score = np.zeros(self.num_agents)
        logger.info("Incentive model reset")
