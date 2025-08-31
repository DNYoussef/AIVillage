from collections import deque
import logging
from typing import Any

import numpy as np
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch

# Import constants for magic literal elimination
from infrastructure.constants import (
    PerformanceConstants, IncentiveDefaults, RewardConstants, TaskDifficultyConstants,
    PerformanceFieldNames, TaskConstants, TaskActionMapping, MessageConstants, get_config_manager
)
from infrastructure.constants.task_constants import TaskType

logger = logging.getLogger(__name__)


class IncentiveModel:
    def __init__(
        self,
        num_agents: int,
        num_actions: int,
        learning_rate: float = None,
        history_length: int = None,
    ) -> None:
        self.num_agents = num_agents
        self.num_actions = num_actions
        self._config_manager = get_config_manager()
        self.learning_rate = learning_rate or self._config_manager.get_learning_rate()
        self.incentive_matrix = np.zeros((num_agents, num_actions))
        history_len = history_length or self._config_manager.get_history_length()
        self.performance_history = {i: deque(maxlen=history_len) for i in range(num_agents)}
        self.task_difficulty_history = deque(maxlen=history_len)
        self.long_term_performance = np.zeros(num_agents)
        self.agent_specialization = np.zeros((num_agents, num_actions))
        self.collaboration_score = np.zeros((num_agents, num_agents))
        self.innovation_score = np.zeros(num_agents)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=PerformanceConstants.PCA_COMPONENTS)

    def calculate_incentive(self, task: dict[str, Any], agent_performance: dict[str, float]) -> dict[str, float]:
        agent_id = self._get_agent_id(task[PerformanceFieldNames.ASSIGNED_AGENT_FIELD])
        action_id = self._map_task_to_action(task)
        base_incentive = self.incentive_matrix[agent_id, action_id]

        # Adjust incentive based on agent's past performance
        performance_factor = agent_performance.get(task[PerformanceFieldNames.ASSIGNED_AGENT_FIELD], PerformanceConstants.NEUTRAL_TREND)

        # Adjust incentive based on task difficulty
        task_difficulty = self._calculate_task_difficulty(task)
        difficulty_factor = PerformanceConstants.NEUTRAL_TREND + task_difficulty

        # Adjust incentive based on long-term performance trend
        trend_factor = self._calculate_performance_trend(agent_id)

        # Adjust incentive based on agent specialization
        specialization_factor = PerformanceConstants.NEUTRAL_TREND + self.agent_specialization[agent_id, action_id]

        # Adjust incentive based on collaboration score
        collaboration_factor = PerformanceConstants.NEUTRAL_TREND + np.mean(self.collaboration_score[agent_id])

        # Adjust incentive based on innovation score
        innovation_factor = PerformanceConstants.NEUTRAL_TREND + self.innovation_score[agent_id]

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
            PerformanceFieldNames.ASSIGNED_AGENT_FIELD: task[PerformanceFieldNames.ASSIGNED_AGENT_FIELD],
            MessageConstants.INCENTIVE: float(adjusted_incentive),
        }

    def update(self, task: dict[str, Any], result: dict[str, Any]) -> None:
        agent_id = self._get_agent_id(task[PerformanceFieldNames.ASSIGNED_AGENT_FIELD])
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
        self.agent_specialization[agent_id, action_id] += PerformanceConstants.SPECIALIZATION_RATE * reward

        # Update collaboration score
        if PerformanceFieldNames.COLLABORATORS_FIELD in task:
            for collaborator in task[PerformanceFieldNames.COLLABORATORS_FIELD]:
                collaborator_id = self._get_agent_id(collaborator)
                self.collaboration_score[agent_id, collaborator_id] += PerformanceConstants.COLLABORATION_RATE * reward

        # Update innovation score
        if result.get(PerformanceFieldNames.INNOVATIVE_SOLUTION_FIELD, False):
            self.innovation_score[agent_id] += PerformanceConstants.INNOVATION_RATE * reward

    def update_agent_performance(
        self,
        agent_performance: dict[str, float],
        agent: str,
        result: Any,
        analytics: Any | None = None,
    ) -> float:
        """Update performance tracking for an agent."""
        success = result.get(PerformanceFieldNames.SUCCESS_FIELD, False)
        current = agent_performance.get(agent, PerformanceConstants.NEUTRAL_TREND)
        if success:
            agent_performance[agent] = min(current * PerformanceConstants.PERFORMANCE_BOOST_FACTOR, 
                                         self._config_manager.get_max_performance_multiplier())
        else:
            agent_performance[agent] = max(current * PerformanceConstants.PERFORMANCE_PENALTY_FACTOR, 
                                         self._config_manager.get_min_performance_multiplier())
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
        task_type = task.get(PerformanceFieldNames.TASK_TYPE_FIELD, TaskType.DEFAULT.value)
        task_priority = task.get(PerformanceFieldNames.PRIORITY_FIELD, TaskConstants.DEFAULT_PRIORITY)
        task_complexity = task.get(PerformanceFieldNames.COMPLEXITY_FIELD, TaskDifficultyConstants.DEFAULT_COMPLEXITY)

        # Example mapping logic
        if task_type == TaskType.CRITICAL.value:
            return TaskActionMapping.CRITICAL.value
        if task_type == TaskType.ROUTINE.value and task_priority > TaskConstants.ROUTINE_PRIORITY_THRESHOLD:
            return TaskActionMapping.ROUTINE_HIGH_PRIORITY.value
        if TaskConstants.ANALYSIS_KEYWORD in task.get(PerformanceFieldNames.DESCRIPTION_FIELD, "").lower():
            return TaskActionMapping.ANALYSIS.value
        if task_complexity > TaskConstants.HIGH_COMPLEXITY_THRESHOLD:
            return TaskActionMapping.HIGH_COMPLEXITY.value
        return TaskActionMapping.DEFAULT.value % self.num_actions

    def _calculate_reward(self, result: dict[str, Any]) -> float:
        # Enhanced reward calculation
        base_reward = result.get(PerformanceFieldNames.SUCCESS_FIELD, 0) * RewardConstants.BASE_SUCCESS_REWARD
        time_factor = max(0, RewardConstants.TIME_WEIGHT - 
                         result.get(PerformanceFieldNames.TIME_TAKEN_FIELD, RewardConstants.DEFAULT_TIME_TAKEN) / 
                         result.get(PerformanceFieldNames.EXPECTED_TIME_FIELD, RewardConstants.DEFAULT_EXPECTED_TIME))
        quality_factor = result.get(PerformanceFieldNames.QUALITY_FIELD, RewardConstants.DEFAULT_QUALITY_SCORE)
        cost_factor = max(0, RewardConstants.COST_WEIGHT - 
                         result.get(PerformanceFieldNames.COST_FIELD, RewardConstants.DEFAULT_COST) / 
                         result.get(PerformanceFieldNames.BUDGET_FIELD, RewardConstants.DEFAULT_BUDGET))
        innovation_factor = PerformanceConstants.NEUTRAL_TREND + (RewardConstants.INNOVATION_BONUS if 
                           result.get(PerformanceFieldNames.INNOVATIVE_SOLUTION_FIELD, False) else 0)
        collaboration_factor = PerformanceConstants.NEUTRAL_TREND + (RewardConstants.COLLABORATION_BONUS * 
                              len(result.get(PerformanceFieldNames.COLLABORATORS_FIELD, [])))

        return base_reward * (time_factor + quality_factor + cost_factor) / RewardConstants.REWARD_NORMALIZATION_DIVISOR * innovation_factor * collaboration_factor

    def _calculate_task_difficulty(self, task: dict[str, Any]) -> float:
        # Implement logic to calculate task difficulty
        # This could be based on task complexity, estimated time, required skills, etc.
        complexity = task.get(PerformanceFieldNames.COMPLEXITY_FIELD, TaskDifficultyConstants.DEFAULT_COMPLEXITY)
        estimated_time = task.get(PerformanceFieldNames.ESTIMATED_TIME_FIELD, TaskDifficultyConstants.DEFAULT_ESTIMATED_TIME)
        required_skills = len(task.get(PerformanceFieldNames.REQUIRED_SKILLS_FIELD, []))
        priority = task.get(PerformanceFieldNames.PRIORITY_FIELD, TaskDifficultyConstants.DEFAULT_PRIORITY)

        difficulty = (complexity * estimated_time * 
                     (PerformanceConstants.NEUTRAL_TREND + required_skills * TaskDifficultyConstants.SKILL_MULTIPLIER) * 
                     priority) / TaskDifficultyConstants.DIFFICULTY_DIVISOR
        return min(max(difficulty, TaskDifficultyConstants.DIFFICULTY_MIN), TaskDifficultyConstants.DIFFICULTY_MAX)

    def _calculate_performance_trend(self, agent_id: int) -> float:
        if len(self.performance_history[agent_id]) < 2:
            return PerformanceConstants.NEUTRAL_TREND

        x = np.arange(len(self.performance_history[agent_id]))
        y = np.array(self.performance_history[agent_id])
        slope, _, _, _, _ = linregress(x, y)

        # Normalize the slope to a factor around 1
        trend_factor = PerformanceConstants.NEUTRAL_TREND + slope

        return max(PerformanceConstants.MIN_TREND, min(PerformanceConstants.MAX_TREND, trend_factor))

    def analyze_agent_performance(self, agent_id: int) -> dict[str, Any]:
        if len(self.performance_history[agent_id]) == 0:
            return {
                PerformanceFieldNames.AVERAGE_FIELD: 0,
                PerformanceFieldNames.TREND_FIELD: 0,
                PerformanceFieldNames.LONG_TERM_FIELD: 0,
                PerformanceFieldNames.SPECIALIZATION_FIELD: [],
                PerformanceFieldNames.COLLABORATION_FIELD: 0,
                PerformanceFieldNames.INNOVATION_FIELD: 0,
            }

        performance_data = np.array(self.performance_history[agent_id]).reshape(-1, 1)
        normalized_data = self.scaler.fit_transform(performance_data)
        pca_result = self.pca.fit_transform(normalized_data)

        return {
            PerformanceFieldNames.AVERAGE_FIELD: np.mean(self.performance_history[agent_id]),
            PerformanceFieldNames.TREND_FIELD: self._calculate_performance_trend(agent_id),
            PerformanceFieldNames.LONG_TERM_FIELD: self.long_term_performance[agent_id],
            PerformanceFieldNames.SPECIALIZATION_FIELD: self.agent_specialization[agent_id].tolist(),
            PerformanceFieldNames.COLLABORATION_FIELD: np.mean(self.collaboration_score[agent_id]),
            PerformanceFieldNames.INNOVATION_FIELD: self.innovation_score[agent_id],
            PerformanceFieldNames.PCA_COMPONENTS_FIELD: pca_result.flatten().tolist(),
        }

    def get_task_difficulty_summary(self) -> dict[str, float]:
        if len(self.task_difficulty_history) == 0:
            return {PerformanceFieldNames.AVERAGE_FIELD: 0, PerformanceFieldNames.TREND_FIELD: 0}

        average = np.mean(self.task_difficulty_history)
        x = np.arange(len(self.task_difficulty_history))
        y = np.array(self.task_difficulty_history)
        slope, _, _, _, _ = linregress(x, y)

        return {PerformanceFieldNames.AVERAGE_FIELD: average, PerformanceFieldNames.TREND_FIELD: slope}

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
        history_len = self._config_manager.get_history_length()
        self.performance_history = {i: deque(maxlen=history_len) for i in range(self.num_agents)}
        self.task_difficulty_history.clear()
        self.long_term_performance = np.zeros(self.num_agents)
        self.agent_specialization = np.zeros((self.num_agents, self.num_actions))
        self.collaboration_score = np.zeros((self.num_agents, self.num_agents))
        self.innovation_score = np.zeros(self.num_agents)
        logger.info("Incentive model reset")
