import numpy as np
import torch
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class IncentiveModel:
    def __init__(self, num_agents: int, num_actions: int, learning_rate: float = 0.01):
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.incentive_matrix = np.zeros((num_agents, num_actions))

    def calculate_incentive(self, task: Dict[str, Any], agent_performance: Dict[str, float]) -> Dict[str, float]:
        agent_id = self._get_agent_id(task['assigned_agent'])
        action_id = self._map_task_to_action(task)
        base_incentive = self.incentive_matrix[agent_id, action_id]
        
        # Adjust incentive based on agent's past performance
        performance_factor = agent_performance.get(task['assigned_agent'], 1.0)
        adjusted_incentive = base_incentive * performance_factor
        
        return {'agent_id': task['assigned_agent'], 'incentive': float(adjusted_incentive)}

    def update(self, task: Dict[str, Any], result: Dict[str, Any]):
        agent_id = self._get_agent_id(task['assigned_agent'])
        action_id = self._map_task_to_action(task)
        reward = self._calculate_reward(result)
        
        # Update incentive matrix using gradient descent
        gradient = reward - self.incentive_matrix[agent_id, action_id]
        self.incentive_matrix[agent_id, action_id] += self.learning_rate * gradient

    def _get_agent_id(self, agent_name: str) -> int:
        # This method should be implemented to map agent names to their corresponding IDs
        # For now, we'll use a simple hash function
        return hash(agent_name) % self.num_agents

    def _map_task_to_action(self, task: Dict[str, Any]) -> int:
        # Implement a more sophisticated mapping of tasks to actions
        # This could involve analyzing the task description, type, or other properties
        task_type = task.get('type', 'default')
        task_priority = task.get('priority', 1)
        
        # Example mapping logic
        if task_type == 'critical':
            return 0
        elif task_type == 'routine' and task_priority > 5:
            return 1
        elif 'analysis' in task.get('description', '').lower():
            return 2
        else:
            return 3 % self.num_actions

    def _calculate_reward(self, result: Dict[str, Any]) -> float:
        # Enhanced reward calculation
        base_reward = result.get('success', 0) * 10
        time_factor = max(0, 1 - result.get('time_taken', 0) / result.get('expected_time', 1))
        quality_factor = result.get('quality', 0.5)
        cost_factor = max(0, 1 - result.get('cost', 0) / result.get('budget', 1))
        
        return base_reward * (time_factor + quality_factor + cost_factor) / 3

    def save(self, path: str):
        try:
            torch.save({
                'incentive_matrix': self.incentive_matrix,
                'num_agents': self.num_agents,
                'num_actions': self.num_actions,
                'learning_rate': self.learning_rate
            }, path)
            logger.info(f"Incentive model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving incentive model: {str(e)}")
            raise

    def load(self, path: str):
        try:
            checkpoint = torch.load(path)
            self.incentive_matrix = checkpoint['incentive_matrix']
            self.num_agents = checkpoint['num_agents']
            self.num_actions = checkpoint['num_actions']
            self.learning_rate = checkpoint['learning_rate']
            logger.info(f"Incentive model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading incentive model: {str(e)}")
            raise

    def get_incentive_matrix(self) -> np.ndarray:
        return self.incentive_matrix.copy()

    def update_learning_rate(self, new_learning_rate: float):
        self.learning_rate = new_learning_rate
        logger.info(f"Learning rate updated to {new_learning_rate}")

    def reset(self):
        self.incentive_matrix = np.zeros((self.num_agents, self.num_actions))
        logger.info("Incentive matrix reset to zero")
