import numpy as np
from typing import Dict, List

class IncentiveModel:
    def __init__(self, num_agents, num_actions, learning_rate=0.01):
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.incentive_matrix = np.zeros((num_agents, num_actions))

    def calculate_incentive(self, task: Dict, agent_performance: Dict) -> Dict[str, float]:
        agent_id = task['assigned_agent']
        action_id = self._map_task_to_action(task)
        base_incentive = self.incentive_matrix[agent_id, action_id]
        
        # Adjust incentive based on agent's past performance
        performance_factor = agent_performance.get(agent_id, 1.0)
        adjusted_incentive = base_incentive * performance_factor
        
        return {'agent_id': agent_id, 'incentive': adjusted_incentive}

    def update(self, task: Dict, result: Dict):
        agent_id = task['assigned_agent']
        action_id = self._map_task_to_action(task)
        reward = self._calculate_reward(result)
        
        # Update incentive matrix using gradient descent
        gradient = reward - self.incentive_matrix[agent_id, action_id]
        self.incentive_matrix[agent_id, action_id] += self.learning_rate * gradient

    def _map_task_to_action(self, task: Dict) -> int:
        # Map task to an action ID based on task properties
        # This is a simplified version; in practice, you'd have a more sophisticated mapping
        return hash(frozenset(task.items())) % self.num_actions

    def _calculate_reward(self, result: Dict) -> float:
        # Calculate reward based on task result
        # This is a simplified version; in practice, you'd have a more sophisticated reward function
        return result.get('success', 0) * 10 - result.get('cost', 0)
