import numpy as np
import torch
from typing import Dict, List, Any
import logging
from collections import deque
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import networkx as nx

logger = logging.getLogger(__name__)

class IncentiveModel:
    def __init__(self, num_agents: int, num_actions: int, graph_manager: nx.DiGraph, learning_rate: float = 0.01, history_length: int = 1000):
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
        self.graph_manager = graph_manager
        logger.info("IncentiveModel initialized with GraphManager")
    
    def calculate_incentive(self, task: Dict[str, Any], agent_performance: Dict[str, float]) -> Dict[str, float]:
        agent_id = self._get_agent_id(task['assigned_agent'])
        action_id = self._map_task_to_action(task)
        base_incentive = self.incentive_matrix[agent_id, action_id]
        
        # Adjust incentive based on agent's past performance
        performance_factor = agent_performance.get(task['assigned_agent'], 1.0)
        
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
        
        # Adjust incentive based on graph node properties (e.g., agent's role, task dependencies)
        node_attributes = self.graph_manager.nodes.get(task['task_id'], {})
        role_factor = node_attributes.get('role_factor', 1.0)
        dependency_factor = node_attributes.get('dependency_factor', 1.0)
        
        adjusted_incentive = (
            base_incentive *
            performance_factor *
            difficulty_factor *
            trend_factor *
            specialization_factor *
            collaboration_factor *
            innovation_factor *
            role_factor *
            dependency_factor
        )
        
        return {'agent_id': task['assigned_agent'], 'incentive': float(adjusted_incentive)}
    
    def update(self, task: Dict[str, Any], result: Dict[str, Any]):
        agent_id = self._get_agent_id(task['assigned_agent'])
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
        if 'collaborators' in task:
            for collaborator in task['collaborators']:
                collaborator_id = self._get_agent_id(collaborator)
                self.collaboration_score[agent_id, collaborator_id] += 0.1 * reward
        
        # Update innovation score
        if result.get('innovative_solution', False):
            self.innovation_score[agent_id] += 0.1 * reward
        
        # Modify graph edge weights based on task outcome
        self._update_edge_weights(task, reward)
    
    def _update_edge_weights(self, task: Dict[str, Any], reward: float):
        agent_id = self._get_agent_id(task['assigned_agent'])
        task_id = task['task_id']
        if self.graph_manager.has_edge(agent_id, task_id):
            # Example: Increase weight if reward is positive, decrease otherwise
            current_weight = self.graph_manager[agent_id][task_id].get('weight', 1.0)
            new_weight = current_weight * (1 + reward) if reward > 0 else current_weight * (1 + reward)
            self.graph_manager[agent_id][task_id]['weight'] = max(new_weight, 0.1)  # Prevent weight from dropping below 0.1
            logger.info(f"Updated edge weight from agent {agent_id} to task {task_id} to {new_weight}")
        else:
            # If no edge exists, create one with initial weight based on reward
            initial_weight = 1.0 + reward
            self.graph_manager.add_edge(agent_id, task_id, weight=max(initial_weight, 0.1))
            logger.info(f"Added edge from agent {agent_id} to task {task_id} with weight {initial_weight}")
    
    def _get_agent_id(self, agent_name: str) -> int:
        # This method should be implemented to map agent names to their corresponding IDs
        # For now, we'll use a simple hash function
        return hash(agent_name) % self.num_agents
    
    def _map_task_to_action(self, task: Dict[str, Any]) -> int:
        # Implement a more sophisticated mapping of tasks to actions
        # This could involve analyzing the task description, type, or other properties
        task_type = task.get('type', 'default')
        task_priority = task.get('priority', 1)
        task_complexity = task.get('complexity', 1)
        
        # Example mapping logic
        if task_type == 'critical':
            return 0
        elif task_type == 'routine' and task_priority > 5:
            return 1
        elif 'analysis' in task.get('description', '').lower():
            return 2
        elif task_complexity > 7:
            return 3
        else:
            return 4 % self.num_actions
    
    def _calculate_reward(self, result: Dict[str, Any]) -> float:
        # Enhanced reward calculation
        base_reward = result.get('success', 0) * 10
        time_factor = max(0, 1 - result.get('time_taken', 0) / result.get('expected_time', 1))
        quality_factor = result.get('quality', 0.5)
        cost_factor = max(0, 1 - result.get('cost', 0) / result.get('budget', 1))
        innovation_factor = 1 + (0.5 if result.get('innovative_solution', False) else 0)
        collaboration_factor = 1 + (0.3 * len(result.get('collaborators', [])))
        
        return base_reward * (time_factor + quality_factor + cost_factor) / 3 * innovation_factor * collaboration_factor
    
    def _calculate_task_difficulty(self, task: Dict[str, Any]) -> float:
        # Implement logic to calculate task difficulty
        # This could be based on task complexity, estimated time, required skills, etc.
        complexity = task.get('complexity', 1)
        estimated_time = task.get('estimated_time', 1)
        required_skills = len(task.get('required_skills', []))
        priority = task.get('priority', 1)
        
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
    
    def analyze_agent_performance(self, agent_id: int) -> Dict[str, Any]:
        if len(self.performance_history[agent_id]) == 0:
            return {"average": 0, "trend": 0, "long_term": 0, "specialization": [], "collaboration": 0, "innovation": 0}
        
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
            "pca_components": pca_result.flatten().tolist()
        }
    
    def get_task_difficulty_summary(self) -> Dict[str, float]:
        if len(self.task_difficulty_history) == 0:
            return {"average": 0, "trend": 0}
        
        average = np.mean(self.task_difficulty_history)
        x = np.arange(len(self.task_difficulty_history))
        y = np.array(self.task_difficulty_history)
        slope, _, _, _, _ = linregress(x, y)
        
        return {
            "average": average,
            "trend": slope
        }
    
    def calculate_incentive_distribution(self) -> Dict[str, Any]:
        try:
            total_incentive = np.sum(self.incentive_matrix)
            if total_incentive == 0:
                distribution = {f"Agent_{i}": 0 for i in range(self.num_agents)}
            else:
                distribution = {f"Agent_{i}": float(np.sum(self.incentive_matrix[i]) / total_incentive) for i in range(self.num_agents)}
            
            return {
                "total_incentive": float(total_incentive),
                "distribution": distribution
            }
        except Exception as e:
            logger.exception(f"Error calculating incentive distribution: {str(e)}")
            return {"total_incentive": 0, "distribution": {}}
    
    def save(self, path: str):
        torch.save({
            'incentive_matrix': self.incentive_matrix,
            'num_agents': self.num_agents,
            'num_actions': self.num_actions,
            'learning_rate': self.learning_rate,
            'performance_history': {k: list(v) for k, v in self.performance_history.items()},
            'task_difficulty_history': list(self.task_difficulty_history),
            'long_term_performance': self.long_term_performance.tolist(),
            'agent_specialization': self.agent_specialization.tolist(),
            'collaboration_score': self.collaboration_score.tolist(),
            'innovation_score': self.innovation_score.tolist()
        }, path)
        logger.info(f"Incentive model saved to {path}")
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.incentive_matrix = checkpoint['incentive_matrix']
        self.num_agents = checkpoint['num_agents']
        self.num_actions = checkpoint['num_actions']
        self.learning_rate = checkpoint['learning_rate']
        self.performance_history = {int(k): deque(v, maxlen=self.performance_history[0].maxlen) for k, v in checkpoint['performance_history'].items()}
        self.task_difficulty_history = deque(checkpoint['task_difficulty_history'], maxlen=self.task_difficulty_history.maxlen)
        self.long_term_performance = np.array(checkpoint['long_term_performance'])
        self.agent_specialization = np.array(checkpoint['agent_specialization'])
        self.collaboration_score = np.array(checkpoint['collaboration_score'])
        self.innovation_score = np.array(checkpoint['innovation_score'])
        logger.info(f"Incentive model loaded from {path}")
    
    def get_incentive_matrix(self) -> np.ndarray:
        return self.incentive_matrix.copy()
    
    def update_learning_rate(self, new_learning_rate: float):
        self.learning_rate = new_learning_rate
        logger.info(f"Learning rate updated to {new_learning_rate}")
    
    def reset(self):
        self.incentive_matrix = np.zeros((self.num_agents, self.num_actions))
        self.performance_history = {i: deque(maxlen=self.performance_history[i].maxlen) for i in range(self.num_agents)}
        self.task_difficulty_history.clear()
        self.long_term_performance = np.zeros(self.num_agents)
        self.agent_specialization = np.zeros((self.num_agents, self.num_actions))
        self.collaboration_score = np.zeros((self.num_agents, self.num_agents))
        self.innovation_score = np.zeros(self.num_agents)
        logger.info("Incentive model reset")
    
    # New methods as per TODO.md
    
    def calculate_incentives_based_on_graph(self):
        """
        Calculate incentives based on graph edges and node properties.
        """
        for agent_id, agent_data in self.graph_manager.G.nodes(data=True):
            if agent_data.get('type') != 'agent':
                continue
            for task_id in self.graph_manager.G.successors(agent_id):
                edge_data = self.graph_manager.G.get_edge_data(agent_id, task_id, default={})
                base_incentive = self.incentive_matrix[agent_id][self._map_task_to_action({'description': self.graph_manager.G.nodes[task_id].get('description', '')})]
                role_factor = edge_data.get('role_factor', 1.0)
                dependency_factor = edge_data.get('dependency_factor', 1.0)
                self.incentive_matrix[agent_id][self._map_task_to_action({'description': self.graph_manager.G.nodes[task_id].get('description', '')})] = base_incentive * role_factor * dependency_factor
                logger.info(f"Adjusted incentive for agent {agent_id} on task {task_id} based on graph factors")
    
    def analyze_incentive_distribution_across_graph(self) -> Dict[str, Any]:
        """
        Analyze how incentives are distributed across the agent-task graph.
        """
        try:
            distribution = self.calculate_incentive_distribution()
            graph_metrics = {
                "average_incentive": float(np.mean(self.incentive_matrix)),
                "total_incentive": float(np.sum(self.incentive_matrix)),
                "max_incentive": float(np.max(self.incentive_matrix)),
                "min_incentive": float(np.min(self.incentive_matrix))
            }
            return {
                "incentive_distribution": distribution,
                "graph_metrics": graph_metrics
            }
        except Exception as e:
            logger.exception(f"Error analyzing incentive distribution across graph: {str(e)}")
            return {"incentive_distribution": {}, "graph_metrics": {}}
    
    def _map_task_to_action(self, task: Dict[str, Any]) -> int:
        """
        Enhanced mapping of tasks to actions based on graph node properties.
        """
        # Implement mapping based on node attributes
        task_type = task.get('type', 'default')
        task_priority = task.get('priority', 1)
        task_complexity = task.get('complexity', 1)
        
        node_attributes = self.graph_manager.G.nodes.get(task.get('task_id', ''), {})
        role_factor = node_attributes.get('role_factor', 1.0)
        dependency_factor = node_attributes.get('dependency_factor', 1.0)
        
        # Example mapping logic incorporating graph factors
        if task_type == 'critical':
            return 0
        elif task_type == 'routine' and task_priority > 5:
            return 1
        elif 'analysis' in task.get('description', '').lower():
            return 2
        elif task_complexity > 7:
            return 3
        else:
            return int((4 * role_factor * dependency_factor)) % self.num_actions
    
    def _update_graph_based_on_task_outcome(self, task: Dict[str, Any], reward: float):
        """
        Update graph edge weights based on task outcomes.
        """
        agent_id = self._get_agent_id(task['assigned_agent'])
        task_id = task['task_id']
        if self.graph_manager.G.has_edge(agent_id, task_id):
            current_weight = self.graph_manager.G[agent_id][task_id].get('weight', 1.0)
            new_weight = current_weight * (1 + reward) if reward > 0 else current_weight * (1 + reward)
            self.graph_manager.G[agent_id][task_id]['weight'] = max(new_weight, 0.1)  # Prevent weight from dropping below 0.1
            logger.info(f"Updated edge weight from agent {agent_id} to task {task_id} to {new_weight}")
        else:
            initial_weight = 1.0 + reward
            self.graph_manager.G.add_edge(agent_id, task_id, weight=max(initial_weight, 0.1))
            logger.info(f"Added edge from agent {agent_id} to task {task_id} with weight {initial_weight}")
    
    # Modified update method to include graph updates
    def update(self, task: Dict[str, Any], result: Dict[str, Any]):
        super().update(task, result)
        self._update_graph_based_on_task_outcome(task, self._calculate_reward(result))
