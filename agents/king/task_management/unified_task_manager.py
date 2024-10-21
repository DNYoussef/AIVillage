import logging
import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import uuid
from communications.protocol import StandardCommunicationProtocol, Message, MessageType, Priority
from agents.utils.exceptions import AIVillageException
from .incentive_model import IncentiveModel
from .subgoal_generator import SubGoalGenerator
from ..analytics.unified_analytics import UnifiedAnalytics
from ..planning.unified_planning_and_decision import UnifiedPlanningAndDecision, GraphManager
from networkx import DiGraph
import time
import networkx as nx

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    assigned_agents: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 1
    deadline: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    result: Any = None
    created_at: float = field(default_factory=lambda: time.time())
    completed_at: Optional[float] = None

    def update_status(self, new_status: TaskStatus):
        self.status = new_status
        if new_status == TaskStatus.COMPLETED:
            self.completed_at = time.time()
        return self

    def update_result(self, result: Any):
        self.result = result
        return self

@dataclass
class Project:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    tasks: Dict[str, Task] = field(default_factory=dict)
    status: str = "initialized"
    progress: float = 0.0
    resources: Dict[str, Any] = field(default_factory=dict)

class UnifiedManagement:
    def __init__(self, communication_protocol: StandardCommunicationProtocol, decision_maker: UnifiedPlanningAndDecision, num_agents: int, num_actions: int):
        self.communication_protocol = communication_protocol
        self.decision_maker = decision_maker
        self.pending_tasks: deque[Task] = deque()
        self.ongoing_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        self.projects: Dict[str, Project] = {}
        self.incentive_model = IncentiveModel(num_agents, num_actions, graph_manager=self.decision_maker.graph_manager)
        self.agent_performance: Dict[str, float] = {}
        self.available_agents: List[str] = []
        self.subgoal_generator = SubGoalGenerator()
        self.unified_analytics = UnifiedAnalytics()
        self.batch_size = 5
        self.graph_manager = self.decision_maker.graph_manager
        logger.info("UnifiedManagement initialized with GraphManager")

    async def create_task(self, description: str, agent: str, priority: int = 1, deadline: Optional[str] = None, project_id: Optional[str] = None) -> Task:
        try:
            task = Task(description=description, assigned_agents=[agent], priority=priority, deadline=deadline)
            self.pending_tasks.append(task)
            logger.info(f"Created task: {task.id} for agent: {agent}")

            # Add task node to graph
            self.graph_manager.add_task_node(task.id, {
                "description": description,
                "priority": priority,
                "deadline": deadline,
                "status": task.status.value,
                "dependencies": task.dependencies
            })

            if project_id:
                await self.add_task_to_project(project_id, task.id, {"description": description, "agent": agent})

            return task
        except Exception as e:
            logger.exception(f"Error creating task: {str(e)}")
            raise AIVillageException(f"Error creating task: {str(e)}")

    async def create_complex_task(self, description: str, context: Dict[str, Any]) -> List[Task]:
        try:
            # Use decision maker to get task breakdown
            breakdown = await self.decision_maker.make_decision(description, eudaimonia_score=0.5)
            plan_tree = breakdown.get('full_plan', {}).get('plan_tree', {})

            # Convert plan_tree into graph nodes
            nx_plan_graph = self.graph_manager._convert_plan_to_graph(plan_tree)
            self.graph_manager.merge_task_graph(nx_plan_graph)
            self.graph_manager.visualize_graph()

            # Extract tasks from plan_tree
            tasks = await self._extract_tasks_from_plan(plan_tree)

            # Create tasks and add to pending_tasks
            created_tasks = []
            for task_info in tasks:
                agent = task_info.get('assigned_agent', self.available_agents[0] if self.available_agents else "default_agent")
                task = await self.create_task(task_info['description'], agent, priority=task_info.get('priority', 1), deadline=task_info.get('deadline'), project_id=task_info.get('project_id'))
                created_tasks.append(task)

            return created_tasks
        except Exception as e:
            logger.exception(f"Error creating complex task: {str(e)}")
            raise AIVillageException(f"Error creating complex task: {str(e)}")

    async def _extract_tasks_from_plan(self, plan_tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        tasks = []

        def extract_tasks_recursive(node):
            if 'tasks' in node:
                for task in node['tasks']:
                    tasks.append(task)
            for sub_goal in node.get('sub_goals', []):
                extract_tasks_recursive(sub_goal)

        extract_tasks_recursive(plan_tree)
        return tasks

    async def assign_task(self, task: Task):
        try:
            # Use graph traversal to select the best agent
            agent = await self._select_optimal_agent(task)
            task.assigned_agents = [agent]
            self.ongoing_tasks[task.id] = task.update_status(TaskStatus.IN_PROGRESS)
            incentive = self.incentive_model.calculate_incentive({'assigned_agent': agent, 'task_id': task.id}, self.agent_performance)
            await self.notify_agent_with_incentive(agent, task, incentive['incentive'])

            # Update graph with task assignment
            self.graph_manager.G.edges[agent, task.id]['status'] = 'assigned'
            self.graph_manager.G.edges[agent, task.id]['incentive'] = incentive['incentive']
            logger.info(f"Assigned task {task.id} to agent {agent} with incentive {incentive['incentive']}")
        except Exception as e:
            logger.exception(f"Error assigning task: {str(e)}")
            raise AIVillageException(f"Error assigning task: {str(e)}")

    async def _select_optimal_agent(self, task: Task) -> str:
        """
        Select the optimal agent for the given task using graph traversal.
        """
        try:
            # For simplicity, select the agent with the highest edge weight to the task
            agents = [n for n, d in self.graph_manager.G.nodes(data=True) if d.get('type') == 'agent']
            best_agent = None
            best_weight = -float('inf')
            for agent in agents:
                if self.graph_manager.G.has_edge(agent, task.id):
                    weight = self.graph_manager.G[agent][task.id].get('weight', 1.0)
                else:
                    weight = self.graph_manager.G[agent][task.id].get('weight', 1.0)
                if weight > best_weight:
                    best_weight = weight
                    best_agent = agent
            return best_agent if best_agent else 'undecided'
        except Exception as e:
            logger.exception(f"Error selecting optimal agent: {str(e)}")
            return 'undecided'

    async def notify_agent_with_incentive(self, agent: str, task: Task, incentive: float):
        try:
            message = Message(
                type=MessageType.TASK,
                sender="UnifiedManagement",
                receiver=agent,
                content={"task_id": task.id, "description": task.description, "incentive": incentive},
                priority=Priority.MEDIUM
            )
            await self.communication_protocol.send_message(message)
            logger.info(f"Notified agent {agent} with task {task.id} and incentive {incentive}")
        except Exception as e:
            logger.exception(f"Error notifying agent with incentive: {str(e)}")
            raise AIVillageException(f"Error notifying agent with incentive: {str(e)}")

    async def complete_task(self, task_id: str, result: Any):
        try:
            if task_id not in self.ongoing_tasks:
                raise AIVillageException(f"Task {task_id} not found in ongoing tasks")
            task = self.ongoing_tasks[task_id]
            updated_task = task.update_status(TaskStatus.COMPLETED).update_result(result)
            self.completed_tasks.append(updated_task)
            del self.ongoing_tasks[task_id]

            # Update dependent tasks
            await self.update_dependent_tasks(updated_task)

            # Update graph based on task completion
            self.graph_manager.G.edges[task.assigned_agents[0], task_id]['status'] = 'completed'
            self.graph_manager.G.nodes[task_id]['result'] = result
            logger.info(f"Task {task_id} completed with result: {result}")

            agent = task.assigned_agents[0]
            self.incentive_model.update({'assigned_agent': agent, 'task_id': task_id}, result)
            self.update_agent_performance(agent, result)

            completion_time = (updated_task.completed_at - updated_task.created_at)
            success = result.get('success', False)
            self.unified_analytics.record_task_completion(task_id, completion_time, success)

            # Update project status if the task belongs to a project
            for project in self.projects.values():
                if task_id in project.tasks:
                    project.tasks[task_id] = updated_task
                    await self.update_project_status(project.id)
                    break
        except Exception as e:
            logger.exception(f"Error completing task: {str(e)}")
            raise AIVillageException(f"Error completing task: {str(e)}")

    async def update_dependent_tasks(self, completed_task: Task):
        try:
            for task in list(self.pending_tasks):
                if completed_task.id in task.dependencies:
                    task.dependencies.remove(completed_task.id)
                    if not task.dependencies:
                        self.pending_tasks.remove(task)
                        await self.assign_task(task)
        except Exception as e:
            logger.exception(f"Error updating dependent tasks: {str(e)}")
            raise AIVillageException(f"Error updating dependent tasks: {str(e)}")

    def update_agent_performance(self, agent: str, result: Any):
        try:
            success = result.get('success', False)
            current_performance = self.agent_performance.get(agent, 1.0)
            if success:
                self.agent_performance[agent] = min(current_performance * 1.1, 2.0)  # Cap at 2.0
            else:
                self.agent_performance[agent] = max(current_performance * 0.9, 0.5)  # Floor at 0.5
            self.unified_analytics.update_performance_history(self.agent_performance[agent])
            logger.info(f"Updated performance for agent {agent}: {self.agent_performance[agent]}")
        except Exception as e:
            logger.exception(f"Error updating agent performance: {str(e)}")
            raise AIVillageException(f"Error updating agent performance: {str(e)}")

    async def create_project(self, name: str, description: str) -> str:
        try:
            project_id = str(uuid.uuid4())
            self.projects[project_id] = Project(id=project_id, name=name, description=description)
            logger.info(f"Created project: {project_id}")
            return project_id
        except Exception as e:
            logger.exception(f"Error creating project: {str(e)}")
            raise AIVillageException(f"Error creating project: {str(e)}")

    async def get_all_projects(self) -> Dict[str, Project]:
        return self.projects

    async def get_project(self, project_id: str) -> Project:
        project = self.projects.get(project_id)
        if not project:
            raise AIVillageException(f"Project with ID {project_id} not found")
        return project

    async def update_project_status(self, project_id: str, status: str = None, progress: float = None):
        try:
            project = await self.get_project(project_id)
            if status:
                project.status = status
            if progress is not None:
                project.progress = progress
            logger.info(f"Updated project {project_id} - Status: {status}, Progress: {progress}")
        except Exception as e:
            logger.exception(f"Error updating project status: {str(e)}")
            raise AIVillageException(f"Error updating project status: {str(e)}")

    async def add_task_to_project(self, project_id: str, task_id: str, task_data: Dict[str, Any]):
        try:
            project = await self.get_project(project_id)
            project.tasks[task_id] = Task(id=task_id, **task_data)
            logger.info(f"Added task {task_id} to project {project_id}")

            # Add dependency in graph if applicable
            dependencies = task_data.get('dependencies', [])
            for dep in dependencies:
                if dep in project.tasks:
                    self.graph_manager.G.add_edge(project.tasks[dep].assigned_agents[0], task_id, dependency=True)
                    logger.info(f"Added dependency from {dep} to {task_id} in graph")
        except Exception as e:
            logger.exception(f"Error adding task to project: {str(e)}")
            raise AIVillageException(f"Error adding task to project: {str(e)}")

    async def get_project_tasks(self, project_id: str) -> List[Task]:
        try:
            project = await self.get_project(project_id)
            return list(project.tasks.values())
        except Exception as e:
            logger.exception(f"Error getting project tasks: {str(e)}")
            raise AIVillageException(f"Error getting project tasks: {str(e)}")

    async def add_resources_to_project(self, project_id: str, resources: Dict[str, Any]):
        try:
            project = await self.get_project(project_id)
            project.resources.update(resources)
            logger.info(f"Added resources to project {project_id}")

            # Update resources in graph
            self.graph_manager.G.nodes[project_id]['resources'] = project.resources
            logger.info(f"Updated resources for project {project_id} in graph")
        except Exception as e:
            logger.exception(f"Error adding resources to project: {str(e)}")
            raise AIVillageException(f"Error adding resources to project: {str(e)}")

    def update_agent_list(self, agent_list: List[str]):
        try:
            self.available_agents = agent_list
            self.incentive_model.update_available_agents(agent_list)
            self.graph_manager.update_agent_list(agent_list)
            logger.info(f"Updated available agents: {self.available_agents}")
        except Exception as e:
            logger.exception(f"Error updating agent list: {str(e)}")
            raise AIVillageException(f"Error updating agent list: {str(e)}")

    async def process_task_batch(self):
        try:
            while len(self.pending_tasks) >= self.batch_size:
                batch = [self.pending_tasks.popleft() for _ in range(self.batch_size)]
                tasks_to_assign = self.graph_manager.get_tasks_for_distribution(batch)
                await asyncio.gather(*[self.assign_task(task) for task in tasks_to_assign])
                logger.info(f"Processed and assigned a batch of {self.batch_size} tasks")
        except Exception as e:
            logger.exception(f"Error processing task batch: {str(e)}")
            raise AIVillageException(f"Error processing task batch: {str(e)}")

    async def process_single_task(self, task: Task) -> Any:
        try:
            agent = task.assigned_agents[0]
            return await self.communication_protocol.send_and_wait(Message(
                type=MessageType.TASK,
                sender="UnifiedManagement",
                receiver=agent,
                content={"task_id": task.id, "description": task.description}
            ))
        except Exception as e:
            logger.exception(f"Error processing single task: {str(e)}")
            raise AIVillageException(f"Error processing single task: {str(e)}")

    async def start_batch_processing(self):
        try:
            while True:
                await self.process_task_batch()
                await asyncio.sleep(1)  # Adjust this delay as needed
        except Exception as e:
            logger.exception(f"Error in batch processing: {str(e)}")
            raise AIVillageException(f"Error in batch processing: {str(e)}")

    def set_batch_size(self, size: int):
        try:
            self.batch_size = size
            logger.info(f"Set batch size to {size}")
        except Exception as e:
            logger.exception(f"Error setting batch size: {str(e)}")
            raise AIVillageException(f"Error setting batch size: {str(e)}")

    async def get_task_status(self, task_id: str) -> TaskStatus:
        try:
            if task_id in self.ongoing_tasks:
                return self.ongoing_tasks[task_id].status
            elif any(task.id == task_id for task in self.completed_tasks):
                return TaskStatus.COMPLETED
            else:
                return TaskStatus.PENDING
        except Exception as e:
            logger.exception(f"Error getting task status: {str(e)}")
            raise AIVillageException(f"Error getting task status: {str(e)}")

    async def get_project_status(self, project_id: str) -> Dict[str, Any]:
        try:
            project = await self.get_project(project_id)
            tasks = [{"task_id": task.id, "status": task.status, "description": task.description} for task in project.tasks.values()]
            return {
                "project_id": project_id,
                "name": project.name,
                "status": project.status,
                "progress": project.progress,
                "tasks": tasks
            }
        except Exception as e:
            logger.exception(f"Error getting project status: {str(e)}")
            raise AIVillageException(f"Error getting project status: {str(e)}")

    async def save_state(self, filename: str):
        try:
            state = {
                'tasks': [task.__dict__ for task in self.pending_tasks] + \
                         [task.__dict__ for task in self.ongoing_tasks.values()] + \
                         [task.__dict__ for task in self.completed_tasks],
                'projects': {pid: project.__dict__ for pid, project in self.projects.items()},
                'agent_performance': self.agent_performance,
                'available_agents': self.available_agents,
                'graph': nx.node_link_data(self.graph_manager.G)  # Serialize graph
            }
            with open(filename, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Saved state to {filename}")
        except Exception as e:
            logger.exception(f"Error saving state: {str(e)}")
            raise AIVillageException(f"Error saving state: {str(e)}")

    async def load_state(self, filename: str):
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            self.pending_tasks = deque(Task(**task) for task in state['tasks'] if task['status'] == TaskStatus.PENDING.value)
            self.ongoing_tasks = {task['id']: Task(**task) for task in state['tasks'] if task['status'] == TaskStatus.IN_PROGRESS.value}
            self.completed_tasks = [Task(**task) for task in state['tasks'] if task['status'] == TaskStatus.COMPLETED.value]
            self.projects = {pid: Project(**project) for pid, project in state['projects'].items()}
            self.agent_performance = state['agent_performance']
            self.available_agents = state['available_agents']
            self.graph_manager.G = nx.node_link_graph(state['graph'])
            logger.info(f"Loaded state from {filename}")
        except Exception as e:
            logger.exception(f"Error loading state: {str(e)}")
            raise AIVillageException(f"Error loading state: {str(e)}")

    async def introspect(self) -> Dict[str, Any]:
        try:
            graph_info = {
                "nodes": self.graph_manager.G.number_of_nodes(),
                "edges": self.graph_manager.G.number_of_edges(),
                "agents": [n for n, d in self.graph_manager.G.nodes(data=True) if d['type'] == 'agent'],
                "tasks": [n for n, d in self.graph_manager.G.nodes(data=True) if d['type'] == 'task']
            }
            return {
                "pending_tasks": len(self.pending_tasks),
                "ongoing_tasks": len(self.ongoing_tasks),
                "completed_tasks": len(self.completed_tasks),
                "projects": len(self.projects),
                "available_agents": self.available_agents,
                "agent_performance": self.agent_performance,
                "batch_size": self.batch_size,
                "analytics_report": self.unified_analytics.generate_summary_report(),
                "graph_info": graph_info
            }
        except Exception as e:
            logger.exception(f"Error in introspection: {str(e)}")
            raise AIVillageException(f"Error in introspection: {str(e)}")

    # New Methods for Graph-based Approach

    async def _optimize_graph_structure(self):
        """
        Periodically optimize the agent-task graph structure for better task routing and agent utilization.
        """
        try:
            # Example optimization: Normalize edge weights
            for agent in self.available_agents:
                outgoing_edges = list(self.graph_manager.G.out_edges(agent, data=True))
                if outgoing_edges:
                    total_weight = sum(edge_data['weight'] for _, _, edge_data in outgoing_edges)
                    for _, task_id, edge_data in outgoing_edges:
                        edge_data['weight'] = edge_data['weight'] / total_weight if total_weight > 0 else 1.0
            logger.info("Optimized graph edge weights by normalizing outgoing weights for each agent")
            
            # Additional optimizations can be implemented here, such as removing obsolete edges or reinforcing beneficial connections.
            # For example, remove edges with weights below a certain threshold
            threshold = 0.1
            edges_to_remove = [(u, v) for u, v, d in self.graph_manager.G.edges(data=True) if d.get('weight', 1.0) < threshold]
            self.graph_manager.G.remove_edges_from(edges_to_remove)
            logger.info(f"Removed {len(edges_to_remove)} edges with weights below {threshold}")
            
            # Optionally, add new edges or adjust existing ones based on specific criteria
            # This can include clustering agents, reinforcing collaboration, etc.
        except Exception as e:
            logger.exception(f"Error optimizing graph structure: {str(e)}")
            raise AIVillageException(f"Error optimizing graph structure: {str(e)}")

    async def periodic_graph_optimization(self, interval: int = 3600):
        """
        Periodically optimize the graph structure at specified intervals.
        
        Args:
            interval (int): Time in seconds between optimizations. Default is 1 hour.
        """
        try:
            while True:
                await self._optimize_graph_structure()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Periodic graph optimization task cancelled")
        except Exception as e:
            logger.exception(f"Error in periodic graph optimization: {str(e)}")
            raise AIVillageException(f"Error in periodic graph optimization: {str(e)}")

    async def start_periodic_graph_optimization(self, interval: int = 3600):
        """
        Initiates the periodic graph optimization task.
        
        Args:
            interval (int): Time in seconds between optimizations. Default is 1 hour.
        """
        asyncio.create_task(self.periodic_graph_optimization(interval))
        logger.info(f"Started periodic graph optimization every {interval} seconds")

    async def complete_task(self, task_id: str, result: Any):
        try:
            if task_id not in self.ongoing_tasks:
                raise AIVillageException(f"Task {task_id} not found in ongoing tasks")
            task = self.ongoing_tasks[task_id]
            updated_task = task.update_status(TaskStatus.COMPLETED).update_result(result)
            self.completed_tasks.append(updated_task)
            del self.ongoing_tasks[task_id]

            # Update dependent tasks
            await self.update_dependent_tasks(updated_task)

            # Update graph based on task completion
            self.graph_manager.G.edges[task.assigned_agents[0], task_id]['status'] = 'completed'
            self.graph_manager.G.nodes[task_id]['result'] = result
            logger.info(f"Task {task_id} completed with result: {result}")

            agent = task.assigned_agents[0]
            self.incentive_model.update({'assigned_agent': agent, 'task_id': task_id}, result)
            self.update_agent_performance(agent, result)

            completion_time = (updated_task.completed_at - updated_task.created_at)
            success = result.get('success', False)
            self.unified_analytics.record_task_completion(task_id, completion_time, success)

            # Update project status if the task belongs to a project
            for project in self.projects.values():
                if task_id in project.tasks:
                    project.tasks[task_id] = updated_task
                    await self.update_project_status(project.id)
                    break

            # Propagate results through the graph
            await self._propagate_results(task_id, result)
        except Exception as e:
            logger.exception(f"Error completing task: {str(e)}")
            raise AIVillageException(f"Error completing task: {str(e)}")

    async def _propagate_results(self, task_id: str, result: Any):
        """
        Propagate task results through the graph, updating related node properties.
        """
        try:
            # Example: Update dependent tasks or agent nodes based on results
            successors = list(self.graph_manager.G.successors(task_id))
            for succ in successors:
                # Example: Update dependency factors or adjust incentives
                if self.graph_manager.G.has_edge(task_id, succ):
                    # Modify edge attributes based on result
                    dependency_factor = self.graph_manager.G[task_id][succ].get('dependency_factor', 1.0)
                    outcome_factor = 1 + (result.get('performance', 0.5) * 0.1)
                    self.graph_manager.G[task_id][succ]['dependency_factor'] = dependency_factor * outcome_factor
                    logger.info(f"Updated dependency factor for edge from {task_id} to {succ} to {self.graph_manager.G[task_id][succ]['dependency_factor']}")

                # Optionally, re-route or re-assign the successor task based on updated graph
                successor_task = self.completed_tasks[-1]  # Assuming the last completed task
                optimal_agent = await self._select_optimal_agent({'id': succ, 'description': self.graph_manager.G.nodes[succ].get('description', '')})
                if optimal_agent != self.graph_manager.G.edges[task_id, succ].get('assigned_agent', 'undecided'):
                    # Reassign task
                    await self.reassign_task(succ, optimal_agent)
        except Exception as e:
            logger.exception(f"Error propagating results: {str(e)}")
            raise AIVillageException(f"Error propagating results: {str(e)}")

    async def reassign_task(self, task_id: str, new_agent: str):
        """
        Reassign a task to a different agent based on updated incentives or dependencies.
        """
        try:
            # Remove previous assignment
            for agent in self.available_agents:
                if self.graph_manager.G.has_edge(agent, task_id):
                    self.graph_manager.G.remove_edge(agent, task_id)
                    logger.info(f"Removed edge from agent {agent} to task {task_id}")

            # Assign to new agent
            self.graph_manager.G.add_edge(new_agent, task_id, weight=self.incentive_model.calculate_incentive({'assigned_agent': new_agent, 'task_id': task_id}, self.agent_performance).get('incentive', 1.0))
            logger.info(f"Reassigned task {task_id} to agent {new_agent} with updated incentive")

            # Update task assignment in ongoing_tasks
            task = self.ongoing_tasks.get(task_id)
            if task:
                task.assigned_agents = [new_agent]
                await self.communication_protocol.send_message(Message(
                    type=MessageType.TASK_REASSIGNMENT,
                    sender="UnifiedManagement",
                    receiver=new_agent,
                    content={"task_id": task.id, "description": task.description},
                    priority=Priority.HIGH
                ))
                logger.info(f"Notified agent {new_agent} about task reassignment for task {task_id}")
        except Exception as e:
            logger.exception(f"Error reassigning task {task_id} to agent {new_agent}: {str(e)}")
            raise AIVillageException(f"Error reassigning task {task_id} to agent {new_agent}: {str(e)}")

    async def _propagate_results(self, task_id: str, result: Any):
        """
        Propagate task results through the graph, updating related node properties.
        """
        try:
            successors = list(self.graph_manager.G.successors(task_id))
            for succ in successors:
                # Example: Update dependency factors or adjust incentives
                if self.graph_manager.G.has_edge(task_id, succ):
                    # Modify edge attributes based on result
                    dependency_factor = self.graph_manager.G[task_id][succ].get('dependency_factor', 1.0)
                    outcome_factor = 1 + (result.get('performance', 0.5) * 0.1)
                    self.graph_manager.G[task_id][succ]['dependency_factor'] = dependency_factor * outcome_factor
                    logger.info(f"Updated dependency factor for edge from {task_id} to {succ} to {self.graph_manager.G[task_id][succ]['dependency_factor']}")

                # Optionally, re-route or re-assign the successor task based on updated graph
                successor_task = self.completed_tasks[-1]  # Assuming the last completed task
                optimal_agent = await self._select_optimal_agent({
                    'id': succ,
                    'description': self.graph_manager.G.nodes[succ].get('description', '')
                })
                if optimal_agent != self.graph_manager.G.edges[task_id, succ].get('assigned_agent', 'undecided'):
                    # Reassign task
                    await self.reassign_task(succ, optimal_agent)
        except Exception as e:
            logger.exception(f"Error propagating results: {str(e)}")
            raise AIVillageException(f"Error propagating results: {str(e)}")

    async def periodic_graph_optimization(self, interval: int = 3600):
        """
        Periodically optimize the agent-task graph structure for better task routing and agent utilization.
        
        Args:
            interval (int): Time in seconds between optimizations. Default is 1 hour.
        """
        try:
            while True:
                await self._optimize_graph_structure()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Periodic graph optimization task cancelled")
        except Exception as e:
            logger.exception(f"Error in periodic graph optimization: {str(e)}")
            raise AIVillageException(f"Error in periodic graph optimization: {str(e)}")

    async def start_periodic_graph_optimization(self, interval: int = 3600):
        """
        Initiates the periodic graph optimization task.
        
        Args:
            interval (int): Time in seconds between optimizations. Default is 1 hour.
        """
        asyncio.create_task(self.periodic_graph_optimization(interval))
        logger.info(f"Started periodic graph optimization every {interval} seconds")

    async def _optimize_graph_structure(self):
        """
        Optimize the agent-task graph structure for better task routing and agent utilization.
        """
        try:
            # Example optimization: Normalize edge weights
            for agent in self.available_agents:
                outgoing_edges = list(self.graph_manager.G.out_edges(agent, data=True))
                if outgoing_edges:
                    total_weight = sum(edge_data['weight'] for _, _, edge_data in outgoing_edges)
                    for _, task_id, edge_data in outgoing_edges:
                        edge_data['weight'] = edge_data['weight'] / total_weight if total_weight > 0 else 1.0
            logger.info("Optimized graph edge weights by normalizing outgoing weights for each agent")
            
            # Additional optimizations can be implemented here, such as removing obsolete edges or reinforcing beneficial connections.
            # For example, remove edges with weights below a certain threshold
            threshold = 0.1
            edges_to_remove = [(u, v) for u, v, d in self.graph_manager.G.edges(data=True) if d.get('weight', 1.0) < threshold]
            self.graph_manager.G.remove_edges_from(edges_to_remove)
            logger.info(f"Removed {len(edges_to_remove)} edges with weights below {threshold}")
            
            # Optionally, add new edges or adjust existing ones based on specific criteria
            # This can include clustering agents, reinforcing collaboration, etc.
        except Exception as e:
            logger.exception(f"Error optimizing graph structure: {str(e)}")
            raise AIVillageException(f"Error optimizing graph structure: {str(e)}")

    # Integrate Graph-based Approach: Implementing Step 6
