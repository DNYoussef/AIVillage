"""
Agent Synchronization Manager for Phase 6 Baking Pipeline

This module manages synchronization and coordination between the 9 specialized
baking agents, ensuring consistent state, workload distribution, and failure recovery.
"""

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
import uuid

from .data_flow_coordinator import DataFlowCoordinator, MessageType, ComponentStatus
from .state_manager import StateManager, Phase, StateStatus
from .serialization_utils import SafeJSONSerializer, SerializationConfig

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Types of baking agents in Phase 6"""
    BAKING_COORDINATOR = "baking_coordinator"
    MODEL_OPTIMIZER = "model_optimizer"
    INFERENCE_ACCELERATOR = "inference_accelerator"
    QUALITY_VALIDATOR = "quality_validator"
    PERFORMANCE_PROFILER = "performance_profiler"
    HARDWARE_ADAPTER = "hardware_adapter"
    GRAPH_OPTIMIZER = "graph_optimizer"
    MEMORY_OPTIMIZER = "memory_optimizer"
    DEPLOYMENT_PREPARER = "deployment_preparer"

class AgentState(Enum):
    """Agent state enumeration"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING_FOR_DEPENDENCY = "waiting_for_dependency"
    COMPLETED = "completed"
    ERROR = "error"
    OFFLINE = "offline"

class WorkloadPriority(Enum):
    """Workload priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class AgentInfo:
    """Information about a baking agent"""
    agent_id: str
    agent_type: AgentType
    state: AgentState
    current_workload: Optional[str]
    capabilities: Set[str]
    dependencies: Set[str]
    dependents: Set[str]
    last_heartbeat: datetime
    processing_start_time: Optional[datetime]
    error_count: int
    completed_tasks: int
    resource_usage: Dict[str, float]
    configuration: Dict[str, Any]

@dataclass
class WorkloadTask:
    """Workload task for agent processing"""
    task_id: str
    task_type: str
    priority: WorkloadPriority
    assigned_agent: Optional[str]
    dependencies: Set[str]
    data: Dict[str, Any]
    created_time: datetime
    started_time: Optional[datetime]
    completed_time: Optional[datetime]
    retry_count: int
    max_retries: int
    timeout_seconds: float
    checkpoint_data: Optional[Dict[str, Any]]

@dataclass
class SynchronizationPoint:
    """Synchronization point for agent coordination"""
    sync_id: str
    participating_agents: Set[str]
    arrived_agents: Set[str]
    sync_data: Dict[str, Any]
    timeout_time: datetime
    success: bool

class AgentSynchronizationManager:
    """
    Manager for synchronizing and coordinating Phase 6 baking agents.

    Provides:
    - Agent registration and lifecycle management
    - Workload distribution and task assignment
    - Dependency resolution and execution ordering
    - Synchronization points for multi-agent coordination
    - Error recovery and fault tolerance
    - Resource monitoring and load balancing
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.manager_id = str(uuid.uuid4())

        # Initialize data flow coordinator
        self.data_flow_coordinator = DataFlowCoordinator(config.get('data_flow_config', {}))

        # Agent management
        self.agents: Dict[str, AgentInfo] = {}
        self.agent_lock = threading.RLock()

        # Workload management
        self.pending_tasks: deque = deque()
        self.active_tasks: Dict[str, WorkloadTask] = {}
        self.completed_tasks: Dict[str, WorkloadTask] = {}
        self.task_lock = threading.RLock()

        # Synchronization management
        self.sync_points: Dict[str, SynchronizationPoint] = {}
        self.sync_lock = threading.RLock()

        # Dependency graph
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependency_graph: Dict[str, Set[str]] = defaultdict(set)

        # Configuration
        self.heartbeat_timeout = config.get('heartbeat_timeout_seconds', 30)
        self.task_timeout = config.get('task_timeout_seconds', 300)
        self.max_retries = config.get('max_retries', 3)
        self.sync_timeout = config.get('sync_timeout_seconds', 60)

        # Background task management
        self.running = False
        self.background_tasks = []

        # Serialization
        self.serializer = SafeJSONSerializer(SerializationConfig())

        logger.info(f"AgentSynchronizationManager initialized with ID: {self.manager_id}")

    async def start(self):
        """Start the agent synchronization manager"""
        if self.running:
            logger.warning("AgentSynchronizationManager already running")
            return

        self.running = True
        logger.info("Starting AgentSynchronizationManager...")

        # Start data flow coordinator
        await self.data_flow_coordinator.start()

        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._agent_monitor()),
            asyncio.create_task(self._workload_scheduler()),
            asyncio.create_task(self._synchronization_manager()),
            asyncio.create_task(self._dependency_resolver()),
            asyncio.create_task(self._health_checker())
        ]

        logger.info("AgentSynchronizationManager started successfully")

    async def stop(self):
        """Stop the agent synchronization manager"""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping AgentSynchronizationManager...")

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()

        # Stop data flow coordinator
        await self.data_flow_coordinator.stop()

        logger.info("AgentSynchronizationManager stopped")

    def register_agent(self, agent_id: str, agent_type: AgentType,
                      capabilities: Optional[Set[str]] = None,
                      dependencies: Optional[Set[str]] = None,
                      configuration: Optional[Dict[str, Any]] = None) -> bool:
        """Register a baking agent"""
        try:
            with self.agent_lock:
                if agent_id in self.agents:
                    logger.warning(f"Agent {agent_id} already registered")
                    return False

                # Create agent info
                agent_info = AgentInfo(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    state=AgentState.INITIALIZING,
                    current_workload=None,
                    capabilities=capabilities or set(),
                    dependencies=dependencies or set(),
                    dependents=set(),
                    last_heartbeat=datetime.now(),
                    processing_start_time=None,
                    error_count=0,
                    completed_tasks=0,
                    resource_usage={},
                    configuration=configuration or {}
                )

                self.agents[agent_id] = agent_info

                # Update dependency graphs
                for dep in agent_info.dependencies:
                    self.dependency_graph[agent_id].add(dep)
                    self.reverse_dependency_graph[dep].add(agent_id)
                    # Update dependent's dependents
                    if dep in self.agents:
                        self.agents[dep].dependents.add(agent_id)

                # Register with data flow coordinator
                self.data_flow_coordinator.register_component(
                    agent_id, agent_type.value, self._agent_message_handler
                )

                logger.info(f"Registered agent: {agent_id} ({agent_type.value})")
                return True

        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return False

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister a baking agent"""
        try:
            with self.agent_lock:
                if agent_id not in self.agents:
                    logger.warning(f"Agent {agent_id} not found")
                    return False

                agent_info = self.agents[agent_id]

                # Update dependency graphs
                for dep in agent_info.dependencies:
                    self.dependency_graph[agent_id].discard(dep)
                    self.reverse_dependency_graph[dep].discard(agent_id)
                    if dep in self.agents:
                        self.agents[dep].dependents.discard(agent_id)

                for dependent in agent_info.dependents:
                    if dependent in self.agents:
                        self.agents[dependent].dependencies.discard(agent_id)
                    self.dependency_graph[dependent].discard(agent_id)

                # Remove from active tasks
                if agent_info.current_workload:
                    await self._handle_agent_failure(agent_id, "Agent unregistered")

                del self.agents[agent_id]

                # Unregister from data flow coordinator
                self.data_flow_coordinator.unregister_component(agent_id)

                logger.info(f"Unregistered agent: {agent_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False

    async def submit_task(self, task_type: str, data: Dict[str, Any],
                         priority: WorkloadPriority = WorkloadPriority.NORMAL,
                         dependencies: Optional[Set[str]] = None,
                         timeout_seconds: Optional[float] = None) -> str:
        """Submit a task for processing"""
        task_id = str(uuid.uuid4())

        task = WorkloadTask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            assigned_agent=None,
            dependencies=dependencies or set(),
            data=data,
            created_time=datetime.now(),
            started_time=None,
            completed_time=None,
            retry_count=0,
            max_retries=self.max_retries,
            timeout_seconds=timeout_seconds or self.task_timeout,
            checkpoint_data=None
        )

        with self.task_lock:
            self.pending_tasks.append(task)

        logger.info(f"Submitted task {task_id} of type {task_type}")
        return task_id

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        with self.task_lock:
            # Check active tasks
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                return {
                    'task_id': task_id,
                    'status': 'active',
                    'assigned_agent': task.assigned_agent,
                    'progress': self._calculate_task_progress(task),
                    'started_time': task.started_time.isoformat() if task.started_time else None,
                    'retry_count': task.retry_count
                }

            # Check completed tasks
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                return {
                    'task_id': task_id,
                    'status': 'completed',
                    'assigned_agent': task.assigned_agent,
                    'completed_time': task.completed_time.isoformat() if task.completed_time else None,
                    'retry_count': task.retry_count
                }

            # Check pending tasks
            for task in self.pending_tasks:
                if task.task_id == task_id:
                    return {
                        'task_id': task_id,
                        'status': 'pending',
                        'created_time': task.created_time.isoformat(),
                        'priority': task.priority.value
                    }

        return None

    async def create_synchronization_point(self, sync_id: str,
                                         participating_agents: Set[str],
                                         sync_data: Optional[Dict[str, Any]] = None,
                                         timeout_seconds: Optional[float] = None) -> bool:
        """Create a synchronization point for multiple agents"""
        try:
            with self.sync_lock:
                if sync_id in self.sync_points:
                    logger.warning(f"Synchronization point {sync_id} already exists")
                    return False

                sync_point = SynchronizationPoint(
                    sync_id=sync_id,
                    participating_agents=participating_agents.copy(),
                    arrived_agents=set(),
                    sync_data=sync_data or {},
                    timeout_time=datetime.now() + timedelta(seconds=timeout_seconds or self.sync_timeout),
                    success=False
                )

                self.sync_points[sync_id] = sync_point

                # Notify participating agents
                await self._notify_agents_of_sync_point(sync_id, participating_agents)

                logger.info(f"Created synchronization point {sync_id} for agents: {participating_agents}")
                return True

        except Exception as e:
            logger.error(f"Failed to create synchronization point {sync_id}: {e}")
            return False

    async def agent_arrive_at_sync_point(self, agent_id: str, sync_id: str,
                                        agent_data: Optional[Dict[str, Any]] = None) -> bool:
        """Mark an agent as arrived at a synchronization point"""
        try:
            with self.sync_lock:
                if sync_id not in self.sync_points:
                    logger.error(f"Synchronization point {sync_id} not found")
                    return False

                sync_point = self.sync_points[sync_id]

                if agent_id not in sync_point.participating_agents:
                    logger.error(f"Agent {agent_id} not registered for sync point {sync_id}")
                    return False

                if agent_id in sync_point.arrived_agents:
                    logger.warning(f"Agent {agent_id} already arrived at sync point {sync_id}")
                    return True

                # Mark agent as arrived
                sync_point.arrived_agents.add(agent_id)

                # Store agent data
                if agent_data:
                    sync_point.sync_data[agent_id] = agent_data

                logger.info(f"Agent {agent_id} arrived at sync point {sync_id} "
                           f"({len(sync_point.arrived_agents)}/{len(sync_point.participating_agents)})")

                # Check if all agents have arrived
                if len(sync_point.arrived_agents) == len(sync_point.participating_agents):
                    sync_point.success = True
                    await self._complete_synchronization_point(sync_id)

                return True

        except Exception as e:
            logger.error(f"Error processing agent arrival at sync point: {e}")
            return False

    async def wait_for_synchronization(self, sync_id: str, timeout_seconds: Optional[float] = None) -> bool:
        """Wait for a synchronization point to complete"""
        timeout_time = datetime.now() + timedelta(seconds=timeout_seconds or self.sync_timeout)

        while datetime.now() < timeout_time:
            with self.sync_lock:
                if sync_id not in self.sync_points:
                    return False

                sync_point = self.sync_points[sync_id]
                if sync_point.success:
                    return True

            await asyncio.sleep(0.1)

        logger.warning(f"Synchronization point {sync_id} timed out")
        return False

    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific agent"""
        with self.agent_lock:
            if agent_id not in self.agents:
                return None

            agent = self.agents[agent_id]
            return {
                'agent_id': agent_id,
                'agent_type': agent.agent_type.value,
                'state': agent.state.value,
                'current_workload': agent.current_workload,
                'capabilities': list(agent.capabilities),
                'dependencies': list(agent.dependencies),
                'dependents': list(agent.dependents),
                'last_heartbeat': agent.last_heartbeat.isoformat(),
                'error_count': agent.error_count,
                'completed_tasks': agent.completed_tasks,
                'resource_usage': agent.resource_usage,
                'is_healthy': self._is_agent_healthy(agent)
            }

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        with self.agent_lock, self.task_lock, self.sync_lock:
            agent_statuses = {}
            healthy_agents = 0
            processing_agents = 0

            for agent_id, agent in self.agents.items():
                is_healthy = self._is_agent_healthy(agent)
                agent_statuses[agent_id] = {
                    'type': agent.agent_type.value,
                    'state': agent.state.value,
                    'healthy': is_healthy,
                    'current_workload': agent.current_workload,
                    'error_count': agent.error_count
                }

                if is_healthy:
                    healthy_agents += 1
                if agent.state == AgentState.PROCESSING:
                    processing_agents += 1

            return {
                'manager_id': self.manager_id,
                'total_agents': len(self.agents),
                'healthy_agents': healthy_agents,
                'processing_agents': processing_agents,
                'system_health': 'HEALTHY' if healthy_agents == len(self.agents) else 'DEGRADED',
                'pending_tasks': len(self.pending_tasks),
                'active_tasks': len(self.active_tasks),
                'completed_tasks': len(self.completed_tasks),
                'active_sync_points': len(self.sync_points),
                'agents': agent_statuses
            }

    async def checkpoint_system_state(self, checkpoint_name: str) -> bool:
        """Create a checkpoint of the entire system state"""
        try:
            checkpoint_data = {
                'timestamp': datetime.now().isoformat(),
                'manager_id': self.manager_id,
                'agents': {},
                'pending_tasks': [],
                'active_tasks': {},
                'sync_points': {}
            }

            with self.agent_lock:
                for agent_id, agent in self.agents.items():
                    checkpoint_data['agents'][agent_id] = {
                        'agent_type': agent.agent_type.value,
                        'state': agent.state.value,
                        'current_workload': agent.current_workload,
                        'capabilities': list(agent.capabilities),
                        'dependencies': list(agent.dependencies),
                        'error_count': agent.error_count,
                        'completed_tasks': agent.completed_tasks,
                        'configuration': agent.configuration
                    }

            with self.task_lock:
                for task in self.pending_tasks:
                    checkpoint_data['pending_tasks'].append({
                        'task_id': task.task_id,
                        'task_type': task.task_type,
                        'priority': task.priority.value,
                        'dependencies': list(task.dependencies),
                        'data': task.data,
                        'created_time': task.created_time.isoformat(),
                        'retry_count': task.retry_count
                    })

                for task_id, task in self.active_tasks.items():
                    checkpoint_data['active_tasks'][task_id] = {
                        'task_type': task.task_type,
                        'assigned_agent': task.assigned_agent,
                        'started_time': task.started_time.isoformat() if task.started_time else None,
                        'retry_count': task.retry_count,
                        'checkpoint_data': task.checkpoint_data
                    }

            with self.sync_lock:
                for sync_id, sync_point in self.sync_points.items():
                    checkpoint_data['sync_points'][sync_id] = {
                        'participating_agents': list(sync_point.participating_agents),
                        'arrived_agents': list(sync_point.arrived_agents),
                        'sync_data': sync_point.sync_data,
                        'timeout_time': sync_point.timeout_time.isoformat(),
                        'success': sync_point.success
                    }

            # Save checkpoint through data flow coordinator
            success = await self.data_flow_coordinator.create_checkpoint(checkpoint_name)

            if success:
                # Save agent synchronization specific data
                checkpoint_file = Path(self.data_flow_coordinator.state_manager.storage_dir) / 'checkpoints' / checkpoint_name / 'agent_sync_state.json'
                self.serializer.serialize_to_file(checkpoint_data, checkpoint_file)

            return success

        except Exception as e:
            logger.error(f"Failed to create checkpoint {checkpoint_name}: {e}")
            return False

    # Private methods

    def _agent_message_handler(self, message):
        """Handle messages from agents"""
        try:
            agent_id = message.source_component
            message_type = message.message_type

            if message_type == MessageType.HEALTH_CHECK:
                self._handle_agent_heartbeat(agent_id)
            elif message_type == MessageType.STATE_UPDATE:
                self._handle_agent_state_update(agent_id, message.data)
            elif message_type == MessageType.ERROR_NOTIFICATION:
                self._handle_agent_error(agent_id, message.data)
            elif message_type == MessageType.SYNCHRONIZATION:
                self._handle_agent_sync_message(agent_id, message.data)

        except Exception as e:
            logger.error(f"Error handling agent message: {e}")

    def _handle_agent_heartbeat(self, agent_id: str):
        """Handle agent heartbeat"""
        with self.agent_lock:
            if agent_id in self.agents:
                self.agents[agent_id].last_heartbeat = datetime.now()

    def _handle_agent_state_update(self, agent_id: str, data: Dict[str, Any]):
        """Handle agent state update"""
        with self.agent_lock:
            if agent_id in self.agents:
                agent = self.agents[agent_id]

                if 'state' in data:
                    agent.state = AgentState(data['state'])

                if 'resource_usage' in data:
                    agent.resource_usage = data['resource_usage']

                if 'current_workload' in data:
                    agent.current_workload = data['current_workload']

    def _handle_agent_error(self, agent_id: str, error_data: Dict[str, Any]):
        """Handle agent error notification"""
        with self.agent_lock:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                agent.error_count += 1
                agent.state = AgentState.ERROR

                logger.error(f"Agent {agent_id} reported error: {error_data}")

                # Handle failure recovery
                asyncio.create_task(self._handle_agent_failure(agent_id, error_data.get('error', 'Unknown error')))

    def _handle_agent_sync_message(self, agent_id: str, sync_data: Dict[str, Any]):
        """Handle agent synchronization message"""
        if 'sync_id' in sync_data:
            sync_id = sync_data['sync_id']
            agent_data = sync_data.get('data', {})
            asyncio.create_task(
                self.agent_arrive_at_sync_point(agent_id, sync_id, agent_data)
            )

    async def _handle_agent_failure(self, agent_id: str, reason: str):
        """Handle agent failure and recovery"""
        try:
            with self.agent_lock, self.task_lock:
                if agent_id not in self.agents:
                    return

                agent = self.agents[agent_id]

                # If agent has current workload, handle task failure
                if agent.current_workload:
                    task_id = agent.current_workload
                    if task_id in self.active_tasks:
                        task = self.active_tasks[task_id]

                        # Increment retry count
                        task.retry_count += 1
                        task.assigned_agent = None

                        if task.retry_count <= task.max_retries:
                            # Requeue task for retry
                            self.pending_tasks.appendleft(task)
                            del self.active_tasks[task_id]
                            logger.info(f"Requeued task {task_id} for retry ({task.retry_count}/{task.max_retries})")
                        else:
                            # Task failed permanently
                            del self.active_tasks[task_id]
                            logger.error(f"Task {task_id} failed permanently after {task.max_retries} retries")

                agent.current_workload = None
                agent.state = AgentState.ERROR

            logger.warning(f"Agent {agent_id} failed: {reason}")

        except Exception as e:
            logger.error(f"Error handling agent failure: {e}")

    def _is_agent_healthy(self, agent: AgentInfo) -> bool:
        """Check if an agent is healthy"""
        # Check heartbeat age
        heartbeat_age = (datetime.now() - agent.last_heartbeat).total_seconds()
        if heartbeat_age > self.heartbeat_timeout:
            return False

        # Check error rate
        if agent.completed_tasks > 0:
            error_rate = agent.error_count / (agent.completed_tasks + agent.error_count)
            if error_rate > 0.2:  # 20% error rate threshold
                return False

        # Check state
        if agent.state in [AgentState.ERROR, AgentState.OFFLINE]:
            return False

        return True

    def _calculate_task_progress(self, task: WorkloadTask) -> float:
        """Calculate task progress (0-100%)"""
        if not task.started_time:
            return 0.0

        elapsed = (datetime.now() - task.started_time).total_seconds()
        if elapsed >= task.timeout_seconds:
            return 100.0

        # Estimate progress based on time
        return min(100.0, (elapsed / task.timeout_seconds) * 100.0)

    async def _notify_agents_of_sync_point(self, sync_id: str, participating_agents: Set[str]):
        """Notify agents of a new synchronization point"""
        for agent_id in participating_agents:
            await self.data_flow_coordinator.send_data(
                source_component=self.manager_id,
                target_component=agent_id,
                data={
                    'sync_id': sync_id,
                    'message': 'synchronization_point_created'
                },
                message_type=MessageType.SYNCHRONIZATION
            )

    async def _complete_synchronization_point(self, sync_id: str):
        """Complete a synchronization point"""
        with self.sync_lock:
            if sync_id not in self.sync_points:
                return

            sync_point = self.sync_points[sync_id]

            # Notify all participating agents of completion
            for agent_id in sync_point.participating_agents:
                await self.data_flow_coordinator.send_data(
                    source_component=self.manager_id,
                    target_component=agent_id,
                    data={
                        'sync_id': sync_id,
                        'message': 'synchronization_completed',
                        'sync_data': sync_point.sync_data
                    },
                    message_type=MessageType.SYNCHRONIZATION
                )

            logger.info(f"Synchronization point {sync_id} completed successfully")

    # Background tasks

    async def _agent_monitor(self):
        """Monitor agent health and status"""
        while self.running:
            try:
                current_time = datetime.now()

                with self.agent_lock:
                    for agent_id, agent in self.agents.items():
                        # Check heartbeat timeout
                        heartbeat_age = (current_time - agent.last_heartbeat).total_seconds()

                        if heartbeat_age > self.heartbeat_timeout:
                            if agent.state != AgentState.OFFLINE:
                                logger.warning(f"Agent {agent_id} appears offline (no heartbeat for {heartbeat_age:.1f}s)")
                                agent.state = AgentState.OFFLINE

                                # Handle agent going offline
                                await self._handle_agent_failure(agent_id, "Heartbeat timeout")

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Error in agent monitor: {e}")
                await asyncio.sleep(10)

    async def _workload_scheduler(self):
        """Schedule workload tasks to available agents"""
        while self.running:
            try:
                with self.task_lock:
                    if not self.pending_tasks:
                        await asyncio.sleep(1)
                        continue

                    # Get next task by priority
                    task = self._get_next_task()
                    if not task:
                        await asyncio.sleep(1)
                        continue

                    # Find available agent for task
                    agent_id = self._find_available_agent(task)
                    if not agent_id:
                        await asyncio.sleep(1)
                        continue

                    # Assign task to agent
                    await self._assign_task_to_agent(task, agent_id)

            except Exception as e:
                logger.error(f"Error in workload scheduler: {e}")
                await asyncio.sleep(1)

    def _get_next_task(self) -> Optional[WorkloadTask]:
        """Get the next task to schedule"""
        if not self.pending_tasks:
            return None

        # Sort by priority (lower number = higher priority)
        sorted_tasks = sorted(self.pending_tasks, key=lambda t: (t.priority.value, t.created_time))

        for task in sorted_tasks:
            # Check if task dependencies are satisfied
            if self._are_task_dependencies_satisfied(task):
                self.pending_tasks.remove(task)
                return task

        return None

    def _are_task_dependencies_satisfied(self, task: WorkloadTask) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_task_id in task.dependencies:
            if dep_task_id not in self.completed_tasks:
                return False
        return True

    def _find_available_agent(self, task: WorkloadTask) -> Optional[str]:
        """Find an available agent for the task"""
        with self.agent_lock:
            available_agents = []

            for agent_id, agent in self.agents.items():
                # Check if agent is available
                if (agent.state == AgentState.IDLE and
                    agent.current_workload is None and
                    self._is_agent_healthy(agent) and
                    self._can_agent_handle_task(agent, task)):
                    available_agents.append((agent_id, agent))

            if not available_agents:
                return None

            # Select best agent (lowest error count, highest completed tasks)
            best_agent = min(available_agents,
                           key=lambda x: (x[1].error_count, -x[1].completed_tasks))

            return best_agent[0]

    def _can_agent_handle_task(self, agent: AgentInfo, task: WorkloadTask) -> bool:
        """Check if agent can handle the task"""
        # Check if agent has required capabilities
        required_capabilities = task.data.get('required_capabilities', set())
        if required_capabilities and not required_capabilities.issubset(agent.capabilities):
            return False

        return True

    async def _assign_task_to_agent(self, task: WorkloadTask, agent_id: str):
        """Assign task to agent"""
        try:
            with self.agent_lock, self.task_lock:
                if agent_id not in self.agents:
                    return False

                agent = self.agents[agent_id]

                # Update task
                task.assigned_agent = agent_id
                task.started_time = datetime.now()

                # Update agent
                agent.current_workload = task.task_id
                agent.state = AgentState.PROCESSING
                agent.processing_start_time = datetime.now()

                # Move to active tasks
                self.active_tasks[task.task_id] = task

            # Send task to agent
            await self.data_flow_coordinator.send_data(
                source_component=self.manager_id,
                target_component=agent_id,
                data={
                    'task_id': task.task_id,
                    'task_type': task.task_type,
                    'task_data': task.data,
                    'timeout_seconds': task.timeout_seconds
                },
                message_type=MessageType.DATA_TRANSFER
            )

            logger.info(f"Assigned task {task.task_id} to agent {agent_id}")
            return True

        except Exception as e:
            logger.error(f"Error assigning task to agent: {e}")
            return False

    async def _synchronization_manager(self):
        """Manage synchronization points"""
        while self.running:
            try:
                current_time = datetime.now()

                with self.sync_lock:
                    expired_sync_points = []

                    for sync_id, sync_point in self.sync_points.items():
                        if current_time > sync_point.timeout_time and not sync_point.success:
                            expired_sync_points.append(sync_id)

                    # Clean up expired sync points
                    for sync_id in expired_sync_points:
                        logger.warning(f"Synchronization point {sync_id} expired")
                        del self.sync_points[sync_id]

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error in synchronization manager: {e}")
                await asyncio.sleep(5)

    async def _dependency_resolver(self):
        """Resolve agent dependencies and manage execution order"""
        while self.running:
            try:
                # This would implement dependency resolution logic
                # For now, just sleep
                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Error in dependency resolver: {e}")
                await asyncio.sleep(10)

    async def _health_checker(self):
        """Perform system health checks"""
        while self.running:
            try:
                # Check overall system health
                status = self.get_system_status()

                if status['healthy_agents'] < status['total_agents']:
                    logger.warning(f"System health degraded: {status['healthy_agents']}/{status['total_agents']} agents healthy")

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in health checker: {e}")
                await asyncio.sleep(30)

# Factory function
def create_agent_synchronization_manager(config: Dict[str, Any]) -> AgentSynchronizationManager:
    """Factory function to create agent synchronization manager"""
    return AgentSynchronizationManager(config)

# Testing utilities
async def test_agent_synchronization():
    """Test agent synchronization functionality"""
    config = {
        'heartbeat_timeout_seconds': 30,
        'task_timeout_seconds': 300,
        'max_retries': 3,
        'sync_timeout_seconds': 60
    }

    manager = AgentSynchronizationManager(config)

    try:
        await manager.start()

        # Register test agents
        manager.register_agent("baking_coordinator", AgentType.BAKING_COORDINATOR, {"coordination"})
        manager.register_agent("model_optimizer", AgentType.MODEL_OPTIMIZER, {"optimization"})
        manager.register_agent("quality_validator", AgentType.QUALITY_VALIDATOR, {"validation"})

        # Submit test task
        task_id = await manager.submit_task(
            "optimization_task",
            {"model_data": "test_model", "optimization_level": 3},
            WorkloadPriority.HIGH
        )

        print(f"Submitted task: {task_id}")

        # Check system status
        status = manager.get_system_status()
        print(f"System status: {status}")

        await asyncio.sleep(2)  # Let system process

    finally:
        await manager.stop()

if __name__ == "__main__":
    asyncio.run(test_agent_synchronization())