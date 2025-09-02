"""
MCP Agent Coordination Hub - Memory-Centric Multi-Agent Coordination

This module provides comprehensive agent coordination using MCP Memory server as the central hub.
Designed for CI/CD pipelines where agents need to collaborate efficiently with persistent state.

Key Features:
- Memory-centric coordination architecture
- Agent lifecycle management with persistent state
- Task distribution and result aggregation
- Inter-agent communication protocols
- Conflict resolution and synchronization
- Performance monitoring and optimization
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union
from uuid import uuid4

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """States in agent lifecycle"""
    INITIALIZING = "initializing"
    READY = "ready"
    WORKING = "working"
    WAITING = "waiting"
    BLOCKED = "blocked"
    ERROR = "error"
    COMPLETED = "completed"
    TERMINATED = "terminated"


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class CoordinationStrategy(Enum):
    """Agent coordination strategies"""
    SEQUENTIAL = "sequential"  # Agents work one after another
    PARALLEL = "parallel"     # Agents work simultaneously
    PIPELINE = "pipeline"     # Output of one agent feeds next
    HIERARCHICAL = "hierarchical"  # Master-worker pattern
    PEER_TO_PEER = "peer_to_peer"  # Agents coordinate directly
    HYBRID = "hybrid"         # Mix of strategies based on context


@dataclass
class AgentCapability:
    """Defines what an agent can do"""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    estimated_duration_seconds: float = 60.0
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class Task:
    """Task to be executed by agents"""
    task_id: str = field(default_factory=lambda: f"task_{uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    
    # Task content
    input_data: Dict[str, Any] = field(default_factory=dict)
    required_capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Other task IDs
    
    # Execution context
    assigned_agent: Optional[str] = None
    state: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_info: Optional[str] = None
    execution_log: List[str] = field(default_factory=list)
    
    # Coordination
    blocks_tasks: List[str] = field(default_factory=list)  # Tasks that depend on this
    coordination_strategy: CoordinationStrategy = CoordinationStrategy.PARALLEL


@dataclass
class Agent:
    """Agent definition and state"""
    agent_id: str = field(default_factory=lambda: f"agent_{uuid4().hex[:8]}")
    agent_type: str = "generic"
    name: str = ""
    description: str = ""
    
    # Capabilities
    capabilities: List[AgentCapability] = field(default_factory=list)
    max_concurrent_tasks: int = 1
    
    # State
    state: AgentState = AgentState.INITIALIZING
    current_tasks: Set[str] = field(default_factory=set)
    completed_tasks: List[str] = field(default_factory=list)
    
    # Performance
    total_tasks_completed: int = 0
    total_execution_time_seconds: float = 0.0
    success_rate: float = 1.0
    last_active: datetime = field(default_factory=datetime.now)
    
    # Coordination
    memory_namespace: str = field(default_factory=lambda: f"agent_{uuid4().hex[:8]}")
    communication_channels: Set[str] = field(default_factory=set)
    blocked_by: Set[str] = field(default_factory=set)  # Agent IDs blocking this agent
    
    @property
    def avg_execution_time(self) -> float:
        return self.total_execution_time_seconds / max(self.total_tasks_completed, 1)
    
    @property
    def is_available(self) -> bool:
        return (
            self.state in [AgentState.READY, AgentState.WAITING] and
            len(self.current_tasks) < self.max_concurrent_tasks
        )


class MCPAgentCoordinationHub:
    """
    MCP Agent Coordination Hub - Memory-Centric Multi-Agent Coordination System
    
    Uses MCP Memory server as the central coordination point for multi-agent workflows.
    Provides comprehensive agent lifecycle management, task distribution, and result aggregation.
    
    Architecture:
    - MCP Memory Server: Central state storage and coordination
    - Agent Registry: Track all active agents and capabilities
    - Task Queue: Distributed task management with priorities
    - Communication System: Inter-agent messaging via memory
    - Conflict Resolution: Handle resource conflicts and dependencies
    - Performance Monitoring: Track and optimize coordination efficiency
    """
    
    def __init__(self, memory_client=None, session_id: str = None):
        self.session_id = session_id or f"coordination_{int(time.time())}"
        self.memory_client = memory_client  # MCP Memory client
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Agent management
        self.agents: Dict[str, Agent] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}  # capability -> agent_ids
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[str] = []  # Task IDs in priority order
        self.completed_tasks: Set[str] = set()
        
        # Coordination state
        self.coordination_strategy = CoordinationStrategy.HYBRID
        self.max_parallel_agents = 10
        self.task_timeout_seconds = 300  # 5 minutes default
        
        # Communication
        self.message_queue: List[Dict[str, Any]] = []
        self.broadcast_channels: Dict[str, Set[str]] = {}  # channel -> agent_ids
        
        # Monitoring
        self.coordination_start_time = datetime.now()
        self.stats = {
            "agents_registered": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "coordination_cycles": 0,
            "total_coordination_time": 0.0,
            "conflicts_resolved": 0,
            "messages_sent": 0
        }
        
        # Background tasks
        self.coordination_task: Optional[asyncio.Task] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        
        self.logger.info(f"MCP Agent Coordination Hub initialized for session {self.session_id}")
    
    async def initialize(self) -> bool:
        """Initialize the coordination hub"""
        try:
            # Initialize memory storage structure
            await self._initialize_memory_structure()
            
            # Start coordination and monitoring loops
            self.coordination_task = asyncio.create_task(self._coordination_loop())
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.logger.info("Agent Coordination Hub initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize coordination hub: {e}")
            return False
    
    async def _initialize_memory_structure(self):
        """Initialize memory structure for coordination"""
        if not self.memory_client:
            self.logger.warning("No memory client available - using local storage")
            self.local_memory = {}
            return
        
        # Create namespaces for different coordination aspects
        namespaces = [
            f"{self.session_id}_agents",
            f"{self.session_id}_tasks", 
            f"{self.session_id}_coordination",
            f"{self.session_id}_messages",
            f"{self.session_id}_state"
        ]
        
        for namespace in namespaces:
            try:
                if hasattr(self.memory_client, 'create_namespace'):
                    await self.memory_client.create_namespace(namespace, f"Coordination namespace: {namespace}")
            except Exception as e:
                self.logger.debug(f"Namespace creation note: {e}")
    
    async def register_agent(self, agent: Agent) -> bool:
        """Register a new agent with the coordination hub"""
        try:
            # Validate agent
            if not agent.capabilities:
                self.logger.warning(f"Agent {agent.agent_id} has no capabilities defined")
            
            # Store agent in memory
            agent_data = {
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "name": agent.name,
                "description": agent.description,
                "capabilities": [
                    {
                        "name": cap.name,
                        "description": cap.description,
                        "input_types": cap.input_types,
                        "output_types": cap.output_types,
                        "estimated_duration_seconds": cap.estimated_duration_seconds
                    } for cap in agent.capabilities
                ],
                "max_concurrent_tasks": agent.max_concurrent_tasks,
                "state": agent.state.value,
                "memory_namespace": agent.memory_namespace,
                "registered_at": datetime.now().isoformat()
            }
            
            await self._store_in_memory(
                f"agent_{agent.agent_id}",
                agent_data,
                f"{self.session_id}_agents"
            )
            
            # Update local registry
            self.agents[agent.agent_id] = agent
            agent.state = AgentState.READY
            
            # Index capabilities
            for capability in agent.capabilities:
                if capability.name not in self.agent_capabilities:
                    self.agent_capabilities[capability.name] = []
                self.agent_capabilities[capability.name].append(agent.agent_id)
            
            self.stats["agents_registered"] += 1
            self.logger.info(f"Registered agent: {agent.name} ({agent.agent_id}) with {len(agent.capabilities)} capabilities")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent.agent_id}: {e}")
            return False
    
    async def submit_task(self, task: Task) -> bool:
        """Submit a new task for execution"""
        try:
            # Validate task
            if not task.required_capabilities:
                self.logger.warning(f"Task {task.task_id} has no required capabilities")
            
            # Check if we have agents with required capabilities
            available_agents = self._find_capable_agents(task.required_capabilities)
            if not available_agents:
                self.logger.error(f"No agents available for task {task.task_id} with capabilities: {task.required_capabilities}")
                return False
            
            # Store task in memory
            task_data = {
                "task_id": task.task_id,
                "name": task.name,
                "description": task.description,
                "priority": task.priority.value,
                "input_data": task.input_data,
                "required_capabilities": task.required_capabilities,
                "dependencies": task.dependencies,
                "state": task.state,
                "created_at": task.created_at.isoformat(),
                "coordination_strategy": task.coordination_strategy.value
            }
            
            await self._store_in_memory(
                f"task_{task.task_id}",
                task_data,
                f"{self.session_id}_tasks"
            )
            
            # Add to local registry and queue
            self.tasks[task.task_id] = task
            self._insert_task_in_priority_order(task.task_id)
            
            self.logger.info(f"Submitted task: {task.name} ({task.task_id}) with priority {task.priority.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to submit task {task.task_id}: {e}")
            return False
    
    def _find_capable_agents(self, required_capabilities: List[str]) -> List[str]:
        """Find agents that have the required capabilities"""
        capable_agents = []
        
        for capability in required_capabilities:
            agents_with_capability = self.agent_capabilities.get(capability, [])
            for agent_id in agents_with_capability:
                agent = self.agents.get(agent_id)
                if agent and agent.is_available:
                    capable_agents.append(agent_id)
        
        return list(set(capable_agents))  # Remove duplicates
    
    def _insert_task_in_priority_order(self, task_id: str):
        """Insert task in queue based on priority"""
        task = self.tasks[task_id]
        
        # Find insertion point based on priority
        insert_index = 0
        for i, existing_task_id in enumerate(self.task_queue):
            existing_task = self.tasks[existing_task_id]
            if task.priority.value < existing_task.priority.value:  # Lower number = higher priority
                break
            insert_index = i + 1
        
        self.task_queue.insert(insert_index, task_id)
    
    async def _coordination_loop(self):
        """Main coordination loop"""
        while True:
            try:
                start_time = time.perf_counter()
                
                # Process pending tasks
                await self._process_task_queue()
                
                # Handle inter-agent communication
                await self._process_messages()
                
                # Check for completed tasks
                await self._check_task_completion()
                
                # Resolve conflicts
                await self._resolve_conflicts()
                
                # Update coordination state
                await self._update_coordination_state()
                
                execution_time = (time.perf_counter() - start_time) * 1000
                self.stats["coordination_cycles"] += 1
                self.stats["total_coordination_time"] += execution_time
                
                # Wait before next cycle
                await asyncio.sleep(1.0)  # 1 second coordination cycle
                
            except asyncio.CancelledError:
                self.logger.info("Coordination loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(5.0)  # Longer wait on error
    
    async def _process_task_queue(self):
        """Process the task queue and assign tasks to agents"""
        tasks_processed = 0
        
        for task_id in self.task_queue.copy():
            if tasks_processed >= 10:  # Limit tasks processed per cycle
                break
            
            task = self.tasks.get(task_id)
            if not task or task.state != "pending":
                self.task_queue.remove(task_id)
                continue
            
            # Check dependencies
            if not self._are_dependencies_satisfied(task):
                continue
            
            # Find available agent
            capable_agents = self._find_capable_agents(task.required_capabilities)
            available_agent = None
            
            for agent_id in capable_agents:
                agent = self.agents.get(agent_id)
                if agent and agent.is_available:
                    available_agent = agent
                    break
            
            if available_agent:
                # Assign task to agent
                await self._assign_task_to_agent(task, available_agent)
                self.task_queue.remove(task_id)
                tasks_processed += 1
            else:
                # No available agents, task stays in queue
                break
    
    def _are_dependencies_satisfied(self, task: Task) -> bool:
        """Check if all task dependencies are satisfied"""
        for dep_task_id in task.dependencies:
            if dep_task_id not in self.completed_tasks:
                return False
        return True
    
    async def _assign_task_to_agent(self, task: Task, agent: Agent):
        """Assign a task to a specific agent"""
        try:
            # Update task state
            task.assigned_agent = agent.agent_id
            task.state = "assigned"
            task.started_at = datetime.now()
            
            # Update agent state
            agent.current_tasks.add(task.task_id)
            agent.state = AgentState.WORKING
            agent.last_active = datetime.now()
            
            # Store assignment in memory
            assignment_data = {
                "task_id": task.task_id,
                "agent_id": agent.agent_id,
                "assigned_at": datetime.now().isoformat(),
                "task_data": task.input_data,
                "agent_namespace": agent.memory_namespace
            }
            
            await self._store_in_memory(
                f"assignment_{task.task_id}",
                assignment_data,
                f"{self.session_id}_coordination"
            )
            
            # Notify agent (simulate task execution)
            await self._notify_agent_of_task(agent, task)
            
            self.logger.info(f"Assigned task {task.name} to agent {agent.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to assign task {task.task_id} to agent {agent.agent_id}: {e}")
            task.state = "error"
            task.error_info = str(e)
    
    async def _notify_agent_of_task(self, agent: Agent, task: Task):
        """Notify agent of new task assignment"""
        # In real implementation, this would send message to agent's execution context
        # For now, simulate task execution
        
        notification = {
            "type": "task_assignment",
            "task_id": task.task_id,
            "task_name": task.name,
            "input_data": task.input_data,
            "agent_namespace": agent.memory_namespace,
            "timestamp": datetime.now().isoformat()
        }
        
        await self._send_message_to_agent(agent.agent_id, notification)
        
        # Simulate task execution completion after some time
        asyncio.create_task(self._simulate_task_execution(task, agent))
    
    async def _simulate_task_execution(self, task: Task, agent: Agent):
        """Simulate task execution (for demonstration)"""
        try:
            # Simulate processing time based on task complexity
            execution_time = 2.0 + len(str(task.input_data)) * 0.01
            await asyncio.sleep(execution_time)
            
            # Simulate successful completion
            task.state = "completed"
            task.completed_at = datetime.now()
            task.output_data = {
                "result": f"Processed by {agent.name}",
                "execution_time_seconds": execution_time,
                "agent_id": agent.agent_id
            }
            
            # Update agent state
            agent.current_tasks.remove(task.task_id)
            agent.completed_tasks.append(task.task_id)
            agent.total_tasks_completed += 1
            agent.total_execution_time_seconds += execution_time
            
            if len(agent.current_tasks) == 0:
                agent.state = AgentState.READY
            
            # Store completion in memory
            completion_data = {
                "task_id": task.task_id,
                "agent_id": agent.agent_id,
                "completed_at": task.completed_at.isoformat(),
                "output_data": task.output_data,
                "execution_time_seconds": execution_time
            }
            
            await self._store_in_memory(
                f"completion_{task.task_id}",
                completion_data,
                f"{self.session_id}_coordination"
            )
            
            self.logger.info(f"Task {task.name} completed by agent {agent.name} in {execution_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Task execution simulation failed: {e}")
            task.state = "error"
            task.error_info = str(e)
            agent.current_tasks.discard(task.task_id)
            if len(agent.current_tasks) == 0:
                agent.state = AgentState.ERROR
    
    async def _process_messages(self):
        """Process inter-agent messages"""
        for message in self.message_queue.copy():
            try:
                await self._deliver_message(message)
                self.message_queue.remove(message)
                self.stats["messages_sent"] += 1
            except Exception as e:
                self.logger.error(f"Failed to deliver message: {e}")
    
    async def _deliver_message(self, message: Dict[str, Any]):
        """Deliver a message to its destination"""
        recipient_id = message.get("recipient_id")
        if recipient_id == "broadcast":
            # Broadcast to all agents
            for agent_id in self.agents.keys():
                await self._store_agent_message(agent_id, message)
        else:
            # Send to specific agent
            await self._store_agent_message(recipient_id, message)
    
    async def _store_agent_message(self, agent_id: str, message: Dict[str, Any]):
        """Store message for agent in their namespace"""
        message_key = f"message_{uuid4().hex[:8]}"
        await self._store_in_memory(
            message_key,
            message,
            f"{self.session_id}_messages"
        )
    
    async def _check_task_completion(self):
        """Check for completed tasks and update state"""
        for task_id, task in self.tasks.items():
            if task.state == "completed" and task_id not in self.completed_tasks:
                self.completed_tasks.add(task_id)
                self.stats["tasks_completed"] += 1
                
                # Notify dependent tasks
                for dependent_task_id, dependent_task in self.tasks.items():
                    if task_id in dependent_task.dependencies and dependent_task.state == "pending":
                        # Dependencies might now be satisfied
                        pass
            
            elif task.state == "error":
                self.stats["tasks_failed"] += 1
                self.logger.error(f"Task {task.name} failed: {task.error_info}")
    
    async def _resolve_conflicts(self):
        """Resolve resource conflicts between agents"""
        # Check for agents blocked by resource conflicts
        conflicts_resolved = 0
        
        for agent_id, agent in self.agents.items():
            if agent.state == AgentState.BLOCKED:
                # Try to resolve blocking conditions
                if self._can_resolve_agent_conflicts(agent):
                    agent.state = AgentState.READY
                    agent.blocked_by.clear()
                    conflicts_resolved += 1
                    self.logger.info(f"Resolved conflicts for agent {agent.name}")
        
        self.stats["conflicts_resolved"] += conflicts_resolved
    
    def _can_resolve_agent_conflicts(self, agent: Agent) -> bool:
        """Check if agent conflicts can be resolved"""
        # Check if blocking agents are no longer busy
        for blocking_agent_id in agent.blocked_by:
            blocking_agent = self.agents.get(blocking_agent_id)
            if blocking_agent and blocking_agent.state == AgentState.WORKING:
                return False  # Still blocked
        return True
    
    async def _update_coordination_state(self):
        """Update overall coordination state in memory"""
        state_data = {
            "session_id": self.session_id,
            "coordination_strategy": self.coordination_strategy.value,
            "active_agents": len([a for a in self.agents.values() if a.state != AgentState.TERMINATED]),
            "pending_tasks": len([t for t in self.tasks.values() if t.state == "pending"]),
            "active_tasks": len([t for t in self.tasks.values() if t.state in ["assigned", "working"]]),
            "completed_tasks": len(self.completed_tasks),
            "last_updated": datetime.now().isoformat(),
            "statistics": self.stats.copy()
        }
        
        await self._store_in_memory(
            "coordination_state",
            state_data,
            f"{self.session_id}_state"
        )
    
    async def _monitoring_loop(self):
        """Monitor agent and task performance"""
        while True:
            try:
                await self._monitor_agent_health()
                await self._monitor_task_timeouts()
                await self._optimize_coordination()
                
                await asyncio.sleep(30.0)  # Monitor every 30 seconds
                
            except asyncio.CancelledError:
                self.logger.info("Monitoring loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30.0)
    
    async def _monitor_agent_health(self):
        """Monitor health and performance of agents"""
        current_time = datetime.now()
        
        for agent in self.agents.values():
            # Check if agent is unresponsive
            time_since_active = current_time - agent.last_active
            if time_since_active > timedelta(minutes=5) and agent.state not in [AgentState.TERMINATED, AgentState.ERROR]:
                self.logger.warning(f"Agent {agent.name} has been inactive for {time_since_active}")
                
                # Mark agent as potentially problematic
                if time_since_active > timedelta(minutes=10):
                    agent.state = AgentState.ERROR
                    self.logger.error(f"Agent {agent.name} marked as error due to inactivity")
    
    async def _monitor_task_timeouts(self):
        """Monitor for tasks that have exceeded timeout"""
        current_time = datetime.now()
        
        for task in self.tasks.values():
            if task.state in ["assigned", "working"] and task.started_at:
                execution_time = current_time - task.started_at
                if execution_time.total_seconds() > self.task_timeout_seconds:
                    self.logger.warning(f"Task {task.name} has exceeded timeout ({execution_time})")
                    
                    # Mark task as failed
                    task.state = "error"
                    task.error_info = f"Task timeout after {execution_time}"
                    
                    # Free up the agent
                    if task.assigned_agent:
                        agent = self.agents.get(task.assigned_agent)
                        if agent:
                            agent.current_tasks.discard(task.task_id)
                            if len(agent.current_tasks) == 0:
                                agent.state = AgentState.READY
    
    async def _optimize_coordination(self):
        """Optimize coordination based on performance metrics"""
        # Analyze agent utilization
        total_agents = len(self.agents)
        active_agents = len([a for a in self.agents.values() if a.state == AgentState.WORKING])
        
        if total_agents > 0:
            utilization = active_agents / total_agents
            
            if utilization > 0.9:
                self.logger.info(f"High agent utilization ({utilization:.2%}) - consider adding more agents")
            elif utilization < 0.3:
                self.logger.info(f"Low agent utilization ({utilization:.2%}) - might have too many agents")
        
        # Optimize task queue based on completion patterns
        avg_completion_time = self._calculate_avg_task_completion_time()
        if avg_completion_time > 120:  # 2 minutes
            self.logger.info(f"Average task completion time is high ({avg_completion_time:.1f}s)")
    
    def _calculate_avg_task_completion_time(self) -> float:
        """Calculate average task completion time"""
        completion_times = []
        
        for task in self.tasks.values():
            if task.state == "completed" and task.started_at and task.completed_at:
                duration = (task.completed_at - task.started_at).total_seconds()
                completion_times.append(duration)
        
        return sum(completion_times) / max(len(completion_times), 1)
    
    async def send_message_to_agent(self, agent_id: str, message: Dict[str, Any]) -> bool:
        """Send a message to a specific agent"""
        try:
            message_envelope = {
                "sender_id": "coordination_hub",
                "recipient_id": agent_id,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "message_id": uuid4().hex
            }
            
            self.message_queue.append(message_envelope)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to queue message for agent {agent_id}: {e}")
            return False
    
    async def _send_message_to_agent(self, agent_id: str, message: Dict[str, Any]):
        """Internal method to send message to agent"""
        return await self.send_message_to_agent(agent_id, message)
    
    async def broadcast_message(self, message: Dict[str, Any]) -> bool:
        """Broadcast a message to all agents"""
        try:
            message_envelope = {
                "sender_id": "coordination_hub",
                "recipient_id": "broadcast",
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "message_id": uuid4().hex
            }
            
            self.message_queue.append(message_envelope)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to queue broadcast message: {e}")
            return False
    
    async def _store_in_memory(self, key: str, data: Any, namespace: str):
        """Store data in memory (MCP or local fallback)"""
        if self.memory_client and hasattr(self.memory_client, 'store'):
            await self.memory_client.store(key, data, namespace)
        else:
            # Fallback to local memory
            if not hasattr(self, 'local_memory'):
                self.local_memory = {}
            self.local_memory[f"{namespace}:{key}"] = data
    
    async def _retrieve_from_memory(self, key: str, namespace: str) -> Any:
        """Retrieve data from memory"""
        if self.memory_client and hasattr(self.memory_client, 'retrieve'):
            return await self.memory_client.retrieve(key, namespace)
        else:
            # Fallback to local memory
            if not hasattr(self, 'local_memory'):
                return None
            return self.local_memory.get(f"{namespace}:{key}")
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get comprehensive coordination status"""
        current_time = datetime.now()
        session_duration = current_time - self.coordination_start_time
        
        agent_status = {}
        for agent_id, agent in self.agents.items():
            agent_status[agent_id] = {
                "name": agent.name,
                "state": agent.state.value,
                "current_tasks": len(agent.current_tasks),
                "completed_tasks": agent.total_tasks_completed,
                "success_rate": agent.success_rate,
                "avg_execution_time": agent.avg_execution_time,
                "capabilities": [cap.name for cap in agent.capabilities],
                "is_available": agent.is_available
            }
        
        task_status = {}
        for task_id, task in self.tasks.items():
            task_status[task_id] = {
                "name": task.name,
                "state": task.state,
                "priority": task.priority.value,
                "assigned_agent": task.assigned_agent,
                "created_at": task.created_at.isoformat(),
                "execution_time": None
            }
            
            if task.started_at and task.completed_at:
                execution_time = (task.completed_at - task.started_at).total_seconds()
                task_status[task_id]["execution_time"] = execution_time
        
        return {
            "session_info": {
                "session_id": self.session_id,
                "coordination_strategy": self.coordination_strategy.value,
                "session_duration_seconds": session_duration.total_seconds(),
                "max_parallel_agents": self.max_parallel_agents
            },
            "agents": agent_status,
            "tasks": {
                "total": len(self.tasks),
                "pending": len([t for t in self.tasks.values() if t.state == "pending"]),
                "active": len([t for t in self.tasks.values() if t.state in ["assigned", "working"]]),
                "completed": len(self.completed_tasks),
                "failed": self.stats["tasks_failed"],
                "queue_length": len(self.task_queue)
            },
            "performance": {
                "avg_coordination_cycle_ms": self.stats["total_coordination_time"] / max(self.stats["coordination_cycles"], 1),
                "avg_task_completion_time_seconds": self._calculate_avg_task_completion_time(),
                "coordination_efficiency": self.stats["tasks_completed"] / max(self.stats["coordination_cycles"], 1),
                "message_throughput": self.stats["messages_sent"]
            },
            "statistics": self.stats.copy()
        }
    
    async def shutdown(self):
        """Shutdown coordination hub and clean up resources"""
        self.logger.info("Shutting down Agent Coordination Hub...")
        
        # Cancel background tasks
        if self.coordination_task and not self.coordination_task.done():
            self.coordination_task.cancel()
        
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
        
        # Mark all agents as terminated
        for agent in self.agents.values():
            agent.state = AgentState.TERMINATED
        
        # Store final state
        await self._update_coordination_state()
        
        self.logger.info("Agent Coordination Hub shutdown complete")


# Example usage and testing
async def example_coordination_usage():
    """Example of how to use the MCP Agent Coordination Hub"""
    
    # Create coordination hub (without real MCP memory client for demo)
    hub = MCPAgentCoordinationHub(session_id="demo_session")
    await hub.initialize()
    
    # Define agent capabilities
    research_capability = AgentCapability(
        name="research",
        description="Research and analyze information",
        input_types=["query", "topic"],
        output_types=["research_report", "analysis"],
        estimated_duration_seconds=30.0
    )
    
    coding_capability = AgentCapability(
        name="coding",
        description="Write and test code",
        input_types=["requirements", "specification"],
        output_types=["code", "tests"],
        estimated_duration_seconds=60.0
    )
    
    # Create agents
    research_agent = Agent(
        agent_type="researcher",
        name="Research Agent Alpha",
        description="Specialized in research and analysis",
        capabilities=[research_capability],
        max_concurrent_tasks=2
    )
    
    coding_agent = Agent(
        agent_type="coder", 
        name="Coding Agent Beta",
        description="Specialized in software development",
        capabilities=[coding_capability],
        max_concurrent_tasks=1
    )
    
    # Register agents
    await hub.register_agent(research_agent)
    await hub.register_agent(coding_agent)
    
    # Create tasks
    research_task = Task(
        name="Research MCP Architecture",
        description="Research best practices for MCP architecture",
        priority=TaskPriority.HIGH,
        required_capabilities=["research"],
        input_data={"topic": "MCP architecture patterns", "depth": "comprehensive"}
    )
    
    coding_task = Task(
        name="Implement MCP Client",
        description="Implement MCP client library",
        priority=TaskPriority.HIGH,
        required_capabilities=["coding"],
        dependencies=[research_task.task_id],  # Depends on research
        input_data={"language": "python", "features": ["connection", "messaging"]}
    )
    
    # Submit tasks
    await hub.submit_task(research_task)
    await hub.submit_task(coding_task)
    
    # Let coordination run for a while
    await asyncio.sleep(10.0)
    
    # Check status
    status = hub.get_coordination_status()
    print(f"Coordination Status:")
    print(f"- Session: {status['session_info']['session_id']}")
    print(f"- Agents: {len(status['agents'])}")
    print(f"- Tasks completed: {status['tasks']['completed']}")
    print(f"- Performance: {status['performance']['coordination_efficiency']:.2f} tasks/cycle")
    
    # Shutdown
    await hub.shutdown()


if __name__ == "__main__":
    asyncio.run(example_coordination_usage())