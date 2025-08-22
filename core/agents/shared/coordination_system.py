"""Agent Coordination System - Prompt J

Comprehensive multi-agent coordination and orchestration framework including:
- Agent discovery and registration
- Task distribution and load balancing
- Inter-agent communication protocols
- Consensus mechanisms and decision making
- Resource allocation and scheduling
- Fault tolerance and recovery

Integration Point: Agent coordination for Phase 4 testing
"""

import heapq
import json
import logging
import sqlite3
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class AgentStatus(Enum):
    """Agent status states."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    FAILED = "failed"


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class MessageType(Enum):
    """Inter-agent message types."""

    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    HEARTBEAT = "heartbeat"
    COORDINATION = "coordination"
    CONSENSUS = "consensus"
    RESOURCE_REQUEST = "resource_request"
    BROADCAST = "broadcast"


class CoordinationStrategy(Enum):
    """Coordination strategies."""

    CENTRALIZED = "centralized"
    DISTRIBUTED = "distributed"
    HIERARCHICAL = "hierarchical"
    PEER_TO_PEER = "peer_to_peer"
    CONSENSUS = "consensus"


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = 1
    NORMAL = 3
    HIGH = 7
    CRITICAL = 10


class ResourceType(Enum):
    """Resource types for allocation management."""

    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    POWER = "power"


@dataclass
class AgentCapability:
    """Agent capability description."""

    name: str
    version: str
    description: str
    resource_requirements: dict[str, float] = field(default_factory=dict)
    max_concurrent_tasks: int = 1
    supported_task_types: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Agent:
    """Agent registration information."""

    agent_id: str
    name: str
    agent_type: str
    capabilities: list[AgentCapability]
    status: AgentStatus
    endpoint: str
    registered_at: float
    last_heartbeat: float
    current_load: float = 0.0
    max_load: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    performance_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class Task:
    """Task definition and tracking."""

    task_id: str
    task_type: str
    description: str
    priority: int
    payload: dict[str, Any]
    requirements: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)

    # Execution tracking
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent_id: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    retry_count: int = 0
    max_retries: int = 3

    # Coordination
    dependencies: list[str] = field(default_factory=list)
    subtasks: list[str] = field(default_factory=list)
    parent_task_id: str | None = None


@dataclass
class Message:
    """Inter-agent message."""

    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: str | None  # None for broadcast
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    reply_to: str | None = None
    ttl_seconds: int = 300


@dataclass
class Resource:
    """System resource definition."""

    resource_id: str
    resource_type: str
    capacity: float
    allocated: float = 0.0
    available: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.available == 0.0:
            self.available = self.capacity - self.allocated


class AgentRegistry:
    """Agent discovery and registration service."""

    def __init__(self, storage_backend: str = ":memory:"):
        """Initialize agent registry.

        Args:
            storage_backend: Storage backend for persistence
        """
        self.storage_backend = storage_backend
        self.agents: dict[str, Agent] = {}
        self.capabilities_index: dict[str, set[str]] = defaultdict(set)
        self._lock = threading.Lock()

        self._init_storage()

    def _init_storage(self):
        """Initialize storage backend."""
        with self._get_db() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    capabilities TEXT NOT NULL,
                    status TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    registered_at REAL NOT NULL,
                    last_heartbeat REAL NOT NULL,
                    current_load REAL DEFAULT 0.0,
                    max_load REAL DEFAULT 1.0,
                    metadata TEXT DEFAULT '{}',
                    performance_metrics TEXT DEFAULT '{}'
                )
            """
            )

    @contextmanager
    def _get_db(self):
        """Get database connection."""
        conn = sqlite3.connect(self.storage_backend)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def register_agent(self, agent: Agent) -> bool:
        """Register an agent.

        Args:
            agent: Agent to register

        Returns:
            True if registration successful
        """
        with self._lock:
            # Update capabilities index
            for capability in agent.capabilities:
                for task_type in capability.supported_task_types:
                    self.capabilities_index[task_type].add(agent.agent_id)

            self.agents[agent.agent_id] = agent

            # Persist to storage
            with self._get_db() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO agents (
                        agent_id, name, agent_type, capabilities, status,
                        endpoint, registered_at, last_heartbeat, current_load,
                        max_load, metadata, performance_metrics
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        agent.agent_id,
                        agent.name,
                        agent.agent_type,
                        json.dumps([cap.__dict__ for cap in agent.capabilities]),
                        agent.status.value,
                        agent.endpoint,
                        agent.registered_at,
                        agent.last_heartbeat,
                        agent.current_load,
                        agent.max_load,
                        json.dumps(agent.metadata),
                        json.dumps(agent.performance_metrics),
                    ),
                )
                conn.commit()

        return True

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent.

        Args:
            agent_id: Agent ID to unregister

        Returns:
            True if unregistration successful
        """
        with self._lock:
            if agent_id not in self.agents:
                return False

            agent = self.agents[agent_id]

            # Remove from capabilities index
            for capability in agent.capabilities:
                for task_type in capability.supported_task_types:
                    self.capabilities_index[task_type].discard(agent_id)

            del self.agents[agent_id]

            # Remove from storage
            with self._get_db() as conn:
                conn.execute("DELETE FROM agents WHERE agent_id = ?", (agent_id,))
                conn.commit()

        return True

    def update_agent_status(
        self,
        agent_id: str,
        status: AgentStatus,
        load: float | None = None,
        metrics: dict[str, float] | None = None,
    ):
        """Update agent status and metrics.

        Args:
            agent_id: Agent ID
            status: New status
            load: Current load (0.0 to 1.0)
            metrics: Performance metrics
        """
        with self._lock:
            if agent_id not in self.agents:
                return

            agent = self.agents[agent_id]
            agent.status = status
            agent.last_heartbeat = time.time()

            if load is not None:
                agent.current_load = load

            if metrics:
                agent.performance_metrics.update(metrics)

    def find_agents_by_capability(self, task_type: str, status_filter: list[AgentStatus] = None) -> list[Agent]:
        """Find agents by capability.

        Args:
            task_type: Required task type capability
            status_filter: Allowed agent statuses

        Returns:
            List of matching agents
        """
        if status_filter is None:
            status_filter = [AgentStatus.ACTIVE, AgentStatus.IDLE]

        candidate_ids = self.capabilities_index.get(task_type, set())

        matching_agents = []
        for agent_id in candidate_ids:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                if agent.status in status_filter:
                    matching_agents.append(agent)

        return matching_agents

    def get_agent(self, agent_id: str) -> Agent | None:
        """Get agent by ID.

        Args:
            agent_id: Agent ID

        Returns:
            Agent if found, None otherwise
        """
        return self.agents.get(agent_id)

    def list_agents(self, status_filter: list[AgentStatus] = None) -> list[Agent]:
        """List all agents.

        Args:
            status_filter: Filter by status

        Returns:
            List of agents
        """
        if status_filter is None:
            return list(self.agents.values())

        return [agent for agent in self.agents.values() if agent.status in status_filter]


class TaskScheduler:
    """Task scheduling and distribution system."""

    def __init__(self, agent_registry: AgentRegistry, storage_backend: str = ":memory:"):
        """Initialize task scheduler.

        Args:
            agent_registry: Agent registry
            storage_backend: Storage backend
        """
        self.agent_registry = agent_registry
        self.storage_backend = storage_backend

        # Task queues (priority queues)
        self.pending_tasks: list[tuple[int, float, Task]] = []  # (priority, timestamp, task)
        self.running_tasks: dict[str, Task] = {}
        self.completed_tasks: dict[str, Task] = {}
        self.failed_tasks: dict[str, Task] = {}

        # Agent assignment tracking
        self.agent_assignments: dict[str, set[str]] = defaultdict(set)  # agent_id -> task_ids

        self._lock = threading.Lock()
        self._init_storage()

    def _init_storage(self):
        """Initialize task storage."""
        with self._get_db() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    requirements TEXT DEFAULT '{}',
                    constraints TEXT DEFAULT '{}',
                    status TEXT NOT NULL,
                    assigned_agent_id TEXT,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    result TEXT,
                    error TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    dependencies TEXT DEFAULT '[]',
                    subtasks TEXT DEFAULT '[]',
                    parent_task_id TEXT
                )
            """
            )

    @contextmanager
    def _get_db(self):
        """Get database connection."""
        conn = sqlite3.connect(self.storage_backend)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def submit_task(self, task: Task) -> str:
        """Submit a task for execution.

        Args:
            task: Task to submit

        Returns:
            Task ID
        """
        with self._lock:
            # Add to pending queue (negative priority for max heap behavior)
            heapq.heappush(self.pending_tasks, (-task.priority, task.created_at, task))

            # Persist to storage
            with self._get_db() as conn:
                conn.execute(
                    """
                    INSERT INTO tasks (
                        task_id, task_type, description, priority, payload,
                        requirements, constraints, status, created_at,
                        retry_count, max_retries, dependencies, subtasks,
                        parent_task_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        task.task_id,
                        task.task_type,
                        task.description,
                        task.priority,
                        json.dumps(task.payload),
                        json.dumps(task.requirements),
                        json.dumps(task.constraints),
                        task.status.value,
                        task.created_at,
                        task.retry_count,
                        task.max_retries,
                        json.dumps(task.dependencies),
                        json.dumps(task.subtasks),
                        task.parent_task_id,
                    ),
                )
                conn.commit()

        return task.task_id

    def schedule_next_task(self) -> tuple[Task, Agent] | None:
        """Schedule the next highest priority task.

        Returns:
            Tuple of (task, assigned_agent) if successful, None otherwise
        """
        with self._lock:
            while self.pending_tasks:
                _, _, task = heapq.heappop(self.pending_tasks)

                # Check dependencies
                if not self._are_dependencies_satisfied(task):
                    # Re-queue task
                    heapq.heappush(self.pending_tasks, (-task.priority, task.created_at, task))
                    continue

                # Find suitable agent
                agent = self._find_best_agent(task)
                if agent is None:
                    # No suitable agent available, re-queue
                    heapq.heappush(self.pending_tasks, (-task.priority, task.created_at, task))
                    return None

                # Assign task to agent
                task.status = TaskStatus.ASSIGNED
                task.assigned_agent_id = agent.agent_id

                self.running_tasks[task.task_id] = task
                self.agent_assignments[agent.agent_id].add(task.task_id)

                # Update agent load
                self.agent_registry.update_agent_status(
                    agent.agent_id,
                    AgentStatus.BUSY,
                    load=len(self.agent_assignments[agent.agent_id]) / agent.max_load,
                )

                return task, agent

        return None

    def _are_dependencies_satisfied(self, task: Task) -> bool:
        """Check if task dependencies are satisfied.

        Args:
            task: Task to check

        Returns:
            True if all dependencies are completed
        """
        for dep_task_id in task.dependencies:
            if dep_task_id not in self.completed_tasks:
                return False
        return True

    def _find_best_agent(self, task: Task) -> Agent | None:
        """Find the best agent for a task.

        Args:
            task: Task to assign

        Returns:
            Best agent if found, None otherwise
        """
        # Find agents with required capability
        capable_agents = self.agent_registry.find_agents_by_capability(task.task_type)

        if not capable_agents:
            return None

        # Score agents based on load, performance, and constraints
        best_agent = None
        best_score = float("-inf")

        for agent in capable_agents:
            # Check if agent can handle more tasks
            current_tasks = len(self.agent_assignments[agent.agent_id])
            max_tasks = max(cap.max_concurrent_tasks for cap in agent.capabilities)

            if current_tasks >= max_tasks:
                continue

            # Calculate score
            load_score = 1.0 - agent.current_load  # Prefer less loaded agents
            performance_score = agent.performance_metrics.get("success_rate", 0.5)
            availability_score = 1.0 if agent.status == AgentStatus.IDLE else 0.5

            total_score = load_score * 0.4 + performance_score * 0.4 + availability_score * 0.2

            if total_score > best_score:
                best_score = total_score
                best_agent = agent

        return best_agent

    def complete_task(self, task_id: str, result: dict[str, Any] = None, error: str = None) -> bool:
        """Mark a task as completed.

        Args:
            task_id: Task ID
            result: Task result
            error: Error message if failed

        Returns:
            True if update successful
        """
        with self._lock:
            if task_id not in self.running_tasks:
                return False

            task = self.running_tasks[task_id]
            task.completed_at = time.time()

            if error:
                task.status = TaskStatus.FAILED
                task.error = error
                self.failed_tasks[task_id] = task
            else:
                task.status = TaskStatus.COMPLETED
                task.result = result
                self.completed_tasks[task_id] = task

            del self.running_tasks[task_id]

            # Update agent assignment
            if task.assigned_agent_id:
                self.agent_assignments[task.assigned_agent_id].discard(task_id)

                # Update agent load
                agent = self.agent_registry.get_agent(task.assigned_agent_id)
                if agent:
                    new_load = len(self.agent_assignments[task.assigned_agent_id]) / agent.max_load
                    new_status = AgentStatus.IDLE if new_load == 0 else AgentStatus.BUSY

                    self.agent_registry.update_agent_status(task.assigned_agent_id, new_status, load=new_load)

        return True

    def retry_task(self, task_id: str) -> bool:
        """Retry a failed task.

        Args:
            task_id: Task ID to retry

        Returns:
            True if retry queued
        """
        with self._lock:
            task = self.failed_tasks.get(task_id)
            if not task or task.retry_count >= task.max_retries:
                return False

            task.retry_count += 1
            task.status = TaskStatus.RETRYING
            task.assigned_agent_id = None
            task.error = None

            # Re-queue task
            heapq.heappush(self.pending_tasks, (-task.priority, time.time(), task))

            del self.failed_tasks[task_id]

        return True

    def get_task_status(self, task_id: str) -> Task | None:
        """Get task status.

        Args:
            task_id: Task ID

        Returns:
            Task if found, None otherwise
        """
        # Check all task collections
        for task_dict in [self.running_tasks, self.completed_tasks, self.failed_tasks]:
            if task_id in task_dict:
                return task_dict[task_id]

        # Check pending tasks
        for _, _, task in self.pending_tasks:
            if task.task_id == task_id:
                return task

        return None


class MessageBroker:
    """Inter-agent message broker."""

    def __init__(self):
        """Initialize message broker."""
        self.message_queues: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.broadcast_queue: deque = deque(maxlen=1000)
        self.message_handlers: dict[str, dict[MessageType, Callable]] = defaultdict(dict)

        self._lock = threading.Lock()

    def register_handler(self, agent_id: str, message_type: MessageType, handler: Callable):
        """Register message handler for agent.

        Args:
            agent_id: Agent ID
            message_type: Message type to handle
            handler: Handler function
        """
        with self._lock:
            self.message_handlers[agent_id][message_type] = handler

    def send_message(self, message: Message):
        """Send message to agent(s).

        Args:
            message: Message to send
        """
        with self._lock:
            if message.recipient_id:
                # Direct message
                self.message_queues[message.recipient_id].append(message)
            else:
                # Broadcast message
                self.broadcast_queue.append(message)

    def get_messages(self, agent_id: str, limit: int = 10) -> list[Message]:
        """Get messages for agent.

        Args:
            agent_id: Agent ID
            limit: Maximum messages to return

        Returns:
            List of messages
        """
        messages = []

        with self._lock:
            # Get direct messages
            agent_queue = self.message_queues[agent_id]
            while agent_queue and len(messages) < limit:
                message = agent_queue.popleft()
                if self._is_message_valid(message):
                    messages.append(message)

            # Get broadcast messages
            if len(messages) < limit:
                remaining_limit = limit - len(messages)
                broadcast_messages = list(self.broadcast_queue)[-remaining_limit:]

                for message in broadcast_messages:
                    if (
                        self._is_message_valid(message) and message.sender_id != agent_id
                    ):  # Don't send own broadcasts back
                        messages.append(message)

        return messages

    def _is_message_valid(self, message: Message) -> bool:
        """Check if message is still valid (not expired).

        Args:
            message: Message to check

        Returns:
            True if valid
        """
        return time.time() - message.timestamp < message.ttl_seconds

    def process_messages(self, agent_id: str):
        """Process pending messages for agent.

        Args:
            agent_id: Agent ID
        """
        messages = self.get_messages(agent_id)
        handlers = self.message_handlers.get(agent_id, {})

        for message in messages:
            handler = handlers.get(message.message_type)
            if handler:
                try:
                    handler(message)
                except Exception as e:
                    logging.error(f"Message handler error: {e}")


class ResourceManager:
    """System resource management."""

    def __init__(self):
        """Initialize resource manager."""
        self.resources: dict[str, Resource] = {}
        self.allocations: dict[str, dict[str, float]] = defaultdict(dict)  # resource_id -> {agent_id: amount}
        self._lock = threading.Lock()

    def register_resource(self, resource: Resource):
        """Register a system resource.

        Args:
            resource: Resource to register
        """
        with self._lock:
            self.resources[resource.resource_id] = resource

    def allocate_resource(self, resource_id: str, agent_id: str, amount: float) -> bool:
        """Allocate resource to agent.

        Args:
            resource_id: Resource ID
            agent_id: Agent ID requesting resource
            amount: Amount to allocate

        Returns:
            True if allocation successful
        """
        with self._lock:
            if resource_id not in self.resources:
                return False

            resource = self.resources[resource_id]

            if resource.available < amount:
                return False

            # Allocate resource
            resource.allocated += amount
            resource.available -= amount
            self.allocations[resource_id][agent_id] = amount

        return True

    def release_resource(self, resource_id: str, agent_id: str) -> bool:
        """Release resource from agent.

        Args:
            resource_id: Resource ID
            agent_id: Agent ID

        Returns:
            True if release successful
        """
        with self._lock:
            if resource_id not in self.resources or agent_id not in self.allocations[resource_id]:
                return False

            resource = self.resources[resource_id]
            amount = self.allocations[resource_id][agent_id]

            # Release resource
            resource.allocated -= amount
            resource.available += amount
            del self.allocations[resource_id][agent_id]

        return True

    def get_resource_usage(self) -> dict[str, dict[str, float]]:
        """Get current resource usage.

        Returns:
            Resource usage information
        """
        with self._lock:
            usage = {}
            for resource_id, resource in self.resources.items():
                usage[resource_id] = {
                    "capacity": resource.capacity,
                    "allocated": resource.allocated,
                    "available": resource.available,
                    "utilization": resource.allocated / resource.capacity if resource.capacity > 0 else 0,
                }
        return usage


class CoordinationEngine:
    """Main coordination engine."""

    def __init__(
        self,
        strategy: CoordinationStrategy = CoordinationStrategy.CENTRALIZED,
        storage_backend: str = ":memory:",
    ):
        """Initialize coordination engine.

        Args:
            strategy: Coordination strategy
            storage_backend: Storage backend
        """
        self.strategy = strategy
        self.storage_backend = storage_backend

        # Core components
        self.agent_registry = AgentRegistry(storage_backend)
        self.task_scheduler = TaskScheduler(self.agent_registry, storage_backend)
        self.message_broker = MessageBroker()
        self.resource_manager = ResourceManager()

        # Coordination state
        self.coordinator_id = str(uuid.uuid4())
        self.running = False
        self._coordination_thread = None

        # Setup default resources
        self._setup_default_resources()

    def _setup_default_resources(self):
        """Setup default system resources."""
        self.resource_manager.register_resource(Resource("cpu", "compute", capacity=100.0))
        self.resource_manager.register_resource(Resource("memory", "memory", capacity=8192.0))  # MB
        self.resource_manager.register_resource(Resource("network", "bandwidth", capacity=1000.0))  # Mbps

    def start(self):
        """Start coordination engine."""
        if self.running:
            return

        self.running = True
        self._coordination_thread = threading.Thread(target=self._coordination_loop, daemon=True)
        self._coordination_thread.start()

        logging.info(f"Coordination engine started with strategy: {self.strategy.value}")

    def stop(self):
        """Stop coordination engine."""
        self.running = False
        if self._coordination_thread:
            self._coordination_thread.join(timeout=5.0)

        logging.info("Coordination engine stopped")

    def _coordination_loop(self):
        """Main coordination loop."""
        while self.running:
            try:
                # Schedule pending tasks
                result = self.task_scheduler.schedule_next_task()
                if result:
                    task, agent = result
                    self._send_task_to_agent(task, agent)

                # Process agent heartbeats and timeouts
                self._process_agent_heartbeats()

                # Clean up expired messages
                self._cleanup_expired_messages()

                time.sleep(1.0)  # Coordination cycle interval

            except Exception as e:
                logging.error(f"Coordination loop error: {e}")
                time.sleep(5.0)

    def _send_task_to_agent(self, task: Task, agent: Agent):
        """Send task to assigned agent.

        Args:
            task: Task to send
            agent: Target agent
        """
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.TASK_REQUEST,
            sender_id=self.coordinator_id,
            recipient_id=agent.agent_id,
            payload={
                "task": {
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "description": task.description,
                    "priority": task.priority,
                    "payload": task.payload,
                    "requirements": task.requirements,
                    "constraints": task.constraints,
                }
            },
        )

        self.message_broker.send_message(message)
        task.started_at = time.time()
        task.status = TaskStatus.RUNNING

    def _process_agent_heartbeats(self):
        """Process agent heartbeats and handle timeouts."""
        current_time = time.time()
        timeout_threshold = 60.0  # 60 seconds

        for agent in self.agent_registry.list_agents():
            if current_time - agent.last_heartbeat > timeout_threshold:
                # Agent timeout - mark as offline
                self.agent_registry.update_agent_status(agent.agent_id, AgentStatus.OFFLINE)

                # Reassign tasks from offline agent
                self._reassign_agent_tasks(agent.agent_id)

    def _reassign_agent_tasks(self, agent_id: str):
        """Reassign tasks from failed/offline agent.

        Args:
            agent_id: Failed agent ID
        """
        tasks_to_reassign = []

        for task_id in list(self.task_scheduler.running_tasks.keys()):
            task = self.task_scheduler.running_tasks[task_id]
            if task.assigned_agent_id == agent_id:
                tasks_to_reassign.append(task)

        for task in tasks_to_reassign:
            # Reset task for reassignment
            task.status = TaskStatus.PENDING
            task.assigned_agent_id = None
            task.started_at = None

            # Remove from running tasks
            del self.task_scheduler.running_tasks[task.task_id]

            # Re-queue for scheduling
            with self.task_scheduler._lock:
                heapq.heappush(
                    self.task_scheduler.pending_tasks,
                    (-task.priority, time.time(), task),
                )

    def _cleanup_expired_messages(self):
        """Clean up expired messages from queues."""
        current_time = time.time()

        # Clean agent message queues
        for agent_id, queue in self.message_broker.message_queues.items():
            valid_messages = deque()
            while queue:
                message = queue.popleft()
                if current_time - message.timestamp < message.ttl_seconds:
                    valid_messages.append(message)
            self.message_broker.message_queues[agent_id] = valid_messages

        # Clean broadcast queue
        valid_broadcasts = deque()
        while self.message_broker.broadcast_queue:
            message = self.message_broker.broadcast_queue.popleft()
            if current_time - message.timestamp < message.ttl_seconds:
                valid_broadcasts.append(message)
        self.message_broker.broadcast_queue = valid_broadcasts

    def register_agent(self, agent: Agent) -> bool:
        """Register agent with coordination system.

        Args:
            agent: Agent to register

        Returns:
            True if registration successful
        """
        success = self.agent_registry.register_agent(agent)

        if success:
            # Send welcome message
            welcome_message = Message(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.BROADCAST,
                sender_id=self.coordinator_id,
                recipient_id=None,
                payload={
                    "event": "agent_joined",
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                },
            )
            self.message_broker.send_message(welcome_message)

        return success

    def submit_task(self, task: Task) -> str:
        """Submit task for execution.

        Args:
            task: Task to submit

        Returns:
            Task ID
        """
        return self.task_scheduler.submit_task(task)

    def get_system_status(self) -> dict[str, Any]:
        """Get overall system status.

        Returns:
            System status information
        """
        agents = self.agent_registry.list_agents()
        active_agents = len([a for a in agents if a.status == AgentStatus.ACTIVE])

        return {
            "coordinator_id": self.coordinator_id,
            "strategy": self.strategy.value,
            "running": self.running,
            "agents": {
                "total": len(agents),
                "active": active_agents,
                "by_status": {status.value: len([a for a in agents if a.status == status]) for status in AgentStatus},
            },
            "tasks": {
                "pending": len(self.task_scheduler.pending_tasks),
                "running": len(self.task_scheduler.running_tasks),
                "completed": len(self.task_scheduler.completed_tasks),
                "failed": len(self.task_scheduler.failed_tasks),
            },
            "resources": self.resource_manager.get_resource_usage(),
            "message_queues": {agent_id: len(queue) for agent_id, queue in self.message_broker.message_queues.items()},
        }


# Convenience functions for agent implementation
def create_agent(
    name: str,
    agent_type: str,
    capabilities: list[AgentCapability],
    endpoint: str = "localhost:0",
) -> Agent:
    """Create agent instance.

    Args:
        name: Agent name
        agent_type: Agent type
        capabilities: Agent capabilities
        endpoint: Agent endpoint

    Returns:
        Agent instance
    """
    return Agent(
        agent_id=str(uuid.uuid4()),
        name=name,
        agent_type=agent_type,
        capabilities=capabilities,
        status=AgentStatus.INITIALIZING,
        endpoint=endpoint,
        registered_at=time.time(),
        last_heartbeat=time.time(),
    )


def create_task(
    task_type: str,
    description: str,
    payload: dict[str, Any],
    priority: int = 5,
    requirements: dict[str, Any] = None,
    constraints: dict[str, Any] = None,
) -> Task:
    """Create task instance."""
    return Task(
        task_id=str(uuid.uuid4()),
        task_type=task_type,
        description=description,
        priority=priority,
        payload=payload,
        requirements=requirements or {},
        constraints=constraints or {},
    )
