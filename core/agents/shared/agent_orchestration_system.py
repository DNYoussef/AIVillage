"""AIVillage Agent Orchestration System

This system provides comprehensive orchestration for all 23+ specialized agents with:
- Unified agent registry and lifecycle management
- Multi-agent communication channels and coordination
- Task distribution and load balancing
- RAG system integration for group memory
- P2P network integration for distributed operation
- MCP tool management and service discovery
- Real-time monitoring and health checks
- Agent-to-agent communication protocols

The orchestration system enables complex multi-agent workflows while maintaining
system-wide coherence, security, and performance optimization.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from typing import Any
from uuid import uuid4

from packages.agents.core.agent_interface import AgentCapability, MessageInterface, TaskInterface

# Core imports
from packages.agents.core.base_agent_template import BaseAgentTemplate

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent lifecycle status"""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    OFFLINE = "offline"


class CommunicationChannelType(Enum):
    """Types of communication channels"""

    DIRECT = "direct"  # Point-to-point agent communication
    BROADCAST = "broadcast"  # One-to-many broadcasting
    GROUP = "group"  # Topic-based group channels
    EMERGENCY = "emergency"  # High-priority emergency channel
    COORDINATION = "coordination"  # Task coordination channel


class TaskDistributionStrategy(Enum):
    """Task distribution strategies"""

    ROUND_ROBIN = "round_robin"
    CAPABILITY_BASED = "capability_based"
    LOAD_BALANCED = "load_balanced"
    OPTIMIZATION_BASED = "optimization_based"
    PRIORITY_WEIGHTED = "priority_weighted"


@dataclass
class AgentRegistration:
    """Agent registration information"""

    agent_id: str
    agent: BaseAgentTemplate
    agent_type: str
    specialized_role: str
    capabilities: set[AgentCapability]
    status: AgentStatus

    # Performance metrics
    tasks_completed: int = 0
    average_response_time: float = 0.0
    success_rate: float = 1.0
    current_load: float = 0.0
    last_activity: datetime = field(default_factory=datetime.now)

    # Health monitoring
    health_score: float = 1.0
    error_count: int = 0
    last_health_check: datetime = field(default_factory=datetime.now)

    # Communication
    active_channels: list[str] = field(default_factory=list)
    message_queue_depth: int = 0


@dataclass
class CommunicationChannel:
    """Communication channel definition"""

    channel_id: str
    channel_type: CommunicationChannelType
    name: str
    description: str
    participants: set[str] = field(default_factory=set)
    message_history: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True

    # Channel policies
    max_participants: int = 100
    message_retention_hours: int = 24
    priority_level: int = 5


@dataclass
class MultiAgentTask:
    """Multi-agent task coordination"""

    task_id: str
    description: str
    requester_id: str
    required_capabilities: list[AgentCapability]
    assigned_agents: list[str] = field(default_factory=list)
    task_status: str = "pending"

    # Coordination
    coordination_strategy: str = "collaborative"
    max_agents: int = 5
    timeout_seconds: int = 300

    # Progress tracking
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    subtasks: list[dict[str, Any]] = field(default_factory=list)

    # Results
    results: dict[str, Any] = field(default_factory=dict)
    success: bool = False


class SystemHealthMonitor:
    """Monitors overall system health and performance"""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.health_metrics = {
            "system_uptime": datetime.now(),
            "total_agents": 0,
            "active_agents": 0,
            "total_tasks_processed": 0,
            "average_system_response_time": 0.0,
            "error_rate": 0.0,
            "communication_throughput": 0.0,
            "resource_utilization": 0.0,
        }

        self.alerts = []
        self.performance_history = []

    async def update_health_metrics(self):
        """Update system-wide health metrics"""
        try:
            # Agent statistics
            agent_count = len(self.orchestrator.agents)
            active_count = len([a for a in self.orchestrator.agents.values() if a.status == AgentStatus.ACTIVE])

            # Task statistics
            total_tasks = sum(a.tasks_completed for a in self.orchestrator.agents.values())
            avg_response = sum(a.average_response_time for a in self.orchestrator.agents.values()) / max(agent_count, 1)

            # Communication statistics
            total_channels = len(self.orchestrator.communication_channels)
            active_messages = sum(
                len(ch.message_history) for ch in self.orchestrator.communication_channels.values() if ch.is_active
            )

            # Update metrics
            self.health_metrics.update(
                {
                    "total_agents": agent_count,
                    "active_agents": active_count,
                    "total_tasks_processed": total_tasks,
                    "average_system_response_time": avg_response,
                    "active_communication_channels": total_channels,
                    "recent_message_count": active_messages,
                    "last_updated": datetime.now().isoformat(),
                }
            )

            # Check for alerts
            await self._check_system_alerts()

        except Exception as e:
            logger.error(f"Health metrics update failed: {e}")

    async def _check_system_alerts(self):
        """Check for system-level alerts"""
        alerts = []

        # Check agent availability
        if self.health_metrics["active_agents"] < self.health_metrics["total_agents"] * 0.8:
            alerts.append(
                {
                    "type": "agent_availability",
                    "severity": "warning",
                    "message": f"Only {self.health_metrics['active_agents']}/{self.health_metrics['total_agents']} agents active",
                }
            )

        # Check response time
        if self.health_metrics["average_system_response_time"] > 5000:  # 5 seconds
            alerts.append(
                {
                    "type": "performance",
                    "severity": "warning",
                    "message": f"High system response time: {self.health_metrics['average_system_response_time']:.1f}ms",
                }
            )

        # Add new alerts
        for alert in alerts:
            if alert not in self.alerts[-10:]:  # Avoid duplicate recent alerts
                alert["timestamp"] = datetime.now().isoformat()
                self.alerts.append(alert)
                logger.warning(f"System alert: {alert}")


class AgentOrchestrationSystem:
    """Complete orchestration system for AIVillage specialized agents

    Provides unified management, communication, and coordination for all agents
    with integration to RAG, P2P, and Agent Forge systems.
    """

    def __init__(self):
        # Agent registry
        self.agents: dict[str, AgentRegistration] = {}
        self.agent_types: dict[str, list[str]] = defaultdict(list)  # type -> agent_ids
        self.capability_index: dict[AgentCapability, list[str]] = defaultdict(list)

        # Communication system
        self.communication_channels: dict[str, CommunicationChannel] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.message_handlers: dict[str, callable] = {}

        # Task coordination
        self.active_tasks: dict[str, MultiAgentTask] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.distribution_strategy = TaskDistributionStrategy.CAPABILITY_BASED

        # System connections (injected during initialization)
        self.rag_client = None
        self.p2p_client = None
        self.agent_forge_client = None

        # Monitoring and health
        self.health_monitor = SystemHealthMonitor(self)
        self.is_running = False
        self.background_tasks: list[asyncio.Task] = []

        # Statistics
        self.stats = {
            "agents_registered": 0,
            "tasks_distributed": 0,
            "messages_routed": 0,
            "multi_agent_collaborations": 0,
            "system_start_time": datetime.now(),
        }

        logger.info("Agent Orchestration System initialized")

    # Agent Registration and Lifecycle Management

    async def register_agent(self, agent: BaseAgentTemplate) -> bool:
        """Register a new agent with the orchestration system"""
        try:
            agent_id = agent.agent_id

            if agent_id in self.agents:
                logger.warning(f"Agent {agent_id} already registered, updating registration")

            # Get agent capabilities
            capabilities = await agent.get_specialized_capabilities()

            # Create registration
            registration = AgentRegistration(
                agent_id=agent_id,
                agent=agent,
                agent_type=agent.agent_type,
                specialized_role=getattr(agent, "specialized_role", "unknown"),
                capabilities=set(capabilities),
                status=AgentStatus.INITIALIZING,
            )

            # Store registration
            self.agents[agent_id] = registration
            self.agent_types[agent.agent_type].append(agent_id)

            # Index by capabilities
            for capability in capabilities:
                self.capability_index[capability].append(agent_id)

            # Set up agent connections
            agent.rag_client = self.rag_client
            agent.p2p_client = self.p2p_client
            agent.agent_forge_client = self.agent_forge_client

            # Initialize agent
            init_success = await agent.initialize()
            if init_success:
                registration.status = AgentStatus.ACTIVE
                self.stats["agents_registered"] += 1
                logger.info(f"Agent registered successfully: {agent_id} ({agent.agent_type})")
            else:
                registration.status = AgentStatus.ERROR
                logger.error(f"Agent initialization failed: {agent_id}")
                return False

            # Add agent to default communication channels
            await self._add_agent_to_default_channels(agent_id)

            return True

        except Exception as e:
            logger.error(f"Agent registration failed: {e}")
            return False

    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the orchestration system"""
        try:
            if agent_id not in self.agents:
                logger.warning(f"Agent {agent_id} not found for unregistration")
                return False

            registration = self.agents[agent_id]
            registration.status = AgentStatus.SHUTTING_DOWN

            # Shutdown the agent
            await registration.agent.shutdown()

            # Remove from indexes
            self.agent_types[registration.agent_type].remove(agent_id)
            for capability in registration.capabilities:
                if agent_id in self.capability_index[capability]:
                    self.capability_index[capability].remove(agent_id)

            # Remove from communication channels
            await self._remove_agent_from_channels(agent_id)

            # Remove from registry
            del self.agents[agent_id]

            logger.info(f"Agent unregistered: {agent_id}")
            return True

        except Exception as e:
            logger.error(f"Agent unregistration failed: {e}")
            return False

    async def get_agent_by_id(self, agent_id: str) -> BaseAgentTemplate | None:
        """Get agent instance by ID"""
        registration = self.agents.get(agent_id)
        return registration.agent if registration else None

    async def get_agents_by_type(self, agent_type: str) -> list[BaseAgentTemplate]:
        """Get all agents of a specific type"""
        agent_ids = self.agent_types.get(agent_type, [])
        return [self.agents[aid].agent for aid in agent_ids if aid in self.agents]

    async def get_agents_by_capability(self, capability: AgentCapability) -> list[BaseAgentTemplate]:
        """Get all agents with a specific capability"""
        agent_ids = self.capability_index.get(capability, [])
        return [
            self.agents[aid].agent
            for aid in agent_ids
            if aid in self.agents and self.agents[aid].status == AgentStatus.ACTIVE
        ]

    # Communication System

    async def create_communication_channel(
        self, channel_type: CommunicationChannelType, name: str, description: str = "", max_participants: int = 100
    ) -> str:
        """Create a new communication channel"""

        channel_id = f"{channel_type.value}_{name}_{uuid4().hex[:8]}"

        channel = CommunicationChannel(
            channel_id=channel_id,
            channel_type=channel_type,
            name=name,
            description=description,
            max_participants=max_participants,
        )

        self.communication_channels[channel_id] = channel
        logger.info(f"Communication channel created: {channel_id} ({channel_type.value})")

        return channel_id

    async def join_channel(self, agent_id: str, channel_id: str) -> bool:
        """Add agent to communication channel"""
        if channel_id not in self.communication_channels:
            return False

        channel = self.communication_channels[channel_id]
        if len(channel.participants) >= channel.max_participants:
            return False

        channel.participants.add(agent_id)

        if agent_id in self.agents:
            self.agents[agent_id].active_channels.append(channel_id)

        logger.debug(f"Agent {agent_id} joined channel {channel_id}")
        return True

    async def leave_channel(self, agent_id: str, channel_id: str) -> bool:
        """Remove agent from communication channel"""
        if channel_id not in self.communication_channels:
            return False

        channel = self.communication_channels[channel_id]
        channel.participants.discard(agent_id)

        if agent_id in self.agents and channel_id in self.agents[agent_id].active_channels:
            self.agents[agent_id].active_channels.remove(channel_id)

        logger.debug(f"Agent {agent_id} left channel {channel_id}")
        return True

    async def send_message(
        self,
        sender_id: str,
        channel_id: str | None = None,
        recipient_id: str | None = None,
        message: str = "",
        message_type: str = "general",
        priority: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Send message through orchestration system"""

        message_data = {
            "message_id": str(uuid4()),
            "sender_id": sender_id,
            "channel_id": channel_id,
            "recipient_id": recipient_id,
            "message": message,
            "message_type": message_type,
            "priority": priority,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }

        # Add to message queue for processing
        await self.message_queue.put(message_data)
        return True

    async def _process_message_queue(self):
        """Process incoming messages"""
        while self.is_running:
            try:
                # Wait for messages
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)

                await self._route_message(message)
                self.stats["messages_routed"] += 1

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Message processing error: {e}")

    async def _route_message(self, message: dict[str, Any]):
        """Route message to appropriate recipients"""
        sender_id = message["sender_id"]
        channel_id = message.get("channel_id")
        recipient_id = message.get("recipient_id")

        recipients = []

        if recipient_id:
            # Direct message
            recipients = [recipient_id]
        elif channel_id and channel_id in self.communication_channels:
            # Channel message
            channel = self.communication_channels[channel_id]
            recipients = list(channel.participants)

            # Store in channel history
            channel.message_history.append(message)

            # Cleanup old messages
            if len(channel.message_history) > 1000:
                channel.message_history = channel.message_history[-500:]

        # Deliver to recipients
        for recipient_id in recipients:
            if recipient_id != sender_id and recipient_id in self.agents:
                await self._deliver_message_to_agent(recipient_id, message)

    async def _deliver_message_to_agent(self, agent_id: str, message: dict[str, Any]):
        """Deliver message to specific agent"""
        try:
            registration = self.agents[agent_id]

            if registration.status != AgentStatus.ACTIVE:
                return

            # Create MessageInterface
            msg_interface = MessageInterface(
                message_id=message["message_id"],
                sender=message["sender_id"],
                receiver=agent_id,
                message_type=message["message_type"],
                content=message["message"],
                priority=message["priority"],
                metadata=message.get("metadata", {}),
            )

            # Deliver to agent
            await registration.agent.receive_message(msg_interface)
            registration.message_queue_depth = max(0, registration.message_queue_depth - 1)

        except Exception as e:
            logger.error(f"Message delivery to {agent_id} failed: {e}")

    # Task Distribution and Coordination

    async def distribute_task(self, task: TaskInterface, strategy: TaskDistributionStrategy | None = None) -> str:
        """Distribute task to appropriate agent(s)"""

        strategy = strategy or self.distribution_strategy

        # Find suitable agents based on task requirements
        suitable_agents = await self._find_suitable_agents(task)

        if not suitable_agents:
            logger.warning(f"No suitable agents found for task: {task.task_id}")
            return task.task_id

        # Select agent(s) based on distribution strategy
        selected_agents = await self._select_agents_by_strategy(suitable_agents, strategy)

        if not selected_agents:
            logger.error(f"Agent selection failed for task: {task.task_id}")
            return task.task_id

        # Distribute to selected agent(s)
        for agent_id in selected_agents:
            if agent_id in self.agents:
                registration = self.agents[agent_id]

                try:
                    # Send task to agent
                    asyncio.create_task(self._execute_task_on_agent(registration, task))
                    registration.current_load += 0.1  # Approximate load increase

                except Exception as e:
                    logger.error(f"Task distribution to {agent_id} failed: {e}")

        self.stats["tasks_distributed"] += 1
        return task.task_id

    async def coordinate_multi_agent_task(
        self,
        description: str,
        required_capabilities: list[AgentCapability],
        max_agents: int = 5,
        coordination_strategy: str = "collaborative",
    ) -> str:
        """Coordinate multi-agent collaborative task"""

        task_id = f"multi_agent_{uuid4().hex[:8]}"

        # Create multi-agent task
        multi_task = MultiAgentTask(
            task_id=task_id,
            description=description,
            requester_id="orchestration_system",
            required_capabilities=required_capabilities,
            max_agents=max_agents,
            coordination_strategy=coordination_strategy,
        )

        # Find agents for each required capability
        assigned_agents = []
        for capability in required_capabilities:
            capable_agents = await self.get_agents_by_capability(capability)

            if capable_agents:
                # Select best available agent
                best_agent = await self._select_best_agent(capable_agents)
                if best_agent and best_agent.agent_id not in assigned_agents:
                    assigned_agents.append(best_agent.agent_id)

                    if len(assigned_agents) >= max_agents:
                        break

        multi_task.assigned_agents = assigned_agents
        multi_task.task_status = "assigned"
        multi_task.started_at = datetime.now()

        self.active_tasks[task_id] = multi_task

        # Create coordination channel
        coord_channel_id = await self.create_communication_channel(
            CommunicationChannelType.COORDINATION,
            f"task_{task_id}",
            f"Coordination channel for multi-agent task: {description[:50]}",
        )

        # Add assigned agents to coordination channel
        for agent_id in assigned_agents:
            await self.join_channel(agent_id, coord_channel_id)

        # Send coordination message
        await self.send_message(
            sender_id="orchestration_system",
            channel_id=coord_channel_id,
            message=f"Multi-agent task coordination: {description}",
            message_type="coordination",
            priority=3,
            metadata={
                "task_id": task_id,
                "coordination_strategy": coordination_strategy,
                "required_capabilities": [cap.value for cap in required_capabilities],
            },
        )

        self.stats["multi_agent_collaborations"] += 1
        logger.info(f"Multi-agent task coordinated: {task_id} with {len(assigned_agents)} agents")

        return task_id

    async def _find_suitable_agents(self, task: TaskInterface) -> list[str]:
        """Find agents suitable for a task"""
        suitable_agents = []

        for agent_id, registration in self.agents.items():
            if registration.status == AgentStatus.ACTIVE:
                try:
                    can_handle = await registration.agent.can_handle_task(task)
                    if can_handle:
                        suitable_agents.append(agent_id)
                except Exception as e:
                    logger.error(f"Error checking if {agent_id} can handle task: {e}")

        return suitable_agents

    async def _select_agents_by_strategy(self, agent_ids: list[str], strategy: TaskDistributionStrategy) -> list[str]:
        """Select agents based on distribution strategy"""

        if not agent_ids:
            return []

        if strategy == TaskDistributionStrategy.ROUND_ROBIN:
            # Simple round-robin selection
            return [agent_ids[self.stats["tasks_distributed"] % len(agent_ids)]]

        elif strategy == TaskDistributionStrategy.LOAD_BALANCED:
            # Select agent with lowest current load
            agents_with_load = [(agent_id, self.agents[agent_id].current_load) for agent_id in agent_ids]
            agents_with_load.sort(key=lambda x: x[1])
            return [agents_with_load[0][0]]

        elif strategy == TaskDistributionStrategy.CAPABILITY_BASED:
            # Select best agent based on success rate and performance
            best_agent = None
            best_score = -1

            for agent_id in agent_ids:
                registration = self.agents[agent_id]
                score = registration.success_rate * registration.health_score

                if score > best_score:
                    best_score = score
                    best_agent = agent_id

            return [best_agent] if best_agent else []

        else:
            # Default: return first available
            return [agent_ids[0]]

    async def _select_best_agent(self, agents: list[BaseAgentTemplate]) -> BaseAgentTemplate | None:
        """Select best agent from list based on performance metrics"""
        if not agents:
            return None

        best_agent = None
        best_score = -1

        for agent in agents:
            registration = self.agents.get(agent.agent_id)
            if not registration or registration.status != AgentStatus.ACTIVE:
                continue

            # Calculate composite score
            score = (
                registration.success_rate * 0.4
                + registration.health_score * 0.3
                + (1.0 - registration.current_load) * 0.3
            )

            if score > best_score:
                best_score = score
                best_agent = agent

        return best_agent

    async def _execute_task_on_agent(self, registration: AgentRegistration, task: TaskInterface):
        """Execute task on specific agent with performance tracking"""
        start_time = datetime.now()

        try:
            # Execute task
            result = await registration.agent.process_task(task)

            # Update performance metrics
            duration = (datetime.now() - start_time).total_seconds()
            registration.tasks_completed += 1
            registration.last_activity = datetime.now()

            # Update average response time
            if registration.average_response_time == 0:
                registration.average_response_time = duration * 1000
            else:
                registration.average_response_time = registration.average_response_time * 0.8 + duration * 1000 * 0.2

            # Update success rate
            success = result.get("status") == "success"
            total_tasks = registration.tasks_completed
            registration.success_rate = (
                registration.success_rate * (total_tasks - 1) + (1.0 if success else 0.0)
            ) / total_tasks

            registration.current_load = max(0, registration.current_load - 0.1)

        except Exception as e:
            logger.error(f"Task execution failed on {registration.agent_id}: {e}")
            registration.error_count += 1
            registration.current_load = max(0, registration.current_load - 0.1)

    # System Management

    async def start_orchestration(self):
        """Start the orchestration system"""
        if self.is_running:
            logger.warning("Orchestration system already running")
            return

        self.is_running = True

        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._process_message_queue()),
            asyncio.create_task(self._periodic_health_checks()),
            asyncio.create_task(self._periodic_cleanup()),
            asyncio.create_task(self._update_system_metrics()),
        ]

        # Create default communication channels
        await self._create_default_channels()

        logger.info("Agent Orchestration System started")

    async def stop_orchestration(self):
        """Stop the orchestration system"""
        if not self.is_running:
            return

        self.is_running = False

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()

        # Shutdown all agents
        for agent_id in list(self.agents.keys()):
            await self.unregister_agent(agent_id)

        logger.info("Agent Orchestration System stopped")

    async def _create_default_channels(self):
        """Create default communication channels"""
        default_channels = [
            (CommunicationChannelType.BROADCAST, "general", "General broadcast channel"),
            (CommunicationChannelType.EMERGENCY, "emergency", "Emergency coordination channel"),
            (CommunicationChannelType.GROUP, "governance", "Governance agents coordination"),
            (CommunicationChannelType.GROUP, "infrastructure", "Infrastructure agents coordination"),
            (CommunicationChannelType.GROUP, "knowledge", "Knowledge agents coordination"),
        ]

        for channel_type, name, description in default_channels:
            await self.create_communication_channel(channel_type, name, description)

    async def _add_agent_to_default_channels(self, agent_id: str):
        """Add agent to appropriate default channels"""
        # Add to general broadcast
        for channel_id, channel in self.communication_channels.items():
            if channel.name == "general" and channel.channel_type == CommunicationChannelType.BROADCAST:
                await self.join_channel(agent_id, channel_id)
                break

        # Add to type-specific group channels
        if agent_id in self.agents:
            agent_type = self.agents[agent_id].agent_type.lower()

            for channel_id, channel in self.communication_channels.items():
                if channel.channel_type == CommunicationChannelType.GROUP and channel.name.lower() in agent_type:
                    await self.join_channel(agent_id, channel_id)

    async def _remove_agent_from_channels(self, agent_id: str):
        """Remove agent from all communication channels"""
        for channel_id in list(self.communication_channels.keys()):
            await self.leave_channel(agent_id, channel_id)

    async def _periodic_health_checks(self):
        """Perform periodic health checks on all agents"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                for agent_id, registration in self.agents.items():
                    try:
                        health_info = await registration.agent.health_check()

                        # Update health metrics
                        registration.last_health_check = datetime.now()
                        registration.health_score = self._calculate_health_score(health_info)

                        # Update status based on health
                        if registration.health_score < 0.3:
                            registration.status = AgentStatus.ERROR
                        elif registration.health_score < 0.7:
                            registration.status = AgentStatus.MAINTENANCE
                        else:
                            registration.status = AgentStatus.ACTIVE

                    except Exception as e:
                        logger.error(f"Health check failed for {agent_id}: {e}")
                        registration.error_count += 1
                        registration.health_score *= 0.9  # Degrade health score

                # Update system health
                await self.health_monitor.update_health_metrics()

            except Exception as e:
                logger.error(f"Periodic health check error: {e}")

    def _calculate_health_score(self, health_info: dict[str, Any]) -> float:
        """Calculate health score from health check information"""
        try:
            # Base score
            score = 1.0

            # Check connections
            connections = health_info.get("connections", {})
            connected_count = sum(1 for connected in connections.values() if connected)
            connection_ratio = connected_count / max(len(connections), 1)
            score *= connection_ratio

            # Check performance
            performance = health_info.get("performance", {})
            recent_perf = performance.get("recent_performance", {})

            if "error_rate" in recent_perf:
                error_rate = recent_perf["error_rate"]
                score *= 1.0 - min(error_rate, 0.5)  # Cap error rate impact

            # Check geometric state if available
            geometric_state = health_info.get("geometric_state", {})
            if not geometric_state.get("is_healthy", True):
                score *= 0.8

            return max(0.0, min(1.0, score))

        except Exception:
            return 0.5  # Default moderate health if calculation fails

    async def _periodic_cleanup(self):
        """Perform periodic cleanup of old data"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Cleanup every hour

                # Cleanup old messages in channels
                for channel in self.communication_channels.values():
                    cutoff_time = datetime.now() - timedelta(hours=channel.message_retention_hours)

                    channel.message_history = [
                        msg for msg in channel.message_history if datetime.fromisoformat(msg["timestamp"]) > cutoff_time
                    ]

                # Cleanup completed multi-agent tasks
                completed_tasks = [
                    task_id
                    for task_id, task in self.active_tasks.items()
                    if task.task_status == "completed"
                    and task.completed_at
                    and task.completed_at < datetime.now() - timedelta(hours=24)
                ]

                for task_id in completed_tasks:
                    del self.active_tasks[task_id]

                logger.debug(f"Cleanup completed: removed {len(completed_tasks)} old tasks")

            except Exception as e:
                logger.error(f"Periodic cleanup error: {e}")

    async def _update_system_metrics(self):
        """Update system-wide metrics"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Update every minute

                # Update runtime statistics
                current_time = datetime.now()
                uptime = (current_time - self.stats["system_start_time"]).total_seconds()

                self.stats.update(
                    {
                        "system_uptime_seconds": uptime,
                        "active_agents": len([a for a in self.agents.values() if a.status == AgentStatus.ACTIVE]),
                        "total_registered_agents": len(self.agents),
                        "active_communication_channels": len(
                            [c for c in self.communication_channels.values() if c.is_active]
                        ),
                        "active_multi_agent_tasks": len(self.active_tasks),
                        "last_metrics_update": current_time.isoformat(),
                    }
                )

            except Exception as e:
                logger.error(f"System metrics update error: {e}")

    # Public API for system status and control

    async def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "orchestration_system": {
                "is_running": self.is_running,
                "uptime_seconds": (datetime.now() - self.stats["system_start_time"]).total_seconds(),
                "statistics": self.stats,
            },
            "agents": {
                "total_registered": len(self.agents),
                "by_status": {
                    status.value: len([a for a in self.agents.values() if a.status == status]) for status in AgentStatus
                },
                "by_type": {agent_type: len(agent_list) for agent_type, agent_list in self.agent_types.items()},
                "capability_coverage": {
                    capability.value: len(agent_list) for capability, agent_list in self.capability_index.items()
                },
            },
            "communication": {
                "active_channels": len([c for c in self.communication_channels.values() if c.is_active]),
                "total_channels": len(self.communication_channels),
                "messages_in_queue": self.message_queue.qsize(),
                "recent_message_count": sum(len(c.message_history) for c in self.communication_channels.values()),
            },
            "tasks": {
                "active_multi_agent_tasks": len(self.active_tasks),
                "tasks_in_queue": self.task_queue.qsize(),
                "distribution_strategy": self.distribution_strategy.value,
            },
            "health": self.health_monitor.health_metrics,
            "alerts": self.health_monitor.alerts[-5:],  # Last 5 alerts
        }

    async def get_agent_status(self, agent_id: str) -> dict[str, Any] | None:
        """Get detailed status for specific agent"""
        if agent_id not in self.agents:
            return None

        registration = self.agents[agent_id]

        return {
            "agent_id": agent_id,
            "agent_type": registration.agent_type,
            "specialized_role": registration.specialized_role,
            "status": registration.status.value,
            "capabilities": [cap.value for cap in registration.capabilities],
            "performance": {
                "tasks_completed": registration.tasks_completed,
                "average_response_time_ms": registration.average_response_time,
                "success_rate": registration.success_rate,
                "current_load": registration.current_load,
                "health_score": registration.health_score,
                "error_count": registration.error_count,
            },
            "communication": {
                "active_channels": len(registration.active_channels),
                "message_queue_depth": registration.message_queue_depth,
            },
            "last_activity": registration.last_activity.isoformat(),
            "last_health_check": registration.last_health_check.isoformat(),
        }


# Factory function for easy instantiation
async def create_orchestration_system() -> AgentOrchestrationSystem:
    """Create and initialize the Agent Orchestration System"""

    orchestrator = AgentOrchestrationSystem()

    # System connections would be injected here in production
    # orchestrator.rag_client = get_rag_client()
    # orchestrator.p2p_client = get_p2p_client()
    # orchestrator.agent_forge_client = get_agent_forge_client()

    await orchestrator.start_orchestration()

    return orchestrator


# Export main classes and functions
__all__ = [
    "AgentOrchestrationSystem",
    "create_orchestration_system",
    "AgentRegistration",
    "CommunicationChannel",
    "MultiAgentTask",
    "AgentStatus",
    "CommunicationChannelType",
    "TaskDistributionStrategy",
]
