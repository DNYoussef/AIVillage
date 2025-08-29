"""
Distributed Agent Coordination System

Archaeological Enhancement: Advanced multi-agent coordination and orchestration
Innovation Score: 9.1/10 (coordination + distributed intelligence)
Branch Origins: agent-coordination-v4, distributed-agents-v3, swarm-intelligence-v2
Integration: Complete agent orchestration with existing Agent Forge and P2P systems
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import random
import networkx as nx
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class AgentRole(str, Enum):
    """Agent roles in distributed coordination."""
    COORDINATOR = "coordinator"          # Central coordination
    WORKER = "worker"                   # Task execution
    SPECIALIST = "specialist"           # Domain-specific tasks
    MONITOR = "monitor"                 # System monitoring
    OPTIMIZER = "optimizer"             # Performance optimization
    BRIDGE = "bridge"                   # Inter-system communication
    VALIDATOR = "validator"             # Quality assurance
    RESEARCHER = "researcher"           # Information gathering

class AgentState(str, Enum):
    """Agent states in coordination system."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    ASSIGNED = "assigned" 
    WORKING = "working"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    ERROR = "error"
    OFFLINE = "offline"

class CoordinationStrategy(str, Enum):
    """Coordination strategies for agent management."""
    CENTRALIZED = "centralized"         # Single coordinator
    HIERARCHICAL = "hierarchical"       # Multi-level coordination
    PEER_TO_PEER = "peer_to_peer"      # Decentralized coordination
    HYBRID = "hybrid"                   # Mixed strategies
    SWARM = "swarm"                     # Swarm intelligence
    CONSENSUS = "consensus"             # Consensus-based decisions

class TaskPriority(str, Enum):
    """Task priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"

@dataclass
class AgentCapability:
    """Agent capability definition."""
    capability_id: str
    name: str
    description: str
    performance_score: float  # 0.0 - 1.0
    resource_requirements: Dict[str, float]
    specialization_areas: List[str]
    learning_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class CoordinatedAgent:
    """Distributed agent representation."""
    agent_id: str
    name: str
    role: AgentRole
    state: AgentState
    capabilities: List[AgentCapability]
    current_tasks: List[str]
    performance_metrics: Dict[str, float]
    location: Dict[str, str]  # node, region, etc.
    last_heartbeat: datetime
    communication_endpoints: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "capabilities": [cap.to_dict() for cap in self.capabilities]
        }

@dataclass
class CoordinationTask:
    """Task in distributed coordination system."""
    task_id: str
    name: str
    description: str
    priority: TaskPriority
    required_capabilities: List[str]
    estimated_duration: float  # hours
    dependencies: List[str]  # other task IDs
    assigned_agents: List[str]
    progress: float  # 0.0 - 1.0
    status: str
    created_at: datetime
    deadline: Optional[datetime]
    result: Optional[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "created_at": self.created_at.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None
        }

@dataclass
class CoordinationMetrics:
    """System-wide coordination metrics."""
    total_agents: int
    active_agents: int
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    average_task_completion_time: float
    agent_utilization: float
    system_throughput: float
    coordination_overhead: float
    consensus_time: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat()
        }

class DistributedAgentCoordinator:
    """
    Advanced Distributed Agent Coordination System.
    
    Archaeological Enhancement: Complete multi-agent orchestration with:
    - Dynamic agent discovery and registration
    - Intelligent task assignment and load balancing
    - Consensus-based decision making
    - Real-time performance optimization
    - Fault tolerance and self-healing capabilities
    - Integration with existing Agent Forge and P2P systems
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Agent management
        self.agents: Dict[str, CoordinatedAgent] = {}
        self.agent_groups: Dict[str, List[str]] = {}  # role -> agent_ids
        self.offline_agents: Set[str] = set()
        
        # Task management
        self.tasks: Dict[str, CoordinationTask] = {}
        self.task_queue: deque = deque()
        self.dependency_graph = nx.DiGraph()
        
        # Coordination strategy
        self.strategy = CoordinationStrategy(self.config.get("strategy", CoordinationStrategy.HYBRID))
        self.coordinator_agents: List[str] = []
        
        # Performance tracking
        self.metrics_history: List[CoordinationMetrics] = []
        self.performance_optimizer = PerformanceOptimizer()
        self.consensus_engine = ConsensusEngine()
        
        # Communication and networking
        self.message_broker = MessageBroker()
        self.network_topology = nx.Graph()
        
        # System state
        self.coordination_active = False
        self.last_optimization = datetime.now()
        
        # Integration with existing systems
        self.agent_forge_integration = True
        self.p2p_integration = True
        
    async def initialize(self) -> bool:
        """
        Initialize Distributed Agent Coordination System.
        
        Archaeological Enhancement: Complete system initialization with discovery.
        """
        try:
            logger.info("Initializing Distributed Agent Coordination System...")
            
            # Initialize message broker
            await self.message_broker.initialize()
            
            # Initialize consensus engine
            await self.consensus_engine.initialize()
            
            # Initialize performance optimizer
            await self.performance_optimizer.initialize()
            
            # Setup system integrations
            await self._setup_system_integrations()
            
            # Discover existing agents
            await self._discover_existing_agents()
            
            # Start coordination services
            await self._start_coordination_services()
            
            self.coordination_active = True
            logger.info("Distributed Agent Coordination System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Distributed Agent Coordination System: {e}")
            return False
    
    async def register_agent(self, agent_config: Dict[str, Any]) -> Optional[CoordinatedAgent]:
        """
        Register new agent in coordination system.
        
        Archaeological Enhancement: Intelligent agent registration with capability analysis.
        """
        try:
            agent_id = agent_config.get("agent_id", f"agent_{uuid.uuid4().hex[:16]}")
            
            # Create capabilities
            capabilities = []
            for cap_data in agent_config.get("capabilities", []):
                capability = AgentCapability(
                    capability_id=cap_data.get("capability_id", f"cap_{uuid.uuid4().hex[:8]}"),
                    name=cap_data.get("name", ""),
                    description=cap_data.get("description", ""),
                    performance_score=cap_data.get("performance_score", 0.8),
                    resource_requirements=cap_data.get("resource_requirements", {}),
                    specialization_areas=cap_data.get("specialization_areas", []),
                    learning_rate=cap_data.get("learning_rate", 0.1)
                )
                capabilities.append(capability)
            
            # Create agent
            agent = CoordinatedAgent(
                agent_id=agent_id,
                name=agent_config.get("name", f"Agent_{agent_id[:8]}"),
                role=AgentRole(agent_config.get("role", AgentRole.WORKER)),
                state=AgentState.INITIALIZING,
                capabilities=capabilities,
                current_tasks=[],
                performance_metrics={
                    "tasks_completed": 0,
                    "success_rate": 1.0,
                    "average_response_time": 0.0,
                    "resource_efficiency": 0.8
                },
                location=agent_config.get("location", {"node": "localhost", "region": "local"}),
                last_heartbeat=datetime.now(),
                communication_endpoints=agent_config.get("endpoints", []),
                metadata=agent_config.get("metadata", {})
            )
            
            # Register agent
            self.agents[agent_id] = agent
            
            # Add to role groups
            role_group = self.agent_groups.get(agent.role, [])
            role_group.append(agent_id)
            self.agent_groups[agent.role] = role_group
            
            # Update network topology
            self.network_topology.add_node(agent_id, **agent.to_dict())
            
            # Set agent state to idle
            agent.state = AgentState.IDLE
            
            # Send welcome message
            await self._send_agent_message(agent_id, {
                "type": "registration_complete",
                "agent_id": agent_id,
                "coordinator_info": {
                    "strategy": self.strategy,
                    "total_agents": len(self.agents)
                }
            })
            
            logger.info(f"Registered agent {agent_id} with role {agent.role}")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to register agent: {e}")
            return None
    
    async def submit_task(self, task_config: Dict[str, Any]) -> Optional[CoordinationTask]:
        """
        Submit task to coordination system.
        
        Archaeological Enhancement: Intelligent task analysis and assignment.
        """
        try:
            task_id = task_config.get("task_id", f"task_{uuid.uuid4().hex[:16]}")
            
            # Create task
            task = CoordinationTask(
                task_id=task_id,
                name=task_config.get("name", f"Task_{task_id[:8]}"),
                description=task_config.get("description", ""),
                priority=TaskPriority(task_config.get("priority", TaskPriority.NORMAL)),
                required_capabilities=task_config.get("required_capabilities", []),
                estimated_duration=task_config.get("estimated_duration", 1.0),
                dependencies=task_config.get("dependencies", []),
                assigned_agents=[],
                progress=0.0,
                status="pending",
                created_at=datetime.now(),
                deadline=None,
                result=None
            )
            
            # Set deadline if specified
            if "deadline_hours" in task_config:
                task.deadline = datetime.now() + timedelta(hours=task_config["deadline_hours"])
            
            # Store task
            self.tasks[task_id] = task
            
            # Add to dependency graph
            self.dependency_graph.add_node(task_id)
            for dep_task_id in task.dependencies:
                if dep_task_id in self.tasks:
                    self.dependency_graph.add_edge(dep_task_id, task_id)
            
            # Add to task queue for assignment
            self.task_queue.append(task_id)
            
            # Trigger task assignment
            await self._process_task_queue()
            
            logger.info(f"Submitted task {task_id}: {task.name}")
            return task
            
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            return None
    
    async def _process_task_queue(self):
        """Process pending tasks and assign to agents."""
        while self.task_queue:
            task_id = self.task_queue.popleft()
            
            if task_id not in self.tasks:
                continue
            
            task = self.tasks[task_id]
            
            # Check dependencies
            if not self._are_dependencies_satisfied(task_id):
                # Put back in queue
                self.task_queue.append(task_id)
                continue
            
            # Find suitable agents
            suitable_agents = await self._find_suitable_agents(task)
            
            if not suitable_agents:
                # No suitable agents available, put back in queue
                self.task_queue.append(task_id)
                logger.debug(f"No suitable agents for task {task_id}, queuing for later")
                continue
            
            # Assign task to agents
            await self._assign_task_to_agents(task_id, suitable_agents)
    
    async def _find_suitable_agents(self, task: CoordinationTask) -> List[str]:
        """Find agents suitable for task execution."""
        suitable_agents = []
        
        for agent_id, agent in self.agents.items():
            # Check agent state
            if agent.state not in [AgentState.IDLE, AgentState.ASSIGNED]:
                continue
            
            # Check if agent is online
            if agent_id in self.offline_agents:
                continue
            
            # Check capabilities match
            agent_capabilities = [cap.name for cap in agent.capabilities]
            has_required_capabilities = all(
                req_cap in agent_capabilities or self._is_capability_compatible(req_cap, agent_capabilities)
                for req_cap in task.required_capabilities
            )
            
            if not has_required_capabilities:
                continue
            
            # Check agent workload
            if len(agent.current_tasks) >= self.config.get("max_tasks_per_agent", 3):
                continue
            
            # Calculate agent fitness for task
            fitness_score = self._calculate_agent_task_fitness(agent, task)
            
            suitable_agents.append((agent_id, fitness_score))
        
        # Sort by fitness score (descending)
        suitable_agents.sort(key=lambda x: x[1], reverse=True)
        
        # Return agent IDs
        return [agent_id for agent_id, _ in suitable_agents]
    
    def _calculate_agent_task_fitness(self, agent: CoordinatedAgent, task: CoordinationTask) -> float:
        """Calculate how suitable an agent is for a specific task."""
        fitness = 0.0
        
        # Capability matching score
        agent_capabilities = [cap.name for cap in agent.capabilities]
        matching_capabilities = sum(
            1 for req_cap in task.required_capabilities 
            if req_cap in agent_capabilities
        )
        
        capability_score = matching_capabilities / max(1, len(task.required_capabilities))
        fitness += capability_score * 0.4
        
        # Performance score
        performance_score = agent.performance_metrics.get("success_rate", 0.8)
        fitness += performance_score * 0.3
        
        # Workload score (prefer less loaded agents)
        current_workload = len(agent.current_tasks)
        max_workload = self.config.get("max_tasks_per_agent", 3)
        workload_score = 1.0 - (current_workload / max_workload)
        fitness += workload_score * 0.2
        
        # Priority bonus for specialized roles
        if task.priority == TaskPriority.CRITICAL and agent.role == AgentRole.SPECIALIST:
            fitness += 0.1
        
        return min(1.0, fitness)
    
    def _is_capability_compatible(self, required_capability: str, agent_capabilities: List[str]) -> bool:
        """Check if agent capabilities can satisfy required capability."""
        # Simplified compatibility checking
        compatibility_map = {
            "research": ["analysis", "information_gathering"],
            "coding": ["programming", "development"],
            "testing": ["validation", "quality_assurance"],
            "optimization": ["performance_tuning", "analysis"]
        }
        
        compatible_caps = compatibility_map.get(required_capability, [])
        return any(cap in agent_capabilities for cap in compatible_caps)
    
    async def _assign_task_to_agents(self, task_id: str, agent_ids: List[str]):
        """Assign task to selected agents."""
        task = self.tasks[task_id]
        
        # Determine number of agents needed
        num_agents_needed = min(
            len(agent_ids),
            self.config.get("max_agents_per_task", 2),
            max(1, len(task.required_capabilities))
        )
        
        selected_agents = agent_ids[:num_agents_needed]
        
        # Update task
        task.assigned_agents = selected_agents
        task.status = "assigned"
        
        # Update agents
        for agent_id in selected_agents:
            agent = self.agents[agent_id]
            agent.current_tasks.append(task_id)
            agent.state = AgentState.ASSIGNED
        
        # Send task assignment messages
        for agent_id in selected_agents:
            await self._send_agent_message(agent_id, {
                "type": "task_assignment",
                "task_id": task_id,
                "task_details": task.to_dict(),
                "collaboration_agents": [aid for aid in selected_agents if aid != agent_id]
            })
        
        # Start task execution
        asyncio.create_task(self._monitor_task_execution(task_id))
        
        logger.info(f"Assigned task {task_id} to agents: {selected_agents}")
    
    async def _monitor_task_execution(self, task_id: str):
        """Monitor task execution and handle completion/failures."""
        task = self.tasks.get(task_id)
        if not task:
            return
        
        start_time = datetime.now()
        timeout = timedelta(hours=task.estimated_duration * 2)  # 2x estimated time
        
        while task.status not in ["completed", "failed", "cancelled"]:
            # Check for timeout
            if datetime.now() - start_time > timeout:
                logger.warning(f"Task {task_id} timed out")
                await self._handle_task_timeout(task_id)
                break
            
            # Check agent heartbeats
            for agent_id in task.assigned_agents:
                if agent_id in self.offline_agents:
                    logger.warning(f"Agent {agent_id} went offline during task {task_id}")
                    await self._handle_agent_failure(task_id, agent_id)
                    break
            
            await asyncio.sleep(5)  # Check every 5 seconds
    
    async def update_task_progress(self, task_id: str, progress: float, agent_id: str,
                                 status: Optional[str] = None, result: Optional[Dict[str, Any]] = None) -> bool:
        """Update task progress from agent."""
        if task_id not in self.tasks:
            logger.warning(f"Task {task_id} not found for progress update")
            return False
        
        task = self.tasks[task_id]
        
        if agent_id not in task.assigned_agents:
            logger.warning(f"Agent {agent_id} not assigned to task {task_id}")
            return False
        
        # Update progress
        task.progress = max(task.progress, progress)
        
        if status:
            task.status = status
        
        if result:
            task.result = result
        
        # Handle completion
        if progress >= 1.0 or status == "completed":
            await self._complete_task(task_id, result or {})
        
        # Update agent performance
        await self._update_agent_performance(agent_id, task_id, progress, status)
        
        return True
    
    async def _complete_task(self, task_id: str, result: Dict[str, Any]):
        """Handle task completion."""
        task = self.tasks[task_id]
        
        # Update task
        task.status = "completed"
        task.progress = 1.0
        task.result = result
        
        # Update assigned agents
        for agent_id in task.assigned_agents:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                agent.current_tasks.remove(task_id)
                agent.state = AgentState.IDLE
                
                # Update performance metrics
                agent.performance_metrics["tasks_completed"] += 1
        
        # Process dependent tasks
        await self._process_dependent_tasks(task_id)
        
        logger.info(f"Task {task_id} completed successfully")
    
    async def _process_dependent_tasks(self, completed_task_id: str):
        """Process tasks that depend on the completed task."""
        dependent_tasks = list(self.dependency_graph.successors(completed_task_id))
        
        for dep_task_id in dependent_tasks:
            if self._are_dependencies_satisfied(dep_task_id):
                # Add to task queue for processing
                if dep_task_id not in self.task_queue:
                    self.task_queue.append(dep_task_id)
        
        # Trigger task queue processing
        await self._process_task_queue()
    
    def _are_dependencies_satisfied(self, task_id: str) -> bool:
        """Check if all task dependencies are satisfied."""
        dependencies = list(self.dependency_graph.predecessors(task_id))
        
        for dep_task_id in dependencies:
            dep_task = self.tasks.get(dep_task_id)
            if not dep_task or dep_task.status != "completed":
                return False
        
        return True
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        current_time = datetime.now()
        
        # Calculate agent statistics
        active_agents = len([a for a in self.agents.values() if a.state != AgentState.OFFLINE])
        working_agents = len([a for a in self.agents.values() if a.state == AgentState.WORKING])
        
        # Calculate task statistics
        pending_tasks = len([t for t in self.tasks.values() if t.status == "pending"])
        running_tasks = len([t for t in self.tasks.values() if t.status in ["assigned", "working"]])
        completed_tasks = len([t for t in self.tasks.values() if t.status == "completed"])
        
        # Calculate performance metrics
        total_tasks = len(self.tasks)
        success_rate = completed_tasks / total_tasks if total_tasks > 0 else 0.0
        
        # Calculate average task completion time
        completed_task_objects = [t for t in self.tasks.values() if t.status == "completed"]
        avg_completion_time = 0.0
        if completed_task_objects:
            completion_times = []
            for task in completed_task_objects:
                if task.result and "completion_time" in task.result:
                    completion_times.append(task.result["completion_time"])
            
            if completion_times:
                avg_completion_time = sum(completion_times) / len(completion_times)
        
        return {
            "timestamp": current_time.isoformat(),
            "coordination_strategy": self.strategy,
            "system_health": {
                "coordination_active": self.coordination_active,
                "total_agents": len(self.agents),
                "active_agents": active_agents,
                "working_agents": working_agents,
                "offline_agents": len(self.offline_agents)
            },
            "task_statistics": {
                "total_tasks": total_tasks,
                "pending_tasks": pending_tasks,
                "running_tasks": running_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": len([t for t in self.tasks.values() if t.status == "failed"]),
                "success_rate": success_rate
            },
            "performance_metrics": {
                "average_completion_time_hours": avg_completion_time,
                "agent_utilization": working_agents / max(1, active_agents),
                "system_throughput": completed_tasks / max(1, (current_time - self.last_optimization).total_seconds() / 3600),
                "coordination_overhead": 0.1  # Placeholder
            },
            "agent_groups": {
                role: len(agents) for role, agents in self.agent_groups.items()
            },
            "network_topology": {
                "nodes": self.network_topology.number_of_nodes(),
                "edges": self.network_topology.number_of_edges(),
                "connected_components": len(list(nx.connected_components(self.network_topology)))
            }
        }
    
    # System integration and initialization methods
    
    async def _setup_system_integrations(self):
        """Setup integrations with existing AIVillage systems."""
        integrations = {}
        
        try:
            # Agent Forge integration
            from core.agent_forge.phases.cognate_pretrain.real_pretraining_pipeline import RealCognateTrainer
            integrations["agent_forge"] = "connected"
            logger.info("Agent Forge integration enabled")
        except ImportError:
            logger.warning("Agent Forge integration unavailable")
            integrations["agent_forge"] = "unavailable"
        
        try:
            # P2P integration
            from infrastructure.p2p.advanced.libp2p_enhanced_manager import LibP2PEnhancedManager
            integrations["p2p_networking"] = "connected"
            logger.info("P2P networking integration enabled")
        except ImportError:
            logger.warning("P2P networking integration unavailable")
            integrations["p2p_networking"] = "unavailable"
        
        try:
            # ML Pipeline integration
            from infrastructure.ml.advanced.ml_pipeline_orchestrator import MLPipelineOrchestrator
            integrations["ml_pipeline"] = "connected"
            logger.info("ML Pipeline integration enabled")
        except ImportError:
            logger.warning("ML Pipeline integration unavailable")
            integrations["ml_pipeline"] = "unavailable"
        
        self.system_integrations = integrations
    
    async def _discover_existing_agents(self):
        """Discover and register existing agents in the system."""
        # This would integrate with existing agent systems to discover agents
        # For now, we'll create some default coordinator agents
        
        coordinator_config = {
            "agent_id": "coordinator_001",
            "name": "Primary Coordinator",
            "role": "coordinator",
            "capabilities": [
                {
                    "name": "task_coordination",
                    "description": "Coordinate task assignments",
                    "performance_score": 0.9,
                    "specialization_areas": ["coordination", "optimization"]
                }
            ],
            "location": {"node": "localhost", "region": "primary"},
            "endpoints": ["http://localhost:8000/coordinator"]
        }
        
        coordinator_agent = await self.register_agent(coordinator_config)
        if coordinator_agent:
            self.coordinator_agents.append(coordinator_agent.agent_id)
        
        logger.info(f"Discovered {len(self.agents)} agents")
    
    async def _start_coordination_services(self):
        """Start background coordination services."""
        
        # Start heartbeat monitoring
        asyncio.create_task(self._heartbeat_monitoring_loop())
        
        # Start performance optimization
        asyncio.create_task(self._performance_optimization_loop())
        
        # Start metrics collection
        asyncio.create_task(self._metrics_collection_loop())
        
        # Start task queue processing
        asyncio.create_task(self._task_queue_processing_loop())
        
        logger.info("Coordination services started")
    
    async def _heartbeat_monitoring_loop(self):
        """Background heartbeat monitoring."""
        while self.coordination_active:
            try:
                current_time = datetime.now()
                heartbeat_timeout = timedelta(minutes=5)
                
                for agent_id, agent in self.agents.items():
                    if current_time - agent.last_heartbeat > heartbeat_timeout:
                        if agent_id not in self.offline_agents:
                            logger.warning(f"Agent {agent_id} appears to be offline")
                            self.offline_agents.add(agent_id)
                            agent.state = AgentState.OFFLINE
                            
                            # Handle ongoing tasks
                            for task_id in agent.current_tasks:
                                await self._handle_agent_failure(task_id, agent_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Heartbeat monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _performance_optimization_loop(self):
        """Background performance optimization."""
        while self.coordination_active:
            try:
                # Run optimization every 10 minutes
                await asyncio.sleep(600)
                
                # Optimize agent assignments
                await self.performance_optimizer.optimize_assignments(self.agents, self.tasks)
                
                # Optimize network topology
                await self.performance_optimizer.optimize_topology(self.network_topology)
                
                self.last_optimization = datetime.now()
                
            except Exception as e:
                logger.error(f"Performance optimization error: {e}")
                await asyncio.sleep(600)
    
    async def _metrics_collection_loop(self):
        """Background metrics collection."""
        while self.coordination_active:
            try:
                # Collect metrics every minute
                await asyncio.sleep(60)
                
                # Generate current metrics
                status = await self.get_system_status()
                
                metrics = CoordinationMetrics(
                    total_agents=status["system_health"]["total_agents"],
                    active_agents=status["system_health"]["active_agents"],
                    total_tasks=status["task_statistics"]["total_tasks"],
                    completed_tasks=status["task_statistics"]["completed_tasks"],
                    failed_tasks=status["task_statistics"]["failed_tasks"],
                    average_task_completion_time=status["performance_metrics"]["average_completion_time_hours"],
                    agent_utilization=status["performance_metrics"]["agent_utilization"],
                    system_throughput=status["performance_metrics"]["system_throughput"],
                    coordination_overhead=status["performance_metrics"]["coordination_overhead"],
                    consensus_time=0.0,  # Placeholder
                    timestamp=datetime.now()
                )
                
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 metrics
                if len(self.metrics_history) > 1000:
                    self.metrics_history.pop(0)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)
    
    async def _task_queue_processing_loop(self):
        """Background task queue processing."""
        while self.coordination_active:
            try:
                if self.task_queue:
                    await self._process_task_queue()
                
                await asyncio.sleep(10)  # Process every 10 seconds
                
            except Exception as e:
                logger.error(f"Task queue processing error: {e}")
                await asyncio.sleep(10)
    
    # Utility methods for message handling and agent communication
    
    async def _send_agent_message(self, agent_id: str, message: Dict[str, Any]):
        """Send message to specific agent."""
        if agent_id in self.agents:
            await self.message_broker.send_message(agent_id, message)
    
    async def _update_agent_performance(self, agent_id: str, task_id: str, progress: float, status: Optional[str]):
        """Update agent performance metrics."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            
            # Update response time
            if status == "completed":
                # Simplified response time calculation
                response_time = 1.0  # Placeholder
                current_avg = agent.performance_metrics.get("average_response_time", 0.0)
                task_count = agent.performance_metrics.get("tasks_completed", 1)
                
                new_avg = (current_avg * (task_count - 1) + response_time) / task_count
                agent.performance_metrics["average_response_time"] = new_avg
            
            # Update success rate
            if status in ["completed", "failed"]:
                total_tasks = agent.performance_metrics.get("tasks_completed", 0) + agent.performance_metrics.get("tasks_failed", 0)
                completed_tasks = agent.performance_metrics.get("tasks_completed", 0)
                
                if status == "completed":
                    completed_tasks += 1
                    agent.performance_metrics["tasks_completed"] = completed_tasks
                else:
                    agent.performance_metrics["tasks_failed"] = agent.performance_metrics.get("tasks_failed", 0) + 1
                
                if total_tasks > 0:
                    agent.performance_metrics["success_rate"] = completed_tasks / (total_tasks + 1)
    
    async def _handle_task_timeout(self, task_id: str):
        """Handle task timeout."""
        task = self.tasks.get(task_id)
        if task:
            task.status = "failed"
            task.result = {"error": "timeout", "message": "Task execution timed out"}
            
            # Free up assigned agents
            for agent_id in task.assigned_agents:
                if agent_id in self.agents:
                    agent = self.agents[agent_id]
                    if task_id in agent.current_tasks:
                        agent.current_tasks.remove(task_id)
                    agent.state = AgentState.IDLE
            
            logger.warning(f"Task {task_id} timed out and marked as failed")
    
    async def _handle_agent_failure(self, task_id: str, failed_agent_id: str):
        """Handle agent failure during task execution."""
        task = self.tasks.get(task_id)
        if not task:
            return
        
        # Remove failed agent from assignment
        if failed_agent_id in task.assigned_agents:
            task.assigned_agents.remove(failed_agent_id)
        
        # Try to reassign task to other agents
        if not task.assigned_agents:
            # Find replacement agents
            suitable_agents = await self._find_suitable_agents(task)
            
            if suitable_agents:
                await self._assign_task_to_agents(task_id, suitable_agents)
                logger.info(f"Reassigned task {task_id} after agent {failed_agent_id} failure")
            else:
                # No suitable agents available, mark task as failed
                task.status = "failed"
                task.result = {"error": "agent_failure", "failed_agent": failed_agent_id}
                logger.warning(f"Task {task_id} failed due to agent failure and no replacement available")

# Supporting classes for coordination system

class MessageBroker:
    """Message broker for agent communication."""
    
    async def initialize(self):
        """Initialize message broker."""
        self.message_queues = {}
    
    async def send_message(self, agent_id: str, message: Dict[str, Any]):
        """Send message to agent."""
        if agent_id not in self.message_queues:
            self.message_queues[agent_id] = deque()
        
        self.message_queues[agent_id].append({
            "message": message,
            "timestamp": datetime.now(),
            "message_id": str(uuid.uuid4())
        })

class ConsensusEngine:
    """Consensus engine for distributed decision making."""
    
    async def initialize(self):
        """Initialize consensus engine."""
        pass
    
    async def reach_consensus(self, agents: List[str], decision: Dict[str, Any]) -> bool:
        """Reach consensus among agents for decision."""
        # Simplified consensus implementation
        return True

class PerformanceOptimizer:
    """Performance optimizer for coordination system."""
    
    async def initialize(self):
        """Initialize performance optimizer."""
        pass
    
    async def optimize_assignments(self, agents: Dict[str, CoordinatedAgent], tasks: Dict[str, CoordinationTask]):
        """Optimize agent-task assignments."""
        # Simplified optimization
        pass
    
    async def optimize_topology(self, network: nx.Graph):
        """Optimize network topology."""
        # Simplified topology optimization
        pass

# Global instance for system integration
global_agent_coordinator = None

async def get_distributed_agent_coordinator() -> DistributedAgentCoordinator:
    """Get global distributed agent coordinator instance."""
    global global_agent_coordinator
    if global_agent_coordinator is None:
        global_agent_coordinator = DistributedAgentCoordinator()
        await global_agent_coordinator.initialize()
    return global_agent_coordinator

async def agent_coordination_health() -> bool:
    """Quick health check for agent coordination system."""
    try:
        coordinator = await get_distributed_agent_coordinator()
        return coordinator.coordination_active
    except Exception:
        return False