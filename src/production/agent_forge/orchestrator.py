"""Fast Agent Orchestrator with <100ms overhead.

Provides high-performance agent coordination for the 18-agent ecosystem.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from src.communications.standard_protocol import StandardCommunicationProtocol

from .agent_factory import AgentFactory
from .base import BaseMetaAgent

logger = logging.getLogger(__name__)


@dataclass
class TaskRequest:
    """High-performance task request."""

    task_id: str
    task_type: str
    payload: dict[str, Any]
    priority: int = 5
    timeout_ms: int = 30000
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time() * 1000  # milliseconds


@dataclass
class TaskResult:
    """Task execution result with timing."""

    task_id: str
    success: bool
    result: dict[str, Any]
    agent_id: str
    execution_time_ms: float
    completed_at: float


class FastAgentOrchestrator:
    """High-performance agent orchestrator with <100ms coordination overhead."""

    def __init__(self, factory: AgentFactory = None) -> None:
        self.factory = factory or AgentFactory()
        self.agents: dict[str, BaseMetaAgent] = {}
        self.communication_protocol = StandardCommunicationProtocol()

        # Performance tracking
        self.orchestration_times: list[float] = []
        self.task_count = 0
        self.agent_loads: dict[str, int] = defaultdict(int)

        # Task routing
        self.task_queue: list[TaskRequest] = []
        self.active_tasks: dict[str, TaskRequest] = {}
        self.completed_tasks: dict[str, TaskResult] = {}

        # Agent-task mapping for optimal routing
        self.agent_capabilities_cache: dict[str, list[str]] = {}
        self.agent_performance_scores: dict[str, float] = defaultdict(lambda: 0.8)

        logger.info("Fast agent orchestrator initialized")

    async def initialize_agents(self) -> None:
        """Initialize all 18 agents with optimized startup."""
        start_time = time.time() * 1000

        agent_types = self.factory.required_agent_types()

        # Initialize agents in parallel for speed
        tasks = []
        for agent_type in agent_types:
            task = self._initialize_single_agent(agent_type)
            tasks.append(task)

        # Wait for all agents to initialize
        await asyncio.gather(*tasks, return_exceptions=True)

        init_time = time.time() * 1000 - start_time
        logger.info(f"Initialized {len(self.agents)} agents in {init_time:.1f}ms")

        # Cache agent capabilities for fast routing
        self._cache_agent_capabilities()

    async def _initialize_single_agent(self, agent_type: str) -> None:
        """Initialize a single agent."""
        try:
            agent = self.factory.create_agent(agent_type)
            self.agents[agent_type] = agent
            logger.debug(f"Initialized {agent_type} agent")
        except Exception as e:
            logger.error(f"Failed to initialize {agent_type}: {e}")

    def _cache_agent_capabilities(self) -> None:
        """Cache agent capabilities for fast task routing."""
        for agent_id, agent in self.agents.items():
            if hasattr(agent, "get_capabilities"):
                self.agent_capabilities_cache[agent_id] = agent.get_capabilities()
            else:
                # Default capabilities based on agent type
                self.agent_capabilities_cache[agent_id] = [agent_id]

    async def execute_task(self, task_request: TaskRequest) -> TaskResult:
        """Execute task with <100ms orchestration overhead."""
        orchestration_start = time.time() * 1000

        # Fast agent selection (target: <10ms)
        agent_id = self._select_optimal_agent(task_request)

        if not agent_id or agent_id not in self.agents:
            return TaskResult(
                task_id=task_request.task_id,
                success=False,
                result={"error": "No suitable agent available"},
                agent_id="none",
                execution_time_ms=0,
                completed_at=time.time() * 1000,
            )

        # Execute task with timing
        agent = self.agents[agent_id]
        execution_start = time.time() * 1000

        try:
            # Update agent load
            self.agent_loads[agent_id] += 1
            self.active_tasks[task_request.task_id] = task_request

            # Execute task
            result = agent.process(task_request.payload)
            success = True

            # Update agent performance score
            self._update_agent_performance(agent_id, True)

        except Exception as e:
            logger.error(f"Task {task_request.task_id} failed on {agent_id}: {e}")
            result = {"error": str(e)}
            success = False
            self._update_agent_performance(agent_id, False)
        finally:
            # Clean up
            self.agent_loads[agent_id] -= 1
            if task_request.task_id in self.active_tasks:
                del self.active_tasks[task_request.task_id]

        execution_time = time.time() * 1000 - execution_start
        orchestration_time = time.time() * 1000 - orchestration_start

        # Track orchestration performance
        self.orchestration_times.append(orchestration_time - execution_time)
        if len(self.orchestration_times) > 1000:  # Keep last 1000 measurements
            self.orchestration_times = self.orchestration_times[-1000:]

        task_result = TaskResult(
            task_id=task_request.task_id,
            success=success,
            result=result,
            agent_id=agent_id,
            execution_time_ms=execution_time,
            completed_at=time.time() * 1000,
        )

        self.completed_tasks[task_request.task_id] = task_result
        self.task_count += 1

        return task_result

    def _select_optimal_agent(self, task_request: TaskRequest) -> str | None:
        """Fast agent selection based on capabilities and load."""
        task_type = task_request.task_type

        # First, try exact match
        if task_type in self.agents and self.agent_loads[task_type] < 3:
            return task_type

        # Find best match based on capabilities and current load
        best_agent = None
        best_score = -1

        for agent_id, capabilities in self.agent_capabilities_cache.items():
            if agent_id not in self.agents:
                continue

            # Calculate suitability score
            capability_score = 1.0 if task_type in capabilities else 0.3
            performance_score = self.agent_performance_scores[agent_id]
            load_penalty = self.agent_loads[agent_id] * 0.2

            total_score = capability_score * performance_score - load_penalty

            if total_score > best_score:
                best_score = total_score
                best_agent = agent_id

        return best_agent

    def _update_agent_performance(self, agent_id: str, success: bool) -> None:
        """Update agent performance score with exponential moving average."""
        current_score = self.agent_performance_scores[agent_id]
        new_score = 1.0 if success else 0.0

        # EMA with alpha=0.1 for smooth updates
        self.agent_performance_scores[agent_id] = current_score * 0.9 + new_score * 0.1

    async def process_batch(self, tasks: list[TaskRequest]) -> list[TaskResult]:
        """Process multiple tasks concurrently for maximum throughput."""
        start_time = time.time() * 1000

        # Execute all tasks concurrently
        task_coroutines = [self.execute_task(task) for task in tasks]
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)

        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = TaskResult(
                    task_id=tasks[i].task_id,
                    success=False,
                    result={"error": str(result)},
                    agent_id="error",
                    execution_time_ms=0,
                    completed_at=time.time() * 1000,
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)

        batch_time = time.time() * 1000 - start_time
        logger.info(f"Processed {len(tasks)} tasks in {batch_time:.1f}ms")

        return processed_results

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get orchestrator performance metrics."""
        if not self.orchestration_times:
            avg_overhead = 0
            max_overhead = 0
        else:
            avg_overhead = sum(self.orchestration_times) / len(self.orchestration_times)
            max_overhead = max(self.orchestration_times)

        return {
            "average_orchestration_overhead_ms": avg_overhead,
            "max_orchestration_overhead_ms": max_overhead,
            "total_tasks_processed": self.task_count,
            "active_agents": len(self.agents),
            "current_active_tasks": len(self.active_tasks),
            "agent_loads": dict(self.agent_loads),
            "agent_performance_scores": dict(self.agent_performance_scores),
            "overhead_target_met": avg_overhead < 100.0,
        }

    def get_agent_status(self, agent_id: str) -> dict[str, Any] | None:
        """Get status of a specific agent."""
        if agent_id not in self.agents:
            return None

        agent = self.agents[agent_id]
        return {
            "agent_id": agent_id,
            "agent_name": getattr(agent, "name", agent_id),
            "current_load": self.agent_loads[agent_id],
            "performance_score": self.agent_performance_scores[agent_id],
            "capabilities": self.agent_capabilities_cache.get(agent_id, []),
            "status": getattr(agent, "get_status", lambda: {})(),
        }

    def list_agents(self) -> list[str]:
        """List all available agents."""
        return list(self.agents.keys())

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on all agents."""
        health_start = time.time() * 1000

        # Simple ping test for all agents
        ping_tasks = []
        for agent_id in self.agents.keys():
            task_request = TaskRequest(
                task_id=f"health_{agent_id}_{int(time.time() * 1000)}",
                task_type="ping",
                payload={"task": "ping"},
            )
            ping_tasks.append(self.execute_task(task_request))

        results = await asyncio.gather(*ping_tasks, return_exceptions=True)

        health_time = time.time() * 1000 - health_start

        # Analyze health
        healthy_agents = sum(
            1 for r in results if not isinstance(r, Exception) and r.success
        )
        total_agents = len(self.agents)

        return {
            "healthy_agents": healthy_agents,
            "total_agents": total_agents,
            "health_check_time_ms": health_time,
            "system_healthy": healthy_agents == total_agents,
            "agent_health": {
                agent_id: not isinstance(results[i], Exception) and results[i].success
                for i, agent_id in enumerate(self.agents.keys())
            },
        }
