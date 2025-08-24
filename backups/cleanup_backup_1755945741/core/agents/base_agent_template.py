"""
Core Agent Base Template

Consolidated base template for all AIVillage agents following clean architecture
and connascence principles. This replaces the scattered agent implementations
with a unified, dependency-injected foundation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any, Protocol

from ..domain.entities.agent_entity import Agent, AgentCapability, AgentId
from ..domain.entities.session_entity import Session
from ..domain.entities.task_entity import Task, TaskId, TaskStatus

logger = logging.getLogger(__name__)


# Infrastructure Dependencies (Injected, not imported directly)


class RAGSystemInterface(Protocol):
    """Interface for RAG system dependency"""

    async def query(self, query: str, mode: str = "balanced", max_results: int = 10) -> dict[str, Any]:
        """Query the RAG system for information"""
        ...


class CommunicationInterface(Protocol):
    """Interface for P2P communication dependency"""

    async def send_message(
        self, recipient: str, message: str, channel_type: str = "direct", priority: int = 5
    ) -> dict[str, Any]:
        """Send message through communication channels"""
        ...


class FogComputeInterface(Protocol):
    """Interface for fog computing dependency"""

    async def submit_job(
        self, computation_type: str, input_data: dict[str, Any], resources: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Submit computation to fog network"""
        ...


class AgentForgeInterface(Protocol):
    """Interface for agent self-modification dependency"""

    async def execute_adas_phase(self, modification_request: dict[str, Any]) -> dict[str, Any]:
        """Execute ADAS self-modification"""
        ...


@dataclass
class AgentDependencies:
    """
    Dependency injection container for agent infrastructure

    Following dependency inversion principle - agents depend on
    interfaces, not concrete implementations.
    """

    rag_system: RAGSystemInterface | None = None
    communication: CommunicationInterface | None = None
    fog_compute: FogComputeInterface | None = None
    agent_forge: AgentForgeInterface | None = None

    # Repository dependencies
    agent_repository: Any | None = None
    task_repository: Any | None = None
    knowledge_repository: Any | None = None
    session_repository: Any | None = None


class BaseAgentTemplate(ABC):
    """
    Consolidated base template for all AIVillage agents

    Provides unified foundation with:
    - Clean dependency injection
    - Domain entity integration
    - Behavioral contracts over implementation details
    - Connascence-aware coupling management
    """

    def __init__(self, agent_entity: Agent, dependencies: AgentDependencies):
        # Core domain entity
        self.agent = agent_entity

        # Injected dependencies (weak coupling)
        self.dependencies = dependencies

        # Internal state
        self.current_tasks: dict[TaskId, Task] = {}
        self.current_session: Session | None = None

        # Performance tracking
        self.operation_history: list[dict[str, Any]] = []

        logger.info(f"BaseAgentTemplate initialized: {self.agent.name}")

    # Core Agent Lifecycle

    async def initialize(self) -> bool:
        """
        Initialize agent with all dependencies

        Template method pattern - subclasses can override specific initialization
        """
        try:
            logger.info(f"Initializing agent: {self.agent.name}")

            # Activate the domain entity
            self.agent.activate()

            # Initialize domain-specific resources
            success = await self.initialize_domain_resources()

            if success:
                logger.info(f"Agent {self.agent.name} initialized successfully")
                return True
            else:
                logger.error(f"Failed to initialize domain resources for {self.agent.name}")
                return False

        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            return False

    async def shutdown(self) -> bool:
        """Gracefully shutdown agent"""
        try:
            logger.info(f"Shutting down agent: {self.agent.name}")

            # Complete any active tasks
            await self._complete_active_tasks()

            # Save agent state
            if self.dependencies.agent_repository:
                await self.dependencies.agent_repository.save(self.agent)

            # Cleanup domain resources
            await self.cleanup_domain_resources()

            # Terminate the domain entity
            self.agent.terminate()

            logger.info(f"Agent {self.agent.name} shutdown complete")
            return True

        except Exception as e:
            logger.error(f"Agent shutdown failed: {e}")
            return False

    # Task Processing (Business Logic)

    async def process_task(self, task: Task) -> dict[str, Any]:
        """
        Process a task using domain-specific logic

        Template method with behavioral contract:
        - Input: Valid Task entity
        - Output: Processing result with status
        - Side effects: Task status updates, performance metrics
        """

        if task.id in self.current_tasks:
            return {"status": "error", "message": "Task already in progress"}

        try:
            start_time = datetime.now()

            # Update task and agent state
            task.start_execution()
            self.current_tasks[task.id] = task

            # Execute domain-specific processing
            result = await self.execute_domain_task(task)

            # Calculate performance metrics
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            success = result.get("status") == "success"

            # Update task completion
            if success:
                task.complete_successfully(result.get("output_data", {}))
            else:
                task.fail_with_error(result.get("error", "Unknown error"))

            # Update agent performance
            self.agent.update_performance(execution_time, success)

            # Clean up
            self.current_tasks.pop(task.id, None)

            # Save state
            if self.dependencies.task_repository:
                await self.dependencies.task_repository.save(task)
            if self.dependencies.agent_repository:
                await self.dependencies.agent_repository.save(self.agent)

            return result

        except Exception as e:
            logger.error(f"Task processing failed: {e}")
            task.fail_with_error(str(e))
            self.current_tasks.pop(task.id, None)

            return {"status": "error", "error": str(e)}

    # Communication (Behavioral Contracts)

    async def send_message(
        self, recipient: str, message: str, channel_type: str = "direct", priority: int = 5
    ) -> dict[str, Any]:
        """Send message through communication system"""

        if not self.dependencies.communication:
            return {"status": "error", "message": "Communication system not available"}

        try:
            result = await self.dependencies.communication.send_message(
                recipient=recipient, message=message, channel_type=channel_type, priority=priority
            )

            # Record communication attempt
            self._record_operation(
                "communication",
                {"recipient": recipient, "channel_type": channel_type, "success": result.get("status") == "success"},
            )

            return result

        except Exception as e:
            logger.error(f"Communication failed: {e}")
            return {"status": "error", "error": str(e)}

    # Knowledge Access (Weak Coupling to RAG)

    async def query_knowledge(self, query: str, mode: str = "balanced", max_results: int = 10) -> dict[str, Any]:
        """Query knowledge through RAG system"""

        if not self.dependencies.rag_system:
            return {"status": "error", "message": "RAG system not available"}

        try:
            result = await self.dependencies.rag_system.query(query=query, mode=mode, max_results=max_results)

            # Record knowledge access
            self._record_operation(
                "knowledge_query",
                {
                    "query": query,
                    "mode": mode,
                    "results_count": len(result.get("results", [])),
                    "success": result.get("status") == "success",
                },
            )

            return result

        except Exception as e:
            logger.error(f"Knowledge query failed: {e}")
            return {"status": "error", "error": str(e)}

    # Computational Offloading (Optional Dependency)

    async def offload_computation(
        self, computation_type: str, input_data: dict[str, Any], resources: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Offload computation to fog network"""

        if not self.dependencies.fog_compute:
            # Fall back to local computation
            return await self.execute_local_computation(computation_type, input_data)

        try:
            result = await self.dependencies.fog_compute.submit_job(
                computation_type=computation_type, input_data=input_data, resources=resources
            )

            # Record computation offload
            self._record_operation(
                "fog_computation", {"computation_type": computation_type, "success": result.get("status") == "success"}
            )

            return result

        except Exception as e:
            logger.error(f"Fog computation failed: {e}")
            return {"status": "error", "error": str(e)}

    # Health and Status

    async def health_check(self) -> dict[str, Any]:
        """Comprehensive agent health check"""

        return {
            "agent_id": str(self.agent.id),
            "name": self.agent.name,
            "status": self.agent.status.value,
            "health": {
                "is_healthy": self.agent.is_healthy(),
                "success_rate": self.agent.success_rate,
                "avg_response_time_ms": self.agent.average_response_time_ms,
                "tasks_completed": self.agent.tasks_completed,
            },
            "dependencies": {
                "rag_system": self.dependencies.rag_system is not None,
                "communication": self.dependencies.communication is not None,
                "fog_compute": self.dependencies.fog_compute is not None,
                "agent_forge": self.dependencies.agent_forge is not None,
            },
            "current_tasks": len(self.current_tasks),
            "domain": self.agent.get_specialization_domain(),
            "capabilities": [cap.value for cap in self.agent.capabilities],
            "last_active": self.agent.last_active.isoformat() if self.agent.last_active else None,
        }

    # Abstract Methods (Domain-Specific Implementation)

    @abstractmethod
    async def initialize_domain_resources(self) -> bool:
        """Initialize domain-specific resources and dependencies"""
        pass

    @abstractmethod
    async def execute_domain_task(self, task: Task) -> dict[str, Any]:
        """Execute domain-specific task processing logic"""
        pass

    @abstractmethod
    async def cleanup_domain_resources(self) -> None:
        """Clean up domain-specific resources"""
        pass

    @abstractmethod
    async def execute_local_computation(self, computation_type: str, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute computation locally when fog compute unavailable"""
        pass

    @abstractmethod
    def get_specialized_capabilities(self) -> list[AgentCapability]:
        """Return domain-specific capabilities"""
        pass

    # Internal Utilities

    def _record_operation(self, operation_type: str, details: dict[str, Any]) -> None:
        """Record operation for monitoring and analysis"""

        operation_record = {
            "timestamp": datetime.now().isoformat(),
            "operation_type": operation_type,
            "agent_id": str(self.agent.id),
            "details": details,
        }

        self.operation_history.append(operation_record)

        # Keep only recent history
        if len(self.operation_history) > 1000:
            self.operation_history = self.operation_history[-1000:]

    async def _complete_active_tasks(self) -> None:
        """Complete or cancel any active tasks during shutdown"""

        for task_id, task in list(self.current_tasks.items()):
            try:
                if task.status == TaskStatus.IN_PROGRESS:
                    task.cancel()

                if self.dependencies.task_repository:
                    await self.dependencies.task_repository.save(task)

            except Exception as e:
                logger.error(f"Failed to complete task {task_id}: {e}")

        self.current_tasks.clear()

    # Factory Method for Agent Creation

    @classmethod
    def create_agent(
        cls,
        name: str,
        agent_type: str,
        capabilities: list[AgentCapability],
        specialized_role: str | None = None,
        dependencies: AgentDependencies | None = None,
    ) -> BaseAgentTemplate:
        """
        Factory method to create properly configured agent

        Ensures all domain invariants are satisfied
        """

        # Create domain entity
        agent_entity = Agent(
            id=AgentId.generate(),
            name=name,
            agent_type=agent_type,
            capabilities=capabilities,
            specialized_role=specialized_role,
        )

        # Use default dependencies if none provided
        if dependencies is None:
            dependencies = AgentDependencies()

        # Create agent instance
        return cls(agent_entity, dependencies)
