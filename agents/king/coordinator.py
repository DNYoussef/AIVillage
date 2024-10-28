"""King agent coordinator module."""

from typing import Dict, Any, List, Optional
from datetime import datetime
from rag_system.core.config import UnifiedConfig
from rag_system.core.base_component import BaseComponent
from rag_system.utils.error_handling import log_and_handle_errors, ErrorContext
from communications.protocol import StandardCommunicationProtocol

class KingCoordinator(BaseComponent):
    """
    Coordinator for King agent operations.
    Manages task distribution, resource allocation, and agent coordination.
    """
    
    def __init__(self, config: UnifiedConfig, communication_protocol: StandardCommunicationProtocol):
        """
        Initialize coordinator.
        
        Args:
            config: Configuration instance
            communication_protocol: Communication protocol instance
        """
        self.config = config
        self.communication_protocol = communication_protocol
        self.tasks = {}
        self.resources = {}
        self.agent_states = {}
        self.initialized = False
        self.coordination_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_completion_time": 0.0
        }
    
    @log_and_handle_errors()
    async def initialize(self) -> None:
        """Initialize coordinator."""
        if not self.initialized:
            # Initialize task and resource tracking
            self.tasks = {}
            self.resources = {}
            self.agent_states = {}
            self.initialized = True
    
    @log_and_handle_errors()
    async def shutdown(self) -> None:
        """Shutdown coordinator."""
        if self.initialized:
            # Clean up resources
            self.tasks.clear()
            self.resources.clear()
            self.agent_states.clear()
            self.initialized = False
    
    @log_and_handle_errors()
    async def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        return {
            "initialized": self.initialized,
            "active_tasks": len(self.tasks),
            "available_resources": len(self.resources),
            "agent_count": len(self.agent_states),
            "coordination_stats": self.coordination_stats
        }
    
    @log_and_handle_errors()
    async def update_config(self, config: Dict[str, Any]) -> None:
        """Update component configuration."""
        self.config = config

    async def assign_task(self, task: Dict[str, Any]) -> str:
        """
        Assign task to appropriate agent.
        
        Args:
            task: Task description and requirements
            
        Returns:
            Task ID
        """
        async with ErrorContext("KingCoordinator"):
            # Generate task ID
            task_id = f"task_{len(self.tasks)}"
            
            # Add task tracking
            self.tasks[task_id] = {
                "description": task,
                "status": "assigned",
                "assigned_to": None,
                "start_time": datetime.now(),
                "completion_time": None
            }
            
            # Update stats
            self.coordination_stats["total_tasks"] += 1
            
            return task_id

    async def update_task_status(self,
                               task_id: str,
                               status: str,
                               result: Optional[Dict[str, Any]] = None) -> None:
        """
        Update task status and results.
        
        Args:
            task_id: Task identifier
            status: New task status
            result: Optional task result
        """
        if task_id not in self.tasks:
            raise ValueError(f"Unknown task ID: {task_id}")
        
        self.tasks[task_id]["status"] = status
        if result:
            self.tasks[task_id]["result"] = result
        
        if status == "completed":
            self.tasks[task_id]["completion_time"] = datetime.now()
            self.coordination_stats["completed_tasks"] += 1
            
            # Update average completion time
            completion_time = (
                self.tasks[task_id]["completion_time"] -
                self.tasks[task_id]["start_time"]
            ).total_seconds()
            
            current_avg = self.coordination_stats["avg_completion_time"]
            completed_tasks = self.coordination_stats["completed_tasks"]
            
            if completed_tasks > 1:
                self.coordination_stats["avg_completion_time"] = (
                    (current_avg * (completed_tasks - 1) + completion_time) /
                    completed_tasks
                )
            else:
                self.coordination_stats["avg_completion_time"] = completion_time
        
        elif status == "failed":
            self.coordination_stats["failed_tasks"] += 1

    async def allocate_resources(self,
                               task_id: str,
                               requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allocate resources for task.
        
        Args:
            task_id: Task identifier
            requirements: Resource requirements
            
        Returns:
            Allocated resources
        """
        if task_id not in self.tasks:
            raise ValueError(f"Unknown task ID: {task_id}")
        
        # Check resource availability
        allocated_resources = {}
        for resource_type, amount in requirements.items():
            if resource_type not in self.resources:
                raise ValueError(f"Unknown resource type: {resource_type}")
            
            if self.resources[resource_type]["available"] < amount:
                raise ValueError(
                    f"Insufficient {resource_type}: "
                    f"requested {amount}, "
                    f"available {self.resources[resource_type]['available']}"
                )
            
            # Allocate resources
            self.resources[resource_type]["available"] -= amount
            allocated_resources[resource_type] = amount
        
        # Update task tracking
        self.tasks[task_id]["allocated_resources"] = allocated_resources
        
        return allocated_resources

    async def release_resources(self, task_id: str) -> None:
        """
        Release resources allocated to task.
        
        Args:
            task_id: Task identifier
        """
        if task_id not in self.tasks:
            raise ValueError(f"Unknown task ID: {task_id}")
        
        # Get allocated resources
        allocated_resources = self.tasks[task_id].get("allocated_resources", {})
        
        # Release resources
        for resource_type, amount in allocated_resources.items():
            self.resources[resource_type]["available"] += amount
        
        # Clear task allocation
        self.tasks[task_id]["allocated_resources"] = {}

    async def register_agent(self, agent_id: str, capabilities: List[str]) -> None:
        """
        Register agent with coordinator.
        
        Args:
            agent_id: Agent identifier
            capabilities: List of agent capabilities
        """
        self.agent_states[agent_id] = {
            "status": "available",
            "capabilities": capabilities,
            "current_task": None,
            "performance_metrics": {
                "tasks_completed": 0,
                "success_rate": 0.0,
                "avg_completion_time": 0.0
            }
        }

    async def update_agent_status(self,
                                agent_id: str,
                                status: str,
                                current_task: Optional[str] = None) -> None:
        """
        Update agent status.
        
        Args:
            agent_id: Agent identifier
            status: New agent status
            current_task: Optional current task ID
        """
        if agent_id not in self.agent_states:
            raise ValueError(f"Unknown agent ID: {agent_id}")
        
        self.agent_states[agent_id]["status"] = status
        if current_task:
            self.agent_states[agent_id]["current_task"] = current_task

    async def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """
        Get agent performance metrics.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent metrics
        """
        if agent_id not in self.agent_states:
            raise ValueError(f"Unknown agent ID: {agent_id}")
        
        return self.agent_states[agent_id]["performance_metrics"]

    async def update_agent_metrics(self,
                                 agent_id: str,
                                 task_result: Dict[str, Any]) -> None:
        """
        Update agent performance metrics.
        
        Args:
            agent_id: Agent identifier
            task_result: Task completion results
        """
        if agent_id not in self.agent_states:
            raise ValueError(f"Unknown agent ID: {agent_id}")
        
        metrics = self.agent_states[agent_id]["performance_metrics"]
        
        # Update task count
        metrics["tasks_completed"] += 1
        
        # Update success rate
        success = task_result.get("success", False)
        current_success_rate = metrics["success_rate"]
        total_tasks = metrics["tasks_completed"]
        
        if total_tasks > 1:
            metrics["success_rate"] = (
                (current_success_rate * (total_tasks - 1) + (1 if success else 0)) /
                total_tasks
            )
        else:
            metrics["success_rate"] = 1.0 if success else 0.0
        
        # Update average completion time
        completion_time = task_result.get("completion_time", 0.0)
        current_avg_time = metrics["avg_completion_time"]
        
        if total_tasks > 1:
            metrics["avg_completion_time"] = (
                (current_avg_time * (total_tasks - 1) + completion_time) /
                total_tasks
            )
        else:
            metrics["avg_completion_time"] = completion_time
