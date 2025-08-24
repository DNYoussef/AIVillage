"""Agent State Manager Component.

Manages agent status, geometric self-awareness, and state transitions.
Encapsulates all state-related logic in a focused component.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent operational states."""

    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    ADAPTING = "adapting"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    OFFLINE = "offline"


class GeometricState(Enum):
    """Geometric self-awareness states (proprioception-like)."""

    BALANCED = "balanced"
    OVERLOADED = "overloaded"
    UNDERUTILIZED = "underutilized"
    ADAPTING = "adapting"
    OPTIMIZING = "optimizing"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics for geometric awareness."""

    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    network_activity: float = 0.0
    task_queue_depth: int = 0
    response_latency_ms: float = 0.0
    accuracy_score: float = 1.0
    energy_efficiency: float = 1.0


@dataclass
class GeometricSelfState:
    """Complete geometric self-awareness state snapshot."""

    timestamp: datetime
    geometric_state: GeometricState
    resource_metrics: ResourceMetrics
    adaptation_rate: float = 0.1
    stability_score: float = 1.0
    optimization_direction: str = "balanced"

    def is_healthy(self) -> bool:
        """Determine if agent is in healthy geometric state."""
        metrics = self.resource_metrics
        return (
            metrics.cpu_utilization < 0.9
            and metrics.memory_utilization < 0.9
            and metrics.response_latency_ms < 5000
            and metrics.accuracy_score > 0.7
            and self.stability_score > 0.5
        )


@dataclass
class StateConfig:
    """Configuration for state management."""

    update_interval_seconds: int = 30
    health_check_interval_seconds: int = 60
    max_history_entries: int = 100
    geometric_awareness_enabled: bool = True
    auto_adaptation_enabled: bool = True


class AgentStateManager:
    """Manages agent state, geometric awareness, and state transitions.

    This component encapsulates all state management logic, providing
    clean separation from other agent responsibilities.
    """

    def __init__(self, agent_id: str, config: StateConfig | None = None):
        """Initialize state manager.

        Args:
            agent_id: Unique identifier for the agent
            config: State management configuration
        """
        self.agent_id = agent_id
        self.config = config or StateConfig()

        # Current state (CoI - Connascence of Identity)
        self._current_state = AgentState.INITIALIZING
        self._current_geometric_state: GeometricSelfState | None = None

        # State history for analysis
        self._state_history: list[tuple[datetime, AgentState]] = []
        self._geometric_history: list[GeometricSelfState] = []

        # Performance tracking for geometric awareness
        self._task_history: list[dict[str, Any]] = []

        # Monitoring task
        self._monitoring_task: asyncio.Task | None = None
        self._shutdown_requested = False

        logger.debug(f"State manager initialized for agent {agent_id}")

    async def start_monitoring(self) -> None:
        """Start background monitoring of geometric state."""
        if self._monitoring_task is not None:
            logger.warning("Monitoring already started")
            return

        if self.config.geometric_awareness_enabled:
            self._monitoring_task = asyncio.create_task(self._periodic_updates())
            logger.info("Geometric state monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._shutdown_requested = True

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

        logger.info("Geometric state monitoring stopped")

    def get_current_state(self) -> AgentState:
        """Get current agent state."""
        return self._current_state

    def set_state(self, new_state: AgentState, reason: str = "") -> None:
        """Set agent state with transition logging.

        Args:
            new_state: Target state to transition to
            reason: Optional reason for state change
        """
        if new_state == self._current_state:
            return

        old_state = self._current_state
        self._current_state = new_state

        # Record state transition
        timestamp = datetime.now()
        self._state_history.append((timestamp, new_state))

        # Maintain history size
        if len(self._state_history) > self.config.max_history_entries:
            self._state_history = self._state_history[-self.config.max_history_entries :]

        logger.info(f"Agent {self.agent_id} state: {old_state.value} -> {new_state.value} ({reason})")

    def get_geometric_state(self) -> GeometricSelfState | None:
        """Get current geometric self-awareness state."""
        return self._current_geometric_state

    async def update_geometric_awareness(self, task_metrics: dict[str, Any] | None = None) -> GeometricSelfState:
        """Update geometric self-awareness state.

        Args:
            task_metrics: Optional recent task performance metrics

        Returns:
            Updated geometric state
        """
        try:
            # Gather system resource metrics
            resource_metrics = await self._gather_resource_metrics()

            # Include task performance if provided
            if task_metrics:
                self._task_history.append({**task_metrics, "timestamp": time.time()})

                # Keep recent history only
                cutoff_time = time.time() - 300  # Last 5 minutes
                self._task_history = [t for t in self._task_history if t.get("timestamp", 0) > cutoff_time]

                # Update performance metrics based on recent tasks
                if self._task_history:
                    recent_latencies = [t.get("latency_ms", 0) for t in self._task_history]
                    recent_accuracies = [t.get("accuracy", 1.0) for t in self._task_history]

                    resource_metrics.response_latency_ms = sum(recent_latencies) / len(recent_latencies)
                    resource_metrics.accuracy_score = sum(recent_accuracies) / len(recent_accuracies)

            # Determine geometric state based on metrics
            geometric_state = self._calculate_geometric_state(resource_metrics)

            # Calculate stability and adaptation metrics
            stability_score = self._calculate_stability_score(resource_metrics)
            adaptation_rate = 0.1  # Could be dynamic based on recent changes

            # Create complete state snapshot
            current_geometric_state = GeometricSelfState(
                timestamp=datetime.now(),
                geometric_state=geometric_state,
                resource_metrics=resource_metrics,
                adaptation_rate=adaptation_rate,
                stability_score=stability_score,
                optimization_direction=self._determine_optimization_direction(resource_metrics),
            )

            self._current_geometric_state = current_geometric_state
            self._geometric_history.append(current_geometric_state)

            # Maintain history size
            if len(self._geometric_history) > self.config.max_history_entries:
                self._geometric_history = self._geometric_history[-self.config.max_history_entries :]

            # Update agent state based on geometric health
            if not current_geometric_state.is_healthy():
                if geometric_state == GeometricState.OVERLOADED:
                    self.set_state(AgentState.OVERLOADED, "geometric state unhealthy")
                elif geometric_state == GeometricState.ADAPTING:
                    self.set_state(AgentState.ADAPTING, "geometric adaptation in progress")
                else:
                    self.set_state(AgentState.ERROR, "geometric state indicates issues")
            elif self._current_state in [AgentState.OVERLOADED, AgentState.ERROR]:
                # Recovery to healthy state
                self.set_state(AgentState.IDLE, "geometric state recovered")

            logger.debug(f"Geometric awareness updated: {geometric_state.value}")
            return current_geometric_state

        except Exception as e:
            logger.error(f"Failed to update geometric awareness: {e}")
            # Return fallback state
            return GeometricSelfState(
                timestamp=datetime.now(), geometric_state=GeometricState.BALANCED, resource_metrics=ResourceMetrics()
            )

    def record_task_performance(
        self, task_id: str, latency_ms: float, accuracy: float = 1.0, status: str = "success"
    ) -> None:
        """Record task performance for geometric awareness.

        Args:
            task_id: Unique task identifier
            latency_ms: Task completion time in milliseconds
            accuracy: Task accuracy score (0.0 to 1.0)
            status: Task completion status
        """
        task_record = {
            "task_id": task_id,
            "timestamp": time.time(),
            "latency_ms": latency_ms,
            "accuracy": accuracy,
            "status": status,
        }

        self._task_history.append(task_record)

        # Maintain reasonable history size
        if len(self._task_history) > 1000:
            self._task_history = self._task_history[-1000:]

    def get_state_metrics(self) -> dict[str, Any]:
        """Get comprehensive state and performance metrics.

        Returns:
            Dict containing state history, geometric metrics, and health status
        """
        current_time = time.time()
        recent_tasks = [t for t in self._task_history if current_time - t.get("timestamp", 0) < 3600]  # Last hour

        return {
            "current_state": self._current_state.value,
            "geometric_state": (
                self._current_geometric_state.geometric_state.value if self._current_geometric_state else "unknown"
            ),
            "is_healthy": (self._current_geometric_state.is_healthy() if self._current_geometric_state else False),
            "recent_performance": {
                "tasks_completed": len(recent_tasks),
                "average_latency_ms": (
                    sum(t.get("latency_ms", 0) for t in recent_tasks) / len(recent_tasks) if recent_tasks else 0
                ),
                "average_accuracy": (
                    sum(t.get("accuracy", 1.0) for t in recent_tasks) / len(recent_tasks) if recent_tasks else 1.0
                ),
                "error_rate": (
                    sum(1 for t in recent_tasks if t.get("status") == "error") / len(recent_tasks)
                    if recent_tasks
                    else 0
                ),
            },
            "state_transitions": len(self._state_history),
            "geometric_updates": len(self._geometric_history),
            "monitoring_active": self._monitoring_task is not None and not self._monitoring_task.done(),
        }

    def get_health_status(self) -> dict[str, Any]:
        """Get current health status for health checks.

        Returns:
            Dict with detailed health information
        """
        is_healthy = True
        issues = []

        # Check current state
        if self._current_state in [AgentState.ERROR, AgentState.OFFLINE]:
            is_healthy = False
            issues.append(f"Agent in {self._current_state.value} state")

        # Check geometric health
        if self._current_geometric_state and not self._current_geometric_state.is_healthy():
            is_healthy = False
            issues.append("Geometric state indicates resource issues")

        # Check monitoring
        if self.config.geometric_awareness_enabled and (not self._monitoring_task or self._monitoring_task.done()):
            is_healthy = False
            issues.append("Geometric monitoring not active")

        return {
            "healthy": is_healthy,
            "current_state": self._current_state.value,
            "issues": issues,
            "geometric_state": (
                self._current_geometric_state.geometric_state.value if self._current_geometric_state else "unknown"
            ),
            "last_geometric_update": (
                self._current_geometric_state.timestamp.isoformat() if self._current_geometric_state else None
            ),
        }

    async def _periodic_updates(self) -> None:
        """Background task for periodic geometric state updates."""
        while not self._shutdown_requested:
            try:
                await self.update_geometric_awareness()
                await asyncio.sleep(self.config.update_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic geometric update: {e}")
                await asyncio.sleep(self.config.update_interval_seconds)

    async def _gather_resource_metrics(self) -> ResourceMetrics:
        """Gather current system resource metrics."""
        try:
            import psutil

            # CPU and memory utilization
            cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent / 100.0

            # Network activity (simplified - would need more sophisticated measurement)
            network_activity = 0.0  # Placeholder

            return ResourceMetrics(
                cpu_utilization=cpu_percent,
                memory_utilization=memory_percent,
                network_activity=network_activity,
                task_queue_depth=len(self._task_history),
            )

        except ImportError:
            # Fallback if psutil not available
            logger.warning("psutil not available, using fallback resource metrics")
            return ResourceMetrics()

    def _calculate_geometric_state(self, metrics: ResourceMetrics) -> GeometricState:
        """Calculate geometric state based on resource metrics."""
        # High resource utilization
        if metrics.cpu_utilization > 0.8 or metrics.memory_utilization > 0.8:
            return GeometricState.OVERLOADED

        # Very low utilization
        if metrics.cpu_utilization < 0.2 and metrics.task_queue_depth < 2:
            return GeometricState.UNDERUTILIZED

        # Recent adaptations or optimization
        if hasattr(self, "_recent_adaptations") and self._recent_adaptations:
            return GeometricState.ADAPTING

        # Default to balanced
        return GeometricState.BALANCED

    def _calculate_stability_score(self, metrics: ResourceMetrics) -> float:
        """Calculate stability score based on resource variance."""
        # Simple stability calculation - in production would analyze variance over time
        variance_score = 1.0 - abs(metrics.cpu_utilization - 0.5) - abs(metrics.memory_utilization - 0.5)
        return max(0.0, min(1.0, variance_score))

    def _determine_optimization_direction(self, metrics: ResourceMetrics) -> str:
        """Determine what aspect should be optimized based on current metrics."""
        if metrics.response_latency_ms > 1000:
            return "latency"
        elif metrics.accuracy_score < 0.8:
            return "accuracy"
        elif metrics.cpu_utilization > 0.7:
            return "efficiency"
        else:
            return "balanced"
