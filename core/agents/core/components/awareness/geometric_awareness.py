"""Geometric self-awareness system for agents.

Provides proprioception-like biofeedback for agents to understand
their resource utilization and performance state. Extracted from
BaseAgentTemplate following single responsibility principle.
"""

from dataclasses import dataclass
from datetime import datetime
import logging
import time
from typing import Any

from ..agent_constants import AgentConstants, GeometricState, GeometricStateConstants

logger = logging.getLogger(__name__)


@dataclass
class GeometricSelfState:
    """Geometric self-awareness state (proprioception-like biofeedback)."""

    timestamp: datetime
    geometric_state: GeometricState

    # Resource awareness (like proprioception for humans)
    cpu_utilization: float  # 0.0 to 1.0
    memory_utilization: float  # 0.0 to 1.0
    network_activity: float  # 0.0 to 1.0
    task_queue_depth: int

    # Performance metrics (like physical awareness)
    response_latency_ms: float
    accuracy_score: float  # Recent task accuracy
    energy_efficiency: float  # Performance per resource unit

    # Self-modification metrics (ADAS-related)
    adaptation_rate: float  # How quickly changing
    stability_score: float  # How stable current config is
    optimization_direction: str  # What aspect being optimized

    def is_healthy(self) -> bool:
        """Determine if agent is in healthy geometric state.

        Uses configurable thresholds from constants.

        Returns:
            True if agent state is healthy
        """
        return (
            self.cpu_utilization < AgentConstants.MAX_CPU_UTILIZATION
            and self.memory_utilization < AgentConstants.MAX_MEMORY_UTILIZATION
            and self.response_latency_ms < AgentConstants.MAX_RESPONSE_LATENCY_MS
            and self.accuracy_score > AgentConstants.MIN_ACCURACY_SCORE
            and self.stability_score > AgentConstants.MIN_STABILITY_SCORE
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "geometric_state": self.geometric_state.value,
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "network_activity": self.network_activity,
            "task_queue_depth": self.task_queue_depth,
            "response_latency_ms": self.response_latency_ms,
            "accuracy_score": self.accuracy_score,
            "energy_efficiency": self.energy_efficiency,
            "adaptation_rate": self.adaptation_rate,
            "stability_score": self.stability_score,
            "optimization_direction": self.optimization_direction,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeometricSelfState":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            geometric_state=GeometricState(data["geometric_state"]),
            cpu_utilization=data["cpu_utilization"],
            memory_utilization=data["memory_utilization"],
            network_activity=data["network_activity"],
            task_queue_depth=data["task_queue_depth"],
            response_latency_ms=data["response_latency_ms"],
            accuracy_score=data["accuracy_score"],
            energy_efficiency=data["energy_efficiency"],
            adaptation_rate=data["adaptation_rate"],
            stability_score=data["stability_score"],
            optimization_direction=data["optimization_direction"],
        )


class GeometricAwarenessManager:
    """Manages geometric self-awareness and proprioception-like monitoring.

    Single responsibility: Monitor and analyze agent resource utilization,
    performance state, and provide health assessments.
    """

    def __init__(self, max_history: int | None = None):
        """Initialize geometric awareness manager.

        Args:
            max_history: Maximum number of states to keep in history
        """
        self._state_history: list[GeometricSelfState] = []
        self._current_state: GeometricSelfState | None = None
        self._max_history = max_history or AgentConstants.MAX_GEOMETRIC_STATES
        self._task_history: list[dict[str, Any]] = []
        self._modification_history: list[dict[str, Any]] = []

    def update_geometric_state(
        self,
        task_history: list[dict[str, Any]],
        modification_history: list[dict[str, Any]],
        adas_config: dict[str, Any],
    ) -> GeometricSelfState:
        """Update geometric self-awareness state.

        Args:
            task_history: Recent task performance history
            modification_history: ADAS modification history
            adas_config: Current ADAS configuration

        Returns:
            Updated geometric state
        """
        try:
            # Gather system metrics
            cpu_percent, memory_percent = self._get_system_metrics()

            # Analyze recent performance
            current_time = time.time()
            recent_tasks = self._filter_recent_tasks(task_history, current_time)

            performance_metrics = self._calculate_performance_metrics(recent_tasks)

            # Determine geometric state
            state = self._determine_geometric_state(
                cpu_percent, memory_percent, recent_tasks, modification_history, current_time
            )

            # Calculate energy efficiency
            total_utilization = cpu_percent + memory_percent
            energy_efficiency = performance_metrics["avg_accuracy"] / max(
                total_utilization, GeometricStateConstants.EFFICIENCY_UTILIZATION_MIN
            )

            # Calculate stability score
            stability_score = self._calculate_stability_score(cpu_percent, memory_percent)

            # Create geometric state snapshot
            geometric_state = GeometricSelfState(
                timestamp=datetime.now(),
                geometric_state=state,
                cpu_utilization=cpu_percent,
                memory_utilization=memory_percent,
                network_activity=0.0,  # Placeholder - would measure actual network I/O
                task_queue_depth=len(task_history) - len(recent_tasks),
                response_latency_ms=performance_metrics["avg_latency"],
                accuracy_score=performance_metrics["avg_accuracy"],
                energy_efficiency=energy_efficiency,
                adaptation_rate=adas_config.get("adaptation_rate", 0.0),
                stability_score=stability_score,
                optimization_direction=adas_config.get("current_optimization", "balanced"),
            )

            self._state_history.append(geometric_state)
            self._current_state = geometric_state

            # Maintain history size limit
            if len(self._state_history) > self._max_history:
                self._state_history = self._state_history[-self._max_history :]

            logger.debug(f"Geometric awareness updated: {state.value}")
            return geometric_state

        except Exception as e:
            logger.error(f"Failed to update geometric self-awareness: {e}")
            # Return a default state in case of error
            return self._create_default_state()

    def _get_system_metrics(self) -> tuple[float, float]:
        """Get current system resource metrics.

        Returns:
            Tuple of (cpu_percent, memory_percent) as 0.0-1.0 values
        """
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent / 100.0

            return cpu_percent, memory_percent
        except ImportError:
            logger.warning("psutil not available, using mock system metrics")
            return 0.5, 0.5  # Default moderate utilization
        except Exception as e:
            logger.warning(f"Failed to get system metrics: {e}")
            return 0.5, 0.5

    def _filter_recent_tasks(self, task_history: list[dict[str, Any]], current_time: float) -> list[dict[str, Any]]:
        """Filter tasks from recent performance window.

        Args:
            task_history: Full task history
            current_time: Current timestamp

        Returns:
            List of recent tasks
        """
        return [
            task
            for task in task_history
            if current_time - task.get("timestamp", 0) < AgentConstants.PERFORMANCE_METRIC_WINDOW
        ]

    def _calculate_performance_metrics(self, recent_tasks: list[dict[str, Any]]) -> dict[str, float]:
        """Calculate performance metrics from recent tasks.

        Args:
            recent_tasks: List of recent task records

        Returns:
            Dictionary with performance metrics
        """
        if not recent_tasks:
            return {
                "avg_latency": 0.0,
                "avg_accuracy": 1.0,
            }

        total_latency = sum(task.get("latency_ms", 0) for task in recent_tasks)
        avg_latency = total_latency / len(recent_tasks)

        accuracy_scores = [task.get("accuracy", 1.0) for task in recent_tasks if "accuracy" in task]
        avg_accuracy = sum(accuracy_scores) / max(len(accuracy_scores), 1)

        return {
            "avg_latency": avg_latency,
            "avg_accuracy": avg_accuracy,
        }

    def _determine_geometric_state(
        self,
        cpu_percent: float,
        memory_percent: float,
        recent_tasks: list[dict[str, Any]],
        modification_history: list[dict[str, Any]],
        current_time: float,
    ) -> GeometricState:
        """Determine the current geometric state based on metrics.

        Args:
            cpu_percent: CPU utilization (0.0-1.0)
            memory_percent: Memory utilization (0.0-1.0)
            recent_tasks: Recent task history
            modification_history: ADAS modification history
            current_time: Current timestamp

        Returns:
            Determined geometric state
        """
        # Check for overloaded state
        if (
            cpu_percent > GeometricStateConstants.OVERLOADED_CPU_THRESHOLD
            or memory_percent > GeometricStateConstants.OVERLOADED_MEMORY_THRESHOLD
        ):
            return GeometricState.OVERLOADED

        # Check for underutilized state
        if (
            cpu_percent < GeometricStateConstants.UNDERUTILIZED_CPU_THRESHOLD
            and len(recent_tasks) < GeometricStateConstants.UNDERUTILIZED_TASK_THRESHOLD
        ):
            return GeometricState.UNDERUTILIZED

        # Check for recent modifications (adapting state)
        if modification_history:
            last_mod = modification_history[-1]
            time_since_mod = current_time - last_mod.get("timestamp", 0)
            if time_since_mod < GeometricStateConstants.RECENT_MODIFICATION_WINDOW:
                return GeometricState.ADAPTING

        # Default to balanced state
        return GeometricState.BALANCED

    def _calculate_stability_score(self, cpu_percent: float, memory_percent: float) -> float:
        """Calculate system stability score.

        Args:
            cpu_percent: CPU utilization
            memory_percent: Memory utilization

        Returns:
            Stability score (0.0-1.0)
        """
        center = GeometricStateConstants.STABILITY_CENTER_POINT
        cpu_deviation = abs(cpu_percent - center)
        memory_deviation = abs(memory_percent - center)

        return 1.0 - cpu_deviation - memory_deviation

    def _create_default_state(self) -> GeometricSelfState:
        """Create a default geometric state for error conditions.

        Returns:
            Default geometric state
        """
        return GeometricSelfState(
            timestamp=datetime.now(),
            geometric_state=GeometricState.BALANCED,
            cpu_utilization=0.5,
            memory_utilization=0.5,
            network_activity=0.0,
            task_queue_depth=0,
            response_latency_ms=1000.0,
            accuracy_score=0.8,
            energy_efficiency=1.0,
            adaptation_rate=0.0,
            stability_score=0.5,
            optimization_direction="unknown",
        )

    def get_current_state(self) -> GeometricSelfState | None:
        """Get the current geometric state.

        Returns:
            Current geometric state or None if not initialized
        """
        return self._current_state

    def get_state_history(self, count: int = 10) -> list[GeometricSelfState]:
        """Get recent state history.

        Args:
            count: Number of recent states to return

        Returns:
            List of recent geometric states
        """
        return self._state_history[-count:] if self._state_history else []

    def is_system_healthy(self) -> bool:
        """Check if the system is currently in a healthy state.

        Returns:
            True if system is healthy
        """
        if not self._current_state:
            return False
        return self._current_state.is_healthy()

    def get_health_summary(self) -> dict[str, Any]:
        """Get comprehensive health summary.

        Returns:
            Dictionary with health metrics and analysis
        """
        if not self._current_state:
            return {
                "status": "unknown",
                "message": "No geometric state available",
            }

        state = self._current_state
        is_healthy = state.is_healthy()

        # Identify health issues
        issues = []
        if state.cpu_utilization >= AgentConstants.MAX_CPU_UTILIZATION:
            issues.append("High CPU utilization")
        if state.memory_utilization >= AgentConstants.MAX_MEMORY_UTILIZATION:
            issues.append("High memory utilization")
        if state.response_latency_ms >= AgentConstants.MAX_RESPONSE_LATENCY_MS:
            issues.append("High response latency")
        if state.accuracy_score <= AgentConstants.MIN_ACCURACY_SCORE:
            issues.append("Low accuracy score")
        if state.stability_score <= AgentConstants.MIN_STABILITY_SCORE:
            issues.append("Low stability")

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "geometric_state": state.geometric_state.value,
            "health_score": self._calculate_health_score(state),
            "issues": issues,
            "metrics": {
                "cpu_utilization": f"{state.cpu_utilization:.2%}",
                "memory_utilization": f"{state.memory_utilization:.2%}",
                "response_latency": f"{state.response_latency_ms:.1f}ms",
                "accuracy": f"{state.accuracy_score:.2%}",
                "stability": f"{state.stability_score:.2f}",
            },
            "timestamp": state.timestamp.isoformat(),
        }

    def _calculate_health_score(self, state: GeometricSelfState) -> float:
        """Calculate overall health score.

        Args:
            state: Geometric state to analyze

        Returns:
            Health score (0.0-1.0)
        """
        # Normalize metrics to 0-1 scale (1 = healthy)
        cpu_score = 1.0 - min(1.0, state.cpu_utilization / AgentConstants.MAX_CPU_UTILIZATION)
        memory_score = 1.0 - min(1.0, state.memory_utilization / AgentConstants.MAX_MEMORY_UTILIZATION)
        latency_score = 1.0 - min(1.0, state.response_latency_ms / AgentConstants.MAX_RESPONSE_LATENCY_MS)
        accuracy_score = state.accuracy_score
        stability_score = state.stability_score

        # Weighted average
        return cpu_score * 0.2 + memory_score * 0.2 + latency_score * 0.2 + accuracy_score * 0.2 + stability_score * 0.2

    def export_state_history(self) -> list[dict[str, Any]]:
        """Export state history as dictionaries.

        Returns:
            List of state dictionaries
        """
        return [state.to_dict() for state in self._state_history]

    def import_state_history(self, state_data: list[dict[str, Any]]) -> int:
        """Import state history from dictionaries.

        Args:
            state_data: List of state dictionaries

        Returns:
            Number of states imported
        """
        imported_count = 0

        for data in state_data:
            try:
                state = GeometricSelfState.from_dict(data)
                self._state_history.append(state)
                imported_count += 1
            except Exception as e:
                logger.warning(f"Failed to import geometric state: {e}")
                continue

        # Update current state to most recent
        if self._state_history:
            self._current_state = self._state_history[-1]

        # Maintain size limit
        if len(self._state_history) > self._max_history:
            self._state_history = self._state_history[-self._max_history :]

        logger.info(f"Imported {imported_count} geometric states")
        return imported_count
