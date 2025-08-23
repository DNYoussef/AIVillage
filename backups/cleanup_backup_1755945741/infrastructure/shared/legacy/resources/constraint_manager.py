"""Resource Constraint Manager for Evolution Tasks."""

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .device_profiler import DeviceProfiler, DeviceType, ResourceSnapshot, ThermalState

logger = logging.getLogger(__name__)


class ConstraintSeverity(Enum):
    """Severity levels for resource constraints."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConstraintType(Enum):
    """Types of resource constraints."""

    MEMORY = "memory"
    CPU = "cpu"
    BATTERY = "battery"
    THERMAL = "thermal"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class ResourceConstraints:
    """Resource constraints for evolution tasks."""

    # Memory constraints (in MB)
    max_memory_mb: int
    memory_warning_mb: int
    memory_critical_mb: int

    # CPU constraints (in percentage)
    max_cpu_percent: float
    cpu_warning_percent: float
    cpu_critical_percent: float

    # Battery constraints (in percentage, None if no battery)
    min_battery_percent: float | None = None
    battery_warning_percent: float | None = None

    # Thermal constraints (in Celsius, None if no temp sensor)
    max_temperature_celsius: float | None = None
    temperature_warning_celsius: float | None = None

    # Storage constraints (in GB)
    min_free_storage_gb: float = 1.0
    storage_warning_gb: float = 2.0

    # Time-based constraints
    max_execution_time_minutes: int = 60
    max_idle_time_minutes: int = 5

    # Device-specific constraints
    require_charging: bool = False
    require_wifi: bool = False
    allow_mobile_data: bool = True

    # Priority and scheduling
    priority_level: int = 1  # 1-5, higher = more important
    can_interrupt: bool = True
    can_pause: bool = True
    can_resume: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert constraints to dictionary."""
        return {
            "max_memory_mb": self.max_memory_mb,
            "memory_warning_mb": self.memory_warning_mb,
            "memory_critical_mb": self.memory_critical_mb,
            "max_cpu_percent": self.max_cpu_percent,
            "cpu_warning_percent": self.cpu_warning_percent,
            "cpu_critical_percent": self.cpu_critical_percent,
            "min_battery_percent": self.min_battery_percent,
            "battery_warning_percent": self.battery_warning_percent,
            "max_temperature_celsius": self.max_temperature_celsius,
            "temperature_warning_celsius": self.temperature_warning_celsius,
            "min_free_storage_gb": self.min_free_storage_gb,
            "storage_warning_gb": self.storage_warning_gb,
            "max_execution_time_minutes": self.max_execution_time_minutes,
            "max_idle_time_minutes": self.max_idle_time_minutes,
            "require_charging": self.require_charging,
            "require_wifi": self.require_wifi,
            "allow_mobile_data": self.allow_mobile_data,
            "priority_level": self.priority_level,
            "can_interrupt": self.can_interrupt,
            "can_pause": self.can_pause,
            "can_resume": self.can_resume,
        }


@dataclass
class ConstraintViolation:
    """Represents a constraint violation."""

    constraint_type: ConstraintType
    severity: ConstraintSeverity
    current_value: float
    limit_value: float
    message: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_type": self.constraint_type.value,
            "severity": self.severity.value,
            "current_value": self.current_value,
            "limit_value": self.limit_value,
            "message": self.message,
            "timestamp": self.timestamp,
        }


class ConstraintManager:
    """Manages resource constraints for evolution tasks."""

    def __init__(self, device_profiler: DeviceProfiler) -> None:
        self.device_profiler = device_profiler

        # Default constraints based on device profile
        self.default_constraints = self._create_default_constraints()

        # Active task constraints
        self.active_tasks: dict[str, ResourceConstraints] = {}
        self.task_start_times: dict[str, float] = {}
        self.task_violations: dict[str, list[ConstraintViolation]] = {}

        # Constraint monitoring
        self.monitoring_active = False
        self.monitoring_task: asyncio.Task | None = None
        self.check_interval = 2.0  # Check constraints every 2 seconds

        # Violation callbacks
        self.violation_callbacks: list[Callable[[str, ConstraintViolation], None]] = []
        self.enforcement_callbacks: list[Callable[[str, str], None]] = []  # task_id, action

        # Statistics
        self.stats = {
            "constraints_checked": 0,
            "violations_detected": 0,
            "tasks_interrupted": 0,
            "tasks_paused": 0,
            "tasks_resumed": 0,
            "enforcement_actions": 0,
        }

        # Pre-defined constraint templates
        self.constraint_templates = self._create_constraint_templates()

    def _create_default_constraints(self) -> ResourceConstraints:
        """Create default constraints based on device profile."""
        profile = self.device_profiler.profile

        # Calculate memory constraints (conservative)
        total_memory_mb = int(profile.total_memory_gb * 1024)
        max_memory = min(
            total_memory_mb // 2,  # Max 50% of total memory
            profile.max_evolution_memory_mb or (total_memory_mb // 2),
        )

        # CPU constraints
        max_cpu = profile.max_evolution_cpu_percent or 70.0

        # Battery constraints (if applicable)
        min_battery = None
        battery_warning = None
        if profile.device_type in [DeviceType.PHONE, DeviceType.TABLET]:
            min_battery = 15.0 if not profile.battery_optimization else 25.0
            battery_warning = 20.0 if not profile.battery_optimization else 30.0

        # Thermal constraints
        max_temp = None
        temp_warning = None
        if profile.thermal_throttling:
            max_temp = 80.0
            temp_warning = 75.0

        return ResourceConstraints(
            max_memory_mb=max_memory,
            memory_warning_mb=int(max_memory * 0.8),
            memory_critical_mb=int(max_memory * 0.95),
            max_cpu_percent=max_cpu,
            cpu_warning_percent=max_cpu * 0.8,
            cpu_critical_percent=max_cpu * 0.95,
            min_battery_percent=min_battery,
            battery_warning_percent=battery_warning,
            max_temperature_celsius=max_temp,
            temperature_warning_celsius=temp_warning,
            min_free_storage_gb=2.0,
            storage_warning_gb=5.0,
            max_execution_time_minutes=90,
            require_charging=profile.battery_optimization,
            can_interrupt=True,
            can_pause=True,
            can_resume=True,
        )

    def _create_constraint_templates(self) -> dict[str, ResourceConstraints]:
        """Create pre-defined constraint templates."""
        base = self.default_constraints

        return {
            "nightly": ResourceConstraints(
                max_memory_mb=base.max_memory_mb,
                memory_warning_mb=base.memory_warning_mb,
                memory_critical_mb=base.memory_critical_mb,
                max_cpu_percent=base.max_cpu_percent,
                cpu_warning_percent=base.cpu_warning_percent,
                cpu_critical_percent=base.cpu_critical_percent,
                min_battery_percent=base.min_battery_percent,
                battery_warning_percent=base.battery_warning_percent,
                max_temperature_celsius=base.max_temperature_celsius,
                temperature_warning_celsius=base.temperature_warning_celsius,
                max_execution_time_minutes=120,  # 2 hours for nightly
                priority_level=2,
                can_interrupt=True,
                can_pause=True,
            ),
            "breakthrough": ResourceConstraints(
                max_memory_mb=int(base.max_memory_mb * 1.2),  # Allow more memory
                memory_warning_mb=int(base.memory_warning_mb * 1.2),
                memory_critical_mb=int(base.memory_critical_mb * 1.2),
                max_cpu_percent=min(85.0, base.max_cpu_percent + 10),  # Allow more CPU
                cpu_warning_percent=min(75.0, base.cpu_warning_percent + 10),
                cpu_critical_percent=min(80.0, base.cpu_critical_percent + 10),
                min_battery_percent=25.0 if base.min_battery_percent else None,
                battery_warning_percent=30.0 if base.battery_warning_percent else None,
                max_temperature_celsius=base.max_temperature_celsius,
                temperature_warning_celsius=base.temperature_warning_celsius,
                max_execution_time_minutes=180,  # 3 hours for breakthrough
                priority_level=4,
                can_interrupt=False,  # Don't interrupt breakthrough evolution
                can_pause=True,
            ),
            "emergency": ResourceConstraints(
                max_memory_mb=int(base.max_memory_mb * 0.8),  # More conservative
                memory_warning_mb=int(base.memory_warning_mb * 0.8),
                memory_critical_mb=int(base.memory_critical_mb * 0.8),
                max_cpu_percent=base.max_cpu_percent * 0.9,
                cpu_warning_percent=base.cpu_warning_percent * 0.9,
                cpu_critical_percent=base.cpu_critical_percent * 0.9,
                min_battery_percent=10.0 if base.min_battery_percent else None,
                battery_warning_percent=15.0 if base.battery_warning_percent else None,
                max_temperature_celsius=base.max_temperature_celsius,
                temperature_warning_celsius=base.temperature_warning_celsius,
                max_execution_time_minutes=30,  # Quick emergency evolution
                priority_level=5,
                can_interrupt=False,
                can_pause=False,  # Don't pause emergency evolution
            ),
            "lightweight": ResourceConstraints(
                max_memory_mb=int(base.max_memory_mb * 0.5),  # Half memory
                memory_warning_mb=int(base.memory_warning_mb * 0.5),
                memory_critical_mb=int(base.memory_critical_mb * 0.5),
                max_cpu_percent=base.max_cpu_percent * 0.6,  # 60% CPU
                cpu_warning_percent=base.cpu_warning_percent * 0.6,
                cpu_critical_percent=base.cpu_critical_percent * 0.6,
                min_battery_percent=10.0 if base.min_battery_percent else None,
                battery_warning_percent=15.0 if base.battery_warning_percent else None,
                max_temperature_celsius=base.max_temperature_celsius,
                temperature_warning_celsius=base.temperature_warning_celsius,
                max_execution_time_minutes=45,
                priority_level=1,
                can_interrupt=True,
                can_pause=True,
            ),
        }

    async def start_constraint_monitoring(self) -> None:
        """Start constraint monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Constraint monitoring started")

    async def stop_constraint_monitoring(self) -> None:
        """Stop constraint monitoring."""
        self.monitoring_active = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitoring_task

        logger.info("Constraint monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main constraint monitoring loop."""
        while self.monitoring_active:
            try:
                await self._check_all_constraints()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.exception(f"Error in constraint monitoring: {e}")
                await asyncio.sleep(self.check_interval)

    async def _check_all_constraints(self) -> None:
        """Check constraints for all active tasks."""
        if not self.active_tasks:
            return

        current_snapshot = self.device_profiler.current_snapshot
        if not current_snapshot:
            return

        for task_id, constraints in self.active_tasks.items():
            violations = await self._check_task_constraints(task_id, constraints, current_snapshot)

            if violations:
                self.task_violations[task_id] = violations
                await self._handle_violations(task_id, violations)
            elif task_id in self.task_violations:
                # Clear previous violations
                del self.task_violations[task_id]

        self.stats["constraints_checked"] += 1

    async def _check_task_constraints(
        self, task_id: str, constraints: ResourceConstraints, snapshot: ResourceSnapshot
    ) -> list[ConstraintViolation]:
        """Check constraints for a specific task."""
        violations = []

        # Memory constraints
        memory_used_mb = snapshot.memory_used / (1024 * 1024)
        if memory_used_mb > constraints.max_memory_mb:
            violations.append(
                ConstraintViolation(
                    constraint_type=ConstraintType.MEMORY,
                    severity=ConstraintSeverity.CRITICAL,
                    current_value=memory_used_mb,
                    limit_value=constraints.max_memory_mb,
                    message=f"Memory usage {memory_used_mb:.0f}MB exceeds limit {constraints.max_memory_mb}MB",
                )
            )
        elif memory_used_mb > constraints.memory_warning_mb:
            violations.append(
                ConstraintViolation(
                    constraint_type=ConstraintType.MEMORY,
                    severity=ConstraintSeverity.MEDIUM,
                    current_value=memory_used_mb,
                    limit_value=constraints.memory_warning_mb,
                    message=f"Memory usage {memory_used_mb:.0f}MB exceeds warning threshold {constraints.memory_warning_mb}MB",
                )
            )

        # CPU constraints
        if snapshot.cpu_percent > constraints.max_cpu_percent:
            violations.append(
                ConstraintViolation(
                    constraint_type=ConstraintType.CPU,
                    severity=ConstraintSeverity.CRITICAL,
                    current_value=snapshot.cpu_percent,
                    limit_value=constraints.max_cpu_percent,
                    message=f"CPU usage {snapshot.cpu_percent:.1f}% exceeds limit {constraints.max_cpu_percent:.1f}%",
                )
            )
        elif snapshot.cpu_percent > constraints.cpu_warning_percent:
            violations.append(
                ConstraintViolation(
                    constraint_type=ConstraintType.CPU,
                    severity=ConstraintSeverity.MEDIUM,
                    current_value=snapshot.cpu_percent,
                    limit_value=constraints.cpu_warning_percent,
                    message=f"CPU usage {snapshot.cpu_percent:.1f}% exceeds warning threshold {constraints.cpu_warning_percent:.1f}%",
                )
            )

        # Battery constraints
        if constraints.min_battery_percent is not None and snapshot.battery_percent is not None:
            if snapshot.battery_percent < constraints.min_battery_percent:
                violations.append(
                    ConstraintViolation(
                        constraint_type=ConstraintType.BATTERY,
                        severity=ConstraintSeverity.CRITICAL,
                        current_value=snapshot.battery_percent,
                        limit_value=constraints.min_battery_percent,
                        message=f"Battery {snapshot.battery_percent:.1f}% below minimum {constraints.min_battery_percent:.1f}%",
                    )
                )
            elif (
                constraints.battery_warning_percent is not None
                and snapshot.battery_percent < constraints.battery_warning_percent
            ):
                violations.append(
                    ConstraintViolation(
                        constraint_type=ConstraintType.BATTERY,
                        severity=ConstraintSeverity.MEDIUM,
                        current_value=snapshot.battery_percent,
                        limit_value=constraints.battery_warning_percent,
                        message=f"Battery {snapshot.battery_percent:.1f}% below warning threshold {constraints.battery_warning_percent:.1f}%",
                    )
                )

        # Thermal constraints
        if constraints.max_temperature_celsius is not None and snapshot.cpu_temp is not None:
            if snapshot.cpu_temp > constraints.max_temperature_celsius:
                violations.append(
                    ConstraintViolation(
                        constraint_type=ConstraintType.THERMAL,
                        severity=ConstraintSeverity.CRITICAL,
                        current_value=snapshot.cpu_temp,
                        limit_value=constraints.max_temperature_celsius,
                        message=f"Temperature {snapshot.cpu_temp:.1f}°C exceeds limit {constraints.max_temperature_celsius:.1f}°C",
                    )
                )
            elif (
                constraints.temperature_warning_celsius is not None
                and snapshot.cpu_temp > constraints.temperature_warning_celsius
            ):
                violations.append(
                    ConstraintViolation(
                        constraint_type=ConstraintType.THERMAL,
                        severity=ConstraintSeverity.MEDIUM,
                        current_value=snapshot.cpu_temp,
                        limit_value=constraints.temperature_warning_celsius,
                        message=f"Temperature {snapshot.cpu_temp:.1f}°C exceeds warning threshold {constraints.temperature_warning_celsius:.1f}°C",
                    )
                )

        # Storage constraints
        storage_free_gb = snapshot.storage_free / (1024**3)
        if storage_free_gb < constraints.min_free_storage_gb:
            violations.append(
                ConstraintViolation(
                    constraint_type=ConstraintType.STORAGE,
                    severity=ConstraintSeverity.HIGH,
                    current_value=storage_free_gb,
                    limit_value=constraints.min_free_storage_gb,
                    message=f"Free storage {storage_free_gb:.1f}GB below minimum {constraints.min_free_storage_gb:.1f}GB",
                )
            )

        # Time constraints
        if task_id in self.task_start_times:
            elapsed_minutes = (time.time() - self.task_start_times[task_id]) / 60
            if elapsed_minutes > constraints.max_execution_time_minutes:
                violations.append(
                    ConstraintViolation(
                        constraint_type=ConstraintType.CPU,  # Use CPU as general constraint type
                        severity=ConstraintSeverity.HIGH,
                        current_value=elapsed_minutes,
                        limit_value=constraints.max_execution_time_minutes,
                        message=f"Task running for {elapsed_minutes:.1f} minutes exceeds limit {constraints.max_execution_time_minutes} minutes",
                    )
                )

        # Power constraints
        if constraints.require_charging and not snapshot.power_plugged:
            violations.append(
                ConstraintViolation(
                    constraint_type=ConstraintType.BATTERY,
                    severity=ConstraintSeverity.HIGH,
                    current_value=0.0,
                    limit_value=1.0,
                    message="Task requires charging but device is not plugged in",
                )
            )

        return violations

    async def _handle_violations(self, task_id: str, violations: list[ConstraintViolation]) -> None:
        """Handle constraint violations."""
        critical_violations = [v for v in violations if v.severity == ConstraintSeverity.CRITICAL]
        high_violations = [v for v in violations if v.severity == ConstraintSeverity.HIGH]

        self.stats["violations_detected"] += len(violations)

        # Determine enforcement action
        constraints = self.active_tasks.get(task_id)
        if not constraints:
            return

        action = None

        if critical_violations:
            if constraints.can_interrupt:
                action = "interrupt"
                self.stats["tasks_interrupted"] += 1
            elif constraints.can_pause:
                action = "pause"
                self.stats["tasks_paused"] += 1
        elif high_violations and constraints.can_pause:
            action = "pause"
            self.stats["tasks_paused"] += 1

        # Execute enforcement action
        if action:
            self.stats["enforcement_actions"] += 1
            await self._enforce_action(task_id, action)

        # Notify callbacks
        for violation in violations:
            for callback in self.violation_callbacks:
                try:
                    await callback(task_id, violation)
                except Exception as e:
                    logger.exception(f"Error in violation callback: {e}")

    async def _enforce_action(self, task_id: str, action: str) -> None:
        """Enforce constraint action."""
        logger.warning(f"Enforcing constraint action '{action}' for task {task_id}")

        # Notify enforcement callbacks
        for callback in self.enforcement_callbacks:
            try:
                await callback(task_id, action)
            except Exception as e:
                logger.exception(f"Error in enforcement callback: {e}")

    def register_task(
        self,
        task_id: str,
        evolution_type: str = "nightly",
        custom_constraints: ResourceConstraints | None = None,
    ) -> bool:
        """Register a task with constraints."""
        if task_id in self.active_tasks:
            logger.warning(f"Task {task_id} already registered")
            return False

        # Get constraints
        if custom_constraints:
            constraints = custom_constraints
        elif evolution_type in self.constraint_templates:
            constraints = self.constraint_templates[evolution_type]
        else:
            constraints = self.default_constraints

        # Check if resources are available
        if not self._check_resource_availability(constraints):
            logger.warning(f"Insufficient resources for task {task_id}")
            return False

        # Register task
        self.active_tasks[task_id] = constraints
        self.task_start_times[task_id] = time.time()

        logger.info(f"Task {task_id} registered with {evolution_type} constraints")
        return True

    def unregister_task(self, task_id: str) -> None:
        """Unregister a task."""
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
        if task_id in self.task_start_times:
            del self.task_start_times[task_id]
        if task_id in self.task_violations:
            del self.task_violations[task_id]

        logger.info(f"Task {task_id} unregistered")

    def _check_resource_availability(self, constraints: ResourceConstraints) -> bool:
        """Check if resources are available for task."""
        current_snapshot = self.device_profiler.current_snapshot
        if not current_snapshot:
            return False

        # Check memory availability
        available_memory_mb = current_snapshot.memory_available / (1024 * 1024)
        if available_memory_mb < constraints.max_memory_mb * 0.5:  # Need at least 50% of requested memory
            return False

        # Check CPU availability
        if current_snapshot.cpu_percent > 80:  # Too much CPU usage already
            return False

        # Check battery requirements
        if (
            constraints.min_battery_percent is not None
            and current_snapshot.battery_percent is not None
            and current_snapshot.battery_percent < constraints.min_battery_percent
        ):
            return False

        # Check charging requirements
        if constraints.require_charging and not current_snapshot.power_plugged:
            return False

        # Check thermal state
        return current_snapshot.thermal_state not in [
            ThermalState.HOT,
            ThermalState.CRITICAL,
            ThermalState.THROTTLING,
        ]

    def can_resume_task(self, task_id: str) -> bool:
        """Check if a paused task can be resumed."""
        if task_id not in self.active_tasks:
            return False

        constraints = self.active_tasks[task_id]
        return self._check_resource_availability(constraints)

    def get_constraint_template(self, evolution_type: str) -> ResourceConstraints | None:
        """Get constraint template for evolution type."""
        return self.constraint_templates.get(evolution_type)

    def update_constraints(self, task_id: str, new_constraints: ResourceConstraints) -> None:
        """Update constraints for active task."""
        if task_id in self.active_tasks:
            self.active_tasks[task_id] = new_constraints
            logger.info(f"Updated constraints for task {task_id}")

    def register_violation_callback(self, callback: Callable[[str, ConstraintViolation], None]) -> None:
        """Register violation callback."""
        self.violation_callbacks.append(callback)

    def register_enforcement_callback(self, callback: Callable[[str, str], None]) -> None:
        """Register enforcement callback."""
        self.enforcement_callbacks.append(callback)

    def get_active_tasks(self) -> dict[str, dict[str, Any]]:
        """Get information about active tasks."""
        result = {}

        for task_id, constraints in self.active_tasks.items():
            violations = self.task_violations.get(task_id, [])
            elapsed_time = time.time() - self.task_start_times.get(task_id, time.time())

            result[task_id] = {
                "constraints": constraints.to_dict(),
                "violations": [v.to_dict() for v in violations],
                "elapsed_minutes": elapsed_time / 60,
                "has_violations": len(violations) > 0,
            }

        return result

    def get_constraint_stats(self) -> dict[str, Any]:
        """Get constraint management statistics."""
        return {
            **self.stats,
            "active_tasks": len(self.active_tasks),
            "tasks_with_violations": len(self.task_violations),
            "monitoring_active": self.monitoring_active,
            "check_interval": self.check_interval,
            "violation_callbacks": len(self.violation_callbacks),
            "enforcement_callbacks": len(self.enforcement_callbacks),
        }
