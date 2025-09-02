"""
Base Service Interface for Fog Computing Services

Provides common patterns for all fog services including:
- Event-driven communication
- Health monitoring
- Configuration management
- Lifecycle management
"""

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
import logging
from typing import Any


class ServiceStatus(Enum):
    """Service status enumeration"""

    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class ServiceEvent:
    """Service event for inter-service communication"""

    def __init__(
        self, event_type: str, source_service: str, data: dict[str, Any], timestamp: datetime | None = None
    ):
        self.event_type = event_type
        self.source_service = source_service
        self.data = data
        self.timestamp = timestamp or datetime.now(UTC)
        self.event_id = f"{source_service}_{event_type}_{self.timestamp.timestamp()}"


@dataclass
class ServiceHealthCheck:
    """Service health check result"""

    service_name: str
    status: ServiceStatus
    last_check: datetime
    error_message: str | None = None
    metrics: dict[str, Any] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class EventBus:
    """Event bus for inter-service communication"""

    def __init__(self):
        self.subscribers: dict[str, list[callable]] = {}
        self.logger = logging.getLogger(f"{__name__}.EventBus")

    def subscribe(self, event_type: str, handler: callable):
        """Subscribe to events of a specific type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
        self.logger.debug(f"Subscribed {handler} to {event_type}")

    async def publish(self, event: ServiceEvent):
        """Publish an event to all subscribers"""
        if event.event_type in self.subscribers:
            tasks = []
            for handler in self.subscribers[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event))
                    else:
                        handler(event)
                except Exception as e:
                    self.logger.error(f"Error in event handler {handler}: {e}")

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        self.logger.debug(f"Published event {event.event_id}")


class BaseFogService(ABC):
    """Base class for all fog computing services"""

    def __init__(self, service_name: str, config: dict[str, Any], event_bus: EventBus):
        self.service_name = service_name
        self.config = config
        self.event_bus = event_bus
        self.logger = logging.getLogger(f"{__name__}.{service_name}")

        # Service state
        self.status = ServiceStatus.STOPPED
        self.start_time: datetime | None = None
        self.last_health_check: datetime | None = None
        self.error_count = 0
        self.metrics: dict[str, Any] = {}

        # Background tasks
        self.background_tasks: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the service. Return True if successful."""
        pass

    @abstractmethod
    async def cleanup(self) -> bool:
        """Cleanup service resources. Return True if successful."""
        pass

    @abstractmethod
    async def health_check(self) -> ServiceHealthCheck:
        """Perform health check. Return health status."""
        pass

    async def start(self) -> bool:
        """Start the service"""
        try:
            self.logger.info(f"Starting {self.service_name}")
            self.status = ServiceStatus.STARTING

            # Initialize the service
            success = await self.initialize()
            if not success:
                self.status = ServiceStatus.ERROR
                return False

            # Start background tasks
            await self._start_background_tasks()

            self.status = ServiceStatus.RUNNING
            self.start_time = datetime.now(UTC)

            # Publish service started event
            event = ServiceEvent("service_started", self.service_name, {"status": self.status.value})
            await self.event_bus.publish(event)

            self.logger.info(f"{self.service_name} started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start {self.service_name}: {e}")
            self.status = ServiceStatus.ERROR
            return False

    async def stop(self) -> bool:
        """Stop the service"""
        try:
            self.logger.info(f"Stopping {self.service_name}")
            self.status = ServiceStatus.STOPPING

            # Signal shutdown to background tasks
            self._shutdown_event.set()

            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()

            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)

            # Cleanup resources
            success = await self.cleanup()

            self.status = ServiceStatus.STOPPED

            # Publish service stopped event
            event = ServiceEvent("service_stopped", self.service_name, {"status": self.status.value})
            await self.event_bus.publish(event)

            self.logger.info(f"{self.service_name} stopped successfully")
            return success

        except Exception as e:
            self.logger.error(f"Error stopping {self.service_name}: {e}")
            self.status = ServiceStatus.ERROR
            return False

    async def restart(self) -> bool:
        """Restart the service"""
        success = await self.stop()
        if success:
            success = await self.start()
        return success

    def get_status(self) -> dict[str, Any]:
        """Get service status information"""
        return {
            "service_name": self.service_name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "error_count": self.error_count,
            "metrics": self.metrics.copy(),
        }

    def add_background_task(self, coro, name: str | None = None):
        """Add a background task to the service"""
        task = asyncio.create_task(coro)
        if name:
            task.set_name(f"{self.service_name}_{name}")
        self.background_tasks.append(task)
        return task

    async def _start_background_tasks(self):
        """Start service-specific background tasks"""
        # Health check task
        self.add_background_task(self._health_check_loop(), "health_check")

    async def _health_check_loop(self):
        """Background task for periodic health checks"""
        while not self._shutdown_event.is_set():
            try:
                health = await self.health_check()
                self.last_health_check = health.last_check

                if health.status != ServiceStatus.RUNNING:
                    self.error_count += 1
                    self.logger.warning(f"Health check failed: {health.error_message}")

                # Publish health check event
                event = ServiceEvent(
                    "health_check",
                    self.service_name,
                    {"status": health.status.value, "error_message": health.error_message, "metrics": health.metrics},
                )
                await self.event_bus.publish(event)

                await asyncio.sleep(60)  # Health check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                self.error_count += 1
                await asyncio.sleep(30)

    async def publish_event(self, event_type: str, data: dict[str, Any]):
        """Publish an event from this service"""
        event = ServiceEvent(event_type, self.service_name, data)
        await self.event_bus.publish(event)

    def subscribe_to_events(self, event_type: str, handler: callable):
        """Subscribe to events from other services"""
        self.event_bus.subscribe(event_type, handler)
