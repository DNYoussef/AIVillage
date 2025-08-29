#!/usr/bin/env python3
"""
Production Service Registry and Discovery System

Implements production-ready service discovery, health monitoring, and load balancing
for the AIVillage distributed architecture.

Key features:
- Service registration and discovery
- Health check monitoring
- Load balancing and routing
- Circuit breaker pattern
- Configuration management
- Service mesh coordination
"""

import asyncio
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import logging
import time
from typing import Any

import aiohttp
from aiohttp import ClientTimeout

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service health status enumeration."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    HEALTH_BASED = "health_based"


@dataclass
class ServiceEndpoint:
    """Service endpoint definition."""

    service_id: str
    name: str
    host: str
    port: int
    protocol: str = "http"
    health_check_path: str = "/health"
    weight: int = 1
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        """Get base URL for this endpoint."""
        return f"{self.protocol}://{self.host}:{self.port}"

    @property
    def health_url(self) -> str:
        """Get health check URL for this endpoint."""
        return f"{self.base_url}{self.health_check_path}"


@dataclass
class ServiceHealth:
    """Service health tracking."""

    service_id: str
    status: ServiceStatus
    last_check: datetime
    response_time_ms: float = 0.0
    error_count: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    circuit_open: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceMetrics:
    """Service performance metrics."""

    service_id: str
    request_count: int = 0
    active_connections: int = 0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 30):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self.last_failure_time and (datetime.now() - self.last_failure_time).seconds > self.timeout_seconds:
                self.state = "half_open"
                return True
            return False
        else:  # half_open
            return True

    def record_success(self):
        """Record successful execution."""
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
        elif self.state == "half_open":
            self.state = "open"


class ServiceRegistry:
    """Production service registry with health monitoring and load balancing."""

    def __init__(
        self, health_check_interval: int = 30, circuit_breaker_threshold: int = 5, circuit_breaker_timeout: int = 30
    ):
        self.services: dict[str, ServiceEndpoint] = {}
        self.service_health: dict[str, ServiceHealth] = {}
        self.service_metrics: dict[str, ServiceMetrics] = {}
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.load_balancer_state: dict[str, dict[str, Any]] = defaultdict(dict)

        self.health_check_interval = health_check_interval
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout

        self._running = False
        self._health_check_task: asyncio.Task | None = None
        self._session: aiohttp.ClientSession | None = None

    async def start(self):
        """Start the service registry."""
        if self._running:
            return

        self._running = True
        self._session = aiohttp.ClientSession(timeout=ClientTimeout(total=10))

        # Start health checking task
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info("Service registry started")

    async def stop(self):
        """Stop the service registry."""
        self._running = False

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._session:
            await self._session.close()

        logger.info("Service registry stopped")

    async def register_service(self, service: ServiceEndpoint) -> bool:
        """Register a service with the registry."""
        try:
            self.services[service.service_id] = service

            # Initialize health tracking
            self.service_health[service.service_id] = ServiceHealth(
                service_id=service.service_id, status=ServiceStatus.STARTING, last_check=datetime.now()
            )

            # Initialize metrics
            self.service_metrics[service.service_id] = ServiceMetrics(service_id=service.service_id)

            # Initialize circuit breaker
            self.circuit_breakers[service.service_id] = CircuitBreaker(
                failure_threshold=self.circuit_breaker_threshold, timeout_seconds=self.circuit_breaker_timeout
            )

            # Initialize load balancer state
            self.load_balancer_state[service.name]["round_robin_index"] = 0

            logger.info(f"Registered service: {service.name} ({service.service_id}) at {service.base_url}")

            # Perform immediate health check
            await self._check_service_health(service)

            return True

        except Exception as e:
            logger.error(f"Failed to register service {service.service_id}: {e}")
            return False

    async def deregister_service(self, service_id: str) -> bool:
        """Deregister a service from the registry."""
        try:
            if service_id in self.services:
                service = self.services.pop(service_id)
                self.service_health.pop(service_id, None)
                self.service_metrics.pop(service_id, None)
                self.circuit_breakers.pop(service_id, None)

                logger.info(f"Deregistered service: {service.name} ({service_id})")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to deregister service {service_id}: {e}")
            return False

    def get_service(
        self, service_name: str, strategy: LoadBalancingStrategy = LoadBalancingStrategy.HEALTH_BASED
    ) -> ServiceEndpoint | None:
        """Get a service endpoint using specified load balancing strategy."""
        candidates = [service for service in self.services.values() if service.name == service_name]

        if not candidates:
            return None

        # Filter out unhealthy services (unless all are unhealthy)
        healthy_candidates = [
            service
            for service in candidates
            if self.service_health.get(service.service_id, {}).status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]
            and not self.circuit_breakers.get(service.service_id, CircuitBreaker()).state == "open"
        ]

        if not healthy_candidates:
            # If no healthy services, use all candidates as fallback
            healthy_candidates = candidates
            logger.warning(f"No healthy instances of {service_name}, using all candidates")

        return self._apply_load_balancing(service_name, healthy_candidates, strategy)

    def _apply_load_balancing(
        self, service_name: str, candidates: list[ServiceEndpoint], strategy: LoadBalancingStrategy
    ) -> ServiceEndpoint:
        """Apply load balancing strategy to select service endpoint."""
        if len(candidates) == 1:
            return candidates[0]

        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            state = self.load_balancer_state[service_name]
            index = state.get("round_robin_index", 0) % len(candidates)
            state["round_robin_index"] = index + 1
            return candidates[index]

        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            # Select service with least active connections
            return min(
                candidates,
                key=lambda s: self.service_metrics.get(s.service_id, ServiceMetrics(s.service_id)).active_connections,
            )

        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            # Weighted selection based on service weight
            total_weight = sum(s.weight for s in candidates)
            if total_weight == 0:
                return candidates[0]

            import random

            r = random.randint(1, total_weight)
            for service in candidates:
                r -= service.weight
                if r <= 0:
                    return service

            return candidates[0]

        elif strategy == LoadBalancingStrategy.HEALTH_BASED:
            # Select service with best health metrics
            def health_score(service: ServiceEndpoint) -> float:
                health = self.service_health.get(service.service_id)
                metrics = self.service_metrics.get(service.service_id)

                if not health or not metrics:
                    return 0.0

                # Calculate composite health score
                status_weight = {
                    ServiceStatus.HEALTHY: 1.0,
                    ServiceStatus.DEGRADED: 0.7,
                    ServiceStatus.UNHEALTHY: 0.0,
                    ServiceStatus.STARTING: 0.5,
                    ServiceStatus.STOPPING: 0.2,
                    ServiceStatus.UNKNOWN: 0.3,
                }.get(health.status, 0.0)

                response_time_factor = max(0.1, 1.0 / (1.0 + health.response_time_ms / 1000.0))
                error_rate_factor = max(0.1, 1.0 - metrics.error_rate)

                return status_weight * response_time_factor * error_rate_factor * service.weight

            return max(candidates, key=health_score)

        else:
            return candidates[0]

    async def _health_check_loop(self):
        """Main health check loop."""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _perform_health_checks(self):
        """Perform health checks for all registered services."""
        if not self.services:
            return

        tasks = []
        for service in self.services.values():
            tasks.append(self._check_service_health(service))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_service_health(self, service: ServiceEndpoint):
        """Check health of a specific service."""
        circuit_breaker = self.circuit_breakers[service.service_id]

        if not circuit_breaker.can_execute():
            # Circuit is open, mark as unhealthy
            self.service_health[service.service_id] = ServiceHealth(
                service_id=service.service_id,
                status=ServiceStatus.UNHEALTHY,
                last_check=datetime.now(),
                circuit_open=True,
                metadata={"circuit_breaker": "open"},
            )
            return

        start_time = time.time()

        try:
            async with self._session.get(service.health_url) as response:
                response_time = (time.time() - start_time) * 1000

                health = self.service_health.get(
                    service.service_id, ServiceHealth(service.service_id, ServiceStatus.UNKNOWN, datetime.now())
                )

                if response.status == 200:
                    # Parse health response
                    try:
                        health_data = await response.json()
                        status = ServiceStatus.HEALTHY
                        metadata = health_data
                    except:
                        status = ServiceStatus.HEALTHY
                        metadata = {"raw_response": True}

                    health.status = status
                    health.response_time_ms = response_time
                    health.consecutive_failures = 0
                    health.consecutive_successes += 1
                    health.circuit_open = False
                    health.metadata = metadata

                    circuit_breaker.record_success()

                else:
                    # Non-200 response
                    health.status = ServiceStatus.DEGRADED if response.status < 500 else ServiceStatus.UNHEALTHY
                    health.response_time_ms = response_time
                    health.consecutive_failures += 1
                    health.consecutive_successes = 0
                    health.error_count += 1

                    if health.consecutive_failures >= 3:
                        circuit_breaker.record_failure()

                health.last_check = datetime.now()
                self.service_health[service.service_id] = health

        except Exception as e:
            # Connection error or timeout
            response_time = (time.time() - start_time) * 1000

            health = self.service_health.get(
                service.service_id, ServiceHealth(service.service_id, ServiceStatus.UNKNOWN, datetime.now())
            )
            health.status = ServiceStatus.UNHEALTHY
            health.response_time_ms = response_time
            health.consecutive_failures += 1
            health.consecutive_successes = 0
            health.error_count += 1
            health.last_check = datetime.now()
            health.metadata = {"error": str(e)}

            self.service_health[service.service_id] = health

            circuit_breaker.record_failure()

    def get_service_status(self, service_id: str) -> ServiceHealth | None:
        """Get health status for a specific service."""
        return self.service_health.get(service_id)

    def get_all_services(self) -> dict[str, dict[str, Any]]:
        """Get all registered services with their health status."""
        result = {}

        for service_id, service in self.services.items():
            health = self.service_health.get(service_id)
            metrics = self.service_metrics.get(service_id)
            circuit_breaker = self.circuit_breakers.get(service_id)

            result[service_id] = {
                "service": asdict(service),
                "health": asdict(health) if health else None,
                "metrics": asdict(metrics) if metrics else None,
                "circuit_breaker": {
                    "state": circuit_breaker.state if circuit_breaker else "unknown",
                    "failure_count": circuit_breaker.failure_count if circuit_breaker else 0,
                },
            }

        return result

    def update_service_metrics(
        self,
        service_id: str,
        request_count: int = None,
        active_connections: int = None,
        response_time: float = None,
        error_occurred: bool = False,
    ):
        """Update service metrics."""
        if service_id not in self.service_metrics:
            return

        metrics = self.service_metrics[service_id]

        if request_count is not None:
            metrics.request_count += request_count

        if active_connections is not None:
            metrics.active_connections = active_connections

        if response_time is not None:
            # Update running average
            if metrics.average_response_time == 0:
                metrics.average_response_time = response_time
            else:
                metrics.average_response_time = (metrics.average_response_time * 0.9) + (response_time * 0.1)

        if error_occurred:
            # Update error rate (simple exponential moving average)
            metrics.error_rate = (metrics.error_rate * 0.95) + 0.05
        else:
            metrics.error_rate = metrics.error_rate * 0.95

        metrics.last_updated = datetime.now()


class ServiceMeshCoordinator:
    """Coordinates service mesh operations."""

    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.service_dependencies: dict[str, set[str]] = defaultdict(set)
        self.service_routes: dict[str, str] = {}

    def add_service_dependency(self, service_name: str, depends_on: str):
        """Add service dependency relationship."""
        self.service_dependencies[service_name].add(depends_on)

    def get_service_dependencies(self, service_name: str) -> set[str]:
        """Get dependencies for a service."""
        return self.service_dependencies.get(service_name, set())

    async def check_service_dependencies(self, service_name: str) -> dict[str, bool]:
        """Check if all dependencies for a service are healthy."""
        dependencies = self.get_service_dependencies(service_name)
        results = {}

        for dep in dependencies:
            service = self.registry.get_service(dep)
            if service:
                health = self.registry.get_service_status(service.service_id)
                results[dep] = health.status == ServiceStatus.HEALTHY if health else False
            else:
                results[dep] = False

        return results

    def add_route(self, path_pattern: str, service_name: str):
        """Add routing rule."""
        self.service_routes[path_pattern] = service_name

    def get_route_target(self, path: str) -> str | None:
        """Get target service for a path."""
        # Simple pattern matching (can be enhanced with regex)
        for pattern, service_name in self.service_routes.items():
            if path.startswith(pattern):
                return service_name
        return None


# Singleton instance
_registry_instance: ServiceRegistry | None = None


async def get_service_registry() -> ServiceRegistry:
    """Get global service registry instance."""
    global _registry_instance

    if _registry_instance is None:
        _registry_instance = ServiceRegistry()
        await _registry_instance.start()

    return _registry_instance


async def shutdown_service_registry():
    """Shutdown global service registry."""
    global _registry_instance

    if _registry_instance:
        await _registry_instance.stop()
        _registry_instance = None
