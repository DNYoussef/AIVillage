"""
Onion Circuit Service

Dedicated service for managing onion routing circuits with privacy-level isolation,
load balancing, and background maintenance for fog computing infrastructure.
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
import logging
from typing import Any

from ..privacy.onion_routing import OnionCircuit, OnionRouter

logger = logging.getLogger(__name__)


class PrivacyLevel(Enum):
    """Privacy classification for circuit pools."""

    PUBLIC = "public"
    PRIVATE = "private"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"  # nosec B105 - config key name, not password


@dataclass
class CircuitMetrics:
    """Statistics and metrics for a circuit."""

    circuit_id: str
    privacy_level: PrivacyLevel
    bytes_sent: int = 0
    bytes_received: int = 0
    requests_handled: int = 0
    created_at: datetime = None
    last_used_at: datetime = None
    health_score: float = 1.0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(UTC)
        if self.last_used_at is None:
            self.last_used_at = self.created_at


class OnionCircuitService:
    """
    Dedicated service for managing onion routing circuits.

    Provides:
    - Privacy-level isolated circuit pools
    - Intelligent load balancing based on usage metrics
    - Background circuit rotation and maintenance
    - Circuit health monitoring and statistics
    - Secure circuit cleanup and destruction
    """

    def __init__(
        self,
        onion_router: OnionRouter,
        max_circuits_per_level: int = 10,
        circuit_lifetime_minutes: int = 30,
        rotation_interval_minutes: int = 5,
    ):
        self.onion_router = onion_router
        self.max_circuits_per_level = max_circuits_per_level
        self.circuit_lifetime = timedelta(minutes=circuit_lifetime_minutes)
        self.rotation_interval = timedelta(minutes=rotation_interval_minutes)

        # Circuit pools organized by privacy level
        self.circuit_pools: dict[PrivacyLevel, list[OnionCircuit]] = {level: [] for level in PrivacyLevel}

        # Circuit metrics and tracking
        self.circuit_metrics: dict[str, CircuitMetrics] = {}
        self.authenticated_clients: set[str] = set()

        # Pool size targets per privacy level
        self.pool_targets = {
            PrivacyLevel.PUBLIC: 1,  # Minimal circuits for public tasks
            PrivacyLevel.PRIVATE: 3,  # Higher pool for common private tasks
            PrivacyLevel.CONFIDENTIAL: 2,  # Dedicated circuits for sensitive tasks
            PrivacyLevel.SECRET: 2,  # Isolated circuits for secret tasks
        }

        # Statistics
        self.stats = {
            "total_circuits_created": 0,
            "circuits_rotated": 0,
            "circuit_failures": 0,
            "load_balancing_decisions": 0,
            "auth_failures": 0,
            "cleanup_operations": 0,
        }

        self._running = False
        self._background_tasks: set[asyncio.Task] = set()

        logger.info("OnionCircuitService initialized")

    async def start(self) -> bool:
        """Start the circuit service and initialize circuit pools."""
        if self._running:
            return True

        try:
            logger.info("Starting OnionCircuitService...")

            # Initialize circuit pools
            await self._initialize_circuit_pools()

            # Start background maintenance tasks
            await self._start_background_tasks()

            self._running = True
            logger.info("OnionCircuitService started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start OnionCircuitService: {e}")
            return False

    async def stop(self):
        """Stop the service and clean up all circuits."""
        if not self._running:
            return

        logger.info("Stopping OnionCircuitService...")
        self._running = False

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        # Securely destroy all circuits
        await self._secure_circuit_cleanup()

        logger.info("OnionCircuitService stopped")

    def authenticate_client(self, client_id: str, auth_token: str) -> bool:
        """Authenticate a client for circuit access."""
        # Simple authentication - in production, use proper auth mechanisms
        if self._validate_auth_token(client_id, auth_token):
            self.authenticated_clients.add(client_id)
            return True
        else:
            self.stats["auth_failures"] += 1
            logger.warning(f"Authentication failed for client {client_id}")
            return False

    async def get_circuit(
        self,
        privacy_level: PrivacyLevel,
        client_id: str,
        preferred_path_length: int | None = None,
    ) -> OnionCircuit | None:
        """
        Get an optimal circuit for the specified privacy level.

        Uses load balancing to select the least used circuit from the pool.
        """
        if not self._running:
            return None

        # Check client authentication
        if client_id not in self.authenticated_clients:
            logger.error(f"Unauthenticated client {client_id} requesting circuit")
            return None

        circuits = self.circuit_pools.get(privacy_level, [])
        if not circuits:
            # Try to create a new circuit
            circuit = await self._create_circuit_for_level(privacy_level, preferred_path_length)
            return circuit

        # Load balance: select circuit with lowest usage
        optimal_circuit = self._select_optimal_circuit(circuits)

        if optimal_circuit:
            # Update usage metrics
            metrics = self.circuit_metrics.get(optimal_circuit.circuit_id)
            if metrics:
                metrics.last_used_at = datetime.now(UTC)
                metrics.requests_handled += 1

            self.stats["load_balancing_decisions"] += 1
            logger.debug(f"Selected circuit {optimal_circuit.circuit_id} for {privacy_level.value}")

        return optimal_circuit

    async def update_circuit_usage(
        self,
        circuit_id: str,
        bytes_sent: int,
        bytes_received: int,
    ):
        """Update circuit usage statistics."""
        metrics = self.circuit_metrics.get(circuit_id)
        if metrics:
            metrics.bytes_sent += bytes_sent
            metrics.bytes_received += bytes_received
            metrics.last_used_at = datetime.now(UTC)

            # Update health score based on performance
            total_bytes = metrics.bytes_sent + metrics.bytes_received
            if total_bytes > 0:
                # Simple health scoring - can be enhanced with latency, error rates
                age_minutes = (datetime.now(UTC) - metrics.created_at).total_seconds() / 60
                metrics.health_score = max(0.1, 1.0 - (age_minutes / 60))  # Decay over 1 hour

    def get_circuit_stats(self) -> dict[str, Any]:
        """Get comprehensive circuit statistics."""
        pool_stats = {}
        for level, circuits in self.circuit_pools.items():
            active_circuits = [c for c in circuits if c.state.value == "established"]
            pool_stats[level.value] = {
                "total": len(circuits),
                "active": len(active_circuits),
                "target": self.pool_targets[level],
                "health_avg": self._calculate_average_health(circuits),
            }

        return {
            "service_stats": self.stats.copy(),
            "pool_statistics": pool_stats,
            "authenticated_clients": len(self.authenticated_clients),
            "total_circuits": sum(len(pool) for pool in self.circuit_pools.values()),
            "metrics_tracked": len(self.circuit_metrics),
        }

    # ============================================================================
    # PRIVATE HELPER METHODS
    # ============================================================================

    async def _initialize_circuit_pools(self):
        """Initialize circuit pools for all privacy levels."""
        logger.info("Initializing circuit pools...")

        tasks = []
        for level, target_size in self.pool_targets.items():
            if level == PrivacyLevel.PUBLIC:
                continue  # Create public circuits on demand

            for _ in range(target_size):
                task = self._create_circuit_for_level(level)
                tasks.append(task)

        # Create circuits concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = sum(1 for r in results if isinstance(r, OnionCircuit))
        logger.info(f"Initialized {successful} circuits across privacy levels")

    async def _create_circuit_for_level(
        self,
        privacy_level: PrivacyLevel,
        preferred_path_length: int | None = None,
    ) -> OnionCircuit | None:
        """Create a new circuit for the specified privacy level."""
        if not self.onion_router:
            return None

        # Determine path length based on privacy level
        if preferred_path_length:
            path_length = preferred_path_length
        else:
            path_length = {
                PrivacyLevel.PUBLIC: 1,
                PrivacyLevel.PRIVATE: 3,
                PrivacyLevel.CONFIDENTIAL: 5,
                PrivacyLevel.SECRET: 7,
            }.get(privacy_level, 3)

        try:
            circuit = await self.onion_router.build_circuit(
                purpose=f"privacy_{privacy_level.value}",
                path_length=path_length,
            )

            if circuit:
                # Add to pool
                self.circuit_pools[privacy_level].append(circuit)

                # Create metrics tracking
                metrics = CircuitMetrics(
                    circuit_id=circuit.circuit_id,
                    privacy_level=privacy_level,
                )
                self.circuit_metrics[circuit.circuit_id] = metrics

                self.stats["total_circuits_created"] += 1
                logger.debug(f"Created {privacy_level.value} circuit: {circuit.circuit_id}")

            return circuit

        except Exception as e:
            logger.error(f"Failed to create circuit for {privacy_level.value}: {e}")
            self.stats["circuit_failures"] += 1
            return None

    def _select_optimal_circuit(self, circuits: list[OnionCircuit]) -> OnionCircuit | None:
        """Select the optimal circuit based on load balancing."""
        if not circuits:
            return None

        # Filter active circuits
        active_circuits = [c for c in circuits if c.state.value == "established"]
        if not active_circuits:
            return None

        # Load balance based on bytes transferred and request count
        def circuit_load_score(circuit):
            metrics = self.circuit_metrics.get(circuit.circuit_id)
            if not metrics:
                return 0

            # Combine bytes transferred and request count for load scoring
            total_bytes = metrics.bytes_sent + metrics.bytes_received
            return (total_bytes / 1024) + (metrics.requests_handled * 10)  # Weight requests more

        # Return circuit with lowest load score
        return min(active_circuits, key=circuit_load_score)

    def _calculate_average_health(self, circuits: list[OnionCircuit]) -> float:
        """Calculate average health score for a circuit pool."""
        if not circuits:
            return 0.0

        health_scores = []
        for circuit in circuits:
            metrics = self.circuit_metrics.get(circuit.circuit_id)
            if metrics:
                health_scores.append(metrics.health_score)

        return sum(health_scores) / len(health_scores) if health_scores else 0.0

    def _validate_auth_token(self, client_id: str, auth_token: str) -> bool:
        """Validate authentication token for client."""
        # Simple validation - in production, use proper cryptographic validation
        expected_token = f"auth_{client_id}_token"
        return auth_token == expected_token

    async def _start_background_tasks(self):
        """Start background maintenance tasks."""
        tasks = [
            self._circuit_rotation_task(),
            self._pool_maintenance_task(),
            self._metrics_cleanup_task(),
        ]

        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        logger.info("Started circuit service background tasks")

    async def _circuit_rotation_task(self):
        """Periodically rotate old circuits."""
        while self._running:
            try:
                now = datetime.now(UTC)
                circuits_rotated = 0

                for level, circuits in self.circuit_pools.items():
                    old_circuits = []

                    for circuit in circuits:
                        metrics = self.circuit_metrics.get(circuit.circuit_id)
                        if metrics and (now - metrics.created_at) > self.circuit_lifetime:
                            old_circuits.append(circuit)

                    # Replace old circuits
                    for old_circuit in old_circuits:
                        await self._replace_circuit(level, old_circuit)
                        circuits_rotated += 1

                if circuits_rotated > 0:
                    self.stats["circuits_rotated"] += circuits_rotated
                    logger.info(f"Rotated {circuits_rotated} circuits")

                await asyncio.sleep(self.rotation_interval.total_seconds())

            except Exception as e:
                logger.error(f"Error in circuit rotation: {e}")
                await asyncio.sleep(60)

    async def _pool_maintenance_task(self):
        """Maintain circuit pool sizes and health."""
        while self._running:
            try:
                for level, target_size in self.pool_targets.items():
                    circuits = self.circuit_pools[level]

                    # Remove failed circuits
                    active_circuits = []
                    failed_circuits = []

                    for circuit in circuits:
                        if circuit.state.value == "established":
                            active_circuits.append(circuit)
                        else:
                            failed_circuits.append(circuit)

                    # Clean up failed circuits
                    for failed_circuit in failed_circuits:
                        await self._remove_circuit(level, failed_circuit)

                    # Add circuits if below target
                    current_size = len(active_circuits)
                    if current_size < target_size:
                        missing = target_size - current_size
                        for _ in range(missing):
                            await self._create_circuit_for_level(level)

                await asyncio.sleep(120)  # Check every 2 minutes

            except Exception as e:
                logger.error(f"Error in pool maintenance: {e}")
                await asyncio.sleep(60)

    async def _metrics_cleanup_task(self):
        """Clean up old metrics and expired data."""
        while self._running:
            try:
                now = datetime.now(UTC)
                expired_metrics = []

                # Find metrics for circuits that no longer exist
                active_circuit_ids = set()
                for circuits in self.circuit_pools.values():
                    active_circuit_ids.update(c.circuit_id for c in circuits)

                for circuit_id, metrics in self.circuit_metrics.items():
                    if circuit_id not in active_circuit_ids:
                        expired_metrics.append(circuit_id)
                    elif (now - metrics.created_at) > timedelta(hours=2):
                        expired_metrics.append(circuit_id)

                # Remove expired metrics
                for circuit_id in expired_metrics:
                    self.circuit_metrics.pop(circuit_id, None)

                if expired_metrics:
                    self.stats["cleanup_operations"] += len(expired_metrics)
                    logger.debug(f"Cleaned up {len(expired_metrics)} old circuit metrics")

                await asyncio.sleep(600)  # Cleanup every 10 minutes

            except Exception as e:
                logger.error(f"Error in metrics cleanup: {e}")
                await asyncio.sleep(300)

    async def _replace_circuit(self, level: PrivacyLevel, old_circuit: OnionCircuit):
        """Replace an old circuit with a new one."""
        try:
            # Create new circuit first
            new_circuit = await self._create_circuit_for_level(level)

            if new_circuit:
                # Remove old circuit
                await self._remove_circuit(level, old_circuit)
                logger.debug(f"Replaced circuit {old_circuit.circuit_id} with {new_circuit.circuit_id}")

        except Exception as e:
            logger.error(f"Error replacing circuit: {e}")

    async def _remove_circuit(self, level: PrivacyLevel, circuit: OnionCircuit):
        """Securely remove a circuit from the pool."""
        try:
            # Remove from pool
            if circuit in self.circuit_pools[level]:
                self.circuit_pools[level].remove(circuit)

            # Close circuit securely
            await self.onion_router.close_circuit(circuit.circuit_id)

            # Remove metrics
            self.circuit_metrics.pop(circuit.circuit_id, None)

        except Exception as e:
            logger.error(f"Error removing circuit {circuit.circuit_id}: {e}")

    async def _secure_circuit_cleanup(self):
        """Securely destroy all circuits during shutdown."""
        logger.info("Performing secure circuit cleanup...")

        total_circuits = 0
        for level, circuits in self.circuit_pools.items():
            for circuit in circuits[:]:  # Copy list to avoid modification during iteration
                await self._remove_circuit(level, circuit)
                total_circuits += 1

        # Clear all data structures
        self.circuit_pools.clear()
        self.circuit_metrics.clear()
        self.authenticated_clients.clear()

        self.stats["cleanup_operations"] += total_circuits
        logger.info(f"Securely cleaned up {total_circuits} circuits")
