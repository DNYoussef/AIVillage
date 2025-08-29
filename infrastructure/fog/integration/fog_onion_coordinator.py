"""
Fog Onion Routing Coordinator

Integrates onion routing privacy layer with fog computing task distribution.
Provides privacy-aware task scheduling, hidden service hosting, and secure gossip protocols.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from ..privacy.onion_routing import OnionRouter, OnionCircuit, NodeType, HiddenService
from ..privacy.mixnet_integration import NymMixnetClient

if TYPE_CHECKING:
    from .fog_coordinator import FogCoordinator

logger = logging.getLogger(__name__)


class PrivacyLevel(Enum):
    """Privacy classification for fog computing tasks."""

    PUBLIC = "public"  # No special privacy requirements
    PRIVATE = "private"  # Basic onion routing (3 hops)
    CONFIDENTIAL = "confidential"  # Extended onion routing (5+ hops) + mixnet
    SECRET = "secret"  # Full anonymity stack with cover traffic  # pragma: allowlist secret  # nosec B105


class TaskPrivacyPolicy(Enum):
    """Privacy policy enforcement modes."""

    OPTIONAL = "optional"  # Privacy is optional
    REQUIRED = "required"  # Privacy is mandatory
    ADAPTIVE = "adaptive"  # Auto-select based on task sensitivity


@dataclass
class PrivacyAwareTask:
    """A fog computing task with privacy requirements."""

    task_id: str
    privacy_level: PrivacyLevel
    task_data: bytes
    compute_requirements: Dict[str, Any]
    client_id: str

    # Privacy settings
    require_onion_circuit: bool = True
    require_mixnet: bool = False
    require_hidden_service: bool = False
    max_latency_ms: int = 5000

    # Circuit preferences
    min_circuit_hops: int = 3
    preferred_regions: List[str] = field(default_factory=list)
    excluded_nodes: Set[str] = field(default_factory=set)

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: Optional[datetime] = None


@dataclass
class PrivacyAwareService:
    """A fog service with privacy configuration."""

    service_id: str
    service_type: str
    privacy_level: PrivacyLevel
    onion_address: Optional[str] = None
    circuit_id: Optional[str] = None
    hidden_service: Optional[HiddenService] = None

    # Service configuration
    ports: Dict[int, int] = field(default_factory=dict)
    access_control: bool = False
    authentication_required: bool = False

    # Performance metrics
    requests_served: int = 0
    bytes_transferred: int = 0
    average_latency_ms: float = 0.0

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class FogOnionCoordinator:
    """
    Privacy-aware fog computing coordinator using onion routing.

    Integrates onion routing and mixnet technologies to provide:
    - Anonymous task submission and execution
    - Hidden service hosting for fog services
    - Private inter-node communication
    - Traffic analysis resistant gossip protocols
    """

    def __init__(
        self,
        node_id: str,
        fog_coordinator: 'FogCoordinator',
        enable_mixnet: bool = True,
        default_privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE,
        max_circuits: int = 50,
    ):
        self.node_id = node_id
        self.fog_coordinator = fog_coordinator
        self.enable_mixnet = enable_mixnet
        self.default_privacy_level = default_privacy_level
        self.max_circuits = max_circuits

        # Components
        self.onion_router: Optional[OnionRouter] = None
        self.mixnet_client: Optional[NymMixnetClient] = None

        # Task and service management
        self.privacy_tasks: Dict[str, PrivacyAwareTask] = {}
        self.privacy_services: Dict[str, PrivacyAwareService] = {}
        self.task_circuits: Dict[str, OnionCircuit] = {}
        self.service_circuits: Dict[str, OnionCircuit] = {}

        # Circuit pools for different privacy levels
        self.circuit_pools: Dict[PrivacyLevel, List[OnionCircuit]] = {level: [] for level in PrivacyLevel}

        # Statistics
        self.stats = {
            "tasks_processed": 0,
            "services_hosted": 0,
            "circuits_created": 0,
            "privacy_violations": 0,
            "average_task_latency": 0.0,
        }

        self._running = False
        self._background_tasks: Set[asyncio.Task] = set()

        logger.info(f"FogOnionCoordinator initialized: {node_id}")

    async def start(self) -> bool:
        """Start the fog onion coordinator."""
        if self._running:
            return True

        try:
            logger.info("Starting fog onion coordinator...")

            # Initialize onion router from fog coordinator
            if self.fog_coordinator.onion_router:
                self.onion_router = self.fog_coordinator.onion_router
                logger.info("Using existing onion router from fog coordinator")
            else:
                # Create new onion router
                node_types = {NodeType.MIDDLE, NodeType.GUARD}
                self.onion_router = OnionRouter(
                    node_id=f"fog-onion-{self.node_id}",
                    node_types=node_types,
                    enable_hidden_services=True,
                    num_guards=3,
                    circuit_lifetime_hours=1,
                )

                # Fetch consensus
                await self.onion_router.fetch_consensus()
                logger.info("Created new onion router")

            # Initialize mixnet client if enabled
            if self.enable_mixnet:
                self.mixnet_client = NymMixnetClient(client_id=f"fog-mixnet-{self.node_id}")
                await self.mixnet_client.start()
                logger.info("Mixnet client initialized")

            # Pre-build circuit pools
            await self._initialize_circuit_pools()

            # Start background tasks
            await self._start_background_tasks()

            self._running = True
            logger.info("Fog onion coordinator started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start fog onion coordinator: {e}")
            return False

    async def stop(self):
        """Stop the fog onion coordinator."""
        if not self._running:
            return

        logger.info("Stopping fog onion coordinator...")
        self._running = False

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        # Close all circuits
        for circuits in self.circuit_pools.values():
            for circuit in circuits:
                if self.onion_router:
                    await self.onion_router.close_circuit(circuit.circuit_id)

        # Stop mixnet client
        if self.mixnet_client:
            await self.mixnet_client.stop()

        logger.info("Fog onion coordinator stopped")

    async def submit_privacy_aware_task(self, task: PrivacyAwareTask) -> bool:
        """Submit a privacy-aware fog computing task."""
        if not self._running:
            raise RuntimeError("Coordinator not running")

        logger.info(f"Submitting privacy-aware task {task.task_id} with level {task.privacy_level.value}")

        try:
            # Validate privacy requirements
            if not await self._validate_privacy_requirements(task):
                logger.error(f"Privacy requirements validation failed for task {task.task_id}")
                return False

            # Get or create appropriate circuit
            circuit = await self._get_circuit_for_privacy_level(task.privacy_level)
            if not circuit:
                logger.error(f"Failed to establish circuit for privacy level {task.privacy_level.value}")
                return False

            # Store task
            self.privacy_tasks[task.task_id] = task
            self.task_circuits[task.task_id] = circuit

            # Route task through privacy layers
            success = await self._route_task_privately(task, circuit)

            if success:
                self.stats["tasks_processed"] += 1
                logger.info(f"Successfully submitted task {task.task_id}")
            else:
                # Cleanup on failure
                self.privacy_tasks.pop(task.task_id, None)
                self.task_circuits.pop(task.task_id, None)
                logger.error(f"Failed to route task {task.task_id}")

            return success

        except Exception as e:
            logger.error(f"Error submitting task {task.task_id}: {e}")
            return False

    async def create_privacy_aware_service(
        self,
        service_id: str,
        service_type: str,
        privacy_level: PrivacyLevel,
        ports: Dict[int, int],
        authentication_required: bool = False,
    ) -> Optional[PrivacyAwareService]:
        """Create a privacy-aware fog service."""
        if not self._running or not self.onion_router:
            return None

        logger.info(f"Creating privacy-aware service {service_id} with level {privacy_level.value}")

        try:
            service = PrivacyAwareService(
                service_id=service_id,
                service_type=service_type,
                privacy_level=privacy_level,
                ports=ports,
                authentication_required=authentication_required,
            )

            # Create hidden service for PRIVATE and above
            if privacy_level != PrivacyLevel.PUBLIC:
                hidden_service = await self.onion_router.create_hidden_service(
                    ports=ports,
                    descriptor_cookie=None,
                )

                service.onion_address = hidden_service.onion_address
                service.hidden_service = hidden_service

                logger.info(f"Created hidden service: {hidden_service.onion_address}")

            # Create dedicated circuit for CONFIDENTIAL and above
            if privacy_level in [PrivacyLevel.CONFIDENTIAL, PrivacyLevel.SECRET]:
                circuit = await self._create_dedicated_circuit(privacy_level)
                if circuit:
                    service.circuit_id = circuit.circuit_id
                    self.service_circuits[service_id] = circuit

            # Store service
            self.privacy_services[service_id] = service
            self.stats["services_hosted"] += 1

            logger.info(f"Successfully created privacy-aware service {service_id}")
            return service

        except Exception as e:
            logger.error(f"Error creating service {service_id}: {e}")
            return None

    async def send_private_gossip(
        self,
        recipient_id: str,
        message: bytes,
        privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE,
    ) -> bool:
        """Send private gossip message using onion routing."""
        if not self._running:
            return False

        try:
            if privacy_level == PrivacyLevel.PUBLIC:
                # Use direct fog coordinator communication
                return await self._send_direct_gossip(recipient_id, message)

            # Use appropriate privacy layer
            if privacy_level in [PrivacyLevel.CONFIDENTIAL, PrivacyLevel.SECRET] and self.mixnet_client:
                # Use mixnet for maximum privacy
                packet_id = await self.mixnet_client.send_anonymous_message(
                    destination=recipient_id,
                    message=message,
                )
                return packet_id is not None
            else:
                # Use onion routing
                circuit = await self._get_circuit_for_privacy_level(privacy_level)
                if circuit and self.onion_router:
                    return await self.onion_router.send_data(
                        circuit.circuit_id,
                        message,
                    )

            return False

        except Exception as e:
            logger.error(f"Error sending private gossip to {recipient_id}: {e}")
            return False

    async def get_service_by_onion_address(self, onion_address: str) -> Optional[PrivacyAwareService]:
        """Get service by its onion address."""
        for service in self.privacy_services.values():
            if service.onion_address == onion_address:
                return service
        return None

    async def get_coordinator_stats(self) -> Dict[str, Any]:
        """Get comprehensive coordinator statistics."""
        onion_stats = {}
        if self.onion_router:
            onion_stats = self.onion_router.get_stats()

        mixnet_stats = {}
        if self.mixnet_client:
            mixnet_stats = await self.mixnet_client.get_mixnet_stats()

        return {
            "node_id": self.node_id,
            "running": self._running,
            "privacy_stats": self.stats.copy(),
            "privacy_tasks": len(self.privacy_tasks),
            "privacy_services": len(self.privacy_services),
            "circuit_pools": {level.value: len(circuits) for level, circuits in self.circuit_pools.items()},
            "onion_routing": onion_stats,
            "mixnet": mixnet_stats,
        }

    # ============================================================================
    # PRIVATE HELPER METHODS
    # ============================================================================

    async def _validate_privacy_requirements(self, task: PrivacyAwareTask) -> bool:
        """Validate that privacy requirements can be satisfied."""
        # Check if we have sufficient nodes for required hops
        if self.onion_router and task.require_onion_circuit:
            available_nodes = len(self.onion_router.consensus)
            if available_nodes < task.min_circuit_hops:
                return False

        # Check mixnet availability for high privacy levels
        if task.privacy_level in [PrivacyLevel.CONFIDENTIAL, PrivacyLevel.SECRET]:
            if not self.enable_mixnet or not self.mixnet_client:
                return False

        return True

    async def _get_circuit_for_privacy_level(self, privacy_level: PrivacyLevel) -> Optional[OnionCircuit]:
        """Get or create a circuit for the specified privacy level."""
        if not self.onion_router:
            return None

        # Check circuit pool first
        circuits = self.circuit_pools.get(privacy_level, [])
        if circuits:
            # Return least used circuit
            circuit = min(circuits, key=lambda c: c.bytes_sent + c.bytes_received)
            return circuit

        # Create new circuit
        return await self._create_dedicated_circuit(privacy_level)

    async def _create_dedicated_circuit(self, privacy_level: PrivacyLevel) -> Optional[OnionCircuit]:
        """Create a dedicated circuit for the privacy level."""
        if not self.onion_router:
            return None

        # Determine path length based on privacy level
        path_length = {
            PrivacyLevel.PUBLIC: 1,
            PrivacyLevel.PRIVATE: 3,
            PrivacyLevel.CONFIDENTIAL: 5,
            PrivacyLevel.SECRET: 7,
        }.get(privacy_level, 3)

        try:
            circuit = await self.onion_router.build_circuit(
                purpose=f"privacy_{privacy_level.value}", path_length=path_length
            )

            if circuit:
                self.circuit_pools[privacy_level].append(circuit)
                self.stats["circuits_created"] += 1
                logger.debug(f"Created {privacy_level.value} circuit: {circuit.circuit_id}")

            return circuit

        except Exception as e:
            logger.error(f"Failed to create circuit for {privacy_level.value}: {e}")
            return None

    async def _route_task_privately(self, task: PrivacyAwareTask, circuit: OnionCircuit) -> bool:
        """Route task through privacy layers."""
        try:
            # Serialize task for transmission
            task_payload = await self._serialize_task(task)

            # Route based on privacy level
            if task.privacy_level == PrivacyLevel.SECRET and self.mixnet_client:
                # Use mixnet for maximum anonymity
                packet_id = await self.mixnet_client.send_anonymous_message(
                    destination="fog_task_processor",
                    message=task_payload,
                )
                return packet_id is not None

            elif task.require_onion_circuit and self.onion_router:
                # Use onion routing
                success = await self.onion_router.send_data(
                    circuit.circuit_id,
                    task_payload,
                    stream_id=f"task_{task.task_id}",
                )
                return success

            else:
                # Fallback to direct routing
                return await self._route_task_directly(task)

        except Exception as e:
            logger.error(f"Error routing task {task.task_id}: {e}")
            return False

    async def _serialize_task(self, task: PrivacyAwareTask) -> bytes:
        """Serialize task for network transmission."""
        import pickle  # nosec B403 - Used for internal task serialization

        return pickle.dumps(
            {
                "task_id": task.task_id,
                "privacy_level": task.privacy_level.value,
                "task_data": task.task_data,
                "compute_requirements": task.compute_requirements,
                "client_id": task.client_id,
            }
        )

    async def _route_task_directly(self, task: PrivacyAwareTask) -> bool:
        """Route task directly through fog coordinator."""
        if not self.fog_coordinator:
            return False

        try:
            # Convert to fog coordinator format
            request_data = {
                "request_id": task.task_id,
                "customer_id": task.client_id,
                "compute_requirements": task.compute_requirements,
                "task_data": task.task_data,
            }

            result = await self.fog_coordinator.process_fog_request("compute_task", request_data)
            return result.get("success", False)

        except Exception as e:
            logger.error(f"Error in direct task routing: {e}")
            return False

    async def _send_direct_gossip(self, recipient_id: str, message: bytes) -> bool:
        """Send gossip message directly through fog coordinator."""
        # This would integrate with the fog coordinator's P2P networking
        # For now, simulate success
        logger.debug(f"Sending direct gossip to {recipient_id}: {len(message)} bytes")
        return True

    async def _initialize_circuit_pools(self):
        """Pre-build circuits for different privacy levels."""
        logger.info("Initializing circuit pools...")

        # Build circuits for each privacy level
        tasks = []
        for privacy_level in PrivacyLevel:
            if privacy_level == PrivacyLevel.PUBLIC:
                continue  # No circuits needed for public

            # Build 2-3 circuits per privacy level
            num_circuits = 3 if privacy_level == PrivacyLevel.PRIVATE else 2
            for _ in range(num_circuits):
                tasks.append(self._create_dedicated_circuit(privacy_level))

        # Wait for all circuits to be built
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_circuits = sum(1 for r in results if isinstance(r, OnionCircuit))
        logger.info(f"Initialized {successful_circuits} circuits across privacy levels")

    async def _start_background_tasks(self):
        """Start background maintenance tasks."""
        tasks = [
            self._circuit_maintenance_task(),
            self._privacy_metrics_task(),
            self._task_cleanup_task(),
        ]

        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        logger.info("Started background tasks for fog onion coordinator")

    async def _circuit_maintenance_task(self):
        """Maintain circuit pools and rotate old circuits."""
        while self._running:
            try:
                # Rotate old circuits
                if self.onion_router:
                    rotated = await self.onion_router.rotate_circuits()
                    if rotated > 0:
                        logger.debug(f"Rotated {rotated} circuits")

                # Maintain circuit pool sizes
                for privacy_level in PrivacyLevel:
                    if privacy_level == PrivacyLevel.PUBLIC:
                        continue

                    circuits = self.circuit_pools[privacy_level]
                    target_size = 3 if privacy_level == PrivacyLevel.PRIVATE else 2

                    # Remove failed circuits
                    active_circuits = [c for c in circuits if c.state.value == "established"]
                    self.circuit_pools[privacy_level] = active_circuits

                    # Add new circuits if needed
                    if len(active_circuits) < target_size:
                        missing = target_size - len(active_circuits)
                        for _ in range(missing):
                            await self._create_dedicated_circuit(privacy_level)

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error in circuit maintenance: {e}")
                await asyncio.sleep(60)

    async def _privacy_metrics_task(self):
        """Collect privacy and performance metrics."""
        while self._running:
            try:
                # Update task latency metrics
                task_latencies = []
                for task in self.privacy_tasks.values():
                    age = (datetime.now(UTC) - task.created_at).total_seconds() * 1000
                    task_latencies.append(age)

                if task_latencies:
                    self.stats["average_task_latency"] = sum(task_latencies) / len(task_latencies)

                # Check for privacy violations
                violations = 0
                for task in self.privacy_tasks.values():
                    if task.task_id not in self.task_circuits and task.require_onion_circuit:
                        violations += 1

                self.stats["privacy_violations"] = violations

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Error collecting privacy metrics: {e}")
                await asyncio.sleep(30)

    async def _task_cleanup_task(self):
        """Clean up completed and expired tasks."""
        while self._running:
            try:
                now = datetime.now(UTC)
                expired_tasks = []

                for task_id, task in self.privacy_tasks.items():
                    # Remove tasks older than 1 hour or explicitly expired
                    age = now - task.created_at
                    if age > timedelta(hours=1) or (task.expires_at and now > task.expires_at):
                        expired_tasks.append(task_id)

                # Clean up expired tasks
                for task_id in expired_tasks:
                    self.privacy_tasks.pop(task_id, None)
                    circuit = self.task_circuits.pop(task_id, None)
                    if circuit:
                        logger.debug(f"Cleaned up expired task {task_id}")

                if expired_tasks:
                    logger.info(f"Cleaned up {len(expired_tasks)} expired tasks")

                await asyncio.sleep(600)  # Cleanup every 10 minutes

            except Exception as e:
                logger.error(f"Error in task cleanup: {e}")
                await asyncio.sleep(300)
