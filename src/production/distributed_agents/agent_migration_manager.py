"""Agent Migration Manager.

Handles dynamic migration of agents between devices based on performance,
resource availability, and network conditions.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any
import uuid

import msgpack

from AIVillage.src.core.p2p.p2p_node import P2PNode

from .distributed_agent_orchestrator import AgentInstance, AgentType, DeviceProfile

logger = logging.getLogger(__name__)


class MigrationReason(Enum):
    """Reasons for agent migration."""

    DEVICE_OVERLOAD = "device_overload"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DEVICE_FAILURE = "device_failure"
    BATTERY_LOW = "battery_low"
    NETWORK_OPTIMIZATION = "network_optimization"
    LOAD_BALANCING = "load_balancing"
    MANUAL_REQUEST = "manual_request"
    DEVICE_MAINTENANCE = "device_maintenance"


class MigrationStrategy(Enum):
    """Migration strategies."""

    IMMEDIATE = "immediate"  # Immediate migration (for failures)
    GRACEFUL = "graceful"  # Wait for good migration window
    SCHEDULED = "scheduled"  # Schedule migration for later
    OPPORTUNISTIC = "opportunistic"  # Migrate when conditions are optimal


@dataclass
class MigrationRequest:
    """Request for agent migration."""

    request_id: str
    agent_instance_id: str
    reason: MigrationReason
    source_device_id: str
    target_device_id: str | None = None
    strategy: MigrationStrategy = MigrationStrategy.GRACEFUL
    priority: int = 5  # 1=highest, 10=lowest
    created_at: float = field(default_factory=time.time)
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MigrationEvent:
    """Migration event record."""

    event_id: str
    request: MigrationRequest
    source_device_id: str
    target_device_id: str
    start_time: float
    end_time: float | None = None
    success: bool = False
    migration_duration: float = 0.0
    downtime_duration: float = 0.0
    data_transferred_mb: float = 0.0
    error_message: str | None = None
    performance_impact: float = 0.0  # 0-1, impact on system performance


@dataclass
class AgentCheckpoint:
    """Agent state checkpoint for migration."""

    instance_id: str
    agent_type: AgentType
    state_data: bytes
    memory_snapshot: bytes | None = None
    configuration: dict[str, Any] = field(default_factory=dict)
    checkpoint_time: float = field(default_factory=time.time)
    size_mb: float = 0.0
    source_device_id: str | None = None


class AgentMigrationManager:
    """Manages agent migration between devices."""

    def __init__(
        self,
        p2p_node: P2PNode,
        agent_orchestrator,  # DistributedAgentOrchestrator
    ) -> None:
        self.p2p_node = p2p_node
        self.agent_orchestrator = agent_orchestrator

        # Migration state
        self.pending_migrations: dict[str, MigrationRequest] = {}
        self.active_migrations: dict[str, MigrationEvent] = {}
        self.migration_history: list[MigrationEvent] = []

        # Migration queues by priority
        self.migration_queues: dict[int, list[MigrationRequest]] = {
            i: [] for i in range(1, 11)
        }

        # Agent checkpoints
        self.agent_checkpoints: dict[str, AgentCheckpoint] = {}

        # Configuration
        self.config = {
            "max_concurrent_migrations": 2,
            "migration_timeout_seconds": 300.0,
            "checkpoint_interval_seconds": 60.0,
            "min_target_device_resources": 0.3,  # 30% resources available
            "migration_cooldown_seconds": 30.0,
            "enable_proactive_migration": True,
            "performance_threshold": 0.7,
        }

        # Statistics
        self.stats = {
            "migrations_requested": 0,
            "migrations_completed": 0,
            "migrations_failed": 0,
            "avg_migration_time": 0.0,
            "avg_downtime": 0.0,
            "data_transferred_gb": 0.0,
            "performance_improvements": 0,
        }

        # Start background tasks
        self._start_background_tasks()

        logger.info("AgentMigrationManager initialized")

    def _start_background_tasks(self) -> None:
        """Start background migration tasks."""
        asyncio.create_task(self._migration_processor())
        asyncio.create_task(self._performance_monitor())
        asyncio.create_task(self._checkpoint_manager())

    async def request_migration(
        self,
        agent_instance_id: str,
        reason: MigrationReason,
        target_device_id: str | None = None,
        strategy: MigrationStrategy = MigrationStrategy.GRACEFUL,
        priority: int = 5,
    ) -> str:
        """Request agent migration."""
        # Get agent instance
        if agent_instance_id not in self.agent_orchestrator.active_agents:
            msg = f"Agent instance not found: {agent_instance_id}"
            raise ValueError(msg)

        agent_instance = self.agent_orchestrator.active_agents[agent_instance_id]

        # Check if agent can migrate
        if not agent_instance.agent_spec.can_migrate:
            msg = f"Agent {agent_instance_id} cannot be migrated"
            raise ValueError(msg)

        # Create migration request
        request = MigrationRequest(
            request_id=str(uuid.uuid4()),
            agent_instance_id=agent_instance_id,
            reason=reason,
            source_device_id=agent_instance.device_id,
            target_device_id=target_device_id,
            strategy=strategy,
            priority=priority,
        )

        # Add to pending migrations
        self.pending_migrations[request.request_id] = request
        self.migration_queues[priority].append(request)

        self.stats["migrations_requested"] += 1

        logger.info(
            f"Migration requested: {agent_instance_id} ({reason.value}) priority {priority}"
        )
        return request.request_id

    async def _migration_processor(self) -> None:
        """Process migration queue."""
        while True:
            try:
                # Check if we can process more migrations
                if (
                    len(self.active_migrations)
                    >= self.config["max_concurrent_migrations"]
                ):
                    await asyncio.sleep(5.0)
                    continue

                # Find highest priority migration request
                migration_request = None
                for priority in range(1, 11):
                    if self.migration_queues[priority]:
                        migration_request = self.migration_queues[priority].pop(0)
                        break

                if migration_request is None:
                    await asyncio.sleep(10.0)
                    continue

                # Process migration
                await self._process_migration_request(migration_request)

            except Exception as e:
                logger.exception(f"Error in migration processor: {e}")
                await asyncio.sleep(30.0)

    async def _process_migration_request(self, request: MigrationRequest) -> None:
        """Process a single migration request."""
        logger.info(f"Processing migration request {request.request_id}")

        try:
            # Remove from pending
            if request.request_id in self.pending_migrations:
                del self.pending_migrations[request.request_id]

            # Find target device if not specified
            target_device_id = request.target_device_id
            if target_device_id is None:
                target_device_id = await self._find_best_target_device(request)

                if target_device_id is None:
                    logger.warning(
                        f"No suitable target device found for migration {request.request_id}"
                    )
                    return

            # Create migration event
            migration_event = MigrationEvent(
                event_id=str(uuid.uuid4()),
                request=request,
                source_device_id=request.source_device_id,
                target_device_id=target_device_id,
                start_time=time.time(),
            )

            self.active_migrations[migration_event.event_id] = migration_event

            # Execute migration based on strategy
            if request.strategy == MigrationStrategy.IMMEDIATE:
                success = await self._execute_immediate_migration(migration_event)
            elif request.strategy == MigrationStrategy.GRACEFUL:
                success = await self._execute_graceful_migration(migration_event)
            elif request.strategy == MigrationStrategy.SCHEDULED:
                success = await self._execute_scheduled_migration(migration_event)
            else:
                success = await self._execute_opportunistic_migration(migration_event)

            # Complete migration event
            migration_event.end_time = time.time()
            migration_event.success = success
            migration_event.migration_duration = (
                migration_event.end_time - migration_event.start_time
            )

            # Update statistics
            if success:
                self.stats["migrations_completed"] += 1
                self.stats["avg_migration_time"] = (
                    self.stats["avg_migration_time"]
                    + migration_event.migration_duration
                ) / 2
            else:
                self.stats["migrations_failed"] += 1

            # Move to history
            self.migration_history.append(migration_event)
            if migration_event.event_id in self.active_migrations:
                del self.active_migrations[migration_event.event_id]

            logger.info(
                f"Migration {request.request_id} completed: {'success' if success else 'failed'}"
            )

        except Exception as e:
            logger.exception(f"Migration processing failed: {e}")
            if request.request_id in self.active_migrations:
                migration_event = self.active_migrations[request.request_id]
                migration_event.success = False
                migration_event.error_message = str(e)

    async def _find_best_target_device(self, request: MigrationRequest) -> str | None:
        """Find best target device for migration."""
        agent_instance = self.agent_orchestrator.active_agents[
            request.agent_instance_id
        ]
        agent_spec = agent_instance.agent_spec

        # Get available devices
        try:
            if (
                hasattr(self.agent_orchestrator, "sharding_engine")
                and self.agent_orchestrator.sharding_engine
            ):
                device_profiles = (
                    await self.agent_orchestrator.sharding_engine._get_device_profiles()
                )
            else:
                # Fallback to simple device discovery
                device_profiles = await self.agent_orchestrator._get_available_devices()
        except Exception as e:
            logger.exception(f"Failed to get device profiles: {e}")
            return None

        # Filter out source device and unsuitable devices
        candidate_devices = []
        for device in device_profiles:
            if device.device_id == request.source_device_id:
                continue

            if not self._is_device_suitable_for_migration(device, agent_spec):
                continue

            candidate_devices.append(device)

        if not candidate_devices:
            return None

        # Score devices based on suitability
        best_device = None
        best_score = -1.0

        for device in candidate_devices:
            score = self._calculate_migration_suitability_score(
                device, agent_spec, request.reason
            )

            if score > best_score:
                best_score = score
                best_device = device

        return best_device.device_id if best_device else None

    def _is_device_suitable_for_migration(
        self, device: DeviceProfile, agent_spec
    ) -> bool:
        """Check if device is suitable for agent migration."""
        # Basic resource requirements
        if device.available_memory_mb < agent_spec.memory_requirement_mb:
            return False

        if device.compute_score < agent_spec.compute_requirement:
            return False

        # Minimum available resources (don't overload target device)
        memory_utilization = (
            agent_spec.memory_requirement_mb / device.available_memory_mb
        )
        if memory_utilization > (1.0 - self.config["min_target_device_resources"]):
            return False

        # Reliability threshold
        if device.reliability_score < 0.6:
            return False

        # Battery constraint for mobile devices
        return not (device.battery_level and device.battery_level < 30)

    def _calculate_migration_suitability_score(
        self, device: DeviceProfile, agent_spec, reason: MigrationReason
    ) -> float:
        """Calculate migration suitability score for device."""
        score = 0.0

        # Resource availability (prefer devices with more resources)
        memory_ratio = device.available_memory_mb / agent_spec.memory_requirement_mb
        memory_score = min(1.0, memory_ratio / 3.0)  # Optimal at 3x requirement
        score += memory_score * 0.3

        compute_ratio = device.compute_score / agent_spec.compute_requirement
        compute_score = min(1.0, compute_ratio / 2.0)
        score += compute_score * 0.3

        # Device reliability
        score += device.reliability_score * 0.2

        # Network latency (lower is better)
        latency_score = max(0.0, 1.0 - device.network_latency_ms / 100.0)
        score += latency_score * 0.1

        # Reason-specific scoring
        if reason == MigrationReason.PERFORMANCE_DEGRADATION:
            # Prefer high-performance devices
            score += (device.compute_score / 10.0) * 0.1
        elif reason == MigrationReason.DEVICE_OVERLOAD:
            # Prefer devices with more available resources
            score += memory_score * 0.1

        return score

    async def _execute_immediate_migration(
        self, migration_event: MigrationEvent
    ) -> bool:
        """Execute immediate migration (for emergencies)."""
        logger.info(f"Executing immediate migration {migration_event.event_id}")

        try:
            agent_instance_id = migration_event.request.agent_instance_id
            agent_instance = self.agent_orchestrator.active_agents[agent_instance_id]

            # Create checkpoint quickly
            checkpoint = await self._create_agent_checkpoint(agent_instance)

            # Stop agent on source device
            downtime_start = time.time()
            await self._stop_agent(agent_instance_id, migration_event.source_device_id)

            # Transfer checkpoint to target device
            transfer_success = await self._transfer_checkpoint(
                checkpoint, migration_event.target_device_id
            )

            if not transfer_success:
                # Rollback - restart agent on source device
                await self._restart_agent(
                    agent_instance_id, migration_event.source_device_id
                )
                return False

            # Start agent on target device
            start_success = await self._start_agent_from_checkpoint(
                checkpoint, migration_event.target_device_id
            )

            downtime_end = time.time()
            migration_event.downtime_duration = downtime_end - downtime_start

            if start_success:
                # Update agent instance location
                agent_instance.device_id = migration_event.target_device_id
                agent_instance.migration_count += 1

                # Update orchestrator state
                await self._update_orchestrator_state(
                    agent_instance_id, migration_event.target_device_id
                )

                return True
            # Rollback
            await self._restart_agent(
                agent_instance_id, migration_event.source_device_id
            )
            return False

        except Exception as e:
            logger.exception(f"Immediate migration failed: {e}")
            migration_event.error_message = str(e)
            return False

    async def _execute_graceful_migration(
        self, migration_event: MigrationEvent
    ) -> bool:
        """Execute graceful migration with minimal disruption."""
        logger.info(f"Executing graceful migration {migration_event.event_id}")

        try:
            agent_instance_id = migration_event.request.agent_instance_id
            agent_instance = self.agent_orchestrator.active_agents[agent_instance_id]

            # Wait for good migration window (low activity)
            await self._wait_for_migration_window(agent_instance)

            # Pre-warm target device
            await self._prepare_target_device(
                migration_event.target_device_id, agent_instance.agent_spec
            )

            # Create comprehensive checkpoint
            checkpoint = await self._create_agent_checkpoint(
                agent_instance, comprehensive=True
            )

            # Transfer checkpoint while agent is still running
            transfer_success = await self._transfer_checkpoint(
                checkpoint, migration_event.target_device_id
            )

            if not transfer_success:
                return False

            # Quick switchover
            downtime_start = time.time()

            # Stop agent on source
            await self._stop_agent(agent_instance_id, migration_event.source_device_id)

            # Start agent on target
            start_success = await self._start_agent_from_checkpoint(
                checkpoint, migration_event.target_device_id
            )

            downtime_end = time.time()
            migration_event.downtime_duration = downtime_end - downtime_start

            if start_success:
                # Update state
                agent_instance.device_id = migration_event.target_device_id
                agent_instance.migration_count += 1
                await self._update_orchestrator_state(
                    agent_instance_id, migration_event.target_device_id
                )

                # Cleanup source device
                await self._cleanup_source_device(
                    migration_event.source_device_id, agent_instance_id
                )

                return True
            # Rollback
            await self._restart_agent(
                agent_instance_id, migration_event.source_device_id
            )
            return False

        except Exception as e:
            logger.exception(f"Graceful migration failed: {e}")
            migration_event.error_message = str(e)
            return False

    async def _execute_scheduled_migration(
        self, migration_event: MigrationEvent
    ) -> bool:
        """Execute scheduled migration at optimal time."""
        # For now, execute as graceful migration
        # Future implementation could add scheduling logic
        return await self._execute_graceful_migration(migration_event)

    async def _execute_opportunistic_migration(
        self, migration_event: MigrationEvent
    ) -> bool:
        """Execute opportunistic migration when conditions are optimal."""
        # Wait for optimal conditions
        max_wait_time = 300.0  # 5 minutes max wait
        start_wait = time.time()

        while time.time() - start_wait < max_wait_time:
            # Check if conditions are optimal
            if await self._are_conditions_optimal_for_migration(migration_event):
                break
            await asyncio.sleep(30.0)  # Check every 30 seconds

        # Execute as graceful migration
        return await self._execute_graceful_migration(migration_event)

    async def _create_agent_checkpoint(
        self, agent_instance: AgentInstance, comprehensive: bool = False
    ) -> AgentCheckpoint:
        """Create checkpoint of agent state."""
        logger.debug(f"Creating checkpoint for agent {agent_instance.instance_id}")

        # Simulate agent state serialization
        # In real implementation, this would serialize the actual agent state
        state_data = {
            "instance_id": agent_instance.instance_id,
            "agent_type": agent_instance.agent_spec.agent_type.value,
            "configuration": agent_instance.agent_spec.metadata,
            "runtime_state": {
                "status": agent_instance.status,
                "health_score": agent_instance.health_score,
                "performance_metrics": agent_instance.performance_metrics,
            },
            "checkpoint_comprehensive": comprehensive,
        }

        serialized_state = msgpack.dumps(state_data)

        checkpoint = AgentCheckpoint(
            instance_id=agent_instance.instance_id,
            agent_type=agent_instance.agent_spec.agent_type,
            state_data=serialized_state,
            configuration=agent_instance.agent_spec.metadata,
            size_mb=len(serialized_state) / (1024 * 1024),
            source_device_id=self.p2p_node.node_id,
        )

        # Store checkpoint
        self.agent_checkpoints[agent_instance.instance_id] = checkpoint

        logger.debug(f"Checkpoint created: {checkpoint.size_mb:.2f}MB")
        return checkpoint

    async def _transfer_checkpoint(
        self, checkpoint: AgentCheckpoint, target_device_id: str
    ) -> bool:
        """Transfer checkpoint to target device."""
        logger.debug(f"Transferring checkpoint to {target_device_id}")

        try:
            # Create transfer message
            transfer_message = {
                "type": "AGENT_CHECKPOINT_TRANSFER",
                "checkpoint": {
                    "instance_id": checkpoint.instance_id,
                    "agent_type": checkpoint.agent_type.value,
                    "state_data": checkpoint.state_data.hex(),  # Convert bytes to hex
                    "configuration": checkpoint.configuration,
                    "size_mb": checkpoint.size_mb,
                    "source_device_id": checkpoint.source_device_id,
                },
            }

            # Send to target device
            success = await self.p2p_node.send_to_peer(
                target_device_id, transfer_message
            )

            if success:
                # Update statistics
                self.stats["data_transferred_gb"] += checkpoint.size_mb / 1024

            return success

        except Exception as e:
            logger.exception(f"Checkpoint transfer failed: {e}")
            return False

    async def _stop_agent(self, agent_instance_id: str, device_id: str) -> None:
        """Stop agent on specified device."""
        logger.debug(f"Stopping agent {agent_instance_id} on {device_id}")

        stop_message = {
            "type": "STOP_AGENT",
            "instance_id": agent_instance_id,
            "reason": "migration",
        }

        if device_id == self.p2p_node.node_id:
            # Local stop
            await self._stop_agent_locally(agent_instance_id)
        else:
            # Remote stop
            await self.p2p_node.send_to_peer(device_id, stop_message)

    async def _start_agent_from_checkpoint(
        self, checkpoint: AgentCheckpoint, target_device_id: str
    ) -> bool:
        """Start agent from checkpoint on target device."""
        logger.debug(
            f"Starting agent {checkpoint.instance_id} from checkpoint on {target_device_id}"
        )

        start_message = {
            "type": "START_AGENT_FROM_CHECKPOINT",
            "checkpoint": {
                "instance_id": checkpoint.instance_id,
                "agent_type": checkpoint.agent_type.value,
                "state_data": checkpoint.state_data.hex(),
                "configuration": checkpoint.configuration,
                "source_device_id": checkpoint.source_device_id,
            },
        }

        if target_device_id == self.p2p_node.node_id:
            # Local start
            return await self._start_agent_locally_from_checkpoint(checkpoint)
        # Remote start
        return await self.p2p_node.send_to_peer(target_device_id, start_message)

    async def _stop_agent_locally(self, agent_instance_id: str) -> None:
        """Stop agent locally."""
        # Simulate agent stop
        if agent_instance_id in self.agent_orchestrator.active_agents:
            agent_instance = self.agent_orchestrator.active_agents[agent_instance_id]
            agent_instance.status = "stopped"
            logger.debug(f"Agent {agent_instance_id} stopped locally")

    def _is_peer_trusted(self, peer_id: str, threshold: float = 0.5) -> bool:
        peer = self.p2p_node.peer_registry.get(peer_id)
        return bool(peer and peer.trust_score >= threshold)

    async def _start_agent_locally_from_checkpoint(
        self, checkpoint: AgentCheckpoint
    ) -> bool:
        """Start agent locally from checkpoint."""
        if checkpoint.source_device_id and not self._is_peer_trusted(
            checkpoint.source_device_id
        ):
            raise PermissionError(
                f"Untrusted checkpoint source: {checkpoint.source_device_id}"
            )

        try:
            state = msgpack.loads(checkpoint.state_data, raw=False)
        except Exception as e:  # pragma: no cover - msgpack error path
            raise ValueError("Invalid checkpoint payload") from e

        required_fields = {
            "instance_id",
            "agent_type",
            "configuration",
            "runtime_state",
            "checkpoint_comprehensive",
        }
        if not isinstance(state, dict) or not required_fields.issubset(state):
            raise ValueError("Checkpoint schema validation failed")

        if checkpoint.instance_id in self.agent_orchestrator.active_agents:
            agent_instance = self.agent_orchestrator.active_agents[
                checkpoint.instance_id
            ]
            agent_instance.status = "running"
            agent_instance.device_id = self.p2p_node.node_id

            logger.debug(
                f"Agent {checkpoint.instance_id} started locally from checkpoint"
            )
            return True

        return False

    async def _restart_agent(self, agent_instance_id: str, device_id: str) -> None:
        """Restart agent (rollback mechanism)."""
        logger.debug(f"Restarting agent {agent_instance_id} on {device_id}")

        restart_message = {"type": "RESTART_AGENT", "instance_id": agent_instance_id}

        if device_id == self.p2p_node.node_id:
            if agent_instance_id in self.agent_orchestrator.active_agents:
                agent_instance = self.agent_orchestrator.active_agents[
                    agent_instance_id
                ]
                agent_instance.status = "running"
        else:
            await self.p2p_node.send_to_peer(device_id, restart_message)

    async def _update_orchestrator_state(
        self, agent_instance_id: str, new_device_id: str
    ) -> None:
        """Update orchestrator state after migration."""
        if agent_instance_id in self.agent_orchestrator.active_agents:
            agent_instance = self.agent_orchestrator.active_agents[agent_instance_id]
            old_device_id = agent_instance.device_id

            # Update device assignments
            if old_device_id in self.agent_orchestrator.device_agent_assignments:
                if (
                    agent_instance_id
                    in self.agent_orchestrator.device_agent_assignments[old_device_id]
                ):
                    self.agent_orchestrator.device_agent_assignments[
                        old_device_id
                    ].remove(agent_instance_id)

            if new_device_id not in self.agent_orchestrator.device_agent_assignments:
                self.agent_orchestrator.device_agent_assignments[new_device_id] = []
            self.agent_orchestrator.device_agent_assignments[new_device_id].append(
                agent_instance_id
            )

            # Update instance
            agent_instance.device_id = new_device_id
            agent_instance.last_heartbeat = time.time()

            logger.info(
                f"Updated orchestrator state: agent {agent_instance_id} moved to {new_device_id}"
            )

    async def _wait_for_migration_window(self, agent_instance: AgentInstance) -> None:
        """Wait for optimal migration window."""
        # Simple implementation - wait for low activity period
        # In real implementation, this would monitor agent activity
        await asyncio.sleep(5.0)  # Simulate waiting for good window

    async def _prepare_target_device(self, target_device_id: str, agent_spec) -> None:
        """Prepare target device for migration."""
        prep_message = {
            "type": "PREPARE_FOR_AGENT_MIGRATION",
            "agent_type": agent_spec.agent_type.value,
            "memory_requirement_mb": agent_spec.memory_requirement_mb,
            "compute_requirement": agent_spec.compute_requirement,
        }

        if target_device_id != self.p2p_node.node_id:
            await self.p2p_node.send_to_peer(target_device_id, prep_message)

    async def _cleanup_source_device(
        self, source_device_id: str, agent_instance_id: str
    ) -> None:
        """Cleanup source device after successful migration."""
        cleanup_message = {
            "type": "CLEANUP_AFTER_MIGRATION",
            "instance_id": agent_instance_id,
        }

        if source_device_id != self.p2p_node.node_id:
            await self.p2p_node.send_to_peer(source_device_id, cleanup_message)

    async def _are_conditions_optimal_for_migration(
        self, migration_event: MigrationEvent
    ) -> bool:
        """Check if conditions are optimal for migration."""
        # Simple heuristic - check network load and device availability
        # In real implementation, this would be more sophisticated

        # Check target device health
        target_device_id = migration_event.target_device_id
        if target_device_id in self.p2p_node.peer_registry:
            peer = self.p2p_node.peer_registry[target_device_id]
            if peer.current_evolution_load > 0.7:  # Device is busy
                return False

        # Check network conditions
        if migration_event.source_device_id in self.p2p_node.peer_registry:
            source_peer = self.p2p_node.peer_registry[migration_event.source_device_id]
            if source_peer.latency_ms > 100:  # High latency
                return False

        return True

    async def _performance_monitor(self) -> None:
        """Monitor agent performance and trigger proactive migrations."""
        while True:
            try:
                if not self.config["enable_proactive_migration"]:
                    await asyncio.sleep(60.0)
                    continue

                # Check all active agents for performance issues
                for (
                    instance_id,
                    agent_instance,
                ) in self.agent_orchestrator.active_agents.items():
                    if not agent_instance.agent_spec.can_migrate:
                        continue

                    # Check health score
                    if (
                        agent_instance.health_score
                        < self.config["performance_threshold"]
                    ):
                        logger.info(
                            f"Performance degradation detected for agent {instance_id}"
                        )
                        await self.request_migration(
                            instance_id,
                            MigrationReason.PERFORMANCE_DEGRADATION,
                            strategy=MigrationStrategy.GRACEFUL,
                            priority=3,
                        )

                    # Check device resource constraints
                    device_id = agent_instance.device_id
                    if device_id in self.p2p_node.peer_registry:
                        peer = self.p2p_node.peer_registry[device_id]

                        # Check battery level
                        if peer.battery_percent and peer.battery_percent < 25:
                            logger.info(f"Low battery detected for device {device_id}")
                            await self.request_migration(
                                instance_id,
                                MigrationReason.BATTERY_LOW,
                                strategy=MigrationStrategy.GRACEFUL,
                                priority=4,
                            )

                        # Check device overload
                        if peer.current_evolution_load > 0.9:
                            logger.info(f"Device overload detected for {device_id}")
                            await self.request_migration(
                                instance_id,
                                MigrationReason.DEVICE_OVERLOAD,
                                strategy=MigrationStrategy.OPPORTUNISTIC,
                                priority=5,
                            )

                # Check every 2 minutes
                await asyncio.sleep(120.0)

            except Exception as e:
                logger.exception(f"Error in performance monitor: {e}")
                await asyncio.sleep(300.0)  # Back off on error

    async def _checkpoint_manager(self) -> None:
        """Manage periodic checkpoints for all agents."""
        while True:
            try:
                checkpoint_interval = self.config["checkpoint_interval_seconds"]

                # Create checkpoints for all running agents
                for (
                    instance_id,
                    agent_instance,
                ) in self.agent_orchestrator.active_agents.items():
                    if agent_instance.status == "running":
                        try:
                            await self._create_agent_checkpoint(agent_instance)
                        except Exception as e:
                            logger.exception(
                                f"Failed to create checkpoint for {instance_id}: {e}"
                            )

                # Cleanup old checkpoints (keep only latest)
                current_instances = set(self.agent_orchestrator.active_agents.keys())
                old_checkpoints = set(self.agent_checkpoints.keys()) - current_instances
                for old_instance_id in old_checkpoints:
                    del self.agent_checkpoints[old_instance_id]

                await asyncio.sleep(checkpoint_interval)

            except Exception as e:
                logger.exception(f"Error in checkpoint manager: {e}")
                await asyncio.sleep(300.0)

    def get_migration_status(self) -> dict[str, Any]:
        """Get current migration status."""
        return {
            "pending_migrations": len(self.pending_migrations),
            "active_migrations": len(self.active_migrations),
            "completed_migrations": len(self.migration_history),
            "migration_queue_sizes": {
                priority: len(requests)
                for priority, requests in self.migration_queues.items()
                if requests
            },
            "checkpoints_stored": len(self.agent_checkpoints),
            "statistics": self.stats.copy(),
            "config": self.config.copy(),
            "recent_migrations": [
                {
                    "event_id": event.event_id,
                    "agent_id": event.request.agent_instance_id,
                    "reason": event.request.reason.value,
                    "source_device": event.source_device_id,
                    "target_device": event.target_device_id,
                    "success": event.success,
                    "duration": event.migration_duration,
                    "downtime": event.downtime_duration,
                }
                for event in self.migration_history[-10:]  # Last 10 migrations
            ],
        }

    async def migrate_for_performance(self, agent_instance_id: str, reason: str) -> str:
        """Migrate agent for performance improvement."""
        migration_reason = MigrationReason.PERFORMANCE_DEGRADATION
        if "overload" in reason.lower():
            migration_reason = MigrationReason.DEVICE_OVERLOAD
        elif "battery" in reason.lower():
            migration_reason = MigrationReason.BATTERY_LOW

        return await self.request_migration(
            agent_instance_id,
            migration_reason,
            strategy=MigrationStrategy.GRACEFUL,
            priority=3,
        )

    async def cancel_migration(self, request_id: str) -> bool:
        """Cancel pending migration request."""
        if request_id in self.pending_migrations:
            request = self.pending_migrations[request_id]

            # Remove from queue
            for priority_queue in self.migration_queues.values():
                if request in priority_queue:
                    priority_queue.remove(request)
                    break

            del self.pending_migrations[request_id]
            logger.info(f"Migration request {request_id} cancelled")
            return True

        return False
