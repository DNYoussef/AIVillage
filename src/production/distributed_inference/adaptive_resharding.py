"""Adaptive Resharding Manager for Dynamic Network Changes.

This module handles dynamic resharding when devices join/leave the network,
ensuring continuous inference operation without interruption.
"""

import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..communications.p2p.p2p_node import P2PNode
from .model_sharding_engine import ModelShard, ModelShardingEngine, ShardingPlan

logger = logging.getLogger(__name__)


class ReshardingReason(Enum):
    """Reasons for triggering resharding."""

    DEVICE_JOINED = "device_joined"
    DEVICE_LEFT = "device_left"
    DEVICE_FAILED = "device_failed"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MEMORY_CONSTRAINT = "memory_constraint"
    LOAD_IMBALANCE = "load_imbalance"
    MANUAL_TRIGGER = "manual_trigger"


class ReshardingStrategy(Enum):
    """Resharding strategies."""

    MINIMAL_DISRUPTION = "minimal_disruption"  # Minimize changes to existing shards
    OPTIMAL_REBALANCE = "optimal_rebalance"  # Full rebalancing for optimal performance
    INCREMENTAL = "incremental"  # Gradual resharding over time
    EMERGENCY = "emergency"  # Fast resharding for critical situations


@dataclass
class ReshardingEvent:
    """Represents a resharding event."""

    event_id: str
    reason: ReshardingReason
    trigger_device_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    old_plan: ShardingPlan | None = None
    new_plan: ShardingPlan | None = None
    success: bool = False
    duration_seconds: float = 0.0
    disruption_score: float = 0.0  # 0-1, how disruptive the resharding was
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize event to a JSON-friendly dict."""
        return {
            "event_id": self.event_id,
            "reason": self.reason.value,
            "trigger_device_id": self.trigger_device_id,
            "timestamp": self.timestamp,
            "success": self.success,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReshardingEvent":
        """Deserialize event from dict."""
        return cls(
            event_id=data["event_id"],
            reason=ReshardingReason(data["reason"]),
            trigger_device_id=data.get("trigger_device_id"),
            timestamp=data.get("timestamp", time.time()),
            success=data.get("success", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ReshardingConfig:
    """Configuration for adaptive resharding."""

    enable_auto_resharding: bool = True
    min_resharding_interval_seconds: float = 30.0
    performance_threshold: float = 0.7  # Trigger resharding if performance drops below this
    load_imbalance_threshold: float = 0.3  # Max acceptable load imbalance
    device_stability_window_seconds: float = 60.0  # Wait time before considering device stable
    max_concurrent_resharding: int = 1
    graceful_handoff_timeout_seconds: float = 30.0
    emergency_resharding_threshold: float = 0.5  # Critical performance threshold


class AdaptiveReshardingManager:
    """Manages adaptive resharding based on network changes."""

    def __init__(
        self,
        sharding_engine: ModelShardingEngine,
        p2p_node: P2PNode,
        config: ReshardingConfig | None = None,
        state_file: str = "adaptive_resharding_state.json",
    ) -> None:
        self.sharding_engine = sharding_engine
        self.p2p_node = p2p_node
        self.config = config or ReshardingConfig()
        self.state_file = state_file

        # Resharding state
        self.resharding_active = False
        self.last_resharding_time = 0.0
        self.resharding_history: list[ReshardingEvent] = []
        self.pending_device_changes: set[str] = set()

        # Device monitoring
        self.device_stability_tracker: dict[str, float] = {}  # device_id -> join_time
        self.device_performance_history: dict[str, list[float]] = {}

        # Performance monitoring
        self.performance_monitor_task: asyncio.Task | None = None
        self.network_change_callbacks: list[Callable] = []

        # Statistics
        self.stats = {
            "total_resharding_events": 0,
            "successful_resharding": 0,
            "failed_resharding": 0,
            "avg_resharding_time": 0.0,
            "avg_disruption_score": 0.0,
            "device_joins_handled": 0,
            "device_failures_handled": 0,
        }

        # Register P2P event handlers
        self._register_p2p_handlers()

        # Load persisted history
        self._load_history_from_disk()

        logger.info("AdaptiveReshardingManager initialized")

    async def start_monitoring(self) -> None:
        """Start monitoring for resharding triggers."""
        if self.performance_monitor_task:
            logger.warning("Resharding monitoring already active")
            return

        await self._replay_incomplete_events()
        self.performance_monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Adaptive resharding monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop monitoring."""
        if self.performance_monitor_task:
            self.performance_monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.performance_monitor_task
            self.performance_monitor_task = None

        logger.info("Adaptive resharding monitoring stopped")

    def _load_history_from_disk(self) -> None:
        """Load resharding history from disk."""
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, encoding="utf-8") as f:
                data = json.load(f)
            self.resharding_history = [ReshardingEvent.from_dict(d) for d in data]
            if self.resharding_history:
                self.last_resharding_time = self.resharding_history[-1].timestamp
        except Exception as e:
            logger.warning(f"Failed to load resharding state: {e}")

    def _save_history_to_disk(self) -> None:
        """Persist resharding history to disk."""
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump([e.to_dict() for e in self.resharding_history], f)
        except Exception as e:
            logger.warning(f"Failed to save resharding state: {e}")

    async def _replay_incomplete_events(self) -> None:
        """Replay any resharding events that were incomplete before restart."""
        pending = [e for e in self.resharding_history if not e.success]
        for event in pending:
            strategy_name = event.metadata.get("strategy", ReshardingStrategy.OPTIMAL_REBALANCE.value)
            strategy = ReshardingStrategy(strategy_name)
            logger.info(f"Replaying incomplete resharding event {event.event_id}")
            await self.trigger_resharding(event.reason, event.trigger_device_id, strategy=strategy)
            event.success = True
        if pending:
            self._save_history_to_disk()

    def _register_p2p_handlers(self) -> None:
        """Register P2P network event handlers."""
        # These would be registered with the P2P node's event system
        # For now, we'll implement polling in the monitoring loop

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for resharding triggers."""
        while True:
            try:
                await asyncio.sleep(5.0)  # Check every 5 seconds

                if not self.config.enable_auto_resharding:
                    continue

                # Check for various resharding triggers
                await self._check_device_changes()
                await self._check_performance_degradation()
                await self._check_load_imbalance()
                await self._process_pending_changes()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in resharding monitoring loop: {e}")
                await asyncio.sleep(10.0)  # Back off on error

    async def _check_device_changes(self) -> None:
        """Check for device joins/leaves."""
        current_devices = set()

        # Get current peer list
        for peer_id in self.p2p_node.peer_registry:
            peer = self.p2p_node.peer_registry[peer_id]
            if peer.is_suitable_for_evolution():
                current_devices.add(peer_id)

        # Add local device if suitable
        if self.p2p_node.local_capabilities and self.p2p_node.local_capabilities.is_suitable_for_evolution():
            current_devices.add(self.p2p_node.node_id)

        # Compare with devices in current sharding plan
        if self.sharding_engine.current_sharding_plan:
            plan_devices = set()
            for shard in self.sharding_engine.current_sharding_plan.shards:
                plan_devices.add(shard.device_id)

            # Check for new devices
            new_devices = current_devices - plan_devices
            for device_id in new_devices:
                await self._handle_device_joined(device_id)

            # Check for missing devices
            missing_devices = plan_devices - current_devices
            for device_id in missing_devices:
                await self._handle_device_left(device_id)

    async def _check_performance_degradation(self) -> None:
        """Check for overall performance degradation."""
        if not self.sharding_engine.current_sharding_plan:
            return

        # Get current performance metrics (placeholder implementation)
        current_performance = await self._get_current_performance()

        if current_performance < self.config.performance_threshold:
            logger.warning(f"Performance degradation detected: {current_performance:.3f}")

            if current_performance < self.config.emergency_resharding_threshold:
                await self.trigger_resharding(
                    ReshardingReason.PERFORMANCE_DEGRADATION,
                    strategy=ReshardingStrategy.EMERGENCY,
                )
            else:
                await self.trigger_resharding(
                    ReshardingReason.PERFORMANCE_DEGRADATION,
                    strategy=ReshardingStrategy.OPTIMAL_REBALANCE,
                )

    async def _check_load_imbalance(self) -> None:
        """Check for compute load imbalance."""
        if not self.sharding_engine.current_sharding_plan:
            return

        current_balance = self.sharding_engine.current_sharding_plan.compute_balance_score

        if current_balance < (1.0 - self.config.load_imbalance_threshold):
            logger.info(f"Load imbalance detected: {current_balance:.3f}")
            await self.trigger_resharding(ReshardingReason.LOAD_IMBALANCE, strategy=ReshardingStrategy.INCREMENTAL)

    async def _handle_device_joined(self, device_id: str) -> None:
        """Handle new device joining the network."""
        logger.info(f"Device joined: {device_id}")

        # Track device stability
        self.device_stability_tracker[device_id] = time.time()
        self.stats["device_joins_handled"] += 1

        # Don't immediately reshard - wait for stability
        self.pending_device_changes.add(device_id)

    async def _handle_device_left(self, device_id: str) -> None:
        """Handle device leaving the network."""
        logger.warning(f"Device left/failed: {device_id}")

        self.stats["device_failures_handled"] += 1

        # Check if this device has active shards
        has_shards = any(shard.device_id == device_id for _, shard in self.sharding_engine.active_shards.items())

        if has_shards:
            # Immediate resharding needed
            await self.trigger_resharding(
                ReshardingReason.DEVICE_FAILED,
                trigger_device_id=device_id,
                strategy=ReshardingStrategy.EMERGENCY,
            )

        # Clean up tracking
        self.device_stability_tracker.pop(device_id, None)
        self.device_performance_history.pop(device_id, None)
        self.pending_device_changes.discard(device_id)

    async def _process_pending_changes(self) -> None:
        """Process pending device changes after stability period."""
        current_time = time.time()
        stable_devices = []

        for device_id in list(self.pending_device_changes):
            join_time = self.device_stability_tracker.get(device_id, current_time)
            if current_time - join_time >= self.config.device_stability_window_seconds:
                stable_devices.append(device_id)
                self.pending_device_changes.remove(device_id)

        if stable_devices:
            logger.info(f"Stable devices ready for resharding: {stable_devices}")
            await self.trigger_resharding(
                ReshardingReason.DEVICE_JOINED,
                strategy=ReshardingStrategy.OPTIMAL_REBALANCE,
            )

    async def trigger_resharding(
        self,
        reason: ReshardingReason,
        trigger_device_id: str | None = None,
        strategy: ReshardingStrategy = ReshardingStrategy.OPTIMAL_REBALANCE,
    ) -> bool:
        """Trigger resharding with specified reason and strategy."""
        # Check if resharding is allowed
        if not self._can_reshard():
            logger.info(f"Resharding not allowed: reason={reason.value}")
            return False

        event = ReshardingEvent(
            event_id=str(uuid.uuid4()),
            reason=reason,
            trigger_device_id=trigger_device_id,
            old_plan=self.sharding_engine.current_sharding_plan,
        )
        event.metadata["strategy"] = strategy.value
        self.resharding_history.append(event)
        self._save_history_to_disk()

        try:
            logger.info(f"Starting resharding: reason={reason.value}, strategy={strategy.value}")
            self.resharding_active = True
            start_time = time.time()

            # Execute resharding
            success = await self._execute_resharding(event, strategy)

            # Update event
            event.success = success
            event.duration_seconds = time.time() - start_time
            event.new_plan = self.sharding_engine.current_sharding_plan

            # Calculate disruption score
            event.disruption_score = self._calculate_disruption_score(event.old_plan, event.new_plan)

            # Update statistics
            self.stats["total_resharding_events"] += 1
            if success:
                self.stats["successful_resharding"] += 1
            else:
                self.stats["failed_resharding"] += 1

            self.stats["avg_resharding_time"] = (self.stats["avg_resharding_time"] + event.duration_seconds) / 2
            self.stats["avg_disruption_score"] = (self.stats["avg_disruption_score"] + event.disruption_score) / 2

            self.last_resharding_time = time.time()
            self._save_history_to_disk()

            logger.info(f"Resharding completed: success={success}, duration={event.duration_seconds:.2f}s")
            return success

        except Exception as e:
            logger.exception(f"Resharding failed: {e}")
            event.success = False
            event.metadata["error"] = str(e)
            self._save_history_to_disk()
            return False

        finally:
            self._save_history_to_disk()
            self.resharding_active = False

    def _can_reshard(self) -> bool:
        """Check if resharding is currently allowed."""
        # Check if already resharding
        if self.resharding_active:
            return False

        # Check minimum interval
        current_time = time.time()
        if current_time - self.last_resharding_time < self.config.min_resharding_interval_seconds:
            return False

        # Check if model is currently sharded
        return self.sharding_engine.current_sharding_plan

    async def _execute_resharding(self, event: ReshardingEvent, strategy: ReshardingStrategy) -> bool:
        """Execute the actual resharding process."""
        try:
            # Choose resharding strategy
            if strategy == ReshardingStrategy.MINIMAL_DISRUPTION:
                return await self._minimal_disruption_resharding(event)
            if strategy == ReshardingStrategy.OPTIMAL_REBALANCE:
                return await self._optimal_rebalance_resharding(event)
            if strategy == ReshardingStrategy.INCREMENTAL:
                return await self._incremental_resharding(event)
            if strategy == ReshardingStrategy.EMERGENCY:
                return await self._emergency_resharding(event)
            logger.error(f"Unknown resharding strategy: {strategy}")
            return False

        except Exception as e:
            logger.exception(f"Resharding execution failed: {e}")
            return False

    async def _minimal_disruption_resharding(self, event: ReshardingEvent) -> bool:
        """Resharding with minimal disruption to existing shards."""
        if not event.old_plan:
            return False

        # Get current device profiles
        device_profiles = await self.sharding_engine._get_device_profiles()

        # Identify which shards need to move
        shards_to_move = []
        stable_shards = []

        for shard in event.old_plan.shards:
            # Check if shard's device is still available
            device_available = any(d.device_id == shard.device_id for d in device_profiles)

            if not device_available:
                shards_to_move.append(shard)
            else:
                stable_shards.append(shard)

        if not shards_to_move:
            logger.info("No shards need to be moved")
            return True

        # Find new homes for shards that need to move
        available_devices = [d for d in device_profiles if not any(s.device_id == d.device_id for s in stable_shards)]

        if len(available_devices) < len(shards_to_move):
            logger.warning("Not enough available devices for minimal disruption")
            # Fall back to full resharding
            return await self._optimal_rebalance_resharding(event)

        # Reassign shards to new devices
        new_shards = stable_shards.copy()
        for i, shard in enumerate(shards_to_move):
            new_device = available_devices[i % len(available_devices)]

            new_shard = ModelShard(
                shard_id=str(uuid.uuid4()),
                device_id=new_device.device_id,
                layer_indices=shard.layer_indices,
                parameters_count=shard.parameters_count,
                memory_mb=shard.memory_mb,
                compute_requirement=shard.compute_requirement,
                dependencies=shard.dependencies,
                metadata=shard.metadata,
            )
            new_shards.append(new_shard)

        # Create new plan
        new_plan = ShardingPlan(
            model_name=event.old_plan.model_name,
            total_shards=len(new_shards),
            shards=new_shards,
            activation_routing=event.old_plan.activation_routing.copy(),  # Keep same routing
            memory_efficiency=self.sharding_engine._calculate_memory_efficiency(new_shards, device_profiles),
            compute_balance_score=self.sharding_engine._calculate_compute_balance(new_shards, device_profiles),
        )

        # Activate new plan
        self.sharding_engine.current_sharding_plan = new_plan
        await self.sharding_engine._activate_sharding_plan(new_plan)

        return True

    async def _optimal_rebalance_resharding(self, event: ReshardingEvent) -> bool:
        """Full rebalancing for optimal performance."""
        if not event.old_plan:
            return False

        # Create completely new sharding plan
        model_analysis = await self.sharding_engine._analyze_model(event.old_plan.model_name)
        device_profiles = await self.sharding_engine._get_device_profiles()

        # Use hybrid strategy for best results
        new_plan = await self.sharding_engine._create_hybrid_plan(model_analysis, device_profiles)
        optimized_plan = await self.sharding_engine._optimize_sharding_plan(new_plan, device_profiles)

        # Activate new plan
        self.sharding_engine.current_sharding_plan = optimized_plan
        await self.sharding_engine._activate_sharding_plan(optimized_plan)

        return True

    async def _incremental_resharding(self, event: ReshardingEvent) -> bool:
        """Gradual resharding to minimize impact."""
        # For now, implement as minimal disruption
        # Future implementation could move shards one at a time
        return await self._minimal_disruption_resharding(event)

    async def _emergency_resharding(self, event: ReshardingEvent) -> bool:
        """Fast resharding for critical situations."""
        # Use fastest possible resharding - optimal rebalance but with reduced optimization
        if not event.old_plan:
            return False

        # Get available devices quickly
        device_profiles = await self.sharding_engine._get_device_profiles()

        if not device_profiles:
            logger.error("No devices available for emergency resharding")
            return False

        # Use simple sequential strategy for speed
        model_analysis = await self.sharding_engine._analyze_model(event.old_plan.model_name)
        new_plan = await self.sharding_engine._create_sequential_plan(model_analysis, device_profiles)

        # Skip optimization for speed
        self.sharding_engine.current_sharding_plan = new_plan
        await self.sharding_engine._activate_sharding_plan(new_plan)

        return True

    def _calculate_disruption_score(self, old_plan: ShardingPlan | None, new_plan: ShardingPlan | None) -> float:
        """Calculate how disruptive the resharding was (0=no disruption, 1=complete change)."""
        if not old_plan or not new_plan:
            return 1.0

        # Count shards that moved to different devices
        old_assignments = {shard.device_id: shard.layer_indices for shard in old_plan.shards}
        new_assignments = {shard.device_id: shard.layer_indices for shard in new_plan.shards}

        total_layers = sum(len(indices) for indices in old_assignments.values())
        moved_layers = 0

        for device_id, new_layers in new_assignments.items():
            old_layers = old_assignments.get(device_id, [])
            # Count layers that are now on different devices
            for layer in new_layers:
                if layer not in old_layers:
                    moved_layers += 1

        disruption_score = moved_layers / total_layers if total_layers > 0 else 1.0
        return min(1.0, disruption_score)

    async def _get_current_performance(self) -> float:
        """Get current system performance score (placeholder)."""
        # This would integrate with actual performance monitoring
        # For now, return a mock score based on network health

        if not self.p2p_node.peer_registry:
            return 0.5

        # Simple heuristic based on connected peers and their capabilities
        total_score = 0.0
        count = 0

        for peer in self.p2p_node.peer_registry.values():
            if peer.is_suitable_for_evolution():
                score = (
                    (peer.trust_score * 0.3)
                    + (peer.evolution_capacity * 0.4)
                    + ((100 - peer.current_evolution_load) / 100 * 0.3)
                )
                total_score += score
                count += 1

        return total_score / count if count > 0 else 0.5

    def get_resharding_status(self) -> dict[str, Any]:
        """Get current resharding status and statistics."""
        return {
            "monitoring_active": self.performance_monitor_task is not None,
            "resharding_active": self.resharding_active,
            "config": {
                "auto_resharding_enabled": self.config.enable_auto_resharding,
                "performance_threshold": self.config.performance_threshold,
                "load_imbalance_threshold": self.config.load_imbalance_threshold,
                "min_interval_seconds": self.config.min_resharding_interval_seconds,
            },
            "statistics": self.stats.copy(),
            "recent_events": [
                {
                    "event_id": event.event_id,
                    "reason": event.reason.value,
                    "timestamp": event.timestamp,
                    "success": event.success,
                    "duration_seconds": event.duration_seconds,
                    "disruption_score": event.disruption_score,
                }
                for event in self.resharding_history[-10:]  # Last 10 events
            ],
            "pending_changes": len(self.pending_device_changes),
            "device_stability": {
                device_id: time.time() - join_time for device_id, join_time in self.device_stability_tracker.items()
            },
        }

    def register_network_change_callback(self, callback: Callable) -> None:
        """Register callback for network change events."""
        self.network_change_callbacks.append(callback)

    async def force_resharding(self, strategy: ReshardingStrategy = ReshardingStrategy.OPTIMAL_REBALANCE) -> bool:
        """Force immediate resharding (for testing/manual intervention)."""
        return await self.trigger_resharding(ReshardingReason.MANUAL_TRIGGER, strategy=strategy)
