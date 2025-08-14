"""
SCION-aware Navigator integration for AIVillage transport selection.
Integrates SCION paths into the Navigator's transport decision-making process.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..core.message_types import Message
from ..core.transport_manager import TransportManager
from ..transport.scion_gateway import (
    GatewayConfig,
    SCIONGateway,
    SCIONGatewayError,
    SCIONPath,
    SCIONTransport,
)

logger = logging.getLogger(__name__)


class TransportPriority(Enum):
    """Transport priority levels."""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    FALLBACK = 5


@dataclass
class TransportCandidate:
    """Transport option with routing information."""

    transport_type: str
    endpoint: str
    priority: TransportPriority
    estimated_latency_ms: float
    reliability_score: float  # 0.0 - 1.0
    cost_factor: float  # 0.0 - 1.0 (lower is cheaper)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Navigation routing decision."""

    primary_transport: TransportCandidate
    backup_transports: list[TransportCandidate]
    decision_reason: str
    confidence_score: float  # 0.0 - 1.0
    estimated_total_time_ms: float
    route_metadata: dict[str, Any] = field(default_factory=dict)


class SCIONAwareNavigator:
    """
    Enhanced Navigator that integrates SCION paths into transport selection.
    Provides intelligent routing decisions based on path quality, cost, and availability.
    """

    def __init__(
        self,
        scion_config: GatewayConfig,
        transport_manager: TransportManager,
        enable_scion_preference: bool = True,
        scion_weight: float = 2.0,  # Preference multiplier for SCION
    ):
        self.scion_config = scion_config
        self.transport_manager = transport_manager
        self.enable_scion_preference = enable_scion_preference
        self.scion_weight = scion_weight

        # SCION components
        self.scion_transport: SCIONTransport | None = None
        self._scion_gateway: SCIONGateway | None = None

        # Path cache and statistics
        self._path_cache: dict[str, list[SCIONPath]] = {}
        self._path_cache_ttl: dict[str, float] = {}
        self._path_cache_duration = 300.0  # 5 minutes

        # Transport performance tracking
        self._transport_stats: dict[str, dict[str, float]] = {}
        self._decision_history: list[RoutingDecision] = []
        self._max_history = 1000

        # Configuration
        self.config = {
            "scion_preference_threshold": 0.7,  # Min reliability to prefer SCION
            "latency_weight": 0.4,
            "reliability_weight": 0.4,
            "cost_weight": 0.2,
            "path_discovery_interval": 60.0,  # seconds
            "max_backup_transports": 3,
            "decision_confidence_threshold": 0.6,
        }

        # Background tasks
        self._path_discovery_task: asyncio.Task | None = None
        self._is_running = False

    async def start(self) -> None:
        """Start the SCION-aware Navigator."""
        if self._is_running:
            return

        logger.info("Starting SCION-aware Navigator")

        try:
            # Initialize SCION transport if enabled
            if self.enable_scion_preference:
                await self._initialize_scion()

            # Start background tasks
            self._path_discovery_task = asyncio.create_task(self._path_discovery_worker())

            self._is_running = True
            logger.info("SCION-aware Navigator started")

        except Exception as e:
            logger.error(f"Failed to start SCION-aware Navigator: {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the Navigator and cleanup resources."""
        if not self._is_running:
            return

        logger.info("Stopping SCION-aware Navigator")
        self._is_running = False

        # Stop background tasks
        if self._path_discovery_task:
            self._path_discovery_task.cancel()
            try:
                await self._path_discovery_task
            except asyncio.CancelledError:
                pass

        # Cleanup SCION transport
        if self.scion_transport:
            await self.scion_transport.cleanup()
            self.scion_transport = None

        if self._scion_gateway:
            await self._scion_gateway.stop()
            self._scion_gateway = None

        logger.info("SCION-aware Navigator stopped")

    async def find_optimal_route(
        self,
        destination: str,
        message: Message,
        constraints: dict[str, Any] | None = None,
    ) -> RoutingDecision:
        """
        Find optimal transport route to destination.

        Args:
            destination: Target destination (ISD-AS for SCION, IP/domain for others)
            message: Message to be sent (affects routing decisions)
            constraints: Optional routing constraints (latency, cost, etc.)

        Returns:
            RoutingDecision with primary and backup transports
        """
        start_time = time.time()
        constraints = constraints or {}

        logger.debug(f"Finding optimal route to {destination}")

        # Gather all available transport candidates
        candidates = await self._gather_transport_candidates(destination, message, constraints)

        if not candidates:
            # Fallback decision
            return RoutingDecision(
                primary_transport=self._create_fallback_transport(),
                backup_transports=[],
                decision_reason="no_transports_available",
                confidence_score=0.1,
                estimated_total_time_ms=float("inf"),
            )

        # Score and rank candidates
        scored_candidates = await self._score_transport_candidates(candidates, destination, message, constraints)

        # Select primary and backup transports
        primary = scored_candidates[0]
        backups = scored_candidates[1 : self.config["max_backup_transports"] + 1]

        # Calculate confidence and decision reason
        confidence = self._calculate_decision_confidence(scored_candidates)
        reason = self._determine_decision_reason(primary, candidates)

        decision = RoutingDecision(
            primary_transport=primary,
            backup_transports=backups,
            decision_reason=reason,
            confidence_score=confidence,
            estimated_total_time_ms=primary.estimated_latency_ms,
            route_metadata={
                "candidates_evaluated": len(candidates),
                "decision_time_ms": (time.time() - start_time) * 1000,
                "scion_available": any(c.transport_type == "scion" for c in candidates),
            },
        )

        # Record decision for analysis
        self._record_decision(decision)

        logger.debug(f"Route decision: {primary.transport_type} to {destination} " f"(confidence: {confidence:.2f})")

        return decision

    async def get_scion_paths(self, destination: str, force_refresh: bool = False) -> list[SCIONPath]:
        """Get available SCION paths to destination."""
        if not self._scion_gateway:
            return []

        cache_key = destination
        current_time = time.time()

        # Check cache first
        if not force_refresh and cache_key in self._path_cache:
            cache_time = self._path_cache_ttl.get(cache_key, 0)
            if current_time - cache_time < self._path_cache_duration:
                return self._path_cache[cache_key]

        # Query fresh paths
        try:
            paths = await self._scion_gateway.query_paths(destination)
            self._path_cache[cache_key] = paths
            self._path_cache_ttl[cache_key] = current_time

            logger.debug(f"Retrieved {len(paths)} SCION paths to {destination}")
            return paths

        except SCIONGatewayError as e:
            logger.warning(f"Failed to query SCION paths to {destination}: {e}")
            return []

    async def send_message_with_routing(
        self,
        message: Message,
        destination: str,
        route_decision: RoutingDecision | None = None,
    ) -> bool:
        """Send message using optimal routing decision."""
        if not route_decision:
            route_decision = await self.find_optimal_route(destination, message)

        # Attempt primary transport first
        success = await self._attempt_transport_send(message, destination, route_decision.primary_transport)

        if success:
            self._record_transport_success(route_decision.primary_transport)
            return True

        # Try backup transports
        for backup_transport in route_decision.backup_transports:
            logger.info(f"Primary transport failed, trying backup: {backup_transport.transport_type}")

            success = await self._attempt_transport_send(message, destination, backup_transport)

            if success:
                self._record_transport_success(backup_transport)
                return True

            self._record_transport_failure(backup_transport)

        logger.error(f"All transports failed for destination {destination}")
        return False

    def get_navigation_statistics(self) -> dict[str, Any]:
        """Get comprehensive navigation statistics."""
        return {
            "decisions_made": len(self._decision_history),
            "transport_stats": self._transport_stats.copy(),
            "path_cache_entries": len(self._path_cache),
            "scion_enabled": self.enable_scion_preference,
            "scion_gateway_connected": self._scion_gateway is not None,
            "recent_decisions": [
                {
                    "primary_transport": d.primary_transport.transport_type,
                    "reason": d.decision_reason,
                    "confidence": d.confidence_score,
                }
                for d in self._decision_history[-10:]
            ],
        }

    # Private methods

    async def _initialize_scion(self) -> None:
        """Initialize SCION transport components."""
        try:
            # Create SCION transport
            self.scion_transport = SCIONTransport(self.scion_config)
            await self.scion_transport.initialize(self.transport_manager)

            # Create gateway for path queries
            self._scion_gateway = SCIONGateway(self.scion_config)
            await self._scion_gateway.start()

            logger.info("SCION components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SCION: {e}")
            # Continue without SCION but log the failure
            self.enable_scion_preference = False
            raise

    async def _gather_transport_candidates(
        self, destination: str, message: Message, constraints: dict[str, Any]
    ) -> list[TransportCandidate]:
        """Gather all available transport candidates for destination."""
        candidates = []

        # Add SCION candidates if available
        if self.enable_scion_preference and self._scion_gateway:
            scion_candidates = await self._get_scion_candidates(destination)
            candidates.extend(scion_candidates)

        # Add traditional transport candidates
        traditional_candidates = await self._get_traditional_transport_candidates(destination, message, constraints)
        candidates.extend(traditional_candidates)

        return candidates

    async def _get_scion_candidates(self, destination: str) -> list[TransportCandidate]:
        """Get SCION transport candidates."""
        candidates = []

        try:
            paths = await self.get_scion_paths(destination)

            for path in paths:
                if not path.is_healthy:
                    continue

                # Calculate scores based on path quality
                reliability_score = max(0.0, 1.0 - path.loss_rate)
                latency_ms = path.rtt_us / 1000.0

                # SCION is generally cost-effective for long distances
                cost_factor = 0.3  # Lower cost factor (cheaper)

                # Apply SCION preference weight
                if path.is_active:
                    reliability_score *= self.scion_weight

                candidate = TransportCandidate(
                    transport_type="scion",
                    endpoint=destination,
                    priority=TransportPriority.HIGH if path.is_active else TransportPriority.NORMAL,
                    estimated_latency_ms=latency_ms,
                    reliability_score=min(1.0, reliability_score),
                    cost_factor=cost_factor,
                    metadata={
                        "path_id": path.path_id,
                        "path_fingerprint": path.fingerprint,
                        "rtt_us": path.rtt_us,
                        "loss_rate": path.loss_rate,
                        "is_active": path.is_active,
                    },
                )

                candidates.append(candidate)

        except Exception as e:
            logger.warning(f"Failed to get SCION candidates: {e}")

        return candidates

    async def _get_traditional_transport_candidates(
        self, destination: str, message: Message, constraints: dict[str, Any]
    ) -> list[TransportCandidate]:
        """Get traditional (non-SCION) transport candidates."""
        candidates = []

        # BitChat P2P transport
        candidates.append(
            TransportCandidate(
                transport_type="bitchat",
                endpoint=destination,
                priority=TransportPriority.HIGH,
                estimated_latency_ms=150.0,  # Typical P2P latency
                reliability_score=0.85,
                cost_factor=0.1,  # Very low cost
                metadata={"transport_method": "p2p_mesh"},
            )
        )

        # HTTP/HTTPS transport
        candidates.append(
            TransportCandidate(
                transport_type="http",
                endpoint=f"https://{destination}",
                priority=TransportPriority.NORMAL,
                estimated_latency_ms=100.0,
                reliability_score=0.9,
                cost_factor=0.5,
                metadata={"transport_method": "https"},
            )
        )

        # WebSocket transport
        candidates.append(
            TransportCandidate(
                transport_type="websocket",
                endpoint=f"wss://{destination}",
                priority=TransportPriority.NORMAL,
                estimated_latency_ms=120.0,
                reliability_score=0.8,
                cost_factor=0.6,
                metadata={"transport_method": "websocket"},
            )
        )

        # Local/offline transport (file-based)
        candidates.append(
            TransportCandidate(
                transport_type="offline",
                endpoint="local",
                priority=TransportPriority.LOW,
                estimated_latency_ms=50.0,  # Local file system
                reliability_score=0.95,  # Very reliable locally
                cost_factor=0.0,  # No cost
                metadata={"transport_method": "file_system"},
            )
        )

        return candidates

    async def _score_transport_candidates(
        self,
        candidates: list[TransportCandidate],
        destination: str,
        message: Message,
        constraints: dict[str, Any],
    ) -> list[TransportCandidate]:
        """Score and rank transport candidates."""

        for candidate in candidates:
            # Base score from weighted factors
            latency_score = max(0.0, 1.0 - (candidate.estimated_latency_ms / 1000.0))
            reliability_score = candidate.reliability_score
            cost_score = 1.0 - candidate.cost_factor

            composite_score = (
                self.config["latency_weight"] * latency_score
                + self.config["reliability_weight"] * reliability_score
                + self.config["cost_weight"] * cost_score
            )

            # Apply priority multipliers
            priority_multiplier = {
                TransportPriority.CRITICAL: 1.5,
                TransportPriority.HIGH: 1.2,
                TransportPriority.NORMAL: 1.0,
                TransportPriority.LOW: 0.8,
                TransportPriority.FALLBACK: 0.5,
            }[candidate.priority]

            composite_score *= priority_multiplier

            # Apply constraint penalties
            composite_score = await self._apply_constraint_penalties(composite_score, candidate, constraints)

            # Apply historical performance adjustments
            composite_score = self._apply_historical_adjustments(composite_score, candidate)

            # Store final score in metadata
            candidate.metadata["composite_score"] = composite_score

        # Sort by composite score (descending)
        return sorted(
            candidates,
            key=lambda c: c.metadata.get("composite_score", 0.0),
            reverse=True,
        )

    async def _apply_constraint_penalties(
        self, score: float, candidate: TransportCandidate, constraints: dict[str, Any]
    ) -> float:
        """Apply penalty for constraint violations."""
        penalty = 0.0

        # Max latency constraint
        if "max_latency_ms" in constraints:
            max_latency = constraints["max_latency_ms"]
            if candidate.estimated_latency_ms > max_latency:
                penalty += 0.3

        # Min reliability constraint
        if "min_reliability" in constraints:
            min_reliability = constraints["min_reliability"]
            if candidate.reliability_score < min_reliability:
                penalty += 0.5

        # Transport type exclusions
        if "excluded_transports" in constraints:
            excluded = constraints["excluded_transports"]
            if candidate.transport_type in excluded:
                penalty += 0.8

        return max(0.0, score - penalty)

    def _apply_historical_adjustments(self, score: float, candidate: TransportCandidate) -> float:
        """Apply adjustments based on historical performance."""
        transport_type = candidate.transport_type

        if transport_type not in self._transport_stats:
            return score  # No history available

        stats = self._transport_stats[transport_type]

        # Success rate adjustment
        success_rate = stats.get("success_rate", 0.5)
        score *= 0.5 + 0.5 * success_rate  # Adjust by 50-100% based on success rate

        # Recent performance adjustment
        recent_latency = stats.get("recent_avg_latency_ms", candidate.estimated_latency_ms)
        if recent_latency > candidate.estimated_latency_ms * 1.5:
            score *= 0.9  # Penalty for recently poor performance

        return score

    def _calculate_decision_confidence(self, scored_candidates: list[TransportCandidate]) -> float:
        """Calculate confidence in the routing decision."""
        if len(scored_candidates) < 2:
            return 0.5  # Low confidence with limited options

        # Get scores of top two candidates
        best_score = scored_candidates[0].metadata.get("composite_score", 0.0)
        second_score = scored_candidates[1].metadata.get("composite_score", 0.0)

        # Confidence based on score gap
        if second_score == 0:
            return 0.8

        score_ratio = best_score / second_score
        confidence = min(0.95, 0.5 + (score_ratio - 1.0) * 0.3)

        return confidence

    def _determine_decision_reason(
        self,
        selected_transport: TransportCandidate,
        all_candidates: list[TransportCandidate],
    ) -> str:
        """Determine reason for transport selection."""
        if selected_transport.transport_type == "scion":
            if len([c for c in all_candidates if c.transport_type == "scion"]) > 1:
                return "scion_best_path"
            else:
                return "scion_only_available"

        scion_candidates = [c for c in all_candidates if c.transport_type == "scion"]
        if scion_candidates:
            return "non_scion_preferred"
        else:
            return "scion_unavailable"

    def _create_fallback_transport(self) -> TransportCandidate:
        """Create fallback transport when no others are available."""
        return TransportCandidate(
            transport_type="offline",
            endpoint="local",
            priority=TransportPriority.FALLBACK,
            estimated_latency_ms=100.0,
            reliability_score=0.7,
            cost_factor=0.0,
            metadata={"fallback": True},
        )

    async def _attempt_transport_send(self, message: Message, destination: str, transport: TransportCandidate) -> bool:
        """Attempt to send message via specified transport."""
        try:
            if transport.transport_type == "scion" and self.scion_transport:
                return await self.scion_transport.send_message(message, destination)

            # Handle other transport types through transport manager
            return await self.transport_manager.send_message_via_transport(
                message, destination, transport.transport_type, transport.metadata
            )

        except Exception as e:
            logger.error(f"Failed to send via {transport.transport_type} to {destination}: {e}")
            return False

    def _record_transport_success(self, transport: TransportCandidate) -> None:
        """Record successful transport usage."""
        transport_type = transport.transport_type

        if transport_type not in self._transport_stats:
            self._transport_stats[transport_type] = {
                "attempts": 0,
                "successes": 0,
                "success_rate": 0.0,
                "total_latency_ms": 0.0,
                "recent_avg_latency_ms": 0.0,
            }

        stats = self._transport_stats[transport_type]
        stats["attempts"] += 1
        stats["successes"] += 1
        stats["success_rate"] = stats["successes"] / stats["attempts"]

        # Update latency tracking
        latency = transport.estimated_latency_ms
        stats["total_latency_ms"] += latency
        stats["recent_avg_latency_ms"] = 0.8 * stats.get("recent_avg_latency_ms", latency) + 0.2 * latency

    def _record_transport_failure(self, transport: TransportCandidate) -> None:
        """Record failed transport attempt."""
        transport_type = transport.transport_type

        if transport_type not in self._transport_stats:
            self._transport_stats[transport_type] = {
                "attempts": 0,
                "successes": 0,
                "success_rate": 0.0,
                "total_latency_ms": 0.0,
                "recent_avg_latency_ms": 0.0,
            }

        stats = self._transport_stats[transport_type]
        stats["attempts"] += 1
        stats["success_rate"] = stats["successes"] / stats["attempts"]

    def _record_decision(self, decision: RoutingDecision) -> None:
        """Record routing decision for analysis."""
        self._decision_history.append(decision)

        # Maintain history size limit
        if len(self._decision_history) > self._max_history:
            self._decision_history.pop(0)

    async def _path_discovery_worker(self) -> None:
        """Background worker for periodic path discovery."""
        logger.info("Starting path discovery worker")

        discovery_interval = self.config["path_discovery_interval"]

        while self._is_running:
            try:
                await asyncio.sleep(discovery_interval)

                if not self._is_running:
                    break

                # Refresh path cache for active destinations
                destinations_to_refresh = list(self._path_cache.keys())

                for destination in destinations_to_refresh:
                    if not self._is_running:
                        break

                    try:
                        await self.get_scion_paths(destination, force_refresh=True)
                        await asyncio.sleep(1.0)  # Rate limiting
                    except Exception as e:
                        logger.warning(f"Failed to refresh paths for {destination}: {e}")

                logger.debug(f"Refreshed paths for {len(destinations_to_refresh)} destinations")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in path discovery worker: {e}")
                await asyncio.sleep(10.0)  # Back off on errors

        logger.info("Path discovery worker stopped")


# Utility functions for integration


async def create_scion_navigator(
    scion_config: GatewayConfig, transport_manager: TransportManager, **kwargs
) -> SCIONAwareNavigator:
    """Create and start SCION-aware Navigator."""
    navigator = SCIONAwareNavigator(scion_config, transport_manager, **kwargs)
    await navigator.start()
    return navigator
