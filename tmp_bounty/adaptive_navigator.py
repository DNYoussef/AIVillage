"""Adaptive Navigator Integration - RTT/Jitter Driven Path Selection

Integrates NetworkMetricsCollector with Navigator agent for real-time adaptive routing
based on measured RTT, jitter, and loss metrics. Provides sub-500ms path switching
when network conditions change.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class PathProtocol(Enum):
    """Available transport protocols"""

    BITCHAT = "bitchat"
    BETANET = "betanet"
    HTX = "htx"
    HTXQUIC = "htxquic"
    STORE_FORWARD = "store_forward"


@dataclass
class AdaptiveNetworkConditions:
    """Network conditions enhanced with real metrics"""

    # Traditional fields
    estimated_latency_ms: float = 200.0
    estimated_bandwidth_kbps: float = 1000.0
    reliability_score: float = 0.9

    # Live metrics from NetworkMetricsCollector
    measured_rtt_ms: float | None = None
    measured_jitter_ms: float | None = None
    measured_loss_rate: float | None = None
    quality_score: float | None = None

    # Adaptive parameters
    optimal_chunk_size: int = 4096
    recommended_protocol: str = "htx"

    # Measurement metadata
    measurement_age_seconds: float = 0.0
    measurement_count: int = 0

    def is_measurements_fresh(self, max_age_seconds: float = 10.0) -> bool:
        """Check if measurements are recent enough for decisions"""
        return (
            self.measurement_age_seconds <= max_age_seconds
            and self.measurement_count > 0
        )

    def get_effective_latency(self) -> float:
        """Get effective latency (measured if available, estimated otherwise)"""
        if self.measured_rtt_ms is not None and self.is_measurements_fresh():
            return self.measured_rtt_ms
        return self.estimated_latency_ms

    def get_effective_reliability(self) -> float:
        """Get effective reliability based on measurements"""
        if self.quality_score is not None and self.is_measurements_fresh():
            return self.quality_score
        if self.measured_loss_rate is not None and self.is_measurements_fresh():
            return max(0.0, 1.0 - (self.measured_loss_rate * 2.0))  # Loss penalty
        return self.reliability_score

    def needs_path_switch(
        self,
        rtt_threshold_ms: float = 1000,
        loss_threshold: float = 0.2,
        quality_threshold: float = 0.3,
    ) -> bool:
        """Determine if path should be switched based on conditions"""
        if not self.is_measurements_fresh():
            return False

        # Switch if RTT too high, loss too high, or quality too low
        return (
            (
                self.measured_rtt_ms is not None
                and self.measured_rtt_ms > rtt_threshold_ms
            )
            or (
                self.measured_loss_rate is not None
                and self.measured_loss_rate > loss_threshold
            )
            or (
                self.quality_score is not None
                and self.quality_score < quality_threshold
            )
        )


@dataclass
class MessageContext:
    """Enhanced message context for path selection"""

    sender: str = ""
    recipient: str = ""
    payload_size: int = 0
    priority: int = 5
    privacy_required: bool = False
    deadline: float | None = None
    retry_count: int = 0

    # Adaptive context
    preferred_protocol: str | None = None
    avoid_protocols: list[str] = field(default_factory=list)
    max_chunk_size: int | None = None


class AdaptiveNavigator:
    """Navigator with live RTT/jitter metrics integration"""

    def __init__(self, metrics_collector=None):
        self.metrics_collector = metrics_collector
        self.protocol_preferences = {
            PathProtocol.HTX: {"latency_max": 200, "reliability_min": 0.8},
            PathProtocol.HTXQUIC: {"latency_max": 100, "reliability_min": 0.9},
            PathProtocol.BETANET: {"latency_max": 1000, "reliability_min": 0.7},
            PathProtocol.BITCHAT: {"latency_max": 2000, "reliability_min": 0.5},
            PathProtocol.STORE_FORWARD: {
                "latency_max": float("inf"),
                "reliability_min": 0.0,
            },
        }

        # Path switching state
        self.last_switch_time = {}  # peer_id -> timestamp
        self.switch_cooldown_seconds = 2.0  # Avoid flapping

        logger.info("AdaptiveNavigator initialized with live metrics integration")

    def get_network_conditions(self, peer_id: str) -> AdaptiveNetworkConditions:
        """Get current network conditions for a peer"""
        conditions = AdaptiveNetworkConditions()

        if self.metrics_collector and peer_id in self.metrics_collector.peer_metrics:
            peer_metrics = self.metrics_collector.peer_metrics[peer_id]

            # Populate with live measurements
            conditions.measured_rtt_ms = (
                peer_metrics.rtt_ewma_ms if peer_metrics.rtt_ewma_ms > 0 else None
            )
            conditions.measured_jitter_ms = peer_metrics.jitter_ms
            conditions.measured_loss_rate = peer_metrics.loss_rate
            conditions.quality_score = peer_metrics.quality_score
            conditions.optimal_chunk_size = peer_metrics.optimal_chunk_size
            conditions.recommended_protocol = peer_metrics.recommended_protocol
            conditions.measurement_age_seconds = (
                time.time() - peer_metrics.last_measurement_time
            )
            conditions.measurement_count = peer_metrics.packets_sent

            logger.debug(
                f"Live conditions for {peer_id}: RTT={conditions.measured_rtt_ms:.1f}ms, "
                f"jitter={conditions.measured_jitter_ms:.1f}ms, "
                f"loss={conditions.measured_loss_rate:.3f}, "
                f"quality={conditions.quality_score:.3f}"
            )
        else:
            logger.debug(f"No live metrics for {peer_id}, using estimated conditions")

        return conditions

    async def select_optimal_protocol(
        self, peer_id: str, context: MessageContext, available_protocols: list[str]
    ) -> tuple[str, dict[str, Any]]:
        """Select optimal protocol based on live network conditions"""
        start_time = time.time()
        conditions = self.get_network_conditions(peer_id)

        # Check if we need to switch paths due to poor conditions
        force_switch = conditions.needs_path_switch()
        if force_switch:
            logger.info(
                f"Path switch triggered for {peer_id} due to poor conditions: "
                f"RTT={conditions.measured_rtt_ms}ms, loss={conditions.measured_loss_rate}"
            )

        # Check switch cooldown to avoid flapping
        last_switch = self.last_switch_time.get(peer_id, 0)
        if (
            time.time() - last_switch < self.switch_cooldown_seconds
            and not force_switch
        ):
            # Use recommended protocol from measurements if available
            if conditions.recommended_protocol in available_protocols:
                protocol = conditions.recommended_protocol
            else:
                protocol = available_protocols[0] if available_protocols else "htx"
        else:
            # Select best protocol based on current conditions
            protocol = self._rank_protocols(conditions, available_protocols, context)

            # Update switch time if we're switching
            current_best = getattr(self, f"_last_best_{peer_id}", None)
            if current_best != protocol:
                self.last_switch_time[peer_id] = time.time()
                setattr(self, f"_last_best_{peer_id}", protocol)
                logger.info(
                    f"Protocol switch for {peer_id}: {current_best} → {protocol}"
                )

        # Build metadata
        metadata = {
            "selected_protocol": protocol,
            "measured_rtt_ms": conditions.measured_rtt_ms,
            "measured_jitter_ms": conditions.measured_jitter_ms,
            "measured_loss_rate": conditions.measured_loss_rate,
            "quality_score": conditions.quality_score,
            "optimal_chunk_size": conditions.optimal_chunk_size,
            "decision_time_ms": (time.time() - start_time) * 1000,
            "measurements_fresh": conditions.is_measurements_fresh(),
            "force_switch": force_switch,
        }

        # Ensure decision time is under 500ms target
        decision_time_ms = metadata["decision_time_ms"]
        if decision_time_ms > 500:
            logger.warning(
                f"Path selection took {decision_time_ms:.1f}ms (>500ms target)"
            )
        else:
            logger.debug(f"Path selection completed in {decision_time_ms:.1f}ms")

        return protocol, metadata

    def _rank_protocols(
        self,
        conditions: AdaptiveNetworkConditions,
        available_protocols: list[str],
        context: MessageContext,
    ) -> str:
        """Rank available protocols by suitability for current conditions"""
        if not available_protocols:
            return "htx"  # Default fallback

        effective_latency = conditions.get_effective_latency()
        effective_reliability = conditions.get_effective_reliability()

        # Score each protocol
        protocol_scores = {}

        for protocol_str in available_protocols:
            try:
                protocol = PathProtocol(protocol_str)
            except ValueError:
                continue  # Unknown protocol

            prefs = self.protocol_preferences[protocol]
            score = 1.0

            # Latency penalty
            if effective_latency > prefs["latency_max"]:
                score *= 0.5  # Heavy penalty for exceeding latency threshold
            else:
                score *= (
                    1.0
                    + (prefs["latency_max"] - effective_latency)
                    / prefs["latency_max"]
                    * 0.2
                )

            # Reliability bonus
            if effective_reliability >= prefs["reliability_min"]:
                score *= 1.0 + (effective_reliability - prefs["reliability_min"]) * 0.3
            else:
                score *= 0.7  # Penalty for low reliability

            # Priority boost for measured recommended protocol
            if (
                conditions.recommended_protocol == protocol_str
                and conditions.is_measurements_fresh()
            ):
                score *= 1.5

            # Context-specific adjustments
            if context.privacy_required and protocol in [
                PathProtocol.BETANET,
                PathProtocol.BITCHAT,
            ]:
                score *= 1.3  # Privacy protocols get bonus

            if context.priority >= 8 and protocol == PathProtocol.HTXQUIC:
                score *= 1.2  # QUIC for urgent messages

            # Avoid blacklisted protocols
            if protocol_str in context.avoid_protocols:
                score *= 0.1

            protocol_scores[protocol_str] = score

        # Return highest scoring protocol
        best_protocol = max(protocol_scores.items(), key=lambda x: x[1])[0]

        logger.debug(f"Protocol ranking: {protocol_scores}, selected: {best_protocol}")
        return best_protocol

    def get_adaptive_chunk_size(self, peer_id: str, default_size: int = 4096) -> int:
        """Get adaptive chunk size based on network conditions"""
        if self.metrics_collector:
            return self.metrics_collector.get_adaptive_chunk_size(peer_id, default_size)
        return default_size

    def should_start_control_ping(self, peer_id: str) -> bool:
        """Determine if control ping loop should be started for a peer"""
        if not self.metrics_collector:
            return False

        # Start ping if we don't have recent metrics
        if peer_id not in self.metrics_collector.peer_metrics:
            return True

        peer_metrics = self.metrics_collector.peer_metrics[peer_id]
        return peer_metrics.is_stale(max_age_seconds=15.0)  # Ping if stale >15s

    async def start_control_ping_if_needed(self, peer_id: str, ping_sender) -> None:
        """Start control ping loop if metrics are needed"""
        if self.should_start_control_ping(peer_id) and self.metrics_collector:
            await self.metrics_collector.start_control_ping_loop(peer_id, ping_sender)
            logger.debug(f"Started control ping loop for {peer_id}")

    def export_metrics(self) -> dict[str, Any]:
        """Export navigator metrics for monitoring"""
        return {
            "switch_times": dict(self.last_switch_time),
            "protocol_preferences": {
                k.value: v for k, v in self.protocol_preferences.items()
            },
            "switch_cooldown_seconds": self.switch_cooldown_seconds,
            "metrics_collector_available": self.metrics_collector is not None,
        }


# Convenience functions for integration
def create_adaptive_navigator(metrics_collector=None):
    """Create AdaptiveNavigator with metrics integration"""
    return AdaptiveNavigator(metrics_collector)


async def test_adaptive_selection():
    """Test adaptive protocol selection with simulated metrics"""
    from ..metrics.net_metrics import NetworkMetricsCollector

    # Create test setup
    metrics = NetworkMetricsCollector()
    navigator = create_adaptive_navigator(metrics)

    # Simulate some measurements
    peer_id = "test_peer"

    # Record high RTT scenario
    seq1 = metrics.record_message_sent(peer_id, "msg1", 1024)
    await asyncio.sleep(0.5)  # Simulate 500ms RTT
    rtt1 = metrics.record_message_acked(seq1, success=True)

    # Record another high RTT
    seq2 = metrics.record_message_sent(peer_id, "msg2", 1024)
    await asyncio.sleep(0.7)  # Simulate 700ms RTT
    rtt2 = metrics.record_message_acked(seq2, success=True)

    print(f"Recorded RTTs: {rtt1:.1f}ms, {rtt2:.1f}ms")

    # Test protocol selection
    context = MessageContext(recipient=peer_id, payload_size=2048, priority=7)
    available = ["htx", "htxquic", "betanet", "bitchat"]

    protocol, metadata = await navigator.select_optimal_protocol(
        peer_id, context, available
    )

    print(f"Selected protocol: {protocol}")
    print(f"Decision metadata: {metadata}")
    print(f"Network conditions: {navigator.get_network_conditions(peer_id).__dict__}")

    return (
        protocol == "betanet" or protocol == "bitchat"
    )  # Should prefer robust protocols for high RTT


if __name__ == "__main__":
    # Run test
    asyncio.run(test_adaptive_selection())
