"""Network Metrics - RTT, Jitter, and Loss Measurement

Provides live measurement of network performance metrics to drive adaptive decisions:
- RTT estimation using EWMA (Exponentially Weighted Moving Average)
- Jitter calculation (p95-p50 difference)
- Packet loss rate tracking
- Control ping system for proactive measurement
- Integration with transport send/recv hooks

Used by Navigator for intelligent path selection and adaptive chunking policies.
"""

import asyncio
import logging
import statistics
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class MeasurementType(Enum):
    """Types of network measurements"""

    CONTROL_PING = "control_ping"  # Active probes
    MESSAGE_RTT = "message_rtt"  # Piggyback on real messages
    ACK_RTT = "ack_rtt"  # Application-level ACKs


@dataclass
class NetworkSample:
    """Single network measurement sample"""

    timestamp: float
    peer_id: str
    measurement_type: MeasurementType
    rtt_ms: float
    success: bool = True
    payload_size: int = 0
    sequence_number: int = 0


@dataclass
class PeerMetrics:
    """Network metrics for a specific peer"""

    peer_id: str

    # RTT tracking
    rtt_samples: deque = field(default_factory=lambda: deque(maxlen=100))
    rtt_ewma_ms: float = 0.0
    rtt_ewma_alpha: float = 0.125  # Standard TCP alpha

    # Jitter tracking (deviation from mean RTT)
    jitter_samples: deque = field(default_factory=lambda: deque(maxlen=50))
    jitter_ms: float = 0.0

    # Loss tracking
    packets_sent: int = 0
    packets_acked: int = 0
    loss_rate: float = 0.0

    # Quality indicators
    last_measurement_time: float = 0.0
    consecutive_failures: int = 0
    quality_score: float = 1.0  # 0.0-1.0

    # Adaptive parameters
    optimal_chunk_size: int = 4096  # Bytes
    recommended_protocol: str = "htx"  # htx, htxquic, bitchat

    def update_rtt(self, rtt_ms: float) -> None:
        """Update RTT using EWMA"""
        if self.rtt_ewma_ms == 0.0:
            # First sample
            self.rtt_ewma_ms = rtt_ms
        else:
            # EWMA: new_avg = alpha * new_sample + (1 - alpha) * old_avg
            self.rtt_ewma_ms = self.rtt_ewma_alpha * rtt_ms + (1 - self.rtt_ewma_alpha) * self.rtt_ewma_ms

        # Track samples for jitter calculation
        self.rtt_samples.append(rtt_ms)
        self.last_measurement_time = time.time()

        # Calculate jitter (deviation from EWMA)
        jitter = abs(rtt_ms - self.rtt_ewma_ms)
        self.jitter_samples.append(jitter)

        # Update jitter metric (p95 - p50 of jitter samples)
        if len(self.jitter_samples) >= 10:
            jitter_values = list(self.jitter_samples)
            p50 = statistics.median(jitter_values)
            p95 = statistics.quantiles(jitter_values, n=20)[18]  # 95th percentile
            self.jitter_ms = p95 - p50

        # Reset consecutive failures on successful measurement
        self.consecutive_failures = 0

        logger.debug(
            f"Peer {self.peer_id}: RTT={rtt_ms:.1f}ms, EWMA={self.rtt_ewma_ms:.1f}ms, jitter={self.jitter_ms:.1f}ms"
        )

    def update_loss(self, success: bool) -> None:
        """Update packet loss statistics"""
        self.packets_sent += 1
        if success:
            self.packets_acked += 1
        else:
            self.consecutive_failures += 1

        # Calculate loss rate
        if self.packets_sent > 0:
            self.loss_rate = 1.0 - (self.packets_acked / self.packets_sent)

        # Update quality score (combines RTT, jitter, and loss)
        self._update_quality_score()

    def _update_quality_score(self) -> None:
        """Calculate overall quality score (0.0-1.0)"""
        # RTT factor (penalty for high RTT)
        rtt_factor = max(0.0, 1.0 - (self.rtt_ewma_ms / 2000.0))  # 2s = 0 score

        # Jitter factor (penalty for high jitter)
        jitter_factor = max(0.0, 1.0 - (self.jitter_ms / 500.0))  # 500ms jitter = 0 score

        # Loss factor (penalty for packet loss)
        loss_factor = max(0.0, 1.0 - (self.loss_rate * 2.0))  # 50% loss = 0 score

        # Consecutive failures penalty
        failure_penalty = max(0.0, 1.0 - (self.consecutive_failures * 0.2))

        # Combined score
        self.quality_score = rtt_factor * jitter_factor * loss_factor * failure_penalty

        # Update adaptive parameters based on quality
        self._update_adaptive_params()

    def _update_adaptive_params(self) -> None:
        """Update adaptive parameters based on current metrics"""
        # Chunk size adaptation
        if self.jitter_ms > 200:
            # High jitter - use smaller chunks
            self.optimal_chunk_size = 1024
        elif self.jitter_ms < 50 and self.rtt_ewma_ms < 100:
            # Low jitter, low RTT - can use larger chunks
            self.optimal_chunk_size = 8192
        else:
            # Default
            self.optimal_chunk_size = 4096

        # Protocol recommendation
        if self.rtt_ewma_ms < 50 and self.loss_rate < 0.01:
            # Low latency, low loss - QUIC is good
            self.recommended_protocol = "htxquic"
        elif self.rtt_ewma_ms > 500 or self.loss_rate > 0.1:
            # High latency or loss - fall back to reliable transport
            self.recommended_protocol = "bitchat"
        else:
            # Standard conditions - use TLS
            self.recommended_protocol = "htx"

    def is_stale(self, max_age_seconds: float = 30.0) -> bool:
        """Check if metrics are stale"""
        return (time.time() - self.last_measurement_time) > max_age_seconds

    def to_dict(self) -> dict[str, Any]:
        """Export metrics as dictionary"""
        return {
            "peer_id": self.peer_id,
            "rtt_ewma_ms": round(self.rtt_ewma_ms, 2),
            "jitter_ms": round(self.jitter_ms, 2),
            "loss_rate": round(self.loss_rate, 4),
            "quality_score": round(self.quality_score, 3),
            "packets_sent": self.packets_sent,
            "packets_acked": self.packets_acked,
            "consecutive_failures": self.consecutive_failures,
            "optimal_chunk_size": self.optimal_chunk_size,
            "recommended_protocol": self.recommended_protocol,
            "last_measurement_age": round(time.time() - self.last_measurement_time, 1),
        }


class NetworkMetricsCollector:
    """Collects and tracks network performance metrics"""

    def __init__(self):
        self.peer_metrics: dict[str, PeerMetrics] = {}
        self.pending_measurements: dict[str, NetworkSample] = {}  # seq_id -> sample
        self.measurement_callbacks: dict[str, list[Callable]] = defaultdict(list)

        # Control ping state
        self.ping_sequence = 0
        self.ping_tasks: dict[str, asyncio.Task] = {}  # peer_id -> task
        self.ping_interval_seconds = 5.0
        self.ping_timeout_seconds = 2.0

        # Statistics
        self.total_measurements = 0
        self.measurement_errors = 0

        logger.info("NetworkMetricsCollector initialized")

    def get_peer_metrics(self, peer_id: str) -> PeerMetrics:
        """Get metrics for a specific peer, create if not exists"""
        if peer_id not in self.peer_metrics:
            self.peer_metrics[peer_id] = PeerMetrics(peer_id=peer_id)
        return self.peer_metrics[peer_id]

    def record_message_sent(self, peer_id: str, message_id: str, payload_size: int = 0) -> str:
        """Record that a message was sent, return sequence ID for RTT tracking"""
        sequence_id = f"{peer_id}_{self.ping_sequence}_{message_id}"
        self.ping_sequence += 1

        sample = NetworkSample(
            timestamp=time.time(),
            peer_id=peer_id,
            measurement_type=MeasurementType.MESSAGE_RTT,
            rtt_ms=0.0,  # Will be filled on ACK
            payload_size=payload_size,
            sequence_number=self.ping_sequence,
        )

        self.pending_measurements[sequence_id] = sample

        # Clean up old pending measurements (timeout after 10s)
        self._cleanup_pending_measurements()

        return sequence_id

    def record_message_acked(self, sequence_id: str, success: bool = True) -> float | None:
        """Record that a message was ACKed, return RTT in ms"""
        if sequence_id not in self.pending_measurements:
            logger.debug(f"No pending measurement for sequence {sequence_id}")
            return None

        sample = self.pending_measurements.pop(sequence_id)
        current_time = time.time()
        rtt_ms = (current_time - sample.timestamp) * 1000.0

        # Update peer metrics
        peer_metrics = self.get_peer_metrics(sample.peer_id)
        peer_metrics.update_rtt(rtt_ms)
        peer_metrics.update_loss(success)

        self.total_measurements += 1
        if not success:
            self.measurement_errors += 1

        # Notify callbacks
        self._notify_callbacks(sample.peer_id, rtt_ms, success)

        logger.debug(f"Recorded RTT: {sample.peer_id} -> {rtt_ms:.1f}ms (success={success})")
        return rtt_ms

    async def send_control_ping(self, peer_id: str, ping_sender: Callable[[str, bytes], bool]) -> bool:
        """Send a control ping to measure RTT"""
        sequence_id = f"ping_{peer_id}_{self.ping_sequence}"
        self.ping_sequence += 1

        # Create ping payload with timestamp and sequence
        ping_data = {
            "type": "control_ping",
            "sequence_id": sequence_id,
            "timestamp": time.time(),
            "sender_id": "metrics_collector",
        }

        import json

        ping_payload = json.dumps(ping_data).encode()

        # Record pending measurement
        sample = NetworkSample(
            timestamp=time.time(),
            peer_id=peer_id,
            measurement_type=MeasurementType.CONTROL_PING,
            rtt_ms=0.0,
            payload_size=len(ping_payload),
            sequence_number=self.ping_sequence,
        )

        self.pending_measurements[sequence_id] = sample

        # Send ping
        try:
            success = await ping_sender(peer_id, ping_payload)
            if not success:
                # Record failed send
                peer_metrics = self.get_peer_metrics(peer_id)
                peer_metrics.update_loss(False)
                self.pending_measurements.pop(sequence_id, None)
                return False

            logger.debug(f"Sent control ping to {peer_id}: {sequence_id}")
            return True

        except Exception as e:
            logger.warning(f"Failed to send control ping to {peer_id}: {e}")
            self.pending_measurements.pop(sequence_id, None)
            return False

    def handle_control_pong(self, peer_id: str, pong_data: dict[str, Any]) -> float | None:
        """Handle received control pong, return RTT"""
        sequence_id = pong_data.get("sequence_id")
        if not sequence_id or sequence_id not in self.pending_measurements:
            return None

        # Calculate RTT and update metrics
        return self.record_message_acked(sequence_id, success=True)

    async def start_control_ping_loop(self, peer_id: str, ping_sender: Callable[[str, bytes], bool]) -> None:
        """Start background control ping loop for a peer"""
        if peer_id in self.ping_tasks:
            return  # Already running

        async def ping_loop():
            while peer_id in self.ping_tasks:
                try:
                    await self.send_control_ping(peer_id, ping_sender)
                    await asyncio.sleep(self.ping_interval_seconds)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"Control ping loop error for {peer_id}: {e}")
                    await asyncio.sleep(self.ping_interval_seconds)

        self.ping_tasks[peer_id] = asyncio.create_task(ping_loop())
        logger.info(f"Started control ping loop for {peer_id}")

    def stop_control_ping_loop(self, peer_id: str) -> None:
        """Stop control ping loop for a peer"""
        if peer_id in self.ping_tasks:
            self.ping_tasks[peer_id].cancel()
            del self.ping_tasks[peer_id]
            logger.info(f"Stopped control ping loop for {peer_id}")

    def stop_all_ping_loops(self) -> None:
        """Stop all control ping loops"""
        for peer_id in list(self.ping_tasks.keys()):
            self.stop_control_ping_loop(peer_id)

    def register_callback(self, peer_id: str, callback: Callable[[str, float, bool], None]) -> None:
        """Register callback for measurement updates"""
        self.measurement_callbacks[peer_id].append(callback)

    def _notify_callbacks(self, peer_id: str, rtt_ms: float, success: bool) -> None:
        """Notify registered callbacks of measurement update"""
        for callback in self.measurement_callbacks.get(peer_id, []):
            try:
                callback(peer_id, rtt_ms, success)
            except Exception as e:
                logger.warning(f"Callback error for {peer_id}: {e}")

    def _cleanup_pending_measurements(self, max_age: float = 10.0) -> None:
        """Clean up old pending measurements"""
        current_time = time.time()
        expired_ids = [
            seq_id
            for seq_id, sample in self.pending_measurements.items()
            if (current_time - sample.timestamp) > max_age
        ]

        for seq_id in expired_ids:
            sample = self.pending_measurements.pop(seq_id)
            # Record as timeout/loss
            peer_metrics = self.get_peer_metrics(sample.peer_id)
            peer_metrics.update_loss(False)

        if expired_ids:
            logger.debug(f"Cleaned up {len(expired_ids)} expired measurements")

    def get_adaptive_chunk_size(self, peer_id: str, default_size: int = 4096) -> int:
        """Get adaptive chunk size for a peer based on network conditions"""
        if peer_id not in self.peer_metrics:
            return default_size

        metrics = self.peer_metrics[peer_id]
        if metrics.is_stale():
            return default_size

        return metrics.optimal_chunk_size

    def get_recommended_protocol(self, peer_id: str, default_protocol: str = "htx") -> str:
        """Get recommended protocol for a peer based on network conditions"""
        if peer_id not in self.peer_metrics:
            return default_protocol

        metrics = self.peer_metrics[peer_id]
        if metrics.is_stale():
            return default_protocol

        return metrics.recommended_protocol

    def should_switch_path(self, peer_id: str, rtt_threshold_ms: float = 1000, loss_threshold: float = 0.2) -> bool:
        """Determine if path should be switched based on metrics"""
        if peer_id not in self.peer_metrics:
            return False

        metrics = self.peer_metrics[peer_id]

        # Switch if RTT too high, loss too high, or quality too low
        return (
            metrics.rtt_ewma_ms > rtt_threshold_ms
            or metrics.loss_rate > loss_threshold
            or metrics.quality_score < 0.3
            or metrics.consecutive_failures > 3
        )

    def export_all_metrics(self) -> dict[str, Any]:
        """Export all metrics for monitoring/debugging"""
        return {
            "collector_stats": {
                "total_measurements": self.total_measurements,
                "measurement_errors": self.measurement_errors,
                "error_rate": self.measurement_errors / max(1, self.total_measurements),
                "active_ping_loops": len(self.ping_tasks),
                "pending_measurements": len(self.pending_measurements),
            },
            "peer_metrics": {peer_id: metrics.to_dict() for peer_id, metrics in self.peer_metrics.items()},
        }


# Global metrics collector instance
_metrics_collector: NetworkMetricsCollector | None = None


def get_metrics_collector() -> NetworkMetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = NetworkMetricsCollector()
    return _metrics_collector


def record_send_timestamp(peer_id: str, message_id: str, payload_size: int = 0) -> str:
    """Convenience function to record message send"""
    collector = get_metrics_collector()
    return collector.record_message_sent(peer_id, message_id, payload_size)


def record_ack_timestamp(sequence_id: str, success: bool = True) -> float | None:
    """Convenience function to record message ACK"""
    collector = get_metrics_collector()
    return collector.record_message_acked(sequence_id, success)


def get_adaptive_chunk_size(peer_id: str, default_size: int = 4096) -> int:
    """Convenience function to get adaptive chunk size"""
    collector = get_metrics_collector()
    return collector.get_adaptive_chunk_size(peer_id, default_size)


def get_recommended_protocol(peer_id: str, default_protocol: str = "htx") -> str:
    """Convenience function to get recommended protocol"""
    collector = get_metrics_collector()
    return collector.get_recommended_protocol(peer_id, default_protocol)
