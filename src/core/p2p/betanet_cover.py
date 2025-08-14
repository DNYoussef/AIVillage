"""Betanet Cover Traffic Generator

Provides steady cover/padding traffic to make Betanet indistinguishable from normal web activity.
Implements configurable constant-rate or randomized dummy traffic patterns.

Features:
- Background cover traffic at configurable rates
- Adaptive padding sizes to mimic web traffic bursts
- Budget controls to prevent resource exhaustion
- Traffic shaping to maintain user QoS
- Integration with Betanet transport lifecycle
"""

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

logger = logging.getLogger(__name__)


class CoverTrafficMode(Enum):
    """Cover traffic generation modes"""

    OFF = "off"  # No cover traffic
    CONSTANT_RATE = "constant"  # Steady rate (e.g., 1 pps)
    RANDOMIZED = "randomized"  # Poisson-like distribution
    WEB_BURST = "web_burst"  # Mimic web browsing patterns
    ADAPTIVE = "adaptive"  # Adapt to real traffic patterns


@dataclass
class CoverTrafficConfig:
    """Configuration for cover traffic generation"""

    mode: CoverTrafficMode = CoverTrafficMode.OFF

    # Rate controls
    base_rate_pps: float = 0.5  # Base packets per second
    burst_rate_pps: float = 2.0  # Burst rate during active periods
    burst_duration_sec: float = 5.0  # How long bursts last
    quiet_duration_sec: float = 15.0  # Time between bursts

    # Padding controls
    min_padding_bytes: int = 64  # Minimum dummy message size
    max_padding_bytes: int = 1024  # Maximum dummy message size
    web_size_distribution: list[int] = field(default_factory=lambda: [128, 256, 512, 768, 1024])

    # Budget controls
    max_bandwidth_bps: int = 10000  # Max bandwidth for cover traffic (10KB/s)
    max_daily_mb: float = 100.0  # Daily budget in MB
    cpu_threshold: float = 0.8  # Pause if CPU usage > 80%

    # Timing controls
    jitter_ms: float = 100.0  # Random jitter in send times
    respect_user_traffic: bool = True  # Reduce cover when user active

    @classmethod
    def from_env(cls) -> "CoverTrafficConfig":
        """Create config from environment variables"""
        return cls(
            mode=CoverTrafficMode(os.getenv("BETANET_COVER_MODE", "off")),
            base_rate_pps=float(os.getenv("BETANET_COVER_RATE", "0.5")),
            max_bandwidth_bps=int(os.getenv("BETANET_COVER_BANDWIDTH", "10000")),
            max_daily_mb=float(os.getenv("BETANET_COVER_DAILY_MB", "100.0")),
        )


@dataclass
class CoverTrafficStats:
    """Statistics for cover traffic generation"""

    packets_sent: int = 0
    bytes_sent: int = 0
    budget_used_mb: float = 0.0
    active_time_sec: float = 0.0
    paused_time_sec: float = 0.0
    user_traffic_detected: int = 0

    # Rate tracking
    current_rate_pps: float = 0.0
    avg_packet_size: float = 0.0
    last_packet_time: float = 0.0

    def to_dict(self) -> dict:
        """Export stats as dictionary"""
        return {
            "packets_sent": self.packets_sent,
            "bytes_sent": self.bytes_sent,
            "budget_used_mb": self.budget_used_mb,
            "active_time_sec": self.active_time_sec,
            "paused_time_sec": self.paused_time_sec,
            "current_rate_pps": self.current_rate_pps,
            "avg_packet_size": self.avg_packet_size,
            "efficiency": self.packets_sent / max(1, self.packets_sent + self.user_traffic_detected),
        }


class CoverTrafficSender(Protocol):
    """Protocol for sending cover traffic"""

    async def send_cover_message(self, payload: bytes, recipient: str | None = None) -> bool:
        """Send a cover traffic message"""
        ...

    def get_active_peers(self) -> list[str]:
        """Get list of active peer IDs for cover traffic"""
        ...


class BetanetCoverTraffic:
    """Cover traffic generator for Betanet"""

    def __init__(self, config: CoverTrafficConfig, sender: CoverTrafficSender):
        self.config = config
        self.sender = sender
        self.stats = CoverTrafficStats()

        # State
        self.is_running = False
        self.is_paused = False
        self.cover_task: asyncio.Task | None = None
        self.start_time = 0.0

        # Traffic patterns
        self.web_patterns = {
            "page_load": [512, 1024, 256, 768, 128],  # Page load pattern
            "api_calls": [128, 256, 128, 64],  # API request pattern
            "file_download": [1024] * 5,  # Download pattern
            "chat_messages": [64, 128, 96],  # Chat pattern
        }

        # Adaptive state
        self.recent_user_traffic = []
        self.current_pattern = "page_load"
        self.burst_start_time = 0.0

        logger.info(f"BetanetCoverTraffic initialized: mode={config.mode.value}")

    async def start(self) -> None:
        """Start cover traffic generation"""
        if self.config.mode == CoverTrafficMode.OFF:
            logger.info("Cover traffic disabled")
            return

        if self.is_running:
            logger.warning("Cover traffic already running")
            return

        logger.info(f"Starting cover traffic: {self.config.mode.value} mode")
        self.is_running = True
        self.start_time = time.time()

        # Start background cover traffic task
        self.cover_task = asyncio.create_task(self._cover_traffic_loop())

    async def stop(self) -> None:
        """Stop cover traffic generation"""
        logger.info("Stopping cover traffic")
        self.is_running = False

        if self.cover_task:
            self.cover_task.cancel()
            try:
                await self.cover_task
            except asyncio.CancelledError:
                pass
            self.cover_task = None

        # Log final stats
        time.time() - self.start_time
        logger.info(f"Cover traffic stopped. Stats: {self.get_stats_summary()}")

    async def _cover_traffic_loop(self) -> None:
        """Main cover traffic generation loop"""
        while self.is_running:
            try:
                if self._should_pause():
                    await self._pause_briefly()
                    continue

                # Generate and send cover message
                await self._send_cover_message()

                # Calculate next send time
                delay = self._calculate_next_delay()
                await asyncio.sleep(delay)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Cover traffic error: {e}")
                await asyncio.sleep(1.0)  # Brief pause on error

    def _should_pause(self) -> bool:
        """Check if cover traffic should be paused"""
        # Check budget
        if self.stats.budget_used_mb >= self.config.max_daily_mb:
            return True

        # Check bandwidth
        current_bps = self.stats.current_rate_pps * self.stats.avg_packet_size
        if current_bps >= self.config.max_bandwidth_bps:
            return True

        # Check CPU usage if psutil is available
        try:
            import psutil

            cpu_usage = psutil.cpu_percent(interval=0.1)
            if cpu_usage >= self.config.cpu_threshold:
                return True
        except ImportError:
            # psutil not available - skip CPU check
            pass

        # Check for user traffic conflicts
        if self.config.respect_user_traffic and self._detect_user_traffic():
            self.stats.user_traffic_detected += 1
            return True

        return False

    def _detect_user_traffic(self) -> bool:
        """Detect if user traffic is active (heuristic)"""
        # Simple heuristic: if we've seen recent real traffic, pause briefly
        current_time = time.time()

        # Remove old entries
        self.recent_user_traffic = [t for t in self.recent_user_traffic if current_time - t < 5.0]  # 5 second window

        # If we've seen user traffic recently, pause
        return len(self.recent_user_traffic) > 0

    def notify_user_traffic(self) -> None:
        """Notify that user traffic was detected"""
        self.recent_user_traffic.append(time.time())

    async def _pause_briefly(self) -> None:
        """Pause cover traffic briefly"""
        self.is_paused = True
        pause_duration = random.uniform(1.0, 3.0)  # Random pause
        await asyncio.sleep(pause_duration)
        self.stats.paused_time_sec += pause_duration
        self.is_paused = False

    async def _send_cover_message(self) -> None:
        """Generate and send a single cover message"""
        # Choose message size based on pattern
        size = self._choose_message_size()

        # Generate dummy payload
        payload = self._generate_dummy_payload(size)

        # Choose recipient (random peer or broadcast)
        recipient = self._choose_cover_recipient()

        # Send the message
        try:
            success = await self.sender.send_cover_message(payload, recipient)

            if success:
                self.stats.packets_sent += 1
                self.stats.bytes_sent += len(payload)
                self.stats.budget_used_mb += len(payload) / (1024 * 1024)
                self.stats.last_packet_time = time.time()

                # Update average packet size
                if self.stats.packets_sent > 0:
                    self.stats.avg_packet_size = self.stats.bytes_sent / self.stats.packets_sent

                logger.debug(f"Sent cover message: {len(payload)} bytes to {recipient}")

        except Exception as e:
            logger.warning(f"Failed to send cover message: {e}")

    def _choose_message_size(self) -> int:
        """Choose message size based on current pattern"""
        if self.config.mode == CoverTrafficMode.CONSTANT_RATE:
            return random.randint(self.config.min_padding_bytes, self.config.max_padding_bytes)

        elif self.config.mode == CoverTrafficMode.WEB_BURST:
            # Use realistic web traffic sizes
            sizes = self.config.web_size_distribution
            return random.choice(sizes)

        elif self.config.mode == CoverTrafficMode.ADAPTIVE:
            # Use current pattern
            pattern_sizes = self.web_patterns.get(self.current_pattern, [256])
            return random.choice(pattern_sizes)

        else:  # RANDOMIZED
            # Log-normal distribution (more realistic)
            (self.config.min_padding_bytes + self.config.max_padding_bytes) // 2
            size = max(
                self.config.min_padding_bytes,
                min(self.config.max_padding_bytes, int(random.lognormvariate(6.0, 0.5))),
            )  # ~400 bytes mean
            return size

    def _generate_dummy_payload(self, size: int) -> bytes:
        """Generate dummy payload that looks realistic"""
        # Create payload that looks like compressed JSON/HTTP data

        # Start with some realistic-looking structure
        base_structure = {
            "type": random.choice(["data", "request", "response", "notification"]),
            "timestamp": int(time.time()),
            "id": f"req_{random.randint(1000000, 9999999)}",
            "version": "1.0",
        }

        # Add random fields to reach desired size
        base_json = json.dumps(base_structure)
        padding_needed = max(0, size - len(base_json) - 50)  # Leave some room

        if padding_needed > 0:
            # Add base64-like padding
            import base64

            random_bytes = os.urandom(padding_needed // 4 * 3)  # Base64 expansion
            base_structure["data"] = base64.b64encode(random_bytes).decode()[:padding_needed]

        payload = json.dumps(base_structure).encode()

        # Pad/truncate to exact size
        if len(payload) < size:
            payload += b" " * (size - len(payload))
        elif len(payload) > size:
            payload = payload[:size]

        return payload

    def _choose_cover_recipient(self) -> str | None:
        """Choose recipient for cover traffic"""
        peers = self.sender.get_active_peers()

        if not peers:
            return None  # No peers available

        # For cover traffic, we want to create realistic patterns
        if self.config.mode == CoverTrafficMode.WEB_BURST:
            # Web traffic often goes to the same server multiple times
            return random.choice(peers[:3])  # Prefer first few peers
        else:
            # Random peer
            return random.choice(peers)

    def _calculate_next_delay(self) -> float:
        """Calculate delay until next cover message"""
        if self.config.mode == CoverTrafficMode.CONSTANT_RATE:
            base_delay = 1.0 / self.config.base_rate_pps

        elif self.config.mode == CoverTrafficMode.WEB_BURST:
            # Burst pattern: alternate between bursts and quiet periods
            current_time = time.time()

            if self.burst_start_time == 0 or (current_time - self.burst_start_time) > self.config.burst_duration_sec:
                # Start new burst or quiet period
                if random.random() < 0.3:  # 30% chance of burst
                    self.burst_start_time = current_time
                    base_delay = 1.0 / self.config.burst_rate_pps
                else:
                    base_delay = self.config.quiet_duration_sec / 2  # Quiet period
            else:
                # Continue current burst
                base_delay = 1.0 / self.config.burst_rate_pps

        elif self.config.mode == CoverTrafficMode.RANDOMIZED:
            # Poisson-like distribution
            base_delay = random.expovariate(self.config.base_rate_pps)

        else:  # ADAPTIVE
            # Adapt to recent user patterns
            base_delay = 1.0 / self.config.base_rate_pps

        # Add jitter
        jitter = random.uniform(-self.config.jitter_ms / 1000, self.config.jitter_ms / 1000)
        delay = max(0.1, base_delay + jitter)  # Minimum 100ms delay

        # Update current rate
        self.stats.current_rate_pps = 1.0 / delay

        return delay

    def get_stats_summary(self) -> str:
        """Get human-readable stats summary"""
        runtime = time.time() - self.start_time if self.start_time > 0 else 0
        return (
            f"packets={self.stats.packets_sent}, "
            f"bytes={self.stats.bytes_sent}, "
            f"rate={self.stats.current_rate_pps:.2f}pps, "
            f"budget={self.stats.budget_used_mb:.1f}MB, "
            f"runtime={runtime:.1f}s"
        )

    def export_metrics(self) -> dict:
        """Export detailed metrics"""
        runtime = time.time() - self.start_time if self.start_time > 0 else 0

        metrics = self.stats.to_dict()
        metrics.update(
            {
                "config_mode": self.config.mode.value,
                "config_base_rate": self.config.base_rate_pps,
                "config_max_bandwidth": self.config.max_bandwidth_bps,
                "config_daily_budget": self.config.max_daily_mb,
                "runtime_seconds": runtime,
                "is_running": self.is_running,
                "is_paused": self.is_paused,
            }
        )

        return metrics
