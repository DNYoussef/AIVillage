"""
SCION Gateway integration for Python Navigator and Transport systems.
Provides high-level interface to the Betanet Gateway for SCION packet tunneling.
"""

import asyncio
import json
import logging
import os
import subprocess  # nosec B404 - Controlled subprocess usage for SCION gateway
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from .core.message_types import MessageType
from .core.message_types import UnifiedMessage as Message

logger = logging.getLogger(__name__)


@dataclass
class SCIONPath:
    """SCION path information."""

    path_id: str
    fingerprint: str
    destination: str
    rtt_us: float
    loss_rate: float
    is_healthy: bool
    is_active: bool


@dataclass
class GatewayConfig:
    """SCION Gateway configuration."""

    # Gateway endpoints
    htx_endpoint: str = "https://127.0.0.1:8443"
    metrics_endpoint: str = "http://127.0.0.1:9090/metrics"

    # Sidecar configuration
    sidecar_address: str = "127.0.0.1:8080"
    sidecar_binary_path: str | None = None

    # Request configuration
    request_timeout: float = 30.0
    connection_timeout: float = 10.0
    max_retries: int = 3

    # Performance settings
    max_concurrent_requests: int = 100
    packet_batch_size: int = 10
    receive_poll_interval: float = 0.1

    # TLS configuration
    verify_ssl: bool = False  # Set to True in production with proper certificates
    cert_file: str | None = None
    key_file: str | None = None


class SCIONGatewayError(Exception):
    """Base exception for SCION Gateway operations."""

    pass


class SCIONConnectionError(SCIONGatewayError):
    """SCION Gateway connection error."""

    pass


class SCIONPacketError(SCIONGatewayError):
    """SCION packet processing error."""

    pass


class SCIONGateway:
    """
    High-level interface to Betanet SCION Gateway.
    Manages Go sidecar process and HTTP/3 communication with Rust gateway.
    """

    def __init__(self, config: GatewayConfig):
        self.config = config
        self._sidecar_process: subprocess.Popen | None = None
        self._http_session: aiohttp.ClientSession | None = None
        self._is_running = False
        self._receive_task: asyncio.Task | None = None
        self._receive_handlers: list[callable] = []
        self._request_semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        # Statistics
        self.stats = {
            "packets_sent": 0,
            "packets_received": 0,
            "errors": 0,
            "connections": 0,
            "avg_latency_ms": 0.0,
        }

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    async def start(self) -> None:
        """Start SCION Gateway and all components."""
        if self._is_running:
            return

        logger.info("Starting SCION Gateway")

        try:
            # Create HTTP session for Rust gateway
            await self._create_http_session()

            # Wait for services to be ready (mock implementation)
            await self._wait_for_ready()

            self._is_running = True
            logger.info("SCION Gateway started successfully")

        except Exception as e:
            logger.error(f"Failed to start SCION Gateway: {e}")
            await self.stop()
            raise SCIONConnectionError(f"Failed to start SCION Gateway: {e}") from e

    async def stop(self) -> None:
        """Stop SCION Gateway and cleanup resources."""
        if not self._is_running:
            return

        logger.info("Stopping SCION Gateway")
        self._is_running = False

        # Close HTTP session
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

        logger.info("SCION Gateway stopped")

    async def health_check(self) -> dict[str, Any]:
        """Check gateway health status."""
        if not self._is_running:
            return {"status": "unhealthy", "scion_connected": False}

        return {
            "status": "healthy",
            "scion_connected": True,
            "connections": self.stats["connections"],
            "packets_sent": self.stats["packets_sent"],
            "packets_received": self.stats["packets_received"],
        }

    async def query_paths(self, destination: str) -> list[SCIONPath]:
        """Query available SCION paths to destination."""
        if not self._is_running:
            raise SCIONConnectionError("SCION Gateway not running")

        # Mock implementation for testing
        return [
            SCIONPath(
                path_id="path_0",
                fingerprint="fp_0",
                destination=destination,
                rtt_us=30000.0,  # 30ms
                loss_rate=0.01,  # 1%
                is_healthy=True,
                is_active=True,
            ),
            SCIONPath(
                path_id="path_1",
                fingerprint="fp_1",
                destination=destination,
                rtt_us=50000.0,  # 50ms
                loss_rate=0.02,  # 2%
                is_healthy=True,
                is_active=False,
            ),
        ]

    async def send_message(self, message: Message, destination: str, path_preference: str | None = None) -> str:
        """Send message via SCION."""
        if not self._is_running:
            raise SCIONConnectionError("SCION Gateway not running")

        # Serialize message
        packet_data = self._serialize_message(message)

        # Send via HTX tunnel
        packet_id = await self.send_packet(packet_data, destination, path_preference)

        self.stats["packets_sent"] += 1
        return packet_id

    async def send_packet(self, packet_data: bytes, destination: str, path_preference: str | None = None) -> str:
        """Send raw packet via SCION."""
        if not self._is_running:
            raise SCIONConnectionError("SCION Gateway not running")

        # Mock implementation for testing
        packet_id = f"packet_{int(time.time() * 1000000)}"
        self.stats["packets_sent"] += 1

        # Simulate some latency
        await asyncio.sleep(0.001)  # 1ms simulated latency

        return packet_id

    async def receive_packets(self, timeout_ms: int = 1000, max_packets: int = 10) -> list[tuple[bytes, str]]:
        """Receive packets from SCION network."""
        if not self._is_running:
            raise SCIONConnectionError("SCION Gateway not running")

        # Mock implementation - return empty list for testing
        await asyncio.sleep(timeout_ms / 1000.0)
        return []

    async def get_statistics(self) -> dict[str, Any]:
        """Get gateway statistics."""
        return self.stats.copy()

    def _serialize_message(self, message: Message) -> bytes:
        """Serialize message to bytes."""
        if hasattr(message, "payload"):
            return message.payload
        elif hasattr(message, "content"):
            return json.dumps(message.content).encode("utf-8")
        else:
            return json.dumps({"message": str(message)}).encode("utf-8")

    async def _create_http_session(self) -> None:
        """Create HTTP session for gateway communication."""
        connector = aiohttp.TCPConnector(ssl=self.config.verify_ssl)
        timeout = aiohttp.ClientTimeout(total=self.config.connection_timeout)

        self._http_session = aiohttp.ClientSession(connector=connector, timeout=timeout)

    async def _wait_for_ready(self) -> None:
        """Wait for gateway to be ready."""
        # Mock implementation - just wait a bit
        await asyncio.sleep(0.1)
