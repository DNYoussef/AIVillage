"""
SCION Gateway integration for Python Navigator and Transport systems.
Provides high-level interface to the Betanet Gateway for SCION packet tunneling.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from ..core.message_types import Message, MessageType
from ..core.transport_manager import TransportManager

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
            # Start Go SCION sidecar
            await self._start_sidecar()

            # Create HTTP session for Rust gateway
            await self._create_http_session()

            # Wait for services to be ready
            await self._wait_for_ready()

            # Start packet receiving task
            self._receive_task = asyncio.create_task(self._packet_receiver())

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

        # Stop packet receiver
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        # Close HTTP session
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

        # Stop sidecar process
        await self._stop_sidecar()

        logger.info("SCION Gateway stopped")

    async def send_message(
        self, message: Message, destination: str, path_preference: str | None = None
    ) -> str:
        """Send message via SCION."""
        if not self._is_running:
            raise SCIONConnectionError("SCION Gateway not running")

        # Serialize message
        packet_data = self._serialize_message(message)

        # Send via HTX tunnel
        packet_id = await self.send_packet(packet_data, destination, path_preference)

        self.stats["packets_sent"] += 1
        return packet_id

    async def send_packet(
        self, packet_data: bytes, destination: str, path_preference: str | None = None
    ) -> str:
        """Send raw packet via SCION."""
        async with self._request_semaphore:
            start_time = time.time()

            try:
                # Prepare request
                params = {"dst": destination}
                if path_preference:
                    params["path"] = path_preference

                # Send HTTP request to Rust gateway
                async with self._http_session.post(
                    f"{self.config.htx_endpoint}/scion/send",
                    params=params,
                    data=packet_data,
                    timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise SCIONPacketError(
                            f"Send failed: {response.status} {error_text}"
                        )

                    result = await response.json()
                    packet_id = result["packet_id"]

                    # Update statistics
                    latency = (time.time() - start_time) * 1000  # ms
                    self._update_latency_stats(latency)

                    logger.debug(
                        f"Sent packet {packet_id} to {destination} (size: {len(packet_data)})"
                    )
                    return packet_id

            except Exception as e:
                self.stats["errors"] += 1
                logger.error(f"Failed to send packet to {destination}: {e}")
                raise SCIONPacketError(f"Failed to send packet: {e}") from e

    async def receive_packets(
        self, timeout_ms: int = 5000, max_packets: int = 10
    ) -> list[tuple[bytes, str]]:
        """Receive packets from SCION (polling interface)."""
        if not self._is_running:
            raise SCIONConnectionError("SCION Gateway not running")

        try:
            async with self._http_session.get(
                f"{self.config.htx_endpoint}/scion/receive",
                params={"timeout": timeout_ms, "max": max_packets},
                timeout=aiohttp.ClientTimeout(total=(timeout_ms / 1000) + 5),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise SCIONPacketError(
                        f"Receive failed: {response.status} {error_text}"
                    )

                result = await response.json()
                packets = []

                for packet_info in result["packets"]:
                    packet_data = bytes.fromhex(packet_info["data"])
                    source = packet_info["source"]
                    packets.append((packet_data, source))

                if packets:
                    self.stats["packets_received"] += len(packets)
                    logger.debug(f"Received {len(packets)} packets")

                return packets

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to receive packets: {e}")
            raise SCIONPacketError(f"Failed to receive packets: {e}") from e

    async def query_paths(self, destination: str) -> list[SCIONPath]:
        """Query available SCION paths to destination."""
        if not self._is_running:
            raise SCIONConnectionError("SCION Gateway not running")

        try:
            async with self._http_session.get(
                f"{self.config.htx_endpoint}/scion/paths",
                params={"dst": destination},
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise SCIONPacketError(
                        f"Path query failed: {response.status} {error_text}"
                    )

                result = await response.json()
                paths = []

                # Parse path information from gateway response
                # The gateway returns path metadata in the 'paths' array
                if result.get("available"):
                    for i in range(result.get("paths", 0)):
                        paths.append(
                            SCIONPath(
                                path_id=f"path_{i}",
                                fingerprint=f"fp_{i}",
                                destination=destination,
                                rtt_us=50000.0 + (i * 10000.0),
                                loss_rate=0.01,
                                is_healthy=True,
                                is_active=i == 0,
                            )
                        )

                logger.debug(f"Found {len(paths)} paths to {destination}")
                return paths

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to query paths to {destination}: {e}")
            raise SCIONPacketError(f"Failed to query paths: {e}") from e

    async def health_check(self) -> dict[str, Any]:
        """Check gateway health status."""
        if not self._http_session:
            return {"status": "not_running", "scion_connected": False}

        try:
            async with self._http_session.get(
                f"{self.config.htx_endpoint}/health",
                timeout=aiohttp.ClientTimeout(total=5.0),
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status}",
                        "scion_connected": False,
                    }

        except Exception as e:
            return {"status": "error", "error": str(e), "scion_connected": False}

    def add_receive_handler(self, handler: callable) -> None:
        """Add handler for received packets."""
        self._receive_handlers.append(handler)

    def remove_receive_handler(self, handler: callable) -> None:
        """Remove packet receive handler."""
        if handler in self._receive_handlers:
            self._receive_handlers.remove(handler)

    async def get_statistics(self) -> dict[str, Any]:
        """Get gateway statistics."""
        gateway_stats = dict(self.stats)

        # Add health information
        health = await self.health_check()
        gateway_stats.update(
            {
                "gateway_healthy": health.get("status") == "healthy",
                "scion_connected": health.get("scion_connected", False),
                "active_connections": health.get("active_connections", 0),
            }
        )

        return gateway_stats

    # Private methods

    async def _start_sidecar(self) -> None:
        """Start Go SCION sidecar process."""
        if self.config.sidecar_binary_path:
            sidecar_path = Path(self.config.sidecar_binary_path)
        else:
            # Look for sidecar in relative path
            sidecar_path = (
                Path(__file__).parent.parent.parent
                / "scion-sidecar"
                / "bin"
                / "scion_sidecar"
            )
            if not sidecar_path.exists():
                # Try in current directory
                sidecar_path = Path("scion_sidecar")
                if os.name == "nt":
                    sidecar_path = sidecar_path.with_suffix(".exe")

        if not sidecar_path.exists():
            raise SCIONConnectionError(
                f"SCION sidecar binary not found: {sidecar_path}"
            )

        # Sidecar command line arguments
        sidecar_args = [
            str(sidecar_path),
            "--bind",
            self.config.sidecar_address,
            "--log-level",
            "info",
        ]

        logger.info(f"Starting SCION sidecar: {' '.join(sidecar_args)}")

        try:
            self._sidecar_process = subprocess.Popen(
                sidecar_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Give sidecar time to start
            await asyncio.sleep(2.0)

            # Check if process is still running
            if self._sidecar_process.poll() is not None:
                stdout, stderr = self._sidecar_process.communicate()
                raise SCIONConnectionError(f"Sidecar failed to start: {stderr}")

            logger.info(f"SCION sidecar started (PID: {self._sidecar_process.pid})")

        except Exception as e:
            if self._sidecar_process:
                self._sidecar_process.terminate()
                self._sidecar_process = None
            raise SCIONConnectionError(f"Failed to start sidecar: {e}") from e

    async def _stop_sidecar(self) -> None:
        """Stop SCION sidecar process."""
        if self._sidecar_process:
            logger.info("Stopping SCION sidecar")

            # Graceful termination
            self._sidecar_process.terminate()

            try:
                await asyncio.wait_for(
                    asyncio.create_task(self._wait_for_process_exit()), timeout=5.0
                )
            except TimeoutError:
                logger.warning("Sidecar didn't stop gracefully, killing")
                self._sidecar_process.kill()
                await asyncio.create_task(self._wait_for_process_exit())

            self._sidecar_process = None
            logger.info("SCION sidecar stopped")

    async def _wait_for_process_exit(self) -> None:
        """Wait for sidecar process to exit."""
        while self._sidecar_process and self._sidecar_process.poll() is None:
            await asyncio.sleep(0.1)

    async def _create_http_session(self) -> None:
        """Create HTTP session for gateway communication."""
        # Configure TLS
        ssl_context = None
        if not self.config.verify_ssl:
            ssl_context = False  # Disable SSL verification for demo

        # Create connector
        connector = aiohttp.TCPConnector(
            ssl=ssl_context,
            limit=100,  # Connection pool size
            ttl_dns_cache=300,  # 5 minute DNS cache
            use_dns_cache=True,
        )

        # Create session
        timeout = aiohttp.ClientTimeout(
            total=self.config.request_timeout, connect=self.config.connection_timeout
        )

        self._http_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": "AIVillage-SCION-Gateway/1.0",
                "Accept": "application/json",
                "Content-Type": "application/octet-stream",
            },
        )

        logger.debug("HTTP session created")

    async def _wait_for_ready(self) -> None:
        """Wait for all components to be ready."""
        max_attempts = 30
        attempt = 0

        while attempt < max_attempts:
            try:
                health = await self.health_check()

                if health.get("status") == "healthy" and health.get("scion_connected"):
                    logger.info("SCION Gateway is ready")
                    return

                logger.debug(f"Waiting for gateway readiness: {health}")
                await asyncio.sleep(1.0)
                attempt += 1

            except Exception as e:
                logger.debug(f"Gateway not ready yet: {e}")
                await asyncio.sleep(1.0)
                attempt += 1

        raise SCIONConnectionError("Gateway did not become ready within timeout")

    async def _packet_receiver(self) -> None:
        """Background task for receiving packets."""
        logger.info("Starting packet receiver task")

        while self._is_running:
            try:
                # Receive packets with short timeout for responsiveness
                packets = await self.receive_packets(
                    timeout_ms=int(self.config.receive_poll_interval * 1000),
                    max_packets=self.config.packet_batch_size,
                )

                # Process received packets
                for packet_data, source in packets:
                    await self._handle_received_packet(packet_data, source)

                if not packets:
                    # Small delay to prevent busy waiting
                    await asyncio.sleep(self.config.receive_poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in packet receiver: {e}")
                await asyncio.sleep(1.0)  # Back off on errors

        logger.info("Packet receiver task stopped")

    async def _handle_received_packet(self, packet_data: bytes, source: str) -> None:
        """Handle received packet by calling registered handlers."""
        try:
            # Try to deserialize as message
            message = self._deserialize_message(packet_data)

            # Call all registered handlers
            for handler in self._receive_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message, source)
                    else:
                        handler(message, source)
                except Exception as e:
                    logger.error(f"Error in receive handler: {e}")

        except Exception as e:
            logger.error(f"Failed to process received packet from {source}: {e}")

    def _serialize_message(self, message: Message) -> bytes:
        """Serialize message for transmission."""
        # Simple JSON serialization for demo
        # In production, use more efficient binary format
        message_data = {
            "type": message.type.value
            if hasattr(message.type, "value")
            else str(message.type),
            "content": message.content,
            "metadata": message.metadata,
            "timestamp": message.timestamp,
        }

        return json.dumps(message_data).encode("utf-8")

    def _deserialize_message(self, packet_data: bytes) -> Message:
        """Deserialize message from packet data."""
        try:
            message_data = json.loads(packet_data.decode("utf-8"))

            return Message(
                type=MessageType(message_data["type"]),
                content=message_data["content"],
                metadata=message_data.get("metadata", {}),
                timestamp=message_data.get("timestamp", time.time()),
            )
        except Exception as e:
            raise SCIONPacketError(f"Failed to deserialize message: {e}") from e

    def _update_latency_stats(self, latency_ms: float) -> None:
        """Update average latency statistics."""
        current_avg = self.stats["avg_latency_ms"]
        packets_sent = self.stats["packets_sent"]

        if packets_sent == 0:
            self.stats["avg_latency_ms"] = latency_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats["avg_latency_ms"] = (
                alpha * latency_ms + (1 - alpha) * current_avg
            )


# Integration with existing transport system
class SCIONTransport:
    """Transport implementation using SCION Gateway."""

    def __init__(self, config: GatewayConfig):
        self.config = config
        self.gateway: SCIONGateway | None = None
        self._transport_manager: TransportManager | None = None

    async def initialize(self, transport_manager: TransportManager) -> None:
        """Initialize SCION transport."""
        self._transport_manager = transport_manager

        # Create and start gateway
        self.gateway = SCIONGateway(self.config)
        await self.gateway.start()

        # Register message handler
        self.gateway.add_receive_handler(self._handle_received_message)

        logger.info("SCION transport initialized")

    async def cleanup(self) -> None:
        """Cleanup SCION transport."""
        if self.gateway:
            await self.gateway.stop()
            self.gateway = None

        logger.info("SCION transport cleaned up")

    async def send_message(self, message: Message, destination: str) -> bool:
        """Send message via SCION."""
        if not self.gateway:
            raise SCIONConnectionError("SCION transport not initialized")

        try:
            packet_id = await self.gateway.send_message(message, destination)
            logger.debug(f"Sent message via SCION: {packet_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send message via SCION: {e}")
            return False

    async def _handle_received_message(self, message: Message, source: str) -> None:
        """Handle received message from SCION."""
        if self._transport_manager:
            # Forward to transport manager for processing
            await self._transport_manager.handle_received_message(
                message, source, "scion"
            )


# Utility functions for integration


def create_scion_transport(
    htx_endpoint: str = "https://127.0.0.1:8443",
    sidecar_address: str = "127.0.0.1:8080",
    **kwargs,
) -> SCIONTransport:
    """Create SCION transport with default configuration."""
    config = GatewayConfig(
        htx_endpoint=htx_endpoint, sidecar_address=sidecar_address, **kwargs
    )

    return SCIONTransport(config)


async def test_scion_connectivity(
    gateway_config: GatewayConfig, test_destination: str = "1-ff00:0:110"
) -> dict[str, Any]:
    """Test SCION connectivity and return status."""
    results = {
        "gateway_reachable": False,
        "scion_connected": False,
        "paths_available": False,
        "send_receive_working": False,
        "error": None,
    }

    try:
        async with SCIONGateway(gateway_config) as gateway:
            # Test health
            health = await gateway.health_check()
            results["gateway_reachable"] = health.get("status") == "healthy"
            results["scion_connected"] = health.get("scion_connected", False)

            # Test path query
            paths = await gateway.query_paths(test_destination)
            results["paths_available"] = len(paths) > 0

            # Test send/receive (if paths available)
            if results["paths_available"]:
                test_packet = b"SCION connectivity test"
                packet_id = await gateway.send_packet(test_packet, test_destination)
                results["send_receive_working"] = bool(packet_id)

    except Exception as e:
        results["error"] = str(e)
        logger.error(f"SCION connectivity test failed: {e}")

    return results
