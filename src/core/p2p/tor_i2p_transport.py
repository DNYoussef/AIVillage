"""
Tor/I2P Privacy Transport Implementation - Phase 3 Advanced Features

Implements anonymous networking transport using Tor and I2P networks for:
- Privacy-preserving P2P communication
- Onion routing and garlic routing
- Hidden service hosting
- Censorship resistance
"""

import asyncio
import hashlib
import logging
import socket
import struct
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class PrivacyNetwork(Enum):
    """Supported privacy networks."""

    TOR = "tor"
    I2P = "i2p"
    BOTH = "both"  # Use both networks for redundancy


@dataclass
class OnionAddress:
    """Represents a Tor hidden service address."""

    address: str  # .onion address
    port: int
    public_key: bytes | None = None

    @property
    def full_address(self) -> str:
        return f"{self.address}:{self.port}"


@dataclass
class I2PDestination:
    """Represents an I2P destination."""

    destination: str  # Base64 I2P destination
    port: int
    public_key: bytes | None = None

    @property
    def full_address(self) -> str:
        return f"{self.destination}:{self.port}"


class TorTransport:
    """Tor network transport implementation."""

    def __init__(
        self,
        tor_proxy_host: str = "127.0.0.1",
        tor_proxy_port: int = 9050,
        tor_control_port: int = 9051,
        tor_control_password: str | None = None,
    ):
        """
        Initialize Tor transport.

        Args:
            tor_proxy_host: Tor SOCKS proxy host
            tor_proxy_port: Tor SOCKS proxy port (usually 9050)
            tor_control_port: Tor control port (usually 9051)
            tor_control_password: Control port password
        """
        self.proxy_host = tor_proxy_host
        self.proxy_port = tor_proxy_port
        self.control_port = tor_control_port
        self.control_password = tor_control_password

        self.controller = None
        self.hidden_service = None
        self.connections: dict[str, asyncio.StreamWriter] = {}

    async def start(self) -> bool:
        """Start Tor transport and create hidden service."""
        try:
            # Check if Tor is running
            if not await self._check_tor_running():
                logger.error("Tor is not running. Please start Tor daemon.")
                return False

            # Connect to control port
            if not await self._connect_control_port():
                logger.warning("Could not connect to Tor control port")
                # Can still use as client without control

            # Create hidden service
            self.hidden_service = await self._create_hidden_service()
            if self.hidden_service:
                logger.info(
                    f"Tor hidden service created: {self.hidden_service.address}"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to start Tor transport: {e}")
            return False

    async def _check_tor_running(self) -> bool:
        """Check if Tor SOCKS proxy is accessible."""
        try:
            # Try to connect to SOCKS port
            reader, writer = await asyncio.open_connection(
                self.proxy_host, self.proxy_port
            )
            writer.close()
            await writer.wait_closed()
            return True
        except:
            return False

    async def _connect_control_port(self) -> bool:
        """Connect to Tor control port for hidden service management."""
        try:
            # Try to import stem for Tor control
            import stem
            from stem.control import Controller

            self.controller = Controller.from_port(
                address=self.proxy_host, port=self.control_port
            )

            if self.control_password:
                self.controller.authenticate(password=self.control_password)
            else:
                self.controller.authenticate()

            logger.info("Connected to Tor control port")
            return True

        except ImportError:
            logger.warning("Stem library not available for Tor control")
            return False
        except Exception as e:
            logger.warning(f"Could not connect to Tor control port: {e}")
            return False

    async def _create_hidden_service(self) -> OnionAddress | None:
        """Create a Tor hidden service."""
        if not self.controller:
            return None

        try:
            # Create hidden service
            service_dir = Path.home() / ".aivillage" / "tor_hidden_service"
            service_dir.mkdir(parents=True, exist_ok=True)

            # Configure hidden service
            response = self.controller.create_ephemeral_hidden_service(
                {80: 8888},
                await_publication=True,  # Map port 80 to local 8888
            )

            onion_address = f"{response.service_id}.onion"

            return OnionAddress(address=onion_address, port=80)

        except Exception as e:
            logger.error(f"Failed to create hidden service: {e}")
            return None

    async def connect_to_onion(self, onion_address: OnionAddress) -> bool:
        """Connect to a Tor hidden service."""
        try:
            # Create SOCKS connection through Tor
            reader, writer = await self._socks_connect(
                onion_address.address, onion_address.port
            )

            self.connections[onion_address.full_address] = writer

            # Start receiving messages
            asyncio.create_task(
                self._receive_loop(onion_address.full_address, reader, writer)
            )

            logger.info(f"Connected to {onion_address.full_address}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to {onion_address.full_address}: {e}")
            return False

    async def _socks_connect(
        self, host: str, port: int
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Connect through Tor SOCKS proxy."""
        # Connect to SOCKS proxy
        reader, writer = await asyncio.open_connection(self.proxy_host, self.proxy_port)

        # SOCKS5 handshake
        # 1. Send greeting
        writer.write(b"\x05\x01\x00")  # SOCKS5, 1 auth method, no auth
        await writer.drain()

        response = await reader.read(2)
        if response != b"\x05\x00":
            raise Exception("SOCKS handshake failed")

        # 2. Send connection request
        # For .onion addresses, we need to send as domain name
        if host.endswith(".onion"):
            host_bytes = host.encode("ascii")
            addr_type = b"\x03"  # Domain name
            addr_data = bytes([len(host_bytes)]) + host_bytes
        else:
            # Regular IP (shouldn't happen for Tor)
            addr_type = b"\x01"  # IPv4
            addr_data = socket.inet_aton(host)

        port_bytes = struct.pack(">H", port)

        request = b"\x05\x01\x00" + addr_type + addr_data + port_bytes
        writer.write(request)
        await writer.drain()

        # Read response
        response = await reader.read(10)
        if response[1] != 0:
            raise Exception(f"SOCKS connection failed: {response[1]}")

        logger.debug(f"SOCKS connection established to {host}:{port}")
        return reader, writer

    async def send_message(self, destination: str, message: bytes) -> bool:
        """Send a message through Tor."""
        if destination not in self.connections:
            # Parse destination and connect
            if ":" in destination:
                host, port = destination.rsplit(":", 1)
                onion = OnionAddress(host, int(port))
                if not await self.connect_to_onion(onion):
                    return False
            else:
                logger.error(f"Invalid destination: {destination}")
                return False

        try:
            writer = self.connections[destination]

            # Add length prefix
            length_prefix = struct.pack(">I", len(message))
            writer.write(length_prefix + message)
            await writer.drain()

            logger.debug(f"Sent {len(message)} bytes to {destination}")
            return True

        except Exception as e:
            logger.error(f"Failed to send to {destination}: {e}")
            # Remove failed connection
            if destination in self.connections:
                del self.connections[destination]
            return False

    async def _receive_loop(
        self, address: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        """Receive messages from a connection."""
        try:
            while True:
                # Read length prefix
                length_data = await reader.readexactly(4)
                if not length_data:
                    break

                length = struct.unpack(">I", length_data)[0]

                # Read message
                message = await reader.readexactly(length)

                # Handle message
                await self._handle_message(address, message)

        except Exception as e:
            logger.error(f"Receive error from {address}: {e}")
        finally:
            # Cleanup connection
            writer.close()
            await writer.wait_closed()
            if address in self.connections:
                del self.connections[address]

    async def _handle_message(self, sender: str, message: bytes):
        """Handle received message."""
        logger.debug(f"Received {len(message)} bytes from {sender}")
        # Process message (would integrate with main message handler)

    async def stop(self):
        """Stop Tor transport."""
        # Close all connections
        for writer in list(self.connections.values()):
            writer.close()
            await writer.wait_closed()

        # Remove hidden service
        if self.controller and self.hidden_service:
            try:
                self.controller.remove_ephemeral_hidden_service(
                    self.hidden_service.address.replace(".onion", "")
                )
            except:
                pass

        # Close controller
        if self.controller:
            self.controller.close()

        logger.info("Tor transport stopped")


class I2PTransport:
    """I2P network transport implementation."""

    def __init__(self, i2p_host: str = "127.0.0.1", i2p_sam_port: int = 7656):
        """
        Initialize I2P transport.

        Args:
            i2p_host: I2P SAM bridge host
            i2p_sam_port: I2P SAM bridge port (usually 7656)
        """
        self.sam_host = i2p_host
        self.sam_port = i2p_sam_port

        self.session_id = None
        self.destination = None
        self.connections: dict[str, asyncio.StreamWriter] = {}

    async def start(self) -> bool:
        """Start I2P transport and create destination."""
        try:
            # Check if I2P is running
            if not await self._check_i2p_running():
                logger.error("I2P is not running. Please start I2P router.")
                return False

            # Create I2P session
            self.session_id = await self._create_session()
            if not self.session_id:
                return False

            # Get our destination
            self.destination = await self._get_destination()
            if self.destination:
                logger.info(
                    f"I2P destination created: {self.destination.destination[:20]}..."
                )

            return True

        except Exception as e:
            logger.error(f"Failed to start I2P transport: {e}")
            return False

    async def _check_i2p_running(self) -> bool:
        """Check if I2P SAM bridge is accessible."""
        try:
            reader, writer = await asyncio.open_connection(self.sam_host, self.sam_port)
            writer.close()
            await writer.wait_closed()
            return True
        except:
            return False

    async def _create_session(self) -> str | None:
        """Create I2P SAM session."""
        try:
            reader, writer = await asyncio.open_connection(self.sam_host, self.sam_port)

            # SAM HELLO
            writer.write(b"HELLO VERSION MIN=3.0 MAX=3.3\n")
            await writer.drain()

            response = await reader.readline()
            if not response.startswith(b"HELLO REPLY"):
                raise Exception("SAM HELLO failed")

            # Create session
            session_id = f"aivillage_{hashlib.md5(str(asyncio.get_event_loop().time()).encode()).hexdigest()[:8]}"

            writer.write(
                f"SESSION CREATE STYLE=STREAM ID={session_id} DESTINATION=TRANSIENT\n".encode()
            )
            await writer.drain()

            response = await reader.readline()
            if b"RESULT=OK" not in response:
                raise Exception("SAM SESSION CREATE failed")

            # Parse destination from response
            parts = response.decode().split()
            for part in parts:
                if part.startswith("DESTINATION="):
                    dest = part.split("=")[1]
                    self.destination = I2PDestination(dest, 0)
                    break

            writer.close()
            await writer.wait_closed()

            logger.info(f"I2P session created: {session_id}")
            return session_id

        except Exception as e:
            logger.error(f"Failed to create I2P session: {e}")
            return None

    async def _get_destination(self) -> I2PDestination | None:
        """Get our I2P destination."""
        return self.destination

    async def connect_to_destination(self, destination: I2PDestination) -> bool:
        """Connect to an I2P destination."""
        try:
            reader, writer = await asyncio.open_connection(self.sam_host, self.sam_port)

            # SAM HELLO
            writer.write(b"HELLO VERSION MIN=3.0 MAX=3.3\n")
            await writer.drain()
            await reader.readline()

            # STREAM CONNECT
            connect_cmd = f"STREAM CONNECT ID={self.session_id} DESTINATION={destination.destination}\n"
            writer.write(connect_cmd.encode())
            await writer.drain()

            response = await reader.readline()
            if b"RESULT=OK" not in response:
                raise Exception("STREAM CONNECT failed")

            self.connections[destination.full_address] = writer

            # Start receiving
            asyncio.create_task(
                self._receive_loop(destination.full_address, reader, writer)
            )

            logger.info("Connected to I2P destination")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to I2P destination: {e}")
            return False

    async def send_message(self, destination: str, message: bytes) -> bool:
        """Send a message through I2P."""
        if destination not in self.connections:
            # Parse and connect
            if ":" in destination:
                dest, port = destination.rsplit(":", 1)
                i2p_dest = I2PDestination(dest, int(port))
                if not await self.connect_to_destination(i2p_dest):
                    return False
            else:
                logger.error(f"Invalid destination: {destination}")
                return False

        try:
            writer = self.connections[destination]

            # Send with length prefix
            length_prefix = struct.pack(">I", len(message))
            writer.write(length_prefix + message)
            await writer.drain()

            logger.debug(f"Sent {len(message)} bytes via I2P")
            return True

        except Exception as e:
            logger.error(f"Failed to send via I2P: {e}")
            if destination in self.connections:
                del self.connections[destination]
            return False

    async def _receive_loop(
        self, address: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        """Receive messages from I2P connection."""
        try:
            while True:
                # Read length prefix
                length_data = await reader.readexactly(4)
                if not length_data:
                    break

                length = struct.unpack(">I", length_data)[0]

                # Read message
                message = await reader.readexactly(length)

                # Handle message
                await self._handle_message(address, message)

        except Exception as e:
            logger.error(f"I2P receive error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            if address in self.connections:
                del self.connections[address]

    async def _handle_message(self, sender: str, message: bytes):
        """Handle received I2P message."""
        logger.debug(f"Received {len(message)} bytes via I2P")
        # Process message

    async def stop(self):
        """Stop I2P transport."""
        # Close connections
        for writer in list(self.connections.values()):
            writer.close()
            await writer.wait_closed()

        # Destroy session
        if self.session_id:
            try:
                reader, writer = await asyncio.open_connection(
                    self.sam_host, self.sam_port
                )

                writer.write(b"HELLO VERSION MIN=3.0 MAX=3.3\n")
                await writer.drain()
                await reader.readline()

                writer.write(f"SESSION REMOVE ID={self.session_id}\n".encode())
                await writer.drain()

                writer.close()
                await writer.wait_closed()
            except:
                pass

        logger.info("I2P transport stopped")


class PrivacyTransport:
    """
    Combined Tor/I2P privacy transport with automatic failover.
    """

    def __init__(
        self,
        networks: PrivacyNetwork = PrivacyNetwork.BOTH,
        message_callback: Callable | None = None,
    ):
        """
        Initialize privacy transport.

        Args:
            networks: Which privacy networks to use
            message_callback: Callback for received messages
        """
        self.networks = networks
        self.message_callback = message_callback

        self.tor_transport = None
        self.i2p_transport = None

        if networks in [PrivacyNetwork.TOR, PrivacyNetwork.BOTH]:
            self.tor_transport = TorTransport()

        if networks in [PrivacyNetwork.I2P, PrivacyNetwork.BOTH]:
            self.i2p_transport = I2PTransport()

    async def start(self) -> bool:
        """Start privacy transports."""
        success = False

        if self.tor_transport:
            if await self.tor_transport.start():
                logger.info("Tor transport started")
                success = True
            else:
                logger.warning("Tor transport failed to start")

        if self.i2p_transport:
            if await self.i2p_transport.start():
                logger.info("I2P transport started")
                success = True
            else:
                logger.warning("I2P transport failed to start")

        return success

    async def send_message(self, destination: str, message: bytes) -> bool:
        """
        Send message through available privacy network.

        Tries Tor first, falls back to I2P if available.
        """
        # Encrypt message for additional privacy
        encrypted = self._encrypt_message(message)

        # Try Tor first
        if self.tor_transport and destination.endswith(".onion"):
            if await self.tor_transport.send_message(destination, encrypted):
                return True

        # Try I2P
        if self.i2p_transport and len(destination) > 100:  # I2P destinations are long
            if await self.i2p_transport.send_message(destination, encrypted):
                return True

        # Try any available transport
        if self.tor_transport:
            if await self.tor_transport.send_message(destination, encrypted):
                return True

        if self.i2p_transport:
            if await self.i2p_transport.send_message(destination, encrypted):
                return True

        logger.error(f"Failed to send message to {destination}")
        return False

    def _encrypt_message(self, message: bytes) -> bytes:
        """Add extra encryption layer for messages."""
        # This would use proper encryption like AES or ChaCha20
        # For now, just add a simple XOR with a key
        key = b"AIVillagePrivacy"
        encrypted = bytearray()

        for i, byte in enumerate(message):
            encrypted.append(byte ^ key[i % len(key)])

        return bytes(encrypted)

    def get_addresses(self) -> dict[str, str]:
        """Get our privacy network addresses."""
        addresses = {}

        if self.tor_transport and self.tor_transport.hidden_service:
            addresses["tor"] = self.tor_transport.hidden_service.full_address

        if self.i2p_transport and self.i2p_transport.destination:
            addresses["i2p"] = self.i2p_transport.destination.full_address

        return addresses

    async def stop(self):
        """Stop all privacy transports."""
        if self.tor_transport:
            await self.tor_transport.stop()

        if self.i2p_transport:
            await self.i2p_transport.stop()

        logger.info("Privacy transport stopped")


# Example usage and testing
async def test_privacy_transport():
    """Test privacy transport functionality."""
    transport = PrivacyTransport(networks=PrivacyNetwork.BOTH)

    if await transport.start():
        addresses = transport.get_addresses()
        logger.info(f"Privacy addresses: {addresses}")

        # Test sending a message
        test_message = b"Hello from AIVillage privacy network!"

        # Would need another node's address to actually send
        # await transport.send_message("destination.onion:80", test_message)

        # Keep running for testing
        await asyncio.sleep(60)

        await transport.stop()
    else:
        logger.error("Failed to start privacy transport")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_privacy_transport())
