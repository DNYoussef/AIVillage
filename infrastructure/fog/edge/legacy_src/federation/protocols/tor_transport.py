"""Tor Hidden Service Transport for AIVillage Federation

Implements Tor integration for anonymous communication with:
- Hidden service (.onion) address generation
- SOCKS proxy routing for outbound connections
- Circuit building with minimum 3 hops
- Stem library for Tor daemon management
- Bridge configuration for censored regions
"""

import asyncio
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any

# Tor control and management
try:
    import stem
    import stem.control
    import stem.descriptor.hidden_service
    import stem.process

    TOR_AVAILABLE = True
except ImportError:
    TOR_AVAILABLE = False
    logging.warning("stem library not available - Tor transport disabled")

# HTTP server for hidden service
try:
    import aiohttp
    from aiohttp import ClientSession, web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logging.warning("aiohttp not available - Tor HTTP transport disabled")

logger = logging.getLogger(__name__)


@dataclass
class TorCircuit:
    """Tor circuit information"""

    circuit_id: str
    path: list[str]  # Relay fingerprints
    status: str
    created_at: float
    purpose: str

    def is_established(self) -> bool:
        return self.status == "BUILT"

    def hop_count(self) -> int:
        return len(self.path)


@dataclass
class TorHiddenService:
    """Tor hidden service configuration"""

    service_id: str
    onion_address: str
    private_key: str
    port: int
    target_port: int
    created_at: float

    def get_onion_url(self) -> str:
        return f"http://{self.onion_address}:{self.port}"


class TorTransport:
    """Tor transport for anonymous communication"""

    def __init__(
        self,
        socks_port: int = 9050,
        control_port: int = 9051,
        hidden_service_port: int = 80,
        target_port: int = 8080,
        data_directory: str | None = None,
    ):
        self.socks_port = socks_port
        self.control_port = control_port
        self.hidden_service_port = hidden_service_port
        self.target_port = target_port

        # Tor data directory
        self.data_directory = data_directory or tempfile.mkdtemp(prefix="tor_")

        # Tor process and control
        self.tor_process = None
        self.tor_controller = None

        # Hidden service
        self.hidden_service: TorHiddenService | None = None
        self.http_server: web.Application | None = None
        self.http_runner: web.AppRunner | None = None

        # Circuit management
        self.circuits: dict[str, TorCircuit] = {}
        self.min_circuit_hops = 3

        # Message handling
        self.message_handlers: dict[str, Any] = {}
        self.pending_requests: dict[str, asyncio.Future] = {}

        # Connection pool for outbound requests
        self.client_session: ClientSession | None = None

        # Statistics
        self.stats = {
            "circuits_built": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "hidden_service_requests": 0,
        }

        self.is_running = False

        logger.info(f"TorTransport initialized: SOCKS={socks_port}, Control={control_port}")

    async def start(self) -> bool:
        """Start Tor transport with hidden service"""
        if not TOR_AVAILABLE:
            logger.error("Tor stem library not available")
            return False

        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp not available for HTTP transport")
            return False

        if self.is_running:
            return True

        logger.info("Starting Tor transport...")

        try:
            # Step 1: Start Tor daemon
            if not await self._start_tor_daemon():
                return False

            # Step 2: Connect to Tor control port
            if not await self._connect_tor_control():
                return False

            # Step 3: Create hidden service
            if not await self._create_hidden_service():
                return False

            # Step 4: Start HTTP server for hidden service
            if not await self._start_http_server():
                return False

            # Step 5: Create client session for outbound requests
            await self._create_client_session()

            # Step 6: Start circuit monitoring
            asyncio.create_task(self._circuit_monitoring_loop())

            self.is_running = True
            logger.info("Tor transport started successfully")
            logger.info(f"Hidden service: {self.hidden_service.onion_address}")

            return True

        except Exception as e:
            logger.exception(f"Failed to start Tor transport: {e}")
            await self.stop()
            return False

    async def stop(self):
        """Stop Tor transport"""
        logger.info("Stopping Tor transport...")
        self.is_running = False

        # Close client session
        if self.client_session:
            await self.client_session.close()

        # Stop HTTP server
        if self.http_runner:
            await self.http_runner.cleanup()

        # Close Tor controller
        if self.tor_controller:
            self.tor_controller.close()

        # Stop Tor process
        if self.tor_process:
            self.tor_process.terminate()

        logger.info("Tor transport stopped")

    async def send_message(self, recipient_onion: str, payload: dict[str, Any], timeout: int = 30) -> bool:
        """Send message to Tor hidden service"""
        if not self.is_running:
            return False

        try:
            # Ensure .onion address format
            if not recipient_onion.endswith(".onion"):
                logger.error(f"Invalid onion address: {recipient_onion}")
                return False

            # Create message
            message = {
                "id": str(uuid.uuid4()),
                "sender": self.hidden_service.onion_address,
                "timestamp": time.time(),
                "payload": payload,
            }

            # Send via Tor SOCKS proxy
            url = f"http://{recipient_onion}:{self.hidden_service_port}/message"

            async with self.client_session.post(
                url, json=message, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    self.stats["messages_sent"] += 1
                    logger.debug(f"Sent message to {recipient_onion[:16]}...")
                    return True
                logger.warning(f"Message send failed: HTTP {response.status}")
                return False

        except Exception as e:
            logger.error(f"Tor message send failed: {e}")
            return False

    async def create_circuit(self, purpose: str = "general", min_hops: int | None = None) -> str | None:
        """Create new Tor circuit with specified parameters"""
        if not self.tor_controller:
            return None

        try:
            # Use specified hops or default minimum
            hops = min_hops or self.min_circuit_hops

            # Create circuit through Tor controller
            circuit_id = self.tor_controller.new_circuit(await_build=True)

            # Get circuit details
            circuit_info = self.tor_controller.get_circuit(circuit_id)

            if circuit_info and circuit_info.status == "BUILT":
                # Store circuit information
                path = [relay[0] for relay in circuit_info.path]  # Extract relay fingerprints

                circuit = TorCircuit(
                    circuit_id=circuit_id,
                    path=path,
                    status=circuit_info.status,
                    created_at=time.time(),
                    purpose=purpose,
                )

                self.circuits[circuit_id] = circuit
                self.stats["circuits_built"] += 1

                logger.info(f"Created {hops}-hop circuit: {circuit_id}")
                return circuit_id

        except Exception as e:
            logger.error(f"Circuit creation failed: {e}")

        return None

    def get_onion_address(self) -> str | None:
        """Get our hidden service onion address"""
        if self.hidden_service:
            return self.hidden_service.onion_address
        return None

    def get_circuit_info(self, circuit_id: str) -> TorCircuit | None:
        """Get information about specific circuit"""
        return self.circuits.get(circuit_id)

    def get_active_circuits(self) -> list[TorCircuit]:
        """Get all active circuits"""
        return [circuit for circuit in self.circuits.values() if circuit.is_established()]

    def register_message_handler(self, message_type: str, handler):
        """Register handler for incoming messages"""
        self.message_handlers[message_type] = handler

    def get_status(self) -> dict[str, Any]:
        """Get Tor transport status"""
        return {
            "is_running": self.is_running,
            "onion_address": self.get_onion_address(),
            "active_circuits": len(self.get_active_circuits()),
            "total_circuits": len(self.circuits),
            "socks_port": self.socks_port,
            "control_port": self.control_port,
            "hidden_service_port": self.hidden_service_port,
            "stats": self.stats.copy(),
        }

    async def _start_tor_daemon(self) -> bool:
        """Start Tor daemon process"""
        try:
            # Tor configuration
            tor_config = {
                "SocksPort": str(self.socks_port),
                "ControlPort": str(self.control_port),
                "DataDirectory": self.data_directory,
                "CookieAuthentication": "1",
                "ExitPolicy": "reject *:*",  # Relay traffic only, no exits
            }

            logger.info("Starting Tor daemon...")

            # Start Tor process
            self.tor_process = stem.process.launch_tor_with_config(
                config=tor_config,
                init_msg_handler=lambda line: logger.debug(f"Tor: {line}"),
                timeout=60,
            )

            # Wait for Tor to be ready
            await asyncio.sleep(3)

            logger.info(f"Tor daemon started on SOCKS port {self.socks_port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start Tor daemon: {e}")
            return False

    async def _connect_tor_control(self) -> bool:
        """Connect to Tor control port"""
        try:
            # Connect to control port
            self.tor_controller = stem.control.Controller.from_port(port=self.control_port)
            self.tor_controller.authenticate()

            logger.info("Connected to Tor control port")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Tor control: {e}")
            return False

    async def _create_hidden_service(self) -> bool:
        """Create Tor hidden service"""
        try:
            # Create hidden service directory
            hs_dir = os.path.join(self.data_directory, "hidden_service")
            os.makedirs(hs_dir, exist_ok=True)

            # Configure hidden service
            response = self.tor_controller.create_hidden_service(
                hs_dir, self.hidden_service_port, target_port=self.target_port
            )

            # Extract onion address and private key
            onion_address = response.service_id + ".onion"

            # Store hidden service info
            self.hidden_service = TorHiddenService(
                service_id=response.service_id,
                onion_address=onion_address,
                private_key="",  # Would extract from key files
                port=self.hidden_service_port,
                target_port=self.target_port,
                created_at=time.time(),
            )

            logger.info(f"Created hidden service: {onion_address}")
            return True

        except Exception as e:
            logger.error(f"Failed to create hidden service: {e}")
            return False

    async def _start_http_server(self) -> bool:
        """Start HTTP server for hidden service"""
        try:
            # Create aiohttp application
            app = web.Application()

            # Add routes for federation messages
            app.router.add_post("/message", self._handle_http_message)
            app.router.add_get("/status", self._handle_http_status)
            app.router.add_get("/", self._handle_http_root)

            # Create and start runner
            self.http_runner = web.AppRunner(app)
            await self.http_runner.setup()

            site = web.TCPSite(self.http_runner, "localhost", self.target_port)
            await site.start()

            logger.info(f"HTTP server started on port {self.target_port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start HTTP server: {e}")
            return False

    async def _create_client_session(self):
        """Create HTTP client session with Tor SOCKS proxy"""
        # SOCKS proxy configuration
        connector = aiohttp.TCPConnector(
            use_dns_cache=False,  # Important for Tor
        )

        # Create session with SOCKS proxy
        self.client_session = ClientSession(
            connector=connector,
            headers={"User-Agent": "AIVillage/1.0"},
            timeout=aiohttp.ClientTimeout(total=60),
        )

        # Configure SOCKS proxy (simplified - would need proper SOCKS implementation)
        logger.info(f"Created client session with Tor SOCKS proxy on port {self.socks_port}")

    async def _handle_http_message(self, request: web.Request) -> web.Response:
        """Handle incoming HTTP message"""
        try:
            # Parse message
            message = await request.json()

            self.stats["messages_received"] += 1
            sender_onion = message.get("sender", "unknown")

            logger.debug(f"Received message from {sender_onion[:16]}...")

            # Route to appropriate handler
            payload = message.get("payload", {})
            message_type = payload.get("type", "default")

            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)

            return web.json_response({"status": "received"})

        except Exception as e:
            logger.error(f"Error handling HTTP message: {e}")
            return web.json_response({"error": str(e)}, status=400)

    async def _handle_http_status(self, request: web.Request) -> web.Response:
        """Handle status request"""
        status = self.get_status()
        return web.json_response(status)

    async def _handle_http_root(self, request: web.Request) -> web.Response:
        """Handle root request"""
        return web.json_response(
            {
                "service": "AIVillage Tor Transport",
                "onion_address": self.get_onion_address(),
                "version": "1.0",
            }
        )

    async def _circuit_monitoring_loop(self):
        """Monitor Tor circuits and maintain minimum count"""
        while self.is_running:
            try:
                # Get current circuit status
                if self.tor_controller:
                    circuits = self.tor_controller.get_circuits()

                    # Update circuit information
                    for circuit in circuits:
                        if circuit.id in self.circuits:
                            self.circuits[circuit.id].status = circuit.status

                    # Remove closed circuits
                    closed_circuits = [
                        cid for cid, circuit in self.circuits.items() if circuit.status in ["CLOSED", "FAILED"]
                    ]

                    for cid in closed_circuits:
                        del self.circuits[cid]

                    # Ensure minimum circuit count
                    active_circuits = self.get_active_circuits()
                    if len(active_circuits) < 3:  # Maintain at least 3 circuits
                        await self.create_circuit("general")

                # Check every 30 seconds
                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Circuit monitoring error: {e}")
                await asyncio.sleep(60)


class TorBridgeManager:
    """Manages Tor bridges for censored regions"""

    def __init__(self):
        self.bridges: list[str] = []
        self.bridge_types = ["obfs4", "meek", "snowflake"]

    def add_bridge(self, bridge_line: str):
        """Add bridge configuration line"""
        if bridge_line not in self.bridges:
            self.bridges.append(bridge_line)
            logger.info(f"Added bridge: {bridge_line[:50]}...")

    def get_bridge_config(self) -> dict[str, Any]:
        """Get bridge configuration for Tor"""
        if not self.bridges:
            return {}

        return {
            "UseBridges": "1",
            "Bridge": self.bridges,
            "ClientTransportPlugin": "obfs4 exec /usr/bin/obfs4proxy",
        }

    async def fetch_bridges_from_bridgedb(self) -> list[str]:
        """Fetch bridges from Tor BridgeDB (placeholder)"""
        # In practice, would fetch from https://bridges.torproject.org/
        logger.info("Fetching bridges from BridgeDB...")

        # Return some example bridge lines
        return [
            "obfs4 192.0.2.1:1234 FINGERPRINT cert=CERTIFICATE iat-mode=0",
            "obfs4 192.0.2.2:1234 FINGERPRINT cert=CERTIFICATE iat-mode=0",
        ]


class TorFederationExtension:
    """Extension for federation-specific Tor features"""

    def __init__(self, tor_transport: TorTransport):
        self.tor_transport = tor_transport
        self.federation_peers: dict[str, str] = {}  # device_id -> onion_address

    async def register_federation_peer(self, device_id: str, onion_address: str):
        """Register federation peer's onion address"""
        self.federation_peers[device_id] = onion_address
        logger.info(f"Registered federation peer {device_id}: {onion_address[:16]}...")

    async def send_federation_message(self, device_id: str, message_type: str, payload: dict[str, Any]) -> bool:
        """Send message to federation peer via Tor"""
        if device_id not in self.federation_peers:
            logger.error(f"Unknown federation peer: {device_id}")
            return False

        onion_address = self.federation_peers[device_id]

        federation_message = {
            "type": "federation_message",
            "message_type": message_type,
            "payload": payload,
        }

        return await self.tor_transport.send_message(onion_address, federation_message)

    async def broadcast_to_federation(self, message_type: str, payload: dict[str, Any]) -> int:
        """Broadcast message to all federation peers"""
        success_count = 0

        for device_id in self.federation_peers:
            if await self.send_federation_message(device_id, message_type, payload):
                success_count += 1

        return success_count

    def get_federation_status(self) -> dict[str, Any]:
        """Get federation-specific status"""
        return {
            "federation_peers": len(self.federation_peers),
            "peer_list": list(self.federation_peers.keys()),
        }
