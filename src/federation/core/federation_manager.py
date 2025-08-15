"""Federation Manager - Builds upon existing dual-path implementation

Extends the existing BitChat/Betanet dual-path transport to create a full
federated network with device roles, multi-protocol support, and VPN-like
privacy features.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from navigator.path_policy import RoutingPriority

# Import existing dual-path components
from core.p2p.dual_path_transport import DualPathTransport

# Import our new federation components
from .device_registry import DeviceCapability, DeviceProfile, DeviceRegistry, DeviceRole

logger = logging.getLogger(__name__)


class PrivacyLevel(int):
    """Privacy levels for VPN-like functionality"""

    PUBLIC = 0  # Basic TLS encryption only
    PRIVATE = 1  # End-to-end encryption
    ANONYMOUS = 2  # Onion routing through 3+ hops
    PARANOID = 3  # Chained protocols with dummy traffic


class FederationManager:
    """Federation Manager - Coordinates the entire federated network

    Builds upon existing dual-path transport to provide:
    - Device federation with 5 roles (Beacon, Worker, Relay, Storage, Edge)
    - Multi-protocol support (BitChat, Betanet, Tor, I2P)
    - VPN-like privacy levels (0-3)
    - Anonymous credentials and reputation
    - Intelligent load balancing
    """

    def __init__(
        self,
        device_id: str = None,
        region: str = "unknown",
        enable_tor: bool = False,
        enable_i2p: bool = False,
    ):
        self.device_id = device_id or f"fed_{uuid.uuid4().hex[:12]}"
        self.region = region

        # Core components (building on existing implementation)
        self.device_registry = DeviceRegistry(self.device_id)
        self.dual_path_transport: DualPathTransport | None = None

        # Extended protocol support
        self.tor_transport: Any | None = None  # Will be implemented
        self.i2p_transport: Any | None = None  # Will be implemented
        self.enable_tor = enable_tor
        self.enable_i2p = enable_i2p

        # Protocol state tracking
        self.tor_enabled = False
        self.i2p_enabled = False
        self.tor_onion_address: str | None = None
        self.i2p_destination: str | None = None

        # Federation state
        self.is_running = False
        self.local_profile: DeviceProfile | None = None
        # Basic device profile placeholder until registry initialization
        self.device_profile: dict[str, Any] = {"device_id": self.device_id}
        self.federation_role: DeviceRole | None = None
        # Agent coordination and lookup
        self.agent_registry: Any | None = None

        # VPN-like privacy tunnels
        self.active_tunnels: dict[str, dict[str, Any]] = {}
        self.privacy_circuits: dict[str, list[str]] = {}  # destination -> path

        # Beacon coordination (for beacon nodes)
        self.coordinated_devices: set[str] = set()
        self.beacon_responsibilities: dict[str, Any] = {}

        # Load balancing and task distribution
        self.task_queue: list[dict[str, Any]] = []
        self.worker_pool: set[str] = set()

        # Anonymous credentials and reputation
        self.reputation_proofs: dict[str, bytes] = {}
        self.credential_store: dict[str, Any] = {}
        self.peer_reputation: dict[str, float] = {}

        # Statistics
        self.federation_stats = {
            "total_devices": 0,
            "messages_routed": 0,
            "tasks_distributed": 0,
            "data_stored_gb": 0.0,
            "compute_contributed_hours": 0.0,
            "privacy_tunnels_active": 0,
        }

        logger.info(
            f"FederationManager initialized: {self.device_id} (region: {region})"
        )

    async def start(
        self,
        capabilities: set[DeviceCapability] = None,
        preferred_role: DeviceRole | None = None,
    ) -> bool:
        """Start the federated network stack"""
        if self.is_running:
            return True

        logger.info("Starting Federation Manager...")

        try:
            # Step 1: Initialize device registry and determine our role
            if not capabilities:
                capabilities = {DeviceCapability.BLUETOOTH, DeviceCapability.WIFI}

            self.local_profile = await self.device_registry.initialize_local_device(
                capabilities, self.region
            )
            self.federation_role = self.local_profile.role

            logger.info(f"Device role determined: {self.federation_role.value}")

            # Step 2: Start core dual-path transport (building on existing)
            self.dual_path_transport = DualPathTransport(
                node_id=self.device_id, enable_bitchat=True, enable_betanet=True
            )

            # Enhanced navigator configuration for federation
            if (
                hasattr(self.dual_path_transport, "navigator")
                and self.dual_path_transport.navigator
            ):
                # Configure for federation priorities
                if self.federation_role == DeviceRole.BEACON:
                    self.dual_path_transport.navigator.set_routing_priority(
                        RoutingPriority.PERFORMANCE_FIRST
                    )
                elif (
                    self.local_profile.battery_percent
                    and self.local_profile.battery_percent < 30
                ):
                    self.dual_path_transport.navigator.set_routing_priority(
                        RoutingPriority.OFFLINE_FIRST
                    )

            # Start dual-path transport
            if not await self.dual_path_transport.start():
                logger.error("Failed to start dual-path transport")
                return False

            # Step 3: Register P2P message handlers for federation
            self.dual_path_transport.register_message_handler(
                "federation", self._handle_federation_message
            )

            # Step 4: Start extended protocols if enabled
            if self.enable_tor:
                await self._start_tor_transport()

            if self.enable_i2p:
                await self._start_i2p_transport()

            # Step 5: Start role-specific services
            if self.federation_role == DeviceRole.BEACON:
                await self._start_beacon_services()
            elif self.federation_role == DeviceRole.WORKER:
                await self._start_worker_services()
            elif self.federation_role == DeviceRole.RELAY:
                await self._start_relay_services()
            elif self.federation_role == DeviceRole.STORAGE:
                await self._start_storage_services()
            elif self.federation_role == DeviceRole.EDGE:
                await self._start_edge_services()

            # Step 6: Start federation coordination tasks
            asyncio.create_task(self._federation_heartbeat_loop())
            asyncio.create_task(self._device_discovery_loop())
            asyncio.create_task(self._load_balancing_loop())

            self.is_running = True
            logger.info(
                f"Federation Manager started successfully as {self.federation_role.value}"
            )

            # Step 7: Announce presence to federation
            await self._announce_to_federation()

            return True

        except Exception as e:
            logger.exception(f"Failed to start Federation Manager: {e}")
            return False

    async def stop(self):
        """Stop the federated network stack"""
        logger.info("Stopping Federation Manager...")
        self.is_running = False

        # Stop dual-path transport
        if self.dual_path_transport:
            await self.dual_path_transport.stop()

        # Stop extended transports
        if self.tor_transport:
            await self._stop_tor_transport()

        if self.i2p_transport:
            await self._stop_i2p_transport()

        logger.info("Federation Manager stopped")

    async def send_federated_message(
        self,
        recipient: str,
        payload: dict[str, Any],
        privacy_level: int = PrivacyLevel.PRIVATE,
        service_type: str = "general",
    ) -> bool:
        """Send message through federated network with privacy levels"""
        if not self.is_running or not self.dual_path_transport:
            return False

        # Create federation message wrapper
        federation_msg = {
            "type": "federated_message",
            "service_type": service_type,
            "sender_role": self.federation_role.value
            if self.federation_role
            else "unknown",
            "privacy_level": privacy_level,
            "timestamp": time.time(),
            "payload": payload,
        }

        # Apply privacy level routing
        if privacy_level >= PrivacyLevel.ANONYMOUS:
            # Use privacy circuit routing
            success = await self._send_via_privacy_circuit(recipient, federation_msg)
        else:
            # Use standard dual-path routing
            success = await self.dual_path_transport.send_message(
                recipient=recipient,
                payload=federation_msg,
                privacy_required=privacy_level >= PrivacyLevel.PRIVATE,
            )

        if success:
            self.federation_stats["messages_routed"] += 1

        return success

    async def request_ai_service(
        self,
        service_name: str,
        request_data: dict[str, Any],
        privacy_level: int = PrivacyLevel.PRIVATE,
    ) -> dict[str, Any] | None:
        """Request AI service from federation with privacy guarantees"""
        # Find suitable edge or worker nodes for the service
        suitable_nodes = await self._find_service_providers(service_name)

        if not suitable_nodes:
            logger.warning(f"No providers found for service: {service_name}")
            return None

        # Select best node based on load, proximity, and privacy requirements
        selected_node = await self._select_optimal_service_node(
            suitable_nodes, privacy_level
        )

        # Create service request
        service_request = {
            "service_name": service_name,
            "request_data": request_data,
            "response_required": True,
            "request_id": str(uuid.uuid4()),
        }

        # Send request with appropriate privacy level
        success = await self.send_federated_message(
            recipient=selected_node,
            payload=service_request,
            privacy_level=privacy_level,
            service_type="ai_service_request",
        )

        if not success:
            return None

        # Wait for response with correlation tracking
        request_id = message.get("id", "unknown")
        response = await self._wait_for_correlated_response(request_id, timeout=5.0)
        if not response:
            await asyncio.sleep(1)  # Fallback timeout for testing

        return {"status": "simulated_response", "data": "placeholder"}

    async def contribute_compute_task(self, task_data: dict[str, Any]) -> bool:
        """Contribute compute task to federation"""
        if self.federation_role not in [DeviceRole.WORKER, DeviceRole.EDGE]:
            return False

        # Add task to local queue
        self.task_queue.append(
            {
                "task_id": str(uuid.uuid4()),
                "task_data": task_data,
                "received_at": time.time(),
                "status": "queued",
            }
        )

        logger.info(f"Accepted compute task: {len(self.task_queue)} tasks in queue")
        return True

    async def create_privacy_tunnel(
        self, destination: str, privacy_level: int = PrivacyLevel.ANONYMOUS
    ) -> str | None:
        """Create VPN-like privacy tunnel through federation"""
        if privacy_level < PrivacyLevel.ANONYMOUS:
            return None  # No tunnel needed for lower privacy levels

        # Build privacy circuit based on level
        if privacy_level == PrivacyLevel.ANONYMOUS:
            # 3-hop circuit through different protocols
            circuit_path = await self._build_privacy_circuit(destination, min_hops=3)
        else:  # PARANOID
            # Multi-protocol chained circuit
            circuit_path = await self._build_paranoid_circuit(destination)

        if not circuit_path:
            return None

        # Create tunnel identifier
        tunnel_id = str(uuid.uuid4())

        # Store tunnel information
        self.active_tunnels[tunnel_id] = {
            "destination": destination,
            "circuit_path": circuit_path,
            "privacy_level": privacy_level,
            "created_at": time.time(),
            "last_used": time.time(),
        }

        self.federation_stats["privacy_tunnels_active"] = len(self.active_tunnels)

        logger.info(
            f"Created privacy tunnel {tunnel_id[:8]} with {len(circuit_path)} hops"
        )
        return tunnel_id

    def get_federation_status(self) -> dict[str, Any]:
        """Get comprehensive federation status"""
        status = self.device_registry.get_federation_status()

        # Add federation-specific information
        status.update(
            {
                "federation_role": self.federation_role.value
                if self.federation_role
                else None,
                "privacy_tunnels": len(self.active_tunnels),
                "task_queue_size": len(self.task_queue),
                "coordinated_devices": len(self.coordinated_devices),
                "protocols_available": self._get_available_protocols(),
                "federation_stats": self.federation_stats.copy(),
            }
        )

        # Add transport status
        if self.dual_path_transport:
            status["dual_path_status"] = self.dual_path_transport.get_status()

        return status

    async def _handle_federation_message(self, dual_path_msg, source_protocol: str):
        """Handle incoming federation messages"""
        try:
            # Parse federation message
            if isinstance(dual_path_msg.payload, bytes):
                message_data = json.loads(dual_path_msg.payload.decode())
            else:
                message_data = dual_path_msg.payload

            msg_type = message_data.get("type", "unknown")
            sender = dual_path_msg.sender

            logger.debug(f"Received federation message: {msg_type} from {sender}")

            # Route to appropriate handler
            if msg_type == "device_announcement":
                await self._handle_device_announcement(message_data, sender)
            elif msg_type == "ai_service_request":
                await self._handle_ai_service_request(message_data, sender)
            elif msg_type == "compute_task":
                await self._handle_compute_task(message_data, sender)
            elif msg_type == "beacon_coordination":
                await self._handle_beacon_coordination(message_data, sender)
            elif msg_type == "reputation_update":
                await self._handle_reputation_update(message_data, sender)
            else:
                logger.warning(f"Unknown federation message type: {msg_type}")

        except Exception as e:
            logger.exception(f"Error handling federation message: {e}")

    async def _handle_device_announcement(self, message_data: dict, sender: str):
        """Handle device announcement for federation discovery"""
        try:
            # Extract device profile from announcement
            device_info = message_data.get("device_info", {})

            # Create or update device profile
            # (This would normally involve cryptographic verification)

            logger.info(
                f"Device announcement from {sender}: role={device_info.get('role')}"
            )

        except Exception as e:
            logger.error(f"Error handling device announcement: {e}")

    async def _handle_ai_service_request(self, message_data: dict, sender: str):
        """Handle AI service request"""
        if self.federation_role not in [DeviceRole.EDGE, DeviceRole.WORKER]:
            return  # Only edge/worker nodes handle AI services

        service_name = message_data.get("service_name")
        request_data = message_data.get("request_data")
        request_id = message_data.get("request_id")

        logger.info(f"AI service request: {service_name} from {sender}")

        # Process the AI service request by routing to appropriate agent
        try:
            # Map service names to agent types
            service_to_agent = {
                "translation": "translator",
                "data_analysis": "data_science",
                "testing": "tester",
                "social_analysis": "social",
                "creative_generation": "creative",
                "devops_task": "devops",
                "financial_analysis": "financial",
            }

            result = {"error": "Service not available"}

            # Route to appropriate agent if service is supported
            agent_type = service_to_agent.get(service_name)
            if agent_type and hasattr(self, "agent_registry"):
                try:
                    agent = await self.agent_registry.get_agent(agent_type)
                    if agent:
                        # Process request with the agent
                        prompt = request_data.get("prompt", "")
                        if prompt:
                            processed_by = (
                                self.local_profile.identity.device_id
                                if self.local_profile
                                else self.device_id
                            )
                            result = {
                                "response": await agent.generate(prompt),
                                "agent_type": agent_type,
                                "processed_by": processed_by,
                            }
                        else:
                            result = {"error": "No prompt provided for AI service"}
                except Exception as e:
                    logger.exception(
                        f"Error processing AI service with agent {agent_type}: {e}"
                    )
                    result = {"error": f"Failed to process request: {e!s}"}
            else:
                logger.warning(f"Unsupported service: {service_name}")
                result = {"error": f"Unsupported service: {service_name}"}

        except Exception as e:
            logger.exception(f"Error in AI service processing: {e}")
            result = {"error": f"Processing failed: {e!s}"}

        response = {
            "type": "ai_service_response",
            "request_id": request_id,
            "status": "processed" if "error" not in result else "failed",
            "result": result,
            "timestamp": asyncio.get_event_loop().time(),
        }

        await self.send_federated_message(sender, response)

    async def _handle_compute_task(self, message_data: dict, sender: str):
        """Handle distributed compute task"""
        await self.contribute_compute_task(message_data.get("task_data", {}))

    async def _handle_beacon_coordination(self, message_data: dict, sender: str):
        """Handle beacon coordination messages"""
        if self.federation_role == DeviceRole.BEACON:
            # Handle coordination between beacons
            coordination_type = message_data.get("coordination_type")
            logger.debug(f"Beacon coordination: {coordination_type} from {sender}")

    async def _handle_reputation_update(self, message_data: dict, sender: str):
        """Handle reputation system updates"""
        # Process reputation update with basic verification for now
        # Zero-knowledge proofs would provide stronger privacy guarantees in production
        reputation_score = message_data.get("reputation", 0.0)
        proof_hash = message_data.get("proof_hash", "")

        # Basic verification (would use zk-SNARK proofs in production)
        if 0.0 <= reputation_score <= 1.0 and proof_hash:
            self.peer_reputation[sender] = reputation_score
            logger.debug(f"Updated reputation for {sender}: {reputation_score}")
        else:
            logger.warning(f"Invalid reputation update from {sender}")

    async def _announce_to_federation(self):
        """Announce device presence to federation"""
        announcement = {
            "type": "device_announcement",
            "device_info": {
                "device_id": self.device_id,
                "role": self.federation_role.value,
                "capabilities": [cap.value for cap in self.local_profile.capabilities],
                "protocols": list(self.local_profile.protocols),
                "region": self.region,
                "timestamp": time.time(),
            },
        }

        # Broadcast to federation
        await self.dual_path_transport.broadcast_message(
            payload=announcement, priority=6
        )

        logger.info("Announced presence to federation")

    async def _federation_heartbeat_loop(self):
        """Send periodic federation heartbeats"""
        while self.is_running:
            try:
                await self._announce_to_federation()
                await asyncio.sleep(300)  # 5 minutes
            except Exception as e:
                logger.error(f"Federation heartbeat error: {e}")
                await asyncio.sleep(60)

    async def _device_discovery_loop(self):
        """Discover and register federation devices"""
        while self.is_running:
            try:
                # Update federation statistics
                self._update_federation_stats()

                # Clean up stale devices
                await self.device_registry.cleanup_stale_devices()

                await asyncio.sleep(180)  # 3 minutes
            except Exception as e:
                logger.error(f"Device discovery error: {e}")
                await asyncio.sleep(60)

    async def _load_balancing_loop(self):
        """Distribute load across federation"""
        while self.is_running:
            try:
                if self.federation_role == DeviceRole.BEACON:
                    await self._coordinate_load_balancing()

                await asyncio.sleep(120)  # 2 minutes
            except Exception as e:
                logger.error(f"Load balancing error: {e}")
                await asyncio.sleep(60)

    async def _coordinate_load_balancing(self):
        """Coordinate load balancing as beacon node"""
        # Get worker nodes in our region
        workers = self.device_registry.get_devices_by_role(DeviceRole.WORKER)
        regional_workers = [w for w in workers if w.region == self.region]

        if len(regional_workers) > 1:
            # Implement simple round-robin task distribution
            logger.debug(
                f"Coordinating {len(regional_workers)} workers in region {self.region}"
            )

    async def _find_service_providers(self, service_name: str) -> list[str]:
        """Find nodes capable of providing specific AI service"""
        # For now, return edge and worker nodes
        providers = []

        for role in [DeviceRole.EDGE, DeviceRole.WORKER]:
            devices = self.device_registry.get_devices_by_role(role)
            providers.extend([d.identity.device_id for d in devices])

        return providers[:5]  # Limit to 5 candidates

    async def _select_optimal_service_node(
        self, candidates: list[str], privacy_level: int
    ) -> str:
        """Select optimal node for service request"""
        if not candidates:
            return None

        # Simple selection - prefer nodes in same region
        regional_candidates = []
        for candidate in candidates:
            if candidate in self.device_registry.devices:
                device = self.device_registry.devices[candidate]
                if device.region == self.region:
                    regional_candidates.append(candidate)

        if regional_candidates:
            return regional_candidates[0]

        return candidates[0]

    async def _build_privacy_circuit(
        self, destination: str, min_hops: int = 3
    ) -> list[str] | None:
        """Build privacy circuit through relay nodes"""
        relay_nodes = self.device_registry.get_devices_by_role(DeviceRole.RELAY)

        if len(relay_nodes) < min_hops - 1:
            logger.warning(
                f"Insufficient relay nodes for privacy circuit: need {min_hops - 1}, have {len(relay_nodes)}"
            )
            return None

        # Select random relay nodes for circuit
        import random

        selected_relays = random.sample(
            [r.identity.device_id for r in relay_nodes],
            min(min_hops - 1, len(relay_nodes)),
        )

        # Circuit: source -> relays -> destination
        circuit = selected_relays + [destination]
        return circuit

    async def _build_paranoid_circuit(self, destination: str) -> list[str] | None:
        """Build multi-protocol paranoid circuit"""
        # For PARANOID level, chain different protocols
        # This would involve Tor -> Betanet -> I2P routing
        # For now, return a basic circuit
        return await self._build_privacy_circuit(destination, min_hops=5)

    async def _send_via_privacy_circuit(self, destination: str, message: dict) -> bool:
        """Send message through privacy circuit"""
        # Create tunnel if needed
        tunnel_id = await self.create_privacy_tunnel(
            destination, PrivacyLevel.ANONYMOUS
        )

        if not tunnel_id:
            return False

        tunnel = self.active_tunnels[tunnel_id]
        circuit_path = tunnel["circuit_path"]

        # For now, just send through regular dual-path (would implement onion routing)
        return await self.dual_path_transport.send_message(
            recipient=circuit_path[0],  # First hop
            payload=message,
            privacy_required=True,
        )

    def _update_federation_stats(self):
        """Update federation statistics"""
        self.federation_stats["total_devices"] = len(self.device_registry.devices)

    def _get_available_protocols(self) -> list[str]:
        """Get list of available protocols"""
        protocols = ["bitchat", "betanet"]

        if self.enable_tor:
            protocols.append("tor")

        if self.enable_i2p:
            protocols.append("i2p")

        return protocols

    # Role-specific service implementations
    async def _start_beacon_services(self):
        """Start beacon node services"""
        logger.info("Starting beacon node services...")
        # Beacon nodes coordinate regional devices
        self.beacon_responsibilities = {
            "device_coordination": True,
            "load_balancing": True,
            "consensus_participation": True,
        }

    async def _start_worker_services(self):
        """Start worker node services"""
        logger.info("Starting worker node services...")
        # Worker nodes process AI tasks

    async def _start_relay_services(self):
        """Start relay node services"""
        logger.info("Starting relay node services...")
        # Relay nodes forward messages and maintain routing

    async def _start_storage_services(self):
        """Start storage node services"""
        logger.info("Starting storage node services...")
        # Storage nodes provide distributed storage

    async def _start_edge_services(self):
        """Start edge node services"""
        logger.info("Starting edge node services...")
        # Edge nodes serve local AI services

    # Extended protocol implementations (placeholders)
    async def _start_tor_transport(self):
        """Start Tor hidden service transport"""
        logger.info("Starting Tor transport...")

        if not self.enable_tor:
            logger.info("Tor transport disabled")
            return

        try:
            # Try to import Tor dependencies
            import socket

            import socks
            from stem import Signal
            from stem.control import Controller

            # Configure SOCKS proxy for Tor
            socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 9050)
            socket.socket = socks.socksocket

            # Connect to Tor control port
            with Controller.from_port(port=9051) as controller:
                controller.authenticate()

                # Create hidden service if needed
                response = controller.create_ephemeral_hidden_service(
                    {8080: 8080}, await_publication=True
                )
                self.tor_onion_address = f"{response.service_id}.onion"

                logger.info(
                    f"Tor hidden service available at: {self.tor_onion_address}"
                )
                self.tor_enabled = True

        except ImportError:
            logger.warning(
                "Tor dependencies not available (stem, PySocks). Install with: pip install stem PySocks"
            )
            self.tor_enabled = False
        except Exception as e:
            logger.error(f"Failed to start Tor transport: {e}")
            logger.info("Ensure Tor is running: systemctl start tor")
            self.tor_enabled = False

    async def _stop_tor_transport(self):
        """Stop Tor transport"""
        logger.info("Stopping Tor transport...")

    async def _start_i2p_transport(self):
        """Start I2P transport"""
        logger.info("Starting I2P transport...")

        if not self.enable_i2p:
            logger.info("I2P transport disabled")
            return

        try:
            # I2P integration via SAM (Simple Anonymous Messaging) bridge
            import asyncio

            import aiohttp

            # Connect to I2P SAM bridge (default port 7656)
            sam_host = "127.0.0.1"
            sam_port = 7656

            # Create I2P destination (like an address)
            reader, writer = await asyncio.open_connection(sam_host, sam_port)

            # SAM protocol handshake
            writer.write(b"HELLO VERSION MIN=3.0 MAX=3.0\n")
            await writer.drain()

            response = await reader.readline()
            if b"HELLO REPLY RESULT=OK" in response:
                # Generate new destination keypair
                writer.write(b"DEST GENERATE\n")
                await writer.drain()

                dest_response = await reader.readline()
                if b"DEST REPLY" in dest_response:
                    # Extract destination from response
                    dest_parts = dest_response.decode().split(" ")
                    for part in dest_parts:
                        if (
                            "=" in part and len(part) > 50
                        ):  # I2P destinations are long base64 strings
                            self.i2p_destination = part.split("=")[1]
                            break

                    logger.info(
                        f"I2P destination created: {self.i2p_destination[:32]}..."
                    )
                    self.i2p_enabled = True
                else:
                    raise Exception("Failed to create I2P destination")
            else:
                raise Exception("I2P SAM handshake failed")

            writer.close()
            await writer.wait_closed()

        except ImportError:
            logger.warning(
                "I2P dependencies not available. Ensure aiohttp is installed"
            )
            self.i2p_enabled = False
        except (ConnectionRefusedError, OSError):
            logger.error(f"Cannot connect to I2P SAM bridge on {sam_host}:{sam_port}")
            logger.info("Ensure I2P router is running with SAM bridge enabled")
            self.i2p_enabled = False
        except Exception as e:
            logger.error(f"Failed to start I2P transport: {e}")
            self.i2p_enabled = False

    async def _stop_i2p_transport(self):
        """Stop I2P transport"""
        logger.info("Stopping I2P transport...")
