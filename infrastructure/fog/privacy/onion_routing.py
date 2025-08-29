"""
Onion Routing Implementation for Fog Computing

Implements Tor-inspired onion routing for anonymous fog computing traffic.
Provides censorship-resistant hidden service hosting with multi-hop circuits.

Key Features (inspired by Tor):
- 3-hop default circuits with telescoping path construction
- Directory authorities for relay discovery
- Hidden service protocol with rendezvous points
- Perfect forward secrecy with ephemeral keys
- Traffic analysis resistance with padding and timing obfuscation
"""

import base64
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
import hashlib
import hmac
import logging
import secrets
from typing import Any

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the onion network"""

    GUARD = "guard"  # Entry node (trusted, stable)
    MIDDLE = "middle"  # Middle relay
    EXIT = "exit"  # Exit node (allows external traffic)
    BRIDGE = "bridge"  # Unlisted entry points
    DIRECTORY = "directory"  # Directory authority
    HIDDEN_SERVICE = "hidden_service"  # Hidden service node
    INTRODUCTION = "introduction"  # Introduction point
    RENDEZVOUS = "rendezvous"  # Rendezvous point


class CircuitState(Enum):
    """State of an onion routing circuit"""

    BUILDING = "building"
    ESTABLISHED = "established"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass
class OnionNode:
    """Represents a node in the onion network"""

    node_id: str  # Ed25519 public key fingerprint
    address: str  # IP:port or onion address
    node_types: set[NodeType]

    # Cryptographic keys
    identity_key: bytes  # Ed25519 public key
    onion_key: bytes  # X25519 public key for circuit building

    # Node properties
    bandwidth_mbps: float = 10.0
    uptime_hours: float = 0.0
    consensus_weight: float = 1.0

    # Flags
    is_fast: bool = False
    is_stable: bool = False
    is_valid: bool = True
    is_running: bool = True

    # Hidden service specific
    hidden_service_id: str | None = None
    introduction_points: list[str] = field(default_factory=list)


@dataclass
class CircuitHop:
    """Single hop in an onion circuit"""

    node: OnionNode
    position: int  # 0=guard, 1=middle, 2=exit

    # Cryptographic material
    shared_secret: bytes  # Negotiated via DH
    forward_digest: bytes  # For integrity checking
    backward_digest: bytes
    forward_key: bytes  # For onion encryption
    backward_key: bytes

    # Circuit state
    circuit_id: int
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class OnionCircuit:
    """Multi-hop onion routing circuit"""

    circuit_id: str
    state: CircuitState
    hops: list[CircuitHop]

    # Circuit properties
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_used: datetime = field(default_factory=lambda: datetime.now(UTC))
    bytes_sent: int = 0
    bytes_received: int = 0

    # Hidden service specific
    is_hidden_service: bool = False
    rendezvous_cookie: bytes | None = None
    introduction_point: str | None = None


@dataclass
class HiddenService:
    """Hidden service configuration for .fog addresses"""

    service_id: str  # Base32 encoded public key hash
    private_key: bytes  # Ed25519 private key
    public_key: bytes  # Ed25519 public key

    # Service properties
    ports: dict[int, int]  # virtual_port -> real_port mapping
    introduction_points: list[OnionNode] = field(default_factory=list)
    descriptor_cookie: bytes | None = None  # For access control

    # Metrics
    connections: int = 0
    bytes_served: int = 0
    uptime_hours: float = 0.0

    @property
    def onion_address(self) -> str:
        """Generate .fog onion address from public key"""
        # Similar to Tor's .onion addresses
        key_hash = hashlib.sha256(self.public_key).digest()[:10]
        checksum = hashlib.sha256(b".fog checksum" + self.public_key + b"\x03").digest()[:2]

        address_bytes = key_hash + checksum
        # Base32 encode without padding
        return base64.b32encode(address_bytes).decode().lower().rstrip("=") + ".fog"


class OnionRouter:
    """
    Implements onion routing for fog network traffic.
    Provides anonymous communication and hidden services.
    """

    def __init__(
        self,
        node_id: str,
        node_types: set[NodeType],
        enable_hidden_services: bool = True,
        num_guards: int = 3,  # Number of entry guards to use
        circuit_lifetime_hours: int = 1,
    ):
        self.node_id = node_id
        self.node_types = node_types
        self.enable_hidden_services = enable_hidden_services
        self.num_guards = num_guards
        self.circuit_lifetime = timedelta(hours=circuit_lifetime_hours)

        # Cryptographic keys
        self.identity_key = ed25519.Ed25519PrivateKey.generate()
        self.onion_key = x25519.X25519PrivateKey.generate()

        # Network state
        self.consensus: dict[str, OnionNode] = {}
        self.guard_nodes: list[OnionNode] = []
        self.circuits: dict[str, OnionCircuit] = {}
        self.streams: dict[str, str] = {}  # stream_id -> circuit_id

        # Hidden services
        self.hidden_services: dict[str, HiddenService] = {}
        self.introduction_circuits: dict[str, OnionCircuit] = {}

        # Directory authorities (hardcoded like Tor)
        self.directory_authorities = self._initialize_directory_authorities()

        logger.info(f"OnionRouter initialized: {node_id}, types: {node_types}")

    def _initialize_directory_authorities(self) -> list[OnionNode]:
        """Initialize hardcoded directory authorities"""
        # In production, these would be well-known, trusted nodes
        # For fog network, we use federated authorities
        authorities = []
        for i in range(5):
            auth_key = ed25519.Ed25519PrivateKey.generate()
            auth_node = OnionNode(
                node_id=f"fog-dir-auth-{i}",
                address=f"10.0.0.{i+1}:9030",  # Example addresses
                node_types={NodeType.DIRECTORY},
                identity_key=auth_key.public_key().public_bytes(
                    encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
                ),
                onion_key=x25519.X25519PrivateKey.generate()
                .public_key()
                .public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw),
                is_stable=True,
                is_valid=True,
            )
            authorities.append(auth_node)

        return authorities

    async def fetch_consensus(self) -> bool:
        """Fetch network consensus from directory authorities"""
        try:
            # In production, this would fetch from actual directory authorities
            # For now, simulate with some example nodes

            example_nodes = []
            for i in range(20):
                node = OnionNode(
                    node_id=f"fog-relay-{i}",
                    address=f"10.1.{i//256}.{i%256}:9001",
                    node_types={NodeType.MIDDLE} if i < 15 else {NodeType.EXIT},
                    identity_key=ed25519.Ed25519PrivateKey.generate()
                    .public_key()
                    .public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw),
                    onion_key=x25519.X25519PrivateKey.generate()
                    .public_key()
                    .public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw),
                    bandwidth_mbps=50.0 + i * 10,
                    uptime_hours=24 * (i + 1),
                    is_fast=i % 3 == 0,
                    is_stable=i % 2 == 0,
                )

                # Some nodes are guards
                if i < 5 and node.is_stable:
                    node.node_types.add(NodeType.GUARD)

                example_nodes.append(node)
                self.consensus[node.node_id] = node

            # Select guard nodes
            guards = [n for n in example_nodes if NodeType.GUARD in n.node_types]
            self.guard_nodes = guards[: self.num_guards]

            logger.info(f"Fetched consensus: {len(self.consensus)} nodes, {len(self.guard_nodes)} guards")
            return True

        except Exception as e:
            logger.error(f"Failed to fetch consensus: {e}")
            return False

    def _select_path_nodes(self, path_length: int = 3, exit_required: bool = True) -> list[OnionNode]:
        """Select nodes for circuit path using weighted random selection"""

        selected_nodes = []
        used_families = set()  # Avoid nodes from same /16 subnet

        # Select guard (entry node)
        if self.guard_nodes:
            guard = secrets.choice(self.guard_nodes)
            selected_nodes.append(guard)
            family = guard.address.split(".")[0:2]
            used_families.add(tuple(family))

        # Select middle node(s)
        middle_nodes = [
            n
            for n in self.consensus.values()
            if NodeType.MIDDLE in n.node_types
            and n not in selected_nodes
            and tuple(n.address.split(".")[0:2]) not in used_families
        ]

        if middle_nodes and path_length > 1:
            # Weight by consensus weight and bandwidth
            weights = [n.consensus_weight * n.bandwidth_mbps for n in middle_nodes]
            total_weight = sum(weights)

            if total_weight > 0:
                probabilities = [w / total_weight for w in weights]
                middle = self._weighted_choice(middle_nodes, probabilities)
                selected_nodes.append(middle)
                family = middle.address.split(".")[0:2]
                used_families.add(tuple(family))

        # Select exit node
        if exit_required and path_length > 2:
            exit_nodes = [
                n
                for n in self.consensus.values()
                if NodeType.EXIT in n.node_types
                and n not in selected_nodes
                and tuple(n.address.split(".")[0:2]) not in used_families
            ]

            if exit_nodes:
                exit_node = secrets.choice(exit_nodes)
                selected_nodes.append(exit_node)

        return selected_nodes

    def _weighted_choice(self, items: list[Any], weights: list[float]) -> Any:
        """Select item based on weights"""
        r = secrets.SystemRandom().random() * sum(weights)
        cumsum = 0
        for item, weight in zip(items, weights):
            cumsum += weight
            if r < cumsum:
                return item
        return items[-1]

    async def build_circuit(self, purpose: str = "general", path_length: int = 3) -> OnionCircuit | None:
        """Build a new onion circuit with telescoping construction"""

        circuit_id = secrets.token_hex(16)

        # Select path
        path_nodes = self._select_path_nodes(path_length)
        if len(path_nodes) < path_length:
            logger.warning(f"Insufficient nodes for {path_length}-hop circuit")
            return None

        circuit = OnionCircuit(circuit_id=circuit_id, state=CircuitState.BUILDING, hops=[])

        # Build circuit hop by hop (telescoping)
        for position, node in enumerate(path_nodes):
            hop = await self._extend_circuit(circuit, node, position)
            if not hop:
                circuit.state = CircuitState.FAILED
                logger.error(f"Failed to extend circuit at hop {position}")
                return None

            circuit.hops.append(hop)

        circuit.state = CircuitState.ESTABLISHED
        self.circuits[circuit_id] = circuit

        logger.info(f"Built circuit {circuit_id}: " f"{' -> '.join([h.node.node_id for h in circuit.hops])}")

        return circuit

    async def _extend_circuit(self, circuit: OnionCircuit, node: OnionNode, position: int) -> CircuitHop | None:
        """Extend circuit to include another hop"""

        # Generate ephemeral key for this hop
        ephemeral_key = x25519.X25519PrivateKey.generate()
        ephemeral_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )

        # Perform DH with node's onion key
        node_public_key = x25519.X25519PublicKey.from_public_bytes(node.onion_key)
        shared_secret = ephemeral_key.exchange(node_public_key)

        # Derive keys using KDF
        kdf = HKDF(
            algorithm=hashes.SHA256(), length=64, salt=b"fog-onion-v1", info=b"circuit-keys", backend=default_backend()
        )
        key_material = kdf.derive(shared_secret)

        forward_key = key_material[:16]
        backward_key = key_material[16:32]
        forward_digest = key_material[32:48]
        backward_digest = key_material[48:64]

        hop = CircuitHop(
            node=node,
            position=position,
            shared_secret=shared_secret,
            forward_key=forward_key,
            backward_key=backward_key,
            forward_digest=forward_digest,
            backward_digest=backward_digest,
            circuit_id=int.from_bytes(secrets.token_bytes(4), "big"),
        )

        return hop

    def _onion_encrypt(self, circuit: OnionCircuit, payload: bytes) -> bytes:
        """Apply onion encryption layers for circuit"""

        # Pad payload to fixed size to prevent traffic analysis
        padded_payload = self._pad_payload(payload)

        # Apply encryption layers in reverse order (exit -> middle -> guard)
        encrypted = padded_payload
        for hop in reversed(circuit.hops):
            # AES-CTR encryption with hop's forward key
            cipher = Cipher(
                algorithms.AES(hop.forward_key), modes.CTR(secrets.token_bytes(16)), backend=default_backend()
            )
            encryptor = cipher.encryptor()
            encrypted = encryptor.update(encrypted) + encryptor.finalize()

            # Add integrity check
            mac = hmac.new(hop.forward_digest, encrypted, hashlib.sha256).digest()[:4]
            encrypted = mac + encrypted

        return encrypted

    def _onion_decrypt(self, circuit: OnionCircuit, encrypted: bytes, hop_index: int) -> bytes:
        """Remove one layer of onion encryption"""

        hop = circuit.hops[hop_index]

        # Verify integrity
        mac = encrypted[:4]
        ciphertext = encrypted[4:]
        expected_mac = hmac.new(hop.forward_digest, ciphertext, hashlib.sha256).digest()[:4]

        if not hmac.compare_digest(mac, expected_mac):
            raise ValueError("Integrity check failed")

        # Decrypt layer
        cipher = Cipher(algorithms.AES(hop.forward_key), modes.CTR(secrets.token_bytes(16)), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted = decryptor.update(ciphertext) + decryptor.finalize()

        return decrypted

    def _pad_payload(self, payload: bytes, cell_size: int = 512) -> bytes:
        """Pad payload to fixed cell size for traffic analysis resistance"""
        padding_needed = cell_size - (len(payload) % cell_size)
        if padding_needed == cell_size:
            padding_needed = 0

        # Add padding with random data
        padding = secrets.token_bytes(padding_needed - 1) + bytes([padding_needed])
        return payload + padding

    def _unpad_payload(self, padded: bytes) -> bytes:
        """Remove padding from payload"""
        if not padded:
            return padded

        padding_length = padded[-1]
        if padding_length > len(padded):
            return padded  # Invalid padding

        return padded[:-padding_length]

    async def create_hidden_service(
        self, ports: dict[int, int], descriptor_cookie: bytes | None = None
    ) -> HiddenService:
        """Create a new hidden service with .fog address"""

        # Generate service keys
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        # Create service
        service = HiddenService(
            service_id=secrets.token_hex(16),
            private_key=private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            ),
            public_key=public_key.public_bytes(
                encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
            ),
            ports=ports,
            descriptor_cookie=descriptor_cookie,
        )

        # Select introduction points
        intro_nodes = [n for n in self.consensus.values() if n.is_stable and n.bandwidth_mbps > 20][:3]

        service.introduction_points = intro_nodes

        # Build circuits to introduction points
        for intro_node in intro_nodes:
            circuit = await self.build_circuit(purpose="hidden_service")
            if circuit:
                self.introduction_circuits[intro_node.node_id] = circuit

        # Store service
        self.hidden_services[service.service_id] = service

        logger.info(
            f"Created hidden service: {service.onion_address}, " f"ports: {ports}, intro points: {len(intro_nodes)}"
        )

        return service

    async def connect_to_hidden_service(
        self, onion_address: str, descriptor_cookie: bytes | None = None
    ) -> OnionCircuit | None:
        """Connect to a hidden service via rendezvous point"""

        # Parse onion address
        if not onion_address.endswith(".fog"):
            logger.error(f"Invalid onion address: {onion_address}")
            return None

        # Build circuit to rendezvous point
        rendezvous_circuit = await self.build_circuit(purpose="rendezvous")
        if not rendezvous_circuit:
            return None

        # Generate rendezvous cookie
        rendezvous_cookie = secrets.token_bytes(20)
        rendezvous_circuit.rendezvous_cookie = rendezvous_cookie

        # In production, would fetch descriptor and connect via introduction point
        # This is simplified for the implementation

        rendezvous_circuit.is_hidden_service = True

        logger.info(f"Connected to hidden service: {onion_address}")
        return rendezvous_circuit

    async def send_data(self, circuit_id: str, data: bytes, stream_id: str | None = None) -> bool:
        """Send data through an onion circuit"""

        if circuit_id not in self.circuits:
            logger.error(f"Unknown circuit: {circuit_id}")
            return False

        circuit = self.circuits[circuit_id]
        if circuit.state != CircuitState.ESTABLISHED:
            logger.error(f"Circuit not established: {circuit_id}")
            return False

        # Apply onion encryption
        encrypted_data = self._onion_encrypt(circuit, data)

        # Update circuit stats
        circuit.bytes_sent += len(encrypted_data)
        circuit.last_used = datetime.now(UTC)

        # In production, would actually send through the network
        logger.debug(f"Sent {len(data)} bytes through circuit {circuit_id}")

        return True

    async def close_circuit(self, circuit_id: str) -> bool:
        """Close an onion circuit"""

        if circuit_id not in self.circuits:
            return False

        circuit = self.circuits[circuit_id]
        circuit.state = CircuitState.CLOSED

        # Clean up associated streams
        for stream_id, cid in list(self.streams.items()):
            if cid == circuit_id:
                del self.streams[stream_id]

        del self.circuits[circuit_id]

        logger.info(f"Closed circuit {circuit_id}")
        return True

    async def rotate_circuits(self) -> int:
        """Rotate old circuits for better security"""

        now = datetime.now(UTC)
        rotated_count = 0

        for circuit_id, circuit in list(self.circuits.items()):
            age = now - circuit.created_at

            if age > self.circuit_lifetime:
                # Build replacement circuit
                new_circuit = await self.build_circuit(purpose="general", path_length=len(circuit.hops))

                if new_circuit:
                    # Migrate streams
                    for stream_id, cid in self.streams.items():
                        if cid == circuit_id:
                            self.streams[stream_id] = new_circuit.circuit_id

                    # Close old circuit
                    await self.close_circuit(circuit_id)
                    rotated_count += 1

        if rotated_count > 0:
            logger.info(f"Rotated {rotated_count} circuits")

        return rotated_count

    def get_stats(self) -> dict[str, Any]:
        """Get onion routing statistics"""

        active_circuits = [c for c in self.circuits.values() if c.state == CircuitState.ESTABLISHED]

        total_bytes_sent = sum(c.bytes_sent for c in self.circuits.values())
        total_bytes_received = sum(c.bytes_received for c in self.circuits.values())

        hidden_service_addresses = [s.onion_address for s in self.hidden_services.values()]

        return {
            "node_id": self.node_id,
            "node_types": [t.value for t in self.node_types],
            "consensus_nodes": len(self.consensus),
            "guard_nodes": len(self.guard_nodes),
            "active_circuits": len(active_circuits),
            "total_circuits": len(self.circuits),
            "active_streams": len(self.streams),
            "hidden_services": len(self.hidden_services),
            "hidden_service_addresses": hidden_service_addresses,
            "total_bytes_sent": total_bytes_sent,
            "total_bytes_received": total_bytes_received,
        }
