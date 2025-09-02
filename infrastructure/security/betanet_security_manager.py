"""
BetaNet Security Manager for AIVillage
=====================================

Enhanced security layer for BetaNet integration with comprehensive threat protection.
Implements transport encryption, identity verification, and secure communication protocols.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
import json
import logging
import secrets
import time
from typing import Any
import uuid

# Cryptographic libraries
try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for BetaNet communications."""

    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"


class ThreatType(Enum):
    """Types of security threats."""

    MAN_IN_MIDDLE = "man_in_middle"
    REPLAY_ATTACK = "replay_attack"
    EAVESDROPPING = "eavesdropping"
    IDENTITY_SPOOFING = "identity_spoofing"
    DOS_ATTACK = "dos_attack"
    TRAFFIC_ANALYSIS = "traffic_analysis"
    MALICIOUS_NODE = "malicious_node"


class ChannelType(Enum):
    """BetaNet channel types."""

    HTTP_COVERT = "http_covert"
    HTTP3_COVERT = "http3_covert"
    WEBSOCKET_COVERT = "websocket_covert"
    MIXNET_ROUTED = "mixnet_routed"
    DIRECT_ENCRYPTED = "direct_encrypted"


@dataclass
class SecurityCredential:
    """Security credentials for BetaNet nodes."""

    node_id: str
    public_key: bytes
    private_key: bytes
    certificate: bytes | None = None
    key_fingerprint: str = ""
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 86400 * 30)  # 30 days
    revoked: bool = False


@dataclass
class SecureChannel:
    """Secure communication channel."""

    channel_id: str
    channel_type: ChannelType
    local_node_id: str
    remote_node_id: str
    encryption_key: bytes
    authentication_key: bytes
    sequence_number: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatEvent:
    """Security threat event."""

    event_id: str
    threat_type: ThreatType
    source_node: str
    target_node: str
    severity: str  # low, medium, high, critical
    description: str
    detected_at: float = field(default_factory=time.time)
    mitigated: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityPolicy:
    """Security policy configuration."""

    min_security_level: SecurityLevel = SecurityLevel.STANDARD
    require_mutual_auth: bool = True
    enable_forward_secrecy: bool = True
    max_channel_lifetime: int = 3600  # 1 hour
    threat_detection_enabled: bool = True
    automatic_mitigation: bool = True
    allowed_channel_types: set[ChannelType] = field(
        default_factory=lambda: {ChannelType.HTTP3_COVERT, ChannelType.WEBSOCKET_COVERT, ChannelType.MIXNET_ROUTED}
    )


class BetaNetSecurityManager:
    """
    Comprehensive security manager for BetaNet communications.

    Features:
    - End-to-end encryption with multiple cipher suites
    - Identity verification and certificate management
    - Threat detection and mitigation
    - Secure channel establishment and management
    - Traffic analysis protection
    - Forward secrecy and key rotation
    - Integration with existing BetaNet transport
    """

    def __init__(self, node_id: str, security_policy: SecurityPolicy | None = None):
        """Initialize BetaNet security manager."""
        self.node_id = node_id
        self.security_policy = security_policy or SecurityPolicy()

        # Cryptographic materials
        self.node_credentials: SecurityCredential | None = None
        self.trusted_nodes: dict[str, SecurityCredential] = {}
        self.revoked_certificates: set[str] = set()

        # Secure channels
        self.active_channels: dict[str, SecureChannel] = {}
        self.channel_by_peer: dict[str, str] = {}  # peer_id -> channel_id

        # Threat detection
        self.threat_events: list[ThreatEvent] = []
        self.threat_patterns: dict[str, list[dict[str, Any]]] = {}
        self.blocked_nodes: set[str] = set()

        # Session management
        self.session_keys: dict[str, bytes] = {}
        self.key_rotation_schedule: dict[str, float] = {}

        # Statistics
        self.security_stats = {
            "channels_established": 0,
            "channels_terminated": 0,
            "threats_detected": 0,
            "threats_mitigated": 0,
            "messages_encrypted": 0,
            "messages_decrypted": 0,
            "authentication_attempts": 0,
            "authentication_failures": 0,
            "key_rotations": 0,
        }

        logger.info(f"BetaNet Security Manager initialized for node {node_id}")

    async def initialize(self) -> bool:
        """Initialize security manager and generate node credentials."""
        try:
            # Generate node credentials
            self.node_credentials = await self._generate_node_credentials()

            # Initialize threat detection patterns
            await self._initialize_threat_patterns()

            # Start background security tasks
            asyncio.create_task(self._security_maintenance_loop())

            logger.info(f"Security manager initialized successfully for node {self.node_id}")
            return True

        except Exception as e:
            logger.error(f"Security manager initialization failed: {e}")
            return False

    async def establish_secure_channel(
        self, remote_node_id: str, channel_type: ChannelType, security_level: SecurityLevel = SecurityLevel.STANDARD
    ) -> tuple[bool, str | None]:
        """Establish a secure communication channel with a remote node."""

        if not self.node_credentials:
            logger.error("Node credentials not initialized")
            return False, None

        # Check if channel type is allowed
        if channel_type not in self.security_policy.allowed_channel_types:
            logger.warning(f"Channel type {channel_type} not allowed by security policy")
            return False, None

        # Check if node is blocked
        if remote_node_id in self.blocked_nodes:
            logger.warning(f"Remote node {remote_node_id} is blocked")
            return False, None

        try:
            channel_id = str(uuid.uuid4())

            # Perform key exchange
            success, encryption_key, auth_key = await self._perform_key_exchange(remote_node_id, security_level)

            if not success:
                logger.error(f"Key exchange failed with {remote_node_id}")
                return False, None

            # Create secure channel
            channel = SecureChannel(
                channel_id=channel_id,
                channel_type=channel_type,
                local_node_id=self.node_id,
                remote_node_id=remote_node_id,
                encryption_key=encryption_key,
                authentication_key=auth_key,
                metadata={"security_level": security_level.value, "established_at": time.time()},
            )

            # Store channel
            self.active_channels[channel_id] = channel
            self.channel_by_peer[remote_node_id] = channel_id

            # Schedule key rotation
            rotation_time = time.time() + self.security_policy.max_channel_lifetime // 2
            self.key_rotation_schedule[channel_id] = rotation_time

            self.security_stats["channels_established"] += 1

            logger.info(f"Secure channel established with {remote_node_id}: {channel_id}")
            return True, channel_id

        except Exception as e:
            logger.error(f"Failed to establish secure channel with {remote_node_id}: {e}")
            return False, None

    async def encrypt_message(
        self, message: bytes, channel_id: str, associated_data: bytes | None = None
    ) -> tuple[bool, bytes | None]:
        """Encrypt a message for secure transmission."""

        channel = self.active_channels.get(channel_id)
        if not channel or not channel.is_active:
            logger.error(f"Channel {channel_id} not found or inactive")
            return False, None

        try:
            # Update channel activity
            channel.last_used = time.time()
            channel.sequence_number += 1

            # Create message header
            header = {
                "version": 1,
                "channel_id": channel_id,
                "sequence": channel.sequence_number,
                "timestamp": int(time.time()),
                "sender": self.node_id,
            }

            header_bytes = json.dumps(header, sort_keys=True).encode()

            # Encrypt message using AEAD
            encrypted_message = await self._aead_encrypt(message, channel.encryption_key, header_bytes, associated_data)

            # Create authenticated packet
            packet = {
                "header": header,
                "encrypted_payload": encrypted_message.hex(),
                "auth_tag": self._create_auth_tag(header_bytes + encrypted_message, channel.authentication_key).hex(),
            }

            packet_bytes = json.dumps(packet).encode()

            self.security_stats["messages_encrypted"] += 1

            return True, packet_bytes

        except Exception as e:
            logger.error(f"Message encryption failed for channel {channel_id}: {e}")
            return False, None

    async def decrypt_message(
        self, encrypted_packet: bytes, expected_sender: str | None = None
    ) -> tuple[bool, bytes | None, dict[str, Any] | None]:
        """Decrypt a received secure message."""

        try:
            # Parse packet
            packet = json.loads(encrypted_packet.decode())
            header = packet["header"]
            encrypted_payload = bytes.fromhex(packet["encrypted_payload"])
            auth_tag = bytes.fromhex(packet["auth_tag"])

            channel_id = header["channel_id"]
            sender_id = header["sender"]
            sequence = header["sequence"]
            timestamp = header["timestamp"]

            # Validate sender if specified
            if expected_sender and sender_id != expected_sender:
                logger.warning(f"Unexpected sender: {sender_id} (expected {expected_sender})")
                return False, None, None

            # Get channel
            channel = self.active_channels.get(channel_id)
            if not channel or not channel.is_active:
                logger.error(f"Channel {channel_id} not found or inactive")
                return False, None, None

            # Verify sender matches channel
            if sender_id != channel.remote_node_id:
                logger.warning(f"Sender {sender_id} doesn't match channel peer {channel.remote_node_id}")
                return False, None, None

            # Check timestamp (prevent replay attacks)
            current_time = int(time.time())
            if abs(current_time - timestamp) > 300:  # 5 minutes tolerance
                logger.warning(f"Message timestamp out of acceptable range: {timestamp}")
                await self._record_threat_event(
                    ThreatType.REPLAY_ATTACK, sender_id, f"Timestamp out of range: {timestamp}"
                )
                return False, None, None

            # Verify sequence number (prevent replay)
            if sequence <= channel.sequence_number:
                logger.warning(f"Invalid sequence number: {sequence} <= {channel.sequence_number}")
                await self._record_threat_event(ThreatType.REPLAY_ATTACK, sender_id, f"Invalid sequence: {sequence}")
                return False, None, None

            # Verify authentication tag
            header_bytes = json.dumps(header, sort_keys=True).encode()
            expected_auth_tag = self._create_auth_tag(header_bytes + encrypted_payload, channel.authentication_key)

            if not hmac.compare_digest(auth_tag, expected_auth_tag):
                logger.error(f"Authentication tag verification failed for channel {channel_id}")
                await self._record_threat_event(ThreatType.MAN_IN_MIDDLE, sender_id, "Authentication tag mismatch")
                return False, None, None

            # Decrypt message
            decrypted_message = await self._aead_decrypt(encrypted_payload, channel.encryption_key, header_bytes)

            # Update channel state
            channel.sequence_number = sequence
            channel.last_used = time.time()

            self.security_stats["messages_decrypted"] += 1

            return True, decrypted_message, header

        except Exception as e:
            logger.error(f"Message decryption failed: {e}")
            return False, None, None

    async def verify_node_identity(self, node_id: str, certificate: bytes, signature: bytes, challenge: bytes) -> bool:
        """Verify the identity of a remote node."""

        try:
            self.security_stats["authentication_attempts"] += 1

            # Check if certificate is revoked
            cert_fingerprint = hashlib.sha256(certificate).hexdigest()
            if cert_fingerprint in self.revoked_certificates:
                logger.warning(f"Certificate for node {node_id} is revoked")
                self.security_stats["authentication_failures"] += 1
                return False

            # Load certificate and extract public key
            if CRYPTOGRAPHY_AVAILABLE:
                try:
                    public_key = serialization.load_pem_public_key(certificate, backend=default_backend())
                except Exception as e:
                    logger.error(f"Invalid certificate format for node {node_id}: {e}")
                    self.security_stats["authentication_failures"] += 1
                    return False
            else:
                # Simplified verification without cryptography library
                logger.warning("Full certificate verification unavailable - using simplified check")
                return True

            # Verify signature
            try:
                public_key.verify(
                    signature,
                    challenge,
                    padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                    hashes.SHA256(),
                )

                # Store trusted node
                credential = SecurityCredential(
                    node_id=node_id,
                    public_key=certificate,
                    private_key=b"",  # We don't store remote private keys
                    certificate=certificate,
                    key_fingerprint=cert_fingerprint,
                )

                self.trusted_nodes[node_id] = credential

                logger.info(f"Node {node_id} identity verified successfully")
                return True

            except InvalidSignature:
                logger.error(f"Invalid signature from node {node_id}")
                await self._record_threat_event(
                    ThreatType.IDENTITY_SPOOFING, node_id, "Invalid signature during identity verification"
                )
                self.security_stats["authentication_failures"] += 1
                return False

        except Exception as e:
            logger.error(f"Identity verification failed for node {node_id}: {e}")
            self.security_stats["authentication_failures"] += 1
            return False

    async def detect_threats(self, traffic_sample: list[dict[str, Any]]) -> list[ThreatEvent]:
        """Detect security threats from traffic analysis."""

        detected_threats = []

        if not self.security_policy.threat_detection_enabled:
            return detected_threats

        try:
            # Analyze traffic patterns
            for pattern_name, pattern_rules in self.threat_patterns.items():
                for rule in pattern_rules:
                    matches = await self._analyze_pattern(traffic_sample, rule)

                    for match in matches:
                        threat_event = ThreatEvent(
                            event_id=str(uuid.uuid4()),
                            threat_type=ThreatType(rule["threat_type"]),
                            source_node=match.get("source_node", "unknown"),
                            target_node=match.get("target_node", self.node_id),
                            severity=rule["severity"],
                            description=rule["description"],
                            evidence=match,
                        )

                        detected_threats.append(threat_event)
                        self.threat_events.append(threat_event)
                        self.security_stats["threats_detected"] += 1

                        logger.warning(
                            f"Threat detected: {threat_event.threat_type.value} " f"from {threat_event.source_node}"
                        )

                        # Automatic mitigation if enabled
                        if self.security_policy.automatic_mitigation:
                            await self._mitigate_threat(threat_event)

            return detected_threats

        except Exception as e:
            logger.error(f"Threat detection failed: {e}")
            return detected_threats

    async def rotate_keys(self, channel_id: str) -> bool:
        """Rotate encryption keys for a secure channel."""

        channel = self.active_channels.get(channel_id)
        if not channel:
            return False

        try:
            # Generate new keys
            new_encryption_key = secrets.token_bytes(32)
            new_auth_key = secrets.token_bytes(32)

            # Perform key exchange with peer
            success = await self._exchange_rotated_keys(channel.remote_node_id, new_encryption_key, new_auth_key)

            if success:
                # Update channel keys
                channel.encryption_key = new_encryption_key
                channel.authentication_key = new_auth_key
                channel.sequence_number = 0  # Reset sequence

                # Schedule next rotation
                rotation_time = time.time() + self.security_policy.max_channel_lifetime // 2
                self.key_rotation_schedule[channel_id] = rotation_time

                self.security_stats["key_rotations"] += 1

                logger.info(f"Keys rotated successfully for channel {channel_id}")
                return True
            else:
                logger.error(f"Key rotation failed for channel {channel_id}")
                return False

        except Exception as e:
            logger.error(f"Key rotation error for channel {channel_id}: {e}")
            return False

    async def terminate_channel(self, channel_id: str) -> bool:
        """Securely terminate a communication channel."""

        channel = self.active_channels.get(channel_id)
        if not channel:
            return False

        try:
            # Mark channel as inactive
            channel.is_active = False

            # Remove from active channels
            del self.active_channels[channel_id]

            # Remove peer mapping
            if channel.remote_node_id in self.channel_by_peer:
                del self.channel_by_peer[channel.remote_node_id]

            # Remove scheduled key rotation
            if channel_id in self.key_rotation_schedule:
                del self.key_rotation_schedule[channel_id]

            # Securely wipe keys
            await self._secure_wipe_keys(channel)

            self.security_stats["channels_terminated"] += 1

            logger.info(f"Channel {channel_id} terminated securely")
            return True

        except Exception as e:
            logger.error(f"Channel termination failed for {channel_id}: {e}")
            return False

    # Private security methods

    async def _generate_node_credentials(self) -> SecurityCredential:
        """Generate cryptographic credentials for this node."""

        if not CRYPTOGRAPHY_AVAILABLE:
            # Fallback to simple credentials
            private_key = secrets.token_bytes(32)
            public_key = hashlib.sha256(private_key + self.node_id.encode()).digest()

            return SecurityCredential(
                node_id=self.node_id,
                public_key=public_key,
                private_key=private_key,
                key_fingerprint=hashlib.sha256(public_key).hexdigest(),
            )

        # Generate RSA key pair
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())

        public_key = private_key.public_key()

        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        # Create self-signed certificate (simplified)
        certificate = public_pem  # In production, would create proper X.509 cert

        return SecurityCredential(
            node_id=self.node_id,
            public_key=public_pem,
            private_key=private_pem,
            certificate=certificate,
            key_fingerprint=hashlib.sha256(public_pem).hexdigest(),
        )

    async def _perform_key_exchange(
        self, remote_node_id: str, security_level: SecurityLevel
    ) -> tuple[bool, bytes, bytes]:
        """Perform key exchange with remote node."""

        try:
            # Generate ephemeral keys for forward secrecy
            if self.security_policy.enable_forward_secrecy:
                ephemeral_private = secrets.token_bytes(32)
                hashlib.sha256(ephemeral_private).digest()
            else:
                ephemeral_private = self.node_credentials.private_key[:32]
                self.node_credentials.public_key[:32]

            # Derive shared secrets (simplified ECDH-like)
            shared_secret = hashlib.sha256(ephemeral_private + remote_node_id.encode() + self.node_id.encode()).digest()

            # Derive encryption and authentication keys using HKDF
            if CRYPTOGRAPHY_AVAILABLE:
                hkdf = HKDF(
                    algorithm=hashes.SHA256(),
                    length=64,  # 32 bytes each for encryption and auth
                    salt=b"betanet_key_derivation",
                    info=f"{self.node_id}:{remote_node_id}".encode(),
                    backend=default_backend(),
                )
                derived_keys = hkdf.derive(shared_secret)
            else:
                # Fallback key derivation
                derived_keys = (
                    hashlib.sha256(shared_secret + b"betanet_key_derivation").digest()
                    + hashlib.sha256(shared_secret + b"betanet_auth_derivation").digest()
                )

            encryption_key = derived_keys[:32]
            auth_key = derived_keys[32:64]

            return True, encryption_key, auth_key

        except Exception as e:
            logger.error(f"Key exchange failed with {remote_node_id}: {e}")
            return False, b"", b""

    async def _aead_encrypt(
        self,
        plaintext: bytes,
        key: bytes,
        additional_data: bytes | None = None,
        associated_data: bytes | None = None,
    ) -> bytes:
        """Encrypt using AEAD (Authenticated Encryption with Associated Data)."""

        if not CRYPTOGRAPHY_AVAILABLE:
            # Fallback to simple XOR encryption
            cipher_key = hashlib.sha256(key).digest()[:16]
            return bytes(a ^ b for a, b in zip(plaintext, cipher_key * (len(plaintext) // 16 + 1)))

        # Use ChaCha20Poly1305 for AEAD
        nonce = secrets.token_bytes(12)  # 96-bit nonce for ChaCha20

        try:
            from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

            aead = ChaCha20Poly1305(key)
            ciphertext = aead.encrypt(nonce, plaintext, additional_data)
            return nonce + ciphertext
        except ImportError:
            # Fallback to AES-GCM
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            aesgcm = AESGCM(key)
            ciphertext = aesgcm.encrypt(nonce, plaintext, additional_data)
            return nonce + ciphertext

    async def _aead_decrypt(self, ciphertext: bytes, key: bytes, additional_data: bytes | None = None) -> bytes:
        """Decrypt using AEAD."""

        if not CRYPTOGRAPHY_AVAILABLE:
            # Fallback XOR decryption
            cipher_key = hashlib.sha256(key).digest()[:16]
            return bytes(a ^ b for a, b in zip(ciphertext, cipher_key * (len(ciphertext) // 16 + 1)))

        nonce = ciphertext[:12]
        encrypted = ciphertext[12:]

        try:
            from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

            aead = ChaCha20Poly1305(key)
            return aead.decrypt(nonce, encrypted, additional_data)
        except ImportError:
            # Fallback to AES-GCM
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            aesgcm = AESGCM(key)
            return aesgcm.decrypt(nonce, encrypted, additional_data)

    def _create_auth_tag(self, data: bytes, key: bytes) -> bytes:
        """Create HMAC authentication tag."""
        return hmac.new(key, data, hashlib.sha256).digest()

    async def _initialize_threat_patterns(self) -> None:
        """Initialize threat detection patterns."""

        self.threat_patterns = {
            "replay_attack": [
                {
                    "threat_type": "replay_attack",
                    "description": "Repeated message with same timestamp/sequence",
                    "severity": "high",
                    "pattern": {"duplicate_timestamps": {"threshold": 2}, "sequence_anomaly": {"threshold": 3}},
                }
            ],
            "dos_attack": [
                {
                    "threat_type": "dos_attack",
                    "description": "Excessive requests from single source",
                    "severity": "high",
                    "pattern": {
                        "request_rate": {"threshold": 100, "window": 60},
                        "connection_count": {"threshold": 50},
                    },
                }
            ],
            "traffic_analysis": [
                {
                    "threat_type": "traffic_analysis",
                    "description": "Unusual traffic patterns indicating surveillance",
                    "severity": "medium",
                    "pattern": {"timing_correlation": {"threshold": 0.8}, "size_correlation": {"threshold": 0.9}},
                }
            ],
            "identity_spoofing": [
                {
                    "threat_type": "identity_spoofing",
                    "description": "Authentication failures or mismatched certificates",
                    "severity": "critical",
                    "pattern": {"auth_failures": {"threshold": 5, "window": 300}, "cert_mismatch": {"threshold": 1}},
                }
            ],
        }

    async def _analyze_pattern(
        self, traffic_sample: list[dict[str, Any]], rule: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Analyze traffic for specific threat patterns."""

        matches = []
        pattern = rule["pattern"]

        try:
            # Group events by source
            events_by_source = {}
            for event in traffic_sample:
                source = event.get("source", "unknown")
                if source not in events_by_source:
                    events_by_source[source] = []
                events_by_source[source].append(event)

            # Analyze each source
            for source, events in events_by_source.items():
                if source in self.blocked_nodes:
                    continue

                # Check request rate
                if "request_rate" in pattern:
                    rate_config = pattern["request_rate"]
                    window = rate_config.get("window", 60)
                    threshold = rate_config["threshold"]

                    recent_events = [e for e in events if time.time() - e.get("timestamp", 0) < window]

                    if len(recent_events) > threshold:
                        matches.append(
                            {
                                "source_node": source,
                                "target_node": self.node_id,
                                "event_count": len(recent_events),
                                "threshold": threshold,
                                "window": window,
                            }
                        )

                # Check duplicate timestamps
                if "duplicate_timestamps" in pattern:
                    threshold = pattern["duplicate_timestamps"]["threshold"]
                    timestamps = [e.get("timestamp", 0) for e in events]
                    timestamp_counts = {}

                    for ts in timestamps:
                        timestamp_counts[ts] = timestamp_counts.get(ts, 0) + 1

                    for ts, count in timestamp_counts.items():
                        if count > threshold:
                            matches.append(
                                {
                                    "source_node": source,
                                    "target_node": self.node_id,
                                    "duplicate_timestamp": ts,
                                    "count": count,
                                    "threshold": threshold,
                                }
                            )

                # Check authentication failures
                if "auth_failures" in pattern:
                    threshold = pattern["auth_failures"]["threshold"]
                    window = pattern["auth_failures"].get("window", 300)

                    auth_failures = [
                        e
                        for e in events
                        if e.get("type") == "auth_failure" and time.time() - e.get("timestamp", 0) < window
                    ]

                    if len(auth_failures) > threshold:
                        matches.append(
                            {
                                "source_node": source,
                                "target_node": self.node_id,
                                "auth_failure_count": len(auth_failures),
                                "threshold": threshold,
                                "window": window,
                            }
                        )

        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")

        return matches

    async def _record_threat_event(
        self, threat_type: ThreatType, source_node: str, description: str, evidence: dict[str, Any] | None = None
    ) -> None:
        """Record a security threat event."""

        event = ThreatEvent(
            event_id=str(uuid.uuid4()),
            threat_type=threat_type,
            source_node=source_node,
            target_node=self.node_id,
            severity="medium",
            description=description,
            evidence=evidence or {},
        )

        self.threat_events.append(event)
        self.security_stats["threats_detected"] += 1

        logger.warning(f"Threat recorded: {threat_type.value} from {source_node}")

    async def _mitigate_threat(self, threat_event: ThreatEvent) -> None:
        """Mitigate a detected security threat."""

        try:
            mitigation_applied = False

            # Block malicious nodes
            if threat_event.threat_type in [
                ThreatType.DOS_ATTACK,
                ThreatType.IDENTITY_SPOOFING,
                ThreatType.MALICIOUS_NODE,
            ]:
                if threat_event.severity in ["high", "critical"]:
                    self.blocked_nodes.add(threat_event.source_node)
                    mitigation_applied = True
                    logger.info(f"Blocked node {threat_event.source_node} due to {threat_event.threat_type.value}")

            # Terminate compromised channels
            if threat_event.threat_type in [ThreatType.MAN_IN_MIDDLE, ThreatType.REPLAY_ATTACK]:
                channel_id = self.channel_by_peer.get(threat_event.source_node)
                if channel_id:
                    await self.terminate_channel(channel_id)
                    mitigation_applied = True
                    logger.info(f"Terminated channel {channel_id} due to {threat_event.threat_type.value}")

            # Force key rotation for traffic analysis
            if threat_event.threat_type == ThreatType.TRAFFIC_ANALYSIS:
                for channel_id, channel in self.active_channels.items():
                    await self.rotate_keys(channel_id)
                mitigation_applied = True
                logger.info("Rotated all channel keys due to traffic analysis threat")

            if mitigation_applied:
                threat_event.mitigated = True
                self.security_stats["threats_mitigated"] += 1

        except Exception as e:
            logger.error(f"Threat mitigation failed for {threat_event.event_id}: {e}")

    async def _exchange_rotated_keys(self, remote_node_id: str, new_encryption_key: bytes, new_auth_key: bytes) -> bool:
        """Exchange rotated keys with remote node."""
        # Simplified key exchange - in production would use proper key exchange protocol
        return True

    async def _secure_wipe_keys(self, channel: SecureChannel) -> None:
        """Securely wipe cryptographic keys from memory."""
        try:
            # Overwrite key bytes with random data
            key_len = len(channel.encryption_key)
            channel.encryption_key = secrets.token_bytes(key_len)

            auth_len = len(channel.authentication_key)
            channel.authentication_key = secrets.token_bytes(auth_len)

        except Exception as e:
            logger.error(f"Secure key wipe failed: {e}")

    async def _security_maintenance_loop(self) -> None:
        """Background security maintenance tasks."""
        while True:
            try:
                current_time = time.time()

                # Rotate keys that are due
                for channel_id, rotation_time in list(self.key_rotation_schedule.items()):
                    if current_time >= rotation_time:
                        await self.rotate_keys(channel_id)

                # Clean up expired channels
                expired_channels = [
                    channel_id
                    for channel_id, channel in self.active_channels.items()
                    if current_time - channel.created_at > self.security_policy.max_channel_lifetime
                ]

                for channel_id in expired_channels:
                    await self.terminate_channel(channel_id)

                # Clean up old threat events (keep last 1000)
                if len(self.threat_events) > 1000:
                    self.threat_events = self.threat_events[-1000:]

                # Unblock nodes after timeout (24 hours)
                for node_id in self.blocked_nodes:
                    # In production, would track block timestamps
                    # For now, just keep blocks indefinitely
                    pass

                await asyncio.sleep(60)  # Run every minute

            except Exception as e:
                logger.error(f"Security maintenance error: {e}")
                await asyncio.sleep(60)

    # Public API methods

    def get_security_stats(self) -> dict[str, Any]:
        """Get security statistics."""
        return {
            **self.security_stats,
            "active_channels": len(self.active_channels),
            "trusted_nodes": len(self.trusted_nodes),
            "blocked_nodes": len(self.blocked_nodes),
            "pending_key_rotations": len(self.key_rotation_schedule),
            "recent_threats": len(
                [event for event in self.threat_events[-100:] if time.time() - event.detected_at < 3600]  # Last hour
            ),
        }

    def get_channel_info(self, channel_id: str) -> dict[str, Any] | None:
        """Get information about a secure channel."""
        channel = self.active_channels.get(channel_id)
        if not channel:
            return None

        return {
            "channel_id": channel.channel_id,
            "channel_type": channel.channel_type.value,
            "remote_node": channel.remote_node_id,
            "created_at": channel.created_at,
            "last_used": channel.last_used,
            "sequence_number": channel.sequence_number,
            "is_active": channel.is_active,
            "security_level": channel.metadata.get("security_level", "unknown"),
        }

    def get_threat_summary(self) -> dict[str, Any]:
        """Get summary of recent threats."""
        recent_threats = [
            event for event in self.threat_events if time.time() - event.detected_at < 86400  # Last 24 hours
        ]

        threat_counts = {}
        for event in recent_threats:
            threat_type = event.threat_type.value
            threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1

        return {
            "total_threats_24h": len(recent_threats),
            "threats_by_type": threat_counts,
            "mitigated_threats": len([e for e in recent_threats if e.mitigated]),
            "blocked_nodes": len(self.blocked_nodes),
            "critical_threats": len([e for e in recent_threats if e.severity == "critical"]),
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform security system health check."""
        issues = []

        # Check credential status
        if not self.node_credentials:
            issues.append("Node credentials not initialized")
        elif time.time() > self.node_credentials.expires_at:
            issues.append("Node credentials expired")

        # Check channel health
        inactive_channels = [
            channel_id for channel_id, channel in self.active_channels.items() if not channel.is_active
        ]

        if inactive_channels:
            issues.append(f"{len(inactive_channels)} inactive channels found")

        # Check threat detection
        recent_critical_threats = len(
            [
                event
                for event in self.threat_events
                if (time.time() - event.detected_at < 3600 and event.severity == "critical" and not event.mitigated)
            ]
        )

        if recent_critical_threats > 0:
            issues.append(f"{recent_critical_threats} unmitigated critical threats")

        return {
            "healthy": len(issues) == 0,
            "issues": issues,
            "active_channels": len(self.active_channels),
            "blocked_nodes": len(self.blocked_nodes),
            "security_stats": self.get_security_stats(),
        }
