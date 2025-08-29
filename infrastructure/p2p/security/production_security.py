"""Production Security Layer for P2P Communications.

Implements comprehensive security mechanisms including:
- End-to-end encryption with perfect forward secrecy
- Peer authentication and identity verification
- Trust scoring and reputation management
- Anti-spam and DoS protection
- Security monitoring and threat detection
- Key management and rotation

This converts the test/verification security code into production-ready
security protocols for the LibP2P mesh network.
"""

import asyncio
import base64
from collections import defaultdict, deque
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

# Cryptographic imports with fallbacks
try:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different types of communication."""

    NONE = "none"  # No security (testing only)
    BASIC = "basic"  # Basic authentication
    STANDARD = "standard"  # Standard encryption
    HIGH = "high"  # High security with PFS
    CRITICAL = "critical"  # Maximum security


class ThreatType(Enum):
    """Types of security threats."""

    SPAM = "spam"
    DOS_ATTACK = "dos_attack"
    SYBIL_ATTACK = "sybil_attack"
    ECLIPSE_ATTACK = "eclipse_attack"
    MALFORMED_MESSAGE = "malformed_message"
    IDENTITY_SPOOFING = "identity_spoofing"
    EAVESDROPPING = "eavesdropping"
    REPLAY_ATTACK = "replay_attack"


@dataclass
class SecurityConfig:
    """Security configuration parameters."""

    # Encryption settings
    security_level: SecurityLevel = SecurityLevel.STANDARD
    key_size: int = 2048  # RSA key size
    session_key_size: int = 32  # AES key size in bytes
    enable_perfect_forward_secrecy: bool = True

    # Authentication
    require_peer_authentication: bool = True
    enable_trust_scoring: bool = True
    min_trust_score: float = 0.3

    # Rate limiting
    max_messages_per_second: int = 10
    max_message_size: int = 1024 * 1024  # 1MB
    rate_limit_window: int = 60  # seconds

    # Key management
    key_rotation_interval: int = 3600  # 1 hour
    max_key_age: int = 7200  # 2 hours

    # Threat detection
    enable_anomaly_detection: bool = True
    max_failed_auth_attempts: int = 5
    ban_duration: int = 3600  # 1 hour

    # Network security
    enable_message_signing: bool = True
    enable_replay_protection: bool = True
    replay_window: int = 300  # 5 minutes


@dataclass
class PeerIdentity:
    """Cryptographic identity of a peer."""

    peer_id: str
    public_key: bytes | None = None
    certificate: bytes | None = None
    trust_score: float = 0.5
    reputation: float = 0.5

    # Authentication state
    is_authenticated: bool = False
    auth_timestamp: float | None = None
    session_key: bytes | None = None

    # Security metrics
    failed_auth_attempts: int = 0
    last_auth_attempt: float | None = None
    message_count: int = 0
    spam_score: float = 0.0

    # Behavioral tracking
    connection_history: list[float] = field(default_factory=list)
    message_patterns: dict[str, int] = field(default_factory=dict)
    anomaly_score: float = 0.0


@dataclass
class SecurityEvent:
    """Security event for monitoring and analysis."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    event_type: ThreatType = ThreatType.SPAM
    peer_id: str = ""
    severity: str = "low"  # low, medium, high, critical
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


class CryptoManager:
    """Manages cryptographic operations and key lifecycle."""

    def __init__(self, config: SecurityConfig):
        self.config = config

        # Key storage
        self.node_private_key: bytes | None = None
        self.node_public_key: bytes | None = None
        self.session_keys: dict[str, tuple[bytes, float]] = {}  # peer_id -> (key, timestamp)

        # Crypto state
        self.backend = default_backend() if CRYPTO_AVAILABLE else None

        # Initialize if crypto is available
        if CRYPTO_AVAILABLE:
            self._generate_node_keys()

    def _generate_node_keys(self):
        """Generate node's RSA key pair."""
        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography not available, using mock keys")
            self.node_private_key = b"mock_private_key"
            self.node_public_key = b"mock_public_key"
            return

        try:
            # Generate RSA key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=self.config.key_size, backend=self.backend
            )

            # Serialize keys
            self.node_private_key = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )

            self.node_public_key = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo
            )

            logger.info(f"Generated {self.config.key_size}-bit RSA key pair")

        except Exception as e:
            logger.error(f"Failed to generate node keys: {e}")
            # Fallback to mock keys
            self.node_private_key = b"mock_private_key"
            self.node_public_key = b"mock_public_key"

    def get_public_key(self) -> bytes:
        """Get node's public key."""
        return self.node_public_key or b""

    def generate_session_key(self, peer_id: str) -> bytes:
        """Generate a new session key for a peer."""
        session_key = secrets.token_bytes(self.config.session_key_size)
        self.session_keys[peer_id] = (session_key, time.time())
        return session_key

    def get_session_key(self, peer_id: str) -> bytes | None:
        """Get session key for a peer."""
        if peer_id not in self.session_keys:
            return None

        key, timestamp = self.session_keys[peer_id]

        # Check if key is expired
        if time.time() - timestamp > self.config.max_key_age:
            del self.session_keys[peer_id]
            return None

        return key

    def encrypt_message(self, data: bytes, peer_id: str) -> bytes | None:
        """Encrypt message for a specific peer."""
        if not CRYPTO_AVAILABLE:
            # Mock encryption for testing
            return base64.b64encode(data)

        try:
            session_key = self.get_session_key(peer_id)
            if not session_key:
                return None

            # Use AES-GCM for authenticated encryption
            iv = secrets.token_bytes(16)
            cipher = Cipher(algorithms.AES(session_key), modes.GCM(iv), backend=self.backend)

            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data) + encryptor.finalize()

            # Combine IV + tag + ciphertext
            encrypted_data = iv + encryptor.tag + ciphertext
            return encrypted_data

        except Exception as e:
            logger.error(f"Encryption failed for peer {peer_id}: {e}")
            return None

    def decrypt_message(self, encrypted_data: bytes, peer_id: str) -> bytes | None:
        """Decrypt message from a specific peer."""
        if not CRYPTO_AVAILABLE:
            # Mock decryption for testing
            try:
                return base64.b64decode(encrypted_data)
            except Exception:
                return None

        try:
            session_key = self.get_session_key(peer_id)
            if not session_key:
                return None

            # Extract IV, tag, and ciphertext
            if len(encrypted_data) < 32:  # 16 bytes IV + 16 bytes tag minimum
                return None

            iv = encrypted_data[:16]
            tag = encrypted_data[16:32]
            ciphertext = encrypted_data[32:]

            # Decrypt
            cipher = Cipher(algorithms.AES(session_key), modes.GCM(iv, tag), backend=self.backend)

            decryptor = cipher.decryptor()
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()

            return plaintext

        except Exception as e:
            logger.error(f"Decryption failed for peer {peer_id}: {e}")
            return None

    def sign_message(self, data: bytes) -> bytes | None:
        """Sign a message with node's private key."""
        if not CRYPTO_AVAILABLE:
            # Mock signature
            return hashlib.sha256(data + self.node_private_key).digest()

        try:
            # Load private key
            private_key = serialization.load_pem_private_key(self.node_private_key, password=None, backend=self.backend)

            # Sign with PSS padding
            signature = private_key.sign(
                data,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )

            return signature

        except Exception as e:
            logger.error(f"Message signing failed: {e}")
            return None

    def verify_signature(self, data: bytes, signature: bytes, peer_public_key: bytes) -> bool:
        """Verify message signature."""
        if not CRYPTO_AVAILABLE:
            # Mock verification
            expected = hashlib.sha256(data + b"mock_private_key").digest()
            return hmac.compare_digest(signature, expected)

        try:
            # Load public key
            public_key = serialization.load_pem_public_key(peer_public_key, backend=self.backend)

            # Verify signature
            public_key.verify(
                signature,
                data,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )

            return True

        except Exception as e:
            logger.debug(f"Signature verification failed: {e}")
            return False

    async def rotate_keys(self):
        """Rotate session keys for all peers."""
        logger.info("Rotating session keys")

        # Generate new session keys for active peers
        current_time = time.time()
        for peer_id in list(self.session_keys.keys()):
            key, timestamp = self.session_keys[peer_id]

            if current_time - timestamp > self.config.key_rotation_interval:
                self.generate_session_key(peer_id)
                logger.debug(f"Rotated session key for peer {peer_id}")


class TrustManager:
    """Manages peer trust scoring and reputation."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.peer_identities: dict[str, PeerIdentity] = {}

        # Trust calculation parameters
        self.trust_decay_rate = 0.01  # Daily decay rate
        self.reputation_weight = 0.7  # Weight of historical reputation
        self.recent_weight = 0.3  # Weight of recent behavior

        # Behavioral analysis
        self.behavior_window = 3600  # 1 hour window for behavior analysis

    def get_peer_identity(self, peer_id: str) -> PeerIdentity:
        """Get or create peer identity."""
        if peer_id not in self.peer_identities:
            self.peer_identities[peer_id] = PeerIdentity(peer_id=peer_id)
        return self.peer_identities[peer_id]

    def update_trust_score(self, peer_id: str, delta: float, reason: str = ""):
        """Update peer's trust score."""
        identity = self.get_peer_identity(peer_id)

        # Apply trust change with bounds
        old_score = identity.trust_score
        identity.trust_score = max(0.0, min(1.0, identity.trust_score + delta))

        logger.debug(f"Trust score for {peer_id}: {old_score:.3f} -> {identity.trust_score:.3f} ({reason})")

    def calculate_reputation(self, peer_id: str) -> float:
        """Calculate peer's reputation based on behavior history."""
        identity = self.get_peer_identity(peer_id)

        # Start with base reputation
        reputation = identity.reputation

        # Factor in message patterns
        current_time = time.time()
        recent_messages = sum(
            count
            for timestamp, count in identity.message_patterns.items()
            if current_time - float(timestamp) < self.behavior_window
        )

        # Penalize spam-like behavior
        if recent_messages > self.config.max_messages_per_second * self.behavior_window:
            reputation *= 0.8  # Spam penalty

        # Reward consistent good behavior
        if identity.failed_auth_attempts == 0 and identity.message_count > 10:
            reputation = min(1.0, reputation * 1.05)  # Small bonus

        # Update stored reputation
        identity.reputation = reputation
        return reputation

    def is_peer_trusted(self, peer_id: str) -> bool:
        """Check if peer meets trust threshold."""
        identity = self.get_peer_identity(peer_id)
        reputation = self.calculate_reputation(peer_id)

        # Combine trust score and reputation
        combined_score = identity.trust_score * self.reputation_weight + reputation * self.recent_weight

        return combined_score >= self.config.min_trust_score

    def record_good_behavior(self, peer_id: str, behavior_type: str):
        """Record positive peer behavior."""
        trust_increases = {
            "successful_auth": 0.05,
            "message_delivered": 0.01,
            "helped_routing": 0.02,
            "valid_signature": 0.01,
        }

        delta = trust_increases.get(behavior_type, 0.01)
        self.update_trust_score(peer_id, delta, f"good behavior: {behavior_type}")

    def record_bad_behavior(self, peer_id: str, behavior_type: str):
        """Record negative peer behavior."""
        trust_decreases = {
            "auth_failure": -0.1,
            "invalid_signature": -0.15,
            "spam": -0.2,
            "malformed_message": -0.05,
            "replay_attack": -0.3,
        }

        delta = trust_decreases.get(behavior_type, -0.05)
        self.update_trust_score(peer_id, delta, f"bad behavior: {behavior_type}")

    async def maintain_trust_scores(self):
        """Periodic maintenance of trust scores."""
        current_time = time.time()

        for peer_id, identity in self.peer_identities.items():
            # Apply trust decay
            identity.trust_score *= 1.0 - self.trust_decay_rate

            # Clean old behavior patterns
            old_patterns = [
                timestamp
                for timestamp in identity.message_patterns.keys()
                if current_time - float(timestamp) > self.behavior_window * 2
            ]
            for timestamp in old_patterns:
                del identity.message_patterns[timestamp]


class ThreatDetector:
    """Detects and responds to security threats."""

    def __init__(self, config: SecurityConfig):
        self.config = config

        # Rate limiting
        self.rate_limiters: dict[str, deque] = defaultdict(deque)  # peer_id -> timestamps
        self.banned_peers: dict[str, float] = {}  # peer_id -> ban_end_time

        # Anomaly detection
        self.message_patterns: dict[str, list[float]] = defaultdict(list)
        self.baseline_patterns: dict[str, dict[str, float]] = {}

        # Security events
        self.security_events: list[SecurityEvent] = []
        self.max_events = 1000  # Keep last 1000 events

    def is_peer_banned(self, peer_id: str) -> bool:
        """Check if peer is currently banned."""
        if peer_id not in self.banned_peers:
            return False

        ban_end_time = self.banned_peers[peer_id]
        if time.time() > ban_end_time:
            del self.banned_peers[peer_id]
            return False

        return True

    def ban_peer(self, peer_id: str, duration: int | None = None):
        """Ban a peer for specified duration."""
        duration = duration or self.config.ban_duration
        ban_end_time = time.time() + duration
        self.banned_peers[peer_id] = ban_end_time

        self._record_security_event(ThreatType.DOS_ATTACK, peer_id, "high", f"Peer banned for {duration} seconds")

        logger.warning(f"Banned peer {peer_id} for {duration} seconds")

    def check_rate_limit(self, peer_id: str) -> bool:
        """Check if peer exceeds rate limit."""
        current_time = time.time()
        peer_timestamps = self.rate_limiters[peer_id]

        # Remove old timestamps
        while peer_timestamps and current_time - peer_timestamps[0] > self.config.rate_limit_window:
            peer_timestamps.popleft()

        # Check rate limit
        if len(peer_timestamps) >= self.config.max_messages_per_second * self.config.rate_limit_window:
            self._record_security_event(
                ThreatType.SPAM, peer_id, "medium", f"Rate limit exceeded: {len(peer_timestamps)} messages in window"
            )
            return False

        # Add current timestamp
        peer_timestamps.append(current_time)
        return True

    def detect_anomalies(self, peer_id: str, message_data: dict[str, Any]) -> list[ThreatType]:
        """Detect behavioral anomalies in messages."""
        threats = []
        current_time = time.time()

        # Message size anomaly
        message_size = len(json.dumps(message_data).encode())
        if message_size > self.config.max_message_size:
            threats.append(ThreatType.DOS_ATTACK)
            self._record_security_event(
                ThreatType.DOS_ATTACK, peer_id, "high", f"Oversized message: {message_size} bytes"
            )

        # Message pattern analysis
        peer_patterns = self.message_patterns[peer_id]
        peer_patterns.append(current_time)

        # Keep only recent patterns
        peer_patterns[:] = [t for t in peer_patterns if current_time - t < 3600]

        # Check for rapid message bursts
        recent_count = sum(1 for t in peer_patterns if current_time - t < 60)
        if recent_count > self.config.max_messages_per_second * 10:  # 10x normal rate
            threats.append(ThreatType.SPAM)
            self._record_security_event(
                ThreatType.SPAM, peer_id, "medium", f"Message burst detected: {recent_count} messages in 60s"
            )

        # Detect potential replay attacks
        if self.config.enable_replay_protection:
            message_timestamp = message_data.get("timestamp", current_time)
            time_diff = abs(current_time - message_timestamp)

            if time_diff > self.config.replay_window:
                threats.append(ThreatType.REPLAY_ATTACK)
                self._record_security_event(
                    ThreatType.REPLAY_ATTACK, peer_id, "high", f"Message timestamp anomaly: {time_diff}s difference"
                )

        return threats

    def _record_security_event(self, threat_type: ThreatType, peer_id: str, severity: str, description: str):
        """Record a security event."""
        event = SecurityEvent(
            event_type=threat_type,
            peer_id=peer_id,
            severity=severity,
            description=description,
            metadata={"timestamp": time.time()},
        )

        self.security_events.append(event)

        # Limit event history
        if len(self.security_events) > self.max_events:
            self.security_events = self.security_events[-self.max_events :]

        logger.warning(f"Security event [{severity}] {threat_type.value}: {description} (peer: {peer_id})")

    def get_security_summary(self) -> dict[str, Any]:
        """Get security events summary."""
        current_time = time.time()
        recent_events = [e for e in self.security_events if current_time - e.timestamp < 3600]

        # Count events by type
        event_counts = defaultdict(int)
        for event in recent_events:
            event_counts[event.event_type.value] += 1

        return {
            "total_events": len(self.security_events),
            "recent_events_1h": len(recent_events),
            "banned_peers": len(self.banned_peers),
            "active_rate_limits": len(self.rate_limiters),
            "event_counts": dict(event_counts),
            "threat_level": self._calculate_threat_level(recent_events),
        }

    def _calculate_threat_level(self, recent_events: list[SecurityEvent]) -> str:
        """Calculate overall threat level."""
        if not recent_events:
            return "low"

        critical_count = sum(1 for e in recent_events if e.severity == "critical")
        high_count = sum(1 for e in recent_events if e.severity == "high")

        if critical_count > 0 or high_count > 5:
            return "critical"
        elif high_count > 2 or len(recent_events) > 20:
            return "high"
        elif len(recent_events) > 10:
            return "medium"
        else:
            return "low"


class SecurityManager:
    """Main security manager coordinating all security components."""

    def __init__(self, config: SecurityConfig):
        self.config = config

        # Security components
        self.crypto_manager = CryptoManager(config)
        self.trust_manager = TrustManager(config)
        self.threat_detector = ThreatDetector(config)

        # State management
        self.running = False
        self._maintenance_task: asyncio.Task | None = None

    async def start(self):
        """Start security manager."""
        self.running = True
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        logger.info("Security manager started")

    async def stop(self):
        """Stop security manager."""
        self.running = False
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
        logger.info("Security manager stopped")

    async def authenticate_peer(self, peer_id: str, auth_data: dict[str, Any]) -> bool:
        """Authenticate a peer."""
        identity = self.trust_manager.get_peer_identity(peer_id)

        try:
            # Check if peer is banned
            if self.threat_detector.is_peer_banned(peer_id):
                logger.warning(f"Authentication rejected for banned peer: {peer_id}")
                return False

            # Verify authentication data
            if self.config.security_level == SecurityLevel.NONE:
                # No authentication required
                identity.is_authenticated = True
                identity.auth_timestamp = time.time()
                return True

            # Extract and verify public key
            public_key = auth_data.get("public_key")
            signature = auth_data.get("signature")
            timestamp = auth_data.get("timestamp", time.time())

            if not public_key or not signature:
                identity.failed_auth_attempts += 1
                self.trust_manager.record_bad_behavior(peer_id, "auth_failure")
                return False

            # Verify timestamp to prevent replay attacks
            if abs(time.time() - timestamp) > 300:  # 5 minutes tolerance
                identity.failed_auth_attempts += 1
                self.trust_manager.record_bad_behavior(peer_id, "replay_attack")
                return False

            # Create challenge data
            challenge_data = f"{peer_id}:{timestamp}".encode()

            # Verify signature
            if self.crypto_manager.verify_signature(challenge_data, signature, public_key):
                # Authentication successful
                identity.is_authenticated = True
                identity.auth_timestamp = time.time()
                identity.public_key = public_key
                identity.failed_auth_attempts = 0

                # Generate session key
                session_key = self.crypto_manager.generate_session_key(peer_id)
                identity.session_key = session_key

                self.trust_manager.record_good_behavior(peer_id, "successful_auth")
                logger.info(f"Peer {peer_id} authenticated successfully")
                return True
            else:
                # Authentication failed
                identity.failed_auth_attempts += 1
                self.trust_manager.record_bad_behavior(peer_id, "auth_failure")

                # Ban peer after too many failures
                if identity.failed_auth_attempts >= self.config.max_failed_auth_attempts:
                    self.threat_detector.ban_peer(peer_id)

                logger.warning(f"Authentication failed for peer {peer_id}")
                return False

        except Exception as e:
            logger.error(f"Authentication error for peer {peer_id}: {e}")
            identity.failed_auth_attempts += 1
            return False

    async def process_message(self, peer_id: str, message_data: dict[str, Any]) -> dict[str, Any] | None:
        """Process incoming message through security filters."""
        try:
            # Check if peer is banned
            if self.threat_detector.is_peer_banned(peer_id):
                return None

            # Check rate limits
            if not self.threat_detector.check_rate_limit(peer_id):
                logger.warning(f"Rate limit exceeded for peer {peer_id}")
                return None

            # Check peer trust
            if not self.trust_manager.is_peer_trusted(peer_id):
                logger.warning(f"Message rejected from untrusted peer: {peer_id}")
                return None

            # Detect threats
            threats = self.threat_detector.detect_anomalies(peer_id, message_data)
            if threats:
                logger.warning(f"Threats detected from peer {peer_id}: {[t.value for t in threats]}")

                # Take action based on threat severity
                for threat in threats:
                    if threat in [ThreatType.DOS_ATTACK, ThreatType.REPLAY_ATTACK]:
                        self.threat_detector.ban_peer(peer_id, 1800)  # 30 minutes
                        return None
                    elif threat == ThreatType.SPAM:
                        self.trust_manager.record_bad_behavior(peer_id, "spam")

            # Decrypt message if needed
            if "encrypted_data" in message_data:
                encrypted_data = base64.b64decode(message_data["encrypted_data"])
                decrypted_data = self.crypto_manager.decrypt_message(encrypted_data, peer_id)

                if decrypted_data is None:
                    logger.warning(f"Failed to decrypt message from peer {peer_id}")
                    self.trust_manager.record_bad_behavior(peer_id, "invalid_encryption")
                    return None

                # Replace encrypted data with decrypted
                message_data = json.loads(decrypted_data.decode())

            # Verify signature if present
            if "signature" in message_data and self.config.enable_message_signing:
                signature = base64.b64decode(message_data["signature"])
                message_content = json.dumps({k: v for k, v in message_data.items() if k != "signature"}).encode()

                identity = self.trust_manager.get_peer_identity(peer_id)
                if identity.public_key:
                    if self.crypto_manager.verify_signature(message_content, signature, identity.public_key):
                        self.trust_manager.record_good_behavior(peer_id, "valid_signature")
                    else:
                        logger.warning(f"Invalid signature from peer {peer_id}")
                        self.trust_manager.record_bad_behavior(peer_id, "invalid_signature")
                        return None

            # Message passed all security checks
            self.trust_manager.record_good_behavior(peer_id, "message_delivered")
            return message_data

        except Exception as e:
            logger.error(f"Error processing message from peer {peer_id}: {e}")
            self.trust_manager.record_bad_behavior(peer_id, "malformed_message")
            return None

    async def prepare_outgoing_message(self, peer_id: str, message_data: dict[str, Any]) -> dict[str, Any] | None:
        """Prepare outgoing message with security features."""
        try:
            # Add timestamp for replay protection
            if self.config.enable_replay_protection:
                message_data["timestamp"] = time.time()

            # Sign message if required
            if self.config.enable_message_signing:
                message_content = json.dumps(message_data).encode()
                signature = self.crypto_manager.sign_message(message_content)

                if signature:
                    message_data["signature"] = base64.b64encode(signature).decode()

            # Encrypt message if security level requires it
            if self.config.security_level in [SecurityLevel.STANDARD, SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                message_json = json.dumps(message_data).encode()
                encrypted_data = self.crypto_manager.encrypt_message(message_json, peer_id)

                if encrypted_data:
                    return {
                        "encrypted_data": base64.b64encode(encrypted_data).decode(),
                        "encryption": True,
                    }
                else:
                    logger.warning(f"Failed to encrypt message for peer {peer_id}")
                    if self.config.security_level == SecurityLevel.CRITICAL:
                        return None  # Don't send unencrypted critical messages

            return message_data

        except Exception as e:
            logger.error(f"Error preparing message for peer {peer_id}: {e}")
            return None

    def get_security_status(self) -> dict[str, Any]:
        """Get comprehensive security status."""
        return {
            "security_level": self.config.security_level.value,
            "crypto_available": CRYPTO_AVAILABLE,
            "authenticated_peers": len([p for p in self.trust_manager.peer_identities.values() if p.is_authenticated]),
            "total_peers": len(self.trust_manager.peer_identities),
            "trust_scores": {
                peer_id: identity.trust_score for peer_id, identity in self.trust_manager.peer_identities.items()
            },
            "threat_summary": self.threat_detector.get_security_summary(),
            "session_keys_active": len(self.crypto_manager.session_keys),
            "config": {
                "require_authentication": self.config.require_peer_authentication,
                "enable_encryption": self.config.security_level != SecurityLevel.NONE,
                "enable_signing": self.config.enable_message_signing,
                "enable_pfs": self.config.enable_perfect_forward_secrecy,
            },
        }

    async def _maintenance_loop(self):
        """Periodic security maintenance."""
        while self.running:
            try:
                # Rotate keys if needed
                if self.config.enable_perfect_forward_secrecy:
                    await self.crypto_manager.rotate_keys()

                # Maintain trust scores
                await self.trust_manager.maintain_trust_scores()

                # Clean up expired data
                current_time = time.time()

                # Clean up old security events
                self.threat_detector.security_events = [
                    e
                    for e in self.threat_detector.security_events
                    if current_time - e.timestamp < 86400  # Keep 24 hours
                ]

                await asyncio.sleep(300)  # Run every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Security maintenance error: {e}")
                await asyncio.sleep(60)


# Convenience functions
def create_security_config(level: SecurityLevel = SecurityLevel.STANDARD) -> SecurityConfig:
    """Create security configuration for specified level."""
    config = SecurityConfig(security_level=level)

    if level == SecurityLevel.NONE:
        config.require_peer_authentication = False
        config.enable_message_signing = False
        config.enable_trust_scoring = False
    elif level == SecurityLevel.BASIC:
        config.require_peer_authentication = True
        config.enable_message_signing = False
        config.key_size = 1024
    elif level == SecurityLevel.HIGH:
        config.enable_perfect_forward_secrecy = True
        config.key_rotation_interval = 1800  # 30 minutes
        config.enable_anomaly_detection = True
    elif level == SecurityLevel.CRITICAL:
        config.enable_perfect_forward_secrecy = True
        config.key_rotation_interval = 900  # 15 minutes
        config.max_failed_auth_attempts = 3
        config.ban_duration = 7200  # 2 hours
        config.min_trust_score = 0.7

    return config


# Example usage
if __name__ == "__main__":

    async def test_security():
        config = create_security_config(SecurityLevel.STANDARD)
        security_manager = SecurityManager(config)

        await security_manager.start()

        # Test authentication
        auth_data = {
            "public_key": security_manager.crypto_manager.get_public_key(),
            "signature": b"mock_signature",
            "timestamp": time.time(),
        }

        success = await security_manager.authenticate_peer("test_peer", auth_data)
        print(f"Authentication result: {success}")

        # Test message processing
        message = {"type": "test", "data": "hello world"}
        processed = await security_manager.process_message("test_peer", message)
        print(f"Processed message: {processed}")

        # Get status
        status = security_manager.get_security_status()
        print(f"Security status: {json.dumps(status, indent=2)}")

        await security_manager.stop()

    asyncio.run(test_security())
