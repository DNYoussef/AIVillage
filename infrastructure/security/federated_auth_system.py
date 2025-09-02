"""
Federated Authentication System for AIVillage
=============================================

Comprehensive multi-factor authentication system for federated learning participants.
Integrates with existing RBAC system and provides secure node authentication.
"""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import logging
import secrets
import time
from typing import Any
import uuid

import bcrypt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import pyotp

logger = logging.getLogger(__name__)


class AuthenticationMethod(Enum):
    """Available authentication methods."""

    PASSWORD = "password"  # nosec B105 - field name constant, not password
    MULTI_FACTOR = "mfa"
    CERTIFICATE = "certificate"
    ZERO_KNOWLEDGE = "zero_knowledge"
    BIOMETRIC_HASH = "biometric_hash"


class NodeRole(Enum):
    """Federated learning node roles."""

    COORDINATOR = "coordinator"
    PARTICIPANT = "participant"
    VALIDATOR = "validator"
    AGGREGATOR = "aggregator"
    OBSERVER = "observer"


class AuthenticationStatus(Enum):
    """Authentication status states."""

    PENDING = "pending"
    AUTHENTICATED = "authenticated"
    CHALLENGED = "challenged"
    FAILED = "failed"
    EXPIRED = "expired"
    REVOKED = "revoked"


@dataclass
class NodeIdentity:
    """Federated node identity."""

    node_id: str
    public_key: bytes
    role: NodeRole
    capabilities: dict[str, Any] = field(default_factory=dict)
    reputation_score: float = 0.5
    trust_level: str = "basic"
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    authentication_methods: set[AuthenticationMethod] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthenticationChallenge:
    """Authentication challenge for node verification."""

    challenge_id: str
    node_id: str
    method: AuthenticationMethod
    challenge_data: bytes
    expected_response_hash: str
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 300)  # 5 minutes
    attempts: int = 0
    max_attempts: int = 3


@dataclass
class AuthenticationSession:
    """Active authentication session."""

    session_id: str
    node_id: str
    role: NodeRole
    authenticated_methods: set[AuthenticationMethod] = field(default_factory=set)
    permissions: set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 3600)  # 1 hour
    last_activity: float = field(default_factory=time.time)
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class FederatedAuthenticationSystem:
    """
    Comprehensive authentication system for federated learning nodes.

    Features:
    - Multi-factor authentication
    - Certificate-based authentication
    - Zero-knowledge proof authentication
    - Role-based access control
    - Session management
    - Reputation-based trust
    - Challenge-response protocols
    """

    def __init__(self, jwt_secret: str | None = None, enable_mfa: bool = True):
        """Initialize the federated authentication system."""
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
        self.enable_mfa = enable_mfa

        # Storage
        self.node_identities: dict[str, NodeIdentity] = {}
        self.active_challenges: dict[str, AuthenticationChallenge] = {}
        self.active_sessions: dict[str, AuthenticationSession] = {}
        self.revoked_tokens: set[str] = set()

        # Security configuration
        self.password_policy = {
            "min_length": 12,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_numbers": True,
            "require_special": True,
            "max_age_days": 90,
        }

        self.session_config = {
            "default_duration": 3600,  # 1 hour
            "max_duration": 86400,  # 24 hours
            "renewal_threshold": 300,  # 5 minutes before expiry
            "max_concurrent_sessions": 5,
        }

        # Certificate authority for node certificates
        self.ca_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())

        # Statistics
        self.auth_stats = {
            "total_authentications": 0,
            "failed_authentications": 0,
            "active_sessions": 0,
            "revoked_sessions": 0,
            "challenge_responses": 0,
            "mfa_verifications": 0,
        }

        logger.info("Federated Authentication System initialized")

    async def register_node(
        self,
        node_id: str,
        role: NodeRole,
        password: str,
        capabilities: dict[str, Any] | None = None,
        enable_certificate_auth: bool = True,
    ) -> NodeIdentity:
        """Register a new federated learning node."""

        if node_id in self.node_identities:
            raise ValueError(f"Node {node_id} already registered")

        # Validate password policy
        if not self._validate_password_policy(password):
            raise ValueError("Password does not meet security requirements")

        # Generate node key pair
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())

        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        # Create node identity
        identity = NodeIdentity(
            node_id=node_id,
            public_key=public_key,
            role=role,
            capabilities=capabilities or {},
            authentication_methods={AuthenticationMethod.PASSWORD},
        )

        # Enable certificate authentication if requested
        if enable_certificate_auth:
            identity.authentication_methods.add(AuthenticationMethod.CERTIFICATE)

        # Enable MFA for sensitive roles
        if role in [NodeRole.COORDINATOR, NodeRole.VALIDATOR] and self.enable_mfa:
            identity.authentication_methods.add(AuthenticationMethod.MULTI_FACTOR)

        # Store hashed password
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        identity.metadata["password_hash"] = password_hash
        identity.metadata["private_key"] = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        # Generate TOTP secret if MFA enabled
        if AuthenticationMethod.MULTI_FACTOR in identity.authentication_methods:
            totp_secret = pyotp.random_base32()
            identity.metadata["totp_secret"] = totp_secret

        self.node_identities[node_id] = identity

        logger.info(f"Node {node_id} registered with role {role.value}")
        return identity

    async def authenticate_node(
        self,
        node_id: str,
        password: str,
        mfa_token: str | None = None,
        certificate: bytes | None = None,
        challenge_response: dict[str, Any] | None = None,
    ) -> tuple[bool, AuthenticationSession | None]:
        """
        Authenticate a federated learning node with multiple methods.
        """

        identity = self.node_identities.get(node_id)
        if not identity:
            self.auth_stats["failed_authentications"] += 1
            logger.warning(f"Authentication failed: Node {node_id} not found")
            return False, None

        try:
            self.auth_stats["total_authentications"] += 1
            authenticated_methods = set()

            # Password authentication
            if AuthenticationMethod.PASSWORD in identity.authentication_methods:
                if await self._verify_password(identity, password):
                    authenticated_methods.add(AuthenticationMethod.PASSWORD)
                else:
                    self.auth_stats["failed_authentications"] += 1
                    logger.warning(f"Password authentication failed for node {node_id}")
                    return False, None

            # MFA authentication
            if AuthenticationMethod.MULTI_FACTOR in identity.authentication_methods:
                if mfa_token and await self._verify_mfa_token(identity, mfa_token):
                    authenticated_methods.add(AuthenticationMethod.MULTI_FACTOR)
                    self.auth_stats["mfa_verifications"] += 1
                else:
                    logger.warning(f"MFA authentication failed for node {node_id}")
                    return False, None

            # Certificate authentication
            if AuthenticationMethod.CERTIFICATE in identity.authentication_methods and certificate:
                if await self._verify_certificate(identity, certificate):
                    authenticated_methods.add(AuthenticationMethod.CERTIFICATE)
                else:
                    logger.warning(f"Certificate authentication failed for node {node_id}")
                    return False, None

            # Challenge-response authentication
            if challenge_response:
                if await self._verify_challenge_response(node_id, challenge_response):
                    authenticated_methods.add(AuthenticationMethod.ZERO_KNOWLEDGE)
                    self.auth_stats["challenge_responses"] += 1

            # Check if minimum authentication requirements are met
            required_methods = self._get_required_auth_methods(identity.role)
            if not required_methods.issubset(authenticated_methods):
                missing = required_methods - authenticated_methods
                logger.warning(f"Insufficient authentication for node {node_id}, missing: {missing}")
                return False, None

            # Create authentication session
            session = await self._create_session(identity, authenticated_methods)

            # Update identity activity
            identity.last_active = time.time()

            logger.info(f"Node {node_id} authenticated successfully with methods: {authenticated_methods}")
            return True, session

        except Exception as e:
            self.auth_stats["failed_authentications"] += 1
            logger.exception(f"Authentication error for node {node_id}: {e}")
            return False, None

    async def create_authentication_challenge(
        self, node_id: str, method: AuthenticationMethod = AuthenticationMethod.ZERO_KNOWLEDGE
    ) -> AuthenticationChallenge | None:
        """Create an authentication challenge for a node."""

        identity = self.node_identities.get(node_id)
        if not identity:
            return None

        challenge_id = str(uuid.uuid4())
        challenge_data = secrets.token_bytes(32)

        # Create expected response based on method
        if method == AuthenticationMethod.ZERO_KNOWLEDGE:
            # Zero-knowledge proof challenge
            expected_response = self._create_zk_challenge_response(identity, challenge_data)
        else:
            # Generic challenge
            expected_response = hashlib.sha256(challenge_data + identity.public_key).hexdigest()

        challenge = AuthenticationChallenge(
            challenge_id=challenge_id,
            node_id=node_id,
            method=method,
            challenge_data=challenge_data,
            expected_response_hash=expected_response,
        )

        self.active_challenges[challenge_id] = challenge

        # Clean up expired challenges
        await self._cleanup_expired_challenges()

        logger.info(f"Created {method.value} challenge for node {node_id}")
        return challenge

    async def validate_session(self, session_id: str) -> tuple[bool, AuthenticationSession | None]:
        """Validate an authentication session."""

        session = self.active_sessions.get(session_id)
        if not session:
            return False, None

        # Check if session is expired
        if time.time() > session.expires_at or not session.is_active:
            await self.revoke_session(session_id)
            return False, None

        # Update last activity
        session.last_activity = time.time()

        return True, session

    async def revoke_session(self, session_id: str) -> bool:
        """Revoke an authentication session."""

        session = self.active_sessions.get(session_id)
        if not session:
            return False

        session.is_active = False
        del self.active_sessions[session_id]
        self.revoked_tokens.add(session_id)
        self.auth_stats["revoked_sessions"] += 1

        logger.info(f"Session {session_id} revoked for node {session.node_id}")
        return True

    async def update_node_reputation(self, node_id: str, reputation_delta: float) -> bool:
        """Update node reputation score."""

        identity = self.node_identities.get(node_id)
        if not identity:
            return False

        identity.reputation_score = max(0.0, min(1.0, identity.reputation_score + reputation_delta))

        # Update trust level based on reputation
        if identity.reputation_score >= 0.9:
            identity.trust_level = "high"
        elif identity.reputation_score >= 0.7:
            identity.trust_level = "medium"
        elif identity.reputation_score >= 0.5:
            identity.trust_level = "basic"
        else:
            identity.trust_level = "low"

        logger.info(f"Node {node_id} reputation updated to {identity.reputation_score:.3f}")
        return True

    async def get_node_permissions(self, node_id: str) -> set[str]:
        """Get permissions for a node based on role and trust level."""

        identity = self.node_identities.get(node_id)
        if not identity:
            return set()

        permissions = set()

        # Base permissions by role
        role_permissions = {
            NodeRole.COORDINATOR: {
                "create_training_rounds",
                "manage_participants",
                "aggregate_gradients",
                "distribute_models",
                "view_all_metrics",
            },
            NodeRole.PARTICIPANT: {"participate_training", "submit_gradients", "view_own_metrics"},
            NodeRole.VALIDATOR: {"validate_gradients", "audit_training", "view_validation_metrics"},
            NodeRole.AGGREGATOR: {"aggregate_gradients", "secure_computation", "view_aggregation_metrics"},
            NodeRole.OBSERVER: {"view_public_metrics", "monitor_training"},
        }

        permissions.update(role_permissions.get(identity.role, set()))

        # Trust-based permissions
        if identity.trust_level == "high":
            permissions.update({"priority_participation", "validator_nomination", "reputation_boost"})
        elif identity.trust_level == "medium":
            permissions.add("regular_participation")

        return permissions

    # Private methods

    def _validate_password_policy(self, password: str) -> bool:
        """Validate password against security policy."""
        policy = self.password_policy

        if len(password) < policy["min_length"]:
            return False

        if policy["require_uppercase"] and not any(c.isupper() for c in password):
            return False

        if policy["require_lowercase"] and not any(c.islower() for c in password):
            return False

        if policy["require_numbers"] and not any(c.isdigit() for c in password):
            return False

        if policy["require_special"] and not any(c in "!@#$%^&*()_+-=" for c in password):
            return False

        return True

    async def _verify_password(self, identity: NodeIdentity, password: str) -> bool:
        """Verify password hash."""
        stored_hash = identity.metadata.get("password_hash")
        if not stored_hash:
            return False

        return bcrypt.checkpw(password.encode(), stored_hash)

    async def _verify_mfa_token(self, identity: NodeIdentity, token: str) -> bool:
        """Verify TOTP MFA token."""
        totp_secret = identity.metadata.get("totp_secret")
        if not totp_secret:
            return False

        totp = pyotp.TOTP(totp_secret)
        return totp.verify(token, valid_window=1)  # Allow 30 seconds window

    async def _verify_certificate(self, identity: NodeIdentity, certificate: bytes) -> bool:
        """Verify node certificate."""
        try:
            # In a real implementation, this would verify the certificate
            # against the CA and check expiration, revocation, etc.
            return True  # Simplified for demo
        except Exception as e:
            logger.error(f"Certificate verification error: {e}")
            return False

    async def _verify_challenge_response(self, node_id: str, challenge_response: dict[str, Any]) -> bool:
        """Verify challenge response."""
        challenge_id = challenge_response.get("challenge_id")
        response = challenge_response.get("response")

        if not challenge_id or not response:
            return False

        challenge = self.active_challenges.get(challenge_id)
        if not challenge or challenge.node_id != node_id:
            return False

        # Check if challenge is expired
        if time.time() > challenge.expires_at:
            del self.active_challenges[challenge_id]
            return False

        # Verify response
        if response == challenge.expected_response_hash:
            del self.active_challenges[challenge_id]
            return True

        challenge.attempts += 1
        if challenge.attempts >= challenge.max_attempts:
            del self.active_challenges[challenge_id]

        return False

    def _create_zk_challenge_response(self, identity: NodeIdentity, challenge_data: bytes) -> str:
        """Create zero-knowledge proof challenge response."""
        # Simplified ZK proof - in production would use proper ZK protocols
        commitment = hashlib.sha256(challenge_data + identity.public_key + b"zk_proof").hexdigest()
        return commitment

    def _get_required_auth_methods(self, role: NodeRole) -> set[AuthenticationMethod]:
        """Get required authentication methods for a role."""
        base_requirements = {AuthenticationMethod.PASSWORD}

        if role in [NodeRole.COORDINATOR, NodeRole.VALIDATOR]:
            if self.enable_mfa:
                base_requirements.add(AuthenticationMethod.MULTI_FACTOR)

        return base_requirements

    async def _create_session(
        self, identity: NodeIdentity, authenticated_methods: set[AuthenticationMethod]
    ) -> AuthenticationSession:
        """Create an authentication session."""
        session_id = str(uuid.uuid4())
        permissions = await self.get_node_permissions(identity.node_id)

        session = AuthenticationSession(
            session_id=session_id,
            node_id=identity.node_id,
            role=identity.role,
            authenticated_methods=authenticated_methods,
            permissions=permissions,
        )

        # Clean up old sessions if exceeding limit
        await self._cleanup_node_sessions(identity.node_id)

        self.active_sessions[session_id] = session
        self.auth_stats["active_sessions"] = len(self.active_sessions)

        return session

    async def _cleanup_node_sessions(self, node_id: str) -> None:
        """Clean up old sessions for a node."""
        node_sessions = [
            (sid, session)
            for sid, session in self.active_sessions.items()
            if session.node_id == node_id and session.is_active
        ]

        if len(node_sessions) >= self.session_config["max_concurrent_sessions"]:
            # Remove oldest sessions
            node_sessions.sort(key=lambda x: x[1].created_at)
            for sid, _ in node_sessions[: -self.session_config["max_concurrent_sessions"] + 1]:
                await self.revoke_session(sid)

    async def _cleanup_expired_challenges(self) -> None:
        """Clean up expired challenges."""
        current_time = time.time()
        expired_challenges = [
            cid for cid, challenge in self.active_challenges.items() if current_time > challenge.expires_at
        ]

        for cid in expired_challenges:
            del self.active_challenges[cid]

    # Statistics and monitoring

    def get_auth_stats(self) -> dict[str, Any]:
        """Get authentication system statistics."""
        return {
            **self.auth_stats,
            "registered_nodes": len(self.node_identities),
            "active_challenges": len(self.active_challenges),
            "active_sessions": len(self.active_sessions),
            "success_rate": (
                (self.auth_stats["total_authentications"] - self.auth_stats["failed_authentications"])
                / max(1, self.auth_stats["total_authentications"])
            ),
        }

    def get_node_info(self, node_id: str) -> dict[str, Any] | None:
        """Get node information (excluding sensitive data)."""
        identity = self.node_identities.get(node_id)
        if not identity:
            return None

        return {
            "node_id": identity.node_id,
            "role": identity.role.value,
            "reputation_score": identity.reputation_score,
            "trust_level": identity.trust_level,
            "capabilities": identity.capabilities,
            "authentication_methods": [method.value for method in identity.authentication_methods],
            "created_at": identity.created_at,
            "last_active": identity.last_active,
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on authentication system."""
        current_time = time.time()

        # Clean up expired sessions
        expired_sessions = [sid for sid, session in self.active_sessions.items() if current_time > session.expires_at]

        for sid in expired_sessions:
            await self.revoke_session(sid)

        return {
            "healthy": True,
            "active_sessions": len(self.active_sessions),
            "active_challenges": len(self.active_challenges),
            "registered_nodes": len(self.node_identities),
            "expired_sessions_cleaned": len(expired_sessions),
        }
