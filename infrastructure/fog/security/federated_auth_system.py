"""
Federated Authentication and Authorization System

This module provides comprehensive authentication and authorization mechanisms
for federated learning networks with BetaNet integration, including:
- Distributed identity management
- Multi-factor authentication
- Role-based access control (RBAC)
- Capability-based security
- Session management with revocation
- Cross-node authentication protocols
"""

import asyncio
import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import jwt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding

logger = logging.getLogger(__name__)


class NodeRole(Enum):
    """Node roles in the federated system"""

    COORDINATOR = "coordinator"
    TRAINER = "trainer"
    AGGREGATOR = "aggregator"
    VALIDATOR = "validator"
    OBSERVER = "observer"
    ADMIN = "admin"


class AuthenticationMethod(Enum):
    """Supported authentication methods"""

    PASSWORD = "password"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"
    HARDWARE_TOKEN = "hardware_token"
    MULTI_FACTOR = "multi_factor"


class SessionStatus(Enum):
    """Session status types"""

    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


@dataclass
class NodeIdentity:
    """Node identity information"""

    node_id: str
    public_key: bytes
    certificate: Optional[bytes] = None
    roles: Set[NodeRole] = field(default_factory=set)
    capabilities: Set[str] = field(default_factory=set)
    reputation_score: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthenticationChallenge:
    """Authentication challenge structure"""

    challenge_id: str
    node_id: str
    challenge_data: bytes
    method: AuthenticationMethod
    expires_at: float
    nonce: bytes
    difficulty: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthenticationResponse:
    """Authentication response structure"""

    challenge_id: str
    node_id: str
    response_data: bytes
    proof: bytes
    timestamp: float
    method: AuthenticationMethod
    additional_factors: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionToken:
    """Secure session token"""

    token_id: str
    node_id: str
    roles: Set[NodeRole]
    capabilities: Set[str]
    issued_at: float
    expires_at: float
    status: SessionStatus = SessionStatus.ACTIVE
    refresh_token: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessRequest:
    """Access request for resources or operations"""

    request_id: str
    node_id: str
    resource_path: str
    operation: str
    required_capabilities: Set[str]
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class CryptographicKeyManager:
    """
    Manages cryptographic keys for authentication and authorization
    """

    def __init__(self):
        self.node_keypairs: Dict[str, Tuple[bytes, bytes]] = {}  # private, public
        self.trusted_certificates: Dict[str, bytes] = {}
        self.revoked_certificates: Set[str] = set()
        self.key_rotation_schedule: Dict[str, float] = {}

    def generate_node_keypair(self, node_id: str) -> Tuple[bytes, bytes]:
        """Generate RSA keypair for a node"""
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        self.node_keypairs[node_id] = (private_pem, public_pem)
        return private_pem, public_pem

    def sign_data(self, node_id: str, data: bytes) -> bytes:
        """Sign data with node's private key"""
        if node_id not in self.node_keypairs:
            raise ValueError(f"No keypair found for node {node_id}")

        private_pem, _ = self.node_keypairs[node_id]
        private_key = serialization.load_pem_private_key(private_pem, password=None)

        signature = private_key.sign(
            data, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256()
        )

        return signature

    def verify_signature(self, node_id: str, data: bytes, signature: bytes) -> bool:
        """Verify signature with node's public key"""
        if node_id not in self.node_keypairs:
            return False

        try:
            _, public_pem = self.node_keypairs[node_id]
            public_key = serialization.load_pem_public_key(public_pem)

            public_key.verify(
                signature,
                data,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )
            return True
        except Exception as e:
            logger.warning(f"Signature verification failed for {node_id}: {e}")
            return False

    def encrypt_for_node(self, node_id: str, data: bytes) -> bytes:
        """Encrypt data for a specific node"""
        if node_id not in self.node_keypairs:
            raise ValueError(f"No public key found for node {node_id}")

        _, public_pem = self.node_keypairs[node_id]
        public_key = serialization.load_pem_public_key(public_pem)

        ciphertext = public_key.encrypt(
            data, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
        )

        return ciphertext

    def decrypt_from_node(self, node_id: str, ciphertext: bytes) -> bytes:
        """Decrypt data from a specific node"""
        if node_id not in self.node_keypairs:
            raise ValueError(f"No private key found for node {node_id}")

        private_pem, _ = self.node_keypairs[node_id]
        private_key = serialization.load_pem_private_key(private_pem, password=None)

        plaintext = private_key.decrypt(
            ciphertext, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
        )

        return plaintext


class MultiFactorAuthenticator:
    """
    Multi-factor authentication system for enhanced security
    """

    def __init__(self):
        self.pending_challenges: Dict[str, AuthenticationChallenge] = {}
        self.completed_authentications: Dict[str, Dict[str, Any]] = {}
        self.factor_weights = {
            AuthenticationMethod.PASSWORD: 0.3,
            AuthenticationMethod.CERTIFICATE: 0.4,
            AuthenticationMethod.BIOMETRIC: 0.5,
            AuthenticationMethod.HARDWARE_TOKEN: 0.6,
        }

    async def create_authentication_challenge(
        self, node_id: str, required_methods: List[AuthenticationMethod], context: Optional[Dict[str, Any]] = None
    ) -> List[AuthenticationChallenge]:
        """
        Create multi-factor authentication challenges

        Args:
            node_id: Node requesting authentication
            required_methods: Required authentication methods
            context: Additional context for challenge generation

        Returns:
            List of authentication challenges
        """
        challenges = []

        for method in required_methods:
            challenge_id = f"{node_id}_{method.value}_{secrets.token_hex(8)}"
            nonce = secrets.token_bytes(32)

            # Generate method-specific challenge data
            if method == AuthenticationMethod.PASSWORD:
                challenge_data = await self._create_password_challenge(node_id, nonce)
            elif method == AuthenticationMethod.CERTIFICATE:
                challenge_data = await self._create_certificate_challenge(node_id, nonce)
            elif method == AuthenticationMethod.BIOMETRIC:
                challenge_data = await self._create_biometric_challenge(node_id, nonce)
            elif method == AuthenticationMethod.HARDWARE_TOKEN:
                challenge_data = await self._create_hardware_token_challenge(node_id, nonce)
            else:
                challenge_data = nonce  # Default challenge

            challenge = AuthenticationChallenge(
                challenge_id=challenge_id,
                node_id=node_id,
                challenge_data=challenge_data,
                method=method,
                expires_at=time.time() + 300,  # 5 minutes
                nonce=nonce,
                difficulty=self._calculate_challenge_difficulty(method, context),
                metadata=context or {},
            )

            challenges.append(challenge)
            self.pending_challenges[challenge_id] = challenge

        logger.info(f"Created {len(challenges)} authentication challenges for node {node_id}")
        return challenges

    async def _create_password_challenge(self, node_id: str, nonce: bytes) -> bytes:
        """Create password-based challenge"""
        # Challenge: prove knowledge of password without revealing it
        salt = secrets.token_bytes(16)
        challenge_data = {"type": "password_proof", "salt": salt.hex(), "nonce": nonce.hex(), "rounds": 10000}
        return json.dumps(challenge_data).encode()

    async def _create_certificate_challenge(self, node_id: str, nonce: bytes) -> bytes:
        """Create certificate-based challenge"""
        challenge_data = {
            "type": "certificate_proof",
            "nonce": nonce.hex(),
            "timestamp": time.time(),
            "node_id": node_id,
        }
        return json.dumps(challenge_data).encode()

    async def _create_biometric_challenge(self, node_id: str, nonce: bytes) -> bytes:
        """Create biometric challenge"""
        challenge_data = {
            "type": "biometric_proof",
            "nonce": nonce.hex(),
            "template_hash": hashlib.sha256(f"{node_id}_biometric".encode()).hexdigest(),
            "challenge_pattern": secrets.token_hex(16),
        }
        return json.dumps(challenge_data).encode()

    async def _create_hardware_token_challenge(self, node_id: str, nonce: bytes) -> bytes:
        """Create hardware token challenge"""
        current_time = int(time.time() // 30)  # TOTP-style time window
        challenge_data = {
            "type": "hardware_token",
            "nonce": nonce.hex(),
            "time_window": current_time,
            "token_serial": f"hw_token_{node_id}",
        }
        return json.dumps(challenge_data).encode()

    def _calculate_challenge_difficulty(self, method: AuthenticationMethod, context: Optional[Dict[str, Any]]) -> int:
        """Calculate appropriate challenge difficulty"""
        base_difficulty = 1

        if context:
            risk_level = context.get("risk_level", "medium")
            if risk_level == "high":
                base_difficulty = 3
            elif risk_level == "critical":
                base_difficulty = 5

        return base_difficulty

    async def verify_authentication_response(self, response: AuthenticationResponse) -> Dict[str, Any]:
        """
        Verify multi-factor authentication response

        Args:
            response: Authentication response from node

        Returns:
            Verification result with confidence score
        """
        challenge_id = response.challenge_id

        if challenge_id not in self.pending_challenges:
            return {"success": False, "error": "Challenge not found or expired", "confidence": 0.0}

        challenge = self.pending_challenges[challenge_id]

        # Check expiration
        if time.time() > challenge.expires_at:
            del self.pending_challenges[challenge_id]
            return {"success": False, "error": "Challenge expired", "confidence": 0.0}

        # Verify response based on method
        verification_result = await self._verify_method_response(challenge, response)

        if verification_result["success"]:
            # Store successful authentication
            if response.node_id not in self.completed_authentications:
                self.completed_authentications[response.node_id] = {}

            self.completed_authentications[response.node_id][challenge.method.value] = {
                "timestamp": time.time(),
                "confidence": verification_result["confidence"],
                "challenge_id": challenge_id,
            }

            # Clean up challenge
            del self.pending_challenges[challenge_id]

            logger.info(f"Authentication successful for {response.node_id} using {challenge.method.value}")

        return verification_result

    async def _verify_method_response(
        self, challenge: AuthenticationChallenge, response: AuthenticationResponse
    ) -> Dict[str, Any]:
        """Verify response for specific authentication method"""
        try:
            if challenge.method == AuthenticationMethod.PASSWORD:
                return await self._verify_password_response(challenge, response)
            elif challenge.method == AuthenticationMethod.CERTIFICATE:
                return await self._verify_certificate_response(challenge, response)
            elif challenge.method == AuthenticationMethod.BIOMETRIC:
                return await self._verify_biometric_response(challenge, response)
            elif challenge.method == AuthenticationMethod.HARDWARE_TOKEN:
                return await self._verify_hardware_token_response(challenge, response)
            else:
                return {"success": False, "error": "Unsupported method", "confidence": 0.0}
        except Exception as e:
            logger.error(f"Authentication verification error: {e}")
            return {"success": False, "error": str(e), "confidence": 0.0}

    async def _verify_password_response(
        self, challenge: AuthenticationChallenge, response: AuthenticationResponse
    ) -> Dict[str, Any]:
        """Verify password-based response"""
        challenge_data = json.loads(challenge.challenge_data.decode())

        # In production, verify against stored password hash
        # This is a simplified verification
        hashlib.pbkdf2_hmac(
            "sha256",
            response.response_data,  # Password hash
            bytes.fromhex(challenge_data["salt"]),
            challenge_data["rounds"],
        )

        # Verify proof includes correct nonce
        proof_data = json.loads(response.proof.decode())
        if proof_data.get("nonce") != challenge.nonce.hex():
            return {"success": False, "error": "Invalid nonce", "confidence": 0.0}

        return {"success": True, "confidence": self.factor_weights[AuthenticationMethod.PASSWORD], "method": "password"}

    async def _verify_certificate_response(
        self, challenge: AuthenticationChallenge, response: AuthenticationResponse
    ) -> Dict[str, Any]:
        """Verify certificate-based response"""
        # Verify certificate signature on challenge data
        # In production, verify against trusted CA

        json.loads(challenge.challenge_data.decode())

        # Simplified certificate verification

        # Verify signature covers challenge nonce and timestamp
        challenge.challenge_data + challenge.nonce

        # In production, use actual certificate verification
        return {
            "success": True,
            "confidence": self.factor_weights[AuthenticationMethod.CERTIFICATE],
            "method": "certificate",
        }

    async def _verify_biometric_response(
        self, challenge: AuthenticationChallenge, response: AuthenticationResponse
    ) -> Dict[str, Any]:
        """Verify biometric response"""
        challenge_data = json.loads(challenge.challenge_data.decode())

        # Simplified biometric verification
        # In production, use proper biometric matching algorithms
        biometric_hash = hashlib.sha256(response.response_data).hexdigest()
        expected_hash = challenge_data["template_hash"]

        # Calculate similarity score (simplified)
        similarity = 1.0 if biometric_hash == expected_hash else 0.8

        return {
            "success": similarity > 0.7,
            "confidence": self.factor_weights[AuthenticationMethod.BIOMETRIC] * similarity,
            "method": "biometric",
        }

    async def _verify_hardware_token_response(
        self, challenge: AuthenticationChallenge, response: AuthenticationResponse
    ) -> Dict[str, Any]:
        """Verify hardware token response"""
        challenge_data = json.loads(challenge.challenge_data.decode())

        # TOTP-style verification
        current_time_window = int(time.time() // 30)
        challenge_time_window = challenge_data["time_window"]

        # Allow one time window tolerance
        if abs(current_time_window - challenge_time_window) > 1:
            return {"success": False, "error": "Token expired", "confidence": 0.0}

        # Verify TOTP code (simplified)
        token_code = response.response_data.decode()

        return {
            "success": len(token_code) == 6 and token_code.isdigit(),
            "confidence": self.factor_weights[AuthenticationMethod.HARDWARE_TOKEN],
            "method": "hardware_token",
        }

    def get_authentication_score(self, node_id: str) -> float:
        """Calculate overall authentication confidence score"""
        if node_id not in self.completed_authentications:
            return 0.0

        authentications = self.completed_authentications[node_id]
        total_score = 0.0

        for method_name, auth_data in authentications.items():
            # Decay confidence over time
            age_hours = (time.time() - auth_data["timestamp"]) / 3600
            decay_factor = max(0.1, 1.0 - (age_hours / 24))  # Decay over 24 hours

            score = auth_data["confidence"] * decay_factor
            total_score += score

        return min(1.0, total_score)  # Cap at 1.0


class RoleBasedAccessControl:
    """
    Role-based access control system for federated nodes
    """

    def __init__(self):
        self.role_permissions: Dict[NodeRole, Set[str]] = {
            NodeRole.ADMIN: {
                "system.manage",
                "node.create",
                "node.delete",
                "node.suspend",
                "data.read",
                "data.write",
                "data.delete",
                "model.train",
                "model.aggregate",
                "model.validate",
                "model.deploy",
                "audit.read",
                "config.modify",
            },
            NodeRole.COORDINATOR: {
                "node.manage",
                "model.train",
                "model.aggregate",
                "model.validate",
                "data.read",
                "audit.read",
            },
            NodeRole.TRAINER: {"model.train", "data.read", "gradient.submit"},
            NodeRole.AGGREGATOR: {"model.aggregate", "gradient.aggregate", "model.validate"},
            NodeRole.VALIDATOR: {"model.validate", "audit.read"},
            NodeRole.OBSERVER: {"audit.read", "metrics.read"},
        }

        self.capability_matrix: Dict[str, Dict[str, Any]] = {
            "model.train": {
                "description": "Train machine learning models",
                "risk_level": "medium",
                "required_reputation": 0.7,
            },
            "model.aggregate": {
                "description": "Aggregate model updates",
                "risk_level": "high",
                "required_reputation": 0.8,
            },
            "gradient.submit": {
                "description": "Submit gradient updates",
                "risk_level": "medium",
                "required_reputation": 0.6,
            },
            "node.manage": {"description": "Manage node operations", "risk_level": "high", "required_reputation": 0.9},
            "data.write": {"description": "Write training data", "risk_level": "high", "required_reputation": 0.8},
            "system.manage": {
                "description": "Manage system configuration",
                "risk_level": "critical",
                "required_reputation": 0.95,
            },
        }

        self.dynamic_permissions: Dict[str, Set[str]] = {}

    def get_node_capabilities(self, node_identity: NodeIdentity) -> Set[str]:
        """Get all capabilities for a node based on roles and dynamic permissions"""
        capabilities = set()

        # Add role-based permissions
        for role in node_identity.roles:
            if role in self.role_permissions:
                capabilities.update(self.role_permissions[role])

        # Add explicit capabilities
        capabilities.update(node_identity.capabilities)

        # Add dynamic permissions
        if node_identity.node_id in self.dynamic_permissions:
            capabilities.update(self.dynamic_permissions[node_identity.node_id])

        # Filter based on reputation
        filtered_capabilities = set()
        for capability in capabilities:
            if capability in self.capability_matrix:
                required_reputation = self.capability_matrix[capability]["required_reputation"]
                if node_identity.reputation_score >= required_reputation:
                    filtered_capabilities.add(capability)
            else:
                filtered_capabilities.add(capability)  # Unknown capabilities allowed

        return filtered_capabilities

    def check_access_permission(self, node_identity: NodeIdentity, access_request: AccessRequest) -> Dict[str, Any]:
        """Check if node has permission for requested access"""
        node_capabilities = self.get_node_capabilities(node_identity)

        # Check if all required capabilities are present
        missing_capabilities = access_request.required_capabilities - node_capabilities

        if missing_capabilities:
            return {
                "granted": False,
                "reason": "insufficient_capabilities",
                "missing": list(missing_capabilities),
                "node_capabilities": list(node_capabilities),
            }

        # Check reputation requirements
        max_risk_level = "low"
        min_required_reputation = 0.0

        for capability in access_request.required_capabilities:
            if capability in self.capability_matrix:
                cap_info = self.capability_matrix[capability]
                risk_level = cap_info["risk_level"]
                required_reputation = cap_info["required_reputation"]

                if required_reputation > min_required_reputation:
                    min_required_reputation = required_reputation
                    max_risk_level = risk_level

        if node_identity.reputation_score < min_required_reputation:
            return {
                "granted": False,
                "reason": "insufficient_reputation",
                "required_reputation": min_required_reputation,
                "current_reputation": node_identity.reputation_score,
                "risk_level": max_risk_level,
            }

        # Additional context-based checks
        context_result = self._check_contextual_access(node_identity, access_request)
        if not context_result["allowed"]:
            return {"granted": False, "reason": "context_restriction", "details": context_result["details"]}

        return {
            "granted": True,
            "capabilities_used": list(access_request.required_capabilities),
            "risk_level": max_risk_level,
            "reputation_score": node_identity.reputation_score,
        }

    def _check_contextual_access(self, node_identity: NodeIdentity, access_request: AccessRequest) -> Dict[str, Any]:
        """Check contextual access restrictions"""
        context = access_request.context

        # Time-based restrictions
        if "time_restrictions" in context:
            current_hour = datetime.now(UTC).hour
            allowed_hours = context["time_restrictions"].get("allowed_hours", [])
            if allowed_hours and current_hour not in allowed_hours:
                return {"allowed": False, "details": f"Access not allowed at hour {current_hour}"}

        # Resource-based restrictions
        if "resource_limits" in context:
            resource_type = access_request.resource_path.split("/")[0]
            limits = context["resource_limits"].get(resource_type, {})

            # Check rate limiting
            if "rate_limit" in limits:
                # In production, implement proper rate limiting
                pass

        # Geographic restrictions
        if "geo_restrictions" in context:
            node_location = node_identity.metadata.get("location", {})
            allowed_regions = context["geo_restrictions"].get("allowed_regions", [])

            if allowed_regions and node_location.get("region") not in allowed_regions:
                return {"allowed": False, "details": "Geographic restriction violated"}

        return {"allowed": True, "details": "All contextual checks passed"}

    def grant_temporary_permission(self, node_id: str, capabilities: Set[str], duration_seconds: int = 3600) -> str:
        """Grant temporary permissions to a node"""
        permission_id = f"temp_{node_id}_{secrets.token_hex(8)}"

        if node_id not in self.dynamic_permissions:
            self.dynamic_permissions[node_id] = set()

        self.dynamic_permissions[node_id].update(capabilities)

        # Schedule removal (simplified - in production use proper scheduler)
        async def remove_permission():
            await asyncio.sleep(duration_seconds)
            if node_id in self.dynamic_permissions:
                self.dynamic_permissions[node_id] -= capabilities
                if not self.dynamic_permissions[node_id]:
                    del self.dynamic_permissions[node_id]

        asyncio.create_task(remove_permission())

        logger.info(f"Granted temporary permissions to {node_id}: {capabilities}")
        return permission_id


class FederatedSessionManager:
    """
    Session management for federated authentication
    """

    def __init__(self, jwt_secret: str):
        self.jwt_secret = jwt_secret
        self.active_sessions: Dict[str, SessionToken] = {}
        self.session_history: Dict[str, List[Dict[str, Any]]] = {}
        self.revoked_tokens: Set[str] = set()

    async def create_session(
        self,
        node_identity: NodeIdentity,
        authentication_score: float,
        rbac: RoleBasedAccessControl,
        session_duration: int = 3600,
    ) -> Dict[str, Any]:
        """
        Create authenticated session for a node

        Args:
            node_identity: Authenticated node identity
            authentication_score: MFA authentication confidence score
            rbac: Role-based access control system
            session_duration: Session duration in seconds

        Returns:
            Session creation result
        """
        if authentication_score < 0.7:  # Minimum authentication threshold
            raise ValueError(f"Insufficient authentication score: {authentication_score}")

        token_id = secrets.token_hex(16)
        current_time = time.time()
        expires_at = current_time + session_duration

        # Generate refresh token
        refresh_token = secrets.token_hex(32)

        # Get node capabilities
        capabilities = rbac.get_node_capabilities(node_identity)

        # Create session token
        session_token = SessionToken(
            token_id=token_id,
            node_id=node_identity.node_id,
            roles=node_identity.roles,
            capabilities=capabilities,
            issued_at=current_time,
            expires_at=expires_at,
            refresh_token=refresh_token,
            metadata={
                "authentication_score": authentication_score,
                "reputation_score": node_identity.reputation_score,
                "creation_ip": node_identity.metadata.get("ip_address"),
                "user_agent": node_identity.metadata.get("user_agent"),
            },
        )

        # Create JWT token
        jwt_payload = {
            "token_id": token_id,
            "node_id": node_identity.node_id,
            "roles": [role.value for role in node_identity.roles],
            "capabilities": list(capabilities),
            "iat": current_time,
            "exp": expires_at,
            "auth_score": authentication_score,
            "reputation": node_identity.reputation_score,
        }

        jwt_token = jwt.encode(jwt_payload, self.jwt_secret, algorithm="HS256")

        # Store session
        self.active_sessions[token_id] = session_token

        # Record in session history
        if node_identity.node_id not in self.session_history:
            self.session_history[node_identity.node_id] = []

        self.session_history[node_identity.node_id].append(
            {
                "token_id": token_id,
                "created_at": current_time,
                "expires_at": expires_at,
                "authentication_score": authentication_score,
                "ip_address": node_identity.metadata.get("ip_address"),
                "status": "created",
            }
        )

        logger.info(f"Created session {token_id} for node {node_identity.node_id}")

        return {
            "token_id": token_id,
            "jwt_token": jwt_token,
            "refresh_token": refresh_token,
            "expires_at": expires_at,
            "capabilities": list(capabilities),
            "session_duration": session_duration,
        }

    async def validate_session(
        self, jwt_token: str, required_capabilities: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """Validate session token and check capabilities"""
        try:
            # Decode JWT token
            payload = jwt.decode(jwt_token, self.jwt_secret, algorithms=["HS256"])
            token_id = payload.get("token_id")

            if not token_id or token_id in self.revoked_tokens:
                return {"valid": False, "reason": "token_revoked", "node_id": payload.get("node_id")}

            # Check session exists and is active
            if token_id not in self.active_sessions:
                return {"valid": False, "reason": "session_not_found", "node_id": payload.get("node_id")}

            session = self.active_sessions[token_id]

            # Check expiration
            if time.time() > session.expires_at:
                session.status = SessionStatus.EXPIRED
                return {"valid": False, "reason": "session_expired", "node_id": session.node_id}

            # Check status
            if session.status != SessionStatus.ACTIVE:
                return {"valid": False, "reason": f"session_{session.status.value}", "node_id": session.node_id}

            # Check required capabilities
            if required_capabilities:
                missing_capabilities = required_capabilities - session.capabilities
                if missing_capabilities:
                    return {
                        "valid": False,
                        "reason": "insufficient_capabilities",
                        "missing_capabilities": list(missing_capabilities),
                        "node_id": session.node_id,
                    }

            # Update last activity
            session.metadata["last_activity"] = time.time()

            return {
                "valid": True,
                "node_id": session.node_id,
                "token_id": token_id,
                "roles": list(session.roles),
                "capabilities": list(session.capabilities),
                "authentication_score": session.metadata.get("authentication_score", 0.0),
                "reputation_score": session.metadata.get("reputation_score", 0.0),
                "expires_at": session.expires_at,
            }

        except jwt.ExpiredSignatureError:
            return {"valid": False, "reason": "jwt_expired"}
        except jwt.InvalidTokenError as e:
            return {"valid": False, "reason": "invalid_jwt", "error": str(e)}

    async def revoke_session(self, token_id: str, reason: str = "manual_revocation") -> bool:
        """Revoke a session token"""
        if token_id not in self.active_sessions:
            return False

        session = self.active_sessions[token_id]
        session.status = SessionStatus.REVOKED

        # Add to revoked tokens set
        self.revoked_tokens.add(token_id)

        # Update session history
        if session.node_id in self.session_history:
            for session_record in self.session_history[session.node_id]:
                if session_record["token_id"] == token_id:
                    session_record["revoked_at"] = time.time()
                    session_record["revocation_reason"] = reason
                    break

        logger.info(f"Revoked session {token_id} for node {session.node_id}: {reason}")
        return True

    async def refresh_session(self, refresh_token: str, new_duration: int = 3600) -> Optional[Dict[str, Any]]:
        """Refresh session using refresh token"""
        # Find session by refresh token
        target_session = None
        for session in self.active_sessions.values():
            if session.refresh_token == refresh_token:
                target_session = session
                break

        if not target_session or target_session.status != SessionStatus.ACTIVE:
            return None

        # Create new session with updated expiration
        current_time = time.time()
        target_session.expires_at = current_time + new_duration
        target_session.refresh_token = secrets.token_hex(32)  # New refresh token

        # Generate new JWT
        jwt_payload = {
            "token_id": target_session.token_id,
            "node_id": target_session.node_id,
            "roles": [role.value for role in target_session.roles],
            "capabilities": list(target_session.capabilities),
            "iat": current_time,
            "exp": target_session.expires_at,
            "auth_score": target_session.metadata.get("authentication_score", 0.0),
            "reputation": target_session.metadata.get("reputation_score", 0.0),
        }

        jwt_token = jwt.encode(jwt_payload, self.jwt_secret, algorithm="HS256")

        logger.info(f"Refreshed session {target_session.token_id} for node {target_session.node_id}")

        return {
            "jwt_token": jwt_token,
            "refresh_token": target_session.refresh_token,
            "expires_at": target_session.expires_at,
        }

    def get_session_statistics(self) -> Dict[str, Any]:
        """Get session management statistics"""
        active_count = len([s for s in self.active_sessions.values() if s.status == SessionStatus.ACTIVE])
        expired_count = len([s for s in self.active_sessions.values() if s.status == SessionStatus.EXPIRED])
        revoked_count = len([s for s in self.active_sessions.values() if s.status == SessionStatus.REVOKED])

        return {
            "total_sessions": len(self.active_sessions),
            "active_sessions": active_count,
            "expired_sessions": expired_count,
            "revoked_sessions": revoked_count,
            "unique_nodes": len(set(s.node_id for s in self.active_sessions.values())),
            "revoked_tokens_count": len(self.revoked_tokens),
        }


class FederatedAuthenticationSystem:
    """
    Main federated authentication system coordinating all components
    """

    def __init__(self, jwt_secret: Optional[str] = None):
        self.key_manager = CryptographicKeyManager()
        self.mfa_authenticator = MultiFactorAuthenticator()
        self.rbac = RoleBasedAccessControl()
        self.session_manager = FederatedSessionManager(jwt_secret or secrets.token_hex(32))
        self.node_identities: Dict[str, NodeIdentity] = {}
        self.pending_registrations: Dict[str, Dict[str, Any]] = {}

    async def register_node(
        self,
        node_id: str,
        roles: Set[NodeRole],
        metadata: Optional[Dict[str, Any]] = None,
        initial_capabilities: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Register a new node in the system"""
        if node_id in self.node_identities:
            raise ValueError(f"Node {node_id} already registered")

        # Generate keypair for node
        private_key, public_key = self.key_manager.generate_node_keypair(node_id)

        # Create node identity
        node_identity = NodeIdentity(
            node_id=node_id,
            public_key=public_key,
            roles=roles,
            capabilities=initial_capabilities or set(),
            metadata=metadata or {},
        )

        self.node_identities[node_id] = node_identity

        logger.info(f"Registered node {node_id} with roles: {[r.value for r in roles]}")

        return {
            "node_id": node_id,
            "public_key": public_key.hex(),
            "roles": [role.value for role in roles],
            "capabilities": list(node_identity.capabilities),
            "registration_successful": True,
        }

    async def authenticate_node(
        self, node_id: str, authentication_methods: List[AuthenticationMethod], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Start authentication process for a node"""
        if node_id not in self.node_identities:
            raise ValueError(f"Node {node_id} not registered")

        # Create authentication challenges
        challenges = await self.mfa_authenticator.create_authentication_challenge(
            node_id, authentication_methods, context
        )

        return {
            "node_id": node_id,
            "challenges": [
                {
                    "challenge_id": c.challenge_id,
                    "method": c.method.value,
                    "challenge_data": c.challenge_data.hex(),
                    "expires_at": c.expires_at,
                    "difficulty": c.difficulty,
                }
                for c in challenges
            ],
            "required_methods": [m.value for m in authentication_methods],
        }

    async def complete_authentication(
        self, responses: List[AuthenticationResponse], session_duration: int = 3600
    ) -> Dict[str, Any]:
        """Complete authentication and create session"""
        if not responses:
            raise ValueError("No authentication responses provided")

        node_id = responses[0].node_id

        if node_id not in self.node_identities:
            raise ValueError(f"Node {node_id} not registered")

        # Verify all responses
        verification_results = []
        for response in responses:
            result = await self.mfa_authenticator.verify_authentication_response(response)
            verification_results.append(result)

        # Check if authentication succeeded
        successful_verifications = [r for r in verification_results if r["success"]]

        if not successful_verifications:
            return {
                "authentication_successful": False,
                "reason": "all_verifications_failed",
                "details": verification_results,
            }

        # Calculate overall authentication score
        auth_score = self.mfa_authenticator.get_authentication_score(node_id)

        # Create session
        node_identity = self.node_identities[node_id]
        session_result = await self.session_manager.create_session(
            node_identity, auth_score, self.rbac, session_duration
        )

        # Update node last activity
        node_identity.last_active = time.time()

        return {
            "authentication_successful": True,
            "node_id": node_id,
            "authentication_score": auth_score,
            "session": session_result,
            "verification_details": verification_results,
        }

    async def authorize_access(self, jwt_token: str, access_request: AccessRequest) -> Dict[str, Any]:
        """Authorize access request using session and RBAC"""
        # Validate session
        session_validation = await self.session_manager.validate_session(
            jwt_token, access_request.required_capabilities
        )

        if not session_validation["valid"]:
            return {"authorized": False, "reason": "invalid_session", "session_validation": session_validation}

        node_id = session_validation["node_id"]

        if node_id not in self.node_identities:
            return {"authorized": False, "reason": "node_not_found"}

        # Check RBAC permissions
        node_identity = self.node_identities[node_id]
        access_check = self.rbac.check_access_permission(node_identity, access_request)

        if not access_check["granted"]:
            return {"authorized": False, "reason": "access_denied", "rbac_result": access_check}

        logger.info(f"Authorized access for {node_id} to {access_request.resource_path}")

        return {
            "authorized": True,
            "node_id": node_id,
            "access_granted": access_check,
            "session_info": {
                "token_id": session_validation["token_id"],
                "authentication_score": session_validation["authentication_score"],
                "reputation_score": session_validation["reputation_score"],
            },
        }

    async def revoke_node_access(self, node_id: str, reason: str = "manual_revocation") -> Dict[str, Any]:
        """Revoke all access for a node"""
        if node_id not in self.node_identities:
            raise ValueError(f"Node {node_id} not found")

        # Revoke all active sessions for the node
        revoked_sessions = []
        for token_id, session in self.session_manager.active_sessions.items():
            if session.node_id == node_id and session.status == SessionStatus.ACTIVE:
                await self.session_manager.revoke_session(token_id, reason)
                revoked_sessions.append(token_id)

        # Update node reputation (optional punishment)
        node_identity = self.node_identities[node_id]
        if reason == "security_violation":
            node_identity.reputation_score = max(0.0, node_identity.reputation_score - 0.5)

        logger.warning(f"Revoked access for node {node_id}: {reason}")

        return {
            "node_id": node_id,
            "revocation_successful": True,
            "revoked_sessions": revoked_sessions,
            "reason": reason,
            "new_reputation": node_identity.reputation_score,
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "registered_nodes": len(self.node_identities),
            "session_statistics": self.session_manager.get_session_statistics(),
            "authentication_statistics": {
                "pending_challenges": len(self.mfa_authenticator.pending_challenges),
                "completed_authentications": len(self.mfa_authenticator.completed_authentications),
            },
            "security_status": {
                "revoked_certificates": len(self.key_manager.revoked_certificates),
                "key_pairs_managed": len(self.key_manager.node_keypairs),
            },
            "rbac_statistics": {
                "defined_roles": len(self.rbac.role_permissions),
                "capability_definitions": len(self.rbac.capability_matrix),
                "dynamic_permissions": len(self.rbac.dynamic_permissions),
            },
        }


# Factory function for system creation
def create_federated_auth_system(
    jwt_secret: Optional[str] = None, admin_node_id: Optional[str] = None
) -> FederatedAuthenticationSystem:
    """
    Factory function to create and initialize federated authentication system

    Args:
        jwt_secret: JWT signing secret (generated if not provided)
        admin_node_id: Optional admin node to register automatically

    Returns:
        Configured authentication system
    """
    auth_system = FederatedAuthenticationSystem(jwt_secret)

    # Register admin node if provided
    if admin_node_id:
        asyncio.create_task(
            auth_system.register_node(admin_node_id, {NodeRole.ADMIN}, {"created_by": "system_initialization"})
        )

    logger.info("Created federated authentication system")
    return auth_system
