"""Security Interfaces and Contracts.

Defines the core interfaces for the modular security architecture,
ensuring clean separation of concerns and dependency injection.
"""

import abc
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol


class SecurityLevel(Enum):
    """Security levels for different operations."""

    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"


class AuthenticationMethod(Enum):
    """Available authentication methods."""

    PASSWORD = "password"  # nosec B105 - field name constant, not password
    TOKEN = "token"  # nosec B105 - field name constant, not password
    CERTIFICATE = "certificate"
    MFA = "mfa"
    BIOMETRIC = "biometric"


class AuthorizationLevel(Enum):
    """Authorization levels."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


@dataclass
class SecurityContext:
    """Security context for operations."""

    user_id: str
    session_id: str
    tenant_id: str | None = None
    roles: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    authentication_method: AuthenticationMethod | None = None
    security_level: SecurityLevel = SecurityLevel.STANDARD
    device_info: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    is_authenticated: bool = False
    mfa_verified: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityResult:
    """Result of security operations."""

    success: bool
    message: str
    error_code: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    security_level: SecurityLevel = SecurityLevel.STANDARD
    requires_mfa: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ISecurityConfig(Protocol):
    """Interface for security configuration."""

    @abc.abstractmethod
    def get_security_level(self) -> SecurityLevel:
        """Get current security level."""
        ...

    @abc.abstractmethod
    def get_encryption_config(self) -> dict[str, Any]:
        """Get encryption configuration."""
        ...

    @abc.abstractmethod
    def get_authentication_config(self) -> dict[str, Any]:
        """Get authentication configuration."""
        ...

    @abc.abstractmethod
    def get_authorization_config(self) -> dict[str, Any]:
        """Get authorization configuration."""
        ...

    @abc.abstractmethod
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if security feature is enabled."""
        ...


class IAuthenticationProvider(Protocol):
    """Interface for authentication providers."""

    @abc.abstractmethod
    async def authenticate(self, credentials: dict[str, Any], context: dict[str, Any]) -> SecurityResult:
        """Authenticate user with provided credentials."""
        ...

    @abc.abstractmethod
    async def verify_token(self, token: str) -> SecurityResult:
        """Verify authentication token."""
        ...

    @abc.abstractmethod
    async def refresh_token(self, refresh_token: str) -> SecurityResult:
        """Refresh authentication token."""
        ...

    @abc.abstractmethod
    async def revoke_token(self, token: str) -> SecurityResult:
        """Revoke authentication token."""
        ...

    @abc.abstractmethod
    async def create_session(self, user_id: str, device_info: dict[str, Any]) -> SecurityResult:
        """Create new session."""
        ...

    @abc.abstractmethod
    async def validate_session(self, session_id: str) -> SecurityResult:
        """Validate existing session."""
        ...


class IAuthorizationProvider(Protocol):
    """Interface for authorization providers."""

    @abc.abstractmethod
    async def check_permission(self, context: SecurityContext, resource: str, action: str) -> SecurityResult:
        """Check if user has permission for action on resource."""
        ...

    @abc.abstractmethod
    async def get_user_roles(self, user_id: str) -> list[str]:
        """Get user roles."""
        ...

    @abc.abstractmethod
    async def get_role_permissions(self, role: str) -> list[str]:
        """Get permissions for role."""
        ...

    @abc.abstractmethod
    async def assign_role(self, user_id: str, role: str) -> SecurityResult:
        """Assign role to user."""
        ...

    @abc.abstractmethod
    async def revoke_role(self, user_id: str, role: str) -> SecurityResult:
        """Revoke role from user."""
        ...


class ICryptographicService(Protocol):
    """Interface for cryptographic services."""

    @abc.abstractmethod
    async def encrypt(
        self, data: bytes, key_id: str | None = None, context: dict[str, Any] | None = None
    ) -> bytes:
        """Encrypt data."""
        ...

    @abc.abstractmethod
    async def decrypt(
        self, encrypted_data: bytes, key_id: str | None = None, context: dict[str, Any] | None = None
    ) -> bytes:
        """Decrypt data."""
        ...

    @abc.abstractmethod
    async def sign(self, data: bytes, key_id: str | None = None) -> bytes:
        """Sign data."""
        ...

    @abc.abstractmethod
    async def verify_signature(self, data: bytes, signature: bytes, public_key: bytes | None = None) -> bool:
        """Verify signature."""
        ...

    @abc.abstractmethod
    async def generate_key(self, key_type: str) -> str:
        """Generate cryptographic key."""
        ...

    @abc.abstractmethod
    async def rotate_keys(self) -> SecurityResult:
        """Rotate encryption keys."""
        ...


class ISecurityMiddleware(Protocol):
    """Interface for security middleware."""

    @abc.abstractmethod
    async def process_request(self, request_data: dict[str, Any], context: SecurityContext) -> SecurityResult:
        """Process incoming request through security filters."""
        ...

    @abc.abstractmethod
    async def process_response(self, response_data: dict[str, Any], context: SecurityContext) -> SecurityResult:
        """Process outgoing response through security filters."""
        ...

    @abc.abstractmethod
    async def handle_security_event(self, event_type: str, event_data: dict[str, Any]) -> SecurityResult:
        """Handle security events."""
        ...


class ISecurityMonitor(Protocol):
    """Interface for security monitoring."""

    @abc.abstractmethod
    async def record_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        context: SecurityContext | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record security event."""
        ...

    @abc.abstractmethod
    async def get_security_metrics(self) -> dict[str, Any]:
        """Get security metrics."""
        ...

    @abc.abstractmethod
    async def check_threat_level(self) -> str:
        """Get current threat level."""
        ...


class IThreatDetector(Protocol):
    """Interface for threat detection."""

    @abc.abstractmethod
    async def analyze_request(self, request_data: dict[str, Any], context: SecurityContext) -> list[str]:
        """Analyze request for threats."""
        ...

    @abc.abstractmethod
    async def is_peer_trusted(self, peer_id: str) -> bool:
        """Check if peer is trusted."""
        ...

    @abc.abstractmethod
    async def update_peer_reputation(self, peer_id: str, behavior: str, score_delta: float) -> None:
        """Update peer reputation."""
        ...


class ISecurityAuditor(Protocol):
    """Interface for security auditing."""

    @abc.abstractmethod
    async def log_access(
        self, user_id: str, resource: str, action: str, result: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Log access attempt."""
        ...

    @abc.abstractmethod
    async def get_audit_logs(self, filters: dict[str, Any], limit: int = 100) -> list[dict[str, Any]]:
        """Get audit logs."""
        ...

    @abc.abstractmethod
    async def generate_security_report(self, start_date: datetime, end_date: datetime) -> dict[str, Any]:
        """Generate security report."""
        ...
