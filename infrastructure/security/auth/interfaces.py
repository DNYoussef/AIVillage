"""Authentication module interfaces.

This module defines the interfaces for authentication, session management,
and MFA services following the Interface Segregation Principle.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class MFAMethodType(Enum):
    """Multi-factor authentication method types."""

    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"
    BACKUP_CODE = "backup_code"


@dataclass
class DeviceInfo:
    """Device information for session tracking."""

    user_agent: str
    ip_address: str
    device_fingerprint: str | None = None
    location: str | None = None


@dataclass
class AuthCredentials:
    """Authentication credentials."""

    username: str
    password: str
    mfa_token: str | None = None
    mfa_method: MFAMethodType | None = None


@dataclass
class AuthResult:
    """Authentication result."""

    success: bool
    user_id: str | None = None
    roles: list[str] = None
    permissions: list[str] = None
    mfa_verified: bool = False
    error_message: str | None = None


@dataclass
class TokenValidationResult:
    """Token validation result."""

    valid: bool
    payload: dict[str, Any] | None = None
    error_message: str | None = None


@dataclass
class SessionData:
    """Session data structure."""

    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    device_info: DeviceInfo
    roles: list[str]
    permissions: list[str]
    tenant_id: str | None = None
    is_active: bool = True
    metadata: dict[str, Any] = None


@dataclass
class MFASetupResult:
    """MFA setup result."""

    success: bool
    qr_code: str | None = None
    backup_codes: list[str] = None
    secret: str | None = None
    error_message: str | None = None


@dataclass
class MFAStatus:
    """MFA status for user."""

    totp_enabled: bool = False
    sms_enabled: bool = False
    email_enabled: bool = False
    backup_codes_available: int = 0
    methods_available: list[str] = None


class IAuthenticationService(ABC):
    """Authentication service interface."""

    @abstractmethod
    async def authenticate_user(self, credentials: AuthCredentials) -> AuthResult:
        """Authenticate user with credentials."""
        pass

    @abstractmethod
    async def validate_token(self, token: str, token_type: str = "access_token") -> TokenValidationResult:  # nosec B107
        """Validate JWT token."""
        pass

    @abstractmethod
    async def create_session_tokens(
        self,
        user_id: str,
        device_info: DeviceInfo,
        roles: list[str] = None,
        permissions: list[str] = None,
        mfa_verified: bool = False,
    ) -> dict[str, Any]:
        """Create JWT tokens with session tracking."""
        pass

    @abstractmethod
    async def revoke_token(self, jti: str) -> bool:
        """Revoke specific token."""
        pass

    @abstractmethod
    async def revoke_session(self, session_id: str) -> bool:
        """Revoke entire session."""
        pass


class ISessionManager(ABC):
    """Session manager interface."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize session manager."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close session manager."""
        pass

    @abstractmethod
    async def create_session(
        self,
        user_id: str,
        device_info: DeviceInfo,
        roles: list[str] = None,
        permissions: list[str] = None,
        tenant_id: str | None = None,
        metadata: dict[str, Any] = None,
    ) -> str:
        """Create new user session."""
        pass

    @abstractmethod
    async def get_session(self, session_id: str) -> SessionData | None:
        """Get session by ID."""
        pass

    @abstractmethod
    async def is_session_active(self, session_id: str) -> bool:
        """Check if session is active."""
        pass

    @abstractmethod
    async def update_session_activity(self, session_id: str) -> bool:
        """Update session last activity."""
        pass

    @abstractmethod
    async def revoke_session(self, session_id: str) -> bool:
        """Revoke session."""
        pass

    @abstractmethod
    async def revoke_all_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for user."""
        pass

    @abstractmethod
    async def get_user_sessions(self, user_id: str) -> list[SessionData]:
        """Get all active sessions for user."""
        pass

    @abstractmethod
    async def add_token_to_session(self, session_id: str, jti: str, token_type: str) -> bool:
        """Add token to session tracking."""
        pass

    @abstractmethod
    async def is_token_revoked(self, jti: str) -> bool:
        """Check if token is revoked."""
        pass

    @abstractmethod
    async def revoke_token(self, jti: str) -> bool:
        """Revoke specific token."""
        pass

    @abstractmethod
    async def detect_suspicious_activity(self, user_id: str, device_info: DeviceInfo) -> bool:
        """Detect suspicious activity."""
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Health check for session manager."""
        pass


class IMFAService(ABC):
    """Multi-factor authentication service interface."""

    @abstractmethod
    def setup_totp(self, user_id: str, email: str) -> MFASetupResult:
        """Setup TOTP for user."""
        pass

    @abstractmethod
    def verify_mfa(self, user_id: str, method: MFAMethodType, token: str, secret: str | None = None) -> bool:
        """Verify MFA token."""
        pass

    @abstractmethod
    def get_user_mfa_status(self, user_id: str) -> MFAStatus:
        """Get MFA status for user."""
        pass

    @abstractmethod
    def generate_backup_codes(self, user_id: str, count: int = 10) -> list[str]:
        """Generate backup codes for user."""
        pass

    @abstractmethod
    def verify_backup_code(self, user_id: str, code: str) -> bool:
        """Verify backup code."""
        pass

    @abstractmethod
    def disable_mfa(self, user_id: str, method: MFAMethodType) -> bool:
        """Disable MFA method for user."""
        pass
