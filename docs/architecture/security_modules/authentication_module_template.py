#!/usr/bin/env python3
"""
Authentication Module Template - Modular Security Architecture
============================================================

This module provides comprehensive authentication services with clean interfaces
and dependency injection support following SOLID principles.

Key Features:
- JWT token management with session tracking
- Multi-factor authentication (MFA) support
- Session lifecycle management
- User identity and role resolution
- Redis-based session storage
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol
import asyncio
import logging
import secrets

logger = logging.getLogger(__name__)


# ===== DOMAIN MODELS =====


class AuthenticationMethod(Enum):
    """Supported authentication methods."""

    JWT_TOKEN = "jwt_token"  # nosec B105 - field name
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BASIC = "basic"


class MFAMethod(Enum):
    """Multi-factor authentication methods."""

    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"
    BACKUP_CODE = "backup_code"


class SessionStatus(Enum):
    """Session status enumeration."""

    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


@dataclass
class DeviceInfo:
    """Device information for session tracking."""

    user_agent: str
    ip_address: str
    device_fingerprint: str = field(default_factory=lambda: secrets.token_urlsafe(16))
    platform: Optional[str] = None
    browser: Optional[str] = None
    location: Optional[str] = None


@dataclass
class AuthCredentials:
    """Authentication credentials container."""

    username: str
    password: Optional[str] = None
    token: Optional[str] = None
    api_key: Optional[str] = None
    mfa_token: Optional[str] = None
    mfa_method: Optional[MFAMethod] = None
    device_info: Optional[DeviceInfo] = None


@dataclass
class AuthResult:
    """Authentication result container."""

    success: bool
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None
    mfa_required: bool = False
    available_mfa_methods: List[MFAMethod] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """User session data model."""

    session_id: str
    user_id: str
    device_info: DeviceInfo
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    status: SessionStatus = SessionStatus.ACTIVE
    access_tokens: List[str] = field(default_factory=list)
    refresh_tokens: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Check if session is active and not expired."""
        return self.status == SessionStatus.ACTIVE and datetime.utcnow() < self.expires_at

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()


@dataclass
class TokenValidationResult:
    """Token validation result."""

    valid: bool
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class MFASetupResult:
    """MFA setup result container."""

    success: bool
    method: Optional[MFAMethod] = None
    secret: Optional[str] = None
    qr_code: Optional[str] = None
    backup_codes: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class MFAStatus:
    """User MFA status information."""

    user_id: str
    totp_enabled: bool = False
    sms_enabled: bool = False
    email_enabled: bool = False
    backup_codes_available: int = 0
    methods_available: List[MFAMethod] = field(default_factory=list)
    last_mfa_verification: Optional[datetime] = None


# ===== INTERFACES =====


class ISessionManager(Protocol):
    """Interface for session management operations."""

    @abstractmethod
    async def create_session(self, user_id: str, device_info: DeviceInfo) -> str:
        """Create a new user session."""
        ...

    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve session by ID."""
        ...

    @abstractmethod
    async def update_session(self, session: Session) -> bool:
        """Update session information."""
        ...

    @abstractmethod
    async def revoke_session(self, session_id: str) -> bool:
        """Revoke a specific session."""
        ...

    @abstractmethod
    async def revoke_all_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user."""
        ...

    @abstractmethod
    async def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all active sessions for a user."""
        ...

    @abstractmethod
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        ...

    @abstractmethod
    async def is_session_active(self, session_id: str) -> bool:
        """Check if session is active."""
        ...


class IMFAService(Protocol):
    """Interface for multi-factor authentication operations."""

    @abstractmethod
    async def setup_mfa(self, user_id: str, method: MFAMethod, **kwargs) -> MFASetupResult:
        """Set up MFA for a user."""
        ...

    @abstractmethod
    async def verify_mfa(self, user_id: str, token: str, method: MFAMethod, **kwargs) -> bool:
        """Verify MFA token."""
        ...

    @abstractmethod
    async def disable_mfa(self, user_id: str, method: MFAMethod) -> bool:
        """Disable MFA method for user."""
        ...

    @abstractmethod
    async def get_user_mfa_status(self, user_id: str) -> MFAStatus:
        """Get user's MFA configuration status."""
        ...

    @abstractmethod
    async def generate_backup_codes(self, user_id: str) -> List[str]:
        """Generate backup codes for user."""
        ...

    @abstractmethod
    async def verify_backup_code(self, user_id: str, code: str) -> bool:
        """Verify backup code."""
        ...


class ITokenManager(Protocol):
    """Interface for token management operations."""

    @abstractmethod
    async def create_access_token(self, user_id: str, session_id: str, **claims) -> str:
        """Create JWT access token."""
        ...

    @abstractmethod
    async def create_refresh_token(self, user_id: str, session_id: str) -> str:
        """Create JWT refresh token."""
        ...

    @abstractmethod
    async def validate_token(self, token: str, token_type: str = "access") -> TokenValidationResult:
        """Validate JWT token."""
        ...

    @abstractmethod
    async def revoke_token(self, jti: str) -> bool:
        """Revoke specific token."""
        ...

    @abstractmethod
    async def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Create new access token from refresh token."""
        ...

    @abstractmethod
    async def is_token_revoked(self, jti: str) -> bool:
        """Check if token is revoked."""
        ...


class IAuthenticationService(Protocol):
    """Main authentication service interface."""

    @abstractmethod
    async def authenticate_user(self, credentials: AuthCredentials) -> AuthResult:
        """Authenticate user with various methods."""
        ...

    @abstractmethod
    async def validate_session(self, session_id: str) -> Optional[Session]:
        """Validate user session."""
        ...

    @abstractmethod
    async def logout_user(self, session_id: str) -> bool:
        """Logout user session."""
        ...

    @abstractmethod
    async def logout_all_sessions(self, user_id: str) -> int:
        """Logout all user sessions."""
        ...

    @abstractmethod
    async def refresh_authentication(self, refresh_token: str) -> AuthResult:
        """Refresh user authentication."""
        ...


# ===== IMPLEMENTATIONS =====


class RedisSessionManager:
    """Redis-based session manager implementation."""

    def __init__(self, redis_client, session_timeout: int = 3600):
        self.redis = redis_client
        self.session_timeout = session_timeout
        self.session_prefix = "session:"
        self.user_sessions_prefix = "user_sessions:"

    async def create_session(self, user_id: str, device_info: DeviceInfo) -> str:
        """Create a new user session in Redis."""
        session_id = f"sess_{secrets.token_urlsafe(32)}"
        now = datetime.utcnow()

        session = Session(
            session_id=session_id,
            user_id=user_id,
            device_info=device_info,
            created_at=now,
            last_activity=now,
            expires_at=now + timedelta(seconds=self.session_timeout),
        )

        # Store session data
        session_key = f"{self.session_prefix}{session_id}"
        await self.redis.hset(
            session_key,
            mapping={
                "user_id": session.user_id,
                "device_fingerprint": session.device_info.device_fingerprint,
                "ip_address": session.device_info.ip_address,
                "user_agent": session.device_info.user_agent,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "expires_at": session.expires_at.isoformat(),
                "status": session.status.value,
            },
        )

        # Set expiration
        await self.redis.expire(session_key, self.session_timeout)

        # Add to user sessions set
        user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
        await self.redis.sadd(user_sessions_key, session_id)
        await self.redis.expire(user_sessions_key, self.session_timeout * 2)

        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve session from Redis."""
        session_key = f"{self.session_prefix}{session_id}"
        session_data = await self.redis.hgetall(session_key)

        if not session_data:
            return None

        device_info = DeviceInfo(
            user_agent=session_data.get("user_agent", "unknown"),
            ip_address=session_data.get("ip_address", "unknown"),
            device_fingerprint=session_data.get("device_fingerprint", "unknown"),
        )

        return Session(
            session_id=session_id,
            user_id=session_data["user_id"],
            device_info=device_info,
            created_at=datetime.fromisoformat(session_data["created_at"]),
            last_activity=datetime.fromisoformat(session_data["last_activity"]),
            expires_at=datetime.fromisoformat(session_data["expires_at"]),
            status=SessionStatus(session_data.get("status", "active")),
        )

    async def update_session(self, session: Session) -> bool:
        """Update session in Redis."""
        session_key = f"{self.session_prefix}{session.session_id}"

        updated_fields = {"last_activity": session.last_activity.isoformat(), "status": session.status.value}

        result = await self.redis.hset(session_key, mapping=updated_fields)
        return result > 0

    async def revoke_session(self, session_id: str) -> bool:
        """Revoke session in Redis."""
        session_key = f"{self.session_prefix}{session_id}"

        # Mark as revoked instead of deleting for audit trail
        result = await self.redis.hset(session_key, "status", SessionStatus.REVOKED.value)

        if result:
            logger.info(f"Revoked session {session_id}")

        return bool(result)

    async def revoke_all_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user."""
        user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
        session_ids = await self.redis.smembers(user_sessions_key)

        revoked_count = 0
        for session_id in session_ids:
            if await self.revoke_session(session_id):
                revoked_count += 1

        logger.info(f"Revoked {revoked_count} sessions for user {user_id}")
        return revoked_count

    async def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all active sessions for a user."""
        user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
        session_ids = await self.redis.smembers(user_sessions_key)

        sessions = []
        for session_id in session_ids:
            session = await self.get_session(session_id)
            if session and session.is_active:
                sessions.append(session)

        return sessions

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        # This would typically be run as a background task
        # Implementation would scan for expired sessions and remove them
        cleaned_count = 0
        logger.info(f"Cleaned up {cleaned_count} expired sessions")
        return cleaned_count

    async def is_session_active(self, session_id: str) -> bool:
        """Check if session is active."""
        session = await self.get_session(session_id)
        return session is not None and session.is_active


class JWTTokenManager:
    """JWT token manager implementation."""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_lifetime = timedelta(hours=1)
        self.refresh_token_lifetime = timedelta(days=30)
        self.revoked_tokens = set()  # In production, use Redis

    async def create_access_token(self, user_id: str, session_id: str, **claims) -> str:
        """Create JWT access token."""
        import jwt

        now = datetime.utcnow()
        jti = secrets.token_urlsafe(16)

        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "iat": now,
            "exp": now + self.access_token_lifetime,
            "type": "access_token",
            "jti": jti,
            **claims,
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.debug(f"Created access token for user {user_id}")
        return token

    async def create_refresh_token(self, user_id: str, session_id: str) -> str:
        """Create JWT refresh token."""
        import jwt

        now = datetime.utcnow()
        jti = secrets.token_urlsafe(16)

        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "iat": now,
            "exp": now + self.refresh_token_lifetime,
            "type": "refresh_token",
            "jti": jti,
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.debug(f"Created refresh token for user {user_id}")
        return token

    async def validate_token(self, token: str, token_type: str = "access") -> TokenValidationResult:
        """Validate JWT token."""
        import jwt

        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check token type
            if payload.get("type") != f"{token_type}_token":
                return TokenValidationResult(
                    valid=False, error_message=f"Invalid token type: expected {token_type}_token"
                )

            # Check if token is revoked
            jti = payload.get("jti")
            if jti in self.revoked_tokens:
                return TokenValidationResult(valid=False, error_message="Token has been revoked")

            return TokenValidationResult(
                valid=True,
                user_id=payload.get("user_id"),
                session_id=payload.get("session_id"),
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
                expires_at=datetime.fromtimestamp(payload["exp"]),
            )

        except jwt.ExpiredSignatureError:
            return TokenValidationResult(valid=False, error_message="Token has expired")
        except jwt.InvalidTokenError as e:
            return TokenValidationResult(valid=False, error_message=f"Invalid token: {str(e)}")

    async def revoke_token(self, jti: str) -> bool:
        """Revoke specific token."""
        self.revoked_tokens.add(jti)
        logger.info(f"Revoked token {jti}")
        return True

    async def is_token_revoked(self, jti: str) -> bool:
        """Check if token is revoked."""
        return jti in self.revoked_tokens


class AuthenticationService:
    """Main authentication service implementation."""

    def __init__(
        self,
        session_manager: ISessionManager,
        token_manager: ITokenManager,
        mfa_service: IMFAService,
        user_repository: Any,  # User data access layer
    ):
        self.session_manager = session_manager
        self.token_manager = token_manager
        self.mfa_service = mfa_service
        self.user_repository = user_repository

    async def authenticate_user(self, credentials: AuthCredentials) -> AuthResult:
        """Authenticate user with various methods."""
        try:
            # Step 1: Basic authentication
            user = await self._authenticate_basic(credentials)
            if not user:
                return AuthResult(success=False, error_message="Invalid credentials")

            # Step 2: Check MFA requirement
            mfa_status = await self.mfa_service.get_user_mfa_status(user["user_id"])
            mfa_required = any([mfa_status.totp_enabled, mfa_status.sms_enabled, mfa_status.email_enabled])

            # Step 3: Verify MFA if required
            mfa_verified = False
            if mfa_required:
                if not credentials.mfa_token or not credentials.mfa_method:
                    return AuthResult(
                        success=False,
                        mfa_required=True,
                        available_mfa_methods=mfa_status.methods_available,
                        error_message="MFA verification required",
                    )

                mfa_verified = await self.mfa_service.verify_mfa(
                    user["user_id"], credentials.mfa_token, credentials.mfa_method
                )

                if not mfa_verified:
                    return AuthResult(success=False, error_message="Invalid MFA token")

            # Step 4: Create session and tokens
            session_id = await self.session_manager.create_session(
                user["user_id"], credentials.device_info or DeviceInfo("unknown", "unknown")
            )

            # Create tokens with user claims
            access_token = await self.token_manager.create_access_token(
                user["user_id"],
                session_id,
                roles=user.get("roles", []),
                permissions=user.get("permissions", []),
                mfa_verified=mfa_verified,
            )

            refresh_token = await self.token_manager.create_refresh_token(user["user_id"], session_id)

            return AuthResult(
                success=True,
                user_id=user["user_id"],
                session_id=session_id,
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=3600,  # 1 hour
                mfa_required=False,
            )

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return AuthResult(success=False, error_message="Authentication system error")

    async def _authenticate_basic(self, credentials: AuthCredentials) -> Optional[Dict[str, Any]]:
        """Perform basic username/password authentication."""
        # This would integrate with your user repository
        # For now, return a mock user
        if credentials.username and credentials.password:
            # In real implementation, verify password hash
            return {
                "user_id": f"user_{credentials.username}",
                "username": credentials.username,
                "roles": ["user"],
                "permissions": ["read", "write"],
            }
        return None

    async def validate_session(self, session_id: str) -> Optional[Session]:
        """Validate user session."""
        return await self.session_manager.get_session(session_id)

    async def logout_user(self, session_id: str) -> bool:
        """Logout user session."""
        return await self.session_manager.revoke_session(session_id)

    async def logout_all_sessions(self, user_id: str) -> int:
        """Logout all user sessions."""
        return await self.session_manager.revoke_all_user_sessions(user_id)

    async def refresh_authentication(self, refresh_token: str) -> AuthResult:
        """Refresh user authentication."""
        try:
            # Validate refresh token
            validation_result = await self.token_manager.validate_token(refresh_token, "refresh")

            if not validation_result.valid:
                return AuthResult(success=False, error_message=validation_result.error_message)

            # Check session is still active
            session = await self.session_manager.get_session(validation_result.session_id)

            if not session or not session.is_active:
                return AuthResult(success=False, error_message="Session is no longer active")

            # Create new access token
            access_token = await self.token_manager.create_access_token(
                validation_result.user_id,
                validation_result.session_id,
                roles=validation_result.roles,
                permissions=validation_result.permissions,
            )

            # Update session activity
            session.update_activity()
            await self.session_manager.update_session(session)

            return AuthResult(
                success=True,
                user_id=validation_result.user_id,
                session_id=validation_result.session_id,
                access_token=access_token,
                expires_in=3600,
            )

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return AuthResult(success=False, error_message="Token refresh failed")


# ===== DEPENDENCY INJECTION SETUP =====


class AuthenticationContainer:
    """Dependency injection container for authentication module."""

    def __init__(self):
        self._services = {}
        self._singletons = {}

    def register_singleton(self, interface: type, implementation: type):
        """Register singleton service."""
        self._services[interface] = (implementation, "singleton")

    def register_transient(self, interface: type, implementation: type):
        """Register transient service."""
        self._services[interface] = (implementation, "transient")

    def get_service(self, interface: type):
        """Get service instance."""
        if interface not in self._services:
            raise ValueError(f"Service {interface.__name__} not registered")

        implementation, lifetime = self._services[interface]

        if lifetime == "singleton":
            if interface not in self._singletons:
                self._singletons[interface] = implementation()
            return self._singletons[interface]
        else:
            return implementation()

    def configure_services(self, redis_client, secret_key: str, user_repository):
        """Configure authentication services."""
        # Register implementations
        session_manager = RedisSessionManager(redis_client)
        token_manager = JWTTokenManager(secret_key)
        # mfa_service would be implemented separately

        self._singletons[ISessionManager] = session_manager
        self._singletons[ITokenManager] = token_manager
        self._singletons[IAuthenticationService] = AuthenticationService(
            session_manager, token_manager, None, user_repository
        )


# ===== MODULE HEALTH CHECK =====


class AuthenticationHealthCheck:
    """Health check for authentication module."""

    def __init__(self, auth_service: IAuthenticationService):
        self.auth_service = auth_service

    async def check_health(self) -> Dict[str, Any]:
        """Perform health check on authentication components."""
        try:
            # Test basic service availability
            # Could add more comprehensive checks

            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "components": {
                    "authentication_service": "operational",
                    "session_manager": "operational",
                    "token_manager": "operational",
                },
            }
        except Exception as e:
            logger.error(f"Authentication health check failed: {e}")
            return {"status": "unhealthy", "error": str(e), "timestamp": datetime.utcnow().isoformat()}


# ===== USAGE EXAMPLE =====


async def example_usage():
    """Example usage of authentication module."""

    # Mock dependencies (would be real in production)
    from unittest.mock import MagicMock

    redis_client = MagicMock()
    user_repository = MagicMock()

    # Configure container
    container = AuthenticationContainer()
    container.configure_services(redis_client, "secret-key-here", user_repository)

    # Get authentication service
    auth_service = container.get_service(IAuthenticationService)

    # Create credentials
    device_info = DeviceInfo(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)", ip_address="192.168.1.100")

    credentials = AuthCredentials(
        username="testuser",
        password="testpassword",  # nosec B106 - test template password, not production
        device_info=device_info,
    )

    # Authenticate user
    result = await auth_service.authenticate_user(credentials)

    if result.success:
        print(f"Authentication successful for user: {result.user_id}")
        print(f"Session ID: {result.session_id}")
        print(f"Access Token: {result.access_token[:20]}...")
    else:
        print(f"Authentication failed: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(example_usage())
