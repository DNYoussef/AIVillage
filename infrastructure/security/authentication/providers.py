"""Authentication Providers.

Implements various authentication providers following clean architecture principles
with dependency injection and clear separation of concerns.
"""

import hashlib
import hmac
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..core.interfaces import (
    IAuthenticationProvider,
    SecurityContext,
    SecurityResult,
    AuthenticationMethod,
)
from ..core.exceptions import (
    TokenError,
)
from .session_manager import SessionManager
from .token_manager import TokenManager
from .mfa_service import MFAService

logger = logging.getLogger(__name__)


class BaseAuthenticationProvider:
    """Base authentication provider with common functionality."""

    def __init__(self, session_manager: SessionManager, token_manager: TokenManager, config: Dict[str, Any]):
        self.session_manager = session_manager
        self.token_manager = token_manager
        self.config = config

        self.max_failed_attempts = config.get("max_failed_attempts", 5)
        self.lockout_duration = config.get("lockout_duration_minutes", 30) * 60

        # Track failed attempts
        self.failed_attempts: Dict[str, List[float]] = {}

    def _is_locked_out(self, identifier: str) -> bool:
        """Check if identifier is locked out."""
        if identifier not in self.failed_attempts:
            return False

        current_time = time.time()
        attempts = self.failed_attempts[identifier]

        # Remove old attempts
        attempts[:] = [attempt for attempt in attempts if current_time - attempt < self.lockout_duration]

        return len(attempts) >= self.max_failed_attempts

    def _record_failed_attempt(self, identifier: str):
        """Record failed authentication attempt."""
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []

        self.failed_attempts[identifier].append(time.time())

    def _clear_failed_attempts(self, identifier: str):
        """Clear failed attempts for identifier."""
        if identifier in self.failed_attempts:
            del self.failed_attempts[identifier]

    async def _create_security_context(
        self, user_id: str, session_data: Dict[str, Any], device_info: Dict[str, Any]
    ) -> SecurityContext:
        """Create security context from authentication data."""
        return SecurityContext(
            user_id=user_id,
            session_id=session_data.get("session_id", ""),
            tenant_id=session_data.get("tenant_id"),
            roles=session_data.get("roles", []),
            permissions=session_data.get("permissions", []),
            authentication_method=AuthenticationMethod(session_data.get("auth_method", "token")),
            device_info=device_info,
            is_authenticated=True,
            mfa_verified=session_data.get("mfa_verified", False),
            metadata=session_data.get("metadata", {}),
        )


class TokenAuthenticationProvider(BaseAuthenticationProvider, IAuthenticationProvider):
    """Token-based authentication provider."""

    def __init__(
        self,
        session_manager: SessionManager,
        token_manager: TokenManager,
        user_store: Any,  # Abstract user store interface
        config: Dict[str, Any],
    ):
        super().__init__(session_manager, token_manager, config)
        self.user_store = user_store

        # Token configuration
        self.token_expiry_hours = config.get("token_expiry_hours", 24)
        self.refresh_expiry_days = config.get("refresh_token_expiry_days", 30)

    async def authenticate(self, credentials: Dict[str, Any], context: Dict[str, Any]) -> SecurityResult:
        """Authenticate user with username/password."""
        try:
            username = credentials.get("username")
            password = credentials.get("password")

            if not username or not password:
                return SecurityResult(
                    success=False, message="Username and password required", error_code="MISSING_CREDENTIALS"
                )

            # Check for lockout
            if self._is_locked_out(username):
                return SecurityResult(
                    success=False,
                    message="Account temporarily locked due to failed attempts",
                    error_code="ACCOUNT_LOCKED",
                )

            # Get user from store
            user = await self.user_store.get_user_by_username(username)
            if not user:
                self._record_failed_attempt(username)
                return SecurityResult(success=False, message="Invalid credentials", error_code="INVALID_CREDENTIALS")

            # Verify password
            if not await self._verify_password(password, user["password_hash"], user.get("salt")):
                self._record_failed_attempt(username)
                return SecurityResult(success=False, message="Invalid credentials", error_code="INVALID_CREDENTIALS")

            # Clear failed attempts on successful authentication
            self._clear_failed_attempts(username)

            # Check if MFA is required
            if user.get("mfa_enabled", False) and not credentials.get("mfa_verified"):
                return SecurityResult(
                    success=False,
                    message="Multi-factor authentication required",
                    error_code="MFA_REQUIRED",
                    requires_mfa=True,
                    data={"user_id": user["user_id"], "mfa_methods": user.get("mfa_methods", ["TOTP"])},
                )

            # Create session
            device_info = context.get("device_info", {})
            session_result = await self.create_session(user["user_id"], device_info)

            if not session_result.success:
                return session_result

            return SecurityResult(success=True, message="Authentication successful", data=session_result.data)

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return SecurityResult(success=False, message="Authentication failed", error_code="AUTH_ERROR")

    async def verify_token(self, token: str) -> SecurityResult:
        """Verify JWT token."""
        try:
            payload = await self.token_manager.verify_token(token)

            if not payload:
                return SecurityResult(success=False, message="Invalid token", error_code="INVALID_TOKEN")

            # Check session validity
            session_id = payload.get("session_id")
            if session_id:
                session = await self.session_manager.get_session(session_id)
                if not session or not session.get("is_active", False):
                    return SecurityResult(success=False, message="Session expired", error_code="SESSION_EXPIRED")

                # Update session activity
                await self.session_manager.update_activity(session_id)

            return SecurityResult(success=True, message="Token verified", data=payload)

        except TokenError as e:
            return SecurityResult(success=False, message=str(e), error_code=e.error_code)
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return SecurityResult(success=False, message="Token verification failed", error_code="TOKEN_ERROR")

    async def refresh_token(self, refresh_token: str) -> SecurityResult:
        """Refresh access token."""
        try:
            result = await self.token_manager.refresh_token(refresh_token)

            if not result["success"]:
                return SecurityResult(success=False, message="Token refresh failed", error_code="REFRESH_FAILED")

            return SecurityResult(success=True, message="Token refreshed", data=result["data"])

        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return SecurityResult(success=False, message="Token refresh failed", error_code="REFRESH_ERROR")

    async def revoke_token(self, token: str) -> SecurityResult:
        """Revoke token."""
        try:
            await self.token_manager.revoke_token(token)

            return SecurityResult(success=True, message="Token revoked")

        except Exception as e:
            logger.error(f"Token revocation error: {e}")
            return SecurityResult(success=False, message="Token revocation failed", error_code="REVOKE_ERROR")

    async def create_session(self, user_id: str, device_info: Dict[str, Any]) -> SecurityResult:
        """Create new user session."""
        try:
            # Get user data
            user = await self.user_store.get_user_by_id(user_id)
            if not user:
                return SecurityResult(success=False, message="User not found", error_code="USER_NOT_FOUND")

            # Create session
            session_data = await self.session_manager.create_session(
                user_id=user_id,
                device_info=device_info,
                roles=user.get("roles", []),
                permissions=user.get("permissions", []),
                tenant_id=user.get("tenant_id"),
                metadata=user.get("metadata", {}),
            )

            # Generate tokens
            tokens = await self.token_manager.create_tokens(
                user_id=user_id,
                session_id=session_data["session_id"],
                roles=user.get("roles", []),
                permissions=user.get("permissions", []),
                mfa_verified=device_info.get("mfa_verified", False),
            )

            return SecurityResult(
                success=True,
                message="Session created",
                data={
                    "session_id": session_data["session_id"],
                    "access_token": tokens["access_token"],
                    "refresh_token": tokens["refresh_token"],
                    "token_type": "Bearer",
                    "expires_in": self.token_expiry_hours * 3600,
                    "user_id": user_id,
                    "roles": user.get("roles", []),
                    "permissions": user.get("permissions", []),
                },
            )

        except Exception as e:
            logger.error(f"Session creation error: {e}")
            return SecurityResult(success=False, message="Session creation failed", error_code="SESSION_ERROR")

    async def validate_session(self, session_id: str) -> SecurityResult:
        """Validate existing session."""
        try:
            session = await self.session_manager.get_session(session_id)

            if not session:
                return SecurityResult(success=False, message="Session not found", error_code="SESSION_NOT_FOUND")

            if not session.get("is_active", False):
                return SecurityResult(success=False, message="Session expired", error_code="SESSION_EXPIRED")

            # Check session timeout
            last_activity = session.get("last_activity")
            if last_activity:
                timeout_minutes = self.config.get("session_timeout_minutes", 60)
                if datetime.utcnow() - last_activity > timedelta(minutes=timeout_minutes):
                    await self.session_manager.revoke_session(session_id)
                    return SecurityResult(success=False, message="Session timed out", error_code="SESSION_TIMEOUT")

            return SecurityResult(success=True, message="Session valid", data=session)

        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return SecurityResult(success=False, message="Session validation failed", error_code="SESSION_ERROR")

    async def _verify_password(self, password: str, password_hash: str, salt: Optional[str] = None) -> bool:
        """Verify password against hash."""
        try:
            if salt:
                # PBKDF2 verification
                computed_hash = hashlib.pbkdf2_hmac(
                    "sha256", password.encode(), salt.encode(), 100000  # iterations
                ).hex()
            else:
                # Simple SHA-256 (less secure, for backward compatibility)
                computed_hash = hashlib.sha256(password.encode()).hexdigest()

            return hmac.compare_digest(password_hash, computed_hash)

        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False


class MFAAuthenticationProvider(BaseAuthenticationProvider):
    """Multi-factor authentication provider."""

    def __init__(
        self,
        session_manager: SessionManager,
        token_manager: TokenManager,
        mfa_service: MFAService,
        config: Dict[str, Any],
    ):
        super().__init__(session_manager, token_manager, config)
        self.mfa_service = mfa_service

    async def verify_mfa(self, user_id: str, mfa_token: str, method: str) -> SecurityResult:
        """Verify MFA token."""
        try:
            is_valid = await self.mfa_service.verify_token(user_id, mfa_token, method)

            if not is_valid:
                self._record_failed_attempt(f"{user_id}:mfa")
                return SecurityResult(success=False, message="Invalid MFA token", error_code="INVALID_MFA_TOKEN")

            self._clear_failed_attempts(f"{user_id}:mfa")

            return SecurityResult(
                success=True, message="MFA verification successful", data={"method": method, "verified": True}
            )

        except Exception as e:
            logger.error(f"MFA verification error: {e}")
            return SecurityResult(success=False, message="MFA verification failed", error_code="MFA_ERROR")

    async def setup_mfa(self, user_id: str, method: str, contact_info: Optional[str] = None) -> SecurityResult:
        """Set up MFA for user."""
        try:
            setup_data = await self.mfa_service.setup_mfa(user_id, method, contact_info)

            return SecurityResult(success=True, message="MFA setup initiated", data=setup_data)

        except Exception as e:
            logger.error(f"MFA setup error: {e}")
            return SecurityResult(success=False, message="MFA setup failed", error_code="MFA_SETUP_ERROR")


class CertificateAuthenticationProvider(BaseAuthenticationProvider):
    """Certificate-based authentication provider."""

    def __init__(
        self,
        session_manager: SessionManager,
        token_manager: TokenManager,
        certificate_store: Any,
        config: Dict[str, Any],
    ):
        super().__init__(session_manager, token_manager, config)
        self.certificate_store = certificate_store

    async def authenticate_certificate(self, certificate: bytes, signature: bytes, challenge: bytes) -> SecurityResult:
        """Authenticate using client certificate."""
        try:
            # Verify certificate validity
            cert_info = await self.certificate_store.verify_certificate(certificate)

            if not cert_info["valid"]:
                return SecurityResult(success=False, message="Invalid certificate", error_code="INVALID_CERTIFICATE")

            # Verify signature
            is_signature_valid = await self.certificate_store.verify_signature(certificate, signature, challenge)

            if not is_signature_valid:
                return SecurityResult(success=False, message="Invalid signature", error_code="INVALID_SIGNATURE")

            user_id = cert_info.get("subject", {}).get("user_id")
            if not user_id:
                return SecurityResult(
                    success=False, message="Certificate does not contain user ID", error_code="MISSING_USER_ID"
                )

            return SecurityResult(
                success=True,
                message="Certificate authentication successful",
                data={"user_id": user_id, "certificate_info": cert_info, "auth_method": "certificate"},
            )

        except Exception as e:
            logger.error(f"Certificate authentication error: {e}")
            return SecurityResult(
                success=False, message="Certificate authentication failed", error_code="CERT_AUTH_ERROR"
            )
