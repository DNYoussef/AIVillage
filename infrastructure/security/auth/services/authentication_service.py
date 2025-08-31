"""Authentication Service.

Handles JWT authentication, token management, and user authentication flows.
Extracted from the EnhancedJWTAuthenticator class for better modularity.
"""

import logging
import os
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

from ..interfaces import (
    IAuthenticationService, ISessionManager, IMFAService,
    AuthCredentials, AuthResult, TokenValidationResult, DeviceInfo, MFAMethodType
)

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Authentication-related error."""
    pass


class AuthenticationService(IAuthenticationService):
    """Enhanced JWT authenticator with session management.
    
    Extracted from the EnhancedJWTAuthenticator class to provide
    dedicated authentication functionality following SRP.
    """

    def __init__(
        self, 
        session_manager: ISessionManager,
        mfa_service: IMFAService,
        rbac_system=None,
        config: Dict[str, Any] = None
    ):
        """Initialize authentication service."""
        self.session_manager = session_manager
        self.mfa_service = mfa_service
        self.rbac_system = rbac_system
        self.config = config or {}
        
        # JWT configuration
        self.secret_key = self.config.get("secret_key") or os.getenv("API_SECRET_KEY") or secrets.token_urlsafe(32)
        self.algorithm = self.config.get("algorithm", "HS256")
        self.token_expiry_hours = self.config.get("token_expiry_hours", 24)
        self.refresh_expiry_days = self.config.get("refresh_expiry_days", 30)
        
        # Validate secret key
        if len(self.secret_key) < 32:
            raise AuthenticationError("API_SECRET_KEY must be at least 32 characters long")
        
        if not JWT_AVAILABLE:
            logger.warning("JWT library not available, authentication functionality limited")

    async def authenticate_user(self, credentials: AuthCredentials) -> AuthResult:
        """Authenticate user with credentials."""
        try:
            if not self.rbac_system:
                return AuthResult(
                    success=False,
                    error_message="RBAC system not configured"
                )
            
            # Get user from RBAC system
            user = self.rbac_system.get_user(credentials.username)
            if not user:
                return AuthResult(
                    success=False,
                    error_message="Invalid credentials"
                )
            
            # Verify password
            if not self._verify_password(credentials.password, user):
                return AuthResult(
                    success=False,
                    error_message="Invalid credentials"
                )
            
            # Check MFA requirement
            user_id = user["user_id"]
            mfa_status = self.mfa_service.get_user_mfa_status(user_id)
            mfa_required = any([
                mfa_status.totp_enabled,
                mfa_status.sms_enabled,
                mfa_status.email_enabled
            ])
            
            mfa_verified = False
            if mfa_required:
                if not credentials.mfa_token or not credentials.mfa_method:
                    return AuthResult(
                        success=False,
                        error_message="MFA verification required"
                    )
                
                # Verify MFA token
                mfa_verified = self.mfa_service.verify_mfa(
                    user_id,
                    credentials.mfa_method,
                    credentials.mfa_token,
                    secret=self.mfa_service.get_user_totp_secret(user_id) if credentials.mfa_method == MFAMethodType.TOTP else None
                )
                
                if not mfa_verified:
                    return AuthResult(
                        success=False,
                        error_message="Invalid MFA token"
                    )
            
            # Get user roles and permissions
            roles = self.rbac_system.get_user_roles(user_id)
            permissions = []
            for role in roles:
                permissions.extend(self.rbac_system.get_role_permissions(role))
            
            return AuthResult(
                success=True,
                user_id=user_id,
                roles=[role.value if hasattr(role, 'value') else str(role) for role in roles],
                permissions=[perm.value if hasattr(perm, 'value') else str(perm) for perm in set(permissions)],
                mfa_verified=mfa_verified
            )
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return AuthResult(
                success=False,
                error_message="Authentication failed"
            )

    async def validate_token(self, token: str, token_type: str = "access_token") -> TokenValidationResult:
        """Validate JWT token and check session validity."""
        try:
            if not JWT_AVAILABLE:
                return TokenValidationResult(
                    valid=False,
                    error_message="JWT library not available"
                )
            
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get("type") != token_type:
                return TokenValidationResult(
                    valid=False,
                    error_message=f"Invalid token type: expected {token_type}"
                )
            
            jti = payload.get("jti")
            if not jti:
                return TokenValidationResult(
                    valid=False,
                    error_message="Token missing JTI"
                )
            
            # Check if token is revoked
            if await self.session_manager.is_token_revoked(jti):
                return TokenValidationResult(
                    valid=False,
                    error_message="Token has been revoked"
                )
            
            # Verify session is still active
            session_id = payload.get("session_id")
            if session_id:
                session_data = await self.session_manager.get_session(session_id)
                if not session_data or not session_data.is_active:
                    return TokenValidationResult(
                        valid=False,
                        error_message="Session is no longer active"
                    )
                
                # Update session activity
                await self.session_manager.update_session_activity(session_id)
            
            return TokenValidationResult(
                valid=True,
                payload=payload
            )
            
        except jwt.ExpiredSignatureError:
            return TokenValidationResult(
                valid=False,
                error_message="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            return TokenValidationResult(
                valid=False,
                error_message=f"Invalid token: {e}"
            )
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return TokenValidationResult(
                valid=False,
                error_message="Token validation failed"
            )

    async def create_session_tokens(
        self,
        user_id: str,
        device_info: DeviceInfo,
        roles: List[str] = None,
        permissions: List[str] = None,
        mfa_verified: bool = False,
    ) -> Dict[str, Any]:
        """Create JWT tokens with session tracking."""
        try:
            if not JWT_AVAILABLE:
                raise AuthenticationError("JWT library not available")
            
            # Create session
            session_id = await self.session_manager.create_session(
                user_id=user_id,
                device_info=device_info,
                roles=roles or [],
                permissions=permissions or []
            )
            
            # Generate unique JTIs
            access_jti = secrets.token_urlsafe(16)
            refresh_jti = secrets.token_urlsafe(16)
            
            now = datetime.utcnow()
            
            # Create access token
            access_payload = {
                "user_id": user_id,
                "session_id": session_id,
                "roles": roles or [],
                "permissions": permissions or [],
                "mfa_verified": mfa_verified,
                "iat": now,
                "exp": now + timedelta(hours=self.token_expiry_hours),
                "type": "access_token",
                "jti": access_jti,
            }
            
            # Create refresh token
            refresh_payload = {
                "user_id": user_id,
                "session_id": session_id,
                "iat": now,
                "exp": now + timedelta(days=self.refresh_expiry_days),
                "type": "refresh_token",
                "jti": refresh_jti,
            }
            
            access_token = jwt.encode(access_payload, self.secret_key, algorithm=self.algorithm)
            refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm=self.algorithm)
            
            # Track tokens in session
            await self.session_manager.add_token_to_session(session_id, access_jti, "access")
            await self.session_manager.add_token_to_session(session_id, refresh_jti, "refresh")
            
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "Bearer",
                "expires_in": self.token_expiry_hours * 3600,
                "session_id": session_id,
                "mfa_verified": mfa_verified,
            }
            
        except Exception as e:
            logger.error(f"Token creation failed: {e}")
            raise AuthenticationError(f"Failed to create session tokens: {e}")

    async def revoke_token(self, jti: str) -> bool:
        """Revoke specific token."""
        try:
            return await self.session_manager.revoke_token(jti)
        except Exception as e:
            logger.error(f"Token revocation failed: {e}")
            return False

    async def revoke_session(self, session_id: str) -> bool:
        """Revoke entire session."""
        try:
            return await self.session_manager.revoke_session(session_id)
        except Exception as e:
            logger.error(f"Session revocation failed: {e}")
            return False

    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token using refresh token."""
        try:
            # Validate refresh token
            validation_result = await self.validate_token(refresh_token, "refresh_token")
            if not validation_result.valid:
                raise AuthenticationError(validation_result.error_message)
            
            payload = validation_result.payload
            user_id = payload["user_id"]
            session_id = payload["session_id"]
            
            # Get session data for roles/permissions
            session_data = await self.session_manager.get_session(session_id)
            if not session_data:
                raise AuthenticationError("Session not found")
            
            # Create new access token
            new_access_jti = secrets.token_urlsafe(16)
            now = datetime.utcnow()
            
            access_payload = {
                "user_id": user_id,
                "session_id": session_id,
                "roles": session_data.roles,
                "permissions": session_data.permissions,
                "mfa_verified": payload.get("mfa_verified", False),
                "iat": now,
                "exp": now + timedelta(hours=self.token_expiry_hours),
                "type": "access_token",
                "jti": new_access_jti,
            }
            
            new_access_token = jwt.encode(access_payload, self.secret_key, algorithm=self.algorithm)
            
            # Track new token
            await self.session_manager.add_token_to_session(session_id, new_access_jti, "access")
            
            return {
                "access_token": new_access_token,
                "token_type": "Bearer",
                "expires_in": self.token_expiry_hours * 3600,
            }
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise AuthenticationError(f"Failed to refresh token: {e}")

    def _verify_password(self, password: str, user: Dict[str, Any]) -> bool:
        """Verify password against user data."""
        try:
            import hashlib
            import hmac
            
            salt = user.get("password_salt", "")
            expected_hash = user.get("password_hash", "")
            
            if not salt or not expected_hash:
                return False
            
            # Use PBKDF2 for password hashing
            actual_hash = hashlib.pbkdf2_hmac(
                "sha256", 
                password.encode(), 
                salt.encode(), 
                100000
            ).hex()
            
            return hmac.compare_digest(expected_hash, actual_hash)
            
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False

    async def logout_user(self, session_id: str) -> bool:
        """Logout user by revoking session."""
        return await self.revoke_session(session_id)

    async def logout_all_user_sessions(self, user_id: str) -> int:
        """Logout all sessions for user."""
        try:
            return await self.session_manager.revoke_all_user_sessions(user_id)
        except Exception as e:
            logger.error(f"Logout all sessions failed: {e}")
            return 0