#!/usr/bin/env python3
"""
JWT Authentication Handler for Unified API Gateway

Implements secure JWT token validation with MFA support and enhanced security.
Integrates with the existing security framework (AES-256-GCM, MFA).
"""

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import os
from typing import Any

from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from pydantic import BaseModel


class TokenPayload(BaseModel):
    """JWT token payload structure."""

    sub: str  # Subject (user ID)
    iat: int  # Issued at
    exp: int  # Expiration time
    scope: list[str] = []  # Permission scopes
    mfa_verified: bool = False  # MFA verification status
    api_key_hash: str | None = None  # API key hash for validation
    rate_limit_tier: str = "standard"  # Rate limiting tier


class JWTHandler:
    """Enhanced JWT authentication handler with security features."""

    def __init__(
        self,
        secret_key: str | None = None,
        algorithm: str = "HS256",
        token_expiry_hours: int = 24,
        require_mfa: bool = True,
        enable_api_key_validation: bool = True,
    ):
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", self._generate_secret())
        self.algorithm = algorithm
        self.token_expiry_hours = token_expiry_hours
        self.require_mfa = require_mfa
        self.enable_api_key_validation = enable_api_key_validation

        # Rate limiting tiers
        self.rate_limit_tiers = {
            "standard": {"requests_per_hour": 1000, "requests_per_minute": 60},
            "premium": {"requests_per_hour": 5000, "requests_per_minute": 200},
            "enterprise": {"requests_per_hour": 10000, "requests_per_minute": 500},
        }

        # Token blacklist for logout/revocation
        self._blacklisted_tokens = set()

    def _generate_secret(self) -> str:
        """Generate a secure secret key if none provided."""
        return base64.urlsafe_b64encode(os.urandom(32)).decode()

    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage in token."""
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]

    def create_token(
        self,
        user_id: str,
        scopes: list[str] = None,
        api_key: str = None,
        mfa_verified: bool = False,
        rate_limit_tier: str = "standard",
        custom_claims: dict[str, Any] = None,
    ) -> str:
        """Create a JWT token with enhanced security claims."""

        now = datetime.now(timezone.utc)
        expiry = now + timedelta(hours=self.token_expiry_hours)

        payload = {
            "sub": user_id,
            "iat": int(now.timestamp()),
            "exp": int(expiry.timestamp()),
            "scope": scopes or ["read"],
            "mfa_verified": mfa_verified,
            "rate_limit_tier": rate_limit_tier,
            "jti": hashlib.md5(f"{user_id}{now.timestamp()}".encode()).hexdigest()[:8],  # JWT ID for revocation
        }

        if api_key and self.enable_api_key_validation:
            payload["api_key_hash"] = self._hash_api_key(api_key)

        if custom_claims:
            payload.update(custom_claims)

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def validate_token(self, token: str, required_scopes: list[str] = None) -> TokenPayload:
        """Validate JWT token and return payload."""

        try:
            # Check if token is blacklisted
            if token in self._blacklisted_tokens:
                raise HTTPException(status_code=401, detail="Token has been revoked")

            # Decode and validate token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Validate required scopes
            if required_scopes:
                token_scopes = payload.get("scope", [])
                if not all(scope in token_scopes for scope in required_scopes):
                    raise HTTPException(
                        status_code=403, detail=f"Insufficient permissions. Required: {required_scopes}"
                    )

            # Validate MFA if required
            if self.require_mfa and not payload.get("mfa_verified", False):
                raise HTTPException(status_code=403, detail="Multi-factor authentication required")

            return TokenPayload(**payload)

        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Token validation failed: {str(e)}")

    def revoke_token(self, token: str) -> bool:
        """Revoke a token by adding it to blacklist."""
        try:
            # Validate token first to ensure it's properly formatted
            jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            self._blacklisted_tokens.add(token)
            return True
        except jwt.InvalidTokenError:
            return False

    def get_rate_limit_config(self, tier: str) -> dict[str, int]:
        """Get rate limiting configuration for a tier."""
        return self.rate_limit_tiers.get(tier, self.rate_limit_tiers["standard"])

    def refresh_token(self, token: str) -> str:
        """Refresh a token if it's valid and not expired soon."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check if token expires within the next hour
            exp_time = datetime.fromtimestamp(payload["exp"], timezone.utc)
            if exp_time - datetime.now(timezone.utc) < timedelta(hours=1):
                # Create new token with same claims
                return self.create_token(
                    user_id=payload["sub"],
                    scopes=payload.get("scope", []),
                    mfa_verified=payload.get("mfa_verified", False),
                    rate_limit_tier=payload.get("rate_limit_tier", "standard"),
                )

            return token  # Token doesn't need refresh yet

        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Cannot refresh expired token")
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token for refresh: {str(e)}")


class JWTBearer(HTTPBearer):
    """FastAPI dependency for JWT authentication."""

    def __init__(self, jwt_handler: JWTHandler, required_scopes: list[str] = None, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
        self.jwt_handler = jwt_handler
        self.required_scopes = required_scopes or []

    async def __call__(self, request: Request) -> TokenPayload:
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)

        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(status_code=401, detail="Invalid authentication scheme")

            # Validate token and return payload
            return self.jwt_handler.validate_token(credentials.credentials, self.required_scopes)
        else:
            raise HTTPException(status_code=401, detail="Invalid authorization code")


class APIKeyValidator:
    """API Key validation with JWT integration."""

    def __init__(self, valid_api_keys: dict[str, dict[str, Any]] = None):
        # Format: {api_key: {"user_id": str, "scopes": List[str], "tier": str}}
        self.valid_api_keys = valid_api_keys or {}

        # Load from environment if no keys provided
        if not self.valid_api_keys:
            self._load_from_env()

    def _load_from_env(self):
        """Load API keys from environment variables."""
        # Example: API_KEY_1=key123:user1:read,write:premium
        for i in range(1, 10):  # Support up to 10 API keys
            env_key = f"API_KEY_{i}"
            if env_key in os.environ:
                parts = os.environ[env_key].split(":")
                if len(parts) >= 3:
                    api_key, user_id, scopes = parts[0], parts[1], parts[2].split(",")
                    tier = parts[3] if len(parts) > 3 else "standard"

                    self.valid_api_keys[api_key] = {"user_id": user_id, "scopes": scopes, "tier": tier}

    def validate_api_key(self, api_key: str) -> dict[str, Any] | None:
        """Validate API key and return user info."""
        return self.valid_api_keys.get(api_key)

    def create_jwt_from_api_key(self, api_key: str, jwt_handler: JWTHandler) -> str | None:
        """Create JWT token from valid API key."""
        user_info = self.validate_api_key(api_key)
        if user_info:
            return jwt_handler.create_token(
                user_id=user_info["user_id"],
                scopes=user_info["scopes"],
                api_key=api_key,
                rate_limit_tier=user_info["tier"],
                mfa_verified=False,  # API keys don't have MFA by default
            )
        return None


# Convenience functions for FastAPI dependency injection
def create_jwt_dependency(
    secret_key: str = None, required_scopes: list[str] = None, require_mfa: bool = True
) -> JWTBearer:
    """Create a JWT dependency for FastAPI endpoints."""
    jwt_handler = JWTHandler(secret_key=secret_key, require_mfa=require_mfa)
    return JWTBearer(jwt_handler, required_scopes)


def create_api_key_to_jwt_dependency(api_keys: dict[str, dict[str, Any]] = None, jwt_secret: str = None) -> callable:
    """Create dependency that converts API key to JWT."""
    api_validator = APIKeyValidator(api_keys)
    jwt_handler = JWTHandler(jwt_secret, require_mfa=False)

    async def api_key_to_jwt(request: Request) -> TokenPayload:
        # Check for API key in header
        api_key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")

        if api_key:
            jwt_token = api_validator.create_jwt_from_api_key(api_key, jwt_handler)
            if jwt_token:
                return jwt_handler.validate_token(jwt_token)

        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return api_key_to_jwt
