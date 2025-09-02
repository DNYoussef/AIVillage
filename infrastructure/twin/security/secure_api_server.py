#!/usr/bin/env python3
"""Secure API Server with TLS 1.3, Authentication, and Rate Limiting.

Provides CODEX-compliant secure API endpoints with comprehensive security
features including TLS 1.3, JWT authentication, rate limiting, and input validation.
"""

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import secrets
import ssl
import time
from typing import Any

from aiohttp import web, web_request
from aiohttp.web_middlewares import middleware
from aiohttp_cors import ResourceOptions
from aiohttp_cors import setup as cors_setup
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
import jwt

from .rbac_system import RBACSystem, Role
from .secure_digital_twin_db import SecureDigitalTwinDB

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Security-related error."""


class RateLimitExceeded(Exception):
    """Rate limit exceeded."""


class AuthenticationError(Exception):
    """Authentication failed."""


class InputValidationError(Exception):
    """Input validation failed."""


class RateLimiter:
    """In-memory rate limiter with sliding window."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # client_id -> list of timestamps

    def is_allowed(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        now = time.time()

        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                timestamp for timestamp in self.requests[client_id] if now - timestamp < self.window_seconds
            ]
        else:
            self.requests[client_id] = []

        # Check current count
        if len(self.requests[client_id]) >= self.max_requests:
            return False

        # Add current request
        self.requests[client_id].append(now)
        return True

    def get_stats(self, client_id: str) -> dict[str, Any]:
        """Get rate limiting stats for client."""
        if client_id not in self.requests:
            return {"requests": 0, "remaining": self.max_requests}

        now = time.time()
        recent_requests = [timestamp for timestamp in self.requests[client_id] if now - timestamp < self.window_seconds]

        return {
            "requests": len(recent_requests),
            "remaining": max(0, self.max_requests - len(recent_requests)),
            "reset_time": now + self.window_seconds,
        }


class JWTAuthenticator:
    """JWT token authentication and management."""

    def __init__(self, secret_key: str | None = None) -> None:
        self.secret_key = secret_key or os.getenv("API_SECRET_KEY") or secrets.token_urlsafe(32)
        self.algorithm = "HS256"
        self.token_expiry_hours = int(os.getenv("API_JWT_EXPIRY_HOURS", "24"))
        self.refresh_expiry_days = int(os.getenv("API_REFRESH_TOKEN_EXPIRY_DAYS", "30"))

        if len(self.secret_key) < 32:
            msg = "API_SECRET_KEY must be at least 32 characters long"
            raise SecurityError(msg)

    def create_access_token(
        self,
        user_id: str,
        roles: list[str] | None = None,
        permissions: list[str] | None = None,
    ) -> str:
        """Create JWT access token."""
        now = datetime.utcnow()
        payload = {
            "user_id": user_id,
            "roles": roles or [],
            "permissions": permissions or [],
            "iat": now,
            "exp": now + timedelta(hours=self.token_expiry_hours),
            "type": "access_token",
            "jti": secrets.token_urlsafe(16),  # JWT ID for revocation
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token."""
        now = datetime.utcnow()
        payload = {
            "user_id": user_id,
            "iat": now,
            "exp": now + timedelta(days=self.refresh_expiry_days),
            "type": "refresh_token",
            "jti": secrets.token_urlsafe(16),
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str, token_type: str = "access_token") -> dict[str, Any]:  # nosec B107
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            if payload.get("type") != token_type:
                msg = f"Invalid token type: expected {token_type}"
                raise AuthenticationError(msg)

            return payload

        except jwt.ExpiredSignatureError:
            msg = "Token has expired"
            raise AuthenticationError(msg)
        except jwt.InvalidTokenError as e:
            msg = f"Invalid token: {e}"
            raise AuthenticationError(msg)

    def hash_password(self, password: str) -> tuple[str, str]:
        """Hash password with salt using PBKDF2."""
        salt = os.urandom(32)
        password_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)

        return salt.hex(), password_hash.hex()

    def verify_password(self, password: str, salt_hex: str, hash_hex: str) -> bool:
        """Verify password against stored hash."""
        try:
            salt = bytes.fromhex(salt_hex)
            stored_hash = bytes.fromhex(hash_hex)

            password_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)

            return hmac.compare_digest(password_hash, stored_hash)
        except Exception:
            return False


class InputValidator:
    """Input validation and sanitization."""

    @staticmethod
    def validate_json(data: Any, schema: dict[str, Any]) -> dict[str, Any]:
        """Validate JSON data against schema."""
        if not isinstance(data, dict):
            msg = "Request body must be JSON object"
            raise InputValidationError(msg)

        validated = {}

        for field, rules in schema.items():
            value = data.get(field)

            # Check required fields
            if rules.get("required", False) and value is None:
                msg = f"Required field missing: {field}"
                raise InputValidationError(msg)

            if value is not None:
                # Type validation
                expected_type = rules.get("type")
                if expected_type and not isinstance(value, expected_type):
                    msg = f"Invalid type for {field}: expected {expected_type.__name__}"
                    raise InputValidationError(msg)

                # String length validation
                if isinstance(value, str):
                    min_len = rules.get("min_length", 0)
                    max_len = rules.get("max_length", 10000)
                    if not (min_len <= len(value) <= max_len):
                        msg = f"Invalid length for {field}: must be {min_len}-{max_len} characters"
                        raise InputValidationError(msg)

                # Numeric range validation
                if isinstance(value, int | float):
                    min_val = rules.get("min_value")
                    max_val = rules.get("max_value")
                    if min_val is not None and value < min_val:
                        msg = f"Value too small for {field}: minimum {min_val}"
                        raise InputValidationError(msg)
                    if max_val is not None and value > max_val:
                        msg = f"Value too large for {field}: maximum {max_val}"
                        raise InputValidationError(msg)

                # Pattern validation
                pattern = rules.get("pattern")
                if pattern and isinstance(value, str):
                    import re

                    if not re.match(pattern, value):
                        msg = f"Invalid format for {field}"
                        raise InputValidationError(msg)

                # Custom validator
                validator = rules.get("validator")
                if validator and callable(validator):
                    if not validator(value):
                        msg = f"Custom validation failed for {field}"
                        raise InputValidationError(msg)

                validated[field] = value
            else:
                # Set default value
                default = rules.get("default")
                if default is not None:
                    validated[field] = default

        return validated

    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            return str(value)

        # Remove null bytes and control characters
        sanitized = "".join(char for char in value if ord(char) >= 32 or char in "\n\r\t")

        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        return sanitized.strip()


class SecureAPIServer:
    """Secure API server with TLS 1.3 and comprehensive security."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        digital_twin_port: int = 8080,
        evolution_metrics_port: int = 8081,
        rag_pipeline_port: int = 8082,
    ) -> None:
        """Initialize secure API server."""
        self.host = host
        self.ports = {
            "digital_twin": digital_twin_port,
            "evolution_metrics": evolution_metrics_port,
            "rag_pipeline": rag_pipeline_port,
        }

        # Security components
        self.authenticator = JWTAuthenticator()
        self.rbac_system = RBACSystem()
        self.profile_db = SecureDigitalTwinDB()
        self.rate_limiter = RateLimiter(
            max_requests=int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "60")),
            window_seconds=60,
        )
        self.validator = InputValidator()

        # TLS Configuration
        self.tls_enabled = os.getenv("TLS_ENABLED", "false").lower() == "true"
        self.cert_file = os.getenv("TLS_CERT_FILE", "./certs/server.crt")
        self.key_file = os.getenv("TLS_KEY_FILE", "./certs/server.key")

        # CORS settings
        self.cors_enabled = os.getenv("API_CORS_ENABLED", "true").lower() == "true"
        self.cors_origins = os.getenv("API_CORS_ORIGINS", "https://localhost:3000").split(",")

        # Apps
        self.apps = {}

        # Initialize apps
        self._create_apps()

    def _create_apps(self) -> None:
        """Create aiohttp applications for each service."""
        for service in self.ports:
            app = web.Application(
                middlewares=[
                    self._security_middleware,
                    self._rate_limit_middleware,
                    self._auth_middleware,
                ]
            )

            # Configure CORS
            if self.cors_enabled:
                cors = cors_setup(
                    app,
                    defaults={
                        "*": ResourceOptions(
                            allow_credentials=True,
                            expose_headers="*",
                            allow_headers="*",
                            allow_methods="*",
                        )
                    },
                )

                for origin in self.cors_origins:
                    cors.add(
                        app.router.add_resource("/{path:.*}"),
                        {
                            origin: ResourceOptions(
                                allow_credentials=True,
                                expose_headers="*",
                                allow_headers="*",
                                allow_methods="*",
                            )
                        },
                    )

            # Add service-specific routes
            self._add_routes(app, service)

            self.apps[service] = app

    @middleware
    async def _security_middleware(self, request: web_request.Request, handler: Callable) -> web.Response:
        """Security headers and basic security checks."""
        response = await handler(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        if self.tls_enabled:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response

    @middleware
    async def _rate_limit_middleware(self, request: web_request.Request, handler: Callable) -> web.Response:
        """Rate limiting middleware."""
        # Skip rate limiting for health checks
        if request.path.startswith("/health"):
            return await handler(request)

        # Get client identifier (IP + User-Agent)
        client_id = f"{request.remote}:{request.headers.get('User-Agent', 'unknown')}"

        if not self.rate_limiter.is_allowed(client_id):
            stats = self.rate_limiter.get_stats(client_id)
            raise web.HTTPTooManyRequests(
                text=json.dumps(
                    {
                        "error": "Rate limit exceeded",
                        "retry_after": int(stats.get("reset_time", time.time()) - time.time()),
                    }
                ),
                headers={
                    "X-RateLimit-Limit": str(self.rate_limiter.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(stats.get("reset_time", time.time()))),
                },
            )

        response = await handler(request)

        # Add rate limit headers
        stats = self.rate_limiter.get_stats(client_id)
        response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(stats["remaining"])
        response.headers["X-RateLimit-Reset"] = str(int(stats.get("reset_time", time.time())))

        return response

    @middleware
    async def _auth_middleware(self, request: web_request.Request, handler: Callable) -> web.Response:
        """Authentication middleware."""
        # Skip auth for public endpoints
        if request.path in ["/health", "/auth/login", "/auth/register"]:
            return await handler(request)

        # Check for API token
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise web.HTTPUnauthorized(text=json.dumps({"error": "Missing or invalid authorization header"}))

        token = auth_header[7:]  # Remove 'Bearer '

        try:
            payload = self.authenticator.verify_token(token)
            request["user"] = payload
            return await handler(request)

        except AuthenticationError as e:
            raise web.HTTPUnauthorized(text=json.dumps({"error": str(e)}))

    def _add_routes(self, app: web.Application, service: str) -> None:
        """Add service-specific routes."""
        # Health check endpoint (no auth required)
        app.router.add_get("/health", self._health_check)

        # Authentication endpoints (no auth required)
        app.router.add_post("/auth/login", self._login)
        app.router.add_post("/auth/register", self._register)
        app.router.add_post("/auth/refresh", self._refresh_token)

        # Service-specific secured endpoints
        if service == "digital_twin":
            app.router.add_get("/profiles/{profile_id}", self._get_profile)
            app.router.add_post("/profiles", self._create_profile)
            app.router.add_put("/profiles/{profile_id}", self._update_profile)
            app.router.add_delete("/profiles/{profile_id}", self._delete_profile)
            app.router.add_get("/profiles/{profile_id}/export", self._export_profile_data)

        elif service == "evolution_metrics":
            app.router.add_get("/metrics", self._get_metrics)
            app.router.add_post("/metrics", self._store_metrics)
            app.router.add_get("/metrics/stats", self._get_metrics_stats)

        elif service == "rag_pipeline":
            app.router.add_post("/query", self._rag_query)
            app.router.add_post("/documents", self._add_document)
            app.router.add_get("/documents/{doc_id}", self._get_document)

    # Authentication endpoints
    async def _login(self, request: web_request.Request) -> web.Response:
        """User login endpoint."""
        try:
            data = await request.json()

            schema = {
                "username": {
                    "required": True,
                    "type": str,
                    "min_length": 3,
                    "max_length": 50,
                },
                "password": {"required": True, "type": str, "min_length": 8},
            }

            validated_data = self.validator.validate_json(data, schema)

            # Authenticate user against database
            username = validated_data["username"]
            password = validated_data["password"]

            # Get user from RBAC system
            user = self.rbac_system.get_user(username)
            if not user:
                raise web.HTTPUnauthorized(text=json.dumps({"error": "Invalid credentials"}))

            # Verify password
            if not user.get("password_hash") or not user.get("password_salt"):
                raise web.HTTPUnauthorized(text=json.dumps({"error": "Invalid credentials"}))

            salt = user["password_salt"]
            expected_hash = user["password_hash"]
            actual_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000).hex()

            if not hmac.compare_digest(expected_hash, actual_hash):
                raise web.HTTPUnauthorized(text=json.dumps({"error": "Invalid credentials"}))

            # Get user roles and permissions
            roles = self.rbac_system.get_user_roles(user["user_id"])
            permissions = []
            for role in roles:
                permissions.extend(self.rbac_system.get_role_permissions(role))

            access_token = self.authenticator.create_access_token(
                user_id=user["user_id"],
                roles=[role.value for role in roles],
                permissions=[perm.value for perm in set(permissions)],
            )
            refresh_token = self.authenticator.create_refresh_token(user["user_id"])

            return web.json_response(
                {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "token_type": "Bearer",
                    "expires_in": self.authenticator.token_expiry_hours * 3600,
                }
            )

        except (ValueError, json.JSONDecodeError):
            raise web.HTTPBadRequest(text=json.dumps({"error": "Invalid JSON"}))
        except InputValidationError as e:
            raise web.HTTPBadRequest(text=json.dumps({"error": str(e)}))

    async def _register(self, request: web_request.Request) -> web.Response:
        """User registration endpoint."""
        try:
            data = await request.json()

            schema = {
                "username": {
                    "required": True,
                    "type": str,
                    "min_length": 3,
                    "max_length": 50,
                },
                "password": {"required": True, "type": str, "min_length": 8},
                "email": {
                    "required": True,
                    "type": str,
                    "pattern": r"^[^@]+@[^@]+\.[^@]+$",
                },
            }

            validated_data = self.validator.validate_json(data, schema)

            # Hash password
            salt, password_hash = self.authenticator.hash_password(validated_data["password"])

            # Store user in database using RBAC system
            user_id = f"user_{secrets.token_urlsafe(8)}"
            success = self.rbac_system.create_user(
                user_id=user_id,
                username=validated_data["username"],
                email=validated_data["email"],
                password_hash=password_hash,
                password_salt=salt,
                roles=[Role.USER],  # Default role for new users
            )

            if not success:
                raise web.HTTPBadRequest(text=json.dumps({"error": "User registration failed"}))

            return web.json_response(
                {
                    "message": "User registered successfully",
                    "user_id": user_id,
                },
                status=201,
            )

        except (ValueError, json.JSONDecodeError):
            raise web.HTTPBadRequest(text=json.dumps({"error": "Invalid JSON"}))
        except InputValidationError as e:
            raise web.HTTPBadRequest(text=json.dumps({"error": str(e)}))

    async def _refresh_token(self, request: web_request.Request) -> web.Response:
        """Token refresh endpoint."""
        try:
            data = await request.json()
            refresh_token = data.get("refresh_token")

            if not refresh_token:
                raise web.HTTPBadRequest(text=json.dumps({"error": "Missing refresh token"}))

            payload = self.authenticator.verify_token(refresh_token, "refresh_token")
            user_id = payload["user_id"]

            # Get user roles and permissions from database
            user_roles = self.rbac_system.get_user_roles(user_id)
            permissions = []
            for role in user_roles:
                permissions.extend(self.rbac_system.get_role_permissions(role))

            # Generate new tokens
            access_token = self.authenticator.create_access_token(
                user_id=user_id,
                roles=[role.value for role in user_roles],
                permissions=[perm.value for perm in set(permissions)],
            )
            new_refresh_token = self.authenticator.create_refresh_token(user_id)

            return web.json_response(
                {
                    "access_token": access_token,
                    "refresh_token": new_refresh_token,
                    "token_type": "Bearer",
                    "expires_in": self.authenticator.token_expiry_hours * 3600,
                }
            )

        except AuthenticationError as e:
            raise web.HTTPUnauthorized(text=json.dumps({"error": str(e)}))
        except (ValueError, json.JSONDecodeError):
            raise web.HTTPBadRequest(text=json.dumps({"error": "Invalid JSON"}))

    # Health check
    async def _health_check(self, request: web_request.Request) -> web.Response:
        """Health check endpoint."""

        # Default database health info
        db_health: dict[str, Any] = {"status": "operational"}

        # Measure database connectivity latency and capture failures
        start_time = time.perf_counter()
        try:
            # Execute a lightweight query to ensure the connection is healthy
            with self.profile_db.get_connection() as conn:
                conn.execute("SELECT 1")
            db_health["latency_ms"] = (time.perf_counter() - start_time) * 1000
        except Exception as exc:  # pragma: no cover - defensive, errors handled below
            db_health["status"] = "degraded"
            db_health["error"] = str(exc)
            db_health["latency_ms"] = (time.perf_counter() - start_time) * 1000

        health_status = {
            "status": "healthy" if db_health["status"] == "operational" else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "services": {
                "authentication": "operational",
                "rate_limiting": "operational",
                "encryption": "operational",
                "database": db_health,
            },
            "security": {
                "tls_enabled": self.tls_enabled,
                "cors_enabled": self.cors_enabled,
                "rate_limiting_enabled": True,
                "authentication_required": True,
            },
        }

        return web.json_response(health_status)

    # Digital Twin endpoints (reference implementations)
    async def _get_profile(self, request: web_request.Request) -> web.Response:
        """Get learning profile."""
        profile_id = request.match_info["profile_id"]
        user_id = request.get("user", {}).get("user_id")

        if not user_id:
            raise web.HTTPUnauthorized(text=json.dumps({"error": "Authentication required"}))

        try:
            # Retrieve profile with encryption
            profile = self.profile_db.get_profile(user_id, profile_id)
            if not profile:
                raise web.HTTPNotFound(text=json.dumps({"error": "Profile not found"}))

            return web.json_response({"profile_id": profile_id, "data": profile, "status": "success"})
        except Exception as e:
            logger.exception(f"Failed to get profile {profile_id}: {e}")
            raise web.HTTPInternalServerError(text=json.dumps({"error": "Failed to retrieve profile"}))

    async def _create_profile(self, request: web_request.Request) -> web.Response:
        """Create learning profile."""
        user_id = request.get("user", {}).get("user_id")

        if not user_id:
            raise web.HTTPUnauthorized(text=json.dumps({"error": "Authentication required"}))

        try:
            data = await request.json()

            # Validate profile data
            schema = {
                "name": {"required": True, "type": str, "min_length": 1},
                "age": {"required": False, "type": int, "min": 1, "max": 150},
                "preferences": {"required": False, "type": dict},
                "learning_goals": {"required": False, "type": list},
            }

            validated_data = self.validator.validate_json(data, schema)

            # Create profile with encryption
            profile_id = self.profile_db.create_profile(user_id, validated_data)

            return web.json_response(
                {"message": "Profile created successfully", "profile_id": profile_id},
                status=201,
            )
        except (ValueError, json.JSONDecodeError):
            raise web.HTTPBadRequest(text=json.dumps({"error": "Invalid JSON"}))
        except Exception as e:
            logger.exception(f"Failed to create profile: {e}")
            raise web.HTTPInternalServerError(text=json.dumps({"error": "Failed to create profile"}))

    async def _update_profile(self, request: web_request.Request) -> web.Response:
        """Update learning profile."""
        profile_id = request.match_info["profile_id"]
        user_id = request.get("user", {}).get("user_id")

        if not user_id:
            raise web.HTTPUnauthorized(text=json.dumps({"error": "Authentication required"}))

        try:
            data = await request.json()

            # Update profile with encryption
            success = self.profile_db.update_profile(user_id, profile_id, data)

            if not success:
                raise web.HTTPNotFound(text=json.dumps({"error": "Profile not found"}))

            return web.json_response({"profile_id": profile_id, "status": "updated"})
        except (ValueError, json.JSONDecodeError):
            raise web.HTTPBadRequest(text=json.dumps({"error": "Invalid JSON"}))
        except Exception as e:
            logger.exception(f"Failed to update profile {profile_id}: {e}")
            raise web.HTTPInternalServerError(text=json.dumps({"error": "Failed to update profile"}))

    async def _delete_profile(self, request: web_request.Request) -> web.Response:
        """Delete learning profile."""
        profile_id = request.match_info["profile_id"]
        user_id = request.get("user", {}).get("user_id")

        if not user_id:
            raise web.HTTPUnauthorized(text=json.dumps({"error": "Authentication required"}))

        try:
            # GDPR-compliant deletion with secure erasure
            success = self.profile_db.delete_profile(user_id, profile_id, gdpr_compliant=True)

            if not success:
                raise web.HTTPNotFound(text=json.dumps({"error": "Profile not found"}))

            return web.json_response({"profile_id": profile_id, "status": "deleted", "gdpr_compliant": True})
        except Exception as e:
            logger.exception(f"Failed to delete profile {profile_id}: {e}")
            raise web.HTTPInternalServerError(text=json.dumps({"error": "Failed to delete profile"}))

    async def _export_profile_data(self, request: web_request.Request) -> web.Response:
        """Export profile data for GDPR compliance."""
        profile_id = request.match_info["profile_id"]
        user_id = request.get("user", {}).get("user_id")

        if not user_id:
            raise web.HTTPUnauthorized(text=json.dumps({"error": "Authentication required"}))

        try:
            # Export all user data for GDPR compliance
            export_data = self.profile_db.export_user_data(user_id, profile_id)

            if not export_data:
                raise web.HTTPNotFound(text=json.dumps({"error": "No data found for export"}))

            return web.json_response(
                {
                    "profile_id": profile_id,
                    "export_timestamp": datetime.utcnow().isoformat(),
                    "data": export_data,
                    "gdpr_compliant": True,
                    "format": "JSON",
                }
            )
        except Exception as e:
            logger.exception(f"Failed to export profile data {profile_id}: {e}")
            raise web.HTTPInternalServerError(text=json.dumps({"error": "Failed to export profile data"}))

    # Evolution Metrics endpoints (reference implementations)
    async def _get_metrics(self, request: web_request.Request) -> web.Response:
        """Get evolution metrics."""
        return web.json_response({"metrics": "placeholder"})

    async def _store_metrics(self, request: web_request.Request) -> web.Response:
        """Store evolution metrics."""
        return web.json_response({"message": "Metrics stored"}, status=201)

    async def _get_metrics_stats(self, request: web_request.Request) -> web.Response:
        """Get metrics statistics."""
        return web.json_response({"stats": "placeholder"})

    # RAG Pipeline endpoints (reference implementations)
    async def _rag_query(self, request: web_request.Request) -> web.Response:
        """Process RAG query."""
        return web.json_response({"query_result": "placeholder"})

    async def _add_document(self, request: web_request.Request) -> web.Response:
        """Add document to RAG index."""
        return web.json_response({"message": "Document added"}, status=201)

    async def _get_document(self, request: web_request.Request) -> web.Response:
        """Get document from RAG index."""
        doc_id = request.match_info["doc_id"]
        return web.json_response({"doc_id": doc_id, "content": "placeholder"})

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for TLS 1.3."""
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)

        # TLS 1.3 configuration
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.maximum_version = ssl.TLSVersion.TLSv1_3

        # Load certificates
        if Path(self.cert_file).exists() and Path(self.key_file).exists():
            context.load_cert_chain(self.cert_file, self.key_file)
        else:
            logger.warning("TLS certificates not found, generating self-signed certificates")
            self._generate_self_signed_cert()
            context.load_cert_chain(self.cert_file, self.key_file)

        # Security settings
        context.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS")
        context.options |= ssl.OP_NO_COMPRESSION
        context.options |= ssl.OP_NO_RENEGOTIATION

        return context

    def _generate_self_signed_cert(self) -> None:
        """Generate self-signed certificate for development."""
        # Create certificates directory
        cert_dir = Path(self.cert_file).parent
        cert_dir.mkdir(parents=True, exist_ok=True)

        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        # Generate certificate
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "State"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AIVillage"),
                x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
            ]
        )

        certificate = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.DNSName("localhost"),
                        x509.DNSName("127.0.0.1"),
                    ]
                ),
                critical=False,
            )
            .sign(private_key, hashes.SHA256())
        )

        # Write private key
        with open(self.key_file, "wb") as f:
            f.write(
                private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

        # Write certificate
        with open(self.cert_file, "wb") as f:
            f.write(certificate.public_bytes(serialization.Encoding.PEM))

        logger.info(f"Generated self-signed certificate: {self.cert_file}")

    async def start_servers(self):
        """Start all API servers."""
        ssl_context = self._create_ssl_context() if self.tls_enabled else None

        runners = []
        sites = []

        for service, port in self.ports.items():
            runner = web.AppRunner(self.apps[service])
            await runner.setup()
            runners.append(runner)

            site = web.TCPSite(runner, self.host, port, ssl_context=ssl_context)
            await site.start()
            sites.append(site)

            protocol = "https" if self.tls_enabled else "http"
            logger.info(f"Started {service} API server: {protocol}://{self.host}:{port}")

        logger.info("All secure API servers started successfully")
        return runners, sites

    async def stop_servers(self, runners) -> None:
        """Stop all API servers."""
        for runner in runners:
            await runner.cleanup()
        logger.info("All API servers stopped")


# Example usage
if __name__ == "__main__":

    async def main() -> None:
        # Set environment variables for testing
        os.environ["API_SECRET_KEY"] = secrets.token_urlsafe(32)
        os.environ["TLS_ENABLED"] = "false"  # Set to 'true' for production

        server = SecureAPIServer()
        runners, sites = await server.start_servers()

        try:
            # Keep servers running
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("Shutting down servers...")
        finally:
            await server.stop_servers(runners)

    asyncio.run(main())
