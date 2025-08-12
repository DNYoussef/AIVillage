"""Secure Redis Client with TLS enforcement for production.

This module provides a Redis client wrapper that enforces TLS in production
and validates connection security.
"""

import logging
import os
import ssl
from typing import Any
from urllib.parse import urlparse

try:
    import redis
except ImportError:
    redis = None

from .http_security_validator import HTTPSecurityError, validate_url_for_production

logger = logging.getLogger(__name__)


class SecureRedisError(Exception):
    """Raised when Redis security validation fails."""


def create_secure_redis_client(
    redis_url: str,
    ssl_required: bool | None = None,
    ssl_cert_reqs: str = "required",
    ssl_ca_certs: str | None = None,
    ssl_certfile: str | None = None,
    ssl_keyfile: str | None = None,
    **kwargs,
) -> object | None:
    """Create a secure Redis client with TLS validation.

    Args:
        redis_url: Redis connection URL
        ssl_required: Whether SSL/TLS is required (auto-detected from environment if None)
        ssl_cert_reqs: SSL certificate requirements ("required", "optional", "none")
        ssl_ca_certs: Path to CA certificates file
        ssl_certfile: Path to client certificate file
        ssl_keyfile: Path to client private key file
        **kwargs: Additional Redis connection parameters

    Returns:
        Redis client instance or None if Redis is unavailable

    Raises:
        SecureRedisError: If security validation fails
        HTTPSecurityError: If HTTP URL is used in production
    """
    if redis is None:
        logger.warning("Redis library not available")
        return None

    # Validate URL security in production
    try:
        validate_url_for_production(redis_url, "Redis connection")
    except HTTPSecurityError as e:
        raise SecureRedisError(f"Redis URL security validation failed: {e}")

    # Parse Redis URL
    parsed = urlparse(redis_url)
    if not parsed.hostname:
        raise SecureRedisError(f"Invalid Redis URL: {redis_url}")

    # Determine if TLS should be required
    is_production = os.getenv("AIVILLAGE_ENV") == "production"
    if ssl_required is None:
        ssl_required = is_production or parsed.scheme == "rediss"

    # Enforce TLS in production
    if is_production and not (ssl_required or parsed.scheme == "rediss"):
        raise SecureRedisError(
            "TLS is required for Redis connections in production environment. "
            f"Use rediss:// URL or set ssl_required=True. Got: {redis_url}"
        )

    # Build connection parameters
    connection_params = {
        "host": parsed.hostname,
        "port": parsed.port or (6380 if ssl_required else 6379),
        "db": int(parsed.path.lstrip("/")) if parsed.path and parsed.path != "/" else 0,
        "decode_responses": True,
        "socket_connect_timeout": kwargs.pop("socket_connect_timeout", 5),
        "socket_timeout": kwargs.pop("socket_timeout", 5),
        **kwargs,
    }

    # Add authentication if provided in URL
    if parsed.password:
        connection_params["password"] = parsed.password
    if parsed.username:
        connection_params["username"] = parsed.username

    # Configure SSL/TLS if required
    if ssl_required or parsed.scheme == "rediss":
        ssl_context = ssl.create_default_context()

        # Configure certificate verification
        if ssl_cert_reqs == "required":
            ssl_context.check_hostname = True
            ssl_context.verify_mode = ssl.CERT_REQUIRED
        elif ssl_cert_reqs == "optional":
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_OPTIONAL
        elif ssl_cert_reqs == "none":
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        # Load custom CA certificates
        if ssl_ca_certs:
            ssl_context.load_verify_locations(ssl_ca_certs)

        # Load client certificate and key
        if ssl_certfile and ssl_keyfile:
            ssl_context.load_cert_chain(ssl_certfile, ssl_keyfile)

        connection_params.update(
            {
                "ssl": True,
                "ssl_context": ssl_context,
            }
        )

        logger.info(
            f"Redis TLS enabled for {parsed.hostname}:{connection_params['port']}"
        )

    try:
        # Create Redis client
        client = redis.Redis(**connection_params)

        # Test connection
        client.ping()

        logger.info(
            f"Connected to Redis at {parsed.hostname}:{connection_params['port']}"
            f" (TLS: {'enabled' if ssl_required else 'disabled'})"
        )

        return client

    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        if is_production:
            raise SecureRedisError(f"Redis connection failed in production: {e}")
        return None


def validate_redis_url_security(redis_url: str) -> None:
    """Validate Redis URL for production security requirements.

    Args:
        redis_url: Redis URL to validate

    Raises:
        SecureRedisError: If URL doesn't meet security requirements
    """
    if not redis_url:
        return

    is_production = os.getenv("AIVILLAGE_ENV") == "production"
    parsed = urlparse(redis_url)

    # Check for HTTP instead of Redis protocol
    if parsed.scheme in ("http", "https"):
        raise SecureRedisError(
            f"HTTP URL provided for Redis connection: {redis_url}. "
            "Use redis:// or rediss:// protocol instead."
        )

    # Enforce TLS in production
    if is_production:
        if parsed.scheme == "redis":
            logger.warning(
                f"Redis URL uses unencrypted connection in production: {redis_url}. "
                "Consider using rediss:// for TLS encryption."
            )
        elif parsed.scheme not in ("redis", "rediss"):
            raise SecureRedisError(
                f"Invalid Redis protocol in production: {parsed.scheme}. "
                "Use redis:// or rediss:// protocols."
            )


def get_production_redis_config() -> dict[str, Any]:
    """Get secure Redis configuration for production environment.

    Returns:
        Dictionary with production Redis configuration
    """
    return {
        "ssl_required": True,
        "ssl_cert_reqs": "required",
        "socket_connect_timeout": 10,
        "socket_timeout": 10,
        "retry_on_timeout": True,
        "health_check_interval": 30,
        "decode_responses": True,
    }


def create_production_redis_pool(redis_url: str, **kwargs) -> object | None:
    """Create a production-ready Redis connection pool with security settings.

    Args:
        redis_url: Redis connection URL
        **kwargs: Additional configuration parameters

    Returns:
        Redis connection pool or None if unavailable
    """
    if redis is None:
        return None

    # Get production configuration
    config = get_production_redis_config()
    config.update(kwargs)

    # Validate URL security
    validate_redis_url_security(redis_url)

    # Create connection pool
    return create_secure_redis_client(redis_url, **config)
