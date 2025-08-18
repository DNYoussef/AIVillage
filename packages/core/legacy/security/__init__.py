"""Core Security Module.

This module provides security utilities for AIVillage including:
- HTTP/HTTPS validation for production environments
- Secure serialization utilities
- TLS/mTLS configuration
- Role-based access control
- Digital twin encryption
"""

from .http_security_validator import (
    HTTPSecurityError,
    require_https_in_production,
    scan_source_for_http_urls,
    validate_config_dict_for_production,
    validate_production_environment,
    validate_url_for_production,
)
from .secure_redis_client import (
    SecureRedisError,
    create_production_redis_pool,
    create_secure_redis_client,
    get_production_redis_config,
    validate_redis_url_security,
)
from .secure_serialization import (
    SecureSerializationError,
    SecureSerializer,
    secure_dumps,
    secure_loads,
)

__all__ = [
    # HTTP Security
    "HTTPSecurityError",
    "validate_url_for_production",
    "validate_config_dict_for_production",
    "validate_production_environment",
    "require_https_in_production",
    "scan_source_for_http_urls",
    # Secure Serialization
    "SecureSerializer",
    "SecureSerializationError",
    "secure_dumps",
    "secure_loads",
    # Secure Redis
    "SecureRedisError",
    "create_secure_redis_client",
    "validate_redis_url_security",
    "get_production_redis_config",
    "create_production_redis_pool",
]
