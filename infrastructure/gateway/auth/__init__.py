"""
Authentication Module for Gateway Layer

Handles JWT tokens, API keys, OAuth2, and user authentication.
"""

from .jwt_handler import (
    APIKeyValidator,
    JWTBearer,
    JWTHandler,
    TokenPayload,
    create_api_key_to_jwt_dependency,
    create_jwt_dependency,
)

__all__ = [
    "JWTHandler",
    "JWTBearer",
    "TokenPayload",
    "APIKeyValidator",
    "create_jwt_dependency",
    "create_api_key_to_jwt_dependency",
]
