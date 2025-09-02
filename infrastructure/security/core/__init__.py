"""Core Security Module for AIVillage.

Provides foundational security interfaces, abstractions, and core components
for the modular security architecture.
"""

from .config import SecurityConfiguration
from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    CryptographicError,
    SecurityConfigurationError,
    SecurityError,
)
from .interfaces import (
    IAuthenticationProvider,
    IAuthorizationProvider,
    ICryptographicService,
    ISecurityConfig,
    ISecurityMiddleware,
    SecurityContext,
    SecurityResult,
)

__all__ = [
    # Interfaces
    "IAuthenticationProvider",
    "IAuthorizationProvider",
    "ICryptographicService",
    "ISecurityMiddleware",
    "ISecurityConfig",
    "SecurityContext",
    "SecurityResult",
    # Configuration
    "SecurityConfiguration",
    # Exceptions
    "SecurityError",
    "AuthenticationError",
    "AuthorizationError",
    "CryptographicError",
    "SecurityConfigurationError",
]
