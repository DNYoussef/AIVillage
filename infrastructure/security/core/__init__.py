"""Core Security Module for AIVillage.

Provides foundational security interfaces, abstractions, and core components
for the modular security architecture.
"""

from .interfaces import (
    IAuthenticationProvider,
    IAuthorizationProvider,
    ICryptographicService,
    ISecurityMiddleware,
    ISecurityConfig,
    SecurityContext,
    SecurityResult,
)
from .config import SecurityConfiguration
from .exceptions import (
    SecurityError,
    AuthenticationError,
    AuthorizationError,
    CryptographicError,
    SecurityConfigurationError,
)

__all__ = [
    # Interfaces
    'IAuthenticationProvider',
    'IAuthorizationProvider',
    'ICryptographicService',
    'ISecurityMiddleware',
    'ISecurityConfig',
    'SecurityContext',
    'SecurityResult',
    # Configuration
    'SecurityConfiguration',
    # Exceptions
    'SecurityError',
    'AuthenticationError',
    'AuthorizationError',
    'CryptographicError',
    'SecurityConfigurationError',
]
