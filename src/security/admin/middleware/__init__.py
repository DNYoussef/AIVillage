"""
Security Admin Middleware Module
Clean aggregator for middleware components.
"""

from .security_middleware import (
    SecurityHeadersMiddleware,
    LocalhostOnlyMiddleware, 
    AuditLoggingMiddleware
)

# Middleware factory for easy configuration
def create_security_middleware_stack():
    """Create a complete security middleware stack"""
    return [
        SecurityHeadersMiddleware,
        AuditLoggingMiddleware,
        LocalhostOnlyMiddleware
    ]

__all__ = [
    'SecurityHeadersMiddleware',
    'LocalhostOnlyMiddleware',
    'AuditLoggingMiddleware',
    'create_security_middleware_stack'
]