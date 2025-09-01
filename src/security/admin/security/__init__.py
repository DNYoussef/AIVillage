"""
Security Admin Security Services Module
Clean aggregator for security service components.
"""

from .security_context import SecurityContextService

# Factory function for easy instantiation
def create_security_context_service(session_service):
    """Create configured security context service"""
    return SecurityContextService(session_service)

__all__ = [
    'SecurityContextService',
    'create_security_context_service'
]