"""
Security Admin Authentication Module
Clean aggregator for authentication components.
"""

from .session_service import SessionService

# Factory function for easy instantiation
def create_session_service():
    """Create configured session service"""
    return SessionService()

__all__ = [
    'SessionService',
    'create_session_service'
]