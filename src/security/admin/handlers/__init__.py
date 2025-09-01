"""
Security Admin Handlers Module
Clean aggregator for route handler components.
"""

from .auth_handlers import AuthHandlers
from .admin_handlers import AdminHandlers

# Factory function for easy instantiation
def create_handler_registry(session_service, admin_boundary):
    """Create complete handler registry"""
    return {
        'auth': AuthHandlers(session_service),
        'admin': AdminHandlers(session_service, admin_boundary)
    }

__all__ = [
    'AuthHandlers',
    'AdminHandlers',
    'create_handler_registry'
]