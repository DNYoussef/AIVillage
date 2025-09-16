"""
Routing Module

Message routing and discovery for unified messaging system.
"""

from .message_router import MessageRouter, RoutingStrategy, RouteInfo

__all__ = [
    "MessageRouter",
    "RoutingStrategy",
    "RouteInfo"
]
