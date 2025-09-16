"""
Backward Compatibility Module

Legacy wrappers for existing communication systems.
"""

from .legacy_wrappers import (
    LegacyMessagePassingSystem,
    LegacyChatEngine, 
    LegacyWebSocketHandler,
    LegacyMessage,
    create_legacy_message_system,
    create_legacy_chat_engine,
    create_legacy_websocket_handler,
    MessagePassingSystem,  # Alias
    MessagePassing        # Alias
)

__all__ = [
    "LegacyMessagePassingSystem",
    "LegacyChatEngine",
    "LegacyWebSocketHandler",
    "LegacyMessage",
    "create_legacy_message_system",
    "create_legacy_chat_engine",
    "create_legacy_websocket_handler",
    "MessagePassingSystem",
    "MessagePassing"
]
