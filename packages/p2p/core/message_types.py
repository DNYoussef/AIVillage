# Bridge for message_types module
try:
    from infrastructure.p2p.messages.unified_message import UnifiedMessage
    UnifiedMessage as Message  # Alias for compatibility
except ImportError:
    try:
        from core.p2p.message_types import *
    except ImportError:
        # Define minimal stub
        class UnifiedMessage:
            pass
        Message = UnifiedMessage