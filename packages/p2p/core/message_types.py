from dataclasses import dataclass
from typing import Any, Dict, Optional

# Fix syntax error - proper import alias
from .unified_message import UnifiedMessage
Message = UnifiedMessage  # Alias for compatibility

@dataclass
class P2PMessage:
    """Base P2P message structure"""
    sender_id: str
    recipient_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: Optional[str] = None
    priority: int = 0

@dataclass
class SystemMessage(P2PMessage):
    """System-level P2P messages"""
    system_level: bool = True

@dataclass  
class UserMessage(P2PMessage):
    """User-level P2P messages"""
    user_level: bool = True