"""
MessagePack Serialization Implementation

MessagePack serializer for unified message format.
Optimized for performance and bandwidth efficiency.
"""

import logging
from typing import Any, Dict
from datetime import datetime

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

from ..message_format import UnifiedMessage

logger = logging.getLogger(__name__)


class MessagePackSerializer:
    """MessagePack serialization for unified messages"""
    
    def __init__(self, use_bin_type: bool = True):
        if not MSGPACK_AVAILABLE:
            raise ImportError("msgpack library is required for MessagePack serialization")
        
        self.use_bin_type = use_bin_type
        
    def serialize(self, message: UnifiedMessage) -> bytes:
        """Serialize unified message to MessagePack bytes"""
        try:
            message_dict = message.to_dict()
            return msgpack.packb(
                message_dict,
                default=self._msgpack_serializer,
                use_bin_type=self.use_bin_type
            )
            
        except Exception as e:
            logger.error(f"Error serializing message to MessagePack: {e}")
            raise
    
    def deserialize(self, data: bytes) -> UnifiedMessage:
        """Deserialize MessagePack bytes to unified message"""
        try:
            message_dict = msgpack.unpackb(
                data,
                raw=False,
                strict_map_key=False
            )
            return UnifiedMessage.from_dict(message_dict)
            
        except Exception as e:
            logger.error(f"Error deserializing MessagePack message: {e}")
            raise
    
    def serialize_dict(self, data: Dict[str, Any]) -> bytes:
        """Serialize arbitrary dictionary to MessagePack"""
        try:
            return msgpack.packb(
                data,
                default=self._msgpack_serializer,
                use_bin_type=self.use_bin_type
            )
            
        except Exception as e:
            logger.error(f"Error serializing dict to MessagePack: {e}")
            raise
    
    def deserialize_dict(self, data: bytes) -> Dict[str, Any]:
        """Deserialize MessagePack bytes to dictionary"""
        try:
            return msgpack.unpackb(
                data,
                raw=False,
                strict_map_key=False
            )
            
        except Exception as e:
            logger.error(f"Error deserializing MessagePack to dict: {e}")
            raise
    
    def _msgpack_serializer(self, obj: Any) -> Any:
        """Custom MessagePack serializer for special types"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            raise TypeError(f"Object of type {type(obj)} is not MessagePack serializable")
    
    def get_content_type(self) -> str:
        """Get content type for HTTP headers"""
        return "application/msgpack"
    
    def get_encoding(self) -> str:
        """Get encoding type"""
        return "binary"
    
    def __repr__(self) -> str:
        return f"MessagePackSerializer(use_bin_type={self.use_bin_type})"
