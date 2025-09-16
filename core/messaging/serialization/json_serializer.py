"""
JSON Serialization Implementation

JSON serializer for unified message format.
Optimized for human-readability and API compatibility.
"""

import json
import logging
from typing import Any, Dict
from datetime import datetime

from ..message_format import UnifiedMessage

logger = logging.getLogger(__name__)


class JsonSerializer:
    """JSON serialization for unified messages"""
    
    def __init__(self, pretty: bool = False):
        self.pretty = pretty
        self.indent = 2 if pretty else None
        
    def serialize(self, message: UnifiedMessage) -> bytes:
        """Serialize unified message to JSON bytes"""
        try:
            message_dict = message.to_dict()
            json_str = json.dumps(
                message_dict,
                indent=self.indent,
                ensure_ascii=False,
                default=self._json_serializer
            )
            return json_str.encode('utf-8')
            
        except Exception as e:
            logger.error(f"Error serializing message to JSON: {e}")
            raise
    
    def deserialize(self, data: bytes) -> UnifiedMessage:
        """Deserialize JSON bytes to unified message"""
        try:
            json_str = data.decode('utf-8')
            message_dict = json.loads(json_str)
            return UnifiedMessage.from_dict(message_dict)
            
        except Exception as e:
            logger.error(f"Error deserializing JSON message: {e}")
            raise
    
    def serialize_dict(self, data: Dict[str, Any]) -> bytes:
        """Serialize arbitrary dictionary to JSON"""
        try:
            json_str = json.dumps(
                data,
                indent=self.indent,
                ensure_ascii=False,
                default=self._json_serializer
            )
            return json_str.encode('utf-8')
            
        except Exception as e:
            logger.error(f"Error serializing dict to JSON: {e}")
            raise
    
    def deserialize_dict(self, data: bytes) -> Dict[str, Any]:
        """Deserialize JSON bytes to dictionary"""
        try:
            json_str = data.decode('utf-8')
            return json.loads(json_str)
            
        except Exception as e:
            logger.error(f"Error deserializing JSON to dict: {e}")
            raise
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for special types"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def get_content_type(self) -> str:
        """Get content type for HTTP headers"""
        return "application/json"
    
    def get_encoding(self) -> str:
        """Get encoding type"""
        return "utf-8"
    
    def __repr__(self) -> str:
        return f"JsonSerializer(pretty={self.pretty})"
