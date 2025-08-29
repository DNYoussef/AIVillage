"""
Serialization utilities for P2P messages.

Provides standardized serialization/deserialization for all P2P components
with support for multiple formats and automatic format detection.
"""

import json
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Type
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
import logging

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False
    msgpack = None

try:
    import protobuf
    HAS_PROTOBUF = True
except ImportError:
    HAS_PROTOBUF = False
    protobuf = None

logger = logging.getLogger(__name__)


class SerializationFormat(Enum):
    """Supported serialization formats."""
    JSON = "json"
    MSGPACK = "msgpack" 
    PICKLE = "pickle"
    PROTOBUF = "protobuf"


class Serializer(ABC):
    """Abstract base class for serializers."""
    
    @abstractmethod
    def serialize(self, obj: Any) -> bytes:
        """Serialize object to bytes."""
        pass
    
    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to object."""
        pass
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """Get serialization format name."""
        pass


class JSONSerializer(Serializer):
    """JSON serializer with custom encoder for P2P objects."""
    
    class P2PJSONEncoder(json.JSONEncoder):
        """Custom JSON encoder for P2P objects."""
        
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Enum):
                return obj.value
            elif is_dataclass(obj):
                return asdict(obj)
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            return super().default(obj)
    
    def serialize(self, obj: Any) -> bytes:
        """Serialize object to JSON bytes."""
        try:
            json_str = json.dumps(obj, cls=self.P2PJSONEncoder, ensure_ascii=False)
            return json_str.encode('utf-8')
        except (TypeError, ValueError) as e:
            logger.error(f"JSON serialization failed: {e}")
            raise
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize JSON bytes to object."""
        try:
            json_str = data.decode('utf-8')
            return json.loads(json_str)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.error(f"JSON deserialization failed: {e}")
            raise
    
    @property
    def format_name(self) -> str:
        return "json"


class MessagePackSerializer(Serializer):
    """MessagePack serializer for efficient binary serialization."""
    
    def __init__(self):
        if not HAS_MSGPACK:
            raise ImportError("msgpack library not available")
    
    def serialize(self, obj: Any) -> bytes:
        """Serialize object to MessagePack bytes."""
        try:
            # Convert P2P objects to dict representation
            if is_dataclass(obj):
                obj = asdict(obj)
            elif hasattr(obj, 'to_dict'):
                obj = obj.to_dict()
            
            return msgpack.packb(obj, use_bin_type=True)
        except (TypeError, ValueError) as e:
            logger.error(f"MessagePack serialization failed: {e}")
            raise
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize MessagePack bytes to object."""
        try:
            return msgpack.unpackb(data, raw=False, strict_map_key=False)
        except (msgpack.exceptions.ExtraData, 
                msgpack.exceptions.UnpackException,
                msgpack.exceptions.UnpackValueError) as e:
            logger.error(f"MessagePack deserialization failed: {e}")
            raise
    
    @property
    def format_name(self) -> str:
        return "msgpack"


class ProtobufSerializer(Serializer):
    """Protocol Buffers serializer."""
    
    def __init__(self, message_class: Optional[Type] = None):
        if not HAS_PROTOBUF:
            raise ImportError("protobuf library not available")
        self.message_class = message_class
    
    def serialize(self, obj: Any) -> bytes:
        """Serialize protobuf message to bytes."""
        try:
            if hasattr(obj, 'SerializeToString'):
                return obj.SerializeToString()
            else:
                raise TypeError(f"Object {type(obj)} is not a protobuf message")
        except Exception as e:
            logger.error(f"Protobuf serialization failed: {e}")
            raise
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to protobuf message."""
        try:
            if self.message_class is None:
                raise ValueError("No message class specified for deserialization")
            
            message = self.message_class()
            message.ParseFromString(data)
            return message
        except Exception as e:
            logger.error(f"Protobuf deserialization failed: {e}")
            raise
    
    @property
    def format_name(self) -> str:
        return "protobuf"


# Default serializers
_serializers: Dict[SerializationFormat, Serializer] = {
    SerializationFormat.JSON: JSONSerializer()
}

if HAS_MSGPACK:
    _serializers[SerializationFormat.MSGPACK] = MessagePackSerializer()


def get_serializer(format: Union[SerializationFormat, str]) -> Serializer:
    """Get serializer for specified format."""
    if isinstance(format, str):
        format = SerializationFormat(format.lower())
    
    if format not in _serializers:
        raise ValueError(f"Serializer not available for format: {format.value}")
    
    return _serializers[format]


def register_serializer(format: SerializationFormat, serializer: Serializer) -> None:
    """Register a custom serializer for a format."""
    _serializers[format] = serializer


def serialize_message(obj: Any, format: Union[SerializationFormat, str] = SerializationFormat.JSON) -> bytes:
    """Serialize an object using specified format."""
    serializer = get_serializer(format)
    return serializer.serialize(obj)


def deserialize_message(data: bytes, format: Union[SerializationFormat, str] = SerializationFormat.JSON) -> Any:
    """Deserialize bytes using specified format."""
    serializer = get_serializer(format)
    return serializer.deserialize(data)


def detect_format(data: bytes) -> Optional[SerializationFormat]:
    """Attempt to detect serialization format from data."""
    if not data:
        return None
    
    # Try JSON first (most common)
    try:
        json.loads(data.decode('utf-8'))
        return SerializationFormat.JSON
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    
    # Try MessagePack
    if HAS_MSGPACK:
        try:
            msgpack.unpackb(data, raw=False)
            return SerializationFormat.MSGPACK
        except:
            pass
    
    # Default to JSON if detection fails
    return SerializationFormat.JSON


def auto_deserialize(data: bytes) -> Any:
    """Automatically detect format and deserialize."""
    detected_format = detect_format(data)
    if detected_format is None:
        raise ValueError("Could not detect serialization format")
    
    return deserialize_message(data, detected_format)
