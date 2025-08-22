"""Secure serialization utilities to replace unsafe pickle usage."""

import json
import logging
from typing import Any, Union

logger = logging.getLogger(__name__)

SerializableType = Union[dict[str, Any], list[Any], str, int, float, bool, None]


class SecureSerializationError(ValueError):
    """Raised when secure serialization operations fail."""


class SecureSerializer:
    """Secure JSON-based serializer to replace pickle usage."""

    @staticmethod
    def dumps(obj: SerializableType) -> bytes:
        """Securely serialize object to bytes using JSON."""
        try:
            json_str = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
            return json_str.encode("utf-8")
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize object: {e}")
            raise ValueError(f"Object not JSON serializable: {type(obj)}") from e

    @staticmethod
    def loads(data: bytes) -> SerializableType:
        """Securely deserialize bytes to object using JSON."""
        try:
            json_str = data.decode("utf-8")
            return json.loads(json_str)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.error(f"Failed to deserialize data: {e}")
            raise ValueError("Invalid serialized data format") from e

    @staticmethod
    def serialize_performance_record(record) -> dict[str, Any]:
        """Convert PerformanceRecord to JSON-serializable dict."""
        return {
            "timestamp": record.timestamp,
            "task_type": record.task_type,
            "success": record.success,
            "execution_time_ms": record.execution_time_ms,
            "accuracy": record.accuracy,
            "confidence": record.confidence,
            "resource_usage": record.resource_usage,
            "context": record.context,
        }

    @staticmethod
    def deserialize_performance_record(data: dict[str, Any]):
        """Convert dict back to PerformanceRecord-like object."""
        # Import here to avoid circular imports
        from production.agent_forge.evolution.base import PerformanceRecord

        return PerformanceRecord(
            timestamp=data["timestamp"],
            task_type=data["task_type"],
            success=data["success"],
            execution_time_ms=data["execution_time_ms"],
            accuracy=data.get("accuracy"),
            confidence=data.get("confidence"),
            resource_usage=data.get("resource_usage"),
            context=data.get("context"),
        )


# Backward-compatible interface
def secure_dumps(obj: SerializableType) -> bytes:
    """Secure replacement for pickle.dumps()."""
    return SecureSerializer.dumps(obj)


def secure_loads(data: bytes) -> SerializableType:
    """Secure replacement for the insecure pickle deserializer."""
    return SecureSerializer.loads(data)
