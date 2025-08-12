from __future__ import annotations

"""Safe serialization utilities for agent checkpoints."""

import json
from typing import Any, Dict

import msgpack

CURRENT_VERSION = 1


def serialize_checkpoint(data: Dict[str, Any]) -> bytes:
    """Serialize checkpoint data with version information."""
    payload = {"version": CURRENT_VERSION, "data": data}
    return msgpack.dumps(payload)


def deserialize_checkpoint(payload: bytes) -> Dict[str, Any]:
    """Deserialize checkpoint payload supporting legacy JSON format."""
    try:
        decoded = msgpack.loads(payload, raw=False)
        if isinstance(decoded, dict) and "version" in decoded and "data" in decoded:
            return decoded["data"]
    except Exception:
        pass

    try:
        return json.loads(payload.decode("utf-8"))
    except Exception as e:  # pragma: no cover - unexpected serialization
        raise ValueError("Invalid checkpoint payload") from e
