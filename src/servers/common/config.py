"""Shared configuration utilities for server modules."""

from __future__ import annotations

# isort: skip_file

import os
from functools import lru_cache
from typing import Any

_DEFAULTS: dict[str, Any] = {
    "MAX_FILE_SIZE": 50 * 1024 * 1024,  # 50MB
    "ALLOWED_EXTENSIONS": ".txt,.md,.pdf,.docx,.html",
    "CHUNK_SIZE": 8192,
    "RATE_LIMIT_REQUESTS": 100,
    "RATE_LIMIT_WINDOW": 60,
    "API_KEY": None,
}


def _coerce_value(key: str, value: str) -> Any:
    """Coerce environment variable strings to appropriate types."""
    if key in {
        "MAX_FILE_SIZE",
        "CHUNK_SIZE",
        "RATE_LIMIT_REQUESTS",
        "RATE_LIMIT_WINDOW",
    }:
        return int(value)
    if key == "ALLOWED_EXTENSIONS":
        return {ext.strip() for ext in value.split(",") if ext.strip()}
    return value


@lru_cache(maxsize=1)
def load_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load configuration from environment with optional overrides."""
    config = _DEFAULTS.copy()
    for key in _DEFAULTS:
        env_val = os.getenv(key)
        if env_val is not None:
            config[key] = _coerce_value(key, env_val)
    if overrides:
        for key, value in overrides.items():
            if key in config and isinstance(config[key], set) and isinstance(value, set | list | tuple):
                config[key] = set(value)
            else:
                config[key] = value
    if isinstance(config["ALLOWED_EXTENSIONS"], str):
        config["ALLOWED_EXTENSIONS"] = {ext.strip() for ext in config["ALLOWED_EXTENSIONS"].split(",") if ext.strip()}
    return config
