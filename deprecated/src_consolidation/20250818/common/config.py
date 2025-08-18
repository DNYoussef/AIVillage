"""Generic configuration loading utilities."""

# ruff: noqa: I001

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml


def _load_file(path: Path) -> dict[str, Any]:
    """Load configuration data from a JSON or YAML file."""
    with path.open() as f:
        if path.suffix == ".json":
            return json.load(f)
        if path.suffix in {".yml", ".yaml"}:
            return yaml.safe_load(f) or {}
        msg = f"Unsupported config format: {path.suffix}"
        raise ValueError(msg)


def _dump_file(path: Path, data: Mapping[str, Any]) -> None:
    """Persist configuration data to a JSON or YAML file."""
    with path.open("w") as f:
        if path.suffix == ".json":
            json.dump(data, f, indent=2)
        elif path.suffix in {".yml", ".yaml"}:
            yaml.safe_dump(data, f, default_flow_style=False)
        else:
            msg = f"Unsupported config format: {path.suffix}"
            raise ValueError(msg)


def load_config(
    defaults: Mapping[str, Any] | None = None,
    config_path: str | None = None,
    *,
    env_prefix: str | None = None,
    required: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Load configuration with optional file and environment overrides.

    Parameters
    ----------
    defaults:
        Base configuration values. These are overridden by values from
        ``config_path`` and environment variables.
    config_path:
        Optional path to a JSON or YAML file.
    env_prefix:
        Optional prefix for environment variables. For example, with a prefix of
        ``"SERVICE_"`` a config key ``"token"`` would be overridden by the
        ``SERVICE_TOKEN`` environment variable.
    required:
        Keys that must be present after loading. Missing keys raise ``ValueError``.
    """
    config: dict[str, Any] = dict(defaults or {})

    if config_path:
        path = Path(config_path)
        if path.exists():
            config.update(_load_file(path))

    prefix = (env_prefix or "").upper()
    for key in list(config.keys()):
        env_key = f"{prefix}{key}".upper()
        if env_key in os.environ:
            value = os.environ[env_key]
            try:
                config[key] = json.loads(value)
            except json.JSONDecodeError:
                config[key] = value

    if required:
        missing = [key for key in required if not config.get(key)]
        if missing:
            msg = f"Missing required configuration: {', '.join(missing)}"
            raise ValueError(msg)

    return config


def save_config(data: Mapping[str, Any], path: str) -> None:
    """Save configuration mapping to a JSON or YAML file."""
    _dump_file(Path(path), data)


__all__ = ["load_config", "save_config"]
