# agents/langroid/utils/configuration.py

"""Utility classes for handling application configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a file."""

    openai_api_key: str = ""
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    class Config:
        env_prefix = ""
        env_file = ".env"
        env_file_encoding = "utf-8"

    @classmethod
    def from_file(cls, path: str | Path) -> "Settings":
        """Create a :class:`Settings` instance from a YAML or JSON file."""

        file_path = Path(path)
        data: Dict[str, Any] = {}

        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.suffix in {".yaml", ".yml"}:
                    data = yaml.safe_load(f) or {}
                elif file_path.suffix == ".json":
                    data = json.load(f)

        return cls(**data)
