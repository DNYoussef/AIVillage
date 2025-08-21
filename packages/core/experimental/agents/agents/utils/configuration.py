"""Utility classes for handling application configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a file."""

    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    neo4j_uri: str = Field(..., env="NEO4J_URI")
    neo4j_user: str = Field(..., env="NEO4J_USER")
    neo4j_password: str = Field(..., env="NEO4J_PASSWORD")

    class Config:
        env_prefix = ""
        env_file = ".env"
        env_file_encoding = "utf-8"

    @classmethod
    def from_file(cls, path: str | Path) -> Settings:
        """Create a :class:`Settings` instance from a YAML or JSON file."""
        file_path = Path(path)
        data: dict[str, Any] = {}

        if file_path.exists():
            with open(file_path, encoding="utf-8") as f:
                if file_path.suffix in {".yaml", ".yml"}:
                    data = yaml.safe_load(f) or {}
                elif file_path.suffix == ".json":
                    data = json.load(f)

        return cls(**data)
