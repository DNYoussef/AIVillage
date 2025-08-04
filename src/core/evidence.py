from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class ConfidenceTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Chunk(BaseModel):
    """Single retrieved text chunk."""

    id: str
    text: str
    score: float = Field(..., ge=0.0, le=1.0)
    source_uri: HttpUrl

    model_config = ConfigDict(frozen=True)


class EvidencePack(BaseModel):
    """Payload describing retrieval evidence."""

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    query: str
    chunks: list[Chunk]
    proto_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_tier: ConfidenceTier | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

    @field_validator("chunks")
    @classmethod
    def _non_empty(cls, v: list[Chunk]) -> list[Chunk]:
        if not v:
            msg = "chunks must not be empty"
            raise ValueError(msg)
        return v

    def to_json(self) -> str:
        """Serialize pack to JSON."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, s: str) -> EvidencePack:
        """Deserialize pack from JSON."""
        return cls.model_validate_json(s)
