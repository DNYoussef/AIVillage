from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl, validator


class Chunk(BaseModel):
    """Single retrieved text chunk."""

    id: str
    text: str
    score: float = Field(..., ge=0.0, le=1.0)
    source_uri: HttpUrl

    class Config:
        frozen = True


class EvidencePack(BaseModel):
    """Payload describing retrieval evidence."""

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    query: str
    chunks: List[Chunk]
    proto_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    meta: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True

    @validator("chunks")
    def _non_empty(cls, v: List[Chunk]) -> List[Chunk]:
        if not v:
            raise ValueError("chunks must not be empty")
        return v

    def to_json(self) -> str:
        """Serialize pack to JSON."""
        return self.json()

    @classmethod
    def from_json(cls, s: str) -> "EvidencePack":
        """Deserialize pack from JSON."""
        return cls.parse_raw(s)
