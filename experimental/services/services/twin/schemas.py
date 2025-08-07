from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from datetime import datetime


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10_000)
    user_id: str = Field(..., description="Unique user identifier")
    conversation_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: datetime
    processing_time_ms: float
    calibrated_prob: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Conformally calibrated confidence (0-1) when CALIBRATION_ENABLED=1",
    )

    model_config = ConfigDict(ser_json_exclude_none=True)


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    timestamp: datetime
