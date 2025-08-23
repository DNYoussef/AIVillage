"""
Shared type definitions for mobile components.
Common types used across mobile integration components.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class DataSource(Enum):
    """Data sources for mobile components."""

    CONVERSATION = "conversation"
    PURCHASE = "purchase"
    LOCATION = "location"
    APP_USAGE = "app_usage"
    CALENDAR = "calendar"
    VOICE = "voice"


class PrivacyLevel(Enum):
    """Privacy levels for data handling."""

    PUBLIC = "public"
    PRIVATE = "private"
    PERSONAL = "personal"
    SENSITIVE = "sensitive"


@dataclass
class MobileConfig:
    """Mobile component configuration."""

    device_id: str
    enable_digital_twin: bool = True
    enable_rag: bool = True
    privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE
    battery_optimization: bool = True
    data_retention_days: int = 30
