"""
Shared types for Digital Twin and Mini-RAG systems
This module contains common enums and types to avoid circular imports.
"""

from enum import Enum


class DataSource(Enum):
    """Data sources for digital twin learning"""

    CONVERSATIONS = "conversations"  # Text messaging, chat apps
    LOCATION = "location"  # GPS, location history
    PURCHASES = "purchases"  # Shopping behavior
    APP_USAGE = "app_usage"  # Application interaction patterns
    CALENDAR = "calendar"  # Scheduling and time management
    VOICE = "voice"  # Voice interaction patterns
    WEB_BROWSING = "web_browsing"  # Website visits and search
    HEALTH = "health"  # Health app integration
    MUSIC = "music"  # Music and entertainment preferences
    PHOTOS = "photos"  # Image analysis patterns


class PrivacyLevel(Enum):
    """Privacy levels for data handling"""

    PUBLIC = 1  # Can be shared if aggregated
    PERSONAL = 2  # Personal but not sensitive
    SENSITIVE = 3  # Sensitive personal data
    CONFIDENTIAL = 4  # Highly confidential


class LearningStage(Enum):
    """Stages of digital twin learning"""

    INITIALIZATION = "initialization"  # Initial data collection
    ADAPTATION = "adaptation"  # Learning user patterns
    OPTIMIZATION = "optimization"  # Fine-tuning responses
    MAINTENANCE = "maintenance"  # Ongoing improvements
    PRIVACY_RESET = "privacy_reset"  # Data cleanup and reset
