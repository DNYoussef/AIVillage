"""Timing and scheduling constants."""

from dataclasses import dataclass
from enum import Enum
from typing import Final


class TimingUnits(Enum):
    """Time unit enumeration for consistency."""

    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"


@dataclass(frozen=True)
class BatchProcessingDefaults:
    """Default timing values for batch processing."""

    PROCESSING_INTERVAL_SECONDS: Final[float] = 1.0
    RETRY_DELAY_SECONDS: Final[float] = 2.0
    TIMEOUT_SECONDS: Final[int] = 30
    MAX_PROCESSING_TIME_SECONDS: Final[int] = 300
    HEARTBEAT_INTERVAL_SECONDS: Final[float] = 10.0


@dataclass(frozen=True)
class TimingConstants:
    """Core timing constants for task management."""

    # Batch processing timing
    BATCH_PROCESSING_INTERVAL: Final[float] = BatchProcessingDefaults.PROCESSING_INTERVAL_SECONDS
    RETRY_DELAY: Final[float] = BatchProcessingDefaults.RETRY_DELAY_SECONDS
    DEFAULT_TIMEOUT: Final[int] = BatchProcessingDefaults.TIMEOUT_SECONDS
    MAX_PROCESSING_TIME: Final[int] = BatchProcessingDefaults.MAX_PROCESSING_TIME_SECONDS
    HEARTBEAT_INTERVAL: Final[float] = BatchProcessingDefaults.HEARTBEAT_INTERVAL_SECONDS

    # Sleep intervals for different operations
    SHORT_SLEEP: Final[float] = 0.1
    MEDIUM_SLEEP: Final[float] = 0.5
    LONG_SLEEP: Final[float] = 1.0

    # Timeout multipliers for different task types
    SIMPLE_TASK_TIMEOUT_MULTIPLIER: Final[float] = 1.0
    COMPLEX_TASK_TIMEOUT_MULTIPLIER: Final[float] = 3.0
    CRITICAL_TASK_TIMEOUT_MULTIPLIER: Final[float] = 5.0


@dataclass(frozen=True)
class PerformanceTimingDefaults:
    """Timing constants for performance monitoring."""

    PERFORMANCE_SAMPLE_INTERVAL: Final[float] = 5.0
    METRICS_COLLECTION_INTERVAL: Final[float] = 60.0
    HISTORY_CLEANUP_INTERVAL: Final[int] = 3600  # 1 hour
    PERFORMANCE_ALERT_THRESHOLD: Final[float] = 30.0
