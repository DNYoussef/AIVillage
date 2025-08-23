#!/usr/bin/env python3
"""
System Constants Module - Eliminates Configuration Magic Literals

This module consolidates all system configuration constants to eliminate
5,179 configuration-related magic literals identified in the analysis.
Provides typed, documented alternatives to scattered magic numbers.
"""

from enum import Enum
from typing import Final


class MobileProfile(Enum):
    """Mobile device performance profiles."""

    LOW_RAM = "low_ram"
    BATTERY_SAVE = "battery_save"
    THERMAL_THROTTLE = "thermal_throttle"
    PERFORMANCE = "performance"
    ADAPTIVE = "adaptive"


class TransportType(Enum):
    """Network transport type identifiers."""

    BITCHAT = "bitchat"
    BETANET = "betanet"
    QUIC = "quic"
    WEBSOCKET = "websocket"
    HTTP = "http"
    HTTPS = "https"


class MessageStatus(Enum):
    """Message processing status values."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class OperationMode(Enum):
    """System operation modes."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    MAINTENANCE = "maintenance"


# System Performance Limits
class SystemLimits:
    """Core system performance and resource limits."""

    # Timeout Values
    DEFAULT_TIMEOUT: Final[float] = 30.0
    CONNECTION_TIMEOUT: Final[float] = 10.0
    REQUEST_TIMEOUT: Final[float] = 60.0
    BACKGROUND_TASK_TIMEOUT: Final[float] = 300.0

    # Retry Logic
    DEFAULT_MAX_RETRIES: Final[int] = 5
    EXPONENTIAL_BACKOFF_BASE: Final[float] = 2.0
    MAX_BACKOFF_DELAY: Final[float] = 60.0
    JITTER_FACTOR: Final[float] = 0.1

    # Connection Pooling
    MAX_CONNECTIONS: Final[int] = 100
    CONNECTION_POOL_SIZE: Final[int] = 20
    CONNECTION_IDLE_TIMEOUT: Final[int] = 300

    # Message Processing
    MAX_MESSAGE_SIZE: Final[int] = 1048576  # 1MB
    MESSAGE_BATCH_SIZE: Final[int] = 50
    MESSAGE_TTL: Final[int] = 300  # 5 minutes

    # Memory and Performance
    DEFAULT_CHUNK_SIZE: Final[int] = 1024
    MAX_MEMORY_USAGE: Final[int] = 1073741824  # 1GB
    GARBAGE_COLLECTION_THRESHOLD: Final[int] = 1000


class CacheSettings:
    """Caching system configuration constants."""

    # Cache TTL Values
    SHORT_TTL: Final[int] = 300  # 5 minutes
    MEDIUM_TTL: Final[int] = 3600  # 1 hour
    LONG_TTL: Final[int] = 86400  # 24 hours

    # Cache Sizes
    DEFAULT_CACHE_SIZE: Final[int] = 1000
    LARGE_CACHE_SIZE: Final[int] = 10000
    MEMORY_CACHE_SIZE: Final[int] = 500

    # Cache Policies
    EVICTION_POLICY: Final[str] = "lru"
    DEFAULT_CACHE_ENABLED: Final[bool] = True


class ProcessingLimits:
    """Processing and computation limits."""

    # Query Processing
    MAX_QUERY_LENGTH: Final[int] = 1000
    MAX_RESULTS_PER_QUERY: Final[int] = 100
    DEFAULT_RESULT_LIMIT: Final[int] = 10

    # Graph Traversal
    MAX_GRAPH_DEPTH: Final[int] = 5
    DEFAULT_GRAPH_DEPTH: Final[int] = 3
    MAX_GRAPH_NODES: Final[int] = 10000

    # Vector Operations
    DEFAULT_VECTOR_DIMENSIONS: Final[int] = 384
    MAX_VECTOR_DIMENSIONS: Final[int] = 4096
    SIMILARITY_THRESHOLD: Final[float] = 0.1

    # Batch Processing
    DEFAULT_BATCH_SIZE: Final[int] = 32
    MAX_BATCH_SIZE: Final[int] = 1000
    MIN_BATCH_SIZE: Final[int] = 1


# File System Constants
class FileSystemLimits:
    """File system operation limits and paths."""

    # File Sizes
    MAX_FILE_SIZE: Final[int] = 104857600  # 100MB
    MAX_LOG_FILE_SIZE: Final[int] = 10485760  # 10MB
    CHUNK_READ_SIZE: Final[int] = 8192  # 8KB

    # Directory Structure
    DEFAULT_DATA_DIR: Final[str] = "data"
    DEFAULT_LOGS_DIR: Final[str] = "logs"
    DEFAULT_CACHE_DIR: Final[str] = "cache"
    DEFAULT_TEMP_DIR: Final[str] = "tmp"

    # File Extensions
    LOG_FILE_EXTENSION: Final[str] = ".log"
    DATA_FILE_EXTENSION: Final[str] = ".db"
    CONFIG_FILE_EXTENSION: Final[str] = ".yaml"
    BACKUP_FILE_EXTENSION: Final[str] = ".bak"


class NetworkConstants:
    """Network communication constants."""

    # Buffer Sizes
    DEFAULT_BUFFER_SIZE: Final[int] = 4096
    MAX_BUFFER_SIZE: Final[int] = 65536
    RECEIVE_BUFFER_SIZE: Final[int] = 8192

    # Protocol Settings
    DEFAULT_PORT: Final[int] = 8080
    HTTP_OK: Final[int] = 200
    HTTP_BAD_REQUEST: Final[int] = 400
    HTTP_UNAUTHORIZED: Final[int] = 401
    HTTP_FORBIDDEN: Final[int] = 403
    HTTP_NOT_FOUND: Final[int] = 404
    HTTP_INTERNAL_ERROR: Final[int] = 500

    # Routing and Mesh
    MAX_HOP_COUNT: Final[int] = 7
    DEFAULT_MAX_HOPS: Final[int] = 3
    ROUTE_DISCOVERY_TTL: Final[int] = 60
    KEEPALIVE_INTERVAL: Final[int] = 30


# Quality Assurance Constants
class QualityThresholds:
    """Code quality and performance thresholds."""

    # Confidence Scores
    MIN_CONFIDENCE: Final[float] = 0.1
    MEDIUM_CONFIDENCE: Final[float] = 0.6
    HIGH_CONFIDENCE: Final[float] = 0.8

    # Performance Metrics
    MAX_RESPONSE_TIME: Final[float] = 1.0  # 1 second
    MAX_PROCESSING_TIME: Final[float] = 5.0  # 5 seconds
    MIN_THROUGHPUT: Final[int] = 100  # requests/second

    # Resource Usage
    MAX_CPU_USAGE: Final[float] = 0.8  # 80%
    MAX_MEMORY_USAGE: Final[float] = 0.9  # 90%
    MIN_DISK_SPACE: Final[int] = 1073741824  # 1GB


# Default Configurations
DEFAULT_SYSTEM_CONFIG = {
    "timeout": SystemLimits.DEFAULT_TIMEOUT,
    "max_retries": SystemLimits.DEFAULT_MAX_RETRIES,
    "cache_enabled": CacheSettings.DEFAULT_CACHE_ENABLED,
    "cache_size": CacheSettings.DEFAULT_CACHE_SIZE,
    "batch_size": ProcessingLimits.DEFAULT_BATCH_SIZE,
    "vector_dimensions": ProcessingLimits.DEFAULT_VECTOR_DIMENSIONS,
    "graph_depth": ProcessingLimits.DEFAULT_GRAPH_DEPTH,
}


def get_mobile_profile_settings(profile: MobileProfile) -> dict:
    """Get optimized settings for mobile profile."""
    profiles = {
        MobileProfile.LOW_RAM: {
            "max_cache_size": 100,
            "batch_size": 10,
            "vector_dimensions": 128,
            "max_connections": 5,
        },
        MobileProfile.BATTERY_SAVE: {
            "connection_timeout": 60.0,
            "keepalive_interval": 120,
            "background_sync": False,
            "aggressive_caching": True,
        },
        MobileProfile.PERFORMANCE: {
            "max_connections": 50,
            "batch_size": 100,
            "parallel_processing": True,
            "prefetch_enabled": True,
        },
        MobileProfile.THERMAL_THROTTLE: {
            "max_cpu_usage": 0.5,
            "processing_delay": 0.1,
            "reduced_quality": True,
        },
    }
    return profiles.get(profile, DEFAULT_SYSTEM_CONFIG)


def get_environment_config(mode: OperationMode) -> dict:
    """Get configuration for operation mode."""
    configs = {
        OperationMode.DEVELOPMENT: {
            "debug_logging": True,
            "cache_enabled": False,
            "strict_validation": False,
        },
        OperationMode.PRODUCTION: {
            "debug_logging": False,
            "cache_enabled": True,
            "strict_validation": True,
            "performance_monitoring": True,
        },
    }
    return configs.get(mode, {})


# Usage Examples for Migration:
"""
# BEFORE (Magic Literals - VIOLATIONS):
if i % 10 == 0:                                 # Magic modulo
if timeout > 30.0:                              # Magic timeout
if mobile_profile == "low_ram":                 # Magic string
if response.status == "completed":              # Magic status

# AFTER (Using Constants - CLEAN):
if i % ProcessingLimits.DEFAULT_RESULT_LIMIT == 0:  # Clear intent
if timeout > SystemLimits.DEFAULT_TIMEOUT:          # Documented limit
if profile == MobileProfile.LOW_RAM:                # Type-safe enum
if response.status == MessageStatus.COMPLETED:      # Explicit status
"""
