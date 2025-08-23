"""Common constants used across AIVillage packages.

This module centralizes shared magic literals to eliminate connascence
of meaning and provide a single source of truth for common values.
"""

from enum import Enum
from typing import Final

# HTTP and networking
HTTP_TIMEOUT_SECONDS: Final[int] = 30
HTTP_MAX_RETRIES: Final[int] = 3
HTTP_RETRY_DELAY_SECONDS: Final[int] = 1
DEFAULT_PORT: Final[int] = 8000
MAX_CONNECTIONS: Final[int] = 100
CONNECTION_POOL_SIZE: Final[int] = 20
KEEP_ALIVE_TIMEOUT_SECONDS: Final[int] = 75

# Logging configuration
LOG_MAX_BYTES: Final[int] = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT: Final[int] = 5
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

# File handling
DEFAULT_ENCODING: Final[str] = "utf-8"
TEMP_FILE_PREFIX: Final[str] = "aivillage_"
MAX_FILENAME_LENGTH: Final[int] = 255
BUFFER_SIZE: Final[int] = 8192

# Time constants
SECONDS_PER_MINUTE: Final[int] = 60
MINUTES_PER_HOUR: Final[int] = 60
HOURS_PER_DAY: Final[int] = 24
DAYS_PER_WEEK: Final[int] = 7
MILLISECONDS_PER_SECOND: Final[int] = 1000

# Memory and performance
DEFAULT_BATCH_SIZE: Final[int] = 100
MAX_QUEUE_SIZE: Final[int] = 1000
WORKER_POOL_SIZE: Final[int] = 4
CACHE_TTL_SECONDS: Final[int] = 3600  # 1 hour
CACHE_MAX_SIZE: Final[int] = 1000

# Database configuration
DB_CONNECTION_TIMEOUT_SECONDS: Final[int] = 10
DB_QUERY_TIMEOUT_SECONDS: Final[int] = 30
DB_POOL_SIZE: Final[int] = 10
DB_MAX_OVERFLOW: Final[int] = 20
DB_POOL_RECYCLE_SECONDS: Final[int] = 3600

# API configuration
API_VERSION: Final[str] = "v1"
API_PREFIX: Final[str] = "/api/v1"
MAX_PAGE_SIZE: Final[int] = 1000
DEFAULT_PAGE_SIZE: Final[int] = 50
MAX_REQUEST_SIZE_MB: Final[int] = 10

# Validation constants
MIN_PASSWORD_LENGTH: Final[int] = 8
MAX_PASSWORD_LENGTH: Final[int] = 128
MAX_USERNAME_LENGTH: Final[int] = 50
MAX_EMAIL_LENGTH: Final[int] = 254
MAX_DESCRIPTION_LENGTH: Final[int] = 1000

# Resource limits
MAX_CPU_PERCENT: Final[int] = 80
MAX_MEMORY_PERCENT: Final[int] = 80
MAX_DISK_PERCENT: Final[int] = 90
MAX_OPEN_FILES: Final[int] = 1000


class EnvironmentType(Enum):
    """Environment types for deployment."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class HttpMethod(Enum):
    """HTTP methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class ContentType(Enum):
    """Common content types."""

    JSON = "application/json"
    XML = "application/xml"
    HTML = "text/html"
    PLAIN_TEXT = "text/plain"
    CSV = "text/csv"
    BINARY = "application/octet-stream"
    FORM_DATA = "multipart/form-data"
    URL_ENCODED = "application/x-www-form-urlencoded"


class CacheStrategy(Enum):
    """Cache strategies."""

    NO_CACHE = "no_cache"
    CACHE_FIRST = "cache_first"
    NETWORK_FIRST = "network_first"
    CACHE_ONLY = "cache_only"
    NETWORK_ONLY = "network_only"


# Error messages
class CommonMessages:
    """Common standardized messages."""

    OPERATION_SUCCESSFUL: Final[str] = "Operation completed successfully"
    OPERATION_FAILED: Final[str] = "Operation failed: {error}"
    VALIDATION_FAILED: Final[str] = "Validation failed: {details}"
    RESOURCE_NOT_FOUND: Final[str] = "Resource not found: {resource_id}"
    RESOURCE_ALREADY_EXISTS: Final[str] = "Resource already exists: {resource_id}"
    INTERNAL_ERROR: Final[str] = "Internal server error occurred"
    BAD_REQUEST: Final[str] = "Bad request: {details}"
    UNAUTHORIZED: Final[str] = "Unauthorized access attempt"
    FORBIDDEN: Final[str] = "Access forbidden"
    TIMEOUT_ERROR: Final[str] = "Operation timed out after {seconds} seconds"
    CONNECTION_ERROR: Final[str] = "Connection error: {details}"


# Configuration file names
CONFIG_FILENAMES: Final[tuple[str, ...]] = (
    "config.yaml",
    "config.yml",
    "settings.yaml",
    "settings.yml",
    ".env",
    "aivillage.conf",
)

# Default directories
DEFAULT_CONFIG_DIR: Final[str] = "config"
DEFAULT_DATA_DIR: Final[str] = "data"
DEFAULT_LOGS_DIR: Final[str] = "logs"
DEFAULT_CACHE_DIR: Final[str] = "cache"
DEFAULT_TEMP_DIR: Final[str] = "tmp"

# Feature flags
FEATURE_FLAGS: Final[dict[str, bool]] = {
    "enable_caching": True,
    "enable_compression": True,
    "enable_encryption": True,
    "enable_monitoring": True,
    "enable_rate_limiting": True,
    "enable_audit_logging": True,
    "enable_health_checks": True,
}

# System metadata
SYSTEM_NAME: Final[str] = "AIVillage"
SYSTEM_VERSION: Final[str] = "2.0.0"
MINIMUM_PYTHON_VERSION: Final[str] = "3.9"
SUPPORTED_PLATFORMS: Final[tuple[str, ...]] = ("linux", "darwin", "win32")

# Monitoring and health check
HEALTH_CHECK_INTERVAL_SECONDS: Final[int] = 30
HEALTH_CHECK_TIMEOUT_SECONDS: Final[int] = 5
METRICS_COLLECTION_INTERVAL_SECONDS: Final[int] = 60
ALERTING_THRESHOLD_CHECKS: Final[int] = 3

# Data formats and serialization
DATETIME_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S.%fZ"
DATE_FORMAT: Final[str] = "%Y-%m-%d"
TIME_FORMAT: Final[str] = "%H:%M:%S"
JSON_INDENT: Final[int] = 2
CSV_DELIMITER: Final[str] = ","
CSV_QUOTE_CHAR: Final[str] = '"'

# Cryptographic constants
RANDOM_SEED_LENGTH: Final[int] = 32
NONCE_LENGTH: Final[int] = 16
KEY_DERIVATION_ITERATIONS: Final[int] = 10000
HASH_ALGORITHM: Final[str] = "sha256"

# Performance monitoring
SLOW_QUERY_THRESHOLD_SECONDS: Final[float] = 1.0
SLOW_REQUEST_THRESHOLD_SECONDS: Final[float] = 5.0
HIGH_MEMORY_THRESHOLD_MB: Final[int] = 1000
HIGH_CPU_THRESHOLD_PERCENT: Final[float] = 75.0
