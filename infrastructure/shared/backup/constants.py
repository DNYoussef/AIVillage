"""Backup and restore system constants for AIVillage.

This module centralizes all backup-related magic literals to eliminate
connascence of meaning and ensure consistent backup configurations.
"""

from enum import Enum
from typing import Final

# Backup timing and scheduling
BACKUP_STATUS_CHECK_INTERVAL_SECONDS: Final[int] = 5
BACKUP_CLEANUP_INTERVAL_HOURS: Final[int] = 24
BACKUP_RETENTION_DAYS: Final[int] = 30
INCREMENTAL_BACKUP_INTERVAL_HOURS: Final[int] = 6
FULL_BACKUP_INTERVAL_DAYS: Final[int] = 7

# File size and compression
MAX_BACKUP_SIZE_GB: Final[int] = 100
COMPRESSION_LEVEL: Final[int] = 6
CHUNK_SIZE_BYTES: Final[int] = 64 * 1024  # 64KB
BUFFER_SIZE_BYTES: Final[int] = 8 * 1024 * 1024  # 8MB

# Backup file naming
BACKUP_TIMESTAMP_FORMAT: Final[str] = "%Y%m%d_%H%M%S"
BACKUP_FILENAME_PATTERN: Final[str] = "{type}_{tenant}_{timestamp}.tar.gz"
BACKUP_MANIFEST_FILENAME: Final[str] = "backup_manifest.json"
BACKUP_METADATA_FILENAME: Final[str] = "backup_metadata.json"
BACKUP_LOG_FILENAME: Final[str] = "backup.log"

# Directory structure
DEFAULT_BACKUP_DIR: Final[str] = "/var/backups/aivillage"
TEMP_BACKUP_DIR: Final[str] = "/tmp/aivillage_backup"  # noqa: S108
RESTORE_TEMP_DIR: Final[str] = "/tmp/aivillage_restore"  # noqa: S108
QUARANTINE_DIR: Final[str] = "/var/quarantine/aivillage"

# Database backup
DB_BACKUP_TIMEOUT_SECONDS: Final[int] = 300
DB_LOCK_TIMEOUT_SECONDS: Final[int] = 30
DB_VACUUM_BEFORE_BACKUP: Final[bool] = True
DB_INTEGRITY_CHECK: Final[bool] = True

# Encryption and security
BACKUP_ENCRYPTION_ENABLED: Final[bool] = True
BACKUP_ENCRYPTION_ALGORITHM: Final[str] = "AES-256-GCM"
BACKUP_KEY_DERIVATION_ITERATIONS: Final[int] = 100_000
BACKUP_SALT_LENGTH: Final[int] = 32
BACKUP_SIGNATURE_ALGORITHM: Final[str] = "HMAC-SHA256"

# Validation and verification
BACKUP_CHECKSUM_ALGORITHM: Final[str] = "sha256"
BACKUP_VERIFICATION_ENABLED: Final[bool] = True
RESTORE_VERIFICATION_ENABLED: Final[bool] = True
CORRUPTION_CHECK_ENABLED: Final[bool] = True

# Performance tuning
PARALLEL_BACKUP_WORKERS: Final[int] = 4
PARALLEL_RESTORE_WORKERS: Final[int] = 2
IO_QUEUE_SIZE: Final[int] = 100
MEMORY_LIMIT_MB: Final[int] = 512

# Progress reporting
PROGRESS_REPORT_INTERVAL_SECONDS: Final[int] = 10
PROGRESS_GRANULARITY_PERCENT: Final[float] = 1.0
LOG_LEVEL_BACKUP: Final[str] = "INFO"
LOG_LEVEL_RESTORE: Final[str] = "INFO"


class BackupPriority(Enum):
    """Backup priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class BackupComponentType(Enum):
    """Types of components that can be backed up."""

    RBAC_DATA = "rbac_data"
    AGENT_CONFIGS = "agent_configs"
    RAG_COLLECTIONS = "rag_collections"
    P2P_CONFIGS = "p2p_configs"
    DIGITAL_TWIN_DATA = "digital_twin_data"
    SYSTEM_LOGS = "system_logs"
    USER_DATA = "user_data"
    KNOWLEDGE_GRAPHS = "knowledge_graphs"
    MODEL_WEIGHTS = "model_weights"
    CACHED_DATA = "cached_data"


class CompressionLevel(Enum):
    """Compression levels for backups."""

    NONE = 0
    FAST = 1
    BALANCED = 6
    BEST = 9


class RestoreStrategy(Enum):
    """Restore strategies."""

    FULL_REPLACE = "full_replace"
    INCREMENTAL_MERGE = "incremental_merge"
    SELECTIVE_RESTORE = "selective_restore"
    POINT_IN_TIME = "point_in_time"


# Error handling
MAX_RETRY_ATTEMPTS: Final[int] = 3
RETRY_DELAY_SECONDS: Final[int] = 5
EXPONENTIAL_BACKOFF_MULTIPLIER: Final[float] = 2.0
MAX_RETRY_DELAY_SECONDS: Final[int] = 300

# Backup exclusion patterns
DEFAULT_EXCLUDE_PATTERNS: Final[tuple[str, ...]] = (
    "*.tmp",
    "*.temp",
    "*.cache",
    "*.pid",
    "*.lock",
    "__pycache__/*",
    ".git/*",
    "node_modules/*",
    "*.log",
)

# Tenant-specific settings
TENANT_BACKUP_QUOTA_GB: Final[int] = 10
TENANT_BACKUP_RETENTION_DAYS: Final[int] = 90
CROSS_TENANT_RESTORE_ALLOWED: Final[bool] = False

# Monitoring and alerting
BACKUP_FAILURE_ALERT_THRESHOLD: Final[int] = 2
BACKUP_SIZE_ALERT_THRESHOLD_GB: Final[int] = 50
BACKUP_DURATION_ALERT_THRESHOLD_HOURS: Final[int] = 4
DISK_SPACE_WARNING_THRESHOLD_PERCENT: Final[int] = 80
DISK_SPACE_CRITICAL_THRESHOLD_PERCENT: Final[int] = 95


class BackupMessages:
    """Standardized backup-related messages."""

    BACKUP_STARTED: Final[str] = "Backup operation started for {backup_type}"
    BACKUP_COMPLETED: Final[str] = "Backup completed successfully: {backup_file}"
    BACKUP_FAILED: Final[str] = "Backup operation failed: {error}"
    RESTORE_STARTED: Final[str] = "Restore operation started from {backup_file}"
    RESTORE_COMPLETED: Final[str] = "Restore completed successfully"
    RESTORE_FAILED: Final[str] = "Restore operation failed: {error}"
    VERIFICATION_PASSED: Final[str] = "Backup verification passed"
    VERIFICATION_FAILED: Final[str] = "Backup verification failed: {error}"
    CORRUPTION_DETECTED: Final[str] = "Backup corruption detected: {details}"
    DISK_SPACE_LOW: Final[str] = "Low disk space: {available_gb}GB remaining"
    CLEANUP_COMPLETED: Final[str] = "Backup cleanup completed: {files_removed} files removed"


# Backup metadata fields
class BackupMetadataFields:
    """Standard metadata fields for backups."""

    VERSION: Final[str] = "version"
    TIMESTAMP: Final[str] = "timestamp"
    TYPE: Final[str] = "type"
    SIZE_BYTES: Final[str] = "size_bytes"
    CHECKSUM: Final[str] = "checksum"
    TENANT_ID: Final[str] = "tenant_id"
    COMPONENTS: Final[str] = "components"
    ENCRYPTION_ENABLED: Final[str] = "encryption_enabled"
    COMPRESSION_LEVEL: Final[str] = "compression_level"
    CREATOR: Final[str] = "creator"
    DESCRIPTION: Final[str] = "description"
    PARENT_BACKUP: Final[str] = "parent_backup"
    RESTORATION_POINTS: Final[str] = "restoration_points"


# File format versions
BACKUP_FORMAT_VERSION: Final[str] = "2.0"
MANIFEST_FORMAT_VERSION: Final[str] = "1.0"
METADATA_FORMAT_VERSION: Final[str] = "1.0"

# Resource monitoring
MEMORY_USAGE_CHECK_INTERVAL_SECONDS: Final[int] = 30
CPU_USAGE_THRESHOLD_PERCENT: Final[int] = 80
NETWORK_BANDWIDTH_LIMIT_MBPS: Final[int] = 100
CONCURRENT_BACKUP_LIMIT: Final[int] = 2
CONCURRENT_RESTORE_LIMIT: Final[int] = 1
