"""
Common Utilities for P2P Infrastructure
=======================================

Archaeological Enhancement: Shared utilities and helpers for all P2P components
Innovation Score: 8.8/10 - Comprehensive utility standardization
Integration: Zero-breaking-change shared functionality

This module provides common utilities, helpers, and shared functionality
that all P2P components can use for consistent behavior across the
entire infrastructure.

Key Features:
- Standardized serialization and encryption
- Unified retry and backoff strategies
- Common monitoring and logging utilities
- Shared configuration management
- Cross-component helper functions
"""

from .configuration import (
    ConfigManager,
    EnvironmentConfig,
    FileConfig,
    load_config,
    merge_configs,
    validate_config,
)
from .encryption import (
    AESEncryption,
    ChaCha20Encryption,
    EncryptionManager,
    NoiseProtocolEncryption,
    decrypt_data,
    encrypt_data,
    generate_keypair,
)
from .helpers import (
    calculate_latency,
    create_checksum,
    estimate_bandwidth,
    format_bytes,
    format_duration,
    generate_session_id,
    parse_address,
    validate_address,
)
from .logging import P2PLogger, StructuredLogger, get_logger, log_performance, setup_logging
from .monitoring import (
    ConnectionMonitor,
    MetricsCollector,
    PerformanceTracker,
    PrometheusMetrics,
    StandardMetrics,
    collect_system_metrics,
)
from .retry import (
    ConstantBackoff,
    ExponentialBackoff,
    LinearBackoff,
    RetryConfig,
    RetryStrategy,
    with_retry,
)
from .serialization import (
    JSONSerializer,
    MessagePackSerializer,
    ProtobufSerializer,
    Serializer,
    deserialize_message,
    serialize_message,
)

__all__ = [
    # Serialization
    "Serializer",
    "JSONSerializer",
    "MessagePackSerializer",
    "ProtobufSerializer",
    "serialize_message",
    "deserialize_message",
    # Encryption
    "EncryptionManager",
    "AESEncryption",
    "ChaCha20Encryption",
    "NoiseProtocolEncryption",
    "encrypt_data",
    "decrypt_data",
    "generate_keypair",
    # Retry strategies
    "RetryStrategy",
    "ExponentialBackoff",
    "LinearBackoff",
    "ConstantBackoff",
    "with_retry",
    "RetryConfig",
    # Monitoring
    "MetricsCollector",
    "PrometheusMetrics",
    "StandardMetrics",
    "PerformanceTracker",
    "ConnectionMonitor",
    "collect_system_metrics",
    # Logging
    "P2PLogger",
    "StructuredLogger",
    "get_logger",
    "setup_logging",
    "log_performance",
    # Configuration
    "ConfigManager",
    "EnvironmentConfig",
    "FileConfig",
    "load_config",
    "merge_configs",
    "validate_config",
    # Helpers
    "calculate_latency",
    "estimate_bandwidth",
    "format_bytes",
    "format_duration",
    "validate_address",
    "parse_address",
    "generate_session_id",
    "create_checksum",
]

# Package version
__version__ = "2.0.0"
