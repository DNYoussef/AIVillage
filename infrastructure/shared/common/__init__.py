"""
Common utilities and shared functionality for AIVillage core systems.
"""

from .flags import (
    FeatureFlag,
    FeatureFlagContext,
    FeatureFlagManager,
    FlagState,
    feature_flag,
    get_flag_manager,
    is_enabled,
    kill_switch,
)
from .http_client import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    DependencyOutageSimulator,
    IdempotencyConfig,
    IdempotencyStore,
    IdempotencyViolationError,
    ResilientHttpClient,
    RetryConfig,
    get_http_client,
)

__all__ = [
    # Feature flags
    "FeatureFlag",
    "FeatureFlagContext",
    "FeatureFlagManager",
    "FlagState",
    "feature_flag",
    "get_flag_manager",
    "is_enabled",
    "kill_switch",
    # HTTP client
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitState",
    "DependencyOutageSimulator",
    "IdempotencyConfig",
    "IdempotencyStore",
    "IdempotencyViolationError",
    "ResilientHttpClient",
    "RetryConfig",
    "get_http_client",
]
