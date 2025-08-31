"""
Fog Computing Infrastructure Constants
Centralized constants to reduce magic literals in fog computing services.
"""

from enum import Enum
from typing import Dict, Any

# Privacy Levels
class PrivacyLevel(Enum):
    """Privacy classification for fog computing operations."""
    PUBLIC = "public"
    PRIVATE = "private"
    CONFIDENTIAL = "confidential" 
    SECRET = "secret"

# Service Types
class ServiceType(Enum):
    """Types of fog computing services."""
    ORCHESTRATION = "orchestration"
    HARVESTING = "harvesting"
    MARKETPLACE = "marketplace"
    PRIVACY = "privacy"
    TOKENOMICS = "tokenomics"
    STATS = "stats"

# Network and Communication Constants
DEFAULT_FOG_PORT = 8080
CIRCUIT_POOL_SIZES = {
    PrivacyLevel.PUBLIC: 2,
    PrivacyLevel.PRIVATE: 3,
    PrivacyLevel.CONFIDENTIAL: 2,
    PrivacyLevel.SECRET: 2
}

# Timing Constants (in seconds)
CIRCUIT_ROTATION_INTERVAL = 300  # 5 minutes
CIRCUIT_LIFETIME = 3600  # 1 hour
HEALTH_CHECK_INTERVAL = 30  # 30 seconds
STATS_UPDATE_INTERVAL = 60  # 1 minute
TASK_CLEANUP_INTERVAL = 300  # 5 minutes
SYSTEM_STARTUP_TIMEOUT = 30  # 30 seconds

# Performance Constants
MAX_CONCURRENT_TASKS = 100
DEFAULT_TASK_TIMEOUT = 5000  # 5 seconds in ms
CIRCUIT_LOAD_THRESHOLD = 1000  # bytes
DEVICE_REGISTRATION_TIMEOUT = 2000  # 2 seconds in ms

# Resource Limits
MIN_BATTERY_PERCENT = 20
MAX_THERMAL_TEMP = 45.0
DEFAULT_TOKEN_RATE_PER_HOUR = 10
MAX_HARVEST_DEVICES = 50

# Privacy Circuit Configuration
PRIVACY_HOPS = {
    PrivacyLevel.PUBLIC: 0,
    PrivacyLevel.PRIVATE: 3,
    PrivacyLevel.CONFIDENTIAL: 5,
    PrivacyLevel.SECRET: 7
}

MAX_LATENCY_MS = {
    PrivacyLevel.PUBLIC: 1000,
    PrivacyLevel.PRIVATE: 3000,
    PrivacyLevel.CONFIDENTIAL: 5000,
    PrivacyLevel.SECRET: 10000
}

# System Configuration Defaults
DEFAULT_CONFIG = {
    "fog_system": {
        "enable_harvesting": True,
        "enable_privacy": True,
        "enable_marketplace": True,
        "enable_tokenomics": True,
        "enable_stats": True
    },
    "harvesting": {
        "min_battery_percent": MIN_BATTERY_PERCENT,
        "max_thermal_temp": MAX_THERMAL_TEMP,
        "require_charging": True,
        "require_wifi": True,
        "token_rate_per_hour": DEFAULT_TOKEN_RATE_PER_HOUR
    },
    "privacy": {
        "default_privacy_level": PrivacyLevel.PRIVATE.value,
        "circuit_pool_maintenance": True,
        "hidden_services_enabled": True,
        "mixnet_integration": False
    },
    "marketplace": {
        "service_tiers": ["basic", "premium", "enterprise"],
        "payment_methods": ["tokens", "crypto"],
        "sla_enforcement": True
    },
    "performance": {
        "max_concurrent_tasks": MAX_CONCURRENT_TASKS,
        "task_timeout_ms": DEFAULT_TASK_TIMEOUT,
        "health_check_interval": HEALTH_CHECK_INTERVAL,
        "stats_update_interval": STATS_UPDATE_INTERVAL
    }
}

# Error Messages
ERROR_MESSAGES = {
    "STARTUP_TIMEOUT": "System startup exceeded timeout limit",
    "DEVICE_REGISTRATION_FAILED": "Device registration failed",
    "PRIVACY_TASK_ROUTING_FAILED": "Privacy task routing failed",
    "CIRCUIT_CREATION_FAILED": "Circuit creation failed",
    "INSUFFICIENT_RESOURCES": "Insufficient system resources",
    "SERVICE_UNAVAILABLE": "Service temporarily unavailable",
    "INVALID_PRIVACY_LEVEL": "Invalid privacy level specified",
    "AUTHENTICATION_REQUIRED": "Authentication required for operation"
}

# Status Codes
class FogStatusCode(Enum):
    """Status codes for fog computing operations."""
    SUCCESS = "success"
    PENDING = "pending"
    FAILED = "failed"
    TIMEOUT = "timeout"
    UNAUTHORIZED = "unauthorized"
    INSUFFICIENT_RESOURCES = "insufficient_resources"
    SERVICE_UNAVAILABLE = "service_unavailable"

__all__ = [
    'PrivacyLevel', 'ServiceType', 'FogStatusCode',
    'DEFAULT_FOG_PORT', 'CIRCUIT_POOL_SIZES', 'PRIVACY_HOPS', 'MAX_LATENCY_MS',
    'CIRCUIT_ROTATION_INTERVAL', 'CIRCUIT_LIFETIME', 'HEALTH_CHECK_INTERVAL',
    'STATS_UPDATE_INTERVAL', 'TASK_CLEANUP_INTERVAL', 'SYSTEM_STARTUP_TIMEOUT',
    'MAX_CONCURRENT_TASKS', 'DEFAULT_TASK_TIMEOUT', 'CIRCUIT_LOAD_THRESHOLD',
    'DEVICE_REGISTRATION_TIMEOUT', 'MIN_BATTERY_PERCENT', 'MAX_THERMAL_TEMP',
    'DEFAULT_TOKEN_RATE_PER_HOUR', 'MAX_HARVEST_DEVICES', 'DEFAULT_CONFIG',
    'ERROR_MESSAGES'
]