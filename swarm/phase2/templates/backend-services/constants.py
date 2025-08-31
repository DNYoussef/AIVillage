"""
Backend Constants Module for AI Village Gateway

This module provides centralized constants for the gateway backend system:
- Service configuration constants
- Error codes and messages
- HTTP status codes and headers
- Default values and limits
- Feature flags and switches
"""

from enum import Enum, IntEnum
from typing import Dict, Any, Final
from dataclasses import dataclass


# Version Information
VERSION: Final[str] = "2.0.0"
API_VERSION: Final[str] = "v2"
BUILD_NUMBER: Final[str] = "2.0.0-phase2"

# Service Names
class ServiceNames:
    """Standard service names used throughout the gateway"""
    DATABASE = "database"
    CACHE = "cache"
    AUTH = "auth"
    API_GATEWAY = "api_gateway"
    RATE_LIMITER = "rate_limiter"
    LOGGER = "logger"
    METRICS = "metrics"
    HEALTH_CHECKER = "health_checker"
    CIRCUIT_BREAKER = "circuit_breaker"
    MESSAGE_QUEUE = "message_queue"
    FILE_STORAGE = "file_storage"
    NOTIFICATION = "notification"
    SEARCH = "search"
    AUDIT = "audit"
    SECURITY = "security"
    
    # AI Village Specific Services
    FOG_COMPUTE = "fog_compute"
    P2P_NETWORK = "p2p_network"
    FEDERATED_LEARNING = "federated_learning"
    EDGE_DEVICE = "edge_device"
    SWARM_COORDINATOR = "swarm_coordinator"
    CONSTITUTIONAL_AI = "constitutional_ai"
    TRANSPARENCY_ENGINE = "transparency_engine"
    BETANET_TRANSPORT = "betanet_transport"


# HTTP Constants
class HTTPStatus(IntEnum):
    """HTTP status codes"""
    # Success
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204
    PARTIAL_CONTENT = 206
    
    # Redirection
    MOVED_PERMANENTLY = 301
    FOUND = 302
    NOT_MODIFIED = 304
    TEMPORARY_REDIRECT = 307
    PERMANENT_REDIRECT = 308
    
    # Client Error
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    NOT_ACCEPTABLE = 406
    REQUEST_TIMEOUT = 408
    CONFLICT = 409
    GONE = 410
    LENGTH_REQUIRED = 411
    PRECONDITION_FAILED = 412
    PAYLOAD_TOO_LARGE = 413
    URI_TOO_LONG = 414
    UNSUPPORTED_MEDIA_TYPE = 415
    RANGE_NOT_SATISFIABLE = 416
    EXPECTATION_FAILED = 417
    UNPROCESSABLE_ENTITY = 422
    LOCKED = 423
    FAILED_DEPENDENCY = 424
    TOO_MANY_REQUESTS = 429
    REQUEST_HEADER_FIELDS_TOO_LARGE = 431
    
    # Server Error
    INTERNAL_SERVER_ERROR = 500
    NOT_IMPLEMENTED = 501
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504
    HTTP_VERSION_NOT_SUPPORTED = 505
    INSUFFICIENT_STORAGE = 507
    LOOP_DETECTED = 508
    NETWORK_AUTHENTICATION_REQUIRED = 511


class HTTPHeaders:
    """Standard HTTP headers"""
    # Request Headers
    AUTHORIZATION = "Authorization"
    CONTENT_TYPE = "Content-Type"
    CONTENT_LENGTH = "Content-Length"
    USER_AGENT = "User-Agent"
    ACCEPT = "Accept"
    ACCEPT_ENCODING = "Accept-Encoding"
    ACCEPT_LANGUAGE = "Accept-Language"
    HOST = "Host"
    REFERER = "Referer"
    ORIGIN = "Origin"
    
    # Response Headers
    CACHE_CONTROL = "Cache-Control"
    SET_COOKIE = "Set-Cookie"
    LOCATION = "Location"
    ETAG = "ETag"
    LAST_MODIFIED = "Last-Modified"
    EXPIRES = "Expires"
    
    # CORS Headers
    ACCESS_CONTROL_ALLOW_ORIGIN = "Access-Control-Allow-Origin"
    ACCESS_CONTROL_ALLOW_METHODS = "Access-Control-Allow-Methods"
    ACCESS_CONTROL_ALLOW_HEADERS = "Access-Control-Allow-Headers"
    ACCESS_CONTROL_ALLOW_CREDENTIALS = "Access-Control-Allow-Credentials"
    ACCESS_CONTROL_MAX_AGE = "Access-Control-Max-Age"
    
    # Custom Headers
    X_REQUEST_ID = "X-Request-ID"
    X_FORWARDED_FOR = "X-Forwarded-For"
    X_FORWARDED_PROTO = "X-Forwarded-Proto"
    X_REAL_IP = "X-Real-IP"
    X_USER_ID = "X-User-ID"
    X_API_KEY = "X-API-Key"
    X_RATE_LIMIT = "X-RateLimit-Limit"
    X_RATE_LIMIT_REMAINING = "X-RateLimit-Remaining"
    X_RATE_LIMIT_RESET = "X-RateLimit-Reset"


class ContentTypes:
    """Common content types"""
    JSON = "application/json"
    XML = "application/xml"
    HTML = "text/html"
    PLAIN = "text/plain"
    CSS = "text/css"
    JAVASCRIPT = "application/javascript"
    PDF = "application/pdf"
    ZIP = "application/zip"
    GZIP = "application/gzip"
    FORM_DATA = "multipart/form-data"
    FORM_ENCODED = "application/x-www-form-urlencoded"
    BINARY = "application/octet-stream"


# Error Codes
class ErrorCodes:
    """Standardized error codes"""
    # Generic Errors
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"
    
    # Service Errors
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    SERVICE_INITIALIZATION_ERROR = "SERVICE_INITIALIZATION_ERROR"
    SERVICE_DEPENDENCY_ERROR = "SERVICE_DEPENDENCY_ERROR"
    SERVICE_TIMEOUT = "SERVICE_TIMEOUT"
    SERVICE_OVERLOAD = "SERVICE_OVERLOAD"
    
    # Authentication & Authorization
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    AUTHORIZATION_FAILED = "AUTHORIZATION_FAILED"
    INVALID_TOKEN = "INVALID_TOKEN"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"
    
    # Validation Errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_PARAMETER = "MISSING_PARAMETER"
    INVALID_PARAMETER = "INVALID_PARAMETER"
    SCHEMA_VALIDATION_ERROR = "SCHEMA_VALIDATION_ERROR"
    
    # Rate Limiting
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    
    # Resource Errors
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_ALREADY_EXISTS = "RESOURCE_ALREADY_EXISTS"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
    RESOURCE_LOCKED = "RESOURCE_LOCKED"
    
    # AI Village Specific Errors
    FOG_COMPUTE_ERROR = "FOG_COMPUTE_ERROR"
    P2P_NETWORK_ERROR = "P2P_NETWORK_ERROR"
    FEDERATED_LEARNING_ERROR = "FEDERATED_LEARNING_ERROR"
    EDGE_DEVICE_ERROR = "EDGE_DEVICE_ERROR"
    SWARM_COORDINATION_ERROR = "SWARM_COORDINATION_ERROR"
    CONSTITUTIONAL_VALIDATION_ERROR = "CONSTITUTIONAL_VALIDATION_ERROR"
    TRANSPARENCY_AUDIT_ERROR = "TRANSPARENCY_AUDIT_ERROR"
    BETANET_TRANSPORT_ERROR = "BETANET_TRANSPORT_ERROR"


# Default Configuration Values
class DefaultConfig:
    """Default configuration values"""
    
    # Server Configuration
    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = 8000
    DEFAULT_WORKERS = 4
    DEFAULT_MAX_CONNECTIONS = 1000
    DEFAULT_KEEPALIVE_TIMEOUT = 5
    DEFAULT_REQUEST_TIMEOUT = 30
    
    # Database Configuration
    DEFAULT_DB_POOL_SIZE = 10
    DEFAULT_DB_MAX_OVERFLOW = 20
    DEFAULT_DB_TIMEOUT = 30
    DEFAULT_DB_RETRY_COUNT = 3
    
    # Cache Configuration
    DEFAULT_CACHE_TTL = 300  # 5 minutes
    DEFAULT_CACHE_MAX_SIZE = 1000
    DEFAULT_CACHE_CLEANUP_INTERVAL = 60
    
    # Rate Limiting
    DEFAULT_RATE_LIMIT = 100
    DEFAULT_RATE_WINDOW = 60  # seconds
    DEFAULT_BURST_LIMIT = 10
    
    # Authentication
    DEFAULT_TOKEN_EXPIRY = 3600  # 1 hour
    DEFAULT_REFRESH_TOKEN_EXPIRY = 86400  # 24 hours
    DEFAULT_SESSION_TIMEOUT = 1800  # 30 minutes
    
    # Logging
    DEFAULT_LOG_LEVEL = "INFO"
    DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DEFAULT_LOG_MAX_SIZE = 100 * 1024 * 1024  # 100MB
    DEFAULT_LOG_BACKUP_COUNT = 5
    
    # Health Check
    DEFAULT_HEALTH_CHECK_INTERVAL = 30  # seconds
    DEFAULT_HEALTH_CHECK_TIMEOUT = 5
    DEFAULT_HEALTH_CHECK_RETRIES = 3
    
    # Circuit Breaker
    DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
    DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60
    DEFAULT_CIRCUIT_BREAKER_EXPECTED_EXCEPTION = Exception
    
    # Message Queue
    DEFAULT_QUEUE_MAX_SIZE = 1000
    DEFAULT_QUEUE_TIMEOUT = 30
    DEFAULT_MESSAGE_TTL = 3600
    
    # File Storage
    DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    DEFAULT_ALLOWED_EXTENSIONS = [".txt", ".json", ".csv", ".png", ".jpg", ".jpeg"]
    DEFAULT_UPLOAD_PATH = "/tmp/uploads"
    
    # AI Village Specific Defaults
    DEFAULT_FOG_COMPUTE_TIMEOUT = 300  # 5 minutes
    DEFAULT_P2P_CONNECTION_TIMEOUT = 60
    DEFAULT_FEDERATED_LEARNING_ROUNDS = 10
    DEFAULT_EDGE_DEVICE_HEARTBEAT = 30
    DEFAULT_SWARM_MAX_AGENTS = 100
    DEFAULT_CONSTITUTIONAL_VALIDATION_TIMEOUT = 10
    DEFAULT_TRANSPARENCY_AUDIT_INTERVAL = 3600
    DEFAULT_BETANET_MIXNODE_COUNT = 3


# Feature Flags
class FeatureFlags:
    """Feature toggle flags"""
    
    # Core Features
    ENABLE_CORS = "enable_cors"
    ENABLE_COMPRESSION = "enable_compression"
    ENABLE_CACHING = "enable_caching"
    ENABLE_RATE_LIMITING = "enable_rate_limiting"
    ENABLE_AUTHENTICATION = "enable_authentication"
    ENABLE_AUTHORIZATION = "enable_authorization"
    ENABLE_AUDIT_LOGGING = "enable_audit_logging"
    ENABLE_METRICS_COLLECTION = "enable_metrics_collection"
    ENABLE_HEALTH_CHECKS = "enable_health_checks"
    ENABLE_CIRCUIT_BREAKER = "enable_circuit_breaker"
    
    # Development Features
    ENABLE_DEBUG_MODE = "enable_debug_mode"
    ENABLE_PROFILING = "enable_profiling"
    ENABLE_REQUEST_LOGGING = "enable_request_logging"
    ENABLE_RESPONSE_LOGGING = "enable_response_logging"
    ENABLE_SQL_LOGGING = "enable_sql_logging"
    ENABLE_PERFORMANCE_MONITORING = "enable_performance_monitoring"
    
    # AI Village Features
    ENABLE_FOG_COMPUTING = "enable_fog_computing"
    ENABLE_P2P_NETWORKING = "enable_p2p_networking"
    ENABLE_FEDERATED_LEARNING = "enable_federated_learning"
    ENABLE_EDGE_DEVICE_INTEGRATION = "enable_edge_device_integration"
    ENABLE_SWARM_COORDINATION = "enable_swarm_coordination"
    ENABLE_CONSTITUTIONAL_AI = "enable_constitutional_ai"
    ENABLE_TRANSPARENCY_ENGINE = "enable_transparency_engine"
    ENABLE_BETANET_TRANSPORT = "enable_betanet_transport"
    ENABLE_PRIVACY_VERIFICATION = "enable_privacy_verification"
    ENABLE_CONSTITUTIONAL_PRICING = "enable_constitutional_pricing"


# System Limits
class SystemLimits:
    """System resource limits"""
    
    # Memory Limits
    MAX_MEMORY_USAGE = 2 * 1024 * 1024 * 1024  # 2GB
    MAX_CACHE_MEMORY = 512 * 1024 * 1024  # 512MB
    MAX_REQUEST_MEMORY = 100 * 1024 * 1024  # 100MB
    
    # Connection Limits
    MAX_CONCURRENT_CONNECTIONS = 10000
    MAX_CONNECTIONS_PER_IP = 100
    MAX_IDLE_CONNECTIONS = 1000
    
    # Request Limits
    MAX_REQUEST_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_HEADER_SIZE = 8 * 1024  # 8KB
    MAX_URL_LENGTH = 2048
    MAX_QUERY_PARAMETERS = 1000
    
    # Time Limits
    MAX_REQUEST_TIMEOUT = 300  # 5 minutes
    MAX_RESPONSE_TIMEOUT = 300  # 5 minutes
    MAX_DATABASE_QUERY_TIMEOUT = 60  # 1 minute
    MAX_CACHE_OPERATION_TIMEOUT = 5  # 5 seconds
    
    # AI Village Specific Limits
    MAX_FOG_COMPUTE_JOBS = 1000
    MAX_P2P_CONNECTIONS = 500
    MAX_FEDERATED_PARTICIPANTS = 100
    MAX_EDGE_DEVICES = 10000
    MAX_SWARM_AGENTS = 1000
    MAX_CONSTITUTIONAL_RULES = 100
    MAX_TRANSPARENCY_LOGS = 1000000
    MAX_BETANET_HOPS = 10


# Priority Levels
class Priority(IntEnum):
    """Priority levels for various operations"""
    CRITICAL = 1000
    HIGH = 800
    MEDIUM = 500
    LOW = 200
    BACKGROUND = 100


# Environment Types
class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


# Logging Levels
class LogLevel(Enum):
    """Logging levels"""
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


# Cache Types
class CacheType(Enum):
    """Cache implementation types"""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"
    DATABASE = "database"


# Database Types
class DatabaseType(Enum):
    """Database types"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"


# Message Queue Types
class MessageQueueType(Enum):
    """Message queue types"""
    MEMORY = "memory"
    REDIS = "redis"
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"
    SQS = "sqs"


# AI Village Specific Constants
class AIVillageConstants:
    """AI Village specific constants"""
    
    # Fog Computing
    FOG_NODE_TYPES = ["edge", "intermediate", "cloud"]
    FOG_RESOURCE_TYPES = ["cpu", "memory", "storage", "gpu"]
    FOG_JOB_STATUSES = ["pending", "running", "completed", "failed", "cancelled"]
    
    # P2P Networking
    P2P_PROTOCOLS = ["libp2p", "kademlia", "gossip", "dht"]
    P2P_TRANSPORT_TYPES = ["tcp", "udp", "websocket", "webrtc"]
    P2P_PEER_TYPES = ["bootstrap", "relay", "client"]
    
    # Federated Learning
    FL_AGGREGATION_METHODS = ["fedavg", "fedprox", "scaffold", "fednova"]
    FL_CLIENT_STATES = ["idle", "training", "uploading", "waiting"]
    FL_ROUND_PHASES = ["initialization", "training", "aggregation", "evaluation"]
    
    # Edge Devices
    EDGE_DEVICE_TYPES = ["mobile", "iot", "embedded", "desktop"]
    EDGE_CAPABILITIES = ["compute", "storage", "sensing", "communication"]
    EDGE_POWER_MODES = ["battery", "plugged", "low_power", "high_performance"]
    
    # Swarm Coordination
    SWARM_TOPOLOGIES = ["mesh", "star", "ring", "tree", "hybrid"]
    SWARM_ROLES = ["coordinator", "worker", "observer", "backup"]
    SWARM_STATES = ["initializing", "active", "degraded", "failed"]
    
    # Constitutional AI
    CONSTITUTIONAL_PRINCIPLES = ["safety", "fairness", "transparency", "privacy", "accountability"]
    VALIDATION_LEVELS = ["strict", "moderate", "permissive"]
    AUDIT_TYPES = ["compliance", "performance", "security", "ethics"]
    
    # Transparency Engine
    TRANSPARENCY_LEVELS = ["public", "restricted", "private", "confidential"]
    AUDIT_EVENTS = ["access", "modification", "deletion", "creation", "execution"]
    COMPLIANCE_FRAMEWORKS = ["gdpr", "ccpa", "hipaa", "sox", "pci"]
    
    # Betanet Transport
    MIXNODE_TYPES = ["entry", "middle", "exit"]
    TRANSPORT_LAYERS = ["physical", "network", "transport", "application"]
    PRIVACY_LEVELS = ["low", "medium", "high", "maximum"]


@dataclass
class ServiceDefaults:
    """Default configuration for services"""
    timeout: int = DefaultConfig.DEFAULT_REQUEST_TIMEOUT
    retry_count: int = DefaultConfig.DEFAULT_DB_RETRY_COUNT
    health_check_interval: int = DefaultConfig.DEFAULT_HEALTH_CHECK_INTERVAL
    max_connections: int = DefaultConfig.DEFAULT_MAX_CONNECTIONS
    enable_metrics: bool = True
    enable_logging: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "health_check_interval": self.health_check_interval,
            "max_connections": self.max_connections,
            "enable_metrics": self.enable_metrics,
            "enable_logging": self.enable_logging
        }


# Utility functions for constants

def get_error_message(error_code: str) -> str:
    """Get human-readable error message for error code"""
    error_messages = {
        ErrorCodes.UNKNOWN_ERROR: "An unknown error occurred",
        ErrorCodes.INTERNAL_ERROR: "Internal server error",
        ErrorCodes.CONFIGURATION_ERROR: "Configuration error",
        ErrorCodes.DEPENDENCY_ERROR: "Dependency error",
        ErrorCodes.SERVICE_UNAVAILABLE: "Service temporarily unavailable",
        ErrorCodes.AUTHENTICATION_REQUIRED: "Authentication required",
        ErrorCodes.AUTHENTICATION_FAILED: "Authentication failed",
        ErrorCodes.AUTHORIZATION_FAILED: "Access denied",
        ErrorCodes.VALIDATION_ERROR: "Input validation failed",
        ErrorCodes.RATE_LIMIT_EXCEEDED: "Rate limit exceeded",
        ErrorCodes.RESOURCE_NOT_FOUND: "Resource not found",
        ErrorCodes.RESOURCE_CONFLICT: "Resource conflict",
    }
    
    return error_messages.get(error_code, "Unknown error")


def get_default_config() -> Dict[str, Any]:
    """Get default configuration dictionary"""
    return {
        "server": {
            "host": DefaultConfig.DEFAULT_HOST,
            "port": DefaultConfig.DEFAULT_PORT,
            "workers": DefaultConfig.DEFAULT_WORKERS,
            "timeout": DefaultConfig.DEFAULT_REQUEST_TIMEOUT,
        },
        "database": {
            "pool_size": DefaultConfig.DEFAULT_DB_POOL_SIZE,
            "timeout": DefaultConfig.DEFAULT_DB_TIMEOUT,
            "retry_count": DefaultConfig.DEFAULT_DB_RETRY_COUNT,
        },
        "cache": {
            "ttl": DefaultConfig.DEFAULT_CACHE_TTL,
            "max_size": DefaultConfig.DEFAULT_CACHE_MAX_SIZE,
        },
        "rate_limiting": {
            "max_requests": DefaultConfig.DEFAULT_RATE_LIMIT,
            "window_seconds": DefaultConfig.DEFAULT_RATE_WINDOW,
        },
        "logging": {
            "level": DefaultConfig.DEFAULT_LOG_LEVEL,
            "format": DefaultConfig.DEFAULT_LOG_FORMAT,
        },
        "features": {
            FeatureFlags.ENABLE_CORS: True,
            FeatureFlags.ENABLE_CACHING: True,
            FeatureFlags.ENABLE_RATE_LIMITING: True,
            FeatureFlags.ENABLE_AUTHENTICATION: True,
            FeatureFlags.ENABLE_HEALTH_CHECKS: True,
        }
    }


def validate_feature_flag(flag: str) -> bool:
    """Validate if a feature flag exists"""
    return hasattr(FeatureFlags, flag.upper())


def get_service_priority(service_name: str) -> int:
    """Get default priority for a service"""
    priority_map = {
        ServiceNames.DATABASE: Priority.CRITICAL,
        ServiceNames.CACHE: Priority.HIGH,
        ServiceNames.AUTH: Priority.HIGH,
        ServiceNames.API_GATEWAY: Priority.MEDIUM,
        ServiceNames.RATE_LIMITER: Priority.MEDIUM,
        ServiceNames.LOGGER: Priority.LOW,
        ServiceNames.METRICS: Priority.LOW,
        ServiceNames.HEALTH_CHECKER: Priority.LOW,
    }
    
    return priority_map.get(service_name, Priority.MEDIUM)


# Export all public constants and classes
__all__ = [
    'VERSION', 'API_VERSION', 'BUILD_NUMBER',
    'ServiceNames', 'HTTPStatus', 'HTTPHeaders', 'ContentTypes',
    'ErrorCodes', 'DefaultConfig', 'FeatureFlags', 'SystemLimits',
    'Priority', 'Environment', 'LogLevel', 'CacheType', 'DatabaseType',
    'MessageQueueType', 'AIVillageConstants', 'ServiceDefaults',
    'get_error_message', 'get_default_config', 'validate_feature_flag',
    'get_service_priority'
]