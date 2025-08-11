"""Environment variable validation system for AIVillage CODEX integration.

This module provides comprehensive validation of all environment variables
required for CODEX components integration, including type checking, security
validation, and dependency verification.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import Enum
import logging
import os
from pathlib import Path
import socket
import sys

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationResult(Enum):
    """Validation result status."""

    VALID = "valid"
    INVALID = "invalid"
    MISSING = "missing"
    INSECURE = "insecure"


@dataclass
class ValidationIssue:
    """Individual validation issue."""

    variable: str
    level: ValidationLevel
    result: ValidationResult
    message: str
    suggestion: str | None = None


@dataclass
class ValidationReport:
    """Complete validation report."""

    issues: list[ValidationIssue] = field(default_factory=list)
    total_variables: int = 0
    valid_variables: int = 0
    warnings: int = 0
    errors: int = 0

    @property
    def is_valid(self) -> bool:
        """Check if configuration is valid (no errors)."""
        return self.errors == 0

    @property
    def has_warnings(self) -> bool:
        """Check if configuration has warnings."""
        return self.warnings > 0

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add validation issue to report."""
        self.issues.append(issue)
        if issue.level == ValidationLevel.ERROR:
            self.errors += 1
        elif issue.level == ValidationLevel.WARNING:
            self.warnings += 1


class EnvironmentValidator:
    """Comprehensive environment variable validator."""

    # Required variables by component
    REQUIRED_VARIABLES = {
        "evolution_metrics": ["AIVILLAGE_DB_PATH", "AIVILLAGE_STORAGE_BACKEND"],
        "rag_pipeline": [
            "RAG_EMBEDDING_MODEL",
            "RAG_VECTOR_DIM",
            "RAG_FAISS_INDEX_PATH",
        ],
        "p2p_networking": ["LIBP2P_HOST", "LIBP2P_PORT", "LIBP2P_PRIVATE_KEY_FILE"],
        "digital_twin": [
            "DIGITAL_TWIN_ENCRYPTION_KEY",
            "DIGITAL_TWIN_DB_PATH",
            "DIGITAL_TWIN_COPPA_COMPLIANT",
            "DIGITAL_TWIN_FERPA_COMPLIANT",
            "DIGITAL_TWIN_GDPR_COMPLIANT",
        ],
        "api_server": [
            "DIGITAL_TWIN_API_PORT",
            "EVOLUTION_METRICS_API_PORT",
            "RAG_PIPELINE_API_PORT",
            "API_SECRET_KEY",
        ],
        "core": ["AIVILLAGE_ENV"],
    }

    # Type validation rules
    TYPE_RULES = {
        # Integer types
        "int": [
            "LIBP2P_PORT",
            "LIBP2P_WEBSOCKET_PORT",
            "MDNS_DISCOVERY_INTERVAL",
            "MDNS_TTL",
            "MESH_MAX_PEERS",
            "MESH_HEARTBEAT_INTERVAL",
            "MESH_CONNECTION_TIMEOUT",
            "MESH_MAX_MESSAGE_SIZE",
            "DIGITAL_TWIN_API_PORT",
            "EVOLUTION_METRICS_API_PORT",
            "RAG_PIPELINE_API_PORT",
            "P2P_STATUS_API_PORT",
            "RAG_L1_CACHE_SIZE",
            "RAG_VECTOR_DIM",
            "RAG_DEFAULT_K",
            "RAG_CHUNK_SIZE",
            "RAG_CHUNK_OVERLAP",
            "RAG_VECTOR_TOP_K",
            "RAG_KEYWORD_TOP_K",
            "RAG_FINAL_TOP_K",
            "RAG_BATCH_SIZE",
            "AIVILLAGE_METRICS_FLUSH_THRESHOLD",
            "REDIS_PORT",
            "REDIS_DB",
            "DIGITAL_TWIN_MAX_PROFILES",
            "DIGITAL_TWIN_PROFILE_TTL_DAYS",
            "DIGITAL_TWIN_DATA_RETENTION_DAYS",
            "API_RATE_LIMIT_PER_MINUTE",
            "API_JWT_EXPIRY_HOURS",
            "API_REFRESH_TOKEN_EXPIRY_DAYS",
            "ANDROID_P2P_PORT_START",
            "ANDROID_P2P_PORT_END",
            "IOS_P2P_PORT_START",
            "IOS_P2P_PORT_END",
            "WIFI_DIRECT_PORT_START",
            "WIFI_DIRECT_PORT_END",
            "MAX_MEMORY_MB",
            "MAX_FAISS_MEMORY_MB",
            "MAX_CACHE_MEMORY_MB",
            "MAX_CONCURRENT_QUERIES",
            "MAX_DOCUMENT_SIZE_MB",
            "MAX_BATCH_PROCESSES",
            "MAX_CONNECTIONS",
            "CONNECTION_TIMEOUT",
            "READ_TIMEOUT",
            "WRITE_TIMEOUT",
            "HEALTH_CHECK_INTERVAL",
            "HEALTH_CHECK_TIMEOUT",
            "BACKUP_INTERVAL_HOURS",
            "BACKUP_RETENTION_DAYS",
            "RECOVERY_MAX_RETRIES",
            "RECOVERY_RETRY_DELAY",
        ],
        # Float types
        "float": ["RAG_SIMILARITY_THRESHOLD", "RAG_CACHE_TTL_HOURS"],
        # Boolean types
        "bool": [
            "RAG_CACHE_ENABLED",
            "RAG_RERANK_ENABLED",
            "MDNS_ENABLED",
            "MESH_ENABLE_TCP",
            "MESH_ENABLE_WEBSOCKET",
            "MESH_ENABLE_BLUETOOTH",
            "MESH_ENABLE_WIFI_DIRECT",
            "MESH_ENABLE_FILE_TRANSPORT",
            "MESH_TLS_ENABLED",
            "MESH_PEER_VERIFICATION",
            "MESH_ENCRYPTION_REQUIRED",
            "DIGITAL_TWIN_SQLITE_WAL",
            "DIGITAL_TWIN_BACKUP_ENCRYPTION",
            "DIGITAL_TWIN_COPPA_COMPLIANT",
            "DIGITAL_TWIN_FERPA_COMPLIANT",
            "DIGITAL_TWIN_GDPR_COMPLIANT",
            "DIGITAL_TWIN_ANALYTICS_ENABLED",
            "DIGITAL_TWIN_TELEMETRY_ENABLED",
            "DIGITAL_TWIN_AUDIT_LOG_ENABLED",
            "API_CORS_ENABLED",
            "API_AUTH_ENABLED",
            "AIVILLAGE_DEBUG_MODE",
            "AIVILLAGE_PROFILE_PERFORMANCE",
            "AIVILLAGE_ENABLE_METRICS",
            "HEALTH_CHECK_ENABLED",
            "TEST_DISABLE_AUTH",
            "TEST_MOCK_EXTERNAL_SERVICES",
            "TEST_FAST_MODE",
            "DEV_AUTO_RELOAD",
            "DEV_DEBUG_TOOLBAR",
            "DEV_MOCK_P2P",
            "DEV_SEED_DATA",
            "MOBILE_P2P_ENABLED",
            "MOBILE_OFFLINE_MODE",
            "MOBILE_BATTERY_OPTIMIZATION",
            "TLS_VERIFY_PEER",
            "SECURITY_HSTS_ENABLED",
            "SECURITY_CSP_ENABLED",
            "RATE_LIMIT_ENABLED",
            "BACKUP_ENABLED",
            "BACKUP_COMPRESS",
            "AUTO_RECOVERY_ENABLED",
            "FEATURE_ADVANCED_ANALYTICS",
            "FEATURE_MULTI_TENANT",
            "FEATURE_BLOCKCHAIN_INTEGRATION",
            "FEATURE_FEDERATED_LEARNING",
            "LEGACY_API_SUPPORT",
            "LEGACY_DATA_FORMAT",
            "ASYNC_PROCESSING",
            "BATCH_OPTIMIZATION",
            "LAZY_LOADING",
        ],
        # Path types (file or directory)
        "path": [
            "AIVILLAGE_DB_PATH",
            "AIVILLAGE_LOG_DIR",
            "RAG_DISK_CACHE_DIR",
            "RAG_FAISS_INDEX_PATH",
            "RAG_BM25_CORPUS_PATH",
            "RAG_CHUNK_STORE_PATH",
            "LIBP2P_PEER_ID_FILE",
            "LIBP2P_PRIVATE_KEY_FILE",
            "MESH_FILE_TRANSPORT_DIR",
            "DIGITAL_TWIN_VAULT_PATH",
            "DIGITAL_TWIN_DB_PATH",
            "AIVILLAGE_LOG_FILE",
            "TLS_CERT_FILE",
            "TLS_KEY_FILE",
            "TLS_CA_FILE",
            "BACKUP_STORAGE_PATH",
            "HUGGINGFACE_CACHE_DIR",
        ],
        # URL types
        "url": ["AIVILLAGE_REDIS_URL", "RAG_REDIS_URL", "MOBILE_API_BASE_URL"],
        # Enum types with allowed values
        "enum": {
            "AIVILLAGE_STORAGE_BACKEND": ["sqlite", "redis", "file"],
            "RAG_DEVICE": ["cpu", "cuda", "mps"],
            "AIVILLAGE_ENV": ["development", "testing", "staging", "production"],
            "AIVILLAGE_LOG_LEVEL": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "AIVILLAGE_LOG_FORMAT": ["structured", "simple", "json"],
            "SECURITY_FRAME_OPTIONS": ["DENY", "SAMEORIGIN", "ALLOW-FROM"],
            "RATE_LIMIT_STRATEGY": ["fixed-window", "sliding-window", "token-bucket"],
            "RATE_LIMIT_STORAGE": ["memory", "redis"],
        },
    }

    # Security-critical variables
    SECURITY_CRITICAL = {
        "DIGITAL_TWIN_ENCRYPTION_KEY",
        "API_SECRET_KEY",
        "LIBP2P_PRIVATE_KEY_FILE",
        "OPENAI_API_KEY",
        "WANDB_API_KEY",
        "HUGGINGFACE_API_TOKEN",
        "TLS_CERT_FILE",
        "TLS_KEY_FILE",
    }

    # Port ranges
    PORT_RANGES = {
        "LIBP2P_PORT": (1024, 65535),
        "LIBP2P_WEBSOCKET_PORT": (1024, 65535),
        "DIGITAL_TWIN_API_PORT": (8000, 9000),
        "EVOLUTION_METRICS_API_PORT": (8000, 9000),
        "RAG_PIPELINE_API_PORT": (8000, 9000),
        "P2P_STATUS_API_PORT": (8000, 9000),
        "ANDROID_P2P_PORT_START": (4000, 4030),
        "ANDROID_P2P_PORT_END": (4000, 4030),
        "IOS_P2P_PORT_START": (4000, 4030),
        "IOS_P2P_PORT_END": (4000, 4030),
        "WIFI_DIRECT_PORT_START": (4000, 4030),
        "WIFI_DIRECT_PORT_END": (4000, 4030),
    }

    def __init__(self, env_profile: str = "development") -> None:
        """Initialize validator with environment profile."""
        self.env_profile = env_profile
        self.report = ValidationReport()

    def validate_all(self, env_vars: dict[str, str] | None = None) -> ValidationReport:
        """Validate all environment variables."""
        if env_vars is None:
            env_vars = dict(os.environ)

        self.report = ValidationReport()

        # Count total variables we're checking
        all_required = set()
        for component_vars in self.REQUIRED_VARIABLES.values():
            all_required.update(component_vars)
        self.report.total_variables = len(all_required)

        # Validate required variables exist
        self._validate_required_variables(env_vars)

        # Validate types
        self._validate_types(env_vars)

        # Validate security configurations
        self._validate_security(env_vars)

        # Validate file paths and directories
        self._validate_paths(env_vars)

        # Validate port availability
        self._validate_ports(env_vars)

        # Validate external service connections
        self._validate_external_services(env_vars)

        # Environment-specific validation
        self._validate_environment_profile(env_vars)

        return self.report

    def _validate_required_variables(self, env_vars: dict[str, str]) -> None:
        """Validate that all required variables are present."""
        for component, variables in self.REQUIRED_VARIABLES.items():
            for var in variables:
                if var not in env_vars:
                    self.report.add_issue(
                        ValidationIssue(
                            variable=var,
                            level=ValidationLevel.ERROR,
                            result=ValidationResult.MISSING,
                            message=f"Required variable for {component} is missing",
                            suggestion=f"Set {var} in your environment or .env file",
                        )
                    )
                elif not env_vars[var].strip():
                    self.report.add_issue(
                        ValidationIssue(
                            variable=var,
                            level=ValidationLevel.ERROR,
                            result=ValidationResult.INVALID,
                            message=f"Required variable {var} is empty",
                            suggestion=f"Provide a value for {var}",
                        )
                    )
                else:
                    self.report.valid_variables += 1

    def _validate_types(self, env_vars: dict[str, str]) -> None:
        """Validate variable types."""
        for var_type, variables in self.TYPE_RULES.items():
            if var_type == "enum":
                # Handle enum validation
                for var, allowed_values in variables.items():
                    if var in env_vars:
                        value = env_vars[var]
                        if value not in allowed_values:
                            self.report.add_issue(
                                ValidationIssue(
                                    variable=var,
                                    level=ValidationLevel.ERROR,
                                    result=ValidationResult.INVALID,
                                    message=f"Invalid value '{value}' for {var}",
                                    suggestion=f"Use one of: {', '.join(allowed_values)}",
                                )
                            )
            else:
                # Handle other type validation
                for var in variables:
                    if var in env_vars:
                        value = env_vars[var]
                        if not self._validate_type(value, var_type):
                            self.report.add_issue(
                                ValidationIssue(
                                    variable=var,
                                    level=ValidationLevel.ERROR,
                                    result=ValidationResult.INVALID,
                                    message=f"Invalid {var_type} value '{value}' for {var}",
                                    suggestion=f"Provide a valid {var_type} value",
                                )
                            )

    def _validate_type(self, value: str, expected_type: str) -> bool:
        """Validate individual type."""
        try:
            if expected_type == "int":
                int(value)
            elif expected_type == "float":
                float(value)
            elif expected_type == "bool":
                if value.lower() not in [
                    "true",
                    "false",
                    "1",
                    "0",
                    "yes",
                    "no",
                    "on",
                    "off",
                ]:
                    return False
            elif expected_type == "url":
                if not (value.startswith(("http://", "https://", "redis://", "mongodb://"))):
                    return False
            elif expected_type == "path":
                # Basic path validation - just check it's not obviously invalid
                if not value or value.isspace():
                    return False
            return True
        except ValueError:
            return False

    def _validate_security(self, env_vars: dict[str, str]) -> None:
        """Validate security-critical configurations."""
        # Validate encryption key
        if "DIGITAL_TWIN_ENCRYPTION_KEY" in env_vars:
            key = env_vars["DIGITAL_TWIN_ENCRYPTION_KEY"]
            if key == "REPLACE_WITH_BASE64_ENCODED_32_BYTE_KEY":
                self.report.add_issue(
                    ValidationIssue(
                        variable="DIGITAL_TWIN_ENCRYPTION_KEY",
                        level=ValidationLevel.ERROR,
                        result=ValidationResult.INSECURE,
                        message="Encryption key is still using template value",
                        suggestion="Generate a secure 32-byte key and base64 encode it",
                    )
                )
            else:
                try:
                    decoded_key = base64.b64decode(key)
                    if len(decoded_key) != 32:
                        self.report.add_issue(
                            ValidationIssue(
                                variable="DIGITAL_TWIN_ENCRYPTION_KEY",
                                level=ValidationLevel.ERROR,
                                result=ValidationResult.INVALID,
                                message=f"Encryption key must be 32 bytes, got {len(decoded_key)}",
                                suggestion="Generate a 32-byte key: python -c 'import secrets, base64; print(base64.b64encode(secrets.token_bytes(32)).decode())'",
                            )
                        )
                except Exception as e:
                    self.report.add_issue(
                        ValidationIssue(
                            variable="DIGITAL_TWIN_ENCRYPTION_KEY",
                            level=ValidationLevel.ERROR,
                            result=ValidationResult.INVALID,
                            message=f"Invalid base64 encoding: {e}",
                            suggestion="Ensure key is properly base64 encoded",
                        )
                    )

        # Validate API secret key
        if "API_SECRET_KEY" in env_vars:
            key = env_vars["API_SECRET_KEY"]
            if key == "REPLACE_WITH_SECURE_SECRET_KEY":
                self.report.add_issue(
                    ValidationIssue(
                        variable="API_SECRET_KEY",
                        level=ValidationLevel.ERROR,
                        result=ValidationResult.INSECURE,
                        message="API secret key is still using template value",
                        suggestion="Generate a secure secret key",
                    )
                )
            elif len(key) < 32:
                self.report.add_issue(
                    ValidationIssue(
                        variable="API_SECRET_KEY",
                        level=ValidationLevel.WARNING,
                        result=ValidationResult.INSECURE,
                        message="API secret key is too short",
                        suggestion="Use at least 32 characters for security",
                    )
                )

        # Validate private key file permissions
        if "LIBP2P_PRIVATE_KEY_FILE" in env_vars:
            key_file = Path(env_vars["LIBP2P_PRIVATE_KEY_FILE"])
            if key_file.exists():
                try:
                    file_mode = key_file.stat().st_mode
                    if file_mode & 0o077:  # Check if others or group have any permissions
                        self.report.add_issue(
                            ValidationIssue(
                                variable="LIBP2P_PRIVATE_KEY_FILE",
                                level=ValidationLevel.WARNING,
                                result=ValidationResult.INSECURE,
                                message="Private key file has insecure permissions",
                                suggestion="Set permissions to 600: chmod 600 " + str(key_file),
                            )
                        )
                except OSError as e:
                    self.report.add_issue(
                        ValidationIssue(
                            variable="LIBP2P_PRIVATE_KEY_FILE",
                            level=ValidationLevel.WARNING,
                            result=ValidationResult.INVALID,
                            message=f"Cannot check file permissions: {e}",
                        )
                    )

        # Validate compliance flags in production
        if self.env_profile == "production":
            compliance_vars = [
                "DIGITAL_TWIN_COPPA_COMPLIANT",
                "DIGITAL_TWIN_FERPA_COMPLIANT",
                "DIGITAL_TWIN_GDPR_COMPLIANT",
            ]
            for var in compliance_vars:
                if var in env_vars and env_vars[var].lower() != "true":
                    self.report.add_issue(
                        ValidationIssue(
                            variable=var,
                            level=ValidationLevel.ERROR,
                            result=ValidationResult.INSECURE,
                            message=f"Compliance flag {var} must be true in production",
                            suggestion=f"Set {var}=true for production deployment",
                        )
                    )

    def _validate_paths(self, env_vars: dict[str, str]) -> None:
        """Validate file paths and directory permissions."""
        for var in self.TYPE_RULES["path"]:
            if var in env_vars:
                path_str = env_vars[var]
                path = Path(path_str)

                # Skip validation for template paths or empty paths
                if not path_str or path_str.startswith("REPLACE_WITH_"):
                    continue

                # Check if path should be a directory or file
                if var.endswith("_DIR") or (var.endswith("_PATH") and not var.endswith("_FILE")):
                    # Directory path
                    if var in ["AIVILLAGE_DB_PATH", "DIGITAL_TWIN_DB_PATH"]:
                        # Database files - check parent directory
                        parent = path.parent
                        if not parent.exists():
                            try:
                                parent.mkdir(parents=True, exist_ok=True)
                                self.report.add_issue(
                                    ValidationIssue(
                                        variable=var,
                                        level=ValidationLevel.INFO,
                                        result=ValidationResult.VALID,
                                        message=f"Created directory for {var}: {parent}",
                                    )
                                )
                            except OSError as e:
                                self.report.add_issue(
                                    ValidationIssue(
                                        variable=var,
                                        level=ValidationLevel.ERROR,
                                        result=ValidationResult.INVALID,
                                        message=f"Cannot create directory {parent}: {e}",
                                        suggestion="Ensure parent directory exists and is writable",
                                    )
                                )
                    # Regular directory
                    elif not path.exists():
                        try:
                            path.mkdir(parents=True, exist_ok=True)
                            self.report.add_issue(
                                ValidationIssue(
                                    variable=var,
                                    level=ValidationLevel.INFO,
                                    result=ValidationResult.VALID,
                                    message=f"Created directory for {var}: {path}",
                                )
                            )
                        except OSError as e:
                            self.report.add_issue(
                                ValidationIssue(
                                    variable=var,
                                    level=ValidationLevel.ERROR,
                                    result=ValidationResult.INVALID,
                                    message=f"Cannot create directory {path}: {e}",
                                    suggestion="Ensure path is valid and parent is writable",
                                )
                            )
                    elif not os.access(path, os.W_OK):
                        self.report.add_issue(
                            ValidationIssue(
                                variable=var,
                                level=ValidationLevel.ERROR,
                                result=ValidationResult.INVALID,
                                message=f"Directory {path} is not writable",
                                suggestion="Check directory permissions",
                            )
                        )
                # File path
                elif path.exists():
                    if not os.access(path, os.R_OK):
                        self.report.add_issue(
                            ValidationIssue(
                                variable=var,
                                level=ValidationLevel.ERROR,
                                result=ValidationResult.INVALID,
                                message=f"File {path} is not readable",
                                suggestion="Check file permissions",
                            )
                        )
                else:
                    # File doesn't exist - check if parent directory is writable
                    parent = path.parent
                    if not parent.exists():
                        self.report.add_issue(
                            ValidationIssue(
                                variable=var,
                                level=ValidationLevel.WARNING,
                                result=ValidationResult.INVALID,
                                message=f"Parent directory {parent} doesn't exist",
                                suggestion="Create parent directory or update path",
                            )
                        )
                    elif not os.access(parent, os.W_OK):
                        self.report.add_issue(
                            ValidationIssue(
                                variable=var,
                                level=ValidationLevel.ERROR,
                                result=ValidationResult.INVALID,
                                message=f"Cannot write to directory {parent}",
                                suggestion="Check directory permissions",
                            )
                        )

    def _validate_ports(self, env_vars: dict[str, str]) -> None:
        """Validate port availability and ranges."""
        used_ports = set()

        for var in self.TYPE_RULES["int"]:
            if "PORT" in var and var in env_vars:
                try:
                    port = int(env_vars[var])

                    # Check port range
                    if var in self.PORT_RANGES:
                        min_port, max_port = self.PORT_RANGES[var]
                        if not (min_port <= port <= max_port):
                            self.report.add_issue(
                                ValidationIssue(
                                    variable=var,
                                    level=ValidationLevel.ERROR,
                                    result=ValidationResult.INVALID,
                                    message=f"Port {port} outside allowed range {min_port}-{max_port}",
                                    suggestion=f"Use port between {min_port} and {max_port}",
                                )
                            )
                            continue

                    # Check for port conflicts
                    if port in used_ports:
                        self.report.add_issue(
                            ValidationIssue(
                                variable=var,
                                level=ValidationLevel.ERROR,
                                result=ValidationResult.INVALID,
                                message=f"Port {port} is used by multiple services",
                                suggestion="Assign unique ports to each service",
                            )
                        )
                    else:
                        used_ports.add(port)

                    # Check if port is available
                    if self._is_port_in_use(port):
                        self.report.add_issue(
                            ValidationIssue(
                                variable=var,
                                level=ValidationLevel.WARNING,
                                result=ValidationResult.INVALID,
                                message=f"Port {port} appears to be in use",
                                suggestion="Choose a different port or stop the conflicting service",
                            )
                        )

                except ValueError:
                    # Type validation will catch this
                    pass

        # Validate port ranges
        range_pairs = [
            ("ANDROID_P2P_PORT_START", "ANDROID_P2P_PORT_END"),
            ("IOS_P2P_PORT_START", "IOS_P2P_PORT_END"),
            ("WIFI_DIRECT_PORT_START", "WIFI_DIRECT_PORT_END"),
        ]

        for start_var, end_var in range_pairs:
            if start_var in env_vars and end_var in env_vars:
                try:
                    start_port = int(env_vars[start_var])
                    end_port = int(env_vars[end_var])

                    if start_port >= end_port:
                        self.report.add_issue(
                            ValidationIssue(
                                variable=f"{start_var}/{end_var}",
                                level=ValidationLevel.ERROR,
                                result=ValidationResult.INVALID,
                                message=f"Invalid port range: {start_port}-{end_port}",
                                suggestion="Start port must be less than end port",
                            )
                        )
                except ValueError:
                    pass

    def _is_port_in_use(self, port: int, host: str = "localhost") -> bool:
        """Check if a port is currently in use."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex((host, port))
                return result == 0
        except Exception:
            return False

    def _validate_external_services(self, env_vars: dict[str, str]) -> None:
        """Validate external service connections."""
        # Validate Redis connection
        redis_vars = ["AIVILLAGE_REDIS_URL", "RAG_REDIS_URL"]
        for var in redis_vars:
            if var in env_vars and REDIS_AVAILABLE:
                redis_url = env_vars[var]
                try:
                    r = redis.Redis.from_url(redis_url, socket_timeout=2)
                    r.ping()
                    self.report.add_issue(
                        ValidationIssue(
                            variable=var,
                            level=ValidationLevel.INFO,
                            result=ValidationResult.VALID,
                            message=f"Redis connection successful: {redis_url}",
                        )
                    )
                except Exception as e:
                    self.report.add_issue(
                        ValidationIssue(
                            variable=var,
                            level=ValidationLevel.WARNING,
                            result=ValidationResult.INVALID,
                            message=f"Cannot connect to Redis: {e}",
                            suggestion="Verify Redis server is running and URL is correct",
                        )
                    )

    def _validate_environment_profile(self, env_vars: dict[str, str]) -> None:
        """Validate environment-specific requirements."""
        env_type = env_vars.get("AIVILLAGE_ENV", "development")

        if env_type == "production":
            # Production must have security enabled
            security_required = [
                ("MESH_TLS_ENABLED", "true"),
                ("MESH_ENCRYPTION_REQUIRED", "true"),
                ("API_AUTH_ENABLED", "true"),
                ("DIGITAL_TWIN_BACKUP_ENCRYPTION", "true"),
            ]

            for var, expected in security_required:
                if var not in env_vars or env_vars[var].lower() != expected:
                    self.report.add_issue(
                        ValidationIssue(
                            variable=var,
                            level=ValidationLevel.ERROR,
                            result=ValidationResult.INSECURE,
                            message=f"Production requires {var}={expected}",
                            suggestion=f"Set {var}={expected} for production deployment",
                        )
                    )

            # Production should not have debug enabled
            debug_vars = [
                "AIVILLAGE_DEBUG_MODE",
                "DEV_AUTO_RELOAD",
                "DEV_DEBUG_TOOLBAR",
            ]
            for var in debug_vars:
                if var in env_vars and env_vars[var].lower() == "true":
                    self.report.add_issue(
                        ValidationIssue(
                            variable=var,
                            level=ValidationLevel.WARNING,
                            result=ValidationResult.INSECURE,
                            message=f"Debug feature {var} should not be enabled in production",
                            suggestion=f"Set {var}=false for production",
                        )
                    )

    def generate_report(self) -> str:
        """Generate human-readable validation report."""
        report_lines = [
            "=" * 80,
            "AIVILLAGE ENVIRONMENT VALIDATION REPORT",
            "=" * 80,
            f"Profile: {self.env_profile}",
            f"Total Variables Checked: {self.report.total_variables}",
            f"Valid Variables: {self.report.valid_variables}",
            f"Errors: {self.report.errors}",
            f"Warnings: {self.report.warnings}",
            "",
            f"Overall Status: {'✅ VALID' if self.report.is_valid else '❌ INVALID'}",
            "",
        ]

        if self.report.issues:
            # Group issues by level
            errors = [issue for issue in self.report.issues if issue.level == ValidationLevel.ERROR]
            warnings = [issue for issue in self.report.issues if issue.level == ValidationLevel.WARNING]
            info = [issue for issue in self.report.issues if issue.level == ValidationLevel.INFO]

            if errors:
                report_lines.extend(["ERRORS:", "-" * 40])
                for issue in errors:
                    report_lines.extend(
                        [
                            f"❌ {issue.variable}: {issue.message}",
                            f"   💡 {issue.suggestion}" if issue.suggestion else "",
                            "",
                        ]
                    )

            if warnings:
                report_lines.extend(["WARNINGS:", "-" * 40])
                for issue in warnings:
                    report_lines.extend(
                        [
                            f"⚠️  {issue.variable}: {issue.message}",
                            f"   💡 {issue.suggestion}" if issue.suggestion else "",
                            "",
                        ]
                    )

            if info:
                report_lines.extend(["INFORMATION:", "-" * 40])
                for issue in info:
                    report_lines.extend([f"ℹ️  {issue.variable}: {issue.message}", ""])
        else:
            report_lines.append("🎉 No issues found! All configurations are valid.")

        report_lines.extend(["", "=" * 80, "Generated by AIVillage Environment Validator", "=" * 80])

        return "\n".join(report_lines)


def validate_environment(env_profile: str | None = None) -> ValidationReport:
    """Main entry point for environment validation."""
    if env_profile is None:
        env_profile = os.environ.get("AIVILLAGE_ENV", "development")

    validator = EnvironmentValidator(env_profile)
    report = validator.validate_all()

    # Print report to console
    print(validator.generate_report())

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate AIVillage environment configuration")
    parser.add_argument(
        "--profile",
        choices=["development", "testing", "staging", "production"],
        default="development",
        help="Environment profile to validate",
    )
    parser.add_argument("--quiet", action="store_true", help="Only show errors and warnings")

    args = parser.parse_args()

    # Set up logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # Run validation
    report = validate_environment(args.profile)

    # Exit with error code if validation failed
    exit_code = 1 if not report.is_valid else 0
    sys.exit(exit_code)
