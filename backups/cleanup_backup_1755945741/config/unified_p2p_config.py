#!/usr/bin/env python3
"""
UNIFIED P2P CONFIGURATION SYSTEM
Consolidated configuration management for all P2P/BitChat/BetaNet/Fog systems

MISSION: Replace scattered configuration files with unified, production-ready config system
- Consolidates BitChat BLE settings
- Consolidates BetaNet HTX transport settings
- Consolidates Fog computing bridge settings
- Mobile platform optimization settings
- Environment-aware configuration loading
- Production-ready defaults with development overrides
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class NetworkProfile(Enum):
    """Network configuration profiles."""

    OFFLINE_FIRST = "offline_first"  # Prioritize BitChat BLE
    PRIVACY_FIRST = "privacy_first"  # Prioritize BetaNet HTX
    PERFORMANCE_FIRST = "performance_first"  # Prioritize fastest available
    BALANCED = "balanced"  # Intelligent adaptive selection
    MOBILE_OPTIMIZED = "mobile_optimized"  # Battery and data aware


class DeploymentMode(Enum):
    """Deployment environment modes."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class BitChatConfig:
    """BitChat BLE mesh configuration."""

    # Core BitChat settings
    enabled: bool = True
    device_name_prefix: str = "AIVillage"
    max_peers: int = 50
    hop_limit: int = 7
    max_message_size: int = 65536

    # BLE specific settings
    advertise_interval_ms: int = 1000
    scan_window_ms: int = 500
    connection_timeout_sec: int = 30

    # Mobile optimization
    battery_optimization: bool = True
    thermal_throttling: bool = True
    background_mode_limits: bool = True

    # Mesh networking
    enable_store_and_forward: bool = True
    offline_message_ttl_hours: int = 24
    peer_discovery_interval_sec: int = 30

    # Reliability features
    enable_acknowledgments: bool = True
    message_retry_attempts: int = 3
    peer_timeout_sec: int = 300

    # Compression and encryption
    enable_compression: bool = True
    enable_encryption: bool = False  # Optional for BitChat
    encryption_algorithm: str = "ChaCha20-Poly1305"


@dataclass
class BetaNetConfig:
    """BetaNet HTX transport configuration."""

    # Core BetaNet settings
    enabled: bool = True
    server_host: str = "127.0.0.1"
    server_port: int = 8443
    device_id_prefix: str = "aivillage"

    # HTX protocol settings
    max_frame_size: int = 16777215  # 2^24 - 1
    max_stream_id: int = 268435455  # 2^28 - 1
    initial_window_size: int = 65536

    # Connection management
    connect_timeout_sec: int = 10
    frame_timeout_sec: int = 5
    keepalive_interval_sec: int = 30
    max_concurrent_streams: int = 100

    # Retry and reliability
    max_retry_attempts: int = 3
    backoff_multiplier: float = 1.5
    max_backoff_sec: int = 30

    # Privacy and security (Noise XK)
    enable_noise_encryption: bool = True
    enable_forward_secrecy: bool = True
    key_rotation_interval_hours: int = 24

    # Mobile optimization
    enable_quic_fallback: bool = True
    mobile_data_conservation: bool = True
    cellular_quality_adaptation: bool = True


@dataclass
class FogBridgeConfig:
    """Fog computing bridge configuration."""

    # Core fog bridge settings
    enabled: bool = False  # Optional component
    gateway_host: str = "localhost"
    gateway_port: int = 8080

    # Integration settings
    privacy_mode: str = "balanced"  # strict, balanced, performance
    enable_covert_operations: bool = True
    job_scheduling_enabled: bool = True

    # Performance settings
    max_concurrent_jobs: int = 10
    job_timeout_sec: int = 300
    result_cache_ttl_sec: int = 3600

    # Cost optimization
    enable_cost_optimization: bool = True
    max_monthly_spend_usd: float | None = None
    preferred_regions: list[str] = field(default_factory=lambda: ["us-east-1"])


@dataclass
class MobileConfig:
    """Mobile platform specific configuration."""

    # Platform detection
    auto_detect_platform: bool = True
    platform_override: str | None = None  # android, ios, desktop

    # Battery optimization
    battery_aware_scheduling: bool = True
    low_battery_threshold: float = 0.2
    critical_battery_threshold: float = 0.1

    # Thermal management
    thermal_throttling_enabled: bool = True
    thermal_critical_temp_celsius: float = 40.0

    # Data usage optimization
    cellular_data_limits: bool = True
    wifi_preferred_operations: list[str] = field(
        default_factory=lambda: ["large_transfers", "video_calls", "bulk_sync"]
    )

    # Background operation limits
    background_message_queue_size: int = 100
    background_heartbeat_interval_sec: int = 300
    foreground_heartbeat_interval_sec: int = 60

    # Native integration
    enable_native_notifications: bool = True
    enable_background_app_refresh: bool = True
    enable_location_services: bool = False


@dataclass
class PerformanceConfig:
    """Performance and resource management configuration."""

    # Resource limits
    max_memory_mb: int = 512
    max_concurrent_connections: int = 100
    max_message_queue_size: int = 10000

    # Threading and async
    max_worker_threads: int = 8
    event_loop_policy: str = "auto"  # auto, asyncio, uvloop

    # Message handling
    message_processing_batch_size: int = 50
    message_cleanup_interval_sec: int = 300

    # Caching
    peer_cache_size: int = 1000
    message_cache_ttl_sec: int = 3600
    route_cache_ttl_sec: int = 1800

    # Monitoring
    enable_metrics_collection: bool = True
    metrics_export_interval_sec: int = 60
    performance_logging_level: str = "INFO"


@dataclass
class SecurityConfig:
    """Security and privacy configuration."""

    # Transport security
    enforce_encryption: bool = False  # Don't break BitChat compatibility
    require_authentication: bool = False

    # Privacy protection
    enable_onion_routing: bool = False
    max_relay_hops: int = 3
    enable_traffic_padding: bool = False

    # Access control
    peer_allowlist: list[str] = field(default_factory=list)
    peer_blocklist: list[str] = field(default_factory=list)
    enable_peer_verification: bool = False

    # Data protection
    enable_message_encryption: bool = False
    key_derivation_iterations: int = 100000
    secure_random_source: str = "system"  # system, hardware, deterministic


@dataclass
class UnifiedP2PConfig:
    """
    UNIFIED P2P CONFIGURATION

    Consolidates all P2P/BitChat/BetaNet/Fog configuration into a single,
    production-ready configuration system with environment awareness.
    """

    # Core system settings
    node_id: str | None = None  # Auto-generated if None
    device_name: str | None = None  # Auto-detected if None
    deployment_mode: DeploymentMode = DeploymentMode.DEVELOPMENT
    network_profile: NetworkProfile = NetworkProfile.BALANCED

    # Component configurations
    bitchat: BitChatConfig = field(default_factory=BitChatConfig)
    betanet: BetaNetConfig = field(default_factory=BetaNetConfig)
    fog_bridge: FogBridgeConfig = field(default_factory=FogBridgeConfig)
    mobile: MobileConfig = field(default_factory=MobileConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    # Advanced settings
    enable_experimental_features: bool = False
    debug_mode: bool = False
    log_level: str = "INFO"
    config_version: str = "2.0.0"

    def __post_init__(self):
        """Apply post-initialization configuration logic."""
        # Auto-generate node_id if not provided
        if not self.node_id:
            import secrets

            self.node_id = f"node_{secrets.token_hex(8)}"

        # Auto-detect device name
        if not self.device_name:
            import socket

            hostname = socket.gethostname()
            self.device_name = f"{hostname}_{self.node_id[:8]}"

        # Apply deployment mode optimizations
        self._apply_deployment_optimizations()

        # Apply network profile optimizations
        self._apply_network_profile_optimizations()

        logger.info(f"Unified P2P config initialized: {self.node_id}")

    def _apply_deployment_optimizations(self):
        """Apply deployment-specific optimizations."""

        if self.deployment_mode == DeploymentMode.PRODUCTION:
            # Production optimizations
            self.debug_mode = False
            self.log_level = "WARNING"
            self.enable_experimental_features = False

            # Enhanced reliability for production
            self.bitchat.message_retry_attempts = 5
            self.betanet.max_retry_attempts = 5

            # Better security for production
            self.security.enforce_encryption = True
            self.bitchat.enable_encryption = True

            # Performance tuning for production
            self.performance.max_memory_mb = 1024
            self.performance.max_concurrent_connections = 200

        elif self.deployment_mode == DeploymentMode.DEVELOPMENT:
            # Development optimizations
            self.debug_mode = True
            self.log_level = "DEBUG"
            self.enable_experimental_features = True

            # Faster iteration for development
            self.bitchat.peer_discovery_interval_sec = 10
            self.betanet.keepalive_interval_sec = 15

        elif self.deployment_mode == DeploymentMode.TESTING:
            # Testing optimizations
            self.debug_mode = True
            self.log_level = "DEBUG"

            # Faster timeouts for testing
            self.bitchat.connection_timeout_sec = 10
            self.betanet.connect_timeout_sec = 5
            self.performance.message_cleanup_interval_sec = 30

    def _apply_network_profile_optimizations(self):
        """Apply network profile specific optimizations."""

        if self.network_profile == NetworkProfile.OFFLINE_FIRST:
            # Prioritize BitChat BLE
            self.bitchat.enabled = True
            self.betanet.enabled = True  # Keep as fallback

            # Optimize for offline scenarios
            self.bitchat.enable_store_and_forward = True
            self.bitchat.offline_message_ttl_hours = 48
            self.bitchat.max_peers = 100

        elif self.network_profile == NetworkProfile.PRIVACY_FIRST:
            # Prioritize BetaNet HTX with maximum privacy
            self.betanet.enabled = True
            self.betanet.enable_noise_encryption = True
            self.betanet.enable_forward_secrecy = True

            # Enhanced privacy settings
            self.security.enable_onion_routing = True
            self.security.max_relay_hops = 5

        elif self.network_profile == NetworkProfile.PERFORMANCE_FIRST:
            # Optimize for maximum performance
            self.betanet.enabled = True
            self.betanet.max_concurrent_streams = 200

            # Performance optimizations
            self.performance.max_worker_threads = 16
            self.performance.message_processing_batch_size = 100
            self.bitchat.enable_compression = True

        elif self.network_profile == NetworkProfile.MOBILE_OPTIMIZED:
            # Optimize for mobile devices
            self.mobile.battery_aware_scheduling = True
            self.mobile.thermal_throttling_enabled = True
            self.mobile.cellular_data_limits = True

            # Conservative resource usage
            self.performance.max_memory_mb = 256
            self.bitchat.max_peers = 25
            self.betanet.max_concurrent_streams = 20


class UnifiedP2PConfigManager:
    """
    Configuration manager for unified P2P system.

    Handles loading, saving, and environment-aware configuration management.
    """

    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or Path("config")
        self.config_dir.mkdir(exist_ok=True)

        # Configuration file paths
        self.default_config_path = self.config_dir / "p2p_default.json"
        self.user_config_path = self.config_dir / "p2p_config.json"
        self.environment_config_path = self.config_dir / "p2p_env_config.json"

        logger.info(f"P2P config manager initialized: {self.config_dir}")

    def load_config(self, config_file: Path | None = None, environment_overrides: bool = True) -> UnifiedP2PConfig:
        """
        Load unified P2P configuration with environment awareness.

        Loading priority (highest to lowest):
        1. Explicit config file parameter
        2. Environment variables
        3. User config file
        4. Default config file
        5. Built-in defaults
        """

        # Start with built-in defaults
        config_dict = asdict(UnifiedP2PConfig())

        # Load default config file if exists
        if self.default_config_path.exists():
            try:
                with open(self.default_config_path) as f:
                    default_config = json.load(f)
                config_dict = self._deep_merge(config_dict, default_config)
                logger.debug("Loaded default P2P configuration")
            except Exception as e:
                logger.warning(f"Failed to load default config: {e}")

        # Load user config file if exists
        if self.user_config_path.exists():
            try:
                with open(self.user_config_path) as f:
                    user_config = json.load(f)
                config_dict = self._deep_merge(config_dict, user_config)
                logger.debug("Loaded user P2P configuration")
            except Exception as e:
                logger.warning(f"Failed to load user config: {e}")

        # Load explicit config file if provided
        if config_file and config_file.exists():
            try:
                with open(config_file) as f:
                    file_config = json.load(f)
                config_dict = self._deep_merge(config_dict, file_config)
                logger.info(f"Loaded explicit P2P config from {config_file}")
            except Exception as e:
                logger.error(f"Failed to load config from {config_file}: {e}")

        # Apply environment variable overrides
        if environment_overrides:
            config_dict = self._apply_environment_overrides(config_dict)

        # Convert back to dataclass
        return self._dict_to_config(config_dict)

    def save_config(self, config: UnifiedP2PConfig, config_file: Path | None = None):
        """Save unified P2P configuration to file."""

        save_path = config_file or self.user_config_path

        try:
            config_dict = asdict(config)

            # Convert enums to strings for JSON serialization
            config_dict = self._serialize_enums(config_dict)

            with open(save_path, "w") as f:
                json.dump(config_dict, f, indent=2)

            logger.info(f"Saved P2P configuration to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save config to {save_path}: {e}")
            raise

    def create_default_config(self, deployment_mode: DeploymentMode = DeploymentMode.DEVELOPMENT):
        """Create and save default configuration."""

        config = UnifiedP2PConfig(deployment_mode=deployment_mode)
        self.save_config(config, self.default_config_path)

        logger.info(f"Created default P2P configuration for {deployment_mode.value}")
        return config

    def _deep_merge(self, base_dict: dict, override_dict: dict) -> dict:
        """Deep merge two dictionaries."""

        result = base_dict.copy()

        for key, value in override_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _apply_environment_overrides(self, config_dict: dict) -> dict:
        """Apply environment variable overrides to configuration."""

        # Map environment variables to config paths
        env_mappings = {
            "P2P_NODE_ID": ["node_id"],
            "P2P_DEVICE_NAME": ["device_name"],
            "P2P_DEPLOYMENT_MODE": ["deployment_mode"],
            "P2P_NETWORK_PROFILE": ["network_profile"],
            # BitChat settings
            "BITCHAT_ENABLED": ["bitchat", "enabled"],
            "BITCHAT_MAX_PEERS": ["bitchat", "max_peers"],
            "BITCHAT_HOP_LIMIT": ["bitchat", "hop_limit"],
            # BetaNet settings
            "BETANET_ENABLED": ["betanet", "enabled"],
            "BETANET_SERVER_HOST": ["betanet", "server_host"],
            "BETANET_SERVER_PORT": ["betanet", "server_port"],
            # Fog bridge settings
            "FOG_BRIDGE_ENABLED": ["fog_bridge", "enabled"],
            "FOG_GATEWAY_HOST": ["fog_bridge", "gateway_host"],
            "FOG_GATEWAY_PORT": ["fog_bridge", "gateway_port"],
            # Debug settings
            "P2P_DEBUG_MODE": ["debug_mode"],
            "P2P_LOG_LEVEL": ["log_level"],
        }

        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Navigate to the correct location in config dict
                current_dict = config_dict
                for key in config_path[:-1]:
                    if key not in current_dict:
                        current_dict[key] = {}
                    current_dict = current_dict[key]

                # Convert environment variable to appropriate type
                final_key = config_path[-1]
                converted_value = self._convert_env_value(env_value)
                current_dict[final_key] = converted_value

                logger.debug(f"Applied environment override: {env_var}={converted_value}")

        return config_dict

    def _convert_env_value(self, value: str) -> str | int | float | bool:
        """Convert environment variable string to appropriate type."""

        # Boolean conversion
        if value.lower() in ("true", "1", "yes", "on"):
            return True
        elif value.lower() in ("false", "0", "no", "off"):
            return False

        # Number conversion
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # String (default)
        return value

    def _serialize_enums(self, obj: Any) -> Any:
        """Convert enums to strings for JSON serialization."""

        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, dict):
            return {key: self._serialize_enums(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_enums(item) for item in obj]
        else:
            return obj

    def _dict_to_config(self, config_dict: dict) -> UnifiedP2PConfig:
        """Convert configuration dictionary back to UnifiedP2PConfig dataclass."""

        try:
            # Convert string enums back to enum objects
            if "deployment_mode" in config_dict:
                config_dict["deployment_mode"] = DeploymentMode(config_dict["deployment_mode"])

            if "network_profile" in config_dict:
                config_dict["network_profile"] = NetworkProfile(config_dict["network_profile"])

            # Create nested dataclass objects
            nested_configs = {
                "bitchat": BitChatConfig,
                "betanet": BetaNetConfig,
                "fog_bridge": FogBridgeConfig,
                "mobile": MobileConfig,
                "performance": PerformanceConfig,
                "security": SecurityConfig,
            }

            for section_name, section_class in nested_configs.items():
                if section_name in config_dict:
                    config_dict[section_name] = section_class(**config_dict[section_name])

            return UnifiedP2PConfig(**config_dict)

        except Exception as e:
            logger.error(f"Failed to convert dict to config: {e}")
            # Fallback to default configuration
            return UnifiedP2PConfig()


# Global configuration instance
_config_manager = UnifiedP2PConfigManager()
_current_config: UnifiedP2PConfig | None = None


def get_p2p_config(reload: bool = False) -> UnifiedP2PConfig:
    """Get the current unified P2P configuration."""
    global _current_config

    if _current_config is None or reload:
        _current_config = _config_manager.load_config()

    return _current_config


def save_p2p_config(config: UnifiedP2PConfig):
    """Save the unified P2P configuration."""
    global _current_config

    _config_manager.save_config(config)
    _current_config = config


def create_development_config() -> UnifiedP2PConfig:
    """Create optimized development configuration."""
    return _config_manager.create_default_config(DeploymentMode.DEVELOPMENT)


def create_production_config() -> UnifiedP2PConfig:
    """Create optimized production configuration."""
    return _config_manager.create_default_config(DeploymentMode.PRODUCTION)


# Factory functions for common configurations
def create_offline_first_config(**overrides) -> UnifiedP2PConfig:
    """Create configuration optimized for offline-first scenarios."""
    config = UnifiedP2PConfig(network_profile=NetworkProfile.OFFLINE_FIRST, **overrides)
    return config


def create_privacy_first_config(**overrides) -> UnifiedP2PConfig:
    """Create configuration optimized for maximum privacy."""
    config = UnifiedP2PConfig(network_profile=NetworkProfile.PRIVACY_FIRST, **overrides)
    return config


def create_mobile_config(**overrides) -> UnifiedP2PConfig:
    """Create configuration optimized for mobile devices."""
    config = UnifiedP2PConfig(network_profile=NetworkProfile.MOBILE_OPTIMIZED, **overrides)
    return config


if __name__ == "__main__":
    # Demo configuration management

    print("=== UNIFIED P2P CONFIGURATION DEMO ===")

    # Create different configuration profiles
    configs = {
        "Development": create_development_config(),
        "Production": create_production_config(),
        "Offline-First": create_offline_first_config(),
        "Privacy-First": create_privacy_first_config(),
        "Mobile-Optimized": create_mobile_config(),
    }

    for name, config in configs.items():
        print(f"\n{name} Configuration:")
        print(f"  Node ID: {config.node_id}")
        print(f"  Network Profile: {config.network_profile.value}")
        print(f"  BitChat Enabled: {config.bitchat.enabled}")
        print(f"  BetaNet Enabled: {config.betanet.enabled}")
        print(f"  Max Peers: {config.bitchat.max_peers}")
        print(f"  Battery Optimization: {config.mobile.battery_aware_scheduling}")

    # Demonstrate configuration saving and loading
    print("\n=== Configuration Management Demo ===")

    # Save and load a configuration
    test_config = create_mobile_config(node_id="demo-mobile-node")
    save_p2p_config(test_config)

    loaded_config = get_p2p_config(reload=True)
    print(f"Loaded config node ID: {loaded_config.node_id}")

    print("\nUnified P2P configuration system ready! 🚀")
