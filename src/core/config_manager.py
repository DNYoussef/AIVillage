#!/usr/bin/env python3
"""Configuration management system for CODEX Integration.

Provides centralized configuration management with:
- Hot-reload capability for configuration changes
- Environment variable overrides
- Configuration validation and consistency checking
- Multiple configuration source support (YAML, JSON, ENV)
"""

import copy
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import yaml

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or inconsistent."""


class ConfigWatcher(FileSystemEventHandler):
    """Watches configuration files for changes and triggers reloads."""

    def __init__(self, config_manager) -> None:
        self.config_manager = config_manager

    def on_modified(self, event) -> None:
        """Handle file modification events."""
        if not event.is_directory and event.src_path in self.config_manager.watched_files:
            logger.info(f"Configuration file changed: {event.src_path}")
            self.config_manager.reload_config()


class CODEXConfigManager:
    """Comprehensive configuration manager for CODEX integration."""

    def __init__(self, config_dir: str = "config", enable_hot_reload: bool = True) -> None:
        self.config_dir = Path(config_dir)
        self.enable_hot_reload = enable_hot_reload
        self.config_data = {}
        self.watched_files = set()
        self.observer = None
        self.lock = threading.RLock()
        self.last_reload = datetime.now()

        # Configuration file paths
        self.main_config = self.config_dir / "aivillage_config.yaml"
        self.p2p_config = self.config_dir / "p2p_config.json"
        self.rag_config = self.config_dir / "rag_config.json"

        # Environment variable mappings from CODEX requirements
        self.env_mappings = {
            # Evolution Metrics System
            "AIVILLAGE_DB_PATH": "integration.evolution_metrics.db_path",
            "AIVILLAGE_STORAGE_BACKEND": "integration.evolution_metrics.backend",
            "AIVILLAGE_REDIS_URL": "integration.evolution_metrics.redis_url",
            "AIVILLAGE_METRICS_FLUSH_THRESHOLD": "integration.evolution_metrics.flush_threshold",
            "AIVILLAGE_METRICS_FILE": "integration.evolution_metrics.metrics_file",
            "AIVILLAGE_LOG_DIR": "integration.evolution_metrics.log_dir",
            "REDIS_HOST": "redis.host",
            "REDIS_PORT": "redis.port",
            "REDIS_DB": "redis.db",
            # RAG Pipeline System
            "RAG_CACHE_ENABLED": "integration.rag_pipeline.cache_enabled",
            "RAG_L1_CACHE_SIZE": "rag_config.cache.l1_size",
            "RAG_REDIS_URL": "integration.rag_pipeline.redis_url",
            "RAG_DISK_CACHE_DIR": "rag_config.cache.l3_directory",
            "RAG_EMBEDDING_MODEL": "integration.rag_pipeline.embedding_model",
            "RAG_CROSS_ENCODER_MODEL": "integration.rag_pipeline.cross_encoder_model",
            "RAG_VECTOR_DIM": "integration.rag_pipeline.vector_dim",
            "RAG_FAISS_INDEX_PATH": "integration.rag_pipeline.faiss_index_path",
            "RAG_BM25_CORPUS_PATH": "integration.rag_pipeline.bm25_corpus_path",
            "RAG_DEFAULT_K": "rag_config.retrieval.final_top_k",
            "RAG_CHUNK_SIZE": "integration.rag_pipeline.chunk_size",
            "RAG_CHUNK_OVERLAP": "integration.rag_pipeline.chunk_overlap",
            # P2P Networking
            "LIBP2P_HOST": "p2p_config.host",
            "LIBP2P_PORT": "p2p_config.port",
            "LIBP2P_PEER_ID_FILE": "p2p_config.peer_id_file",
            "LIBP2P_PRIVATE_KEY_FILE": "p2p_config.private_key_file",
            "MDNS_SERVICE_NAME": "p2p_config.mdns_service_name",
            "MDNS_DISCOVERY_INTERVAL": "p2p_config.peer_discovery.discovery_interval",
            "MDNS_TTL": "p2p_config.mdns_ttl",
            "MESH_MAX_PEERS": "integration.p2p_networking.max_peers",
            "MESH_HEARTBEAT_INTERVAL": "p2p_config.heartbeat_interval",
            "MESH_CONNECTION_TIMEOUT": "p2p_config.connection_timeout",
            "MESH_ENABLE_BLUETOOTH": "p2p_config.transports.bluetooth_enabled",
            "MESH_ENABLE_WIFI_DIRECT": "p2p_config.transports.wifi_direct_enabled",
            "MESH_ENABLE_FILE_TRANSPORT": "p2p_config.enable_file_transport",
            "MESH_FILE_TRANSPORT_DIR": "p2p_config.file_transport_dir",
            # Digital Twin System
            "DIGITAL_TWIN_ENCRYPTION_KEY": "integration.digital_twin.encryption_key",
            "DIGITAL_TWIN_VAULT_PATH": "integration.digital_twin.vault_path",
            "DIGITAL_TWIN_DB_PATH": "integration.digital_twin.db_path",
            "DIGITAL_TWIN_SQLITE_WAL": "integration.digital_twin.sqlite_wal",
            "DIGITAL_TWIN_COPPA_COMPLIANT": "integration.digital_twin.coppa_compliant",
            "DIGITAL_TWIN_FERPA_COMPLIANT": "integration.digital_twin.ferpa_compliant",
            "DIGITAL_TWIN_GDPR_COMPLIANT": "integration.digital_twin.gdpr_compliant",
            "DIGITAL_TWIN_MAX_PROFILES": "integration.digital_twin.max_profiles",
            "DIGITAL_TWIN_PROFILE_TTL_DAYS": "integration.digital_twin.profile_ttl_days",
        }

        # Load initial configuration
        self.reload_config()

        # Start file watcher if hot reload is enabled
        if self.enable_hot_reload:
            self.start_file_watcher()

    def load_yaml_file(self, file_path: Path) -> dict[str, Any]:
        """Load and parse YAML configuration file."""
        try:
            if not file_path.exists():
                logger.warning(f"Configuration file not found: {file_path}")
                return {}

            with open(file_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            logger.info(f"Loaded YAML configuration: {file_path}")
            return data if data is not None else {}

        except yaml.YAMLError as e:
            msg = f"Invalid YAML in {file_path}: {e}"
            raise ConfigurationError(msg)
        except Exception as e:
            msg = f"Error loading {file_path}: {e}"
            raise ConfigurationError(msg)

    def load_json_file(self, file_path: Path) -> dict[str, Any]:
        """Load and parse JSON configuration file."""
        try:
            if not file_path.exists():
                logger.warning(f"Configuration file not found: {file_path}")
                return {}

            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            logger.info(f"Loaded JSON configuration: {file_path}")
            return data if data is not None else {}

        except json.JSONDecodeError as e:
            msg = f"Invalid JSON in {file_path}: {e}"
            raise ConfigurationError(msg)
        except Exception as e:
            msg = f"Error loading {file_path}: {e}"
            raise ConfigurationError(msg)

    def get_nested_value(self, data: dict, key_path: str, default=None):
        """Get nested dictionary value using dot notation."""
        keys = key_path.split(".")
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def set_nested_value(self, data: dict, key_path: str, value: Any) -> None:
        """Set nested dictionary value using dot notation."""
        keys = key_path.split(".")
        current = data

        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def apply_environment_overrides(self, config: dict[str, Any]) -> dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        config_copy = copy.deepcopy(config)

        for env_var, config_path in self.env_mappings.items():
            env_value = os.getenv(env_var)

            if env_value is not None:
                # Type conversion based on existing config value
                existing_value = self.get_nested_value(config_copy, config_path)

                if existing_value is not None:
                    if isinstance(existing_value, bool):
                        converted_value = env_value.lower() in (
                            "true",
                            "1",
                            "yes",
                            "on",
                        )
                    elif isinstance(existing_value, int):
                        converted_value = int(env_value)
                    elif isinstance(existing_value, float):
                        converted_value = float(env_value)
                    else:
                        converted_value = env_value
                # Attempt intelligent type conversion
                elif env_value.lower() in ("true", "false"):
                    converted_value = env_value.lower() == "true"
                elif env_value.isdigit():
                    converted_value = int(env_value)
                elif env_value.replace(".", "").isdigit():
                    converted_value = float(env_value)
                else:
                    converted_value = env_value

                self.set_nested_value(config_copy, config_path, converted_value)
                logger.info(f"Applied environment override: {env_var} -> {config_path} = {converted_value}")

        return config_copy

    def reload_config(self) -> None:
        """Reload all configuration files and apply overrides."""
        with self.lock:
            try:
                logger.info("Reloading configuration...")

                # Load all configuration files
                main_config = self.load_yaml_file(self.main_config)
                p2p_config = self.load_json_file(self.p2p_config)
                rag_config = self.load_json_file(self.rag_config)

                # Merge configurations
                merged_config = {
                    **main_config,
                    "p2p_config": p2p_config,
                    "rag_config": rag_config,
                }

                # Apply environment variable overrides
                final_config = self.apply_environment_overrides(merged_config)

                # Validate configuration
                self.validate_configuration(final_config)

                # Update instance configuration
                self.config_data = final_config
                self.last_reload = datetime.now()

                # Update watched files
                self.watched_files = {
                    str(self.main_config),
                    str(self.p2p_config),
                    str(self.rag_config),
                }

                logger.info("Configuration reloaded successfully")

            except Exception as e:
                logger.exception(f"Failed to reload configuration: {e}")
                msg = f"Configuration reload failed: {e}"
                raise ConfigurationError(msg)

    def validate_configuration(self, config: dict[str, Any]) -> None:
        """Validate configuration consistency and requirements."""
        errors = []
        warnings = []

        # Validate main integration settings
        integration = config.get("integration", {})

        # Evolution metrics validation
        if integration.get("evolution_metrics", {}).get("enabled"):
            evo_config = integration["evolution_metrics"]
            db_path = evo_config.get("db_path", "./data/evolution_metrics.db")

            # Check if database path parent directory exists
            db_path_obj = Path(db_path)
            if not db_path_obj.parent.exists():
                warnings.append(f"Evolution metrics database directory does not exist: {db_path_obj.parent}")

        # RAG pipeline validation
        if integration.get("rag_pipeline", {}).get("enabled"):
            rag_integration = integration["rag_pipeline"]
            embedding_model = rag_integration.get("embedding_model")

            if not embedding_model:
                errors.append("RAG pipeline enabled but no embedding_model specified")

            # Validate RAG config structure
            rag_config = config.get("rag_config", {})
            if not rag_config.get("embedder", {}).get("model_name"):
                errors.append("RAG config missing embedder.model_name")

            cache_dir = rag_config.get("cache", {}).get("l3_directory")
            if cache_dir:
                cache_path = Path(cache_dir)
                if not cache_path.parent.exists():
                    warnings.append(f"RAG cache directory parent does not exist: {cache_path.parent}")

        # P2P networking validation
        if integration.get("p2p_networking", {}).get("enabled"):
            p2p_config = config.get("p2p_config", {})

            if not p2p_config.get("host"):
                errors.append("P2P networking enabled but no host specified")

            port = p2p_config.get("port")
            if not port or not isinstance(port, int) or port <= 0 or port > 65535:
                errors.append(f"Invalid P2P port: {port}")

            # Check required P2P settings match CODEX requirements
            expected_port = 4001
            if port != expected_port:
                warnings.append(f"P2P port {port} differs from CODEX requirement: {expected_port}")

        # Digital twin validation
        if integration.get("digital_twin", {}).get("enabled"):
            dt_config = integration["digital_twin"]

            if dt_config.get("encryption_enabled") and not dt_config.get("encryption_key"):
                errors.append("Digital twin encryption enabled but no encryption_key provided")

            max_profiles = dt_config.get("max_profiles", 10000)
            if not isinstance(max_profiles, int) or max_profiles <= 0:
                errors.append(f"Invalid digital_twin.max_profiles: {max_profiles}")

        # Port conflict checking
        ports_in_use = []
        if config.get("p2p_config", {}).get("port"):
            ports_in_use.append(("P2P", config["p2p_config"]["port"]))

        # Check for port conflicts
        seen_ports = set()
        for _service, port in ports_in_use:
            if port in seen_ports:
                errors.append(f"Port conflict: {port} used by multiple services")
            seen_ports.add(port)

        # Log validation results
        if errors:
            error_msg = f"Configuration validation failed: {'; '.join(errors)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)

        if warnings:
            for warning in warnings:
                logger.warning(f"Configuration warning: {warning}")

        logger.info("Configuration validation passed")

    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation."""
        with self.lock:
            return self.get_nested_value(self.config_data, key_path, default)

    def get_all(self) -> dict[str, Any]:
        """Get complete configuration dictionary."""
        with self.lock:
            return copy.deepcopy(self.config_data)

    def is_enabled(self, component: str) -> bool:
        """Check if a component is enabled."""
        return self.get(f"integration.{component}.enabled", False)

    def start_file_watcher(self) -> None:
        """Start file system watcher for hot reload."""
        if self.observer is not None:
            return

        try:
            self.observer = Observer()
            handler = ConfigWatcher(self)

            # Watch the config directory
            self.observer.schedule(handler, str(self.config_dir), recursive=False)
            self.observer.start()

            logger.info(f"Started configuration file watcher on {self.config_dir}")

        except Exception as e:
            logger.exception(f"Failed to start file watcher: {e}")
            self.observer = None

    def stop_file_watcher(self) -> None:
        """Stop file system watcher."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("Stopped configuration file watcher")

    def get_config_sources(self) -> dict[str, Any]:
        """Get information about configuration sources."""
        return {
            "config_dir": str(self.config_dir),
            "main_config": str(self.main_config),
            "p2p_config": str(self.p2p_config),
            "rag_config": str(self.rag_config),
            "last_reload": self.last_reload.isoformat(),
            "hot_reload_enabled": self.enable_hot_reload,
            "watched_files": list(self.watched_files),
            "environment_overrides_applied": len(
                [env_var for env_var in self.env_mappings if os.getenv(env_var) is not None]
            ),
        }

    def export_effective_config(self, output_path: str | None = None) -> dict[str, Any]:
        """Export the effective configuration after all overrides."""
        effective_config = {
            "timestamp": datetime.now().isoformat(),
            "sources": self.get_config_sources(),
            "configuration": self.get_all(),
        }

        if output_path:
            output_file = Path(output_path)
            with open(output_file, "w", encoding="utf-8") as f:
                if output_file.suffix == ".yaml":
                    yaml.dump(effective_config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(effective_config, f, indent=2)
            logger.info(f"Exported effective configuration to: {output_file}")

        return effective_config

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup watchers."""
        self.stop_file_watcher()


# Global configuration instance
_config_manager = None


def get_config_manager(**kwargs) -> CODEXConfigManager:
    """Get or create the global configuration manager."""
    global _config_manager

    if _config_manager is None:
        _config_manager = CODEXConfigManager(**kwargs)

    return _config_manager


def reload_global_config() -> None:
    """Reload the global configuration."""
    global _config_manager

    if _config_manager:
        _config_manager.reload_config()
    else:
        _config_manager = CODEXConfigManager()


def get_config(key_path: str, default=None):
    """Get configuration value from global config manager."""
    return get_config_manager().get(key_path, default)


def is_component_enabled(component: str) -> bool:
    """Check if a component is enabled in global config."""
    return get_config_manager().is_enabled(component)
