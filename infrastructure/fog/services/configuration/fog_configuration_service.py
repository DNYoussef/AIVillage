"""
Fog Configuration Service

Manages configuration across the fog computing system including:
- Centralized configuration management
- Dynamic configuration updates
- Configuration validation
- Environment-specific settings
"""

import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import yaml

from ..interfaces.base_service import BaseFogService, ServiceHealthCheck, ServiceStatus


class FogConfigurationService(BaseFogService):
    """Service for managing fog computing system configuration"""

    def __init__(self, service_name: str, config: dict[str, Any], event_bus):
        super().__init__(service_name, config, event_bus)

        # Configuration management
        self.config_sources: dict[str, dict[str, Any]] = {}
        self.active_config: dict[str, Any] = config.copy()
        self.config_schema: dict[str, Any] = {}
        self.config_watchers: dict[str, list[callable]] = {}

        # Configuration file paths
        self.config_paths = {
            "main": config.get("config_path"),
            "schema": config.get("schema_path"),
            "overrides": config.get("overrides_path"),
        }

        # Service metrics
        self.metrics = {
            "config_updates": 0,
            "validation_errors": 0,
            "config_sources_count": 0,
            "active_watchers": 0,
            "last_update_timestamp": None,
            "schema_violations": 0,
        }

    async def initialize(self) -> bool:
        """Initialize the configuration service"""
        try:
            # Load configuration schema
            await self._load_configuration_schema()

            # Load configuration from all sources
            await self._load_all_configurations()

            # Validate loaded configuration
            is_valid = await self._validate_configuration(self.active_config)
            if not is_valid:
                self.logger.error("Initial configuration validation failed")
                return False

            # Subscribe to configuration events
            self.subscribe_to_events("config_update_request", self._handle_config_update)
            self.subscribe_to_events("config_reload_request", self._handle_config_reload)
            self.subscribe_to_events("service_config_request", self._handle_service_config_request)

            # Start background tasks
            self.add_background_task(self._config_file_watcher_task(), "file_watcher")
            self.add_background_task(self._config_validation_task(), "validation")

            self.logger.info("Fog configuration service initialized")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize configuration service: {e}")
            return False

    async def cleanup(self) -> bool:
        """Cleanup configuration service resources"""
        try:
            # Save current configuration if needed
            if self.config.get("save_on_exit", False):
                await self._save_configuration()

            self.logger.info("Fog configuration service cleaned up")
            return True

        except Exception as e:
            self.logger.error(f"Error cleaning up configuration service: {e}")
            return False

    async def health_check(self) -> ServiceHealthCheck:
        """Perform health check on configuration service"""
        try:
            error_messages = []

            # Check configuration validity
            if not await self._validate_configuration(self.active_config):
                error_messages.append("Active configuration is invalid")

            # Check for high validation error rate
            total_updates = self.metrics["config_updates"] + self.metrics["validation_errors"]
            if total_updates > 0:
                error_rate = self.metrics["validation_errors"] / total_updates
                if error_rate > 0.1:  # More than 10% error rate
                    error_messages.append(f"High configuration error rate: {error_rate:.2%}")

            # Check if required config files exist
            for name, path in self.config_paths.items():
                if path and not Path(path).exists():
                    error_messages.append(f"Missing config file: {name} at {path}")

            status = ServiceStatus.RUNNING if not error_messages else ServiceStatus.ERROR

            return ServiceHealthCheck(
                service_name=self.service_name,
                status=status,
                last_check=datetime.now(UTC),
                error_message="; ".join(error_messages) if error_messages else None,
                metrics=self.metrics.copy(),
            )

        except Exception as e:
            return ServiceHealthCheck(
                service_name=self.service_name,
                status=ServiceStatus.ERROR,
                last_check=datetime.now(UTC),
                error_message=f"Health check failed: {e}",
                metrics=self.metrics.copy(),
            )

    async def get_configuration(self, key_path: str | None = None) -> Any:
        """Get configuration value by key path"""
        try:
            if key_path is None:
                return self.active_config.copy()

            # Navigate nested configuration using dot notation
            keys = key_path.split(".")
            value = self.active_config

            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None

            return value

        except Exception as e:
            self.logger.error(f"Failed to get configuration for {key_path}: {e}")
            return None

    async def update_configuration(
        self, key_path: str, value: Any, validate: bool = True, persist: bool = True
    ) -> bool:
        """Update configuration value"""
        try:
            # Create backup of current config
            backup_config = self.active_config.copy()

            # Update the configuration
            keys = key_path.split(".")
            current = self.active_config

            # Navigate to the parent of the target key
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Set the value
            current[keys[-1]] = value

            # Validate if requested
            if validate:
                is_valid = await self._validate_configuration(self.active_config)
                if not is_valid:
                    # Restore backup
                    self.active_config = backup_config
                    self.metrics["validation_errors"] += 1
                    return False

            # Update metrics
            self.metrics["config_updates"] += 1
            self.metrics["last_update_timestamp"] = datetime.now(UTC).isoformat()

            # Persist if requested
            if persist:
                await self._save_configuration()

            # Notify watchers
            await self._notify_config_watchers(key_path, value)

            # Publish configuration update event
            await self.publish_event(
                "configuration_updated",
                {"key_path": key_path, "value": value, "timestamp": datetime.now(UTC).isoformat()},
            )

            self.logger.info(f"Updated configuration: {key_path} = {value}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            # Restore backup if something went wrong
            if "backup_config" in locals():
                self.active_config = backup_config
            return False

    async def add_config_watcher(self, key_path: str, callback: callable):
        """Add a watcher for configuration changes"""
        try:
            if key_path not in self.config_watchers:
                self.config_watchers[key_path] = []

            self.config_watchers[key_path].append(callback)
            self.metrics["active_watchers"] = sum(len(watchers) for watchers in self.config_watchers.values())

            self.logger.debug(f"Added config watcher for {key_path}")

        except Exception as e:
            self.logger.error(f"Failed to add config watcher: {e}")

    async def reload_configuration(self) -> bool:
        """Reload configuration from all sources"""
        try:
            # Backup current config
            backup_config = self.active_config.copy()

            # Reload from sources
            success = await self._load_all_configurations()

            if success:
                # Validate reloaded configuration
                is_valid = await self._validate_configuration(self.active_config)
                if not is_valid:
                    # Restore backup
                    self.active_config = backup_config
                    return False

                # Publish reload event
                await self.publish_event("configuration_reloaded", {"timestamp": datetime.now(UTC).isoformat()})

                self.logger.info("Configuration reloaded successfully")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {e}")
            return False

    async def get_service_configuration(self, service_name: str) -> dict[str, Any]:
        """Get configuration specific to a service"""
        try:
            service_config = {}

            # Get global configuration
            service_config.update(self.active_config.get("global", {}))

            # Get service-specific configuration
            services_config = self.active_config.get("services", {})
            if service_name in services_config:
                service_config.update(services_config[service_name])

            return service_config

        except Exception as e:
            self.logger.error(f"Failed to get service configuration for {service_name}: {e}")
            return {}

    async def get_configuration_stats(self) -> dict[str, Any]:
        """Get configuration service statistics"""
        try:
            stats = self.metrics.copy()

            stats.update(
                {
                    "config_sources": list(self.config_sources.keys()),
                    "config_size_bytes": len(json.dumps(self.active_config).encode()),
                    "schema_loaded": bool(self.config_schema),
                    "watchers_by_path": {path: len(watchers) for path, watchers in self.config_watchers.items()},
                }
            )

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get configuration stats: {e}")
            return self.metrics.copy()

    async def _load_configuration_schema(self):
        """Load configuration schema for validation"""
        try:
            schema_path = self.config_paths.get("schema")
            if schema_path and Path(schema_path).exists():
                with open(schema_path) as f:
                    if schema_path.endswith(".yaml") or schema_path.endswith(".yml"):
                        self.config_schema = yaml.safe_load(f)
                    else:
                        self.config_schema = json.load(f)

                self.logger.info(f"Loaded configuration schema from {schema_path}")

        except Exception as e:
            self.logger.error(f"Failed to load configuration schema: {e}")

    async def _load_all_configurations(self) -> bool:
        """Load configuration from all sources"""
        try:
            # Load main configuration
            main_path = self.config_paths.get("main")
            if main_path and Path(main_path).exists():
                with open(main_path) as f:
                    if main_path.endswith(".yaml") or main_path.endswith(".yml"):
                        main_config = yaml.safe_load(f)
                    else:
                        main_config = json.load(f)

                self.config_sources["main"] = main_config
                self.active_config.update(main_config)

            # Load override configuration
            overrides_path = self.config_paths.get("overrides")
            if overrides_path and Path(overrides_path).exists():
                with open(overrides_path) as f:
                    if overrides_path.endswith(".yaml") or overrides_path.endswith(".yml"):
                        override_config = yaml.safe_load(f)
                    else:
                        override_config = json.load(f)

                self.config_sources["overrides"] = override_config
                self._deep_merge(self.active_config, override_config)

            # Load environment variables
            env_config = self._load_environment_config()
            if env_config:
                self.config_sources["environment"] = env_config
                self._deep_merge(self.active_config, env_config)

            self.metrics["config_sources_count"] = len(self.config_sources)
            return True

        except Exception as e:
            self.logger.error(f"Failed to load configurations: {e}")
            return False

    async def _validate_configuration(self, config: dict[str, Any]) -> bool:
        """Validate configuration against schema"""
        try:
            if not self.config_schema:
                return True  # No schema to validate against

            # Simple validation - in production would use jsonschema
            # This is a mock validation
            required_keys = self.config_schema.get("required", [])

            for key in required_keys:
                if key not in config:
                    self.metrics["schema_violations"] += 1
                    self.logger.error(f"Missing required configuration key: {key}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False

    def _load_environment_config(self) -> dict[str, Any]:
        """Load configuration from environment variables"""
        try:
            import os

            env_config = {}

            # Look for FOG_ prefixed environment variables
            for key, value in os.environ.items():
                if key.startswith("FOG_"):
                    config_key = key[4:].lower().replace("_", ".")

                    # Try to parse as JSON, fall back to string
                    try:
                        parsed_value = json.loads(value)
                    except:
                        parsed_value = value

                    # Set nested configuration
                    keys = config_key.split(".")
                    current = env_config
                    for nested_key in keys[:-1]:
                        if nested_key not in current:
                            current[nested_key] = {}
                        current = current[nested_key]
                    current[keys[-1]] = parsed_value

            return env_config

        except Exception as e:
            self.logger.error(f"Failed to load environment configuration: {e}")
            return {}

    def _deep_merge(self, target: dict[str, Any], source: dict[str, Any]):
        """Deep merge source into target dictionary"""
        try:
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    self._deep_merge(target[key], value)
                else:
                    target[key] = value
        except Exception as e:
            self.logger.error(f"Deep merge error: {e}")

    async def _save_configuration(self):
        """Save current configuration to file"""
        try:
            main_path = self.config_paths.get("main")
            if main_path:
                with open(main_path, "w") as f:
                    if main_path.endswith(".yaml") or main_path.endswith(".yml"):
                        yaml.dump(self.active_config, f, indent=2)
                    else:
                        json.dump(self.active_config, f, indent=2)

                self.logger.debug(f"Saved configuration to {main_path}")

        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")

    async def _notify_config_watchers(self, key_path: str, value: Any):
        """Notify configuration watchers of changes"""
        try:
            # Notify exact path watchers
            if key_path in self.config_watchers:
                for callback in self.config_watchers[key_path]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(key_path, value)
                        else:
                            callback(key_path, value)
                    except Exception as e:
                        self.logger.error(f"Config watcher error: {e}")

            # Notify wildcard watchers
            parts = key_path.split(".")
            for i in range(len(parts)):
                wildcard_path = ".".join(parts[: i + 1]) + ".*"
                if wildcard_path in self.config_watchers:
                    for callback in self.config_watchers[wildcard_path]:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(key_path, value)
                            else:
                                callback(key_path, value)
                        except Exception as e:
                            self.logger.error(f"Config watcher error: {e}")

        except Exception as e:
            self.logger.error(f"Failed to notify config watchers: {e}")

    async def _handle_config_update(self, event):
        """Handle configuration update requests"""
        key_path = event.data.get("key_path")
        value = event.data.get("value")
        validate = event.data.get("validate", True)
        persist = event.data.get("persist", True)

        success = await self.update_configuration(key_path, value, validate, persist)

        await self.publish_event(
            "config_update_response",
            {"request_id": event.data.get("request_id"), "success": success, "key_path": key_path},
        )

    async def _handle_config_reload(self, event):
        """Handle configuration reload requests"""
        success = await self.reload_configuration()

        await self.publish_event(
            "config_reload_response", {"request_id": event.data.get("request_id"), "success": success}
        )

    async def _handle_service_config_request(self, event):
        """Handle service configuration requests"""
        service_name = event.data.get("service_name")
        service_config = await self.get_service_configuration(service_name)

        await self.publish_event(
            "service_config_response",
            {"request_id": event.data.get("request_id"), "service_name": service_name, "config": service_config},
        )

    async def _config_file_watcher_task(self):
        """Background task to watch configuration files for changes"""
        while not self._shutdown_event.is_set():
            try:
                # Check if any config files have been modified
                # This is a simplified version - production would use file system events
                config_modified = False

                for name, path in self.config_paths.items():
                    if path and Path(path).exists():
                        # Check modification time (simplified)
                        # In production, would use watchdog or similar
                        pass

                if config_modified:
                    await self.reload_configuration()

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Config file watcher error: {e}")
                await asyncio.sleep(60)

    async def _config_validation_task(self):
        """Background task to periodically validate configuration"""
        while not self._shutdown_event.is_set():
            try:
                # Validate current configuration
                is_valid = await self._validate_configuration(self.active_config)

                if not is_valid:
                    await self.publish_event("configuration_invalid", {"timestamp": datetime.now(UTC).isoformat()})

                await asyncio.sleep(300)  # Validate every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Config validation task error: {e}")
                await asyncio.sleep(120)
