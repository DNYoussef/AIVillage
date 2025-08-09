"""AIVillage Debug Manager

Centralized debug mode management following CODEX Integration Requirements.
"""

from dataclasses import dataclass
from datetime import datetime
import logging
import os
import threading
import time
from typing import Any, Optional

import psutil


@dataclass
class DebugConfig:
    """Debug configuration settings."""
    log_level: str = "DEBUG"
    verbose_output: bool = True
    profile_performance: bool = True
    track_resources: bool = True
    debug_endpoints: bool = True
    dashboard_enabled: bool = True

    # CODEX Integration Requirements environment variables
    aivillage_debug_mode: bool = False
    aivillage_log_level: str = "INFO"
    aivillage_profile_performance: bool = False

class DebugManager:
    """Centralized debug management system.
    
    Manages debug flags, logging configuration, and troubleshooting tools
    according to CODEX Integration Requirements.
    """

    _instance: Optional["DebugManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "DebugManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
        self._debug_data: dict[str, Any] = {}
        self._monitoring_active = False
        self._monitoring_thread: threading.Thread | None = None

        # Initialize debug mode
        self._setup_debug_environment()

    def _load_config(self) -> DebugConfig:
        """Load debug configuration from environment variables."""
        config = DebugConfig()

        # CODEX required environment variables
        config.aivillage_debug_mode = os.getenv("AIVILLAGE_DEBUG_MODE", "false").lower() == "true"
        config.aivillage_log_level = os.getenv("AIVILLAGE_LOG_LEVEL", "INFO")
        config.aivillage_profile_performance = os.getenv("AIVILLAGE_PROFILE_PERFORMANCE", "false").lower() == "true"

        # Set derived configuration
        if config.aivillage_debug_mode:
            config.log_level = "DEBUG"
            config.verbose_output = True
            config.profile_performance = True
            config.track_resources = True
            config.debug_endpoints = True
            config.dashboard_enabled = True
        else:
            config.log_level = config.aivillage_log_level
            config.verbose_output = False
            config.profile_performance = config.aivillage_profile_performance

        return config

    def _setup_debug_environment(self):
        """Set up debug environment variables and configuration."""
        if self.config.aivillage_debug_mode:
            # Set all CODEX required debug environment variables
            os.environ["AIVILLAGE_LOG_LEVEL"] = "DEBUG"
            os.environ["AIVILLAGE_DEBUG_MODE"] = "true"
            os.environ["AIVILLAGE_PROFILE_PERFORMANCE"] = "true"

            # Configure logging
            logging.getLogger().setLevel(logging.DEBUG)

            self.logger.info("Debug mode activated - all debugging features enabled")
            self.logger.debug(f"Debug configuration: {self.config}")

    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled."""
        return self.config.aivillage_debug_mode

    def is_verbose_enabled(self) -> bool:
        """Check if verbose output is enabled."""
        return self.config.verbose_output

    def is_profiling_enabled(self) -> bool:
        """Check if performance profiling is enabled."""
        return self.config.profile_performance

    def log_debug_info(self, component: str, message: str, data: dict[str, Any] | None = None):
        """Log debug information for a specific component."""
        if not self.is_debug_enabled():
            return

        timestamp = datetime.now().isoformat()
        debug_entry = {
            "timestamp": timestamp,
            "component": component,
            "message": message,
            "data": data or {}
        }

        # Store debug data
        if component not in self._debug_data:
            self._debug_data[component] = []
        self._debug_data[component].append(debug_entry)

        # Log the information
        if self.is_verbose_enabled():
            self.logger.debug(f"[{component}] {message}")
            if data:
                for key, value in data.items():
                    self.logger.debug(f"  {key}: {value}")

    def log_request_response(self, endpoint: str, request_data: Any, response_data: Any, duration_ms: float):
        """Log request/response data for debugging API calls."""
        if not self.is_debug_enabled():
            return

        debug_info = {
            "endpoint": endpoint,
            "request": str(request_data)[:1000] if request_data else None,  # Truncate large payloads
            "response": str(response_data)[:1000] if response_data else None,
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat()
        }

        self.log_debug_info("api_requests", f"API call to {endpoint}", debug_info)

    def log_database_query(self, query: str, params: Any, duration_ms: float, result_count: int):
        """Log database queries with timing information."""
        if not self.is_debug_enabled():
            return

        debug_info = {
            "query": query[:500],  # Truncate very long queries
            "params": str(params)[:200] if params else None,
            "duration_ms": duration_ms,
            "result_count": result_count,
            "timestamp": datetime.now().isoformat()
        }

        self.log_debug_info("database", f"Query executed in {duration_ms:.2f}ms", debug_info)

    def log_cache_operation(self, operation: str, key: str, hit: bool, duration_ms: float | None = None):
        """Log cache operations for debugging cache performance."""
        if not self.is_debug_enabled():
            return

        debug_info = {
            "operation": operation,
            "key": key[:100],  # Truncate long keys
            "cache_hit": hit,
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat()
        }

        status = "HIT" if hit else "MISS"
        self.log_debug_info("cache", f"Cache {operation} ({status}): {key}", debug_info)

    def start_resource_monitoring(self, interval_seconds: int = 10):
        """Start monitoring system resources."""
        if not self.is_debug_enabled() or self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitor_resources,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitoring_thread.start()

        self.logger.info(f"Started resource monitoring (interval: {interval_seconds}s)")

    def stop_resource_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)

        self.logger.info("Stopped resource monitoring")

    def _monitor_resources(self, interval_seconds: int):
        """Monitor system resources in background thread."""
        while self._monitoring_active:
            try:
                # Get system resource information
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage("/")

                resource_info = {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_mb": memory.used / (1024 * 1024),
                    "memory_available_mb": memory.available / (1024 * 1024),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024 * 1024 * 1024)
                }

                self.log_debug_info("resources", "System resource usage", resource_info)

                # Check for resource warnings
                if cpu_percent > 80:
                    self.logger.warning(f"High CPU usage detected: {cpu_percent:.1f}%")

                if memory.percent > 85:
                    self.logger.warning(f"High memory usage detected: {memory.percent:.1f}%")

                if disk.percent > 90:
                    self.logger.warning(f"Low disk space detected: {disk.percent:.1f}% used")

                time.sleep(interval_seconds)

            except Exception as e:
                self.logger.error(f"Error during resource monitoring: {e}")
                time.sleep(interval_seconds)

    def get_debug_summary(self) -> dict[str, Any]:
        """Get summary of debug information."""
        summary = {
            "debug_enabled": self.is_debug_enabled(),
            "config": self.config.__dict__,
            "components_tracked": list(self._debug_data.keys()),
            "total_debug_entries": sum(len(entries) for entries in self._debug_data.values()),
            "monitoring_active": self._monitoring_active,
            "environment_variables": {
                "AIVILLAGE_DEBUG_MODE": os.getenv("AIVILLAGE_DEBUG_MODE"),
                "AIVILLAGE_LOG_LEVEL": os.getenv("AIVILLAGE_LOG_LEVEL"),
                "AIVILLAGE_PROFILE_PERFORMANCE": os.getenv("AIVILLAGE_PROFILE_PERFORMANCE")
            }
        }

        # Add component summaries
        component_summaries = {}
        for component, entries in self._debug_data.items():
            component_summaries[component] = {
                "entry_count": len(entries),
                "latest_entry": entries[-1]["timestamp"] if entries else None,
                "first_entry": entries[0]["timestamp"] if entries else None
            }

        summary["component_summaries"] = component_summaries

        return summary

    def get_debug_data(self, component: str | None = None, limit: int = 100) -> dict[str, Any]:
        """Get debug data for analysis."""
        if component:
            return {
                component: self._debug_data.get(component, [])[-limit:]
            }

        # Return all components with limit applied
        result = {}
        for comp, entries in self._debug_data.items():
            result[comp] = entries[-limit:]

        return result

    def clear_debug_data(self, component: str | None = None):
        """Clear debug data."""
        if component and component in self._debug_data:
            self._debug_data[component].clear()
            self.logger.info(f"Cleared debug data for component: {component}")
        else:
            self._debug_data.clear()
            self.logger.info("Cleared all debug data")


# Global debug manager instance
_debug_manager: DebugManager | None = None

def get_debug_manager() -> DebugManager:
    """Get the global debug manager instance."""
    global _debug_manager
    if _debug_manager is None:
        _debug_manager = DebugManager()
    return _debug_manager
