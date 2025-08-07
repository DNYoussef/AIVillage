#!/usr/bin/env python3
"""Base class for all AIVillage scripts.

This module provides a standardized base class that all AIVillage scripts
should inherit from. It provides:
- Standardized lifecycle management
- Built-in error handling and recovery
- Metrics collection integration
- Configuration management
- Resource monitoring
"""

from abc import ABC, abstractmethod
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any

from .common_utils import (
    ResourceMonitor,
    ScriptTimeoutError,
    format_duration,
    get_system_info,
    handle_errors,
    monitor_resources,
    setup_logging,
    timeout_handler,
)
from .config_manager import ConfigManager


class ScriptResult:
    """Container for script execution results."""

    def __init__(
        self,
        success: bool = True,
        message: str = "",
        data: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
    ):
        """Initialize script result.

        Args:
            success: Whether the script execution was successful
            message: Summary message
            data: Result data
            metrics: Performance metrics
            errors: List of error messages
            warnings: List of warning messages
        """
        self.success = success
        self.message = message
        self.data = data or {}
        self.metrics = metrics or {}
        self.errors = errors or []
        self.warnings = warnings or []
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary.

        Returns:
            Dictionary representation of the result
        """
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "metrics": self.metrics,
            "errors": self.errors,
            "warnings": self.warnings,
            "timestamp": self.timestamp,
        }

    def save_to_file(self, file_path: str | Path) -> None:
        """Save result to JSON file.

        Args:
            file_path: Path to save the result file
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class BaseScript(ABC):
    """Base class for all AIVillage scripts."""

    def __init__(
        self,
        name: str,
        description: str = "",
        config_name: str | None = None,
        log_level: int | str = logging.INFO,
        log_file: str | Path | None = None,
        timeout: int | None = None,
    ):
        """Initialize the base script.

        Args:
            name: Script name
            description: Script description
            config_name: Configuration file name (defaults to script name)
            log_level: Logging level
            log_file: Log file path
            timeout: Script timeout in seconds
        """
        self.name = name
        self.description = description
        self.config_name = config_name or name
        self.timeout = timeout

        # Set up logging
        self.logger = setup_logging(
            level=log_level,
            log_file=log_file,
            include_console=True,
        )

        # Initialize configuration manager
        self.config_manager = ConfigManager()

        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor()

        # Script state
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.result: ScriptResult | None = None
        self.dry_run = False

        self.logger.info(f"Initialized script: {self.name}")

    @abstractmethod
    def execute(self) -> ScriptResult:
        """Execute the main script logic.

        This method must be implemented by subclasses.

        Returns:
            ScriptResult with execution results
        """

    def pre_execute(self) -> None:
        """Pre-execution hook for setup tasks.

        Override this method to add custom pre-execution logic.
        """
        self.logger.info(f"Starting pre-execution setup for {self.name}")

        # Load configuration
        try:
            self.config = self.config_manager.load_config(self.config_name)
            self.logger.debug("Configuration loaded successfully")
        except Exception as e:
            self.logger.warning(f"Failed to load configuration: {e}")
            self.config = {}

        # Log system information
        system_info = get_system_info()
        self.logger.debug(f"System info: {system_info}")

    def post_execute(self, result: ScriptResult) -> None:
        """Post-execution hook for cleanup tasks.

        Override this method to add custom post-execution logic.

        Args:
            result: Script execution result
        """
        self.logger.info(f"Starting post-execution cleanup for {self.name}")

        # Add resource metrics to result
        if hasattr(self, "resource_monitor"):
            resource_summary = self.resource_monitor.get_summary()
            result.metrics.update(resource_summary)

        # Log execution summary
        execution_time = result.metrics.get("total_execution_time", 0)
        self.logger.info(
            f"Script {self.name} completed - "
            f"Success: {result.success}, "
            f"Duration: {format_duration(execution_time)}, "
            f"Errors: {len(result.errors)}, "
            f"Warnings: {len(result.warnings)}"
        )

    def handle_error(self, error: Exception) -> ScriptResult:
        """Handle script errors.

        Override this method to add custom error handling logic.

        Args:
            error: The exception that occurred

        Returns:
            ScriptResult with error information
        """
        error_message = f"Script {self.name} failed: {error}"
        self.logger.error(error_message)

        return ScriptResult(
            success=False,
            message=error_message,
            errors=[str(error)],
            metrics=(
                self.resource_monitor.get_summary()
                if hasattr(self, "resource_monitor")
                else {}
            ),
        )

    def validate_configuration(self) -> list[str]:
        """Validate script configuration.

        Override this method to add custom configuration validation.

        Returns:
            List of validation error messages (empty if valid)
        """
        return []

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.config_manager.get_config_value(self.config_name, key, default)

    def set_config_value(self, key: str, value: Any) -> None:
        """Set configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        self.config_manager.set_config_value(self.config_name, key, value)

    @handle_errors(exit_on_error=False)
    @monitor_resources(log_interval=30)
    def run(
        self,
        dry_run: bool = False,
        save_result: bool = True,
        result_file: str | Path | None = None,
    ) -> ScriptResult:
        """Run the complete script lifecycle.

        Args:
            dry_run: Whether to perform a dry run
            save_result: Whether to save results to file
            result_file: Path to save results (defaults to {name}_result.json)

        Returns:
            ScriptResult with execution results
        """
        self.dry_run = dry_run
        self.start_time = time.time()
        self.resource_monitor.start()

        try:
            # Validate configuration
            config_errors = self.validate_configuration()
            if config_errors:
                return ScriptResult(
                    success=False,
                    message="Configuration validation failed",
                    errors=config_errors,
                )

            # Pre-execution setup
            self.pre_execute()

            # Execute main logic with timeout if configured
            if self.timeout:

                @timeout_handler(self.timeout)
                def execute_with_timeout():
                    return self.execute()

                result = execute_with_timeout()
            else:
                result = self.execute()

            # Post-execution cleanup
            self.post_execute(result)

            self.result = result

        except ScriptTimeoutError as e:
            self.logger.error(f"Script timed out: {e}")
            result = ScriptResult(
                success=False,
                message=f"Script timed out after {self.timeout} seconds",
                errors=[str(e)],
                metrics=self.resource_monitor.get_summary(),
            )
            self.result = result

        except Exception as e:
            result = self.handle_error(e)
            self.result = result

        finally:
            self.end_time = time.time()

            # Ensure metrics are populated
            if not result.metrics:
                result.metrics = self.resource_monitor.get_summary()

            # Save results if requested
            if save_result:
                if result_file is None:
                    result_file = f"{self.name}_result.json"
                result.save_to_file(result_file)
                self.logger.info(f"Results saved to {result_file}")

        return result

    def get_status(self) -> dict[str, Any]:
        """Get current script status.

        Returns:
            Dictionary with script status information
        """
        status = {
            "name": self.name,
            "description": self.description,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "running": self.start_time is not None and self.end_time is None,
            "dry_run": self.dry_run,
        }

        if self.start_time:
            if self.end_time:
                status["duration"] = self.end_time - self.start_time
            else:
                status["duration"] = time.time() - self.start_time

        if self.result:
            status["result"] = {
                "success": self.result.success,
                "message": self.result.message,
                "errors": len(self.result.errors),
                "warnings": len(self.result.warnings),
            }

        # Add resource usage if available
        if hasattr(self, "resource_monitor") and self.resource_monitor.start_time:
            try:
                resource_stats = self.resource_monitor.update()
                status["resources"] = resource_stats
            except Exception as e:
                self.logger.debug(f"Failed to get resource stats: {e}")

        return status


class SimpleScript(BaseScript):
    """Simple script implementation for quick scripts.

    This class allows creating scripts without subclassing by passing
    the execution function as a parameter.
    """

    def __init__(
        self, name: str, execute_func: callable, description: str = "", **kwargs
    ):
        """Initialize simple script.

        Args:
            name: Script name
            execute_func: Function to execute (should return ScriptResult)
            description: Script description
            **kwargs: Additional arguments for BaseScript
        """
        super().__init__(name, description, **kwargs)
        self.execute_func = execute_func

    def execute(self) -> ScriptResult:
        """Execute the provided function.

        Returns:
            ScriptResult from the provided function
        """
        return self.execute_func()


# Utility function for creating simple scripts
def create_script(
    name: str, execute_func: callable, description: str = "", **kwargs
) -> SimpleScript:
    """Create a simple script.

    Args:
        name: Script name
        execute_func: Function to execute
        description: Script description
        **kwargs: Additional arguments for BaseScript

    Returns:
        SimpleScript instance
    """
    return SimpleScript(name, execute_func, description, **kwargs)
