#!/usr/bin/env python3
"""Common utilities for AIVillage scripts.

This module provides shared functionality including:
- Standardized logging setup
- Common argument parsing patterns
- Error handling decorators
- Resource monitoring utilities
- Performance measurement tools
"""

import argparse
import functools
import logging
import os
import signal
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn, TypeVar

import psutil

from common.logging import setup_logging

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


class ScriptError(Exception):
    """Base exception for script errors."""


class ScriptTimeoutError(ScriptError):
    """Raised when script execution times out."""


class ResourceMonitor:
    """Monitor system resources during script execution."""

    def __init__(self) -> None:
        """Initialize the resource monitor."""
        self.start_time: float | None = None
        self.peak_memory: float = 0.0
        self.peak_cpu: float = 0.0
        self.initial_memory: float = 0.0

    def start(self) -> None:
        """Start resource monitoring."""
        self.start_time = time.time()
        self.initial_memory = psutil.virtual_memory().used / (1024**3)  # GB
        self.peak_memory = self.initial_memory
        self.peak_cpu = 0.0

    def update(self) -> dict[str, float]:
        """Update resource measurements.

        Returns:
            Dictionary with current resource usage
        """
        if self.start_time is None:
            msg = "Resource monitor not started"
            raise RuntimeError(msg)

        current_memory = psutil.virtual_memory().used / (1024**3)  # GB
        current_cpu = psutil.cpu_percent(interval=0.1)

        self.peak_memory = max(self.peak_memory, current_memory)
        self.peak_cpu = max(self.peak_cpu, current_cpu)

        return {
            "elapsed_time": time.time() - self.start_time,
            "current_memory_gb": current_memory,
            "peak_memory_gb": self.peak_memory,
            "memory_delta_gb": current_memory - self.initial_memory,
            "current_cpu_percent": current_cpu,
            "peak_cpu_percent": self.peak_cpu,
        }

    def get_summary(self) -> dict[str, float]:
        """Get resource usage summary.

        Returns:
            Dictionary with resource usage summary
        """
        if self.start_time is None:
            return {}

        final_stats = self.update()
        return {
            "total_execution_time": final_stats["elapsed_time"],
            "peak_memory_usage_gb": self.peak_memory,
            "memory_increase_gb": final_stats["memory_delta_gb"],
            "peak_cpu_usage_percent": self.peak_cpu,
        }
def create_argument_parser(
    description: str,
    add_common_args: bool = True,
) -> argparse.ArgumentParser:
    """Create a standardized argument parser.

    Args:
        description: Description of the script
        add_common_args: Whether to add common arguments

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    if add_common_args:
        # Logging arguments
        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",
            help="Set the logging level (default: INFO)",
        )

        parser.add_argument(
            "--log-file",
            type=Path,
            help="Write logs to the specified file",
        )

        parser.add_argument(
            "--quiet",
            action="store_true",
            help="Suppress console output (only log to file)",
        )

        # Execution arguments
        parser.add_argument(
            "--timeout",
            type=int,
            default=3600,
            help="Script timeout in seconds (default: 3600)",
        )

        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Perform a dry run without making changes",
        )

        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose output",
        )

        # Configuration arguments
        parser.add_argument(
            "--config-dir",
            type=Path,
            help="Configuration directory path",
        )

        parser.add_argument(
            "--environment",
            choices=["dev", "staging", "prod"],
            default="dev",
            help="Target environment (default: dev)",
        )

    return parser


def handle_errors(
    exit_on_error: bool = True,
    log_traceback: bool = True,
    return_value_on_error: Any = None,
) -> Callable[[F], F]:
    """Decorator for standardized error handling.

    Args:
        exit_on_error: Whether to exit the program on error
        log_traceback: Whether to log the full traceback
        return_value_on_error: Value to return on error (if not exiting)

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                logger = logging.getLogger(func.__module__)
                logger.info("Script interrupted by user")
                if exit_on_error:
                    sys.exit(130)  # Standard exit code for Ctrl+C
                return return_value_on_error
            except Exception as e:
                logger = logging.getLogger(func.__module__)

                if log_traceback:
                    logger.exception(f"Error in {func.__name__}: {e}")
                else:
                    logger.exception(f"Error in {func.__name__}: {e}")

                if exit_on_error:
                    sys.exit(1)
                return return_value_on_error

        return wrapper

    return decorator


def monitor_resources(log_interval: int = 30) -> Callable[[F], F]:
    """Decorator to monitor resource usage during function execution.

    Args:
        log_interval: Interval in seconds to log resource usage

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            monitor = ResourceMonitor()

            def log_resources() -> None:
                stats = monitor.update()
                logger.info(
                    f"Resource usage - Memory: {stats['current_memory_gb']:.1f}GB "
                    f"(+{stats['memory_delta_gb']:.1f}GB), "
                    f"CPU: {stats['current_cpu_percent']:.1f}%, "
                    f"Elapsed: {stats['elapsed_time']:.1f}s"
                )

            monitor.start()
            logger.info(
                f"Starting execution of {func.__name__} with resource monitoring"
            )

            try:
                # Set up periodic resource logging
                if log_interval > 0:

                    def signal_handler(signum, frame) -> None:
                        log_resources()
                        # Re-schedule the signal
                        signal.alarm(log_interval)

                    signal.signal(signal.SIGALRM, signal_handler)
                    signal.alarm(log_interval)

                # Execute the function
                result = func(*args, **kwargs)

                # Log final resource usage
                summary = monitor.get_summary()
                logger.info(
                    f"Completed {func.__name__} - "
                    f"Total time: {summary['total_execution_time']:.1f}s, "
                    f"Peak memory: {summary['peak_memory_usage_gb']:.1f}GB "
                    f"(+{summary['memory_increase_gb']:.1f}GB), "
                    f"Peak CPU: {summary['peak_cpu_usage_percent']:.1f}%"
                )

                return result

            finally:
                # Cancel the alarm
                if log_interval > 0:
                    signal.alarm(0)

        return wrapper

    return decorator


def timeout_handler(timeout_seconds: int) -> Callable[[F], F]:
    """Decorator to add timeout handling to functions.

    Args:
        timeout_seconds: Timeout in seconds

    Returns:
        Decorated function

    Raises:
        ScriptTimeoutError: If function execution exceeds timeout
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)

            def timeout_signal_handler(signum, frame) -> NoReturn:
                msg = f"Function {func.__name__} timed out after {timeout_seconds} seconds"
                raise ScriptTimeoutError(msg)

            # Set up timeout signal
            old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
            signal.alarm(timeout_seconds)

            try:
                logger.debug(
                    f"Starting {func.__name__} with {timeout_seconds}s timeout"
                )
                result = func(*args, **kwargs)
                logger.debug(f"Completed {func.__name__} within timeout")
                return result
            finally:
                # Restore original signal handler and cancel alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper

    return decorator


def ensure_directory(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def safe_file_write(
    file_path: str | Path,
    content: str,
    backup: bool = True,
    encoding: str = "utf-8",
) -> None:
    """Safely write content to a file with optional backup.

    Args:
        file_path: Target file path
        content: Content to write
        backup: Whether to create a backup of existing file
        encoding: File encoding
    """
    path = Path(file_path)

    # Create backup if file exists and backup is requested
    if backup and path.exists():
        backup_path = path.with_suffix(path.suffix + ".backup")
        backup_path.write_text(path.read_text(encoding=encoding), encoding=encoding)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write content
    path.write_text(content, encoding=encoding)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    hours = seconds / 3600
    return f"{hours:.1f}h"


def format_bytes(bytes_value: float) -> str:
    """Format bytes to human-readable string.

    Args:
        bytes_value: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB"


def get_system_info() -> dict[str, Any]:
    """Get current system information.

    Returns:
        Dictionary with system information
    """
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "platform": sys.platform,
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "memory_percent": memory.percent,
            "disk_total_gb": disk.total / (1024**3),
            "disk_free_gb": disk.free / (1024**3),
            "disk_percent": (disk.used / disk.total) * 100,
            "load_average": (
                list(os.getloadavg()) if hasattr(os, "getloadavg") else None
            ),
        }
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to get system info: {e}")
        return {"timestamp": datetime.now().isoformat(), "error": str(e)}
