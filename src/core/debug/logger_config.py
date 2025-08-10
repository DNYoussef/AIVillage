"""AIVillage Debug Logging Configuration

Enhanced logging setup for debug mode following CODEX Integration Requirements.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class DebugFormatter(logging.Formatter):
    """Custom formatter for debug logging with enhanced information."""

    def __init__(self, include_thread_info: bool = True):
        self.include_thread_info = include_thread_info

        # Color codes for different log levels (if terminal supports it)
        self.colors = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
            "RESET": "\033[0m",  # Reset
        }

        base_format = "[{levelname}] {asctime} | {name} | {message}"

        if self.include_thread_info:
            base_format = "[{levelname}] {asctime} | {threadName} | {name} | {message}"

        super().__init__(fmt=base_format, datefmt="%Y-%m-%d %H:%M:%S", style="{")

    def format(self, record: logging.LogRecord) -> str:
        # Add color if we're outputting to a terminal
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            color = self.colors.get(record.levelname, "")
            reset = self.colors["RESET"]

            # Save original levelname
            original_levelname = record.levelname
            record.levelname = f"{color}{record.levelname}{reset}"

            # Format the message
            formatted = super().format(record)

            # Restore original levelname
            record.levelname = original_levelname

            return formatted

        return super().format(record)


def setup_debug_logging(
    log_level: str = "DEBUG",
    log_file: str | None = None,
    enable_console: bool = True,
    enable_file: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """Set up comprehensive debug logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (default: logs/aivillage_debug.log)
        enable_console: Whether to log to console
        enable_file: Whether to log to file
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    if log_file is None:
        log_file = logs_dir / f"aivillage_debug_{datetime.now().strftime('%Y%m%d')}.log"

    # Configure root logger
    root_logger = logging.getLogger()

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.DEBUG)
    root_logger.setLevel(numeric_level)

    # Create formatters
    debug_formatter = DebugFormatter(include_thread_info=True)
    simple_formatter = logging.Formatter(
        fmt="[{levelname}] {asctime} | {name} | {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
    )

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(debug_formatter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if enable_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_file_size, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(simple_formatter)
        root_logger.addHandler(file_handler)

    # Create AIVillage specific logger
    aivillage_logger = logging.getLogger("aivillage")
    aivillage_logger.setLevel(numeric_level)

    # Log initial debug setup information
    aivillage_logger.info("=" * 60)
    aivillage_logger.info("AIVillage Debug Logging Initialized")
    aivillage_logger.info("=" * 60)
    aivillage_logger.info(f"Log Level: {log_level}")
    aivillage_logger.info(f"Console Logging: {enable_console}")
    aivillage_logger.info(f"File Logging: {enable_file}")
    if enable_file:
        aivillage_logger.info(f"Log File: {log_file}")

    # Log environment information
    debug_mode = os.getenv("AIVILLAGE_DEBUG_MODE", "false")
    aivillage_logger.info(f"AIVILLAGE_DEBUG_MODE: {debug_mode}")
    aivillage_logger.info(
        f"AIVILLAGE_LOG_LEVEL: {os.getenv('AIVILLAGE_LOG_LEVEL', 'not set')}"
    )
    aivillage_logger.info(
        f"AIVILLAGE_PROFILE_PERFORMANCE: {os.getenv('AIVILLAGE_PROFILE_PERFORMANCE', 'not set')}"
    )

    return aivillage_logger


def get_debug_logger(name: str = None) -> logging.Logger:
    """Get a debug logger instance.

    Args:
        name: Logger name (defaults to calling module name)

    Returns:
        Logger instance configured for debug mode
    """
    if name is None:
        # Get the calling module name
        import inspect

        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "aivillage")

    logger = logging.getLogger(name)

    # Ensure logger respects debug mode
    debug_mode = os.getenv("AIVILLAGE_DEBUG_MODE", "false").lower() == "true"
    if debug_mode:
        logger.setLevel(logging.DEBUG)

    return logger


class RequestResponseLogger:
    """Logger for request/response data in debug mode.

    Provides structured logging for API calls, database queries, and other operations.
    """

    def __init__(self, logger_name: str = "aivillage.requests"):
        self.logger = logging.getLogger(logger_name)
        self.enabled = os.getenv("AIVILLAGE_DEBUG_MODE", "false").lower() == "true"

    def log_request(
        self,
        endpoint: str,
        method: str = "POST",
        headers: dict[str, str] | None = None,
        data: Any | None = None,
        params: dict[str, Any] | None = None,
    ):
        """Log API request details."""
        if not self.enabled:
            return

        self.logger.debug(f"[REQUEST] {method} {endpoint}")

        if headers:
            self.logger.debug(f"  Headers: {self._sanitize_headers(headers)}")

        if params:
            self.logger.debug(f"  Params: {params}")

        if data:
            data_str = str(data)
            if len(data_str) > 1000:
                data_str = data_str[:1000] + "... (truncated)"
            self.logger.debug(f"  Data: {data_str}")

    def log_response(
        self,
        endpoint: str,
        status_code: int | None = None,
        response_data: Any | None = None,
        duration_ms: float | None = None,
        error: str | None = None,
    ):
        """Log API response details."""
        if not self.enabled:
            return

        if error:
            self.logger.error(f"[RESPONSE] {endpoint} - ERROR: {error}")
            return

        status_str = f" ({status_code})" if status_code else ""
        duration_str = f" in {duration_ms:.2f}ms" if duration_ms else ""

        self.logger.debug(f"[RESPONSE] {endpoint}{status_str}{duration_str}")

        if response_data:
            data_str = str(response_data)
            if len(data_str) > 1000:
                data_str = data_str[:1000] + "... (truncated)"
            self.logger.debug(f"  Response: {data_str}")

    def log_database_query(
        self,
        query: str,
        params: Any | None = None,
        duration_ms: float | None = None,
        result_count: int | None = None,
        error: str | None = None,
    ):
        """Log database query details."""
        if not self.enabled:
            return

        # Sanitize query (remove sensitive data)
        sanitized_query = self._sanitize_query(query)

        if error:
            self.logger.error(f"[DB_ERROR] {sanitized_query} - ERROR: {error}")
            return

        duration_str = f" ({duration_ms:.2f}ms)" if duration_ms else ""
        count_str = f" -> {result_count} rows" if result_count is not None else ""

        self.logger.debug(f"[DB_QUERY]{duration_str} {sanitized_query}{count_str}")

        if params:
            params_str = str(params)
            if len(params_str) > 500:
                params_str = params_str[:500] + "... (truncated)"
            self.logger.debug(f"  Params: {params_str}")

    def log_cache_operation(
        self,
        operation: str,
        key: str,
        hit: bool,
        duration_ms: float | None = None,
        value_size: int | None = None,
    ):
        """Log cache operation details."""
        if not self.enabled:
            return

        # Truncate long keys
        display_key = key[:100] + "..." if len(key) > 100 else key

        hit_str = "HIT" if hit else "MISS"
        duration_str = f" ({duration_ms:.2f}ms)" if duration_ms else ""
        size_str = f" [{value_size} bytes]" if value_size else ""

        self.logger.debug(
            f"[CACHE_{hit_str}]{duration_str} {operation}: {display_key}{size_str}"
        )

    def _sanitize_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Remove sensitive information from headers."""
        sensitive_headers = ["authorization", "api-key", "x-api-key", "cookie"]

        sanitized = {}
        for key, value in headers.items():
            if key.lower() in sensitive_headers:
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value

        return sanitized

    def _sanitize_query(self, query: str) -> str:
        """Remove or mask sensitive data from SQL queries."""
        # Truncate very long queries
        if len(query) > 500:
            query = query[:500] + "... (truncated)"

        # Replace potential sensitive values (basic approach)
        # This is a simple implementation - more sophisticated detection could be added
        import re

        # Replace string literals that might contain sensitive data
        query = re.sub(r"'[^']{20,}'", "'***MASKED***'", query)
        query = re.sub(r'"[^"]{20,}"', '"***MASKED***"', query)

        return query


class PerformanceTimer:
    """Context manager for timing operations in debug mode."""

    def __init__(self, operation_name: str, logger: logging.Logger | None = None):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger("aivillage.performance")
        self.start_time = None
        self.enabled = (
            os.getenv("AIVILLAGE_PROFILE_PERFORMANCE", "false").lower() == "true"
        )

    def __enter__(self):
        if self.enabled:
            import time

            self.start_time = time.time()
            self.logger.debug(f"[TIMER_START] {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self.start_time:
            import time

            duration = time.time() - self.start_time
            duration_ms = duration * 1000

            if exc_type:
                self.logger.error(
                    f"[TIMER_ERROR] {self.operation_name} failed after {duration_ms:.2f}ms: {exc_val}"
                )
            else:
                level = logging.WARNING if duration_ms > 1000 else logging.DEBUG
                self.logger.log(
                    level, f"[TIMER_END] {self.operation_name}: {duration_ms:.2f}ms"
                )

    def get_duration_ms(self) -> float | None:
        """Get current duration in milliseconds."""
        if self.enabled and self.start_time:
            import time

            return (time.time() - self.start_time) * 1000
        return None
