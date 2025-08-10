"""Standardized Logging Configuration for AIVillage.

This module provides centralized logging configuration with support for
structured logging, multiple outputs, and component-specific loggers.
"""

from datetime import datetime
import json
import logging
import logging.config
from pathlib import Path
from typing import Any


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
        }

        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_data[key] = value

        return json.dumps(log_data, default=str)


class AIVillageLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds contextual information to all log messages."""

    def __init__(
        self, logger: logging.Logger, extra: dict[str, Any] | None = None
    ) -> None:
        super().__init__(logger, extra or {})

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple:
        """Process log message with additional context."""
        # Merge extra context
        if "extra" in kwargs:
            kwargs["extra"].update(self.extra)
        else:
            kwargs["extra"] = self.extra.copy()

        return msg, kwargs


def create_logging_config(
    log_level: str = "INFO",
    log_dir: str | None = None,
    structured_logging: bool = True,
    enable_console: bool = True,
    enable_file: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> dict[str, Any]:
    """Create logging configuration dictionary.

    Args:
        log_level: Default logging level
        log_dir: Directory for log files (defaults to ./logs)
        structured_logging: Whether to use JSON structured logging
        enable_console: Whether to enable console logging
        enable_file: Whether to enable file logging
        max_file_size: Maximum file size before rotation
        backup_count: Number of backup files to keep

    Returns:
        Logging configuration dictionary
    """
    if log_dir is None:
        log_dir = "logs"

    # Ensure log directory exists
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # If we can't create the directory, fall back to console logging only
        log_dir = None

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "structured": {"()": StructuredFormatter},
        },
        "handlers": {},
        "loggers": {
            "AIVillage": {"level": log_level, "handlers": [], "propagate": False}
        },
        "root": {"level": "WARNING", "handlers": []},
    }

    # Console handler
    if enable_console:
        config["handlers"]["console"] = {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "structured" if structured_logging else "standard",
            "stream": "ext://sys.stdout",
        }
        config["loggers"]["AIVillage"]["handlers"].append("console")

    # File handlers (only if log_dir is available)
    if enable_file and log_dir is not None:
        # Main application log
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "structured" if structured_logging else "detailed",
            "filename": f"{log_dir}/aivillage.log",
            "maxBytes": max_file_size,
            "backupCount": backup_count,
            "encoding": "utf-8",
        }
        config["loggers"]["AIVillage"]["handlers"].append("file")

        # Error-only log
        config["handlers"]["error_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "structured" if structured_logging else "detailed",
            "filename": f"{log_dir}/errors.log",
            "maxBytes": max_file_size,
            "backupCount": backup_count,
            "encoding": "utf-8",
        }
        config["loggers"]["AIVillage"]["handlers"].append("error_file")

        # Component-specific logs
        components = [
            "AIVillage.Agent",
            "AIVillage.RAG",
            "AIVillage.Communication",
            "AIVillage.Training",
            "AIVillage.Geometry",
            "AIVillage.ErrorHandler",
        ]

        for component in components:
            handler_name = f"{component.lower().replace('.', '_')}_file"
            config["handlers"][handler_name] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "structured" if structured_logging else "detailed",
                "filename": f"{log_dir}/{component.split('.')[-1].lower()}.log",
                "maxBytes": max_file_size,
                "backupCount": backup_count,
                "encoding": "utf-8",
            }

            config["loggers"][component] = {
                "level": log_level,
                "handlers": [handler_name],
                "propagate": True,
            }

    return config


def setup_aivillage_logging(
    log_level: str = "INFO",
    log_dir: str | None = None,
    structured_logging: bool = True,
    **kwargs,
) -> logging.Logger:
    """Setup AIVillage logging with standardized configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        structured_logging: Whether to use JSON structured logging
        **kwargs: Additional arguments for create_logging_config

    Returns:
        Main AIVillage logger
    """
    # Create and apply logging configuration
    config = create_logging_config(
        log_level=log_level,
        log_dir=log_dir,
        structured_logging=structured_logging,
        **kwargs,
    )

    logging.config.dictConfig(config)

    # Get main logger
    logger = logging.getLogger("AIVillage")

    # Log startup message
    logger.info(
        "AIVillage logging system initialized",
        extra={
            "log_level": log_level,
            "log_dir": log_dir,
            "structured_logging": structured_logging,
        },
    )

    return logger


def get_component_logger(component_name: str, **context) -> AIVillageLoggerAdapter:
    """Get a logger for a specific component with contextual information.

    Args:
        component_name: Name of the component (e.g., "Agent.King", "RAG.Pipeline")
        **context: Additional context to include in all log messages

    Returns:
        Logger adapter with component context
    """
    full_name = f"AIVillage.{component_name}"
    base_logger = logging.getLogger(full_name)

    # Add component context
    extra_context = {"component": component_name, **context}

    return AIVillageLoggerAdapter(base_logger, extra_context)


def log_function_call(
    logger: logging.Logger,
    level: int = logging.DEBUG,
    include_args: bool = False,
    include_result: bool = False,
):
    """Decorator to automatically log function calls.

    Args:
        logger: Logger to use for logging
        level: Log level for function call logs
        include_args: Whether to include function arguments
        include_result: Whether to include function result
    """

    def decorator(func):
        import functools

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"

            # Log function entry
            log_data = {"function": func_name, "event": "entry"}
            if include_args:
                log_data["args"] = str(args)
                log_data["kwargs"] = str(kwargs)

            logger.log(level, f"Entering {func_name}", extra=log_data)

            try:
                result = func(*args, **kwargs)

                # Log function exit
                exit_data = {"function": func_name, "event": "exit"}
                if include_result:
                    exit_data["result"] = str(result)

                logger.log(level, f"Exiting {func_name}", extra=exit_data)
                return result

            except Exception as e:
                # Log function exception
                logger.exception(
                    f"Exception in {func_name}: {e}",
                    extra={
                        "function": func_name,
                        "event": "exception",
                        "exception_type": type(e).__name__,
                    },
                )
                raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"

            # Log function entry
            log_data = {"function": func_name, "event": "entry"}
            if include_args:
                log_data["args"] = str(args)
                log_data["kwargs"] = str(kwargs)

            logger.log(level, f"Entering async {func_name}", extra=log_data)

            try:
                result = await func(*args, **kwargs)

                # Log function exit
                exit_data = {"function": func_name, "event": "exit"}
                if include_result:
                    exit_data["result"] = str(result)

                logger.log(level, f"Exiting async {func_name}", extra=exit_data)
                return result

            except Exception as e:
                # Log function exception
                logger.exception(
                    f"Exception in async {func_name}: {e}",
                    extra={
                        "function": func_name,
                        "event": "exception",
                        "exception_type": type(e).__name__,
                    },
                )
                raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def configure_third_party_logging() -> None:
    """Configure logging for third-party libraries to reduce noise."""
    # Reduce verbosity of common libraries
    library_configs = {
        "urllib3": logging.WARNING,
        "requests": logging.WARNING,
        "httpx": logging.WARNING,
        "transformers": logging.WARNING,
        "torch": logging.WARNING,
        "openai": logging.WARNING,
        "langroid": logging.INFO,
        "asyncio": logging.WARNING,
        "aiohttp": logging.WARNING,
    }

    for library, level in library_configs.items():
        logging.getLogger(library).setLevel(level)


# Performance monitoring utilities
class PerformanceLogger:
    """Logger for performance metrics and timing."""

    def __init__(self, component_name: str) -> None:
        self.logger = get_component_logger(f"Performance.{component_name}")
        self.timers = {}

    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        import time

        self.timers[operation] = time.time()
        self.logger.debug(f"Started timing {operation}")

    def end_timer(self, operation: str, **metadata):
        """End timing an operation and log the result."""
        import time

        if operation in self.timers:
            duration = time.time() - self.timers[operation]
            del self.timers[operation]

            self.logger.info(
                f"Operation {operation} completed",
                extra={
                    "operation": operation,
                    "duration_seconds": duration,
                    "duration_ms": duration * 1000,
                    **metadata,
                },
            )

            return duration
        self.logger.warning(f"Timer for {operation} was not started")
        return None

    def log_metric(
        self, metric_name: str, value: float, unit: str = "", **metadata
    ) -> None:
        """Log a performance metric."""
        self.logger.info(
            f"Metric: {metric_name}",
            extra={
                "metric_name": metric_name,
                "metric_value": value,
                "metric_unit": unit,
                **metadata,
            },
        )


# Context manager for operation timing
class timed_operation:
    """Context manager for timing operations."""

    def __init__(self, logger: logging.Logger, operation_name: str, **metadata) -> None:
        self.logger = logger
        self.operation_name = operation_name
        self.metadata = metadata
        self.start_time = None

    def __enter__(self):
        import time

        self.start_time = time.time()
        self.logger.debug(f"Starting timed operation: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        duration = time.time() - self.start_time

        if exc_type is None:
            self.logger.info(
                f"Completed timed operation: {self.operation_name}",
                extra={
                    "operation": self.operation_name,
                    "duration_seconds": duration,
                    "duration_ms": duration * 1000,
                    "success": True,
                    **self.metadata,
                },
            )
        else:
            self.logger.error(
                f"Failed timed operation: {self.operation_name}",
                extra={
                    "operation": self.operation_name,
                    "duration_seconds": duration,
                    "duration_ms": duration * 1000,
                    "success": False,
                    "exception_type": exc_type.__name__ if exc_type else None,
                    **self.metadata,
                },
            )


# Initialize logging on module import
if not logging.getLogger("AIVillage").handlers:
    setup_aivillage_logging()
    configure_third_party_logging()


__all__ = [
    "AIVillageLoggerAdapter",
    "PerformanceLogger",
    "StructuredFormatter",
    "configure_third_party_logging",
    "get_component_logger",
    "log_function_call",
    "setup_aivillage_logging",
    "timed_operation",
]
