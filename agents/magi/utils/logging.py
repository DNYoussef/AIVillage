"""Logging configuration for MAGI agent system."""

import os
import sys
import json
import logging
import logging.handlers
from typing import Any, Dict, Optional
from pathlib import Path
from datetime import datetime
from functools import wraps
import traceback
import threading
from ..core.config import LoggingConfig
from ..core.constants import SYSTEM_CONSTANTS

class JSONFormatter(logging.Formatter):
    """Format log records as JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        # Basic log data
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data
        
        return json.dumps(log_data)

class ContextFilter(logging.Filter):
    """Add contextual information to log records."""
    
    def __init__(self):
        super().__init__()
        self.context = threading.local()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context data to the log record."""
        # Add request ID if available
        if hasattr(self.context, "request_id"):
            record.request_id = self.context.request_id
        
        # Add user info if available
        if hasattr(self.context, "user"):
            record.user = self.context.user
        
        return True

class PerformanceHandler(logging.Handler):
    """Handler for performance-related logs."""
    
    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename
        self.formatter = JSONFormatter()
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit performance log record."""
        try:
            with open(self.filename, 'a') as f:
                f.write(self.formatter.format(record) + '\n')
        except Exception:
            self.handleError(record)

def setup_logging(config: LoggingConfig) -> None:
    """Set up logging configuration."""
    # Create log directory if needed
    if config.log_file:
        os.makedirs(os.path.dirname(config.log_file), exist_ok=True)
    
    # Create handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter(config.log_format)
    )
    handlers.append(console_handler)
    
    # File handler
    if config.log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=config.log_file,
            maxBytes=config.max_log_size,
            backupCount=config.backup_count
        )
        file_handler.setFormatter(JSONFormatter())
        handlers.append(file_handler)
    
    # Performance handler
    if config.enable_performance_logging:
        perf_handler = PerformanceHandler("logs/performance.log")
        perf_handler.addFilter(
            lambda record: getattr(record, 'performance_log', False)
        )
        handlers.append(perf_handler)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(config.log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add new handlers
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Add context filter
    context_filter = ContextFilter()
    root_logger.addFilter(context_filter)
    
    # Set debug logging
    if config.enable_debug_logging:
        logging.getLogger('debug').setLevel(logging.DEBUG)

def log_execution(logger: logging.Logger):
    """Decorator to log function execution."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = await func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"{func.__name__} completed in {duration:.2f}s",
                    extra={"duration": duration, "success": True}
                )
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(
                    f"{func.__name__} failed after {duration:.2f}s: {str(e)}",
                    exc_info=True,
                    extra={"duration": duration, "success": False}
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"{func.__name__} completed in {duration:.2f}s",
                    extra={"duration": duration, "success": True}
                )
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(
                    f"{func.__name__} failed after {duration:.2f}s: {str(e)}",
                    exc_info=True,
                    extra={"duration": duration, "success": False}
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def log_performance(metric_name: str, logger: logging.Logger):
    """Decorator to log performance metrics."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = await func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"Performance metric: {metric_name}",
                    extra={
                        "performance_log": True,
                        "metric_name": metric_name,
                        "duration": duration,
                        "success": True
                    }
                )
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(
                    f"Performance metric: {metric_name} (failed)",
                    exc_info=True,
                    extra={
                        "performance_log": True,
                        "metric_name": metric_name,
                        "duration": duration,
                        "success": False,
                        "error": str(e)
                    }
                )
                raise
        return async_wrapper
    return decorator

class LogContext:
    """Context manager for adding temporary context to logs."""
    
    def __init__(self, **kwargs):
        self.context_filter = logging.getLogger().filters[0]
        self.previous_context = {}
        self.new_context = kwargs
    
    def __enter__(self):
        # Save current context
        for key in self.new_context:
            if hasattr(self.context_filter.context, key):
                self.previous_context[key] = getattr(self.context_filter.context, key)
        
        # Set new context
        for key, value in self.new_context.items():
            setattr(self.context_filter.context, key, value)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context
        for key in self.new_context:
            if key in self.previous_context:
                setattr(self.context_filter.context, key, self.previous_context[key])
            else:
                delattr(self.context_filter.context, key)

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)
