"""
Logging utilities for P2P infrastructure.

Provides standardized logging with structured output,
performance tracking, and cross-component consistency.
"""

import logging
import logging.handlers
import json
import time
import sys
import os
from typing import Dict, Any, Optional, Union
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from pathlib import Path


class LogLevel(Enum):
    """Standard log levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    logger_name: str
    message: str
    component: str
    peer_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    extra_fields: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        if self.extra_fields:
            data.update(self.extra_fields)
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""
    
    def __init__(self, component_name: str):
        super().__init__()
        self.component_name = component_name
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Extract extra fields
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'lineno', 'funcName',
                          'created', 'msecs', 'relativeCreated', 'thread', 
                          'threadName', 'processName', 'process', 'stack_info',
                          'exc_info', 'exc_text', 'message']:
                extra_fields[key] = value
        
        # Create structured log entry
        entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created).isoformat(),
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            component=self.component_name,
            peer_id=getattr(record, 'peer_id', None),
            session_id=getattr(record, 'session_id', None),
            correlation_id=getattr(record, 'correlation_id', None),
            extra_fields=extra_fields if extra_fields else None
        )
        
        # Add exception info if present
        if record.exc_info:
            entry.extra_fields = entry.extra_fields or {}
            entry.extra_fields['exception'] = self.formatException(record.exc_info)
        
        return entry.to_json()


class P2PLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for P2P components with context."""
    
    def __init__(self, logger: logging.Logger, component_name: str, 
                 peer_id: Optional[str] = None, session_id: Optional[str] = None):
        self.component_name = component_name
        self.peer_id = peer_id
        self.session_id = session_id
        extra = {
            'component': component_name,
            'peer_id': peer_id,
            'session_id': session_id
        }
        super().__init__(logger, extra)
    
    def process(self, msg, kwargs):
        """Process log message with context."""
        # Add context to extra fields
        extra = kwargs.get('extra', {})
        extra.update(self.extra)
        
        # Remove None values
        extra = {k: v for k, v in extra.items() if v is not None}
        kwargs['extra'] = extra
        
        return msg, kwargs
    
    def with_peer(self, peer_id: str) -> 'P2PLoggerAdapter':
        """Create new adapter with peer context."""
        return P2PLoggerAdapter(self.logger, self.component_name, peer_id, self.session_id)
    
    def with_session(self, session_id: str) -> 'P2PLoggerAdapter':
        """Create new adapter with session context."""
        return P2PLoggerAdapter(self.logger, self.component_name, self.peer_id, session_id)


class P2PLogger:
    """Main P2P logger class with component-specific loggers."""
    
    def __init__(self, component_name: str, log_level: LogLevel = LogLevel.INFO,
                 structured: bool = True, log_file: Optional[str] = None):
        self.component_name = component_name
        self.structured = structured
        self.log_file = log_file
        
        # Create base logger
        self.logger = logging.getLogger(f"p2p.{component_name}")
        self.logger.setLevel(log_level.value)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
        
        # Create adapter
        self.adapter = P2PLoggerAdapter(self.logger, component_name)
    
    def _setup_handlers(self):
        """Setup log handlers."""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        if self.structured:
            console_handler.setFormatter(StructuredFormatter(self.component_name))
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if self.log_file:
            # Ensure log directory exists
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            
            if self.structured:
                file_handler.setFormatter(StructuredFormatter(self.component_name))
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                )
            
            self.logger.addHandler(file_handler)
    
    def debug(self, msg: str, **kwargs):
        """Log debug message."""
        self.adapter.debug(msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        """Log info message."""
        self.adapter.info(msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message."""
        self.adapter.warning(msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        """Log error message."""
        self.adapter.error(msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        """Log critical message."""
        self.adapter.critical(msg, **kwargs)
    
    def exception(self, msg: str, **kwargs):
        """Log exception message with traceback."""
        self.adapter.exception(msg, **kwargs)
    
    def with_peer(self, peer_id: str) -> P2PLoggerAdapter:
        """Create logger adapter with peer context."""
        return self.adapter.with_peer(peer_id)
    
    def with_session(self, session_id: str) -> P2PLoggerAdapter:
        """Create logger adapter with session context."""
        return self.adapter.with_session(session_id)
    
    def with_correlation(self, correlation_id: str) -> P2PLoggerAdapter:
        """Create logger adapter with correlation context."""
        extra = {'correlation_id': correlation_id}
        return P2PLoggerAdapter(self.logger, self.component_name, 
                              extra=extra)


class StructuredLogger:
    """Simplified structured logger for quick use."""
    
    _loggers: Dict[str, P2PLogger] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_logger(cls, component_name: str, **kwargs) -> P2PLogger:
        """Get or create logger for component."""
        with cls._lock:
            if component_name not in cls._loggers:
                cls._loggers[component_name] = P2PLogger(component_name, **kwargs)
            return cls._loggers[component_name]


@contextmanager
def log_performance(logger: P2PLogger, operation_name: str, **extra_fields):
    """Context manager for logging operation performance."""
    start_time = time.time()
    
    logger.info(f"Starting {operation_name}", **extra_fields)
    
    try:
        yield
        duration = time.time() - start_time
        logger.info(f"Completed {operation_name}", 
                   duration_seconds=duration, **extra_fields)
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed {operation_name}: {e}",
                    duration_seconds=duration, 
                    error_type=e.__class__.__name__, **extra_fields)
        raise


def setup_logging(root_level: LogLevel = LogLevel.INFO,
                 structured: bool = True,
                 log_dir: Optional[str] = None) -> None:
    """Setup global logging configuration for P2P infrastructure."""
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(root_level.value)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    if structured:
        console_handler.setFormatter(StructuredFormatter("p2p"))
    else:
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
    
    root_logger.addHandler(console_handler)
    
    # File handler if log directory specified
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / "p2p.log",
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10
        )
        
        if structured:
            file_handler.setFormatter(StructuredFormatter("p2p"))
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.INFO)


def get_logger(component_name: str, **kwargs) -> P2PLogger:
    """Get logger for P2P component."""
    return StructuredLogger.get_logger(component_name, **kwargs)


def disable_urllib_warnings():
    """Disable urllib3 SSL warnings."""
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except ImportError:
        pass


# Performance logging decorator
def log_performance_decorator(logger: P2PLogger, operation_name: Optional[str] = None):
    """Decorator for automatic performance logging."""
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = func.__name__
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                logger.info(f"Starting {operation_name}")
                
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    logger.info(f"Completed {operation_name}", 
                               duration_seconds=duration)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(f"Failed {operation_name}: {e}",
                                duration_seconds=duration,
                                error_type=e.__class__.__name__)
                    raise
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                logger.info(f"Starting {operation_name}")
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    logger.info(f"Completed {operation_name}", 
                               duration_seconds=duration)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(f"Failed {operation_name}: {e}",
                                duration_seconds=duration,
                                error_type=e.__class__.__name__)
                    raise
            return sync_wrapper
    
    return decorator


# Import asyncio for decorator
try:
    import asyncio
except ImportError:
    asyncio = None
