"""
AIVillage Unified Linting - Logging and Monitoring Integration
Advanced logging, metrics collection, and monitoring capabilities
"""

import asyncio
import json
import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

# Optional monitoring dependencies with fallbacks
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


class LogLevel(Enum):
    """Standard logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(Enum):
    """Types of metrics to collect"""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"


@dataclass
class LintingMetric:
    """Standardized linting metric"""
    name: str
    metric_type: MetricType
    value: Union[int, float]
    labels: Dict[str, str]
    timestamp: float
    description: str = ""


@dataclass
class LintingEvent:
    """Standardized linting event for structured logging"""
    event_type: str
    level: LogLevel
    message: str
    tool: Optional[str] = None
    operation: Optional[str] = None
    duration: Optional[float] = None
    issues_found: Optional[int] = None
    files_processed: Optional[int] = None
    timestamp: Optional[float] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}


class AdvancedLintingLogger:
    """
    Advanced logging system with structured logging, multiple outputs, and filtering
    """
    
    def __init__(self, 
                 logger_name: str = "unified_linting",
                 log_level: LogLevel = LogLevel.INFO,
                 enable_file_logging: bool = True,
                 enable_structured_logging: bool = True,
                 log_directory: str = "logs",
                 max_log_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        
        self.logger_name = logger_name
        self.log_level = log_level
        self.enable_file_logging = enable_file_logging
        self.enable_structured_logging = enable_structured_logging and STRUCTLOG_AVAILABLE
        self.log_directory = Path(log_directory)
        self.max_log_size = max_log_size
        self.backup_count = backup_count
        
        # Create log directory
        self.log_directory.mkdir(exist_ok=True)
        
        # Initialize loggers
        self.standard_logger = self._setup_standard_logger()
        self.structured_logger = self._setup_structured_logger() if self.enable_structured_logging else None
        
        # Event buffer for batch processing
        self.event_buffer: List[LintingEvent] = []
        self.buffer_size = 100
        self.last_flush = time.time()
        self.flush_interval = 60  # seconds
    
    def _setup_standard_logger(self) -> logging.Logger:
        """Set up standard Python logger with multiple handlers"""
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(getattr(logging, self.log_level.value))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        if self.enable_file_logging:
            # File handler with rotation
            file_path = self.log_directory / "unified_linting.log"
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=self.max_log_size,
                backupCount=self.backup_count
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # JSON handler for structured logs
            json_file_path = self.log_directory / "unified_linting.jsonl"
            json_handler = logging.handlers.RotatingFileHandler(
                json_file_path,
                maxBytes=self.max_log_size,
                backupCount=self.backup_count
            )
            json_handler.setFormatter(self._JSONFormatter())
            logger.addHandler(json_handler)
        
        return logger
    
    def _setup_structured_logger(self):
        """Set up structured logger with structlog"""
        if not STRUCTLOG_AVAILABLE:
            return None
        
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        return structlog.get_logger(self.logger_name)
    
    class _JSONFormatter(logging.Formatter):
        """Custom JSON formatter for structured logging"""
        
        def format(self, record):
            log_data = {
                'timestamp': self.formatTime(record),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            # Add exception info if present
            if record.exc_info:
                log_data['exception'] = self.formatException(record.exc_info)
            
            # Add extra fields
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                              'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                              'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                              'thread', 'threadName', 'processName', 'process', 'message']:
                    log_data[key] = value
            
            return json.dumps(log_data)
    
    def log_event(self, event: LintingEvent):
        """Log a structured linting event"""
        # Add to event buffer
        self.event_buffer.append(event)
        
        # Log immediately with standard logger
        extra_data = {
            'tool': event.tool,
            'operation': event.operation,
            'duration': event.duration,
            'issues_found': event.issues_found,
            'files_processed': event.files_processed,
            'session_id': event.session_id
        }
        
        # Filter out None values
        extra_data = {k: v for k, v in extra_data.items() if v is not None}
        
        log_level = getattr(logging, event.level.value)
        self.standard_logger.log(log_level, event.message, extra=extra_data)
        
        # Log with structured logger if available
        if self.structured_logger:
            self.structured_logger.log(
                log_level,
                event.message,
                event_type=event.event_type,
                **extra_data,
                **event.metadata
            )
        
        # Flush buffer if needed
        self._maybe_flush_buffer()
    
    def log_tool_execution(self, 
                          tool: str, 
                          operation: str, 
                          duration: float, 
                          issues_found: int, 
                          files_processed: int,
                          session_id: str = None,
                          success: bool = True):
        """Log tool execution with standardized format"""
        event = LintingEvent(
            event_type="tool_execution",
            level=LogLevel.INFO if success else LogLevel.ERROR,
            message=f"Tool {tool} {operation} completed in {duration:.2f}s",
            tool=tool,
            operation=operation,
            duration=duration,
            issues_found=issues_found,
            files_processed=files_processed,
            session_id=session_id,
            metadata={"success": success}
        )
        self.log_event(event)
    
    def log_error(self, 
                  message: str, 
                  tool: str = None, 
                  operation: str = None,
                  exception: Exception = None,
                  session_id: str = None):
        """Log error with context"""
        metadata = {}
        if exception:
            metadata.update({
                "exception_type": type(exception).__name__,
                "exception_message": str(exception)
            })
        
        event = LintingEvent(
            event_type="error",
            level=LogLevel.ERROR,
            message=message,
            tool=tool,
            operation=operation,
            session_id=session_id,
            metadata=metadata
        )
        self.log_event(event)
    
    def log_performance_metric(self,
                              metric_name: str,
                              value: Union[int, float],
                              labels: Dict[str, str] = None,
                              session_id: str = None):
        """Log performance metric"""
        event = LintingEvent(
            event_type="performance_metric",
            level=LogLevel.INFO,
            message=f"Performance metric: {metric_name} = {value}",
            session_id=session_id,
            metadata={
                "metric_name": metric_name,
                "metric_value": value,
                "labels": labels or {}
            }
        )
        self.log_event(event)
    
    def _maybe_flush_buffer(self):
        """Flush event buffer if conditions are met"""
        current_time = time.time()
        
        if (len(self.event_buffer) >= self.buffer_size or 
            current_time - self.last_flush >= self.flush_interval):
            asyncio.create_task(self._flush_buffer())
    
    async def _flush_buffer(self):
        """Flush event buffer to persistent storage"""
        if not self.event_buffer:
            return
        
        try:
            # Save events to file
            events_file = self.log_directory / "linting_events.jsonl"
            with open(events_file, 'a') as f:
                for event in self.event_buffer:
                    f.write(json.dumps(asdict(event)) + '\n')
            
            # Clear buffer
            self.event_buffer.clear()
            self.last_flush = time.time()
            
        except Exception as e:
            self.standard_logger.error(f"Failed to flush event buffer: {e}")
    
    async def generate_log_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Generate log summary for the last N hours"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            # Read events from buffer and file
            all_events = list(self.event_buffer)
            
            events_file = self.log_directory / "linting_events.jsonl"
            if events_file.exists():
                with open(events_file, 'r') as f:
                    for line in f:
                        try:
                            event_data = json.loads(line.strip())
                            if event_data.get('timestamp', 0) >= cutoff_time:
                                # Convert back to LintingEvent
                                event = LintingEvent(**event_data)
                                all_events.append(event)
                        except (json.JSONDecodeError, TypeError):
                            continue
            
            # Analyze events
            summary = {
                "total_events": len(all_events),
                "event_types": {},
                "tools_used": {},
                "error_count": 0,
                "warning_count": 0,
                "total_issues_found": 0,
                "total_files_processed": 0,
                "average_execution_time": 0.0,
                "time_range_hours": hours
            }
            
            total_duration = 0
            duration_count = 0
            
            for event in all_events:
                # Count event types
                summary["event_types"][event.event_type] = summary["event_types"].get(event.event_type, 0) + 1
                
                # Count tools
                if event.tool:
                    summary["tools_used"][event.tool] = summary["tools_used"].get(event.tool, 0) + 1
                
                # Count errors and warnings
                if event.level == LogLevel.ERROR:
                    summary["error_count"] += 1
                elif event.level == LogLevel.WARNING:
                    summary["warning_count"] += 1
                
                # Sum issues and files
                if event.issues_found is not None:
                    summary["total_issues_found"] += event.issues_found
                if event.files_processed is not None:
                    summary["total_files_processed"] += event.files_processed
                
                # Calculate average duration
                if event.duration is not None:
                    total_duration += event.duration
                    duration_count += 1
            
            if duration_count > 0:
                summary["average_execution_time"] = total_duration / duration_count
            
            return summary
            
        except Exception as e:
            self.standard_logger.error(f"Failed to generate log summary: {e}")
            return {"error": str(e)}


class PrometheusMetricsCollector:
    """
    Prometheus metrics collector for linting operations
    """
    
    def __init__(self, enable_prometheus: bool = True):
        self.enabled = enable_prometheus and PROMETHEUS_AVAILABLE
        
        if not self.enabled:
            return
        
        # Create custom registry
        self.registry = CollectorRegistry()
        
        # Define metrics
        self.tool_executions = Counter(
            'linting_tool_executions_total',
            'Total number of linting tool executions',
            ['tool', 'status'],
            registry=self.registry
        )
        
        self.execution_duration = Histogram(
            'linting_execution_duration_seconds',
            'Duration of linting tool executions',
            ['tool'],
            registry=self.registry
        )
        
        self.issues_found = Counter(
            'linting_issues_found_total',
            'Total number of linting issues found',
            ['tool', 'severity'],
            registry=self.registry
        )
        
        self.files_processed = Counter(
            'linting_files_processed_total',
            'Total number of files processed',
            ['tool'],
            registry=self.registry
        )
        
        self.cache_operations = Counter(
            'linting_cache_operations_total',
            'Total number of cache operations',
            ['operation', 'status'],
            registry=self.registry
        )
        
        self.active_sessions = Gauge(
            'linting_active_sessions',
            'Number of active linting sessions',
            registry=self.registry
        )
        
        self.quality_score = Gauge(
            'linting_quality_score',
            'Overall code quality score',
            ['metric_type'],
            registry=self.registry
        )
    
    def record_tool_execution(self, tool: str, duration: float, success: bool):
        """Record tool execution metrics"""
        if not self.enabled:
            return
        
        status = "success" if success else "failure"
        self.tool_executions.labels(tool=tool, status=status).inc()
        self.execution_duration.labels(tool=tool).observe(duration)
    
    def record_issues_found(self, tool: str, issues: Dict[str, int]):
        """Record issues found by severity"""
        if not self.enabled:
            return
        
        for severity, count in issues.items():
            self.issues_found.labels(tool=tool, severity=severity).inc(count)
    
    def record_files_processed(self, tool: str, count: int):
        """Record number of files processed"""
        if not self.enabled:
            return
        
        self.files_processed.labels(tool=tool).inc(count)
    
    def record_cache_operation(self, operation: str, success: bool):
        """Record cache operation"""
        if not self.enabled:
            return
        
        status = "hit" if success else "miss"
        self.cache_operations.labels(operation=operation, status=status).inc()
    
    def set_active_sessions(self, count: int):
        """Set number of active sessions"""
        if not self.enabled:
            return
        
        self.active_sessions.set(count)
    
    def set_quality_score(self, metric_type: str, score: float):
        """Set quality score"""
        if not self.enabled:
            return
        
        self.quality_score.labels(metric_type=metric_type).set(score)
    
    def generate_metrics(self) -> str:
        """Generate Prometheus metrics output"""
        if not self.enabled:
            return "# Prometheus not available\n"
        
        return generate_latest(self.registry).decode('utf-8')


class MonitoringIntegration:
    """
    Unified monitoring integration for the linting system
    """
    
    def __init__(self, 
                 logger_config: Dict[str, Any] = None,
                 enable_prometheus: bool = True,
                 enable_health_checks: bool = True):
        
        # Initialize logger
        logger_config = logger_config or {}
        self.logger = AdvancedLintingLogger(**logger_config)
        
        # Initialize metrics collector
        self.metrics = PrometheusMetricsCollector(enable_prometheus)
        
        # Health check configuration
        self.enable_health_checks = enable_health_checks
        self.last_health_check = time.time()
        self.health_check_interval = 300  # 5 minutes
        
        # Session tracking
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.performance_data: List[Dict[str, Any]] = []
    
    async def start_session(self, session_id: str, operation: str, metadata: Dict[str, Any] = None):
        """Start a new linting session"""
        session_data = {
            "session_id": session_id,
            "operation": operation,
            "start_time": time.time(),
            "metadata": metadata or {},
            "tools_executed": [],
            "total_issues": 0,
            "total_files": 0
        }
        
        self.active_sessions[session_id] = session_data
        self.metrics.set_active_sessions(len(self.active_sessions))
        
        self.logger.log_event(LintingEvent(
            event_type="session_start",
            level=LogLevel.INFO,
            message=f"Started linting session: {operation}",
            operation=operation,
            session_id=session_id,
            metadata=metadata
        ))
    
    async def end_session(self, session_id: str, success: bool = True):
        """End a linting session"""
        if session_id not in self.active_sessions:
            return
        
        session_data = self.active_sessions.pop(session_id)
        duration = time.time() - session_data["start_time"]
        
        self.metrics.set_active_sessions(len(self.active_sessions))
        
        # Record session performance
        performance_record = {
            "session_id": session_id,
            "operation": session_data["operation"],
            "duration": duration,
            "tools_executed": len(session_data["tools_executed"]),
            "total_issues": session_data["total_issues"],
            "total_files": session_data["total_files"],
            "success": success,
            "timestamp": time.time()
        }
        
        self.performance_data.append(performance_record)
        
        self.logger.log_event(LintingEvent(
            event_type="session_end",
            level=LogLevel.INFO if success else LogLevel.ERROR,
            message=f"Ended linting session in {duration:.2f}s",
            operation=session_data["operation"],
            duration=duration,
            issues_found=session_data["total_issues"],
            files_processed=session_data["total_files"],
            session_id=session_id,
            metadata={"success": success}
        ))
    
    async def record_tool_execution(self, 
                                   session_id: str,
                                   tool: str, 
                                   operation: str,
                                   duration: float,
                                   issues_by_severity: Dict[str, int],
                                   files_processed: int,
                                   success: bool = True):
        """Record tool execution metrics"""
        
        # Update session data
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session["tools_executed"].append(tool)
            session["total_issues"] += sum(issues_by_severity.values())
            session["total_files"] += files_processed
        
        # Record metrics
        self.metrics.record_tool_execution(tool, duration, success)
        self.metrics.record_issues_found(tool, issues_by_severity)
        self.metrics.record_files_processed(tool, files_processed)
        
        # Log event
        self.logger.log_tool_execution(
            tool=tool,
            operation=operation,
            duration=duration,
            issues_found=sum(issues_by_severity.values()),
            files_processed=files_processed,
            session_id=session_id,
            success=success
        )
    
    async def record_quality_metrics(self, metrics: Dict[str, float]):
        """Record quality metrics"""
        for metric_name, value in metrics.items():
            self.metrics.set_quality_score(metric_name, value)
            self.logger.log_performance_metric(metric_name, value)
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        if not self.enable_health_checks:
            return {"status": "disabled"}
        
        current_time = time.time()
        if current_time - self.last_health_check < self.health_check_interval:
            return {"status": "recently_checked"}
        
        health_status = {
            "timestamp": current_time,
            "logger_status": "healthy",
            "metrics_status": "healthy" if self.metrics.enabled else "disabled",
            "active_sessions": len(self.active_sessions),
            "performance_records": len(self.performance_data),
            "disk_space": self._check_disk_space(),
            "log_file_sizes": self._check_log_file_sizes()
        }
        
        self.last_health_check = current_time
        
        self.logger.log_event(LintingEvent(
            event_type="health_check",
            level=LogLevel.INFO,
            message="System health check completed",
            metadata=health_status
        ))
        
        return health_status
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.logger.log_directory)
            return {
                "total_gb": total / (1024**3),
                "used_gb": used / (1024**3),
                "free_gb": free / (1024**3),
                "usage_percent": (used / total) * 100
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _check_log_file_sizes(self) -> Dict[str, Any]:
        """Check log file sizes"""
        try:
            log_files = {}
            for log_file in self.logger.log_directory.glob("*.log"):
                size_mb = log_file.stat().st_size / (1024**2)
                log_files[log_file.name] = f"{size_mb:.2f}MB"
            return log_files
        except Exception as e:
            return {"error": str(e)}
    
    async def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        log_summary = await self.logger.generate_log_summary()
        health_status = await self.perform_health_check()
        
        # Performance summary
        recent_performance = self.performance_data[-100:]  # Last 100 sessions
        avg_duration = sum(p["duration"] for p in recent_performance) / len(recent_performance) if recent_performance else 0
        success_rate = sum(1 for p in recent_performance if p["success"]) / len(recent_performance) if recent_performance else 0
        
        return {
            "monitoring_timestamp": datetime.now().isoformat(),
            "log_summary": log_summary,
            "health_status": health_status,
            "performance_summary": {
                "recent_sessions": len(recent_performance),
                "average_duration": avg_duration,
                "success_rate": success_rate * 100,
                "active_sessions": len(self.active_sessions)
            },
            "prometheus_metrics_available": self.metrics.enabled
        }


# Global monitoring instance
monitoring = MonitoringIntegration()


# Convenience functions
async def start_linting_session(session_id: str, operation: str, metadata: Dict[str, Any] = None):
    """Start a linting session with monitoring"""
    await monitoring.start_session(session_id, operation, metadata)


async def end_linting_session(session_id: str, success: bool = True):
    """End a linting session with monitoring"""
    await monitoring.end_session(session_id, success)


def log_tool_execution(tool: str, operation: str, duration: float, issues: int, files: int, session_id: str = None):
    """Log tool execution (sync wrapper)"""
    issues_by_severity = {"all": issues}  # Simplified for now
    asyncio.create_task(monitoring.record_tool_execution(
        session_id or "unknown", tool, operation, duration, issues_by_severity, files
    ))


def log_error(message: str, tool: str = None, operation: str = None, exception: Exception = None):
    """Log error (sync wrapper)"""
    monitoring.logger.log_error(message, tool, operation, exception)


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Test logging system
        session_id = "test_session_123"
        
        await start_linting_session(session_id, "test_operation", {"test": True})
        
        # Simulate tool executions
        await monitoring.record_tool_execution(
            session_id, "ruff", "check", 2.5, {"error": 5, "warning": 10}, 50
        )
        
        await monitoring.record_tool_execution(
            session_id, "black", "format_check", 1.2, {"style": 3}, 50
        )
        
        # Record quality metrics
        await monitoring.record_quality_metrics({
            "overall_score": 85.5,
            "security_score": 90.0
        })
        
        await end_linting_session(session_id, success=True)
        
        # Generate report
        report = await monitoring.generate_monitoring_report()
        print("Monitoring Report:")
        print(json.dumps(report, indent=2))
        
        # Generate Prometheus metrics
        if monitoring.metrics.enabled:
            print("\nPrometheus Metrics:")
            print(monitoring.metrics.generate_metrics())
    
    asyncio.run(main())