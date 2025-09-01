"""
AIVillage Centralized Log Aggregation System
Provides structured logging, log shipping, and log analysis capabilities
"""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import time
from typing import Any

from prometheus_client import Counter, Histogram
import structlog

# Log metrics
LOG_MESSAGES_TOTAL = Counter("log_messages_total", "Total log messages", ["level", "service", "component"])
LOG_PROCESSING_DURATION = Histogram("log_processing_duration_seconds", "Log processing duration", ["processor"])
LOG_ERRORS_TOTAL = Counter("log_errors_total", "Total log processing errors", ["error_type", "service"])


class LogLevel(Enum):
    """Standardized log levels"""

    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    FATAL = "fatal"


@dataclass
class LogContext:
    """Standard log context for AIVillage services"""

    service: str
    component: str
    version: str = "1.0.0"
    environment: str = "production"
    trace_id: str | None = None
    span_id: str | None = None
    user_id: str | None = None
    request_id: str | None = None
    session_id: str | None = None


class StructuredLogger:
    """Structured logger with standardized format for AIVillage"""

    def __init__(self, context: LogContext):
        self.context = context
        self.logger = self._setup_structured_logger()

    def _setup_structured_logger(self) -> structlog.BoundLogger:
        """Setup structured logger with JSON formatting"""

        # Configure structlog
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
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Create base logger
        base_logger = structlog.get_logger()

        # Bind context
        return base_logger.bind(
            service=self.context.service,
            component=self.context.component,
            version=self.context.version,
            environment=self.context.environment,
            trace_id=self.context.trace_id,
            span_id=self.context.span_id,
        )

    def trace(self, message: str, **kwargs):
        """Log trace level message"""
        self._log(LogLevel.TRACE, message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug level message"""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info level message"""
        self._log(LogLevel.INFO, message, **kwargs)

    def warn(self, message: str, **kwargs):
        """Log warning level message"""
        self._log(LogLevel.WARN, message, **kwargs)

    def error(self, message: str, error: Exception | None = None, **kwargs):
        """Log error level message"""
        if error:
            kwargs.update({"error_type": type(error).__name__, "error_message": str(error)})
        self._log(LogLevel.ERROR, message, **kwargs)

    def fatal(self, message: str, error: Exception | None = None, **kwargs):
        """Log fatal level message"""
        if error:
            kwargs.update({"error_type": type(error).__name__, "error_message": str(error)})
        self._log(LogLevel.FATAL, message, **kwargs)

    def _log(self, level: LogLevel, message: str, **kwargs):
        """Internal log method with metrics"""
        start_time = time.time()

        try:
            # Add standard fields
            log_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": level.value,
                "message": message,
                **kwargs,
            }

            # Log using appropriate method
            if level == LogLevel.TRACE:
                self.logger.debug(message, **log_data)
            elif level == LogLevel.DEBUG:
                self.logger.debug(message, **log_data)
            elif level == LogLevel.INFO:
                self.logger.info(message, **log_data)
            elif level == LogLevel.WARN:
                self.logger.warning(message, **log_data)
            elif level == LogLevel.ERROR:
                self.logger.error(message, **log_data)
            elif level == LogLevel.FATAL:
                self.logger.critical(message, **log_data)

            # Record metrics
            LOG_MESSAGES_TOTAL.labels(
                level=level.value, service=self.context.service, component=self.context.component
            ).inc()

            duration = time.time() - start_time
            LOG_PROCESSING_DURATION.labels(processor="structured_logger").observe(duration)

        except Exception as e:
            LOG_ERRORS_TOTAL.labels(error_type=type(e).__name__, service=self.context.service).inc()
            # Fall back to standard logging
            logging.error(f"Structured logging failed: {e}")
            logging.log(getattr(logging, level.value.upper(), logging.INFO), message)


class ServiceLoggers:
    """Pre-configured loggers for AIVillage services"""

    @staticmethod
    def get_agent_forge_logger(trace_id: str | None = None) -> StructuredLogger:
        """Get Agent Forge pipeline logger"""
        context = LogContext(service="agent_forge", component="ml-pipeline", version="2.0.0", trace_id=trace_id)
        return StructuredLogger(context)

    @staticmethod
    def get_hyperrag_logger(trace_id: str | None = None) -> StructuredLogger:
        """Get HyperRAG memory system logger"""
        context = LogContext(service="hyperrag", component="memory-system", version="1.0.0", trace_id=trace_id)
        return StructuredLogger(context)

    @staticmethod
    def get_p2p_logger(trace_id: str | None = None) -> StructuredLogger:
        """Get P2P mesh network logger"""
        context = LogContext(service="p2p-mesh", component="networking", version="1.0.0", trace_id=trace_id)
        return StructuredLogger(context)

    @staticmethod
    def get_api_gateway_logger(trace_id: str | None = None) -> StructuredLogger:
        """Get API gateway logger"""
        context = LogContext(service="api-gateway", component="api-gateway", version="1.0.0", trace_id=trace_id)
        return StructuredLogger(context)

    @staticmethod
    def get_edge_logger(device_id: str, trace_id: str | None = None) -> StructuredLogger:
        """Get edge computing logger"""
        context = LogContext(service="edge-computing", component="edge-device", version="1.0.0", trace_id=trace_id)
        logger = StructuredLogger(context)
        # Bind device-specific context
        logger.logger = logger.logger.bind(device_id=device_id)
        return logger


class LogAnalyzer:
    """Analyze logs for patterns, errors, and performance insights"""

    def __init__(self):
        self.error_patterns = {}
        self.performance_metrics = {}
        self.anomaly_detectors = {}

    def analyze_error_patterns(self, logs: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze error patterns in logs"""
        error_analysis = {
            "total_errors": 0,
            "error_types": {},
            "error_trends": {},
            "top_errors": [],
            "error_rate_by_service": {},
        }

        for log_entry in logs:
            if log_entry.get("level") == "error":
                error_analysis["total_errors"] += 1

                # Count error types
                error_type = log_entry.get("error_type", "unknown")
                error_analysis["error_types"][error_type] = error_analysis["error_types"].get(error_type, 0) + 1

                # Track by service
                service = log_entry.get("service", "unknown")
                if service not in error_analysis["error_rate_by_service"]:
                    error_analysis["error_rate_by_service"][service] = 0
                error_analysis["error_rate_by_service"][service] += 1

        # Calculate top errors
        error_analysis["top_errors"] = sorted(error_analysis["error_types"].items(), key=lambda x: x[1], reverse=True)[
            :10
        ]

        return error_analysis

    def analyze_performance_patterns(self, logs: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze performance patterns in logs"""
        performance_analysis = {
            "slow_operations": [],
            "latency_trends": {},
            "throughput_metrics": {},
            "bottlenecks": [],
        }

        for log_entry in logs:
            # Look for duration fields
            duration_ms = log_entry.get("duration_ms")
            if duration_ms and duration_ms > 1000:  # >1 second
                performance_analysis["slow_operations"].append(
                    {
                        "service": log_entry.get("service"),
                        "component": log_entry.get("component"),
                        "operation": log_entry.get("operation"),
                        "duration_ms": duration_ms,
                        "timestamp": log_entry.get("timestamp"),
                    }
                )

        # Sort slow operations by duration
        performance_analysis["slow_operations"].sort(key=lambda x: x.get("duration_ms", 0), reverse=True)

        return performance_analysis

    def detect_anomalies(self, logs: list[dict[str, Any]]) -> dict[str, Any]:
        """Detect anomalies in log patterns"""
        anomalies = {"error_spikes": [], "unusual_patterns": [], "service_disruptions": []}

        # Group logs by service and time
        service_logs = {}
        for log_entry in logs:
            service = log_entry.get("service", "unknown")
            if service not in service_logs:
                service_logs[service] = []
            service_logs[service].append(log_entry)

        # Check for error spikes
        for service, service_log_entries in service_logs.items():
            error_count = sum(1 for entry in service_log_entries if entry.get("level") == "error")
            total_count = len(service_log_entries)

            if total_count > 0:
                error_rate = error_count / total_count
                if error_rate > 0.1:  # >10% error rate
                    anomalies["error_spikes"].append(
                        {
                            "service": service,
                            "error_rate": error_rate,
                            "error_count": error_count,
                            "total_count": total_count,
                        }
                    )

        return anomalies


class LogShipper:
    """Ship logs to centralized aggregation system"""

    def __init__(self, loki_url: str = "http://localhost:3100"):
        self.loki_url = loki_url
        self.buffer = []
        self.buffer_size = 100
        self.flush_interval = 10  # seconds

    async def ship_log(self, log_entry: dict[str, Any]):
        """Ship single log entry"""
        self.buffer.append(log_entry)

        if len(self.buffer) >= self.buffer_size:
            await self.flush_buffer()

    async def flush_buffer(self):
        """Flush log buffer to Loki"""
        if not self.buffer:
            return

        try:
            # Format logs for Loki
            loki_payload = self._format_for_loki(self.buffer)

            # Send to Loki (implement HTTP client call)
            # This is a placeholder - in production, implement actual HTTP call
            await self._send_to_loki(loki_payload)

            self.buffer.clear()

        except Exception as e:
            LOG_ERRORS_TOTAL.labels(error_type=type(e).__name__, service="log-shipper").inc()
            logging.error(f"Failed to ship logs to Loki: {e}")

    def _format_for_loki(self, logs: list[dict[str, Any]]) -> dict[str, Any]:
        """Format logs for Loki ingestion"""
        streams = {}

        for log_entry in logs:
            # Create label set for stream
            labels = {
                "service": log_entry.get("service", "unknown"),
                "component": log_entry.get("component", "unknown"),
                "level": log_entry.get("level", "info"),
            }

            label_string = ",".join([f'{k}="{v}"' for k, v in sorted(labels.items())])

            if label_string not in streams:
                streams[label_string] = []

            # Format timestamp and message for Loki
            timestamp = log_entry.get("timestamp", datetime.now(timezone.utc).isoformat())
            message = json.dumps(log_entry)

            streams[label_string].append([timestamp, message])

        # Convert to Loki format
        loki_streams = []
        for label_string, entries in streams.items():
            loki_streams.append(
                {
                    "stream": dict(item.split('="') for item in label_string.replace('"', "").split(",")),
                    "values": entries,
                }
            )

        return {"streams": loki_streams}

    async def _send_to_loki(self, payload: dict[str, Any]):
        """Send payload to Loki (placeholder)"""
        # In production, implement actual HTTP POST to Loki
        pass


# Global log analyzer and shipper
log_analyzer = LogAnalyzer()
log_shipper = LogShipper()


# Convenience functions for service logging
@contextmanager
def agent_forge_logging_context(phase: str, model: str, trace_id: str | None = None):
    """Context manager for Agent Forge logging"""
    logger = ServiceLoggers.get_agent_forge_logger(trace_id)
    logger = logger.logger.bind(phase=phase, model=model)

    logger.info("Agent Forge phase started", phase=phase, model=model)
    start_time = time.time()

    try:
        yield logger
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "Agent Forge phase failed", error_type=type(e).__name__, error_message=str(e), duration_ms=duration * 1000
        )
        raise
    else:
        duration = time.time() - start_time
        logger.info("Agent Forge phase completed", duration_ms=duration * 1000)


@contextmanager
def hyperrag_logging_context(query_type: str, collection: str = "", trace_id: str | None = None):
    """Context manager for HyperRAG logging"""
    logger = ServiceLoggers.get_hyperrag_logger(trace_id)
    logger = logger.logger.bind(query_type=query_type, collection=collection)

    logger.info("HyperRAG query started", query_type=query_type, collection=collection)
    start_time = time.time()

    try:
        yield logger
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "HyperRAG query failed", error_type=type(e).__name__, error_message=str(e), duration_ms=duration * 1000
        )
        raise
    else:
        duration = time.time() - start_time
        logger.info("HyperRAG query completed", duration_ms=duration * 1000)

        # Check against baseline
        if duration > 0.002:  # >2ms, above 1.19ms baseline
            logger.warn(
                "HyperRAG query slower than baseline",
                baseline_ms=1.19,
                actual_ms=duration * 1000,
                performance_ratio=duration / 0.00119,
            )


def initialize_logging_system():
    """Initialize centralized logging system"""
    # Setup log directory structure
    import os

    log_dirs = [
        "/var/log/aivillage/agent_forge",
        "/var/log/aivillage/hyperrag",
        "/var/log/aivillage/p2p",
        "/var/log/aivillage/gateway",
        "/var/log/aivillage/edge",
        "/var/log/aivillage/errors",
    ]

    for log_dir in log_dirs:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info("AIVillage centralized logging system initialized")


if __name__ == "__main__":
    initialize_logging_system()
