"""
Monitoring and metrics utilities for P2P infrastructure.

Provides standardized metrics collection, performance tracking,
and monitoring capabilities across all P2P components.
"""

import time
import asyncio
import psutil
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import logging
import threading

try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, Summary
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    prometheus_client = None

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


@dataclass
class MetricSample:
    """A single metric measurement."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels
        }


@dataclass
class ConnectionMetrics:
    """Connection-specific metrics."""
    peer_id: str
    connected_at: datetime
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    latency_samples: List[float] = field(default_factory=list)
    error_count: int = 0
    last_activity: Optional[datetime] = None
    
    def get_average_latency(self) -> float:
        """Calculate average latency."""
        if not self.latency_samples:
            return 0.0
        return sum(self.latency_samples) / len(self.latency_samples)
    
    def add_latency_sample(self, latency: float, max_samples: int = 100):
        """Add latency sample with sliding window."""
        self.latency_samples.append(latency)
        if len(self.latency_samples) > max_samples:
            self.latency_samples.pop(0)


class MetricsCollector(ABC):
    """Abstract base class for metrics collection."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.metrics: Dict[str, List[MetricSample]] = defaultdict(list)
        self.lock = threading.Lock()
    
    @abstractmethod
    async def record_counter(self, name: str, value: float = 1.0, 
                           labels: Optional[Dict[str, str]] = None) -> None:
        """Record counter metric."""
        pass
    
    @abstractmethod
    async def record_gauge(self, name: str, value: float,
                         labels: Optional[Dict[str, str]] = None) -> None:
        """Record gauge metric."""
        pass
    
    @abstractmethod
    async def record_histogram(self, name: str, value: float,
                             labels: Optional[Dict[str, str]] = None) -> None:
        """Record histogram metric."""
        pass
    
    async def record_metric(self, name: str, value: float, metric_type: MetricType,
                          labels: Optional[Dict[str, str]] = None) -> None:
        """Record a generic metric."""
        sample = MetricSample(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            labels=labels or {}
        )
        
        with self.lock:
            self.metrics[name].append(sample)
            
            # Keep sliding window of samples (max 1000 per metric)
            if len(self.metrics[name]) > 1000:
                self.metrics[name] = self.metrics[name][-1000:]
    
    def get_metrics(self, name: Optional[str] = None, 
                   since: Optional[datetime] = None) -> Dict[str, List[MetricSample]]:
        """Get metrics with optional filtering."""
        with self.lock:
            if name:
                metrics = {name: self.metrics.get(name, [])}
            else:
                metrics = dict(self.metrics)
        
        if since:
            filtered_metrics = {}
            for metric_name, samples in metrics.items():
                filtered_samples = [s for s in samples if s.timestamp >= since]
                if filtered_samples:
                    filtered_metrics[metric_name] = filtered_samples
            return filtered_metrics
        
        return metrics
    
    def clear_metrics(self, older_than: Optional[datetime] = None):
        """Clear old metrics."""
        if older_than is None:
            older_than = datetime.now() - timedelta(hours=24)
        
        with self.lock:
            for name, samples in self.metrics.items():
                self.metrics[name] = [s for s in samples if s.timestamp >= older_than]


class StandardMetrics(MetricsCollector):
    """Standard metrics collector with in-memory storage."""
    
    async def record_counter(self, name: str, value: float = 1.0,
                           labels: Optional[Dict[str, str]] = None) -> None:
        """Record counter metric."""
        await self.record_metric(f"{self.component_name}_{name}", value, 
                               MetricType.COUNTER, labels)
    
    async def record_gauge(self, name: str, value: float,
                         labels: Optional[Dict[str, str]] = None) -> None:
        """Record gauge metric."""
        await self.record_metric(f"{self.component_name}_{name}", value,
                               MetricType.GAUGE, labels)
    
    async def record_histogram(self, name: str, value: float,
                             labels: Optional[Dict[str, str]] = None) -> None:
        """Record histogram metric."""
        await self.record_metric(f"{self.component_name}_{name}", value,
                               MetricType.HISTOGRAM, labels)


class PrometheusMetrics(MetricsCollector):
    """Prometheus-compatible metrics collector."""
    
    def __init__(self, component_name: str):
        if not HAS_PROMETHEUS:
            raise ImportError("prometheus_client required for Prometheus metrics")
        
        super().__init__(component_name)
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._summaries: Dict[str, Summary] = {}
    
    def _get_or_create_counter(self, name: str, labels: Dict[str, str]) -> Counter:
        """Get or create counter metric."""
        full_name = f"{self.component_name}_{name}"
        if full_name not in self._counters:
            label_names = list(labels.keys()) if labels else []
            self._counters[full_name] = Counter(
                full_name, f"{name} counter for {self.component_name}",
                label_names
            )
        return self._counters[full_name]
    
    def _get_or_create_gauge(self, name: str, labels: Dict[str, str]) -> Gauge:
        """Get or create gauge metric."""
        full_name = f"{self.component_name}_{name}"
        if full_name not in self._gauges:
            label_names = list(labels.keys()) if labels else []
            self._gauges[full_name] = Gauge(
                full_name, f"{name} gauge for {self.component_name}",
                label_names
            )
        return self._gauges[full_name]
    
    def _get_or_create_histogram(self, name: str, labels: Dict[str, str]) -> Histogram:
        """Get or create histogram metric."""
        full_name = f"{self.component_name}_{name}"
        if full_name not in self._histograms:
            label_names = list(labels.keys()) if labels else []
            self._histograms[full_name] = Histogram(
                full_name, f"{name} histogram for {self.component_name}",
                label_names
            )
        return self._histograms[full_name]
    
    async def record_counter(self, name: str, value: float = 1.0,
                           labels: Optional[Dict[str, str]] = None) -> None:
        """Record counter metric."""
        labels = labels or {}
        counter = self._get_or_create_counter(name, labels)
        
        if labels:
            counter.labels(**labels).inc(value)
        else:
            counter.inc(value)
        
        # Also store in parent class for querying
        await super().record_counter(name, value, labels)
    
    async def record_gauge(self, name: str, value: float,
                         labels: Optional[Dict[str, str]] = None) -> None:
        """Record gauge metric."""
        labels = labels or {}
        gauge = self._get_or_create_gauge(name, labels)
        
        if labels:
            gauge.labels(**labels).set(value)
        else:
            gauge.set(value)
        
        # Also store in parent class for querying
        await super().record_gauge(name, value, labels)
    
    async def record_histogram(self, name: str, value: float,
                             labels: Optional[Dict[str, str]] = None) -> None:
        """Record histogram metric."""
        labels = labels or {}
        histogram = self._get_or_create_histogram(name, labels)
        
        if labels:
            histogram.labels(**labels).observe(value)
        else:
            histogram.observe(value)
        
        # Also store in parent class for querying
        await super().record_histogram(name, value, labels)


class PerformanceTracker:
    """Performance tracking utilities."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self._timers: Dict[str, float] = {}
    
    def start_timer(self, name: str) -> str:
        """Start a performance timer."""
        timer_id = f"{name}_{id(self)}_{time.time()}"
        self._timers[timer_id] = time.time()
        return timer_id
    
    async def end_timer(self, timer_id: str, labels: Optional[Dict[str, str]] = None):
        """End timer and record duration."""
        if timer_id not in self._timers:
            logger.warning(f"Timer {timer_id} not found")
            return
        
        duration = time.time() - self._timers[timer_id]
        del self._timers[timer_id]
        
        # Extract name from timer_id
        name = timer_id.split('_')[0]
        await self.metrics.record_histogram(f"{name}_duration_seconds", duration, labels)
    
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        return TimerContext(self, name, labels)
    
    async def time_async_function(self, func: Callable, name: str,
                                labels: Optional[Dict[str, str]] = None) -> Any:
        """Time an async function execution."""
        start_time = time.time()
        try:
            result = await func()
            duration = time.time() - start_time
            await self.metrics.record_histogram(f"{name}_duration_seconds", duration, labels)
            await self.metrics.record_counter(f"{name}_calls_total", labels=labels)
            return result
        except Exception as e:
            duration = time.time() - start_time
            error_labels = {**(labels or {}), "error": e.__class__.__name__}
            await self.metrics.record_histogram(f"{name}_duration_seconds", duration, error_labels)
            await self.metrics.record_counter(f"{name}_errors_total", labels=error_labels)
            raise


class TimerContext:
    """Context manager for performance timing."""
    
    def __init__(self, tracker: PerformanceTracker, name: str, labels: Optional[Dict[str, str]]):
        self.tracker = tracker
        self.name = name
        self.labels = labels
        self.timer_id = None
    
    def __enter__(self):
        self.timer_id = self.tracker.start_timer(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer_id:
            asyncio.create_task(self.tracker.end_timer(self.timer_id, self.labels))


class ConnectionMonitor:
    """Monitor P2P connection health and performance."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.connections: Dict[str, ConnectionMetrics] = {}
        self.lock = asyncio.Lock()
    
    async def add_connection(self, peer_id: str):
        """Add a new connection to monitor."""
        async with self.lock:
            self.connections[peer_id] = ConnectionMetrics(
                peer_id=peer_id,
                connected_at=datetime.now()
            )
            await self.metrics.record_gauge("active_connections", len(self.connections))
    
    async def remove_connection(self, peer_id: str):
        """Remove connection from monitoring."""
        async with self.lock:
            if peer_id in self.connections:
                conn = self.connections[peer_id]
                
                # Record final metrics
                duration = (datetime.now() - conn.connected_at).total_seconds()
                await self.metrics.record_histogram("connection_duration_seconds", duration,
                                                  {"peer_id": peer_id})
                
                del self.connections[peer_id]
                await self.metrics.record_gauge("active_connections", len(self.connections))
    
    async def record_message_sent(self, peer_id: str, size_bytes: int):
        """Record sent message metrics."""
        async with self.lock:
            if peer_id in self.connections:
                conn = self.connections[peer_id]
                conn.messages_sent += 1
                conn.bytes_sent += size_bytes
                conn.last_activity = datetime.now()
                
                await self.metrics.record_counter("messages_sent_total", labels={"peer_id": peer_id})
                await self.metrics.record_counter("bytes_sent_total", size_bytes, {"peer_id": peer_id})
    
    async def record_message_received(self, peer_id: str, size_bytes: int):
        """Record received message metrics."""
        async with self.lock:
            if peer_id in self.connections:
                conn = self.connections[peer_id]
                conn.messages_received += 1
                conn.bytes_received += size_bytes
                conn.last_activity = datetime.now()
                
                await self.metrics.record_counter("messages_received_total", labels={"peer_id": peer_id})
                await self.metrics.record_counter("bytes_received_total", size_bytes, {"peer_id": peer_id})
    
    async def record_latency(self, peer_id: str, latency_ms: float):
        """Record connection latency."""
        async with self.lock:
            if peer_id in self.connections:
                conn = self.connections[peer_id]
                conn.add_latency_sample(latency_ms)
                
                await self.metrics.record_histogram("connection_latency_ms", latency_ms,
                                                  {"peer_id": peer_id})
    
    async def record_error(self, peer_id: str, error_type: str):
        """Record connection error."""
        async with self.lock:
            if peer_id in self.connections:
                conn = self.connections[peer_id]
                conn.error_count += 1
                
                await self.metrics.record_counter("connection_errors_total",
                                                labels={"peer_id": peer_id, "error_type": error_type})
    
    def get_connection_info(self, peer_id: str) -> Optional[ConnectionMetrics]:
        """Get connection information."""
        return self.connections.get(peer_id)
    
    def get_all_connections(self) -> Dict[str, ConnectionMetrics]:
        """Get all connection information."""
        return dict(self.connections)


async def collect_system_metrics(metrics_collector: MetricsCollector):
    """Collect system-level metrics."""
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    await metrics_collector.record_gauge("cpu_usage_percent", cpu_percent)
    
    # Memory usage
    memory = psutil.virtual_memory()
    await metrics_collector.record_gauge("memory_usage_percent", memory.percent)
    await metrics_collector.record_gauge("memory_used_bytes", memory.used)
    await metrics_collector.record_gauge("memory_available_bytes", memory.available)
    
    # Disk usage
    disk = psutil.disk_usage('/')
    await metrics_collector.record_gauge("disk_usage_percent", disk.percent)
    await metrics_collector.record_gauge("disk_used_bytes", disk.used)
    await metrics_collector.record_gauge("disk_free_bytes", disk.free)
    
    # Network I/O
    net_io = psutil.net_io_counters()
    await metrics_collector.record_counter("network_bytes_sent_total", net_io.bytes_sent)
    await metrics_collector.record_counter("network_bytes_received_total", net_io.bytes_recv)
    await metrics_collector.record_counter("network_packets_sent_total", net_io.packets_sent)
    await metrics_collector.record_counter("network_packets_received_total", net_io.packets_recv)


def create_metrics_collector(component_name: str, use_prometheus: bool = False) -> MetricsCollector:
    """Create appropriate metrics collector based on availability."""
    if use_prometheus and HAS_PROMETHEUS:
        try:
            return PrometheusMetrics(component_name)
        except Exception as e:
            logger.warning(f"Failed to create Prometheus metrics, using standard: {e}")
    
    return StandardMetrics(component_name)


# Decorator for automatic performance tracking
def track_performance(metrics_collector: MetricsCollector, metric_name: str):
    """Decorator to automatically track function performance."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                tracker = PerformanceTracker(metrics_collector)
                return await tracker.time_async_function(
                    lambda: func(*args, **kwargs), metric_name
                )
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    asyncio.create_task(metrics_collector.record_histogram(
                        f"{metric_name}_duration_seconds", duration
                    ))
                    asyncio.create_task(metrics_collector.record_counter(
                        f"{metric_name}_calls_total"
                    ))
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    error_labels = {"error": e.__class__.__name__}
                    asyncio.create_task(metrics_collector.record_histogram(
                        f"{metric_name}_duration_seconds", duration, error_labels
                    ))
                    asyncio.create_task(metrics_collector.record_counter(
                        f"{metric_name}_errors_total", labels=error_labels
                    ))
                    raise
            return sync_wrapper
    
    return decorator