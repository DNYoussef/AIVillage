"""Mobile metrics collection and Prometheus integration."""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

# Prometheus client
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    Summary,
    generate_latest,
    push_to_gateway,
)

# Handle safe imports for cross-platform compatibility
try:
    from .device_profiler import DeviceProfiler
except ImportError:
    # Mock DeviceProfiler for safe importing
    class DeviceProfiler:
        def __init__(self, *args, **kwargs):
            """Initialize mock device profiler."""
            self.device_info = {}


try:
    from .resource_allocator import ResourceAllocator
except ImportError:
    # Mock ResourceAllocator for safe importing
    class ResourceAllocator:
        def __init__(self, *args, **kwargs):
            """Initialize mock resource allocator."""
            self.resources = {}


logger = logging.getLogger(__name__)


@dataclass
class MetricDefinition:
    """Definition of a custom metric."""

    name: str
    metric_type: str  # counter, gauge, histogram, summary, info
    description: str
    labels: list[str] = field(default_factory=list)
    buckets: list[float] | None = None  # For histograms


class MetricsCollector:
    """Collects and aggregates mobile device metrics."""

    def __init__(
        self,
        device_profiler: DeviceProfiler,
        resource_allocator: ResourceAllocator | None = None,
        collection_interval: float = 10.0,
    ) -> None:
        self.device_profiler = device_profiler
        self.resource_allocator = resource_allocator
        self.collection_interval = collection_interval

        # Metrics registry
        self.registry = CollectorRegistry()

        # Core device metrics
        self._setup_device_metrics()

        # Resource allocation metrics
        if resource_allocator:
            self._setup_allocation_metrics()

        # P2P metrics (if available)
        self._setup_p2p_metrics()

        # Custom metrics
        self.custom_metrics: dict[str, Any] = {}

        # Collection state
        self.collecting = False
        self.collection_thread: threading.Thread | None = None

        # Metric aggregation
        self.metric_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        logger.info("Mobile metrics collector initialized")

    def _setup_device_metrics(self) -> None:
        """Setup core device monitoring metrics."""
        # Memory metrics
        self.memory_total = Gauge(
            "device_memory_total_bytes",
            "Total device memory in bytes",
            ["device_id", "device_type"],
            registry=self.registry,
        )

        self.memory_available = Gauge(
            "device_memory_available_bytes",
            "Available device memory in bytes",
            ["device_id", "device_type"],
            registry=self.registry,
        )

        self.memory_used = Gauge(
            "device_memory_used_bytes",
            "Used device memory in bytes",
            ["device_id", "device_type"],
            registry=self.registry,
        )

        self.memory_percent = Gauge(
            "device_memory_usage_percent",
            "Memory usage percentage",
            ["device_id", "device_type"],
            registry=self.registry,
        )

        # CPU metrics
        self.cpu_percent = Gauge(
            "device_cpu_usage_percent",
            "CPU usage percentage",
            ["device_id", "device_type"],
            registry=self.registry,
        )

        self.cpu_cores = Gauge(
            "device_cpu_cores_total",
            "Total CPU cores",
            ["device_id", "device_type"],
            registry=self.registry,
        )

        self.cpu_frequency = Gauge(
            "device_cpu_frequency_hz",
            "Current CPU frequency in Hz",
            ["device_id", "device_type"],
            registry=self.registry,
        )

        self.cpu_temperature = Gauge(
            "device_cpu_temperature_celsius",
            "CPU temperature in Celsius",
            ["device_id", "device_type"],
            registry=self.registry,
        )

        # Battery metrics
        self.battery_percent = Gauge(
            "device_battery_percent",
            "Battery charge percentage",
            ["device_id", "device_type"],
            registry=self.registry,
        )

        self.power_plugged = Gauge(
            "device_power_plugged",
            "Power adapter connected (1=yes, 0=no)",
            ["device_id", "device_type"],
            registry=self.registry,
        )

        # Storage metrics
        self.storage_total = Gauge(
            "device_storage_total_bytes",
            "Total storage in bytes",
            ["device_id", "device_type"],
            registry=self.registry,
        )

        self.storage_used = Gauge(
            "device_storage_used_bytes",
            "Used storage in bytes",
            ["device_id", "device_type"],
            registry=self.registry,
        )

        self.storage_percent = Gauge(
            "device_storage_usage_percent",
            "Storage usage percentage",
            ["device_id", "device_type"],
            registry=self.registry,
        )

        # Network metrics
        self.network_sent = Counter(
            "device_network_sent_bytes_total",
            "Total bytes sent over network",
            ["device_id", "device_type"],
            registry=self.registry,
        )

        self.network_received = Counter(
            "device_network_received_bytes_total",
            "Total bytes received over network",
            ["device_id", "device_type"],
            registry=self.registry,
        )

        self.network_connections = Gauge(
            "device_network_connections_active",
            "Active network connections",
            ["device_id", "device_type"],
            registry=self.registry,
        )

        # Performance metrics
        self.performance_score = Gauge(
            "device_performance_score",
            "Overall device performance score (0-1)",
            ["device_id", "device_type"],
            registry=self.registry,
        )

        self.resource_constrained = Gauge(
            "device_resource_constrained",
            "Device is resource constrained (1=yes, 0=no)",
            ["device_id", "device_type"],
            registry=self.registry,
        )

        # Process metrics
        self.process_count = Gauge(
            "device_processes_total",
            "Total number of processes",
            ["device_id", "device_type"],
            registry=self.registry,
        )

        # Monitoring metrics
        self.snapshots_taken = Counter(
            "device_snapshots_taken_total",
            "Total resource snapshots taken",
            ["device_id", "device_type"],
            registry=self.registry,
        )

        self.alerts_triggered = Counter(
            "device_alerts_triggered_total",
            "Total alerts triggered",
            ["device_id", "device_type", "alert_type"],
            registry=self.registry,
        )

    def _setup_allocation_metrics(self) -> None:
        """Setup resource allocation metrics."""
        # Allocation requests
        self.allocation_requests = Counter(
            "resource_allocation_requests_total",
            "Total resource allocation requests",
            ["device_id", "priority", "status"],
            registry=self.registry,
        )

        # Active allocations
        self.active_allocations = Gauge(
            "resource_allocations_active",
            "Currently active resource allocations",
            ["device_id", "strategy"],
            registry=self.registry,
        )

        # Resource utilization
        self.resource_utilization = Gauge(
            "resource_utilization_percent",
            "Resource utilization percentage",
            ["device_id", "resource_type"],
            registry=self.registry,
        )

        # Allocation duration
        self.allocation_duration = Histogram(
            "resource_allocation_duration_seconds",
            "Duration of resource allocations",
            ["device_id", "strategy"],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, float("inf")],
            registry=self.registry,
        )

        # Preemptions
        self.preemptions = Counter(
            "resource_preemptions_total",
            "Total resource preemptions",
            ["device_id", "reason"],
            registry=self.registry,
        )

        # Strategy changes
        self.strategy_changes = Counter(
            "allocation_strategy_changes_total",
            "Total allocation strategy changes",
            ["device_id", "from_strategy", "to_strategy"],
            registry=self.registry,
        )

    def _setup_p2p_metrics(self) -> None:
        """Setup P2P networking metrics."""
        # P2P connections
        self.p2p_connections = Gauge(
            "p2p_connections_active",
            "Active P2P connections",
            ["device_id", "connection_type"],
            registry=self.registry,
        )

        # P2P messages
        self.p2p_messages_sent = Counter(
            "p2p_messages_sent_total",
            "Total P2P messages sent",
            ["device_id", "message_type"],
            registry=self.registry,
        )

        self.p2p_messages_received = Counter(
            "p2p_messages_received_total",
            "Total P2P messages received",
            ["device_id", "message_type"],
            registry=self.registry,
        )

        # P2P latency
        self.p2p_latency = Histogram(
            "p2p_message_latency_seconds",
            "P2P message latency",
            ["device_id", "peer_id"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, float("inf")],
            registry=self.registry,
        )

        # Tensor streaming
        self.tensor_streams = Counter(
            "tensor_streams_total",
            "Total tensor streams",
            ["device_id", "direction", "status"],
            registry=self.registry,
        )

        self.tensor_bytes = Counter(
            "tensor_bytes_total",
            "Total tensor bytes transferred",
            ["device_id", "direction"],
            registry=self.registry,
        )

        # Mesh network
        self.mesh_routes = Gauge(
            "mesh_routes_active",
            "Active mesh routes",
            ["device_id", "protocol"],
            registry=self.registry,
        )

        self.mesh_broadcasts = Counter(
            "mesh_broadcasts_total",
            "Total mesh broadcasts",
            ["device_id", "hops"],
            registry=self.registry,
        )

    def start_collection(self) -> None:
        """Start metrics collection."""
        if self.collecting:
            return

        self.collecting = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop, daemon=True
        )
        self.collection_thread.start()

        logger.info("Metrics collection started")

    def stop_collection(self) -> None:
        """Stop metrics collection."""
        self.collecting = False

        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)

        logger.info("Metrics collection stopped")

    def _collection_loop(self) -> None:
        """Main metrics collection loop."""
        while self.collecting:
            try:
                self._collect_device_metrics()

                if self.resource_allocator:
                    self._collect_allocation_metrics()

                time.sleep(self.collection_interval)

            except Exception as e:
                logger.exception(f"Error in metrics collection: {e}")
                time.sleep(self.collection_interval)

    def _collect_device_metrics(self) -> None:
        """Collect device metrics from profiler."""
        snapshot = self.device_profiler.current_snapshot
        profile = self.device_profiler.profile

        if not snapshot:
            return

        device_id = profile.device_id
        device_type = profile.device_type.value
        labels = [device_id, device_type]

        # Memory metrics
        self.memory_total.labels(*labels).set(snapshot.memory_total)
        self.memory_available.labels(*labels).set(snapshot.memory_available)
        self.memory_used.labels(*labels).set(snapshot.memory_used)
        self.memory_percent.labels(*labels).set(snapshot.memory_percent)

        # CPU metrics
        self.cpu_percent.labels(*labels).set(snapshot.cpu_percent)
        self.cpu_cores.labels(*labels).set(snapshot.cpu_cores)

        if snapshot.cpu_freq_current:
            self.cpu_frequency.labels(*labels).set(
                snapshot.cpu_freq_current * 1000000
            )  # Convert to Hz

        if snapshot.cpu_temp:
            self.cpu_temperature.labels(*labels).set(snapshot.cpu_temp)

        # Battery metrics
        if snapshot.battery_percent is not None:
            self.battery_percent.labels(*labels).set(snapshot.battery_percent)

        if snapshot.power_plugged is not None:
            self.power_plugged.labels(*labels).set(1 if snapshot.power_plugged else 0)

        # Storage metrics
        if snapshot.storage_total > 0:
            self.storage_total.labels(*labels).set(snapshot.storage_total)
            self.storage_used.labels(*labels).set(snapshot.storage_used)
            self.storage_percent.labels(*labels).set(snapshot.storage_percent)

        # Network metrics (incremental)
        current_sent = snapshot.network_sent
        current_received = snapshot.network_received

        # Store previous values to calculate deltas
        if not hasattr(self, "_prev_network_sent"):
            self._prev_network_sent = current_sent
            self._prev_network_received = current_received
        else:
            sent_delta = max(0, current_sent - self._prev_network_sent)
            received_delta = max(0, current_received - self._prev_network_received)

            if sent_delta > 0:
                self.network_sent.labels(*labels).inc(sent_delta)
            if received_delta > 0:
                self.network_received.labels(*labels).inc(received_delta)

            self._prev_network_sent = current_sent
            self._prev_network_received = current_received

        self.network_connections.labels(*labels).set(snapshot.network_connections)

        # Performance metrics
        self.performance_score.labels(*labels).set(snapshot.performance_score)
        self.resource_constrained.labels(*labels).set(
            1 if snapshot.is_resource_constrained else 0
        )

        # Process metrics
        self.process_count.labels(*labels).set(snapshot.process_count)

        # Monitoring metrics
        monitoring_stats = self.device_profiler.get_monitoring_stats()

        # Update snapshot counter (incremental)
        current_snapshots = monitoring_stats["snapshots_taken"]
        if not hasattr(self, "_prev_snapshots"):
            self._prev_snapshots = current_snapshots
        else:
            snapshots_delta = max(0, current_snapshots - self._prev_snapshots)
            if snapshots_delta > 0:
                self.snapshots_taken.labels(*labels).inc(snapshots_delta)
            self._prev_snapshots = current_snapshots

        # Store metrics in history for analysis
        self.metric_history["memory_percent"].append(snapshot.memory_percent)
        self.metric_history["cpu_percent"].append(snapshot.cpu_percent)
        self.metric_history["performance_score"].append(snapshot.performance_score)

    def _collect_allocation_metrics(self) -> None:
        """Collect resource allocation metrics."""
        if not self.resource_allocator:
            return

        status = self.resource_allocator.get_allocation_status()
        device_id = self.device_profiler.profile.device_id

        # Active allocations
        strategy = status["strategy"]
        active_count = status["active_allocations"]
        self.active_allocations.labels(device_id, strategy).set(active_count)

        # Resource utilization
        utilization = status["resource_utilization"]
        for resource_type, percent in utilization.items():
            self.resource_utilization.labels(device_id, resource_type).set(percent)

        # Allocation statistics (incremental)
        stats = status["statistics"]

        # Track requests
        current_requests = stats["requests_received"]
        current_granted = stats["requests_granted"]
        current_denied = stats["requests_denied"]

        if not hasattr(self, "_prev_requests"):
            self._prev_requests = current_requests
            self._prev_granted = current_granted
            self._prev_denied = current_denied
        else:
            max(0, current_requests - self._prev_requests)
            granted_delta = max(0, current_granted - self._prev_granted)
            denied_delta = max(0, current_denied - self._prev_denied)

            if granted_delta > 0:
                self.allocation_requests.labels(device_id, "any", "granted").inc(
                    granted_delta
                )
            if denied_delta > 0:
                self.allocation_requests.labels(device_id, "any", "denied").inc(
                    denied_delta
                )

            self._prev_requests = current_requests
            self._prev_granted = current_granted
            self._prev_denied = current_denied

        # Preemptions
        current_preemptions = stats["preemptions_performed"]
        if not hasattr(self, "_prev_preemptions"):
            self._prev_preemptions = current_preemptions
        else:
            preemptions_delta = max(0, current_preemptions - self._prev_preemptions)
            if preemptions_delta > 0:
                self.preemptions.labels(device_id, "resource_pressure").inc(
                    preemptions_delta
                )
            self._prev_preemptions = current_preemptions

    def register_custom_metric(self, definition: MetricDefinition) -> Any:
        """Register a custom metric."""
        if definition.metric_type == "counter":
            metric = Counter(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry,
            )
        elif definition.metric_type == "gauge":
            metric = Gauge(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry,
            )
        elif definition.metric_type == "histogram":
            buckets = definition.buckets or [0.1, 0.5, 1.0, 5.0, 10.0, float("inf")]
            metric = Histogram(
                definition.name,
                definition.description,
                definition.labels,
                buckets=buckets,
                registry=self.registry,
            )
        elif definition.metric_type == "summary":
            metric = Summary(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry,
            )
        elif definition.metric_type == "info":
            metric = Info(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry,
            )
        else:
            msg = f"Unknown metric type: {definition.metric_type}"
            raise ValueError(msg)

        self.custom_metrics[definition.name] = metric
        logger.info(f"Registered custom metric: {definition.name}")

        return metric

    def get_custom_metric(self, name: str) -> Any | None:
        """Get a custom metric by name."""
        return self.custom_metrics.get(name)

    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return generate_latest(self.registry).decode("utf-8")

    def push_to_gateway(self, gateway_url: str, job: str) -> None:
        """Push metrics to Prometheus pushgateway."""
        try:
            push_to_gateway(
                gateway_url,
                job=job,
                registry=self.registry,
                grouping_key={"device_id": self.device_profiler.profile.device_id},
            )
            logger.info(f"Pushed metrics to gateway: {gateway_url}")
        except Exception as e:
            logger.exception(f"Failed to push metrics to gateway: {e}")

    def get_metric_summary(self) -> dict[str, Any]:
        """Get summary of collected metrics."""
        device_profile = self.device_profiler.profile
        current_snapshot = self.device_profiler.current_snapshot

        summary = {
            "device_info": {
                "device_id": device_profile.device_id,
                "device_type": device_profile.device_type.value,
                "total_memory_gb": device_profile.total_memory_gb,
                "cpu_cores": device_profile.cpu_cores,
            },
            "collection_info": {
                "collecting": self.collecting,
                "interval_seconds": self.collection_interval,
                "custom_metrics": len(self.custom_metrics),
            },
        }

        if current_snapshot:
            summary["current_metrics"] = {
                "memory_percent": current_snapshot.memory_percent,
                "cpu_percent": current_snapshot.cpu_percent,
                "performance_score": current_snapshot.performance_score,
                "resource_constrained": current_snapshot.is_resource_constrained,
                "battery_percent": current_snapshot.battery_percent,
                "power_plugged": current_snapshot.power_plugged,
            }

        if self.resource_allocator:
            allocation_status = self.resource_allocator.get_allocation_status()
            summary["resource_allocation"] = {
                "strategy": allocation_status["strategy"],
                "active_allocations": allocation_status["active_allocations"],
                "pending_requests": allocation_status["pending_requests"],
                "utilization": allocation_status["resource_utilization"],
            }

        return summary

    def analyze_trends(self, duration_minutes: int = 60) -> dict[str, Any]:
        """Analyze metric trends over specified duration."""
        trends = {}

        for metric_name, history in self.metric_history.items():
            if len(history) < 2:
                continue

            # Calculate basic statistics
            values = list(history)
            recent_values = values[
                -min(len(values), duration_minutes // (self.collection_interval / 60)) :
            ]

            if recent_values:
                trends[metric_name] = {
                    "current": recent_values[-1],
                    "avg": sum(recent_values) / len(recent_values),
                    "min": min(recent_values),
                    "max": max(recent_values),
                    "trend": (
                        "increasing"
                        if recent_values[-1] > recent_values[0]
                        else "decreasing"
                    ),
                    "samples": len(recent_values),
                }

        return trends

    def get_alerts(self) -> list[dict[str, Any]]:
        """Get current metric-based alerts."""
        alerts = []
        current_snapshot = self.device_profiler.current_snapshot

        if not current_snapshot:
            return alerts

        # Memory alerts
        if current_snapshot.memory_percent > 90:
            alerts.append(
                {
                    "type": "memory_critical",
                    "severity": "critical",
                    "message": f"Memory usage at {current_snapshot.memory_percent:.1f}%",
                    "value": current_snapshot.memory_percent,
                    "threshold": 90,
                }
            )
        elif current_snapshot.memory_percent > 80:
            alerts.append(
                {
                    "type": "memory_warning",
                    "severity": "warning",
                    "message": f"Memory usage at {current_snapshot.memory_percent:.1f}%",
                    "value": current_snapshot.memory_percent,
                    "threshold": 80,
                }
            )

        # CPU alerts
        if current_snapshot.cpu_percent > 90:
            alerts.append(
                {
                    "type": "cpu_critical",
                    "severity": "critical",
                    "message": f"CPU usage at {current_snapshot.cpu_percent:.1f}%",
                    "value": current_snapshot.cpu_percent,
                    "threshold": 90,
                }
            )

        # Battery alerts
        if current_snapshot.battery_percent is not None:
            if current_snapshot.battery_percent < 10:
                alerts.append(
                    {
                        "type": "battery_critical",
                        "severity": "critical",
                        "message": f"Battery at {current_snapshot.battery_percent:.1f}%",
                        "value": current_snapshot.battery_percent,
                        "threshold": 10,
                    }
                )
            elif current_snapshot.battery_percent < 20:
                alerts.append(
                    {
                        "type": "battery_low",
                        "severity": "warning",
                        "message": f"Battery at {current_snapshot.battery_percent:.1f}%",
                        "value": current_snapshot.battery_percent,
                        "threshold": 20,
                    }
                )

        # Temperature alerts
        if current_snapshot.cpu_temp is not None:
            if current_snapshot.cpu_temp > 85:
                alerts.append(
                    {
                        "type": "temperature_critical",
                        "severity": "critical",
                        "message": f"CPU temperature at {current_snapshot.cpu_temp:.1f}°C",
                        "value": current_snapshot.cpu_temp,
                        "threshold": 85,
                    }
                )

        return alerts


class MobileMetrics:
    """Main mobile metrics management class."""

    def __init__(
        self,
        device_profiler: DeviceProfiler,
        resource_allocator: ResourceAllocator | None = None,
        prometheus_gateway: str | None = None,
        push_interval: float = 60.0,
    ) -> None:
        self.device_profiler = device_profiler
        self.resource_allocator = resource_allocator
        self.prometheus_gateway = prometheus_gateway
        self.push_interval = push_interval

        # Initialize collector
        self.collector = MetricsCollector(
            device_profiler=device_profiler,
            resource_allocator=resource_allocator,
        )

        # Push to gateway if configured
        self.push_active = False
        self.push_thread: threading.Thread | None = None

        logger.info("Mobile metrics management initialized")

    def start(self) -> None:
        """Start metrics collection and pushing."""
        # Start collection
        self.collector.start_collection()

        # Start pushing if gateway configured
        if self.prometheus_gateway:
            self.start_pushing()

        logger.info("Mobile metrics started")

    def stop(self) -> None:
        """Stop metrics collection and pushing."""
        self.collector.stop_collection()

        if self.push_active:
            self.stop_pushing()

        logger.info("Mobile metrics stopped")

    def start_pushing(self) -> None:
        """Start pushing metrics to Prometheus gateway."""
        if not self.prometheus_gateway or self.push_active:
            return

        self.push_active = True
        self.push_thread = threading.Thread(target=self._push_loop, daemon=True)
        self.push_thread.start()

        logger.info(f"Started pushing metrics to {self.prometheus_gateway}")

    def stop_pushing(self) -> None:
        """Stop pushing metrics."""
        self.push_active = False

        if self.push_thread and self.push_thread.is_alive():
            self.push_thread.join(timeout=5.0)

        logger.info("Stopped pushing metrics")

    def _push_loop(self) -> None:
        """Push metrics to gateway periodically."""
        while self.push_active:
            try:
                job_name = (
                    f"aivillage_mobile_{self.device_profiler.profile.device_type.value}"
                )
                self.collector.push_to_gateway(self.prometheus_gateway, job_name)

                time.sleep(self.push_interval)

            except Exception as e:
                logger.exception(f"Error pushing metrics: {e}")
                time.sleep(self.push_interval)

    def export_metrics(self) -> str:
        """Export metrics for scraping."""
        return self.collector.export_metrics()

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive metrics summary."""
        return self.collector.get_metric_summary()

    def get_trends(self, duration_minutes: int = 60) -> dict[str, Any]:
        """Get metric trends."""
        return self.collector.analyze_trends(duration_minutes)

    def get_alerts(self) -> list[dict[str, Any]]:
        """Get current alerts."""
        return self.collector.get_alerts()

    def register_custom_metric(self, definition: MetricDefinition) -> Any:
        """Register custom metric."""
        return self.collector.register_custom_metric(definition)
