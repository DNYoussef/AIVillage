"""
Fog Metrics Collector

Provides metrics collection and monitoring for fog computing infrastructure.
"""

from dataclasses import dataclass, field
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class FogMetricsCollector:
    """Collects and manages fog infrastructure metrics"""

    node_metrics: dict[str, dict] = field(default_factory=dict)
    job_metrics: dict[str, dict] = field(default_factory=dict)
    network_metrics: dict[str, dict] = field(default_factory=dict)

    def record_node_metric(self, node_id: str, metric_type: str, value: float):
        """Record a metric for a specific node"""
        if node_id not in self.node_metrics:
            self.node_metrics[node_id] = {}

        self.node_metrics[node_id][metric_type] = {"value": value, "timestamp": time.time()}
        logger.debug(f"Recorded {metric_type}={value} for node {node_id}")

    def record_job_metric(self, job_id: str, metric_type: str, value: float):
        """Record a metric for a specific job"""
        if job_id not in self.job_metrics:
            self.job_metrics[job_id] = {}

        self.job_metrics[job_id][metric_type] = {"value": value, "timestamp": time.time()}
        logger.debug(f"Recorded {metric_type}={value} for job {job_id}")

    def get_node_metrics(self, node_id: str) -> dict:
        """Get all metrics for a specific node"""
        return self.node_metrics.get(node_id, {})

    def get_job_metrics(self, job_id: str) -> dict:
        """Get all metrics for a specific job"""
        return self.job_metrics.get(job_id, {})

    def get_network_latency(self, source: str, target: str) -> float | None:
        """Get network latency between two nodes"""
        key = f"{source}->{target}"
        if key in self.network_metrics:
            return self.network_metrics[key].get("latency")
        return None

    def record_network_latency(self, source: str, target: str, latency: float):
        """Record network latency between two nodes"""
        key = f"{source}->{target}"
        self.network_metrics[key] = {"latency": latency, "timestamp": time.time()}
        logger.debug(f"Recorded latency {source}->{target}: {latency}ms")
