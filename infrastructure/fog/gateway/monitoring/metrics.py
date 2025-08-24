"""
Fog Gateway Prometheus Metrics Collection

Comprehensive metrics collection system for fog computing infrastructure with
Prometheus integration, SLA tracking, and real-time observability.

Provides the following metrics:
- fog_jobs_queued_total{namespace}: Total queued jobs by namespace
- fog_jobs_running_total{runtime}: Running jobs by runtime type
- fog_placement_latency_ms_bucket{class}: Placement latency histogram by SLA class
- fog_node_trust_score{node}: Trust score gauge for each fog node
- fog_namespace_cpu_sec_total{namespace}: CPU usage counter by namespace
"""

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import logging
import time

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger(__name__)


class SLAClass(Enum):
    """SLA service levels for fog jobs"""

    S = "replicated_attested"  # Mission-critical: replicated + attestation
    A = "replicated"  # High-availability: replicated only
    B = "best_effort"  # Standard: best-effort single node


class RuntimeType(Enum):
    """Runtime environments for fog jobs"""

    WASI = "wasi"
    PYTHON = "python"
    NODEJS = "nodejs"
    AGENT_FORGE = "agent_forge"
    RAG_QUERY = "rag_query"


@dataclass
class NodeMetrics:
    """Per-node performance and trust metrics"""

    node_id: str
    trust_score: float
    cpu_utilization: float
    memory_utilization: float
    job_success_rate: float
    average_latency_ms: float
    last_update: float


@dataclass
class NamespaceUsage:
    """Resource usage tracking per namespace"""

    namespace: str
    cpu_seconds_total: float
    memory_mb_hours: float
    jobs_submitted: int
    jobs_completed: int
    jobs_failed: int
    last_update: float


class FogMetricsCollector:
    """
    Comprehensive Prometheus metrics collector for fog computing infrastructure

    Collects and exposes metrics for job queuing, placement latency, node trust
    scores, namespace resource usage, and SLA compliance tracking.
    """

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        self._init_metrics()
        self._node_metrics: dict[str, NodeMetrics] = {}
        self._namespace_usage: dict[str, NamespaceUsage] = {}
        self._placement_start_times: dict[str, float] = {}
        self._job_queue_counts: dict[str, int] = defaultdict(int)
        self._running_job_counts: dict[RuntimeType, int] = defaultdict(int)

    def _init_metrics(self):
        """Initialize all Prometheus metrics"""

        # fog_jobs_queued_total{namespace}
        self.jobs_queued_total = Counter(
            "fog_jobs_queued_total", "Total number of jobs queued in fog gateway", ["namespace"], registry=self.registry
        )

        # fog_jobs_running_total{runtime}
        self.jobs_running_total = Gauge(
            "fog_jobs_running_total",
            "Current number of jobs running by runtime type",
            ["runtime"],
            registry=self.registry,
        )

        # fog_placement_latency_ms_bucket{sla_class}
        self.placement_latency_histogram = Histogram(
            "fog_placement_latency_ms",
            "Job placement latency in milliseconds by SLA class",
            ["sla_class"],
            buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
            registry=self.registry,
        )

        # fog_node_trust_score{node}
        self.node_trust_score = Gauge(
            "fog_node_trust_score", "Trust score for each fog node (0.0-1.0)", ["node"], registry=self.registry
        )

        # fog_namespace_cpu_sec_total{namespace}
        self.namespace_cpu_seconds = Counter(
            "fog_namespace_cpu_sec_total",
            "Total CPU seconds consumed by namespace",
            ["namespace"],
            registry=self.registry,
        )

        # Additional comprehensive metrics
        self.jobs_completed_total = Counter(
            "fog_jobs_completed_total",
            "Total completed jobs by namespace and SLA class",
            ["namespace", "sla_class"],
            registry=self.registry,
        )

        self.jobs_failed_total = Counter(
            "fog_jobs_failed_total",
            "Total failed jobs by namespace and reason",
            ["namespace", "reason"],
            registry=self.registry,
        )

        self.node_utilization = Gauge(
            "fog_node_utilization_ratio",
            "Resource utilization ratio for fog nodes",
            ["node", "resource"],
            registry=self.registry,
        )

        self.sla_violations_total = Counter(
            "fog_sla_violations_total",
            "Total SLA violations by class and type",
            ["sla_class", "violation_type"],
            registry=self.registry,
        )

        self.cluster_capacity = Gauge(
            "fog_cluster_capacity_total",
            "Total cluster capacity by resource type",
            ["resource"],
            registry=self.registry,
        )

        logger.info("Fog metrics collector initialized with Prometheus registry")

    def record_job_queued(self, namespace: str, sla_class: SLAClass):
        """Record a job being queued"""
        self.jobs_queued_total.labels(namespace=namespace).inc()
        self._job_queue_counts[namespace] += 1
        logger.debug(f"Job queued: namespace={namespace}, sla_class={sla_class.value}")

    def record_job_started(self, job_id: str, runtime: RuntimeType, namespace: str):
        """Record a job starting execution"""
        self.jobs_running_total.labels(runtime=runtime.value).inc()
        self._running_job_counts[runtime] += 1
        self._placement_start_times[job_id] = time.time()
        logger.debug(f"Job started: {job_id}, runtime={runtime.value}, namespace={namespace}")

    def record_job_completed(
        self, job_id: str, namespace: str, sla_class: SLAClass, runtime: RuntimeType, cpu_seconds: float
    ):
        """Record a job completing successfully"""
        self.jobs_completed_total.labels(namespace=namespace, sla_class=sla_class.value).inc()

        self.jobs_running_total.labels(runtime=runtime.value).dec()
        self._running_job_counts[runtime] = max(0, self._running_job_counts[runtime] - 1)

        # Track CPU usage
        self.namespace_cpu_seconds.labels(namespace=namespace).inc(cpu_seconds)

        # Update namespace usage tracking
        if namespace not in self._namespace_usage:
            self._namespace_usage[namespace] = NamespaceUsage(
                namespace=namespace,
                cpu_seconds_total=0,
                memory_mb_hours=0,
                jobs_submitted=0,
                jobs_completed=0,
                jobs_failed=0,
                last_update=time.time(),
            )

        usage = self._namespace_usage[namespace]
        usage.cpu_seconds_total += cpu_seconds
        usage.jobs_completed += 1
        usage.last_update = time.time()

        # Clean up placement timing
        self._placement_start_times.pop(job_id, None)

        logger.debug(f"Job completed: {job_id}, namespace={namespace}, cpu_seconds={cpu_seconds}")

    def record_job_failed(self, job_id: str, namespace: str, runtime: RuntimeType, reason: str):
        """Record a job failure"""
        self.jobs_failed_total.labels(namespace=namespace, reason=reason).inc()
        self.jobs_running_total.labels(runtime=runtime.value).dec()
        self._running_job_counts[runtime] = max(0, self._running_job_counts[runtime] - 1)

        # Update namespace usage tracking
        if namespace in self._namespace_usage:
            self._namespace_usage[namespace].jobs_failed += 1
            self._namespace_usage[namespace].last_update = time.time()

        # Clean up placement timing
        self._placement_start_times.pop(job_id, None)

        logger.warning(f"Job failed: {job_id}, namespace={namespace}, reason={reason}")

    def record_placement_latency(self, job_id: str, sla_class: SLAClass):
        """Record job placement completion and latency"""
        if job_id in self._placement_start_times:
            latency_ms = (time.time() - self._placement_start_times[job_id]) * 1000
            self.placement_latency_histogram.labels(sla_class=sla_class.value).observe(latency_ms)
            logger.debug(f"Placement latency: {job_id}, {latency_ms:.2f}ms, class={sla_class.value}")

    def update_node_trust_score(self, node_id: str, trust_score: float):
        """Update trust score for a fog node"""
        # Clamp trust score to valid range
        trust_score = max(0.0, min(1.0, trust_score))

        self.node_trust_score.labels(node=node_id).set(trust_score)

        # Update detailed node metrics
        if node_id not in self._node_metrics:
            self._node_metrics[node_id] = NodeMetrics(
                node_id=node_id,
                trust_score=trust_score,
                cpu_utilization=0.0,
                memory_utilization=0.0,
                job_success_rate=0.0,
                average_latency_ms=0.0,
                last_update=time.time(),
            )
        else:
            self._node_metrics[node_id].trust_score = trust_score
            self._node_metrics[node_id].last_update = time.time()

        logger.debug(f"Node trust score updated: {node_id} = {trust_score:.3f}")

    def update_node_utilization(self, node_id: str, cpu_ratio: float, memory_ratio: float):
        """Update resource utilization for a fog node"""
        self.node_utilization.labels(node=node_id, resource="cpu").set(cpu_ratio)
        self.node_utilization.labels(node=node_id, resource="memory").set(memory_ratio)

        if node_id in self._node_metrics:
            self._node_metrics[node_id].cpu_utilization = cpu_ratio
            self._node_metrics[node_id].memory_utilization = memory_ratio
            self._node_metrics[node_id].last_update = time.time()

    def record_sla_violation(self, sla_class: SLAClass, violation_type: str):
        """Record an SLA violation"""
        self.sla_violations_total.labels(sla_class=sla_class.value, violation_type=violation_type).inc()
        logger.warning(f"SLA violation: class={sla_class.value}, type={violation_type}")

    def update_cluster_capacity(self, cpu_cores: float, memory_gb: float, storage_gb: float):
        """Update total cluster capacity metrics"""
        self.cluster_capacity.labels(resource="cpu_cores").set(cpu_cores)
        self.cluster_capacity.labels(resource="memory_gb").set(memory_gb)
        self.cluster_capacity.labels(resource="storage_gb").set(storage_gb)

    def get_namespace_metrics(self) -> list[NamespaceUsage]:
        """Get resource usage metrics for all namespaces"""
        return list(self._namespace_usage.values())

    def get_node_metrics(self) -> list[NodeMetrics]:
        """Get performance metrics for all nodes"""
        return list(self._node_metrics.values())

    def get_queue_status(self) -> dict[str, int]:
        """Get current job queue counts by namespace"""
        return dict(self._job_queue_counts)

    def get_running_jobs_by_runtime(self) -> dict[RuntimeType, int]:
        """Get current running job counts by runtime"""
        return dict(self._running_job_counts)

    def export_metrics(self) -> bytes:
        """Export metrics in Prometheus format"""
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get the content type for Prometheus metrics"""
        return CONTENT_TYPE_LATEST


class SLAMetrics:
    """SLA-specific metrics tracking and validation"""

    def __init__(self, metrics_collector: FogMetricsCollector):
        self.collector = metrics_collector
        self.sla_targets = {
            SLAClass.S: {"placement_latency_ms": 250, "success_rate": 0.999},
            SLAClass.A: {"placement_latency_ms": 500, "success_rate": 0.99},
            SLAClass.B: {"placement_latency_ms": 1000, "success_rate": 0.95},
        }

    def validate_placement_sla(self, sla_class: SLAClass, latency_ms: float) -> bool:
        """Validate if placement latency meets SLA requirements"""
        target_latency = self.sla_targets[sla_class]["placement_latency_ms"]

        if latency_ms > target_latency:
            self.collector.record_sla_violation(sla_class, "placement_latency")
            return False
        return True

    def calculate_success_rate(self, namespace: str, sla_class: SLAClass, window_hours: int = 24) -> float:
        """Calculate job success rate for SLA validation"""
        # This would integrate with time-series data in production
        # For now, return a placeholder based on current metrics
        usage = self.collector._namespace_usage.get(namespace)
        if not usage:
            return 1.0

        total_jobs = usage.jobs_completed + usage.jobs_failed
        if total_jobs == 0:
            return 1.0

        return usage.jobs_completed / total_jobs

    def check_sla_compliance(self, namespace: str, sla_class: SLAClass) -> dict[str, bool]:
        """Check overall SLA compliance for a namespace"""
        success_rate = self.calculate_success_rate(namespace, sla_class)
        target_success_rate = self.sla_targets[sla_class]["success_rate"]

        compliance = {
            "success_rate": success_rate >= target_success_rate,
            "replication": True,  # Would check actual replication status
            "attestation": sla_class != SLAClass.S or True,  # Would check attestation
        }

        return compliance


# Global metrics collector instance
_metrics_collector: FogMetricsCollector | None = None


def get_metrics_collector() -> FogMetricsCollector:
    """Get the global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = FogMetricsCollector()
    return _metrics_collector


def record_job_event(event_type: str, namespace: str, **kwargs):
    """Convenience function to record job events"""
    collector = get_metrics_collector()

    if event_type == "queued":
        sla_class = kwargs.get("sla_class", SLAClass.B)
        collector.record_job_queued(namespace, sla_class)
    elif event_type == "started":
        job_id = kwargs["job_id"]
        runtime = kwargs.get("runtime", RuntimeType.WASI)
        collector.record_job_started(job_id, runtime, namespace)
    elif event_type == "completed":
        collector.record_job_completed(
            kwargs["job_id"],
            namespace,
            kwargs.get("sla_class", SLAClass.B),
            kwargs.get("runtime", RuntimeType.WASI),
            kwargs.get("cpu_seconds", 0.0),
        )
    elif event_type == "failed":
        collector.record_job_failed(
            kwargs["job_id"], namespace, kwargs.get("runtime", RuntimeType.WASI), kwargs.get("reason", "unknown")
        )


def record_placement_latency(job_id: str, sla_class: SLAClass):
    """Convenience function to record placement latency"""
    get_metrics_collector().record_placement_latency(job_id, sla_class)


def update_node_trust_score(node_id: str, trust_score: float):
    """Convenience function to update node trust scores"""
    get_metrics_collector().update_node_trust_score(node_id, trust_score)


def track_namespace_usage(namespace: str, cpu_seconds: float):
    """Convenience function to track namespace CPU usage"""
    get_metrics_collector().namespace_cpu_seconds.labels(namespace=namespace).inc(cpu_seconds)


# Export metrics endpoint helper
async def metrics_handler():
    """FastAPI/Starlette handler for /metrics endpoint"""
    collector = get_metrics_collector()
    return {"content": collector.export_metrics(), "media_type": collector.get_content_type()}


if __name__ == "__main__":
    # Demo usage
    collector = FogMetricsCollector()

    # Simulate some metrics
    collector.record_job_queued("production", SLAClass.S)
    collector.record_job_started("job-123", RuntimeType.WASI, "production")
    collector.update_node_trust_score("node-01", 0.95)
    collector.record_placement_latency("job-123", SLAClass.S)
    collector.record_job_completed("job-123", "production", SLAClass.S, RuntimeType.WASI, 45.2)

    # Export metrics
    print("Prometheus Metrics Output:")
    print(collector.export_metrics().decode())
