"""Production Monitoring and Performance Optimization for BayesRAG-CODEX System.

Features:
- Enhanced health checks and monitoring
- Performance optimization for <100ms latency
- Circuit breakers and graceful degradation
- Real-time metrics and alerting
- Adaptive performance tuning
"""

import asyncio
import contextlib
import json
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class HealthCheck:
    """Health check configuration."""

    name: str
    check_function: Callable
    interval_seconds: int = 30
    timeout_seconds: int = 5
    failure_threshold: int = 3
    recovery_threshold: int = 2
    critical: bool = False


@dataclass
class PerformanceMetric:
    """Performance metric tracking."""

    name: str
    value: float
    timestamp: float
    target: float | None = None
    unit: str = "ms"

    @property
    def meets_target(self) -> bool:
        """Check if metric meets target."""
        if self.target is None:
            return True
        return self.value <= self.target


@dataclass
class CircuitBreaker:
    """Circuit breaker for service protection."""

    name: str
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_requests: int = 3

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float | None = None
    success_count: int = 0

    def record_success(self) -> None:
        """Record successful request."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_requests:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker '{self.name}' closed")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self) -> None:
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' opened")
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.success_count = 0
            logger.warning(f"Circuit breaker '{self.name}' reopened")

    def can_proceed(self) -> bool:
        """Check if request can proceed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' half-open")
                    return True
            return False

        # Half-open state
        return self.success_count < self.half_open_requests


class ProductionMonitor:
    """Production monitoring system for BayesRAG-CODEX pipeline."""

    def __init__(self, pipeline, cache, metrics_dir: Path = Path("/tmp/rag_metrics")) -> None:
        self.pipeline = pipeline
        self.cache = cache
        self.metrics_dir = metrics_dir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Health checks
        self.health_checks: dict[str, HealthCheck] = {}
        self.health_status = HealthStatus.HEALTHY
        self.health_history: deque = deque(maxlen=100)

        # Performance tracking
        self.performance_history: deque = deque(maxlen=1000)
        self.latency_target = 100  # ms

        # Circuit breakers
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

        # Prometheus metrics
        self.setup_prometheus_metrics()

        # Background tasks
        self.monitoring_task = None

        # Initialize default health checks
        self._setup_default_health_checks()

        # Initialize circuit breakers
        self._setup_circuit_breakers()

    def setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics for monitoring."""
        # Request metrics
        self.request_counter = Counter("rag_requests_total", "Total number of RAG requests", ["method", "status"])

        self.latency_histogram = Histogram(
            "rag_request_latency_ms",
            "Request latency in milliseconds",
            buckets=[10, 25, 50, 75, 100, 150, 200, 500, 1000],
        )

        # Cache metrics
        self.cache_hit_rate = Gauge("rag_cache_hit_rate", "Cache hit rate")

        self.cache_size = Gauge("rag_cache_size", "Cache size by tier", ["tier"])

        # Health metrics
        self.health_status_gauge = Gauge(
            "rag_health_status",
            "Health status (0=healthy, 1=degraded, 2=unhealthy, 3=critical)",
        )

        # Performance metrics
        self.p95_latency = Gauge("rag_p95_latency_ms", "P95 request latency")

        self.throughput = Gauge("rag_throughput_rps", "Request throughput (requests per second)")

    def _setup_default_health_checks(self) -> None:
        """Setup default health checks."""
        # Pipeline health
        self.add_health_check(
            HealthCheck(
                name="pipeline_ready",
                check_function=self._check_pipeline_ready,
                interval_seconds=30,
                critical=True,
            )
        )

        # Index health
        self.add_health_check(
            HealthCheck(
                name="index_accessible",
                check_function=self._check_index_accessible,
                interval_seconds=60,
                critical=True,
            )
        )

        # Cache health
        self.add_health_check(
            HealthCheck(
                name="cache_operational",
                check_function=self._check_cache_operational,
                interval_seconds=30,
            )
        )

        # Performance health
        self.add_health_check(
            HealthCheck(
                name="latency_target",
                check_function=self._check_latency_target,
                interval_seconds=10,
            )
        )

        # Memory health
        self.add_health_check(
            HealthCheck(
                name="memory_usage",
                check_function=self._check_memory_usage,
                interval_seconds=60,
            )
        )

    def _setup_circuit_breakers(self) -> None:
        """Setup circuit breakers for services."""
        # Embedding service breaker
        self.circuit_breakers["embedding"] = CircuitBreaker(name="embedding", failure_threshold=5, recovery_timeout=30)

        # Cache service breaker
        self.circuit_breakers["cache"] = CircuitBreaker(name="cache", failure_threshold=10, recovery_timeout=20)

        # Index service breaker
        self.circuit_breakers["index"] = CircuitBreaker(name="index", failure_threshold=3, recovery_timeout=60)

    async def _check_pipeline_ready(self) -> bool:
        """Check if pipeline is ready."""
        try:
            # Check pipeline attributes
            if not self.pipeline:
                return False

            # Check critical components
            if not hasattr(self.pipeline, "index") or self.pipeline.index is None:
                return False

            return not (not hasattr(self.pipeline, "embedder") or self.pipeline.embedder is None)

        except Exception as e:
            logger.exception(f"Pipeline health check failed: {e}")
            return False

    async def _check_index_accessible(self) -> bool:
        """Check if index is accessible."""
        try:
            if not self.pipeline or not self.pipeline.index:
                return False

            # Try a simple search
            test_embedding = np.random.randn(1, self.pipeline.vector_dim).astype("float32")
            scores, ids = self.pipeline.index.search(test_embedding, 1)

            return True

        except Exception as e:
            logger.exception(f"Index health check failed: {e}")
            return False

    async def _check_cache_operational(self) -> bool:
        """Check if cache is operational."""
        try:
            if not self.cache:
                return False

            # Try cache operations
            test_key = "__health_check__"
            test_value = ["health_check_result"]

            await self.cache.set(test_key, test_value, trust_score=0.5)
            result = await self.cache.get(test_key)

            return result is not None

        except Exception as e:
            logger.exception(f"Cache health check failed: {e}")
            return False

    async def _check_latency_target(self) -> bool:
        """Check if latency target is being met."""
        if not self.performance_history:
            return True  # No data yet

        # Get recent latencies
        recent_latencies = [p.value for p in list(self.performance_history)[-100:] if p.name == "request_latency"]

        if not recent_latencies:
            return True

        # Check P95 latency
        p95 = np.percentile(recent_latencies, 95)

        return p95 <= self.latency_target

    async def _check_memory_usage(self) -> bool:
        """Check memory usage."""
        try:
            import psutil

            # Get current process
            process = psutil.Process()
            memory_info = process.memory_info()

            # Check if memory usage is reasonable (< 4GB)
            memory_gb = memory_info.rss / (1024**3)

            if memory_gb > 4:
                logger.warning(f"High memory usage: {memory_gb:.2f} GB")
                return False

            return True

        except ImportError:
            # psutil not available, skip check
            return True
        except Exception as e:
            logger.exception(f"Memory health check failed: {e}")
            return False

    def add_health_check(self, health_check: HealthCheck) -> None:
        """Add a health check."""
        self.health_checks[health_check.name] = health_check

    async def run_health_checks(self) -> dict[str, Any]:
        """Run all health checks."""
        results = {}
        failed_critical = False
        failed_count = 0

        for name, check in self.health_checks.items():
            try:
                # Run check with timeout
                result = await asyncio.wait_for(check.check_function(), timeout=check.timeout_seconds)

                results[name] = {
                    "status": "healthy" if result else "unhealthy",
                    "critical": check.critical,
                }

                if not result:
                    failed_count += 1
                    if check.critical:
                        failed_critical = True

            except asyncio.TimeoutError:
                results[name] = {"status": "timeout", "critical": check.critical}
                failed_count += 1
                if check.critical:
                    failed_critical = True

            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e),
                    "critical": check.critical,
                }
                failed_count += 1
                if check.critical:
                    failed_critical = True

        # Determine overall health status
        if failed_critical:
            self.health_status = HealthStatus.CRITICAL
        elif failed_count >= len(self.health_checks) // 2:
            self.health_status = HealthStatus.UNHEALTHY
        elif failed_count > 0:
            self.health_status = HealthStatus.DEGRADED
        else:
            self.health_status = HealthStatus.HEALTHY

        # Update metrics
        status_value = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 1,
            HealthStatus.UNHEALTHY: 2,
            HealthStatus.CRITICAL: 3,
        }
        self.health_status_gauge.set(status_value[self.health_status])

        # Record in history
        self.health_history.append(
            {
                "timestamp": time.time(),
                "status": self.health_status.value,
                "checks": results,
            }
        )

        return {
            "status": self.health_status.value,
            "checks": results,
            "timestamp": datetime.now().isoformat(),
        }

    def record_request(self, latency_ms: float, success: bool = True, method: str = "retrieve") -> None:
        """Record request metrics."""
        # Update counters
        status = "success" if success else "failure"
        self.request_counter.labels(method=method, status=status).inc()

        # Update latency
        if success:
            self.latency_histogram.observe(latency_ms)

            # Record performance metric
            metric = PerformanceMetric(
                name="request_latency",
                value=latency_ms,
                timestamp=time.time(),
                target=self.latency_target,
            )
            self.performance_history.append(metric)

    async def optimize_performance(self) -> dict[str, Any]:
        """Optimize system performance based on metrics."""
        optimizations = {
            "applied": [],
            "recommendations": [],
            "current_p95": 0,
            "target": self.latency_target,
        }

        if not self.performance_history:
            return optimizations

        # Calculate current P95 latency
        latencies = [p.value for p in list(self.performance_history)[-100:] if p.name == "request_latency"]

        if latencies:
            p95 = np.percentile(latencies, 95)
            optimizations["current_p95"] = p95
            self.p95_latency.set(p95)

            # Apply optimizations if needed
            if p95 > self.latency_target:
                # Optimization 1: Increase cache size if hit rate is low
                cache_metrics = self.cache.get_metrics()
                if cache_metrics["hit_rate"] < 0.5:
                    # Increase hot cache size
                    if hasattr(self.cache, "tiers"):
                        self.cache.tiers["hot"].max_size = min(256, self.cache.tiers["hot"].max_size * 2)
                        optimizations["applied"].append("Increased hot cache size")

                # Optimization 2: Enable prefetching if not enabled
                if hasattr(self.cache, "enable_prefetch") and not self.cache.enable_prefetch:
                    self.cache.enable_prefetch = True
                    optimizations["applied"].append("Enabled cache prefetching")

                # Optimization 3: Reduce retrieval K if too high
                if hasattr(self.pipeline, "default_k") and self.pipeline.default_k > 5:
                    self.pipeline.default_k = 5
                    optimizations["applied"].append("Reduced default retrieval K")

                # Recommendations for manual intervention
                if p95 > self.latency_target * 2:
                    optimizations["recommendations"].append("Consider scaling horizontally or upgrading hardware")

                if cache_metrics["hit_rate"] < 0.3:
                    optimizations["recommendations"].append(
                        "Cache hit rate very low - consider warming cache with common queries"
                    )

        return optimizations

    async def handle_with_circuit_breaker(
        self,
        service_name: str,
        operation: Callable,
        fallback: Callable | None = None,
    ) -> Any:
        """Execute operation with circuit breaker protection."""
        breaker = self.circuit_breakers.get(service_name)
        if not breaker:
            # No breaker configured, execute directly
            return await operation()

        # Check if circuit allows request
        if not breaker.can_proceed():
            logger.warning(f"Circuit breaker '{service_name}' is open")

            # Use fallback if available
            if fallback:
                return await fallback()
            msg = f"Service '{service_name}' unavailable (circuit open)"
            raise Exception(msg)

        # Execute operation
        try:
            result = await operation()
            breaker.record_success()
            return result

        except Exception:
            breaker.record_failure()
            raise

    async def graceful_degradation(self, query: str, normal_k: int = 10) -> tuple[list[Any], dict[str, Any]]:
        """Retrieve with graceful degradation under load.

        Reduces quality/features to maintain latency target.
        """
        start_time = time.perf_counter()

        # Check system health
        if self.health_status == HealthStatus.CRITICAL:
            # Minimal response
            return [], {"degraded": True, "reason": "system_critical"}

        # Adjust K based on health
        if self.health_status == HealthStatus.UNHEALTHY:
            k = max(3, normal_k // 3)
        elif self.health_status == HealthStatus.DEGRADED:
            k = max(5, normal_k // 2)
        else:
            k = normal_k

        # Try with cache first (fastest)
        try:
            cached = await self.handle_with_circuit_breaker("cache", lambda: self.cache.get(query))

            if cached:
                latency = (time.perf_counter() - start_time) * 1000
                self.record_request(latency, success=True)
                return cached[0], {**cached[1], "degraded": False}

        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")

        # Try normal retrieval with reduced K
        try:
            results = await self.handle_with_circuit_breaker(
                "index", lambda: self.pipeline.retrieve(query, k=k, use_cache=False)
            )

            latency = (time.perf_counter() - start_time) * 1000
            self.record_request(latency, success=True)

            return results[0], {**results[1], "degraded": k < normal_k, "reduced_k": k}

        except Exception as e:
            logger.exception(f"Retrieval failed: {e}")
            latency = (time.perf_counter() - start_time) * 1000
            self.record_request(latency, success=False)

            # Return empty results
            return [], {"degraded": True, "error": str(e), "latency_ms": latency}

    async def start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        if self.monitoring_task and not self.monitoring_task.done():
            return

        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Production monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitoring_task

        logger.info("Production monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                # Run health checks
                await self.run_health_checks()

                # Update cache metrics
                if self.cache:
                    metrics = self.cache.get_metrics()
                    self.cache_hit_rate.set(metrics["hit_rate"])

                    if "cache_sizes" in metrics:
                        for tier, size in metrics["cache_sizes"].items():
                            self.cache_size.labels(tier=tier).set(size)

                # Calculate throughput
                recent_requests = [p for p in list(self.performance_history)[-60:] if p.timestamp > time.time() - 60]
                throughput = len(recent_requests) / 60
                self.throughput.set(throughput)

                # Check if optimization needed
                if self.health_status in [
                    HealthStatus.DEGRADED,
                    HealthStatus.UNHEALTHY,
                ]:
                    await self.optimize_performance()

                # Save metrics to disk periodically
                if len(self.performance_history) % 100 == 0:
                    await self._save_metrics()

                # Wait for next iteration
                await asyncio.sleep(10)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Monitoring loop error: {e}")
                await asyncio.sleep(30)

    async def _save_metrics(self) -> None:
        """Save metrics to disk."""
        try:
            metrics_file = self.metrics_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            metrics = {
                "timestamp": datetime.now().isoformat(),
                "health_status": self.health_status.value,
                "performance": {
                    "p95_latency": (self.p95_latency._value.get() if hasattr(self.p95_latency, "_value") else 0),
                    "throughput": (self.throughput._value.get() if hasattr(self.throughput, "_value") else 0),
                    "cache_hit_rate": (
                        self.cache_hit_rate._value.get() if hasattr(self.cache_hit_rate, "_value") else 0
                    ),
                },
                "circuit_breakers": {
                    name: {
                        "state": breaker.state.value,
                        "failure_count": breaker.failure_count,
                    }
                    for name, breaker in self.circuit_breakers.items()
                },
            }

            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

        except Exception as e:
            logger.exception(f"Failed to save metrics: {e}")

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get data for monitoring dashboard."""
        # Calculate current metrics
        latencies = [p.value for p in list(self.performance_history)[-100:] if p.name == "request_latency"]

        if latencies:
            current_p50 = np.percentile(latencies, 50)
            current_p95 = np.percentile(latencies, 95)
            current_p99 = np.percentile(latencies, 99)
        else:
            current_p50 = current_p95 = current_p99 = 0

        return {
            "health": {
                "status": self.health_status.value,
                "checks": len(self.health_checks),
                "last_check": self.health_history[-1] if self.health_history else None,
            },
            "performance": {
                "p50_latency_ms": current_p50,
                "p95_latency_ms": current_p95,
                "p99_latency_ms": current_p99,
                "target_ms": self.latency_target,
                "meets_target": current_p95 <= self.latency_target,
            },
            "cache": self.cache.get_metrics() if self.cache else {},
            "circuit_breakers": {
                name: {
                    "state": breaker.state.value,
                    "can_proceed": breaker.can_proceed(),
                }
                for name, breaker in self.circuit_breakers.items()
            },
            "timestamp": datetime.now().isoformat(),
        }


async def test_production_monitoring() -> None:
    """Test production monitoring system."""
    print("=== Testing Production Monitoring ===\n")

    # Mock pipeline and cache for testing
    from bayesrag_codex_enhanced import BayesRAGEnhancedPipeline
    from semantic_cache_advanced import SemanticMultiTierCache

    pipeline = BayesRAGEnhancedPipeline()
    cache = SemanticMultiTierCache()

    # Initialize monitor
    monitor = ProductionMonitor(pipeline, cache)

    # Start monitoring
    await monitor.start_monitoring()

    # Simulate some requests
    print("Simulating requests...")
    for i in range(20):
        latency = np.random.normal(80, 30)  # Mean 80ms, std 30ms
        latency = max(10, latency)  # Minimum 10ms

        success = np.random.random() > 0.1  # 90% success rate
        monitor.record_request(latency, success)

        await asyncio.sleep(0.1)

    # Run health checks
    print("\n=== Health Check Results ===")
    health_results = await monitor.run_health_checks()
    print(f"Overall Status: {health_results['status']}")

    for check_name, result in health_results["checks"].items():
        status_icon = "✅" if result["status"] == "healthy" else "❌"
        print(f"  {status_icon} {check_name}: {result['status']}")

    # Test circuit breaker
    print("\n=== Testing Circuit Breaker ===")

    async def failing_operation() -> NoReturn:
        msg = "Test failure"
        raise Exception(msg)

    async def fallback_operation() -> str:
        return "Fallback response"

    # Simulate failures
    for i in range(6):
        try:
            result = await monitor.handle_with_circuit_breaker("test_service", failing_operation, fallback_operation)
            print(f"  Request {i + 1}: {result}")
        except Exception as e:
            print(f"  Request {i + 1}: Failed - {e}")

    # Test graceful degradation
    print("\n=== Testing Graceful Degradation ===")

    # Simulate degraded health
    monitor.health_status = HealthStatus.DEGRADED

    results, metrics = await monitor.graceful_degradation(query="test query", normal_k=10)

    print(f"  Degraded: {metrics.get('degraded', False)}")
    print(f"  Reduced K: {metrics.get('reduced_k', 'N/A')}")

    # Get dashboard data
    print("\n=== Dashboard Data ===")
    dashboard = monitor.get_dashboard_data()

    print(f"Health Status: {dashboard['health']['status']}")
    print(f"P50 Latency: {dashboard['performance']['p50_latency_ms']:.1f}ms")
    print(f"P95 Latency: {dashboard['performance']['p95_latency_ms']:.1f}ms")
    print(f"Meets Target: {dashboard['performance']['meets_target']}")

    # Stop monitoring
    await monitor.stop_monitoring()

    print("\n✅ Production monitoring test complete")


if __name__ == "__main__":
    asyncio.run(test_production_monitoring())
