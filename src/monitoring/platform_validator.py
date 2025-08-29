"""
AIVillage Observability Platform Production Readiness Validator
Validates complete monitoring and observability infrastructure
"""

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
import json
import logging
import time
from typing import Any

import aiohttp
import docker
import redis

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation status levels"""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class ValidationResult:
    """Single validation test result"""

    component: str
    test_name: str
    status: ValidationStatus
    message: str
    details: dict[str, Any] = None
    execution_time_ms: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class ComponentStatus:
    """Overall component status"""

    component: str
    overall_status: ValidationStatus
    tests_passed: int
    tests_failed: int
    tests_warned: int
    tests_skipped: int
    execution_time_ms: float
    critical_issues: list[str]


class ObservabilityValidator:
    """Comprehensive observability platform validator"""

    def __init__(self):
        self.docker_client = docker.from_env()
        self.validation_results = []
        self.component_statuses = {}

        # Service endpoints
        self.endpoints = {
            "prometheus": "http://localhost:9090",
            "grafana": "http://localhost:3000",
            "jaeger": "http://localhost:16686",
            "loki": "http://localhost:3100",
            "alertmanager": "http://localhost:9093",
            "redis": "redis://localhost:6379",
            "memcached": "localhost:11211",
            "elasticsearch": "http://localhost:9200",
            "otel_collector": "http://localhost:8888",
        }

        # Expected containers
        self.expected_containers = [
            "aivillage-prometheus-master",
            "aivillage-grafana-master",
            "aivillage-jaeger-master",
            "aivillage-loki-master",
            "aivillage-promtail-master",
            "aivillage-alertmanager-master",
            "aivillage-otel-collector",
            "aivillage-redis-master-perf",
            "aivillage-redis-replica-1-perf",
            "aivillage-memcached-perf",
            "aivillage-elasticsearch",
            "aivillage-node-exporter",
            "aivillage-cadvisor",
        ]

    async def validate_complete_platform(self) -> dict[str, Any]:
        """Run complete platform validation"""
        logger.info("Starting AIVillage observability platform validation")
        start_time = time.time()

        validation_tasks = [
            self.validate_container_infrastructure(),
            self.validate_metrics_collection(),
            self.validate_distributed_tracing(),
            self.validate_log_aggregation(),
            self.validate_alerting_system(),
            self.validate_performance_caching(),
            self.validate_service_discovery(),
            self.validate_dashboard_functionality(),
            self.validate_data_retention_policies(),
            self.validate_security_configuration(),
        ]

        # Run all validations concurrently
        await asyncio.gather(*validation_tasks, return_exceptions=True)

        # Calculate overall results
        total_time = (time.time() - start_time) * 1000
        overall_result = self._calculate_overall_status()

        validation_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_execution_time_ms": total_time,
            "overall_status": overall_result["status"].value,
            "summary": overall_result["summary"],
            "component_statuses": {k: asdict(v) for k, v in self.component_statuses.items()},
            "validation_results": [asdict(result) for result in self.validation_results],
            "recommendations": self._generate_recommendations(),
            "production_readiness_score": self._calculate_readiness_score(),
        }

        logger.info(f"Platform validation completed in {total_time:.2f}ms")
        return validation_report

    async def validate_container_infrastructure(self):
        """Validate Docker container infrastructure"""
        component = "container_infrastructure"
        results = []
        start_time = time.time()

        try:
            # Check if all expected containers exist and are running
            containers = {container.name: container for container in self.docker_client.containers.list(all=True)}

            for expected_container in self.expected_containers:
                if expected_container in containers:
                    container = containers[expected_container]
                    if container.status == "running":
                        results.append(
                            ValidationResult(
                                component,
                                f"container_{expected_container}_running",
                                ValidationStatus.PASS,
                                f"Container {expected_container} is running",
                                {"status": container.status, "image": container.image.tags},
                            )
                        )
                    else:
                        results.append(
                            ValidationResult(
                                component,
                                f"container_{expected_container}_status",
                                ValidationStatus.FAIL,
                                f"Container {expected_container} is not running: {container.status}",
                                {"status": container.status},
                            )
                        )
                else:
                    results.append(
                        ValidationResult(
                            component,
                            f"container_{expected_container}_exists",
                            ValidationStatus.FAIL,
                            f"Container {expected_container} does not exist",
                        )
                    )

            # Check container resource usage
            running_containers = [c for c in containers.values() if c.status == "running"]
            total_memory_usage = 0

            for container in running_containers:
                try:
                    stats = container.stats(stream=False)
                    memory_usage = stats["memory_stats"]["usage"] if "memory_stats" in stats else 0
                    total_memory_usage += memory_usage
                except Exception as e:
                    logger.warning(f"Could not get stats for {container.name}: {e}")

            # Memory usage validation (should be reasonable)
            total_memory_gb = total_memory_usage / (1024**3)
            if total_memory_gb < 8:  # Less than 8GB total
                results.append(
                    ValidationResult(
                        component,
                        "memory_usage",
                        ValidationStatus.PASS,
                        f"Total container memory usage: {total_memory_gb:.2f}GB",
                        {"memory_gb": total_memory_gb},
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        component,
                        "memory_usage",
                        ValidationStatus.WARN,
                        f"High container memory usage: {total_memory_gb:.2f}GB",
                        {"memory_gb": total_memory_gb},
                    )
                )

        except Exception as e:
            results.append(
                ValidationResult(
                    component, "docker_connectivity", ValidationStatus.FAIL, f"Failed to connect to Docker: {e}"
                )
            )

        execution_time = (time.time() - start_time) * 1000
        self._record_component_results(component, results, execution_time)

    async def validate_metrics_collection(self):
        """Validate Prometheus metrics collection"""
        component = "metrics_collection"
        results = []
        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                # Test Prometheus health
                async with session.get(f"{self.endpoints['prometheus']}/-/healthy", timeout=10) as response:
                    if response.status == 200:
                        results.append(
                            ValidationResult(
                                component, "prometheus_health", ValidationStatus.PASS, "Prometheus is healthy"
                            )
                        )
                    else:
                        results.append(
                            ValidationResult(
                                component,
                                "prometheus_health",
                                ValidationStatus.FAIL,
                                f"Prometheus unhealthy: {response.status}",
                            )
                        )

                # Test metrics endpoint
                async with session.get(f"{self.endpoints['prometheus']}/api/v1/query?query=up", timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "success":
                            results.append(
                                ValidationResult(
                                    component,
                                    "prometheus_query",
                                    ValidationStatus.PASS,
                                    "Prometheus query API working",
                                    {"result_count": len(data.get("data", {}).get("result", []))},
                                )
                            )
                        else:
                            results.append(
                                ValidationResult(
                                    component,
                                    "prometheus_query",
                                    ValidationStatus.FAIL,
                                    "Prometheus query failed",
                                    {"response": data},
                                )
                            )
                    else:
                        results.append(
                            ValidationResult(
                                component,
                                "prometheus_query",
                                ValidationStatus.FAIL,
                                f"Prometheus query API error: {response.status}",
                            )
                        )

                # Check for AIVillage-specific metrics
                aivillage_metrics = [
                    "agent_forge_request_duration_seconds",
                    "hyperrag_query_duration_seconds",
                    "p2p_message_delivery_failures_total",
                    "http_request_duration_seconds",
                ]

                for metric in aivillage_metrics:
                    async with session.get(
                        f"{self.endpoints['prometheus']}/api/v1/query?query={metric}", timeout=5
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("data", {}).get("result"):
                                results.append(
                                    ValidationResult(
                                        component,
                                        f"metric_{metric}",
                                        ValidationStatus.PASS,
                                        f"Metric {metric} is being collected",
                                    )
                                )
                            else:
                                results.append(
                                    ValidationResult(
                                        component,
                                        f"metric_{metric}",
                                        ValidationStatus.WARN,
                                        f"Metric {metric} has no data",
                                    )
                                )

                # Check targets
                async with session.get(f"{self.endpoints['prometheus']}/api/v1/targets", timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "success":
                            targets = data.get("data", {}).get("activeTargets", [])
                            healthy_targets = [t for t in targets if t.get("health") == "up"]

                            results.append(
                                ValidationResult(
                                    component,
                                    "prometheus_targets",
                                    ValidationStatus.PASS,
                                    f"{len(healthy_targets)}/{len(targets)} targets healthy",
                                    {"healthy_targets": len(healthy_targets), "total_targets": len(targets)},
                                )
                            )

        except Exception as e:
            results.append(
                ValidationResult(
                    component, "prometheus_connectivity", ValidationStatus.FAIL, f"Failed to connect to Prometheus: {e}"
                )
            )

        execution_time = (time.time() - start_time) * 1000
        self._record_component_results(component, results, execution_time)

    async def validate_distributed_tracing(self):
        """Validate Jaeger distributed tracing"""
        component = "distributed_tracing"
        results = []
        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                # Test Jaeger health
                async with session.get(f"{self.endpoints['jaeger']}/api/health", timeout=10) as response:
                    if response.status == 200:
                        results.append(
                            ValidationResult(component, "jaeger_health", ValidationStatus.PASS, "Jaeger is healthy")
                        )
                    else:
                        results.append(
                            ValidationResult(
                                component,
                                "jaeger_health",
                                ValidationStatus.FAIL,
                                f"Jaeger unhealthy: {response.status}",
                            )
                        )

                # Test OpenTelemetry collector
                async with session.get(f"{self.endpoints['otel_collector']}/metrics", timeout=10) as response:
                    if response.status == 200:
                        results.append(
                            ValidationResult(
                                component,
                                "otel_collector_metrics",
                                ValidationStatus.PASS,
                                "OpenTelemetry collector is serving metrics",
                            )
                        )
                    else:
                        results.append(
                            ValidationResult(
                                component,
                                "otel_collector_metrics",
                                ValidationStatus.FAIL,
                                f"OTEL collector metrics unavailable: {response.status}",
                            )
                        )

                # Check for recent traces
                async with session.get(f"{self.endpoints['jaeger']}/api/services", timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        services = data.get("data", [])

                        aivillage_services = [
                            s
                            for s in services
                            if "aivillage" in s.lower() or any(svc in s for svc in ["agent-forge", "hyperrag", "p2p"])
                        ]

                        if aivillage_services:
                            results.append(
                                ValidationResult(
                                    component,
                                    "aivillage_services_traced",
                                    ValidationStatus.PASS,
                                    f"Found {len(aivillage_services)} AIVillage services in traces",
                                    {"services": aivillage_services},
                                )
                            )
                        else:
                            results.append(
                                ValidationResult(
                                    component,
                                    "aivillage_services_traced",
                                    ValidationStatus.WARN,
                                    "No AIVillage services found in traces",
                                )
                            )
                    else:
                        results.append(
                            ValidationResult(
                                component,
                                "jaeger_services",
                                ValidationStatus.WARN,
                                f"Could not retrieve Jaeger services: {response.status}",
                            )
                        )

        except Exception as e:
            results.append(
                ValidationResult(
                    component,
                    "tracing_connectivity",
                    ValidationStatus.FAIL,
                    f"Failed to connect to tracing services: {e}",
                )
            )

        execution_time = (time.time() - start_time) * 1000
        self._record_component_results(component, results, execution_time)

    async def validate_log_aggregation(self):
        """Validate Loki log aggregation"""
        component = "log_aggregation"
        results = []
        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                # Test Loki readiness
                async with session.get(f"{self.endpoints['loki']}/ready", timeout=10) as response:
                    if response.status == 200:
                        results.append(
                            ValidationResult(component, "loki_ready", ValidationStatus.PASS, "Loki is ready")
                        )
                    else:
                        results.append(
                            ValidationResult(
                                component, "loki_ready", ValidationStatus.FAIL, f"Loki not ready: {response.status}"
                            )
                        )

                # Test log query API
                query = '{job="system-logs"}'
                async with session.get(
                    f"{self.endpoints['loki']}/loki/api/v1/query?query={query}", timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "success":
                            results.append(
                                ValidationResult(
                                    component, "loki_query", ValidationStatus.PASS, "Loki query API working"
                                )
                            )
                        else:
                            results.append(
                                ValidationResult(
                                    component, "loki_query", ValidationStatus.WARN, "Loki query returned no data"
                                )
                            )
                    else:
                        results.append(
                            ValidationResult(
                                component, "loki_query", ValidationStatus.FAIL, f"Loki query failed: {response.status}"
                            )
                        )

                # Check for AIVillage log streams
                async with session.get(f"{self.endpoints['loki']}/loki/api/v1/labels", timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        labels = data.get("data", [])

                        aivillage_labels = [
                            l
                            for l in labels
                            if any(svc in l for svc in ["agent-forge", "hyperrag", "p2p", "aivillage"])
                        ]

                        if aivillage_labels:
                            results.append(
                                ValidationResult(
                                    component,
                                    "aivillage_log_streams",
                                    ValidationStatus.PASS,
                                    f"Found AIVillage log streams: {aivillage_labels}",
                                )
                            )
                        else:
                            results.append(
                                ValidationResult(
                                    component,
                                    "aivillage_log_streams",
                                    ValidationStatus.WARN,
                                    "No AIVillage-specific log streams detected",
                                )
                            )

        except Exception as e:
            results.append(
                ValidationResult(
                    component,
                    "log_aggregation_connectivity",
                    ValidationStatus.FAIL,
                    f"Failed to connect to log aggregation: {e}",
                )
            )

        execution_time = (time.time() - start_time) * 1000
        self._record_component_results(component, results, execution_time)

    async def validate_alerting_system(self):
        """Validate Alertmanager alerting system"""
        component = "alerting_system"
        results = []
        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                # Test Alertmanager health
                async with session.get(f"{self.endpoints['alertmanager']}/-/healthy", timeout=10) as response:
                    if response.status == 200:
                        results.append(
                            ValidationResult(
                                component, "alertmanager_health", ValidationStatus.PASS, "Alertmanager is healthy"
                            )
                        )
                    else:
                        results.append(
                            ValidationResult(
                                component,
                                "alertmanager_health",
                                ValidationStatus.FAIL,
                                f"Alertmanager unhealthy: {response.status}",
                            )
                        )

                # Check alert configuration
                async with session.get(f"{self.endpoints['alertmanager']}/api/v1/status", timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        config = data.get("data", {}).get("configYAML", "")

                        if "aivillage" in config.lower():
                            results.append(
                                ValidationResult(
                                    component,
                                    "alertmanager_config",
                                    ValidationStatus.PASS,
                                    "AIVillage alerting configuration detected",
                                )
                            )
                        else:
                            results.append(
                                ValidationResult(
                                    component,
                                    "alertmanager_config",
                                    ValidationStatus.WARN,
                                    "No AIVillage-specific alerting configuration found",
                                )
                            )

                # Check alert receivers
                async with session.get(f"{self.endpoints['alertmanager']}/api/v1/receivers", timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        receivers = data.get("data", [])

                        if receivers:
                            results.append(
                                ValidationResult(
                                    component,
                                    "alert_receivers",
                                    ValidationStatus.PASS,
                                    f"Found {len(receivers)} alert receivers configured",
                                )
                            )
                        else:
                            results.append(
                                ValidationResult(
                                    component, "alert_receivers", ValidationStatus.WARN, "No alert receivers configured"
                                )
                            )

        except Exception as e:
            results.append(
                ValidationResult(
                    component,
                    "alerting_connectivity",
                    ValidationStatus.FAIL,
                    f"Failed to connect to alerting system: {e}",
                )
            )

        execution_time = (time.time() - start_time) * 1000
        self._record_component_results(component, results, execution_time)

    async def validate_performance_caching(self):
        """Validate performance caching infrastructure"""
        component = "performance_caching"
        results = []
        start_time = time.time()

        try:
            # Test Redis connectivity and configuration
            redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True, password="aivillage2024")

            # Test Redis ping
            if redis_client.ping():
                results.append(
                    ValidationResult(
                        component, "redis_connectivity", ValidationStatus.PASS, "Redis master is responding"
                    )
                )

                # Test Redis configuration
                config = redis_client.config_get()

                # Check memory settings
                maxmemory = config.get("maxmemory", "0")
                maxmemory_policy = config.get("maxmemory-policy", "")

                if maxmemory != "0":
                    results.append(
                        ValidationResult(
                            component,
                            "redis_memory_config",
                            ValidationStatus.PASS,
                            f"Redis maxmemory configured: {maxmemory}",
                        )
                    )
                else:
                    results.append(
                        ValidationResult(
                            component, "redis_memory_config", ValidationStatus.WARN, "Redis maxmemory not configured"
                        )
                    )

                if maxmemory_policy in ["allkeys-lru", "allkeys-lfu"]:
                    results.append(
                        ValidationResult(
                            component,
                            "redis_eviction_policy",
                            ValidationStatus.PASS,
                            f"Appropriate eviction policy: {maxmemory_policy}",
                        )
                    )
                else:
                    results.append(
                        ValidationResult(
                            component,
                            "redis_eviction_policy",
                            ValidationStatus.WARN,
                            f"Suboptimal eviction policy: {maxmemory_policy}",
                        )
                    )

                # Test cache operations
                test_key = "aivillage:validation:test"
                test_value = "validation_data"

                redis_client.setex(test_key, 60, test_value)
                retrieved_value = redis_client.get(test_key)

                if retrieved_value == test_value:
                    results.append(
                        ValidationResult(
                            component, "redis_operations", ValidationStatus.PASS, "Redis cache operations working"
                        )
                    )
                    redis_client.delete(test_key)  # Cleanup
                else:
                    results.append(
                        ValidationResult(
                            component, "redis_operations", ValidationStatus.FAIL, "Redis cache operations failed"
                        )
                    )

            else:
                results.append(
                    ValidationResult(
                        component, "redis_connectivity", ValidationStatus.FAIL, "Redis master not responding"
                    )
                )

            # Test Memcached connectivity
            import telnetlib

            try:
                tn = telnetlib.Telnet("localhost", 11211, timeout=5)
                tn.write(b"stats\r\n")
                stats = tn.read_until(b"END\r\n", timeout=5).decode()
                tn.close()

                if "STAT" in stats:
                    results.append(
                        ValidationResult(
                            component, "memcached_connectivity", ValidationStatus.PASS, "Memcached is responding"
                        )
                    )
                else:
                    results.append(
                        ValidationResult(
                            component,
                            "memcached_connectivity",
                            ValidationStatus.WARN,
                            "Memcached responded but stats unavailable",
                        )
                    )

            except Exception as e:
                results.append(
                    ValidationResult(
                        component, "memcached_connectivity", ValidationStatus.FAIL, f"Memcached not accessible: {e}"
                    )
                )

        except Exception as e:
            results.append(
                ValidationResult(
                    component,
                    "caching_infrastructure",
                    ValidationStatus.FAIL,
                    f"Failed to validate caching infrastructure: {e}",
                )
            )

        execution_time = (time.time() - start_time) * 1000
        self._record_component_results(component, results, execution_time)

    async def validate_service_discovery(self):
        """Validate service discovery and health monitoring"""
        component = "service_discovery"
        results = []
        start_time = time.time()

        # This would integrate with the service dependency mapping
        try:
            from .service_dependency_mapping import service_monitoring_system

            # Get service health status
            system_status = await service_monitoring_system.get_comprehensive_status()

            service_health = system_status.get("service_health", {})
            healthy_services = [name for name, health in service_health.items() if health.get("status") == 1.0]

            if len(healthy_services) >= 5:  # Expect at least 5 core services
                results.append(
                    ValidationResult(
                        component,
                        "core_services_healthy",
                        ValidationStatus.PASS,
                        f"{len(healthy_services)} core services are healthy",
                        {"healthy_services": healthy_services},
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        component,
                        "core_services_healthy",
                        ValidationStatus.WARN,
                        f"Only {len(healthy_services)} core services are healthy",
                    )
                )

            # Check dependency health
            dependency_health = system_status.get("dependency_health", {})
            healthy_dependencies = [k for k, v in dependency_health.items() if v.get("status") == 1.0]

            results.append(
                ValidationResult(
                    component,
                    "service_dependencies",
                    ValidationStatus.PASS,
                    f"{len(healthy_dependencies)} service dependencies are healthy",
                    {"healthy_dependencies": len(healthy_dependencies)},
                )
            )

        except ImportError:
            results.append(
                ValidationResult(
                    component,
                    "service_monitoring_integration",
                    ValidationStatus.SKIP,
                    "Service monitoring system not available for validation",
                )
            )
        except Exception as e:
            results.append(
                ValidationResult(
                    component,
                    "service_discovery_validation",
                    ValidationStatus.FAIL,
                    f"Service discovery validation failed: {e}",
                )
            )

        execution_time = (time.time() - start_time) * 1000
        self._record_component_results(component, results, execution_time)

    async def validate_dashboard_functionality(self):
        """Validate Grafana dashboard functionality"""
        component = "dashboard_functionality"
        results = []
        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                # Test Grafana health
                async with session.get(f"{self.endpoints['grafana']}/api/health", timeout=10) as response:
                    if response.status == 200:
                        results.append(
                            ValidationResult(component, "grafana_health", ValidationStatus.PASS, "Grafana is healthy")
                        )
                    else:
                        results.append(
                            ValidationResult(
                                component,
                                "grafana_health",
                                ValidationStatus.FAIL,
                                f"Grafana unhealthy: {response.status}",
                            )
                        )

                # Test datasources
                auth = aiohttp.BasicAuth("admin", "aivillage2024")
                async with session.get(
                    f"{self.endpoints['grafana']}/api/datasources", auth=auth, timeout=10
                ) as response:
                    if response.status == 200:
                        datasources = await response.json()

                        expected_datasources = ["Prometheus", "Jaeger", "Loki"]
                        found_datasources = [ds["name"] for ds in datasources]

                        missing_datasources = [ds for ds in expected_datasources if ds not in found_datasources]

                        if not missing_datasources:
                            results.append(
                                ValidationResult(
                                    component,
                                    "grafana_datasources",
                                    ValidationStatus.PASS,
                                    "All expected datasources configured",
                                    {"datasources": found_datasources},
                                )
                            )
                        else:
                            results.append(
                                ValidationResult(
                                    component,
                                    "grafana_datasources",
                                    ValidationStatus.WARN,
                                    f"Missing datasources: {missing_datasources}",
                                )
                            )
                    else:
                        results.append(
                            ValidationResult(
                                component,
                                "grafana_datasources",
                                ValidationStatus.FAIL,
                                f"Could not retrieve datasources: {response.status}",
                            )
                        )

                # Test dashboards
                async with session.get(f"{self.endpoints['grafana']}/api/search", auth=auth, timeout=10) as response:
                    if response.status == 200:
                        dashboards = await response.json()
                        aivillage_dashboards = [d for d in dashboards if "aivillage" in d.get("title", "").lower()]

                        if aivillage_dashboards:
                            results.append(
                                ValidationResult(
                                    component,
                                    "aivillage_dashboards",
                                    ValidationStatus.PASS,
                                    f"Found {len(aivillage_dashboards)} AIVillage dashboards",
                                    {"dashboards": [d["title"] for d in aivillage_dashboards]},
                                )
                            )
                        else:
                            results.append(
                                ValidationResult(
                                    component,
                                    "aivillage_dashboards",
                                    ValidationStatus.WARN,
                                    "No AIVillage-specific dashboards found",
                                )
                            )
                    else:
                        results.append(
                            ValidationResult(
                                component,
                                "grafana_dashboards",
                                ValidationStatus.WARN,
                                f"Could not retrieve dashboards: {response.status}",
                            )
                        )

        except Exception as e:
            results.append(
                ValidationResult(
                    component,
                    "dashboard_connectivity",
                    ValidationStatus.FAIL,
                    f"Failed to connect to dashboard system: {e}",
                )
            )

        execution_time = (time.time() - start_time) * 1000
        self._record_component_results(component, results, execution_time)

    async def validate_data_retention_policies(self):
        """Validate data retention policies"""
        component = "data_retention"
        results = []
        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                # Check Prometheus retention
                async with session.get(
                    f"{self.endpoints['prometheus']}/api/v1/status/runtimeinfo", timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        storage_retention = data.get("data", {}).get("storageRetention", "")

                        if storage_retention:
                            results.append(
                                ValidationResult(
                                    component,
                                    "prometheus_retention",
                                    ValidationStatus.PASS,
                                    f"Prometheus retention configured: {storage_retention}",
                                )
                            )
                        else:
                            results.append(
                                ValidationResult(
                                    component,
                                    "prometheus_retention",
                                    ValidationStatus.WARN,
                                    "Prometheus retention not explicitly configured",
                                )
                            )

                # Check Elasticsearch retention (for Jaeger traces)
                async with session.get(
                    f"{self.endpoints['elasticsearch']}/_cat/indices?format=json", timeout=10
                ) as response:
                    if response.status == 200:
                        indices = await response.json()
                        trace_indices = [idx for idx in indices if "trace" in idx.get("index", "")]

                        if trace_indices:
                            results.append(
                                ValidationResult(
                                    component,
                                    "jaeger_trace_retention",
                                    ValidationStatus.PASS,
                                    f"Found {len(trace_indices)} trace indices",
                                    {"trace_indices": len(trace_indices)},
                                )
                            )
                        else:
                            results.append(
                                ValidationResult(
                                    component,
                                    "jaeger_trace_retention",
                                    ValidationStatus.WARN,
                                    "No trace indices found in Elasticsearch",
                                )
                            )
                    else:
                        results.append(
                            ValidationResult(
                                component,
                                "elasticsearch_indices",
                                ValidationStatus.WARN,
                                f"Could not retrieve Elasticsearch indices: {response.status}",
                            )
                        )

        except Exception as e:
            results.append(
                ValidationResult(
                    component,
                    "data_retention_validation",
                    ValidationStatus.FAIL,
                    f"Failed to validate data retention: {e}",
                )
            )

        execution_time = (time.time() - start_time) * 1000
        self._record_component_results(component, results, execution_time)

    async def validate_security_configuration(self):
        """Validate security configuration"""
        component = "security_configuration"
        results = []
        start_time = time.time()

        try:
            # Check Grafana authentication
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.endpoints['grafana']}/api/admin/users", timeout=5) as response:
                    if response.status == 401:
                        results.append(
                            ValidationResult(
                                component,
                                "grafana_auth_required",
                                ValidationStatus.PASS,
                                "Grafana requires authentication",
                            )
                        )
                    elif response.status == 200:
                        results.append(
                            ValidationResult(
                                component,
                                "grafana_auth_required",
                                ValidationStatus.WARN,
                                "Grafana allows unauthenticated admin access",
                            )
                        )

                # Check Prometheus security
                async with session.get(
                    f"{self.endpoints['prometheus']}/api/v1/admin/tsdb/snapshot", timeout=5
                ) as response:
                    if response.status == 403:
                        results.append(
                            ValidationResult(
                                component,
                                "prometheus_admin_disabled",
                                ValidationStatus.PASS,
                                "Prometheus admin API properly secured",
                            )
                        )
                    else:
                        results.append(
                            ValidationResult(
                                component,
                                "prometheus_admin_disabled",
                                ValidationStatus.WARN,
                                "Prometheus admin API may be accessible",
                            )
                        )

            # Check Redis authentication
            try:
                redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
                redis_client.ping()  # Should fail without password
                results.append(
                    ValidationResult(
                        component, "redis_auth_required", ValidationStatus.WARN, "Redis does not require authentication"
                    )
                )
            except redis.AuthenticationError:
                results.append(
                    ValidationResult(
                        component, "redis_auth_required", ValidationStatus.PASS, "Redis requires authentication"
                    )
                )
            except Exception:
                results.append(
                    ValidationResult(
                        component, "redis_auth_check", ValidationStatus.SKIP, "Could not test Redis authentication"
                    )
                )

        except Exception as e:
            results.append(
                ValidationResult(
                    component, "security_validation", ValidationStatus.FAIL, f"Security validation failed: {e}"
                )
            )

        execution_time = (time.time() - start_time) * 1000
        self._record_component_results(component, results, execution_time)

    def _record_component_results(self, component: str, results: list[ValidationResult], execution_time: float):
        """Record results for a component"""
        self.validation_results.extend(results)

        # Calculate component status
        passed = len([r for r in results if r.status == ValidationStatus.PASS])
        failed = len([r for r in results if r.status == ValidationStatus.FAIL])
        warned = len([r for r in results if r.status == ValidationStatus.WARN])
        skipped = len([r for r in results if r.status == ValidationStatus.SKIP])

        # Determine overall status
        if failed > 0:
            overall_status = ValidationStatus.FAIL
        elif warned > 0:
            overall_status = ValidationStatus.WARN
        else:
            overall_status = ValidationStatus.PASS

        # Critical issues
        critical_issues = [r.message for r in results if r.status == ValidationStatus.FAIL]

        self.component_statuses[component] = ComponentStatus(
            component=component,
            overall_status=overall_status,
            tests_passed=passed,
            tests_failed=failed,
            tests_warned=warned,
            tests_skipped=skipped,
            execution_time_ms=execution_time,
            critical_issues=critical_issues,
        )

    def _calculate_overall_status(self) -> dict[str, Any]:
        """Calculate overall platform status"""
        total_components = len(self.component_statuses)
        failed_components = len(
            [s for s in self.component_statuses.values() if s.overall_status == ValidationStatus.FAIL]
        )
        warned_components = len(
            [s for s in self.component_statuses.values() if s.overall_status == ValidationStatus.WARN]
        )
        passed_components = len(
            [s for s in self.component_statuses.values() if s.overall_status == ValidationStatus.PASS]
        )

        if failed_components > 0:
            overall_status = ValidationStatus.FAIL
            status_message = f"Platform validation FAILED - {failed_components} components have critical issues"
        elif warned_components > 0:
            overall_status = ValidationStatus.WARN
            status_message = f"Platform validation has WARNINGS - {warned_components} components need attention"
        else:
            overall_status = ValidationStatus.PASS
            status_message = f"Platform validation PASSED - All {passed_components} components are healthy"

        return {
            "status": overall_status,
            "message": status_message,
            "summary": {
                "total_components": total_components,
                "passed": passed_components,
                "warned": warned_components,
                "failed": failed_components,
            },
        }

    def _calculate_readiness_score(self) -> dict[str, Any]:
        """Calculate production readiness score"""
        total_tests = len(self.validation_results)
        passed_tests = len([r for r in self.validation_results if r.status == ValidationStatus.PASS])

        if total_tests == 0:
            score = 0
        else:
            score = (passed_tests / total_tests) * 100

        # Determine readiness level
        if score >= 95:
            readiness_level = "PRODUCTION_READY"
        elif score >= 85:
            readiness_level = "STAGING_READY"
        elif score >= 70:
            readiness_level = "DEVELOPMENT_READY"
        else:
            readiness_level = "NOT_READY"

        return {
            "score": score,
            "readiness_level": readiness_level,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "recommendation": self._get_readiness_recommendation(score),
        }

    def _get_readiness_recommendation(self, score: float) -> str:
        """Get readiness recommendation based on score"""
        if score >= 95:
            return "Platform is production-ready. Deploy with confidence."
        elif score >= 85:
            return "Platform is staging-ready. Address warnings before production deployment."
        elif score >= 70:
            return "Platform is development-ready. Significant improvements needed for production."
        else:
            return "Platform is not ready for deployment. Address critical failures immediately."

    def _generate_recommendations(self) -> list[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Component-specific recommendations
        for component, status in self.component_statuses.items():
            if status.overall_status == ValidationStatus.FAIL:
                recommendations.append(
                    f"🚨 CRITICAL: Fix {component} failures immediately - {', '.join(status.critical_issues)}"
                )
            elif status.overall_status == ValidationStatus.WARN:
                recommendations.append(
                    f"⚠️ WARNING: Address {component} issues - {status.tests_warned} warnings detected"
                )

        # General recommendations
        failed_components = [s for s in self.component_statuses.values() if s.overall_status == ValidationStatus.FAIL]
        if failed_components:
            recommendations.append("🔧 PRIORITY: Focus on critical components before proceeding with deployment")

        warned_components = [s for s in self.component_statuses.values() if s.overall_status == ValidationStatus.WARN]
        if warned_components:
            recommendations.append("📊 OPTIMIZE: Review warnings to improve platform reliability")

        if not failed_components and not warned_components:
            recommendations.append("✅ EXCELLENT: Platform is fully operational and production-ready")

        return recommendations


# Global validator instance
platform_validator = ObservabilityValidator()


async def validate_production_readiness() -> dict[str, Any]:
    """Run complete production readiness validation"""
    global platform_validator
    platform_validator = ObservabilityValidator()
    return await platform_validator.validate_complete_platform()


if __name__ == "__main__":
    # Run validation if executed directly
    import asyncio

    result = asyncio.run(validate_production_readiness())
    print(json.dumps(result, indent=2, default=str))
