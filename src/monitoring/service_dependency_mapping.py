"""
AIVillage Service Dependency Mapping and Health Monitoring
Provides comprehensive service topology and health status tracking
"""

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
import logging
import time
from typing import Any

import networkx as nx
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# Service health metrics
SERVICE_HEALTH_STATUS = Gauge("service_health_status", "Service health status", ["service", "component"])
SERVICE_DEPENDENCY_HEALTH = Gauge("service_dependency_health", "Dependency health status", ["source", "target", "type"])
SERVICE_RESPONSE_TIME = Histogram(
    "service_dependency_response_time_seconds", "Service dependency response time", ["source", "target"]
)
DEPENDENCY_FAILURES = Counter(
    "service_dependency_failures_total", "Service dependency failures", ["source", "target", "error_type"]
)


class ServiceStatus(Enum):
    """Service health status levels"""

    HEALTHY = 1.0
    DEGRADED = 0.5
    UNHEALTHY = 0.0
    UNKNOWN = -1.0


class DependencyType(Enum):
    """Types of service dependencies"""

    HTTP_API = "http_api"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    P2P_NETWORK = "p2p_network"
    STORAGE = "storage"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class ServiceInfo:
    """Service information and metadata"""

    name: str
    component: str
    version: str
    host: str
    port: int
    health_endpoint: str = "/health"
    metrics_endpoint: str = "/metrics"
    description: str = ""
    tags: list[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class ServiceDependency:
    """Service dependency definition"""

    source_service: str
    target_service: str
    dependency_type: DependencyType
    endpoint: str
    timeout_seconds: float = 5.0
    critical: bool = False
    description: str = ""


@dataclass
class HealthCheckResult:
    """Health check result"""

    service: str
    status: ServiceStatus
    response_time_ms: float
    timestamp: datetime
    error_message: str | None = None
    details: dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class ServiceRegistry:
    """Registry of all AIVillage services and their dependencies"""

    def __init__(self):
        self.services: dict[str, ServiceInfo] = {}
        self.dependencies: list[ServiceDependency] = []
        self.dependency_graph = nx.DiGraph()
        self._initialize_aivillage_services()

    def _initialize_aivillage_services(self):
        """Initialize known AIVillage services"""

        # Core services
        self.register_service(
            ServiceInfo(
                name="agent_forge",
                component="ml-pipeline",
                version="2.0.0",
                host="localhost",
                port=8001,
                description="7-phase ML pipeline with 2.8-4.4x performance boost",
                tags=["core", "ml", "pipeline"],
            )
        )

        self.register_service(
            ServiceInfo(
                name="hyperrag",
                component="memory-system",
                version="1.0.0",
                host="localhost",
                port=8010,
                description="Neural-biological memory system with ~1.19ms queries",
                tags=["core", "memory", "rag"],
            )
        )

        self.register_service(
            ServiceInfo(
                name="p2p-mesh",
                component="networking",
                version="1.0.0",
                host="localhost",
                port=9001,
                description="LibP2P mesh networking with >95% message delivery",
                tags=["core", "networking", "p2p"],
            )
        )

        self.register_service(
            ServiceInfo(
                name="api-gateway",
                component="api-gateway",
                version="1.0.0",
                host="localhost",
                port=8000,
                description="Unified REST API gateway with <100ms response times",
                tags=["core", "api", "gateway"],
            )
        )

        self.register_service(
            ServiceInfo(
                name="edge-computing",
                component="edge-device",
                version="1.0.0",
                host="localhost",
                port=8020,
                description="Battery-efficient edge computing infrastructure",
                tags=["core", "edge", "mobile"],
            )
        )

        # Infrastructure services
        self.register_service(
            ServiceInfo(
                name="postgresql",
                component="database",
                version="15",
                host="localhost",
                port=5432,
                health_endpoint="/health",
                description="Primary database with optimized connection pooling",
                tags=["infrastructure", "database"],
            )
        )

        self.register_service(
            ServiceInfo(
                name="redis-master",
                component="cache",
                version="7",
                host="localhost",
                port=6379,
                health_endpoint="/health",
                description="Redis master for high-performance caching",
                tags=["infrastructure", "cache", "master"],
            )
        )

        self.register_service(
            ServiceInfo(
                name="neo4j",
                component="graph-database",
                version="5",
                host="localhost",
                port=7474,
                health_endpoint="/db/data/",
                description="Graph database for knowledge relationships",
                tags=["infrastructure", "database", "graph"],
            )
        )

        # Monitoring services
        self.register_service(
            ServiceInfo(
                name="prometheus",
                component="monitoring",
                version="2.45.0",
                host="localhost",
                port=9090,
                description="Metrics collection and alerting",
                tags=["monitoring", "metrics"],
            )
        )

        self.register_service(
            ServiceInfo(
                name="grafana",
                component="visualization",
                version="10.0.0",
                host="localhost",
                port=3000,
                description="Monitoring dashboards and visualization",
                tags=["monitoring", "visualization"],
            )
        )

        self.register_service(
            ServiceInfo(
                name="jaeger",
                component="tracing",
                version="1.47.0",
                host="localhost",
                port=16686,
                description="Distributed tracing and observability",
                tags=["monitoring", "tracing"],
            )
        )

        # Register service dependencies
        self._initialize_service_dependencies()

    def _initialize_service_dependencies(self):
        """Initialize service dependency relationships"""

        # Agent Forge dependencies
        self.add_dependency(
            ServiceDependency(
                "agent_forge",
                "postgresql",
                DependencyType.DATABASE,
                "/health",
                critical=True,
                description="Model storage and pipeline state",
            )
        )
        self.add_dependency(
            ServiceDependency(
                "agent_forge",
                "redis-master",
                DependencyType.CACHE,
                "/health",
                critical=False,
                description="Pipeline result caching",
            )
        )

        # HyperRAG dependencies
        self.add_dependency(
            ServiceDependency(
                "hyperrag",
                "neo4j",
                DependencyType.DATABASE,
                "/db/data/",
                critical=True,
                description="Knowledge graph storage",
            )
        )
        self.add_dependency(
            ServiceDependency(
                "hyperrag",
                "postgresql",
                DependencyType.DATABASE,
                "/health",
                critical=True,
                description="Vector embeddings and metadata",
            )
        )

        # API Gateway dependencies
        self.add_dependency(
            ServiceDependency(
                "api-gateway",
                "agent_forge",
                DependencyType.HTTP_API,
                "/health",
                critical=True,
                description="ML pipeline access",
            )
        )
        self.add_dependency(
            ServiceDependency(
                "api-gateway",
                "hyperrag",
                DependencyType.HTTP_API,
                "/health",
                critical=True,
                description="Memory system access",
            )
        )
        self.add_dependency(
            ServiceDependency(
                "api-gateway",
                "p2p-mesh",
                DependencyType.HTTP_API,
                "/health",
                critical=False,
                description="P2P network access",
            )
        )

        # P2P mesh dependencies
        self.add_dependency(
            ServiceDependency(
                "p2p-mesh",
                "postgresql",
                DependencyType.DATABASE,
                "/health",
                critical=False,
                description="Peer registry and routing tables",
            )
        )

        # Edge computing dependencies
        self.add_dependency(
            ServiceDependency(
                "edge-computing",
                "api-gateway",
                DependencyType.HTTP_API,
                "/health",
                critical=True,
                description="Central API access",
            )
        )
        self.add_dependency(
            ServiceDependency(
                "edge-computing",
                "p2p-mesh",
                DependencyType.P2P_NETWORK,
                "/health",
                critical=False,
                description="Mesh network communication",
            )
        )

        # Monitoring dependencies
        self.add_dependency(
            ServiceDependency(
                "grafana",
                "prometheus",
                DependencyType.HTTP_API,
                "/health",
                critical=True,
                description="Metrics data source",
            )
        )
        self.add_dependency(
            ServiceDependency(
                "grafana",
                "jaeger",
                DependencyType.HTTP_API,
                "/health",
                critical=False,
                description="Tracing data source",
            )
        )

    def register_service(self, service_info: ServiceInfo):
        """Register a service in the registry"""
        self.services[service_info.name] = service_info
        self.dependency_graph.add_node(service_info.name, **asdict(service_info))
        logger.info(f"Registered service: {service_info.name}")

    def add_dependency(self, dependency: ServiceDependency):
        """Add a service dependency"""
        self.dependencies.append(dependency)
        self.dependency_graph.add_edge(dependency.source_service, dependency.target_service, **asdict(dependency))
        logger.debug(f"Added dependency: {dependency.source_service} -> {dependency.target_service}")

    def get_service_dependencies(self, service_name: str) -> list[ServiceDependency]:
        """Get all dependencies for a service"""
        return [dep for dep in self.dependencies if dep.source_service == service_name]

    def get_dependent_services(self, service_name: str) -> list[str]:
        """Get services that depend on this service"""
        return [dep.source_service for dep in self.dependencies if dep.target_service == service_name]

    def get_critical_path(self, service_name: str) -> list[str]:
        """Get critical dependency path for a service"""
        try:
            # Find critical dependencies (nodes this service critically depends on)
            critical_deps = []
            for dep in self.get_service_dependencies(service_name):
                if dep.critical:
                    critical_deps.append(dep.target_service)
                    # Recursively find critical dependencies
                    critical_deps.extend(self.get_critical_path(dep.target_service))

            return list(set(critical_deps))  # Remove duplicates
        except RecursionError:
            logger.warning(f"Circular dependency detected for service: {service_name}")
            return []


class HealthChecker:
    """Health checker for services and dependencies"""

    def __init__(self, service_registry: ServiceRegistry):
        self.registry = service_registry
        self.health_results: dict[str, HealthCheckResult] = {}
        self.dependency_results: dict[tuple[str, str], HealthCheckResult] = {}

    async def check_service_health(self, service_name: str) -> HealthCheckResult:
        """Check health of a specific service"""
        service_info = self.registry.services.get(service_name)
        if not service_info:
            return HealthCheckResult(
                service=service_name,
                status=ServiceStatus.UNKNOWN,
                response_time_ms=0,
                timestamp=datetime.utcnow(),
                error_message="Service not registered",
            )

        start_time = time.time()

        try:
            # Simulate health check (in production, make actual HTTP call)
            await asyncio.sleep(0.01)  # Simulate network call

            # For demo purposes, simulate some failures
            import random

            if random.random() < 0.05:  # 5% failure rate
                raise Exception("Simulated service failure")

            response_time = (time.time() - start_time) * 1000

            # Determine status based on response time
            if response_time > 5000:  # >5 seconds
                status = ServiceStatus.UNHEALTHY
            elif response_time > 1000:  # >1 second
                status = ServiceStatus.DEGRADED
            else:
                status = ServiceStatus.HEALTHY

            result = HealthCheckResult(
                service=service_name,
                status=status,
                response_time_ms=response_time,
                timestamp=datetime.utcnow(),
                details={"host": service_info.host, "port": service_info.port, "version": service_info.version},
            )

            # Record metrics
            SERVICE_HEALTH_STATUS.labels(service=service_name, component=service_info.component).set(status.value)

            SERVICE_RESPONSE_TIME.labels(source="health_checker", target=service_name).observe(response_time / 1000)

            return result

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

            result = HealthCheckResult(
                service=service_name,
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=response_time,
                timestamp=datetime.utcnow(),
                error_message=str(e),
            )

            # Record metrics
            SERVICE_HEALTH_STATUS.labels(service=service_name, component=service_info.component).set(
                ServiceStatus.UNHEALTHY.value
            )

            DEPENDENCY_FAILURES.labels(source="health_checker", target=service_name, error_type=type(e).__name__).inc()

            return result

    async def check_dependency_health(self, dependency: ServiceDependency) -> HealthCheckResult:
        """Check health of a service dependency"""
        start_time = time.time()

        try:
            # Check if target service is healthy first
            target_health = await self.check_service_health(dependency.target_service)

            if target_health.status == ServiceStatus.UNHEALTHY:
                return HealthCheckResult(
                    service=f"{dependency.source_service}->{dependency.target_service}",
                    status=ServiceStatus.UNHEALTHY,
                    response_time_ms=target_health.response_time_ms,
                    timestamp=datetime.utcnow(),
                    error_message=f"Target service {dependency.target_service} is unhealthy",
                )

            # Simulate dependency check
            await asyncio.sleep(dependency.timeout_seconds / 100)  # Scaled down for demo

            response_time = (time.time() - start_time) * 1000

            result = HealthCheckResult(
                service=f"{dependency.source_service}->{dependency.target_service}",
                status=ServiceStatus.HEALTHY,
                response_time_ms=response_time,
                timestamp=datetime.utcnow(),
                details={
                    "dependency_type": dependency.dependency_type.value,
                    "endpoint": dependency.endpoint,
                    "critical": dependency.critical,
                },
            )

            # Record metrics
            SERVICE_DEPENDENCY_HEALTH.labels(
                source=dependency.source_service,
                target=dependency.target_service,
                type=dependency.dependency_type.value,
            ).set(ServiceStatus.HEALTHY.value)

            return result

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

            result = HealthCheckResult(
                service=f"{dependency.source_service}->{dependency.target_service}",
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=response_time,
                timestamp=datetime.utcnow(),
                error_message=str(e),
            )

            # Record metrics
            SERVICE_DEPENDENCY_HEALTH.labels(
                source=dependency.source_service,
                target=dependency.target_service,
                type=dependency.dependency_type.value,
            ).set(ServiceStatus.UNHEALTHY.value)

            DEPENDENCY_FAILURES.labels(
                source=dependency.source_service, target=dependency.target_service, error_type=type(e).__name__
            ).inc()

            return result

    async def check_all_services(self) -> dict[str, HealthCheckResult]:
        """Check health of all registered services"""
        tasks = []
        for service_name in self.registry.services:
            tasks.append(self.check_service_health(service_name))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        health_results = {}
        for i, result in enumerate(results):
            service_name = list(self.registry.services.keys())[i]
            if isinstance(result, Exception):
                health_results[service_name] = HealthCheckResult(
                    service=service_name,
                    status=ServiceStatus.UNHEALTHY,
                    response_time_ms=0,
                    timestamp=datetime.utcnow(),
                    error_message=str(result),
                )
            else:
                health_results[service_name] = result

        self.health_results = health_results
        return health_results

    async def check_all_dependencies(self) -> dict[tuple[str, str], HealthCheckResult]:
        """Check health of all service dependencies"""
        tasks = []
        for dependency in self.registry.dependencies:
            tasks.append(self.check_dependency_health(dependency))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        dependency_results = {}
        for i, result in enumerate(results):
            dependency = self.registry.dependencies[i]
            key = (dependency.source_service, dependency.target_service)

            if isinstance(result, Exception):
                dependency_results[key] = HealthCheckResult(
                    service=f"{dependency.source_service}->{dependency.target_service}",
                    status=ServiceStatus.UNHEALTHY,
                    response_time_ms=0,
                    timestamp=datetime.utcnow(),
                    error_message=str(result),
                )
            else:
                dependency_results[key] = result

        self.dependency_results = dependency_results
        return dependency_results


class ServiceTopologyAnalyzer:
    """Analyze service topology and provide insights"""

    def __init__(self, service_registry: ServiceRegistry, health_checker: HealthChecker):
        self.registry = service_registry
        self.health_checker = health_checker

    def analyze_service_impact(self, service_name: str) -> dict[str, Any]:
        """Analyze the impact if a service goes down"""
        graph = self.registry.dependency_graph

        # Find services that directly depend on this service
        direct_dependents = self.registry.get_dependent_services(service_name)

        # Find all services that would be affected (transitive dependencies)
        affected_services = set()
        for dependent in direct_dependents:
            affected_services.add(dependent)
            # Use graph traversal to find transitive dependents
            try:
                descendants = nx.descendants(graph, dependent)
                affected_services.update(descendants)
            except nx.NetworkXError:
                pass

        # Classify impact level
        impact_level = "low"
        if len(affected_services) > 5:
            impact_level = "critical"
        elif len(affected_services) > 2:
            impact_level = "high"
        elif len(affected_services) > 0:
            impact_level = "medium"

        return {
            "service": service_name,
            "impact_level": impact_level,
            "direct_dependents": direct_dependents,
            "total_affected_services": len(affected_services),
            "affected_services": list(affected_services),
            "critical_dependencies": self.registry.get_critical_path(service_name),
        }

    def get_service_criticality_ranking(self) -> list[dict[str, Any]]:
        """Rank services by criticality based on dependency impact"""
        service_criticality = []

        for service_name in self.registry.services:
            impact = self.analyze_service_impact(service_name)
            service_info = self.registry.services[service_name]

            criticality_score = len(impact["affected_services"])

            # Boost score for core services
            if "core" in service_info.tags:
                criticality_score *= 2

            service_criticality.append(
                {
                    "service": service_name,
                    "criticality_score": criticality_score,
                    "impact_analysis": impact,
                    "service_info": service_info,
                }
            )

        # Sort by criticality score
        service_criticality.sort(key=lambda x: x["criticality_score"], reverse=True)

        return service_criticality

    def generate_topology_report(self) -> dict[str, Any]:
        """Generate comprehensive topology analysis report"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_services": len(self.registry.services),
            "total_dependencies": len(self.registry.dependencies),
            "service_criticality_ranking": self.get_service_criticality_ranking(),
            "graph_metrics": {
                "nodes": self.registry.dependency_graph.number_of_nodes(),
                "edges": self.registry.dependency_graph.number_of_edges(),
                "strongly_connected_components": len(
                    list(nx.strongly_connected_components(self.registry.dependency_graph))
                ),
                "is_dag": nx.is_directed_acyclic_graph(self.registry.dependency_graph),
            },
            "health_summary": self._get_health_summary(),
        }

    def _get_health_summary(self) -> dict[str, Any]:
        """Get summary of current service health"""
        if not self.health_checker.health_results:
            return {"status": "no_health_data"}

        healthy_count = sum(
            1 for result in self.health_checker.health_results.values() if result.status == ServiceStatus.HEALTHY
        )
        degraded_count = sum(
            1 for result in self.health_checker.health_results.values() if result.status == ServiceStatus.DEGRADED
        )
        unhealthy_count = sum(
            1 for result in self.health_checker.health_results.values() if result.status == ServiceStatus.UNHEALTHY
        )

        return {
            "healthy_services": healthy_count,
            "degraded_services": degraded_count,
            "unhealthy_services": unhealthy_count,
            "total_services": len(self.health_checker.health_results),
            "overall_health_percentage": (healthy_count / len(self.health_checker.health_results)) * 100,
        }


# Global service monitoring system
class ServiceMonitoringSystem:
    """Unified service monitoring and dependency tracking system"""

    def __init__(self):
        self.registry = ServiceRegistry()
        self.health_checker = HealthChecker(self.registry)
        self.topology_analyzer = ServiceTopologyAnalyzer(self.registry, self.health_checker)
        self.monitoring_active = False

    async def start_monitoring(self, check_interval: int = 30):
        """Start continuous service monitoring"""
        self.monitoring_active = True
        logger.info("Starting AIVillage service monitoring system")

        while self.monitoring_active:
            try:
                # Check all services and dependencies
                await self.health_checker.check_all_services()
                await self.health_checker.check_all_dependencies()

                logger.info("Service health checks completed")

                # Wait for next check interval
                await asyncio.sleep(check_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)  # Brief pause before retrying

    def stop_monitoring(self):
        """Stop service monitoring"""
        self.monitoring_active = False
        logger.info("Stopped service monitoring")

    async def get_comprehensive_status(self) -> dict[str, Any]:
        """Get comprehensive system status"""
        # Ensure we have recent health data
        await self.health_checker.check_all_services()
        await self.health_checker.check_all_dependencies()

        return {
            "system_overview": self.topology_analyzer.generate_topology_report(),
            "service_health": {name: asdict(result) for name, result in self.health_checker.health_results.items()},
            "dependency_health": {
                f"{k[0]}->{k[1]}": asdict(result) for k, result in self.health_checker.dependency_results.items()
            },
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> list[str]:
        """Generate operational recommendations based on current state"""
        recommendations = []

        # Check for unhealthy core services
        for service_name, health_result in self.health_checker.health_results.items():
            service_info = self.registry.services[service_name]

            if health_result.status == ServiceStatus.UNHEALTHY and "core" in service_info.tags:
                recommendations.append(f"🚨 Core service {service_name} is unhealthy - immediate attention required")
            elif health_result.status == ServiceStatus.DEGRADED and "core" in service_info.tags:
                recommendations.append(f"⚠️ Core service {service_name} is degraded - investigate performance issues")

        # Check for dependency failures
        critical_deps_failed = []
        for (source, target), dep_result in self.health_checker.dependency_results.items():
            if dep_result.status == ServiceStatus.UNHEALTHY:
                # Check if this is a critical dependency
                for dep in self.registry.dependencies:
                    if dep.source_service == source and dep.target_service == target and dep.critical:
                        critical_deps_failed.append(f"{source} -> {target}")
                        break

        if critical_deps_failed:
            recommendations.append(f"💥 Critical dependencies failed: {', '.join(critical_deps_failed)}")

        # Performance recommendations
        slow_services = [
            name for name, result in self.health_checker.health_results.items() if result.response_time_ms > 1000
        ]
        if slow_services:
            recommendations.append(f"🐌 Slow response times detected: {', '.join(slow_services)}")

        if not recommendations:
            recommendations.append("✅ All systems operating normally")

        return recommendations


# Global monitoring system instance
service_monitoring_system = ServiceMonitoringSystem()


async def initialize_service_monitoring():
    """Initialize service dependency mapping and health monitoring"""
    global service_monitoring_system
    service_monitoring_system = ServiceMonitoringSystem()
    logger.info("AIVillage service dependency mapping and health monitoring initialized")
    return service_monitoring_system
