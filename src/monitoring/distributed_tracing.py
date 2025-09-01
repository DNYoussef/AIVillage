"""
AIVillage Distributed Tracing Integration
Provides OpenTelemetry instrumentation for all services
"""

from contextlib import contextmanager
from functools import wraps
import logging

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

logger = logging.getLogger(__name__)


class AIVillageTracing:
    """Central tracing configuration for AIVillage ecosystem"""

    def __init__(self, service_name: str, service_version: str = "1.0.0"):
        self.service_name = service_name
        self.service_version = service_version
        self.tracer = None
        self.meter = None
        self._initialized = False

    def initialize(
        self, otlp_endpoint: str = "http://localhost:4317", sampling_rate: float = 0.1, environment: str = "production"
    ) -> None:
        """Initialize OpenTelemetry tracing for the service"""

        if self._initialized:
            return

        try:
            # Create resource with service information
            resource = Resource.create(
                {
                    ResourceAttributes.SERVICE_NAME: self.service_name,
                    ResourceAttributes.SERVICE_VERSION: self.service_version,
                    ResourceAttributes.SERVICE_NAMESPACE: "aivillage",
                    ResourceAttributes.DEPLOYMENT_ENVIRONMENT: environment,
                    "cluster": "aivillage-main",
                    "component": self._get_component_type(),
                }
            )

            # Configure tracing
            trace.set_tracer_provider(TracerProvider(resource=resource))
            tracer_provider = trace.get_tracer_provider()

            # OTLP Span Exporter
            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint, insecure=True, headers={"service-name": self.service_name}
            )

            # Batch processor for better performance
            span_processor = BatchSpanProcessor(
                otlp_exporter,
                max_queue_size=2048,
                export_timeout_millis=30000,
                max_export_batch_size=512,
                schedule_delay_millis=1000,
            )
            tracer_provider.add_span_processor(span_processor)

            # Configure metrics
            metric_reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True), export_interval_millis=10000
            )
            metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[metric_reader]))

            # Set global propagator
            set_global_textmap(B3MultiFormat())

            # Get tracer and meter instances
            self.tracer = trace.get_tracer(self.service_name, self.service_version)
            self.meter = metrics.get_meter(self.service_name, self.service_version)

            # Auto-instrument popular libraries
            self._setup_auto_instrumentation()

            self._initialized = True
            logger.info(f"OpenTelemetry initialized for {self.service_name}")

        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            raise

    def _get_component_type(self) -> str:
        """Determine component type based on service name"""
        component_map = {
            "agent_forge": "ml-pipeline",
            "hyperrag": "memory-system",
            "p2p-mesh": "networking",
            "api-gateway": "api-gateway",
            "edge-computing": "edge-device",
            "service-mesh": "infrastructure",
        }

        for key, component in component_map.items():
            if key in self.service_name.lower():
                return component

        return "application"

    def _setup_auto_instrumentation(self) -> None:
        """Setup automatic instrumentation for common libraries"""
        try:
            # Web frameworks
            FlaskInstrumentor().instrument()

            # HTTP clients
            RequestsInstrumentor().instrument()

            # Database
            Psycopg2Instrumentor().instrument()
            RedisInstrumentor().instrument()

            # Async support
            AsyncioInstrumentor().instrument()

            logger.info("Auto-instrumentation configured")

        except Exception as e:
            logger.warning(f"Some auto-instrumentation failed: {e}")

    @contextmanager
    def trace_operation(self, operation_name: str, **attributes):
        """Context manager for tracing operations"""
        if not self.tracer:
            yield None
            return

        with self.tracer.start_as_current_span(operation_name) as span:
            # Add custom attributes
            for key, value in attributes.items():
                span.set_attribute(key, value)

            try:
                yield span
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                span.set_attribute("error.type", type(e).__name__)
                raise

    def trace_function(self, operation_name: str | None = None, **attributes):
        """Decorator for tracing functions"""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = operation_name or f"{self.service_name}.{func.__name__}"
                with self.trace_operation(name, **attributes):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def create_counter(self, name: str, description: str = "", unit: str = ""):
        """Create a counter metric"""
        if not self.meter:
            return None
        return self.meter.create_counter(name=f"{self.service_name}.{name}", description=description, unit=unit)

    def create_histogram(self, name: str, description: str = "", unit: str = ""):
        """Create a histogram metric"""
        if not self.meter:
            return None
        return self.meter.create_histogram(name=f"{self.service_name}.{name}", description=description, unit=unit)

    def create_gauge(self, name: str, description: str = "", unit: str = ""):
        """Create a gauge metric"""
        if not self.meter:
            return None
        return self.meter.create_observable_gauge(
            name=f"{self.service_name}.{name}", description=description, unit=unit
        )


# Service-specific tracing instances
agent_forge_tracing = AIVillageTracing("agent_forge", "2.0.0")
hyperrag_tracing = AIVillageTracing("hyperrag", "1.0.0")
p2p_tracing = AIVillageTracing("p2p-mesh", "1.0.0")
api_gateway_tracing = AIVillageTracing("api-gateway", "1.0.0")
edge_tracing = AIVillageTracing("edge-computing", "1.0.0")


# Convenience functions for common operations
def trace_agent_forge_operation(operation_name: str, **attributes):
    """Trace Agent Forge pipeline operations"""
    return agent_forge_tracing.trace_operation(f"agent_forge.{operation_name}", **attributes)


def trace_hyperrag_query(query_type: str, **attributes):
    """Trace HyperRAG memory queries"""
    return hyperrag_tracing.trace_operation(f"hyperrag.query.{query_type}", **attributes)


def trace_p2p_message(message_type: str, **attributes):
    """Trace P2P message operations"""
    return p2p_tracing.trace_operation(f"p2p.message.{message_type}", **attributes)


def initialize_service_tracing(
    service_name: str, otlp_endpoint: str = "http://localhost:4317", environment: str = "production"
) -> AIVillageTracing:
    """Initialize tracing for any AIVillage service"""
    tracing = AIVillageTracing(service_name)
    tracing.initialize(otlp_endpoint=otlp_endpoint, environment=environment)
    return tracing


# Performance metrics helpers
class PerformanceMetrics:
    """Helper class for tracking performance metrics"""

    def __init__(self, tracing_instance: AIVillageTracing):
        self.tracing = tracing_instance
        self.request_counter = tracing_instance.create_counter("requests_total", "Total number of requests")
        self.request_duration = tracing_instance.create_histogram(
            "request_duration_seconds", "Request duration in seconds", "s"
        )
        self.error_counter = tracing_instance.create_counter("errors_total", "Total number of errors")

    @contextmanager
    def track_request(self, endpoint: str, method: str = "GET"):
        """Track request metrics"""
        import time

        start_time = time.time()

        attributes = {"endpoint": endpoint, "method": method}

        try:
            if self.request_counter:
                self.request_counter.add(1, attributes)

            yield

            attributes["status"] = "success"

        except Exception as e:
            if self.error_counter:
                self.error_counter.add(1, {**attributes, "error_type": type(e).__name__})
            attributes["status"] = "error"
            raise

        finally:
            if self.request_duration:
                duration = time.time() - start_time
                self.request_duration.record(duration, attributes)


# Initialize metrics for core services
agent_forge_metrics = PerformanceMetrics(agent_forge_tracing)
hyperrag_metrics = PerformanceMetrics(hyperrag_tracing)
p2p_metrics = PerformanceMetrics(p2p_tracing)
api_gateway_metrics = PerformanceMetrics(api_gateway_tracing)


# Export initialization function
def initialize_all_services():
    """Initialize tracing for all core AIVillage services"""
    services = [agent_forge_tracing, hyperrag_tracing, p2p_tracing, api_gateway_tracing, edge_tracing]

    for service_tracing in services:
        try:
            service_tracing.initialize()
            logger.info(f"Initialized tracing for {service_tracing.service_name}")
        except Exception as e:
            logger.error(f"Failed to initialize tracing for {service_tracing.service_name}: {e}")


if __name__ == "__main__":
    # Initialize all services if run directly
    initialize_all_services()
