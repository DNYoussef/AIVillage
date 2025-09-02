"""
Service-Specific Instrumentation for AIVillage Components
Provides custom tracing and metrics for each core service
"""

from contextlib import contextmanager
import logging
import time

from .distributed_tracing import (
    agent_forge_tracing,
    api_gateway_tracing,
    edge_tracing,
    hyperrag_tracing,
    p2p_tracing,
    trace_agent_forge_operation,
    trace_hyperrag_query,
    trace_p2p_message,
)

logger = logging.getLogger(__name__)


class AgentForgeInstrumentation:
    """Instrumentation for Agent Forge 7-phase pipeline"""

    def __init__(self):
        self.phase_duration = agent_forge_tracing.create_histogram(
            "phase_duration_seconds", "Duration of each pipeline phase", "s"
        )
        self.phase_counter = agent_forge_tracing.create_counter("phases_executed_total", "Total phases executed")
        self.model_operations = agent_forge_tracing.create_counter("model_operations_total", "Total model operations")
        self.pipeline_success = agent_forge_tracing.create_counter(
            "pipeline_success_total", "Successful pipeline completions"
        )

    @contextmanager
    def trace_pipeline_phase(self, phase_name: str, model_name: str = "", **attributes):
        """Trace a specific pipeline phase"""
        start_time = time.time()

        phase_attributes = {"phase": phase_name, "model": model_name, **attributes}

        with trace_agent_forge_operation(f"phase.{phase_name}", **phase_attributes) as span:
            try:
                if self.phase_counter:
                    self.phase_counter.add(1, phase_attributes)

                yield span

                if self.pipeline_success:
                    self.pipeline_success.add(1, phase_attributes)

            finally:
                if self.phase_duration:
                    duration = time.time() - start_time
                    self.phase_duration.record(duration, phase_attributes)

    def trace_model_operation(self, operation: str, model: str, **kwargs):
        """Decorator for model operations"""

        def decorator(func):
            def wrapper(*args, **kwargs):
                with self.trace_pipeline_phase(operation, model, **kwargs):
                    result = func(*args, **kwargs)
                    if self.model_operations:
                        self.model_operations.add(1, {"operation": operation, "model": model})
                    return result

            return wrapper

        return decorator


class HyperRAGInstrumentation:
    """Instrumentation for HyperRAG neural-biological memory system"""

    def __init__(self):
        self.query_duration = hyperrag_tracing.create_histogram(
            "query_duration_seconds", "HyperRAG query duration", "s"
        )
        self.query_counter = hyperrag_tracing.create_counter("queries_total", "Total queries processed")
        self.memory_usage = hyperrag_tracing.create_gauge("memory_usage_bytes", "Memory usage in bytes", "bytes")
        self.embedding_operations = hyperrag_tracing.create_counter(
            "embedding_operations_total", "Embedding operations performed"
        )
        self.retrieval_success = hyperrag_tracing.create_counter("retrieval_success_total", "Successful retrievals")

    @contextmanager
    def trace_memory_query(self, query_type: str, collection: str = "", **attributes):
        """Trace memory system queries"""
        start_time = time.time()

        query_attributes = {"query_type": query_type, "collection": collection, **attributes}

        with trace_hyperrag_query(query_type, **query_attributes) as span:
            try:
                if self.query_counter:
                    self.query_counter.add(1, query_attributes)

                yield span

                if self.retrieval_success:
                    self.retrieval_success.add(1, query_attributes)

            finally:
                if self.query_duration:
                    duration = time.time() - start_time
                    self.query_duration.record(duration, query_attributes)

                    # Track against 1.19ms baseline
                    if span:
                        span.set_attribute("baseline_ms", 1.19)
                        span.set_attribute("performance_ratio", duration / 0.00119)

    def track_memory_usage(self, usage_bytes: int, component: str = ""):
        """Track memory usage metrics"""
        if self.memory_usage:
            self.memory_usage.add(usage_bytes, {"component": component})

    def trace_embedding_operation(self, operation_type: str):
        """Trace embedding operations"""
        if self.embedding_operations:
            self.embedding_operations.add(1, {"operation": operation_type})


class P2PMeshInstrumentation:
    """Instrumentation for P2P mesh networking"""

    def __init__(self):
        self.message_duration = p2p_tracing.create_histogram(
            "message_duration_seconds", "P2P message delivery duration", "s"
        )
        self.messages_sent = p2p_tracing.create_counter("messages_sent_total", "Total messages sent")
        self.messages_received = p2p_tracing.create_counter("messages_received_total", "Total messages received")
        self.message_failures = p2p_tracing.create_counter(
            "message_delivery_failures_total", "Message delivery failures"
        )
        self.connected_peers = p2p_tracing.create_gauge("connected_peers", "Number of connected peers")
        self.bandwidth_usage = p2p_tracing.create_histogram(
            "bandwidth_usage_bytes", "Bandwidth usage per message", "bytes"
        )

    @contextmanager
    def trace_message_delivery(self, message_type: str, peer_id: str = "", **attributes):
        """Trace P2P message delivery"""
        start_time = time.time()

        message_attributes = {"message_type": message_type, "peer_id": peer_id, **attributes}

        with trace_p2p_message(message_type, **message_attributes) as span:
            try:
                if self.messages_sent:
                    self.messages_sent.add(1, message_attributes)

                yield span

                # Track success - aim for >95% delivery rate
                message_attributes["status"] = "success"

            except Exception as e:
                if self.message_failures:
                    self.message_failures.add(1, {**message_attributes, "error": str(e)})
                message_attributes["status"] = "failed"
                raise

            finally:
                if self.message_duration:
                    duration = time.time() - start_time
                    self.message_duration.record(duration, message_attributes)

    def track_peer_connection(self, peer_count: int, network_type: str = "mesh"):
        """Track peer connection metrics"""
        if self.connected_peers:
            self.connected_peers.add(peer_count, {"network_type": network_type})

    def track_bandwidth_usage(self, bytes_transferred: int, direction: str = "outbound"):
        """Track bandwidth usage"""
        if self.bandwidth_usage:
            self.bandwidth_usage.record(bytes_transferred, {"direction": direction})


class APIGatewayInstrumentation:
    """Instrumentation for Unified REST API Gateway"""

    def __init__(self):
        self.request_duration = api_gateway_tracing.create_histogram(
            "http_request_duration_seconds", "HTTP request duration", "s"
        )
        self.requests_total = api_gateway_tracing.create_counter("http_requests_total", "Total HTTP requests")
        self.response_size = api_gateway_tracing.create_histogram(
            "http_response_size_bytes", "HTTP response size", "bytes"
        )
        self.active_connections = api_gateway_tracing.create_gauge("active_connections", "Active HTTP connections")
        self.rate_limit_hits = api_gateway_tracing.create_counter("rate_limit_hits_total", "Rate limit violations")

    @contextmanager
    def trace_http_request(self, method: str, endpoint: str, **attributes):
        """Trace HTTP requests"""
        start_time = time.time()

        request_attributes = {"method": method, "endpoint": endpoint, **attributes}

        with api_gateway_tracing.trace_operation(f"http.{method.lower()}", **request_attributes) as span:
            try:
                if self.requests_total:
                    self.requests_total.add(1, request_attributes)

                yield span

                # Aim for <100ms response time
                duration = time.time() - start_time
                if span and duration > 0.1:
                    span.set_attribute("slow_request", True)

            finally:
                if self.request_duration:
                    duration = time.time() - start_time
                    self.request_duration.record(duration, request_attributes)

    def track_response_size(self, size_bytes: int, endpoint: str):
        """Track response sizes"""
        if self.response_size:
            self.response_size.record(size_bytes, {"endpoint": endpoint})

    def track_rate_limit_hit(self, client_id: str, endpoint: str):
        """Track rate limit violations"""
        if self.rate_limit_hits:
            self.rate_limit_hits.add(1, {"client_id": client_id, "endpoint": endpoint})


class EdgeComputingInstrumentation:
    """Instrumentation for Edge Computing Infrastructure"""

    def __init__(self):
        self.battery_level = edge_tracing.create_gauge("device_battery_percentage", "Device battery level", "%")
        self.resource_usage = edge_tracing.create_histogram(
            "resource_usage_percentage", "Resource usage percentage", "%"
        )
        self.task_execution = edge_tracing.create_histogram(
            "task_execution_duration_seconds", "Task execution time", "s"
        )
        self.energy_consumption = edge_tracing.create_histogram(
            "energy_consumption_joules", "Energy consumption per task", "J"
        )
        self.device_temperature = edge_tracing.create_gauge("device_temperature_celsius", "Device temperature", "°C")

    @contextmanager
    def trace_edge_task(self, task_type: str, device_id: str, **attributes):
        """Trace edge computing tasks"""
        start_time = time.time()
        start_energy = self._get_energy_reading()

        task_attributes = {"task_type": task_type, "device_id": device_id, **attributes}

        with edge_tracing.trace_operation(f"edge.task.{task_type}", **task_attributes) as span:
            try:
                yield span

            finally:
                # Track execution time
                duration = time.time() - start_time
                if self.task_execution:
                    self.task_execution.record(duration, task_attributes)

                # Track energy consumption for battery efficiency
                energy_used = self._get_energy_reading() - start_energy
                if self.energy_consumption and energy_used > 0:
                    self.energy_consumption.record(energy_used, task_attributes)

    def track_battery_level(self, battery_percentage: float, device_id: str):
        """Track device battery levels"""
        if self.battery_level:
            self.battery_level.add(battery_percentage, {"device_id": device_id})

    def track_device_temperature(self, temperature: float, device_id: str):
        """Track device temperature for thermal management"""
        if self.device_temperature:
            self.device_temperature.add(temperature, {"device_id": device_id})

    def _get_energy_reading(self) -> float:
        """Get current energy reading (reference implementation)"""
        # In real implementation, this would read from device sensors
        return 0.0


# Create instrumentation instances for all services
agent_forge_instrumentation = AgentForgeInstrumentation()
hyperrag_instrumentation = HyperRAGInstrumentation()
p2p_instrumentation = P2PMeshInstrumentation()
api_gateway_instrumentation = APIGatewayInstrumentation()
edge_instrumentation = EdgeComputingInstrumentation()


# Export convenience functions
def instrument_agent_forge_pipeline():
    """Initialize Agent Forge pipeline instrumentation"""
    agent_forge_tracing.initialize()
    return agent_forge_instrumentation


def instrument_hyperrag_memory():
    """Initialize HyperRAG memory system instrumentation"""
    hyperrag_tracing.initialize()
    return hyperrag_instrumentation


def instrument_p2p_mesh():
    """Initialize P2P mesh networking instrumentation"""
    p2p_tracing.initialize()
    return p2p_instrumentation


def instrument_api_gateway():
    """Initialize API gateway instrumentation"""
    api_gateway_tracing.initialize()
    return api_gateway_instrumentation


def instrument_edge_computing():
    """Initialize edge computing instrumentation"""
    edge_tracing.initialize()
    return edge_instrumentation


def initialize_all_instrumentation():
    """Initialize instrumentation for all services"""
    instruments = [
        instrument_agent_forge_pipeline(),
        instrument_hyperrag_memory(),
        instrument_p2p_mesh(),
        instrument_api_gateway(),
        instrument_edge_computing(),
    ]

    logger.info("All service instrumentation initialized")
    return instruments
