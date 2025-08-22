"""Fog computing constants for AIVillage.

This module centralizes all fog computing and distributed processing
magic literals to eliminate connascence of meaning.
"""

from enum import Enum
from typing import Final

# Fog node configuration
DEFAULT_FOG_PORT: Final[int] = 8080
FOG_DISCOVERY_PORT: Final[int] = 8081
FOG_METRICS_PORT: Final[int] = 8082
MAX_FOG_NODES: Final[int] = 1000
NODE_HEARTBEAT_INTERVAL_SECONDS: Final[int] = 30

# Job management
MAX_CONCURRENT_JOBS: Final[int] = 10
JOB_TIMEOUT_SECONDS: Final[int] = 3600  # 1 hour
JOB_QUEUE_SIZE: Final[int] = 100
JOB_RETRY_ATTEMPTS: Final[int] = 3
JOB_RETRY_DELAY_SECONDS: Final[int] = 30

# Resource limits
DEFAULT_CPU_LIMIT: Final[float] = 2.0  # CPU cores
DEFAULT_MEMORY_LIMIT_MB: Final[int] = 4096  # 4GB
DEFAULT_STORAGE_LIMIT_GB: Final[int] = 100  # 100GB
DEFAULT_NETWORK_BANDWIDTH_MBPS: Final[int] = 100
GPU_MEMORY_LIMIT_GB: Final[int] = 8

# Scheduling and load balancing
LOAD_BALANCING_ALGORITHM: Final[str] = "least_loaded"
SCHEDULING_INTERVAL_SECONDS: Final[int] = 10
REBALANCING_THRESHOLD_PERCENT: Final[int] = 20
NODE_SELECTION_TIMEOUT_SECONDS: Final[int] = 5

# Network configuration
CONNECTION_TIMEOUT_SECONDS: Final[int] = 30
MESSAGE_TIMEOUT_SECONDS: Final[int] = 60
MAX_MESSAGE_SIZE_MB: Final[int] = 100
NETWORK_RETRY_ATTEMPTS: Final[int] = 3
KEEP_ALIVE_INTERVAL_SECONDS: Final[int] = 60

# Edge optimization
EDGE_CACHE_SIZE_MB: Final[int] = 1024
EDGE_CACHE_TTL_SECONDS: Final[int] = 3600
EDGE_PREFETCH_ENABLED: Final[bool] = True
EDGE_COMPRESSION_ENABLED: Final[bool] = True
EDGE_ENCRYPTION_ENABLED: Final[bool] = True

# Data synchronization
SYNC_INTERVAL_SECONDS: Final[int] = 300  # 5 minutes
CONFLICT_RESOLUTION_STRATEGY: Final[str] = "last_write_wins"
REPLICATION_FACTOR: Final[int] = 3
CONSISTENCY_LEVEL: Final[str] = "eventual"

# Mobile integration
MOBILE_BATTERY_THRESHOLD_PERCENT: Final[int] = 20
MOBILE_NETWORK_QUALITY_THRESHOLD: Final[float] = 0.7
MOBILE_OFFLOAD_DELAY_MS: Final[int] = 100
MOBILE_TASK_SIZE_LIMIT_MB: Final[int] = 10

# P2P mesh networking
P2P_DISCOVERY_INTERVAL_SECONDS: Final[int] = 60
P2P_CONNECTION_POOL_SIZE: Final[int] = 20
P2P_ROUTING_TABLE_SIZE: Final[int] = 1000
P2P_MESSAGE_PROPAGATION_TTL: Final[int] = 10
P2P_BANDWIDTH_LIMIT_MBPS: Final[int] = 50


class FogNodeType(Enum):
    """Types of fog nodes in the network."""

    EDGE_GATEWAY = "edge_gateway"
    COMPUTE_NODE = "compute_node"
    STORAGE_NODE = "storage_node"
    BROKER_NODE = "broker_node"
    MOBILE_DEVICE = "mobile_device"
    IOT_DEVICE = "iot_device"


class JobStatus(Enum):
    """Status of fog computing jobs."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ResourceType(Enum):
    """Types of computational resources."""

    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    SENSOR = "sensor"


class NetworkCondition(Enum):
    """Network condition classifications."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    OFFLINE = "offline"


class DeploymentStrategy(Enum):
    """Fog deployment strategies."""

    REPLICATED = "replicated"
    SHARDED = "sharded"
    FEDERATED = "federated"
    HYBRID = "hybrid"


# Monitoring and observability
METRICS_COLLECTION_INTERVAL_SECONDS: Final[int] = 30
LOG_RETENTION_HOURS: Final[int] = 168  # 1 week
ALERT_THRESHOLD_CPU_PERCENT: Final[int] = 80
ALERT_THRESHOLD_MEMORY_PERCENT: Final[int] = 85
ALERT_THRESHOLD_STORAGE_PERCENT: Final[int] = 90

# Security configuration
ENCRYPTION_ALGORITHM: Final[str] = "AES-256-GCM"
KEY_ROTATION_INTERVAL_HOURS: Final[int] = 24
CERTIFICATE_VALIDITY_DAYS: Final[int] = 90
AUTHENTICATION_TIMEOUT_SECONDS: Final[int] = 300

# Quality of Service (QoS)
QOS_HIGH_PRIORITY_WEIGHT: Final[int] = 10
QOS_NORMAL_PRIORITY_WEIGHT: Final[int] = 5
QOS_LOW_PRIORITY_WEIGHT: Final[int] = 1
QOS_LATENCY_SLA_MS: Final[int] = 100
QOS_THROUGHPUT_SLA_MBPS: Final[int] = 10

# Namespace and isolation
DEFAULT_NAMESPACE: Final[str] = "default"
MAX_NAMESPACES: Final[int] = 100
NAMESPACE_QUOTA_JOBS: Final[int] = 50
NAMESPACE_QUOTA_CPU_CORES: Final[int] = 10
NAMESPACE_QUOTA_MEMORY_GB: Final[int] = 20


class FogMessages:
    """Standardized fog computing messages."""

    NODE_REGISTERED: Final[str] = "Node {node_id} registered as {node_type}"
    JOB_SUBMITTED: Final[str] = "Job {job_id} submitted to namespace {namespace}"
    JOB_SCHEDULED: Final[str] = "Job {job_id} scheduled on node {node_id}"
    JOB_COMPLETED: Final[str] = "Job {job_id} completed successfully in {duration}s"
    JOB_FAILED: Final[str] = "Job {job_id} failed: {error}"
    NODE_OFFLINE: Final[str] = "Node {node_id} went offline"
    RESOURCE_EXHAUSTED: Final[str] = "Resource {resource_type} exhausted on node {node_id}"
    NETWORK_PARTITION: Final[str] = "Network partition detected in region {region}"
    LOAD_REBALANCED: Final[str] = "Load rebalanced: {jobs_moved} jobs moved"
    FAILOVER_INITIATED: Final[str] = "Failover initiated for node {node_id}"


# Performance benchmarks
BENCHMARK_CPU_OPERATIONS_PER_SECOND: Final[int] = 1_000_000
BENCHMARK_MEMORY_BANDWIDTH_GBPS: Final[float] = 10.0
BENCHMARK_STORAGE_IOPS: Final[int] = 10_000
BENCHMARK_NETWORK_LATENCY_MS: Final[float] = 1.0

# Data processing
BATCH_PROCESSING_SIZE: Final[int] = 1000
STREAM_PROCESSING_WINDOW_SECONDS: Final[int] = 60
DATA_COMPRESSION_LEVEL: Final[int] = 6
MAX_DATA_AGE_HOURS: Final[int] = 24

# Fault tolerance
FAILURE_DETECTION_TIMEOUT_SECONDS: Final[int] = 60
RECOVERY_ATTEMPT_LIMIT: Final[int] = 5
CIRCUIT_BREAKER_FAILURE_THRESHOLD: Final[int] = 10
CIRCUIT_BREAKER_RECOVERY_TIME_SECONDS: Final[int] = 300

# API and interfaces
FOG_API_VERSION: Final[str] = "v1"
GRPC_MAX_MESSAGE_SIZE_MB: Final[int] = 64
HTTP_REQUEST_TIMEOUT_SECONDS: Final[int] = 30
WEBSOCKET_PING_INTERVAL_SECONDS: Final[int] = 30

# Resource discovery
SERVICE_DISCOVERY_TTL_SECONDS: Final[int] = 300
CAPABILITY_ADVERTISEMENT_INTERVAL_SECONDS: Final[int] = 120
RESOURCE_PROBE_INTERVAL_SECONDS: Final[int] = 60

# Edge-cloud coordination
CLOUD_SYNC_INTERVAL_SECONDS: Final[int] = 3600  # 1 hour
EDGE_AUTONOMY_MODE_ENABLED: Final[bool] = True
CLOUD_FALLBACK_TIMEOUT_SECONDS: Final[int] = 10
HYBRID_EXECUTION_THRESHOLD: Final[float] = 0.5

# Cost optimization
COST_AWARENESS_ENABLED: Final[bool] = True
ENERGY_EFFICIENCY_WEIGHT: Final[float] = 0.3
LATENCY_WEIGHT: Final[float] = 0.4
THROUGHPUT_WEIGHT: Final[float] = 0.3

# Multi-tenancy in fog
TENANT_ISOLATION_ENABLED: Final[bool] = True
SHARED_RESOURCE_POOL_ENABLED: Final[bool] = True
TENANT_PRIORITY_LEVELS: Final[int] = 5
CROSS_TENANT_COMMUNICATION_ALLOWED: Final[bool] = False
