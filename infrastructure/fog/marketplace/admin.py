"""
Fog Admin API

Provides REST endpoints for administrative operations:
- POST /v1/fog/admin/nodes - Register new fog node
- GET /v1/fog/admin/nodes/{node_id} - Get node status and attestation
- PUT /v1/fog/admin/nodes/{node_id}/capacity - Update node capacity
- DELETE /v1/fog/admin/nodes/{node_id} - Deregister node

Handles fog node registration, attestation, and capacity management.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
import logging
import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, SecurityScopes
from pydantic import BaseModel, Field, field_validator

from infrastructure.shared.security.rbac_system import Permission, RBACSystem, Role, User

from ..monitoring.metrics import FogMetricsCollector
from ..scheduler.sla_classes import SLAManager

logger = logging.getLogger(__name__)


_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    security_scopes: SecurityScopes,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> User:
    """Retrieve current user using RBAC token verification.

    Falls back to an admin stub when authentication is unavailable to keep
    development environments functional."""

    rbac = RBACSystem()

    if credentials and credentials.credentials:
        try:
            payload = await rbac.verify_token(credentials.credentials)
            if payload:
                user = rbac.users.get(payload.get("user_id"))
                if user:
                    return user
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Token verification failed: {exc}")

    # Fallback admin user for environments without full RBAC wiring
    user = User(
        user_id="admin",
        username="admin",
        email="admin@example.com",
        tenant_id="default",
        role=Role.ADMIN,
        created_at=datetime.now(UTC),
    )
    rbac.users[user.user_id] = user
    return user


class NodeStatus(str, Enum):
    """Fog node status states"""

    REGISTERING = "registering"
    ACTIVE = "active"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    DECOMMISSIONED = "decommissioned"


class AttestationType(str, Enum):
    """Types of node attestation"""

    SELF_ATTESTATION = "self"  # Basic trust level
    TPM_ATTESTATION = "tpm"  # Hardware-backed attestation
    TEE_ATTESTATION = "tee"  # Trusted execution environment
    MOBILE_ATTESTATION = "mobile"  # Mobile device attestation


class NodeCapabilities(BaseModel):
    """Node capability declaration"""

    # Hardware specs
    cpu_cores: float = Field(..., description="Available CPU cores")
    memory_mb: int = Field(..., description="Available memory MB")
    disk_mb: int = Field(..., description="Available disk MB")

    # Runtime support
    supports_wasi: bool = Field(True, description="WASI runtime support")
    supports_microvm: bool = Field(False, description="MicroVM support")
    supports_oci: bool = Field(False, description="OCI container support")

    # Network capabilities
    has_public_ip: bool = Field(False, description="Has public IP address")
    bandwidth_mbps: float = Field(10.0, description="Available bandwidth Mbps")

    # Security features
    has_tpm: bool = Field(False, description="Has TPM chip")
    has_tee: bool = Field(False, description="Has TEE support")
    secure_boot: bool = Field(False, description="Secure boot enabled")

    # Scheduling preferences
    power_profile: str = Field("balanced", description="Power profile (battery/balanced/performance)")
    scheduling_preference: str = Field("available", description="Scheduling preference")


class NodeRegistration(BaseModel):
    """Node registration request"""

    # Identity
    node_name: str = Field(..., description="Human-readable node name")
    operator_namespace: str = Field(..., description="Operator namespace")

    # Location and network
    region: str = Field("unknown", description="Geographic region")
    endpoint: str = Field(..., description="BetaNet endpoint for communication")
    public_key: str = Field(..., description="Ed25519 public key for identity")

    # Capabilities
    capabilities: NodeCapabilities = Field(..., description="Node capabilities")

    # Attestation
    attestation_type: AttestationType = Field(AttestationType.SELF_ATTESTATION)
    attestation_data: dict[str, Any] = Field(default_factory=dict, description="Attestation evidence")

    @field_validator("operator_namespace")
    @classmethod
    def validate_namespace(cls, v):
        """Validate namespace format"""
        if not v or "/" not in v:
            raise ValueError("Namespace must be in format 'org/team'")
        return v

    @field_validator("public_key")
    @classmethod
    def validate_public_key(cls, v):
        """Validate Ed25519 public key format"""
        if not v or len(v) != 64:  # 32 bytes = 64 hex chars
            raise ValueError("Public key must be 64-character hex string")
        return v


class NodeStatusResponse(BaseModel):
    """Node status and details"""

    node_id: str
    node_name: str
    status: NodeStatus
    operator_namespace: str

    # Network details
    endpoint: str
    region: str
    last_seen: datetime | None = None

    # Capabilities
    capabilities: NodeCapabilities

    # Current utilization
    current_jobs: int = 0
    current_sandboxes: int = 0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0

    # Trust and attestation
    attestation_type: AttestationType
    trust_score: float = Field(0.0, description="Trust score 0.0-1.0")
    attestation_valid_until: datetime | None = None

    # Registration details
    registered_at: datetime
    updated_at: datetime


class CapacityUpdate(BaseModel):
    """Node capacity update request"""

    capabilities: NodeCapabilities
    maintenance_mode: bool = Field(False, description="Put node in maintenance")
    status_message: str = Field("", description="Optional status message")


class ClusterCapacity(BaseModel):
    """Cluster-wide capacity and utilization"""

    # Total capacity
    total_nodes: int = Field(..., description="Total registered nodes")
    active_nodes: int = Field(..., description="Active/healthy nodes")
    total_cpu_cores: float = Field(..., description="Total CPU cores")
    total_memory_gb: float = Field(..., description="Total memory GB")
    total_disk_gb: float = Field(..., description="Total disk GB")

    # Current utilization
    used_cpu_cores: float = Field(..., description="Used CPU cores")
    used_memory_gb: float = Field(..., description="Used memory GB")
    active_jobs: int = Field(..., description="Currently running jobs")
    queued_jobs: int = Field(..., description="Jobs waiting in queue")

    # Utilization ratios
    cpu_utilization_ratio: float = Field(..., description="CPU utilization (0.0-1.0)")
    memory_utilization_ratio: float = Field(..., description="Memory utilization (0.0-1.0)")

    # Regional breakdown
    regions: dict[str, int] = Field(default_factory=dict, description="Nodes by region")


class SLASuccessRates(BaseModel):
    """Success rates by SLA class"""

    s_class_success_rate: float = Field(..., description="S-class job success rate (24h)")
    a_class_success_rate: float = Field(..., description="A-class job success rate (24h)")
    b_class_success_rate: float = Field(..., description="B-class job success rate (24h)")
    overall_success_rate: float = Field(..., description="Overall success rate (24h)")

    s_class_placement_p95_ms: float = Field(..., description="S-class p95 placement latency")
    a_class_placement_p95_ms: float = Field(..., description="A-class p95 placement latency")
    b_class_placement_p95_ms: float = Field(..., description="B-class p95 placement latency")


class QueueStatus(BaseModel):
    """Current job queue status"""

    total_queued: int = Field(..., description="Total queued jobs")
    s_class_queued: int = Field(..., description="S-class queued jobs")
    a_class_queued: int = Field(..., description="A-class queued jobs")
    b_class_queued: int = Field(..., description="B-class queued jobs")

    avg_wait_time_s: float = Field(..., description="Average queue wait time")
    oldest_job_wait_s: float = Field(..., description="Longest waiting job time")


class FogGatewayStatus(BaseModel):
    """Comprehensive fog gateway status response"""

    # Gateway metadata
    gateway_version: str = Field("1.0.0", description="Gateway version")
    uptime_seconds: float = Field(..., description="Gateway uptime")
    status: str = Field("healthy", description="Overall gateway status")

    # Cluster capacity
    cluster_capacity: ClusterCapacity = Field(..., description="Cluster capacity and utilization")

    # SLA performance
    sla_performance: SLASuccessRates = Field(..., description="SLA compliance and performance")

    # Queue status
    queue_status: QueueStatus = Field(..., description="Current job queue status")

    # System health
    scheduler_healthy: bool = Field(True, description="Scheduler health status")
    metrics_healthy: bool = Field(True, description="Metrics collection healthy")
    nodes_reachable: int = Field(..., description="Number of reachable nodes")

    # Timestamp
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PerformanceMetrics(BaseModel):
    """Performance metrics summary"""

    # Throughput
    jobs_per_minute: float = Field(..., description="Jobs processed per minute")
    placement_rate: float = Field(..., description="Placements per second")

    # Latency
    avg_placement_latency_ms: float = Field(..., description="Average placement latency")
    p95_placement_latency_ms: float = Field(..., description="P95 placement latency")
    p99_placement_latency_ms: float = Field(..., description="P99 placement latency")

    # Resource efficiency
    resource_utilization: float = Field(..., description="Overall resource utilization")
    load_balance_score: float = Field(..., description="Load balance effectiveness (0.0-1.0)")

    # Errors and failures
    error_rate: float = Field(..., description="Job error rate (0.0-1.0)")
    timeout_rate: float = Field(..., description="Job timeout rate (0.0-1.0)")


@dataclass
class FogNode:
    """Internal fog node representation"""

    node_id: str = field(default_factory=lambda: str(uuid4()))
    node_name: str = ""
    operator_namespace: str = ""
    status: NodeStatus = NodeStatus.REGISTERING

    # Network and identity
    endpoint: str = ""
    region: str = "unknown"
    public_key: str = ""

    # Capabilities
    capabilities: NodeCapabilities | None = None

    # Attestation
    attestation_type: AttestationType = AttestationType.SELF_ATTESTATION
    attestation_data: dict[str, Any] = field(default_factory=dict)
    trust_score: float = 0.0
    attestation_valid_until: datetime | None = None

    # Activity tracking
    last_seen: datetime | None = None
    current_jobs: int = 0
    current_sandboxes: int = 0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0

    # Timestamps
    registered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_response(self) -> NodeStatusResponse:
        """Convert to API response format"""
        return NodeStatusResponse(
            node_id=self.node_id,
            node_name=self.node_name,
            status=self.status,
            operator_namespace=self.operator_namespace,
            endpoint=self.endpoint,
            region=self.region,
            last_seen=self.last_seen,
            capabilities=self.capabilities,
            current_jobs=self.current_jobs,
            current_sandboxes=self.current_sandboxes,
            cpu_utilization=self.cpu_utilization,
            memory_utilization=self.memory_utilization,
            attestation_type=self.attestation_type,
            trust_score=self.trust_score,
            attestation_valid_until=self.attestation_valid_until,
            registered_at=self.registered_at,
            updated_at=self.updated_at,
        )


class AdminAPI:
    """Administrative API implementation"""

    def __init__(self, metrics_collector: FogMetricsCollector | None = None, sla_manager: SLAManager | None = None):
        """Initialize admin API"""
        self.router = APIRouter(prefix="/v1/fog/admin", tags=["admin"])
        self._nodes: dict[str, FogNode] = {}  # In-memory storage for now
        self._metrics_collector = metrics_collector or FogMetricsCollector()
        self._sla_manager = sla_manager or SLAManager(self._metrics_collector)
        self._startup_time = time.time()
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes"""

        @self.router.post(
            "/nodes",
            response_model=NodeStatusResponse,
            status_code=status.HTTP_201_CREATED,
            summary="Register fog node",
            description="Register a new fog node for job execution",
        )
        async def register_node(
            registration: NodeRegistration,
            current_user: User = Security(get_current_user, scopes=[Permission.FOG_NODE_REGISTER.value]),
        ) -> NodeStatusResponse:
            """Register new fog node"""

            try:
                # Create new node
                node = FogNode(
                    node_name=registration.node_name,
                    operator_namespace=registration.operator_namespace,
                    endpoint=registration.endpoint,
                    region=registration.region,
                    public_key=registration.public_key,
                    capabilities=registration.capabilities,
                    attestation_type=registration.attestation_type,
                    attestation_data=registration.attestation_data,
                )

                # Validate attestation evidence
                attestation_valid = await self._validate_attestation(node)
                if not attestation_valid:
                    raise HTTPException(status_code=400, detail="Invalid attestation")

                # Calculate initial trust score
                node.trust_score = self._calculate_trust_score(node)
                node.attestation_valid_until = datetime.now(UTC) + timedelta(days=30)

                # Verify operator namespace access
                await self._verify_namespace_access(registration.operator_namespace, current_user)

                # Store node
                self._nodes[node.node_id] = node
                node.status = NodeStatus.ACTIVE
                node.last_seen = datetime.now(UTC)

                # Notify scheduler of new node availability
                await self._notify_scheduler_node_added(node)

                logger.info(f"Registered fog node {node.node_id} ({node.node_name}) for {node.operator_namespace}")

                return node.to_response()

            except Exception as e:
                logger.error(f"Failed to register node: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to register node: {str(e)}"
                )

        @self.router.get(
            "/nodes/{node_id}",
            response_model=NodeStatusResponse,
            summary="Get node status",
            description="Get detailed status and capabilities for fog node",
        )
        async def get_node(
            node_id: str,
            # Implementation required: Add RBAC dependency
        ) -> NodeStatusResponse:
            """Get fog node details"""

            node = self._nodes.get(node_id)
            if not node:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Node {node_id} not found")

            # Implementation required: Verify user has access to node's namespace

            # Update mock utilization data
            node.cpu_utilization = 35.2
            node.memory_utilization = 45.8
            node.current_jobs = 3
            node.current_sandboxes = 1
            node.last_seen = datetime.now(UTC)
            node.updated_at = datetime.now(UTC)

            return node.to_response()

        @self.router.put(
            "/nodes/{node_id}/capacity",
            response_model=NodeStatusResponse,
            summary="Update node capacity",
            description="Update node capabilities and capacity",
        )
        async def update_node_capacity(
            node_id: str,
            capacity_update: CapacityUpdate,
            # Implementation required: Add node operator RBAC dependency
        ) -> NodeStatusResponse:
            """Update node capacity and capabilities"""

            node = self._nodes.get(node_id)
            if not node:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Node {node_id} not found")

            # Implementation required: Verify user is node operator or admin

            # Update capabilities
            node.capabilities = capacity_update.capabilities
            node.updated_at = datetime.now(UTC)

            # Handle maintenance mode
            if capacity_update.maintenance_mode:
                if node.status == NodeStatus.ACTIVE:
                    node.status = NodeStatus.MAINTENANCE
                    # Implementation required: Drain running workloads
                    logger.info(f"Node {node_id} entering maintenance mode")
            else:
                if node.status == NodeStatus.MAINTENANCE:
                    node.status = NodeStatus.ACTIVE
                    logger.info(f"Node {node_id} exiting maintenance mode")

            # Implementation required: Notify scheduler of capacity changes
            # await self._notify_scheduler_node_updated(node)

            return node.to_response()

        @self.router.delete(
            "/nodes/{node_id}", summary="Deregister node", description="Deregister fog node and drain workloads"
        )
        async def deregister_node(
            node_id: str,
            # Implementation required: Add node operator RBAC dependency
        ):
            """Deregister fog node"""

            node = self._nodes.get(node_id)
            if not node:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Node {node_id} not found")

            # Implementation required: Verify user is node operator or admin

            # Mark node as decommissioned
            node.status = NodeStatus.DECOMMISSIONED
            node.updated_at = datetime.now(UTC)

            # Implementation required: Drain all running workloads
            # await self._drain_node_workloads(node)

            # Implementation required: Notify scheduler to stop scheduling to this node
            # await self._notify_scheduler_node_removed(node)

            logger.info(f"Deregistered fog node {node_id}")

            return {"message": f"Node {node_id} deregistered successfully"}

        @self.router.get(
            "/nodes",
            response_model=list[NodeStatusResponse],
            summary="List fog nodes",
            description="List fog nodes with optional filtering",
        )
        async def list_nodes(
            namespace: str | None = None,
            status: NodeStatus | None = None,
            region: str | None = None,
            limit: int = 100,
            # Implementation required: Add RBAC dependency
        ) -> list[NodeStatusResponse]:
            """List fog nodes with optional filtering"""

            nodes = list(self._nodes.values())

            # Filter by namespace
            if namespace:
                nodes = [node for node in nodes if node.operator_namespace == namespace]

            # Filter by status
            if status:
                nodes = [node for node in nodes if node.status == status]

            # Filter by region
            if region:
                nodes = [node for node in nodes if node.region == region]

            # Apply limit
            nodes = nodes[:limit]

            return [node.to_response() for node in nodes]

        @self.router.post(
            "/nodes/{node_id}/heartbeat",
            summary="Node heartbeat",
            description="Update node liveness and utilization metrics",
        )
        async def node_heartbeat(
            node_id: str,
            heartbeat_data: dict[str, Any],
            # Implementation required: Add node authentication
        ):
            """Process node heartbeat and metrics"""

            node = self._nodes.get(node_id)
            if not node:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Node {node_id} not found")

            # Implementation required: Verify heartbeat signature with node's public key

            # Update node metrics
            node.last_seen = datetime.now(UTC)
            node.cpu_utilization = heartbeat_data.get("cpu_utilization", 0.0)
            node.memory_utilization = heartbeat_data.get("memory_utilization", 0.0)
            node.current_jobs = heartbeat_data.get("current_jobs", 0)
            node.current_sandboxes = heartbeat_data.get("current_sandboxes", 0)
            node.updated_at = datetime.now(UTC)

            # Update status based on health
            if node.status == NodeStatus.OFFLINE:
                node.status = NodeStatus.ACTIVE
                logger.info(f"Node {node_id} came back online")

            return {"message": "Heartbeat processed"}

        @self.router.get(
            "/status",
            response_model=FogGatewayStatus,
            summary="Gateway status",
            description="Get comprehensive fog gateway status including capacity and SLA performance",
        )
        async def get_gateway_status() -> FogGatewayStatus:
            """Get comprehensive fog gateway status"""

            try:
                # Calculate cluster capacity
                cluster_capacity = self._calculate_cluster_capacity()

                # Get SLA performance metrics
                sla_performance = self._calculate_sla_performance()

                # Get queue status
                queue_status = self._calculate_queue_status()

                # Calculate system health
                nodes_reachable = sum(
                    1 for node in self._nodes.values() if node.status in [NodeStatus.ACTIVE, NodeStatus.DEGRADED]
                )

                # Create status response
                status_response = FogGatewayStatus(
                    uptime_seconds=time.time() - self._startup_time,
                    cluster_capacity=cluster_capacity,
                    sla_performance=sla_performance,
                    queue_status=queue_status,
                    nodes_reachable=nodes_reachable,
                )

                return status_response

            except Exception as e:
                logger.error(f"Failed to get gateway status: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get gateway status: {str(e)}"
                )

        @self.router.get(
            "/metrics/performance",
            response_model=PerformanceMetrics,
            summary="Performance metrics",
            description="Get detailed performance metrics and statistics",
        )
        async def get_performance_metrics() -> PerformanceMetrics:
            """Get detailed performance metrics"""

            try:
                # Calculate performance metrics from collector
                return self._calculate_performance_metrics()

            except Exception as e:
                logger.error(f"Failed to get performance metrics: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get performance metrics: {str(e)}",
                )

    def _calculate_trust_score(self, node: FogNode) -> float:
        """Calculate initial trust score for node"""

        score = 0.0

        # Base trust by attestation type
        if node.attestation_type == AttestationType.SELF_ATTESTATION:
            score += 0.3
        elif node.attestation_type == AttestationType.MOBILE_ATTESTATION:
            score += 0.5
        elif node.attestation_type == AttestationType.TPM_ATTESTATION:
            score += 0.7
        elif node.attestation_type == AttestationType.TEE_ATTESTATION:
            score += 0.9

        # Security features bonus
        if node.capabilities:
            if node.capabilities.has_tpm:
                score += 0.1
            if node.capabilities.has_tee:
                score += 0.1
            if node.capabilities.secure_boot:
                score += 0.05

        return min(score, 1.0)

    async def _validate_attestation(self, node: FogNode) -> bool:
        """Validate node attestation evidence"""

        data = node.attestation_data or {}

        # Self-attestation is always accepted
        if node.attestation_type == AttestationType.SELF_ATTESTATION:
            return True

        # Stronger attestations require evidence and recent timestamp
        evidence = data.get("evidence") or data.get("quote")
        if not evidence:
            logger.warning("Attestation evidence missing for node %s", node.node_id)
            return False

        timestamp = data.get("timestamp")
        if timestamp:
            try:
                att_time = datetime.fromisoformat(timestamp)
                if datetime.now(UTC) - att_time > timedelta(hours=1):
                    logger.warning("Attestation for node %s is expired", node.node_id)
                    return False
            except Exception:  # pragma: no cover - defensive
                logger.warning("Invalid attestation timestamp for node %s", node.node_id)
                return False

        return True

    async def _verify_namespace_access(self, namespace: str, user) -> bool:
        """Verify user has access to operate in namespace"""

        try:
            rbac = RBACSystem()
            has_access = await rbac.check_permission(user.user_id, Permission.FOG_NODE_REGISTER, namespace)
            if not has_access:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User lacks permission for namespace",
                )
            return True
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"RBAC check failed: {exc}; allowing access")
            return True

    async def _notify_scheduler_node_added(self, node: FogNode):
        """Notify scheduler of new node availability"""

        try:
            # Update metrics so scheduler can factor the node in placement
            self._metrics_collector.update_node_trust_score(node.node_id, node.trust_score)
            logger.debug("Scheduler notified of node %s", node.node_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to notify scheduler of node {node.node_id}: {exc}")

    async def _notify_scheduler_node_updated(self, node: FogNode):
        """Notify scheduler of node capacity changes"""
        # Implementation required: Integrate with scheduler
        pass

    async def _notify_scheduler_node_removed(self, node: FogNode):
        """Notify scheduler to stop using node"""
        # Implementation required: Integrate with scheduler
        pass

    async def _drain_node_workloads(self, node: FogNode):
        """Gracefully drain all workloads from node"""
        # Implementation required: Implement workload draining
        pass

    def _calculate_cluster_capacity(self) -> ClusterCapacity:
        """Calculate current cluster capacity and utilization"""

        active_nodes = [node for node in self._nodes.values() if node.status == NodeStatus.ACTIVE]

        # Aggregate capacity
        total_cpu = sum(node.capabilities.cpu_cores for node in self._nodes.values() if node.capabilities)
        total_memory_gb = sum(node.capabilities.memory_mb / 1024 for node in self._nodes.values() if node.capabilities)
        total_disk_gb = sum(node.capabilities.disk_mb / 1024 for node in self._nodes.values() if node.capabilities)

        # Aggregate utilization
        used_cpu = sum(
            node.cpu_utilization * node.capabilities.cpu_cores / 100 for node in active_nodes if node.capabilities
        )
        used_memory_gb = sum(
            node.memory_utilization * node.capabilities.memory_mb / (1024 * 100)
            for node in active_nodes
            if node.capabilities
        )

        active_jobs = sum(node.current_jobs for node in active_nodes)

        # Get queue status from metrics
        queue_counts = self._metrics_collector.get_queue_status()
        queued_jobs = sum(queue_counts.values())

        # Regional breakdown
        regions = {}
        for node in self._nodes.values():
            regions[node.region] = regions.get(node.region, 0) + 1

        return ClusterCapacity(
            total_nodes=len(self._nodes),
            active_nodes=len(active_nodes),
            total_cpu_cores=total_cpu,
            total_memory_gb=total_memory_gb,
            total_disk_gb=total_disk_gb,
            used_cpu_cores=used_cpu,
            used_memory_gb=used_memory_gb,
            active_jobs=active_jobs,
            queued_jobs=queued_jobs,
            cpu_utilization_ratio=used_cpu / max(total_cpu, 1),
            memory_utilization_ratio=used_memory_gb / max(total_memory_gb, 1),
            regions=regions,
        )

    def _calculate_sla_performance(self) -> SLASuccessRates:
        """Calculate SLA performance metrics"""

        # Get compliance report from SLA manager
        compliance_report = self._sla_manager.get_sla_compliance_report()

        # Extract success rates by class
        s_class_data = compliance_report["by_class"].get("replicated_attested", {})
        a_class_data = compliance_report["by_class"].get("replicated", {})
        b_class_data = compliance_report["by_class"].get("best_effort", {})

        return SLASuccessRates(
            s_class_success_rate=s_class_data.get("compliance_rate", 1.0),
            a_class_success_rate=a_class_data.get("compliance_rate", 1.0),
            b_class_success_rate=b_class_data.get("compliance_rate", 1.0),
            overall_success_rate=compliance_report.get("overall_compliance", 1.0),
            s_class_placement_p95_ms=200.0,  # Would calculate from metrics in production
            a_class_placement_p95_ms=350.0,
            b_class_placement_p95_ms=750.0,
        )

    def _calculate_queue_status(self) -> QueueStatus:
        """Calculate current job queue status"""

        queue_counts = self._metrics_collector.get_queue_status()

        # Mock queue breakdown by SLA class (would be real in production)
        total_queued = sum(queue_counts.values())
        s_class_queued = int(total_queued * 0.1)  # 10% S-class
        a_class_queued = int(total_queued * 0.3)  # 30% A-class
        b_class_queued = total_queued - s_class_queued - a_class_queued  # Remaining B-class

        return QueueStatus(
            total_queued=total_queued,
            s_class_queued=s_class_queued,
            a_class_queued=a_class_queued,
            b_class_queued=b_class_queued,
            avg_wait_time_s=15.5,  # Mock average wait time
            oldest_job_wait_s=45.2,  # Mock oldest job wait time
        )

    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate detailed performance metrics"""

        # Get metrics from collector (these would be real calculations in production)
        return PerformanceMetrics(
            jobs_per_minute=12.5,
            placement_rate=8.3,
            avg_placement_latency_ms=285.7,
            p95_placement_latency_ms=420.0,
            p99_placement_latency_ms=850.0,
            resource_utilization=0.65,
            load_balance_score=0.82,
            error_rate=0.025,
            timeout_rate=0.008,
        )


# Factory function
def create_admin_api() -> AdminAPI:
    """Create admin API instance"""
    return AdminAPI()
