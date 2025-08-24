"""
Fog Jobs API

Provides REST endpoints for fog job management:
- POST /v1/fog/jobs - Submit new fog job
- GET /v1/fog/jobs/{job_id} - Get job status
- GET /v1/fog/jobs/{job_id}/logs - Stream job logs
- DELETE /v1/fog/jobs/{job_id} - Cancel job

Integrates with existing AIVillage RBAC and BetaNet routing.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class RuntimeType(str, Enum):
    """Supported execution runtimes"""

    WASI = "wasi"
    MICROVM = "microvm"
    OCI = "oci"


class JobStatus(str, Enum):
    """Job lifecycle states"""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    """Job priority classes with SLA guarantees"""

    BEST_EFFORT = "B"  # Best effort, no SLA
    STANDARD = "A"  # Replicated execution
    PREMIUM = "S"  # Replicated + attested execution


@dataclass
class JobResources:
    """Resource requirements for fog job"""

    cpu_cores: float = 1.0
    memory_mb: int = 512
    disk_mb: int = 1024
    max_duration_s: int = 300
    network_egress: bool = False  # Deny egress by default for security


@dataclass
class JobMount:
    """Data mount specification"""

    type: str  # "rag", "tmp", "ro", "rw"
    handle: str  # CAS hash or namespace handle
    path: str  # Mount point in container
    mode: str = "ro"  # "ro" or "rw"


class JobSpec(BaseModel):
    """Job specification for submission"""

    namespace: str = Field(..., description="Namespace for RBAC and quota enforcement")
    runtime: RuntimeType = Field(RuntimeType.WASI, description="Execution runtime")
    image: str = Field(..., description="CAS hash or registry reference")
    priority: JobPriority = Field(JobPriority.BEST_EFFORT, description="Priority class")

    # Execution environment
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    args: list[str] = Field(default_factory=list, description="Command arguments")

    # Resource requirements
    resources: dict[str, Any] = Field(
        default_factory=lambda: {
            "cpu_cores": 1.0,
            "memory_mb": 512,
            "disk_mb": 1024,
            "max_duration_s": 300,
            "network_egress": False,
        },
        description="Resource requirements",
    )

    # Data mounts
    mounts: list[dict[str, Any]] = Field(default_factory=list, description="Data mounts")

    # Metadata
    labels: dict[str, str] = Field(default_factory=dict, description="User labels")

    @validator("namespace")
    def validate_namespace(cls, v):
        """Validate namespace format"""
        if not v or "/" not in v:
            raise ValueError("Namespace must be in format 'org/team'")
        return v

    @validator("image")
    def validate_image(cls, v):
        """Validate image reference"""
        if not v:
            raise ValueError("Image reference required")
        return v


class JobResponse(BaseModel):
    """Job creation response"""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    estimated_start_time: datetime | None = Field(None, description="Estimated start time")

    # Placement information
    placement_score: float | None = Field(None, description="Placement optimization score")
    assigned_nodes: list[str] = Field(default_factory=list, description="Assigned fog nodes")


class JobStatusResponse(BaseModel):
    """Detailed job status response"""

    job_id: str
    status: JobStatus
    spec: JobSpec

    # Timestamps
    created_at: datetime
    queued_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Execution details
    assigned_nodes: list[str] = Field(default_factory=list)
    exit_code: int | None = None
    error_message: str | None = None

    # Resource usage
    cpu_seconds_used: float = 0.0
    memory_mb_peak: int = 0
    network_bytes_egress: int = 0

    # SLA metrics
    placement_latency_ms: float | None = None
    execution_latency_ms: float | None = None


@dataclass
class FogJob:
    """Internal fog job representation"""

    job_id: str = field(default_factory=lambda: str(uuid4()))
    spec: JobSpec | None = None
    status: JobStatus = JobStatus.PENDING

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    queued_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Placement results
    assigned_nodes: list[str] = field(default_factory=list)
    placement_score: float | None = None

    # Execution results
    exit_code: int | None = None
    error_message: str | None = None
    logs: str = ""

    # Resource usage tracking
    cpu_seconds_used: float = 0.0
    memory_mb_peak: int = 0
    network_bytes_egress: int = 0

    def to_response(self) -> JobStatusResponse:
        """Convert to API response format"""
        return JobStatusResponse(
            job_id=self.job_id,
            status=self.status,
            spec=self.spec,
            created_at=self.created_at,
            queued_at=self.queued_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            assigned_nodes=self.assigned_nodes,
            exit_code=self.exit_code,
            error_message=self.error_message,
            cpu_seconds_used=self.cpu_seconds_used,
            memory_mb_peak=self.memory_mb_peak,
            network_bytes_egress=self.network_bytes_egress,
            placement_latency_ms=self._calculate_placement_latency_ms(),
            execution_latency_ms=self._calculate_execution_latency_ms(),
        )

    def _calculate_placement_latency_ms(self) -> float | None:
        """Calculate placement latency in milliseconds"""
        if not self.queued_at or not self.created_at:
            return None
        return (self.queued_at - self.created_at).total_seconds() * 1000

    def _calculate_execution_latency_ms(self) -> float | None:
        """Calculate execution latency in milliseconds"""
        if not self.started_at or not self.completed_at:
            return None
        return (self.completed_at - self.started_at).total_seconds() * 1000


class JobsAPI:
    """Jobs API implementation"""

    def __init__(self):
        """Initialize jobs API"""
        self.router = APIRouter(prefix="/v1/fog/jobs", tags=["jobs"])
        self._jobs: dict[str, FogJob] = {}  # In-memory storage for now
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes"""

        @self.router.post(
            "",
            response_model=JobResponse,
            status_code=status.HTTP_201_CREATED,
            summary="Submit fog job",
            description="Submit a new job for execution on the fog network",
        )
        async def submit_job(
            job_spec: JobSpec,
            current_user: str = Header(alias="X-User-ID"),  # User ID from header
            authorization: str = Header(alias="Authorization"),  # JWT token
        ) -> JobResponse:
            """Submit new fog job for execution"""

            try:
                # Create new job
                job = FogJob(spec=job_spec, status=JobStatus.PENDING)

                # Validate RBAC permissions
                await self._validate_job_permissions(current_user, authorization, job_spec.namespace)

                # Validate namespace quota (integrate with existing quota system)
                await self._validate_namespace_quota(job_spec.namespace, current_user)

                # PII/PHI compliance scan on job inputs
                await self._scan_job_inputs(job_spec)

                # Store job (in production: persist to database)
                self._jobs[job.job_id] = job

                # TODO: Submit to scheduler for placement
                # placement_result = await self._schedule_job(job)
                # job.assigned_nodes = placement_result.assigned_nodes
                # job.placement_score = placement_result.score

                # Mock placement for now
                job.status = JobStatus.QUEUED
                job.queued_at = datetime.now(UTC)

                logger.info(f"Job {job.job_id} submitted to namespace {job_spec.namespace}")

                return JobResponse(
                    job_id=job.job_id,
                    status=job.status,
                    created_at=job.created_at,
                    estimated_start_time=datetime.now(UTC) + timedelta(seconds=30),
                    placement_score=0.8,  # Mock score
                    assigned_nodes=["mobile-node-1"],  # Mock node
                )

            except Exception as e:
                logger.error(f"Failed to submit job: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to submit job: {str(e)}"
                )

        @self.router.get(
            "/{job_id}",
            response_model=JobStatusResponse,
            summary="Get job status",
            description="Get detailed status and execution results for a job",
        )
        async def get_job(
            job_id: str,
            # TODO: Add RBAC dependency
            # current_user: User = Security(get_current_user, scopes=["fog.jobs.read"])
        ) -> JobStatusResponse:
            """Get job status and details"""

            job = self._jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")

            # TODO: Verify user has access to this job's namespace

            return job.to_response()

        @self.router.get(
            "/{job_id}/logs", summary="Stream job logs", description="Stream real-time logs from job execution"
        )
        async def get_job_logs(
            job_id: str,
            follow: bool = False,
            # TODO: Add RBAC dependency
            # current_user: User = Security(get_current_user, scopes=["fog.jobs.read"])
        ):
            """Stream job logs"""

            job = self._jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")

            # TODO: Stream logs from BetaNet transport
            # For now, return mock logs
            return {
                "job_id": job_id,
                "logs": job.logs or f"Mock logs for job {job_id}\nJob is {job.status.value}",
                "follow": follow,
            }

        @self.router.delete("/{job_id}", summary="Cancel job", description="Cancel a running or queued job")
        async def cancel_job(
            job_id: str,
            # TODO: Add RBAC dependency
            # current_user: User = Security(get_current_user, scopes=["fog.jobs.delete"])
        ):
            """Cancel job execution"""

            job = self._jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")

            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=f"Cannot cancel job in {job.status.value} state"
                )

            # TODO: Send cancellation over BetaNet to assigned nodes
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now(UTC)

            logger.info(f"Job {job_id} cancelled")

            return {"message": f"Job {job_id} cancelled successfully"}

        @self.router.get(
            "",
            response_model=list[JobStatusResponse],
            summary="List jobs",
            description="List jobs in accessible namespaces",
        )
        async def list_jobs(
            namespace: str | None = None,
            status: JobStatus | None = None,
            limit: int = 100,
            # TODO: Add RBAC dependency
            # current_user: User = Security(get_current_user, scopes=["fog.jobs.read"])
        ) -> list[JobStatusResponse]:
            """List jobs with optional filtering"""

            jobs = list(self._jobs.values())

            # Filter by namespace
            if namespace:
                # TODO: Verify user has access to namespace
                jobs = [job for job in jobs if job.spec and job.spec.namespace == namespace]

            # Filter by status
            if status:
                jobs = [job for job in jobs if job.status == status]

            # Apply limit
            jobs = jobs[:limit]

            return [job.to_response() for job in jobs]

    async def _validate_namespace_quota(self, namespace: str, user):
        """Validate user has quota in namespace"""
        # Integrate with existing quota system
        try:
            from infrastructure.shared.security.rbac_system import RBACSystem

            rbac = RBACSystem()
            # Extract tenant_id from namespace or user context
            tenant_id = namespace.split("-")[0] if "-" in namespace else user
            quota_available = await rbac.check_quota(tenant_id, "fog_jobs", 1)
            if not quota_available:
                raise HTTPException(status_code=429, detail="Job quota exceeded for namespace")
        except ImportError:
            logger.warning("RBAC system not available, allowing job")

    async def _scan_job_inputs(self, job_spec: JobSpec):
        """Scan job inputs for PII/PHI violations"""
        # Integrate with existing compliance scanner
        try:
            from packages.core.compliance.pii_scanner import PIIScanner

            scanner = PIIScanner()

            # Scan image and command for PII/PHI
            content_to_scan = f"{job_spec.image} {' '.join(job_spec.command)}"
            scan_result = await scanner.scan_content(content_to_scan)

            if scan_result.has_violations:
                violations = ", ".join(scan_result.violations)
                raise HTTPException(status_code=400, detail=f"Job contains sensitive data: {violations}")
        except ImportError:
            logger.warning("PII scanner not available, skipping compliance check")

    async def _schedule_job(self, job: FogJob):
        """Schedule job for execution"""
        # Integrate with scheduler
        try:
            from packages.fog.core.scheduler import FogScheduler

            scheduler = FogScheduler()
            placement = await scheduler.schedule_job(job)

            if placement:
                job.status = JobStatus.SCHEDULED
                job.assigned_nodes = placement.get("nodes", [])
            else:
                job.status = JobStatus.QUEUED
        except ImportError:
            logger.warning("Fog scheduler not available, job queued locally")
            job.status = JobStatus.QUEUED

    async def _validate_job_permissions(self, user_id: str, authorization: str, namespace: str):
        """Validate user has permissions to submit job in namespace."""
        try:
            from infrastructure.shared.security.rbac_system import Permission, RBACSystem

            rbac = RBACSystem()

            # Validate JWT token
            import os

            import jwt

            jwt_secret = os.getenv("JWT_SECRET")
            jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
            if not jwt_secret:
                logger.error("JWT secret not configured")
                raise HTTPException(status_code=500, detail="JWT secret not configured")

            token = authorization.replace("Bearer ", "")
            try:
                payload = jwt.decode(token, jwt_secret, algorithms=[jwt_algorithm])
            except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
                raise HTTPException(status_code=403, detail="Invalid authorization token")

            if payload.get("user_id") != user_id:
                raise HTTPException(status_code=403, detail="Invalid token")

            # Check fog job creation permission
            has_permission = await rbac.check_permission(user_id, Permission.FOG_JOB_SUBMIT)
            if not has_permission:
                raise HTTPException(status_code=403, detail="Insufficient permissions for fog job submission")

        except ImportError:
            logger.warning("RBAC system not available, allowing job")

    async def _validate_job_access(self, user_id: str, namespace: str):
        """Validate user has access to job's namespace."""
        try:
            from infrastructure.shared.security.rbac_system import RBACSystem

            rbac = RBACSystem()
            user = rbac.users.get(user_id)
            if user and namespace.startswith(user.tenant_id):
                return  # User owns the namespace
            raise HTTPException(status_code=403, detail="Access denied to job namespace")
        except ImportError:
            logger.warning("RBAC system not available, allowing access")

    async def _validate_admin_access(self, user_id: str, namespace: str):
        """Validate user has admin access to namespace."""
        try:
            from infrastructure.shared.security.rbac_system import RBACSystem, Role

            rbac = RBACSystem()
            user = rbac.users.get(user_id)
            if user and user.role in [Role.ADMIN, Role.SUPER_ADMIN]:
                return  # Admin access
            raise HTTPException(status_code=403, detail="Admin access required")
        except ImportError:
            logger.warning("RBAC system not available, allowing admin access")

    async def _submit_to_scheduler(self, job):
        """Submit job to scheduler for placement."""
        await self._schedule_job(job)

    async def _stream_job_logs(self, job_id: str):
        """Stream job logs from BetaNet transport."""
        try:
            from packages.p2p.betanet.htx_transport import HtxClient

            HtxClient()

            # In production: stream real logs from assigned nodes
            # For now: simulate log streaming
            import asyncio

            logs = [
                f"[{job_id}] Job started",
                f"[{job_id}] Initializing container...",
                f"[{job_id}] Running user command...",
                f"[{job_id}] Job completed",
            ]

            for log in logs:
                yield log
                await asyncio.sleep(0.5)

        except ImportError:
            yield f"[{job_id}] BetaNet transport not available, using local logs"

    async def _cancel_job_on_nodes(self, job):
        """Send cancellation to assigned nodes via BetaNet."""
        try:
            from packages.p2p.betanet.htx_transport import HtxClient

            HtxClient()

            # In production: send cancellation messages to assigned nodes
            job.status = JobStatus.CANCELLED
            logger.info(f"Cancelled job {job.job_id} on {len(job.assigned_nodes or [])} nodes")

        except ImportError:
            job.status = JobStatus.CANCELLED
            logger.warning("BetaNet transport not available, job marked cancelled locally")


# Factory function
def create_jobs_api() -> JobsAPI:
    """Create jobs API instance"""
    return JobsAPI()
