"""
Fog Sandboxes API

Provides REST endpoints for sandbox management:
- POST /v1/fog/sandboxes - Create new sandbox
- GET /v1/fog/sandboxes/{sandbox_id} - Get sandbox details
- POST /v1/fog/sandboxes/{sandbox_id}/exec - Execute command in sandbox
- DELETE /v1/fog/sandboxes/{sandbox_id} - Destroy sandbox

Sandboxes provide persistent execution environments for interactive development.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class SandboxStatus(str, Enum):
    """Sandbox lifecycle states"""

    CREATING = "creating"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    ERROR = "error"


class SandboxType(str, Enum):
    """Types of sandboxes"""

    INTERACTIVE = "interactive"  # Long-running development environment
    BATCH = "batch"  # Short-lived batch processing
    PERSISTENT = "persistent"  # Persistent data + compute


class SandboxSpec(BaseModel):
    """Sandbox creation specification"""

    namespace: str = Field(..., description="Namespace for RBAC and billing")
    type: SandboxType = Field(SandboxType.INTERACTIVE, description="Sandbox type")
    image: str = Field(..., description="Base image (CAS hash or registry ref)")

    # Resource allocation
    resources: dict[str, Any] = Field(
        default_factory=lambda: {"cpu_cores": 1.0, "memory_mb": 1024, "disk_mb": 5120, "max_idle_minutes": 60},
        description="Resource requirements and limits",
    )

    # Environment
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    working_dir: str = Field("/workspace", description="Working directory")

    # Networking (restricted by default)
    network_access: bool = Field(False, description="Allow network access")
    exposed_ports: list[int] = Field(default_factory=list, description="Ports to expose")

    # Data persistence
    persistent_volumes: list[dict[str, str]] = Field(default_factory=list, description="Persistent volume mounts")

    # Labels and metadata
    labels: dict[str, str] = Field(default_factory=dict, description="User labels")

    @validator("namespace")
    def validate_namespace(cls, v):
        """Validate namespace format"""
        if not v or "/" not in v:
            raise ValueError("Namespace must be in format 'org/team'")
        return v


class SandboxResponse(BaseModel):
    """Sandbox creation response"""

    sandbox_id: str = Field(..., description="Unique sandbox identifier")
    status: SandboxStatus = Field(..., description="Current sandbox status")
    created_at: datetime = Field(..., description="Creation timestamp")

    # Connection information
    connection_url: str | None = Field(None, description="Connection URL (when active)")
    ssh_command: str | None = Field(None, description="SSH connection command")


class SandboxStatusResponse(BaseModel):
    """Detailed sandbox status"""

    sandbox_id: str
    status: SandboxStatus
    spec: SandboxSpec

    # Timestamps
    created_at: datetime
    started_at: datetime | None = None
    last_activity: datetime | None = None
    terminated_at: datetime | None = None

    # Connection details
    connection_url: str | None = None
    ssh_command: str | None = None
    assigned_node: str | None = None

    # Resource usage
    cpu_usage_percent: float = 0.0
    memory_usage_mb: int = 0
    disk_usage_mb: int = 0

    # Activity metrics
    idle_minutes: int = 0
    commands_executed: int = 0


class ExecRequest(BaseModel):
    """Execute command in sandbox request"""

    command: str = Field(..., description="Command to execute")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    working_dir: str | None = Field(None, description="Working directory override")
    env: dict[str, str] = Field(default_factory=dict, description="Additional environment vars")
    timeout_seconds: int = Field(30, description="Command timeout")


class ExecResponse(BaseModel):
    """Command execution response"""

    execution_id: str = Field(..., description="Execution identifier")
    exit_code: int | None = Field(None, description="Exit code (when completed)")
    stdout: str = Field("", description="Standard output")
    stderr: str = Field("", description="Standard error")
    duration_ms: float = Field(0.0, description="Execution duration")


@dataclass
class FogSandbox:
    """Internal sandbox representation"""

    sandbox_id: str = field(default_factory=lambda: str(uuid4()))
    spec: SandboxSpec | None = None
    status: SandboxStatus = SandboxStatus.CREATING

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    last_activity: datetime | None = None
    terminated_at: datetime | None = None

    # Connection details
    assigned_node: str | None = None
    connection_url: str | None = None
    ssh_port: int | None = None

    # Resource usage tracking
    cpu_usage_percent: float = 0.0
    memory_usage_mb: int = 0
    disk_usage_mb: int = 0
    commands_executed: int = 0

    def to_response(self) -> SandboxStatusResponse:
        """Convert to API response format"""

        ssh_command = None
        if self.assigned_node and self.ssh_port:
            ssh_command = f"ssh -p {self.ssh_port} user@{self.assigned_node}"

        idle_minutes = 0
        if self.last_activity:
            idle_minutes = int((datetime.now(UTC) - self.last_activity).total_seconds() / 60)

        return SandboxStatusResponse(
            sandbox_id=self.sandbox_id,
            status=self.status,
            spec=self.spec,
            created_at=self.created_at,
            started_at=self.started_at,
            last_activity=self.last_activity,
            terminated_at=self.terminated_at,
            connection_url=self.connection_url,
            ssh_command=ssh_command,
            assigned_node=self.assigned_node,
            cpu_usage_percent=self.cpu_usage_percent,
            memory_usage_mb=self.memory_usage_mb,
            disk_usage_mb=self.disk_usage_mb,
            idle_minutes=idle_minutes,
            commands_executed=self.commands_executed,
        )


class SandboxAPI:
    """Sandbox API implementation"""

    def __init__(self):
        """Initialize sandbox API"""
        self.router = APIRouter(prefix="/v1/fog/sandboxes", tags=["sandboxes"])
        self._sandboxes: dict[str, FogSandbox] = {}  # In-memory storage for now
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes"""

        @self.router.post(
            "",
            response_model=SandboxResponse,
            status_code=status.HTTP_201_CREATED,
            summary="Create sandbox",
            description="Create a new interactive sandbox environment",
        )
        async def create_sandbox(
            sandbox_spec: SandboxSpec,
            # TODO: Add RBAC dependency
            # current_user: User = Security(get_current_user, scopes=["fog.sandboxes.create"])
        ) -> SandboxResponse:
            """Create new sandbox"""

            try:
                # Create new sandbox
                sandbox = FogSandbox(spec=sandbox_spec, status=SandboxStatus.CREATING)

                # TODO: Validate namespace quota
                # TODO: Compliance scan on image and environment

                # Store sandbox
                self._sandboxes[sandbox.sandbox_id] = sandbox

                # TODO: Schedule sandbox creation on fog node
                # For now, mock the creation process
                sandbox.status = SandboxStatus.ACTIVE
                sandbox.started_at = datetime.now(UTC)
                sandbox.assigned_node = "fog-node-1"  # Mock assignment
                sandbox.connection_url = f"https://sandbox-{sandbox.sandbox_id}.fog.aivillage.dev"
                sandbox.ssh_port = 2222

                logger.info(f"Sandbox {sandbox.sandbox_id} created in namespace {sandbox_spec.namespace}")

                return SandboxResponse(
                    sandbox_id=sandbox.sandbox_id,
                    status=sandbox.status,
                    created_at=sandbox.created_at,
                    connection_url=sandbox.connection_url,
                    ssh_command=f"ssh -p {sandbox.ssh_port} user@{sandbox.assigned_node}",
                )

            except Exception as e:
                logger.error(f"Failed to create sandbox: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create sandbox: {str(e)}"
                )

        @self.router.get(
            "/{sandbox_id}",
            response_model=SandboxStatusResponse,
            summary="Get sandbox status",
            description="Get detailed status and connection info for sandbox",
        )
        async def get_sandbox(
            sandbox_id: str,
            # TODO: Add RBAC dependency
        ) -> SandboxStatusResponse:
            """Get sandbox status and details"""

            sandbox = self._sandboxes.get(sandbox_id)
            if not sandbox:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Sandbox {sandbox_id} not found")

            # TODO: Verify user has access to this sandbox's namespace

            # Update activity metrics (mock)
            sandbox.cpu_usage_percent = 25.5
            sandbox.memory_usage_mb = 512
            sandbox.disk_usage_mb = 1024

            return sandbox.to_response()

        @self.router.post(
            "/{sandbox_id}/exec",
            response_model=ExecResponse,
            summary="Execute command",
            description="Execute a command in the sandbox environment",
        )
        async def exec_command(
            sandbox_id: str,
            exec_request: ExecRequest,
            # TODO: Add RBAC dependency
        ) -> ExecResponse:
            """Execute command in sandbox"""

            sandbox = self._sandboxes.get(sandbox_id)
            if not sandbox:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Sandbox {sandbox_id} not found")

            if sandbox.status != SandboxStatus.ACTIVE:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Cannot execute command in {sandbox.status.value} sandbox",
                )

            # TODO: Send command execution over BetaNet to assigned node
            # For now, return mock execution result

            execution_id = str(uuid4())

            # Mock command execution
            if exec_request.command == "echo":
                stdout = " ".join(exec_request.args) if exec_request.args else "Hello from fog sandbox!"
                exit_code = 0
            elif exec_request.command == "pwd":
                stdout = sandbox.spec.working_dir if sandbox.spec else "/workspace"
                exit_code = 0
            else:
                stdout = f"Mock execution of: {exec_request.command} {' '.join(exec_request.args)}"
                exit_code = 0

            # Update sandbox activity
            sandbox.last_activity = datetime.now(UTC)
            sandbox.commands_executed += 1

            logger.info(f"Executed command in sandbox {sandbox_id}: {exec_request.command}")

            return ExecResponse(
                execution_id=execution_id, exit_code=exit_code, stdout=stdout, stderr="", duration_ms=45.2
            )

        @self.router.delete(
            "/{sandbox_id}", summary="Delete sandbox", description="Terminate and delete sandbox environment"
        )
        async def delete_sandbox(
            sandbox_id: str,
            # TODO: Add RBAC dependency
        ):
            """Delete sandbox"""

            sandbox = self._sandboxes.get(sandbox_id)
            if not sandbox:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Sandbox {sandbox_id} not found")

            if sandbox.status == SandboxStatus.TERMINATED:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Sandbox already terminated")

            # TODO: Send termination command over BetaNet
            sandbox.status = SandboxStatus.TERMINATING
            sandbox.terminated_at = datetime.now(UTC)

            # Mock immediate termination
            sandbox.status = SandboxStatus.TERMINATED

            logger.info(f"Sandbox {sandbox_id} terminated")

            return {"message": f"Sandbox {sandbox_id} terminated successfully"}

        @self.router.get(
            "",
            response_model=list[SandboxStatusResponse],
            summary="List sandboxes",
            description="List sandboxes in accessible namespaces",
        )
        async def list_sandboxes(
            namespace: str | None = None,
            status: SandboxStatus | None = None,
            limit: int = 100,
            # TODO: Add RBAC dependency
        ) -> list[SandboxStatusResponse]:
            """List sandboxes with optional filtering"""

            sandboxes = list(self._sandboxes.values())

            # Filter by namespace
            if namespace:
                sandboxes = [s for s in sandboxes if s.spec and s.spec.namespace == namespace]

            # Filter by status
            if status:
                sandboxes = [s for s in sandboxes if s.status == status]

            # Apply limit
            sandboxes = sandboxes[:limit]

            return [sandbox.to_response() for sandbox in sandboxes]


# Factory function
def create_sandbox_api() -> SandboxAPI:
    """Create sandbox API instance"""
    return SandboxAPI()
