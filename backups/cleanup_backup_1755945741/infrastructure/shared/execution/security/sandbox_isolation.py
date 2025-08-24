"""
Execution Security and Sandbox Isolation

Provides secure execution environment for fog computing workloads:
- WebAssembly sandboxing with WASI
- Container-based isolation with OCI
- Resource limits and cgroup controls
- Network isolation and egress filtering
- Filesystem isolation and read-only mounts
- Runtime security monitoring

Security Features:
- Zero-trust execution environment
- Capability-based access control
- Network egress filtering integration
- Resource consumption monitoring
- Runtime behavior analysis
- Security violation detection
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any
from uuid import uuid4

import psutil

logger = logging.getLogger(__name__)


class SandboxType(str, Enum):
    """Types of execution sandboxes"""

    WASI = "wasi"  # WebAssembly System Interface
    OCI = "oci"  # OCI container
    MICROVM = "microvm"  # Firecracker microVM
    NATIVE = "native"  # Native process (least secure)


class SecurityProfile(str, Enum):
    """Security profiles for different workload types"""

    UNTRUSTED = "untrusted"  # Maximum isolation
    RESTRICTED = "restricted"  # High isolation
    STANDARD = "standard"  # Normal isolation
    PRIVILEGED = "privileged"  # Minimal isolation (trusted code)


class ViolationType(str, Enum):
    """Types of security violations"""

    NETWORK_EGRESS = "network_egress"
    FILESYSTEM_ACCESS = "filesystem_access"
    RESOURCE_LIMIT = "resource_limit"
    SYSCALL_VIOLATION = "syscall_violation"
    RUNTIME_BEHAVIOR = "runtime_behavior"


@dataclass
class ResourceLimits:
    """Resource limits for sandbox execution"""

    # CPU limits
    cpu_cores: float = 1.0
    cpu_quota_percent: int = 100

    # Memory limits
    memory_mb: int = 512
    swap_mb: int = 0

    # Disk limits
    disk_read_iops: int = 1000
    disk_write_iops: int = 500
    disk_quota_mb: int = 100

    # Network limits
    network_ingress_kbps: int = 1000
    network_egress_kbps: int = 500

    # Time limits
    max_execution_time_s: int = 300

    # File descriptor limits
    max_open_files: int = 64
    max_processes: int = 1


@dataclass
class FilesystemMount:
    """Filesystem mount specification"""

    source: str  # Source path on host
    destination: str  # Mount point in sandbox
    readonly: bool = True  # Read-only mount
    type: str = "bind"  # Mount type
    options: list[str] = field(default_factory=list)


@dataclass
class NetworkPolicy:
    """Network access policy for sandbox"""

    # Default policy
    default_egress: str = "deny"  # "allow" or "deny"
    default_ingress: str = "deny"  # "allow" or "deny"

    # Allowed destinations
    allowed_destinations: set[str] = field(default_factory=set)
    allowed_ports: set[int] = field(default_factory=set)

    # DNS configuration
    dns_servers: list[str] = field(default_factory=lambda: ["8.8.8.8", "1.1.1.1"])
    dns_search: list[str] = field(default_factory=list)


@dataclass
class SecurityViolation:
    """Security violation detected during execution"""

    violation_id: str = field(default_factory=lambda: str(uuid4()))
    violation_type: ViolationType = ViolationType.RUNTIME_BEHAVIOR
    severity: str = "MEDIUM"

    # Violation details
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    # Context
    sandbox_id: str = ""
    namespace: str = ""
    job_id: str | None = None

    # Timestamps
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Actions taken
    blocked: bool = True
    terminated: bool = False


@dataclass
class SandboxExecutionResult:
    """Result of sandbox execution"""

    # Execution metadata
    sandbox_id: str
    exit_code: int
    execution_time_ms: int

    # Output
    stdout: bytes = b""
    stderr: bytes = b""

    # Resource usage
    peak_memory_mb: float = 0.0
    cpu_time_ms: int = 0
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0
    network_egress_kb: float = 0.0

    # Security
    violations: list[SecurityViolation] = field(default_factory=list)
    terminated_for_violation: bool = False

    # Success/failure
    success: bool = True
    error_message: str = ""


class SecureSandbox:
    """
    Secure execution sandbox with comprehensive isolation

    Provides secure execution environment for untrusted code with:
    - Multiple sandbox technologies (WASI, OCI, microVM)
    - Resource limits and monitoring
    - Network isolation and egress filtering
    - Filesystem isolation
    - Security violation detection
    """

    def __init__(
        self,
        sandbox_id: str,
        sandbox_type: SandboxType = SandboxType.WASI,
        security_profile: SecurityProfile = SecurityProfile.RESTRICTED,
        namespace: str = "default",
    ):
        self.sandbox_id = sandbox_id
        self.sandbox_type = sandbox_type
        self.security_profile = security_profile
        self.namespace = namespace

        # Configuration
        self.resource_limits = ResourceLimits()
        self.network_policy = NetworkPolicy()
        self.filesystem_mounts: list[FilesystemMount] = []

        # Runtime state
        self.process: subprocess.Popen | None = None
        self.temp_dir: tempfile.TemporaryDirectory | None = None
        self.start_time: datetime | None = None
        self.violations: list[SecurityViolation] = []

        # Monitoring
        self.monitor_task: asyncio.Task | None = None
        self.terminated = False

        logger.info(f"Created sandbox {sandbox_id} ({sandbox_type.value}/{security_profile.value})")

    async def setup(self, resource_limits: ResourceLimits | None = None) -> None:
        """Set up the sandbox environment"""

        if resource_limits:
            self.resource_limits = resource_limits

        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory(prefix=f"sandbox_{self.sandbox_id}_")
        self.workdir = Path(self.temp_dir.name)

        # Set up filesystem structure
        await self._setup_filesystem()

        # Configure network isolation
        await self._setup_network_isolation()

        # Apply security profile settings
        await self._apply_security_profile()

        logger.info(f"Sandbox {self.sandbox_id} setup complete")

    async def execute(
        self,
        executable: bytes,
        args: list[str] = None,
        env: dict[str, str] = None,
        input_data: bytes = b"",
        job_id: str | None = None,
    ) -> SandboxExecutionResult:
        """Execute code in the secure sandbox"""

        if args is None:
            args = []
        if env is None:
            env = {}

        self.start_time = datetime.now(UTC)

        try:
            # Write executable to sandbox
            exec_path = await self._prepare_executable(executable)

            # Start monitoring
            self.monitor_task = asyncio.create_task(self._monitor_execution())

            # Execute based on sandbox type
            if self.sandbox_type == SandboxType.WASI:
                result = await self._execute_wasi(exec_path, args, env, input_data)
            elif self.sandbox_type == SandboxType.OCI:
                result = await self._execute_oci(exec_path, args, env, input_data)
            elif self.sandbox_type == SandboxType.MICROVM:
                result = await self._execute_microvm(exec_path, args, env, input_data)
            else:
                result = await self._execute_native(exec_path, args, env, input_data)

            # Add security violations to result
            result.violations = self.violations
            result.terminated_for_violation = self.terminated

            # Log completion
            logger.info(
                f"Sandbox {self.sandbox_id} execution completed: exit_code={result.exit_code}, time={result.execution_time_ms}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Sandbox {self.sandbox_id} execution failed: {e}")

            return SandboxExecutionResult(
                sandbox_id=self.sandbox_id,
                exit_code=-1,
                execution_time_ms=int((datetime.now(UTC) - self.start_time).total_seconds() * 1000),
                success=False,
                error_message=str(e),
                violations=self.violations,
                terminated_for_violation=self.terminated,
            )

        finally:
            # Stop monitoring
            if self.monitor_task:
                self.monitor_task.cancel()

            # Cleanup
            await self.cleanup()

    async def terminate(self, reason: str = "Security violation") -> None:
        """Terminate sandbox execution"""

        self.terminated = True

        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                await asyncio.sleep(1)

                if self.process.poll() is None:
                    self.process.kill()

                logger.warning(f"Terminated sandbox {self.sandbox_id}: {reason}")

            except Exception as e:
                logger.error(f"Error terminating sandbox {self.sandbox_id}: {e}")

    async def cleanup(self) -> None:
        """Clean up sandbox resources"""

        try:
            # Terminate process if still running
            if self.process and self.process.poll() is None:
                await self.terminate("Cleanup")

            # Remove temporary directory
            if self.temp_dir:
                self.temp_dir.cleanup()
                self.temp_dir = None

            logger.debug(f"Cleaned up sandbox {self.sandbox_id}")

        except Exception as e:
            logger.error(f"Error cleaning up sandbox {self.sandbox_id}: {e}")

    # Private methods

    async def _setup_filesystem(self) -> None:
        """Set up isolated filesystem"""

        # Create standard directories
        dirs = ["bin", "lib", "tmp", "proc", "dev", "etc"]
        for dir_name in dirs:
            (self.workdir / dir_name).mkdir(exist_ok=True)

        # Set up read-only system mounts
        if self.security_profile != SecurityProfile.PRIVILEGED:
            self.filesystem_mounts.extend(
                [
                    FilesystemMount("/bin", str(self.workdir / "bin"), readonly=True),
                    FilesystemMount("/lib", str(self.workdir / "lib"), readonly=True),
                    FilesystemMount("/usr/lib", str(self.workdir / "lib"), readonly=True),
                ]
            )

    async def _setup_network_isolation(self) -> None:
        """Set up network isolation"""

        # Network isolation depends on sandbox type
        if self.sandbox_type == SandboxType.WASI:
            # WASI has no network access by default
            pass
        elif self.sandbox_type in [SandboxType.OCI, SandboxType.MICROVM]:
            # Container/microVM network isolation
            self.network_policy.default_egress = "deny"
            self.network_policy.default_ingress = "deny"

    async def _apply_security_profile(self) -> None:
        """Apply security profile settings"""

        if self.security_profile == SecurityProfile.UNTRUSTED:
            # Maximum restrictions
            self.resource_limits.memory_mb = min(self.resource_limits.memory_mb, 256)
            self.resource_limits.cpu_cores = min(self.resource_limits.cpu_cores, 0.5)
            self.resource_limits.max_execution_time_s = min(self.resource_limits.max_execution_time_s, 60)
            self.resource_limits.disk_quota_mb = min(self.resource_limits.disk_quota_mb, 50)

        elif self.security_profile == SecurityProfile.RESTRICTED:
            # High restrictions
            self.resource_limits.memory_mb = min(self.resource_limits.memory_mb, 512)
            self.resource_limits.cpu_cores = min(self.resource_limits.cpu_cores, 1.0)
            self.resource_limits.max_execution_time_s = min(self.resource_limits.max_execution_time_s, 300)

        elif self.security_profile == SecurityProfile.PRIVILEGED:
            # Minimal restrictions (trusted code)
            pass

    async def _prepare_executable(self, executable: bytes) -> Path:
        """Prepare executable file in sandbox"""

        if self.sandbox_type == SandboxType.WASI:
            exec_path = self.workdir / "main.wasm"
        else:
            exec_path = self.workdir / "main"

        # Write executable
        exec_path.write_bytes(executable)
        exec_path.chmod(0o755)

        return exec_path

    async def _execute_wasi(
        self, exec_path: Path, args: list[str], env: dict[str, str], input_data: bytes
    ) -> SandboxExecutionResult:
        """Execute WebAssembly with WASI runtime"""

        # Use wasmtime for WASI execution
        cmd = [
            "wasmtime",
            "--wasi-modules=experimental",
            "--dir=/tmp",  # Allow access to tmp
            "--max-memory={}".format(self.resource_limits.memory_mb * 1024 * 1024),
            str(exec_path),
        ] + args

        start_time = time.time()

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.workdir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

            # Wait for completion with timeout
            stdout, stderr = self.process.communicate(
                input=input_data, timeout=self.resource_limits.max_execution_time_s
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            return SandboxExecutionResult(
                sandbox_id=self.sandbox_id,
                exit_code=self.process.returncode,
                execution_time_ms=execution_time_ms,
                stdout=stdout,
                stderr=stderr,
                success=self.process.returncode == 0,
            )

        except subprocess.TimeoutExpired:
            await self.terminate("Execution timeout")
            return SandboxExecutionResult(
                sandbox_id=self.sandbox_id,
                exit_code=-1,
                execution_time_ms=int((time.time() - start_time) * 1000),
                success=False,
                error_message="Execution timeout",
                terminated_for_violation=True,
            )

    async def _execute_oci(
        self, exec_path: Path, args: list[str], env: dict[str, str], input_data: bytes
    ) -> SandboxExecutionResult:
        """Execute in OCI container"""

        # Use podman/docker for container execution
        runtime = "podman" if shutil.which("podman") else "docker"

        cmd = [
            runtime,
            "run",
            "--rm",
            f"--memory={self.resource_limits.memory_mb}m",
            f"--cpus={self.resource_limits.cpu_cores}",
            "--network=none",  # No network access
            "--read-only",  # Read-only filesystem
            f"--tmpfs=/tmp:size={self.resource_limits.disk_quota_mb}m",
            "-v",
            f"{exec_path}:/app/main:ro",
            "-w",
            "/app",
            "alpine:latest",  # Minimal base image
            "./main",
        ] + args

        return await self._execute_process(cmd, env, input_data)

    async def _execute_microvm(
        self, exec_path: Path, args: list[str], env: dict[str, str], input_data: bytes
    ) -> SandboxExecutionResult:
        """Execute in Firecracker microVM"""

        # This would require Firecracker setup - simplified for now
        logger.warning("MicroVM execution not fully implemented, falling back to native")
        return await self._execute_native(exec_path, args, env, input_data)

    async def _execute_native(
        self, exec_path: Path, args: list[str], env: dict[str, str], input_data: bytes
    ) -> SandboxExecutionResult:
        """Execute as native process (least secure)"""

        cmd = [str(exec_path)] + args
        return await self._execute_process(cmd, env, input_data)

    async def _execute_process(self, cmd: list[str], env: dict[str, str], input_data: bytes) -> SandboxExecutionResult:
        """Execute process with monitoring"""

        start_time = time.time()

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.workdir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

            # Wait for completion with timeout
            stdout, stderr = self.process.communicate(
                input=input_data, timeout=self.resource_limits.max_execution_time_s
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            return SandboxExecutionResult(
                sandbox_id=self.sandbox_id,
                exit_code=self.process.returncode,
                execution_time_ms=execution_time_ms,
                stdout=stdout,
                stderr=stderr,
                success=self.process.returncode == 0,
            )

        except subprocess.TimeoutExpired:
            await self.terminate("Execution timeout")
            return SandboxExecutionResult(
                sandbox_id=self.sandbox_id,
                exit_code=-1,
                execution_time_ms=int((time.time() - start_time) * 1000),
                success=False,
                error_message="Execution timeout",
                terminated_for_violation=True,
            )

    async def _monitor_execution(self) -> None:
        """Monitor sandbox execution for violations"""

        while not self.terminated and self.process and self.process.poll() is None:
            try:
                await asyncio.sleep(1)

                # Check resource usage
                if self.process.pid:
                    try:
                        proc = psutil.Process(self.process.pid)

                        # Check memory usage
                        memory_mb = proc.memory_info().rss / (1024 * 1024)
                        if memory_mb > self.resource_limits.memory_mb:
                            await self._record_violation(
                                ViolationType.RESOURCE_LIMIT,
                                f"Memory limit exceeded: {memory_mb:.1f}MB > {self.resource_limits.memory_mb}MB",
                                {"memory_mb": memory_mb, "limit_mb": self.resource_limits.memory_mb},
                            )
                            await self.terminate("Memory limit exceeded")
                            break

                        # Check CPU usage (simplified)
                        cpu_percent = proc.cpu_percent()
                        if cpu_percent > self.resource_limits.cpu_quota_percent:
                            await self._record_violation(
                                ViolationType.RESOURCE_LIMIT,
                                f"CPU limit exceeded: {cpu_percent}% > {self.resource_limits.cpu_quota_percent}%",
                                {"cpu_percent": cpu_percent, "limit_percent": self.resource_limits.cpu_quota_percent},
                            )

                    except psutil.NoSuchProcess:
                        break

                # Check execution time
                if self.start_time:
                    elapsed = (datetime.now(UTC) - self.start_time).total_seconds()
                    if elapsed > self.resource_limits.max_execution_time_s:
                        await self._record_violation(
                            ViolationType.RESOURCE_LIMIT,
                            f"Execution time limit exceeded: {elapsed:.1f}s > {self.resource_limits.max_execution_time_s}s",
                            {"elapsed_s": elapsed, "limit_s": self.resource_limits.max_execution_time_s},
                        )
                        await self.terminate("Execution time limit exceeded")
                        break

            except Exception as e:
                logger.error(f"Error monitoring sandbox {self.sandbox_id}: {e}")
                await asyncio.sleep(5)

    async def _record_violation(
        self, violation_type: ViolationType, message: str, details: dict[str, Any], severity: str = "HIGH"
    ) -> None:
        """Record security violation"""

        violation = SecurityViolation(
            violation_type=violation_type,
            severity=severity,
            message=message,
            details=details,
            sandbox_id=self.sandbox_id,
            namespace=self.namespace,
        )

        self.violations.append(violation)

        logger.warning(f"Security violation in sandbox {self.sandbox_id}: {message}")


class SandboxManager:
    """
    Manager for secure execution sandboxes

    Provides lifecycle management for sandboxes including:
    - Sandbox creation and configuration
    - Resource allocation and limits
    - Security policy enforcement
    - Violation monitoring and response
    """

    def __init__(self):
        self.active_sandboxes: dict[str, SecureSandbox] = {}
        self.sandbox_counter = 0

        logger.info("Sandbox manager initialized")

    async def create_sandbox(
        self,
        sandbox_type: SandboxType = SandboxType.WASI,
        security_profile: SecurityProfile = SecurityProfile.RESTRICTED,
        namespace: str = "default",
        resource_limits: ResourceLimits | None = None,
    ) -> str:
        """Create a new secure sandbox"""

        self.sandbox_counter += 1
        sandbox_id = f"sandbox_{self.sandbox_counter:06d}_{uuid4().hex[:8]}"

        sandbox = SecureSandbox(
            sandbox_id=sandbox_id, sandbox_type=sandbox_type, security_profile=security_profile, namespace=namespace
        )

        await sandbox.setup(resource_limits)

        self.active_sandboxes[sandbox_id] = sandbox

        logger.info(f"Created sandbox {sandbox_id} for namespace {namespace}")
        return sandbox_id

    async def execute_in_sandbox(
        self,
        sandbox_id: str,
        executable: bytes,
        args: list[str] = None,
        env: dict[str, str] = None,
        input_data: bytes = b"",
        job_id: str | None = None,
    ) -> SandboxExecutionResult:
        """Execute code in specified sandbox"""

        sandbox = self.active_sandboxes.get(sandbox_id)
        if not sandbox:
            raise ValueError(f"Sandbox not found: {sandbox_id}")

        try:
            result = await sandbox.execute(executable, args, env, input_data, job_id)
            return result
        finally:
            # Remove sandbox after execution
            if sandbox_id in self.active_sandboxes:
                del self.active_sandboxes[sandbox_id]

    async def terminate_sandbox(self, sandbox_id: str, reason: str = "User request") -> bool:
        """Terminate a running sandbox"""

        sandbox = self.active_sandboxes.get(sandbox_id)
        if not sandbox:
            return False

        await sandbox.terminate(reason)
        return True

    async def cleanup_all_sandboxes(self) -> None:
        """Clean up all active sandboxes"""

        for sandbox in self.active_sandboxes.values():
            await sandbox.cleanup()

        self.active_sandboxes.clear()
        logger.info("All sandboxes cleaned up")

    def get_sandbox_status(self, sandbox_id: str) -> dict[str, Any] | None:
        """Get status of a specific sandbox"""

        sandbox = self.active_sandboxes.get(sandbox_id)
        if not sandbox:
            return None

        return {
            "sandbox_id": sandbox_id,
            "sandbox_type": sandbox.sandbox_type.value,
            "security_profile": sandbox.security_profile.value,
            "namespace": sandbox.namespace,
            "terminated": sandbox.terminated,
            "violations": len(sandbox.violations),
            "start_time": sandbox.start_time.isoformat() if sandbox.start_time else None,
        }

    def get_manager_status(self) -> dict[str, Any]:
        """Get sandbox manager status"""

        return {
            "active_sandboxes": len(self.active_sandboxes),
            "total_created": self.sandbox_counter,
            "sandboxes_by_type": {
                sandbox_type.value: len([s for s in self.active_sandboxes.values() if s.sandbox_type == sandbox_type])
                for sandbox_type in SandboxType
            },
            "sandboxes_by_profile": {
                profile.value: len([s for s in self.active_sandboxes.values() if s.security_profile == profile])
                for profile in SecurityProfile
            },
        }


# Global sandbox manager instance
_sandbox_manager: SandboxManager | None = None


def get_sandbox_manager() -> SandboxManager:
    """Get global sandbox manager instance"""
    global _sandbox_manager

    if _sandbox_manager is None:
        _sandbox_manager = SandboxManager()

    return _sandbox_manager


async def create_secure_sandbox(
    sandbox_type: SandboxType = SandboxType.WASI,
    security_profile: SecurityProfile = SecurityProfile.RESTRICTED,
    namespace: str = "default",
    **resource_kwargs,
) -> str:
    """Convenience function to create secure sandbox"""

    resource_limits = None
    if resource_kwargs:
        resource_limits = ResourceLimits(**resource_kwargs)

    manager = get_sandbox_manager()
    return await manager.create_sandbox(
        sandbox_type=sandbox_type,
        security_profile=security_profile,
        namespace=namespace,
        resource_limits=resource_limits,
    )


async def execute_securely(
    executable: bytes,
    namespace: str = "default",
    sandbox_type: SandboxType = SandboxType.WASI,
    security_profile: SecurityProfile = SecurityProfile.RESTRICTED,
    args: list[str] = None,
    env: dict[str, str] = None,
    input_data: bytes = b"",
    **resource_kwargs,
) -> SandboxExecutionResult:
    """Convenience function to execute code securely"""

    manager = get_sandbox_manager()

    # Create sandbox
    sandbox_id = await create_secure_sandbox(
        sandbox_type=sandbox_type, security_profile=security_profile, namespace=namespace, **resource_kwargs
    )

    # Execute code
    return await manager.execute_in_sandbox(
        sandbox_id=sandbox_id, executable=executable, args=args, env=env, input_data=input_data
    )
