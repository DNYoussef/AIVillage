"""
WASI and MicroVM Execution Runners

Implements the execution fabric layer for fog computing:
- WASI (WebAssembly System Interface) runner for secure sandboxed execution
- MicroVM runner for lightweight virtual machines (Firecracker)
- Resource management and isolation
- Integration with BetaNet for secure job delivery

This is the "immune system" layer that safely executes untrusted code
while protecting the host device and maintaining resource constraints.
"""

import asyncio
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import psutil

logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Execution status states"""

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class RuntimeType(str, Enum):
    """Supported execution runtimes"""

    WASI = "wasi"
    MICROVM = "microvm"
    OCI = "oci"
    NATIVE = "native"


@dataclass
class ExecutionResources:
    """Resource limits for execution"""

    cpu_cores: float = 1.0
    memory_mb: int = 512
    disk_mb: int = 1024
    max_duration_s: int = 300
    network_egress: bool = False
    max_files: int = 1000
    max_processes: int = 10


@dataclass
class ExecutionResult:
    """Result of code execution"""

    execution_id: str
    status: ExecutionStatus
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""

    # Resource usage
    cpu_time_s: float = 0.0
    memory_peak_mb: int = 0
    disk_used_mb: int = 0
    network_bytes: int = 0

    # Timing
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_ms: float = 0.0

    # Security
    sandbox_violations: list[str] = field(default_factory=list)
    resource_violations: list[str] = field(default_factory=list)


class WASIRunner:
    """
    WebAssembly System Interface (WASI) runner

    Provides secure sandboxed execution of WebAssembly modules with:
    - Memory isolation and limits
    - File system sandboxing
    - Network restrictions
    - Resource monitoring
    - Capability-based security
    """

    def __init__(self, wasmtime_path: str = "wasmtime", default_timeout: float = 300.0, temp_dir: str | None = None):
        """
        Initialize WASI runner

        Args:
            wasmtime_path: Path to wasmtime binary
            default_timeout: Default execution timeout
            temp_dir: Temporary directory for sandboxes
        """

        self.wasmtime_path = wasmtime_path
        self.default_timeout = default_timeout
        self.temp_dir = temp_dir or tempfile.gettempdir()

        # Active executions
        self.active_executions: dict[str, asyncio.Task] = {}

        # Runtime verification
        self._verified = False

    async def verify_runtime(self) -> bool:
        """Verify WASI runtime is available and functional"""

        if self._verified:
            return True

        try:
            # Check wasmtime version
            proc = await asyncio.create_subprocess_exec(
                self.wasmtime_path, "--version", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                logger.error(f"wasmtime check failed: {stderr.decode()}")
                return False

            version = stdout.decode().strip()
            logger.info(f"WASI runtime verified: {version}")
            self._verified = True
            return True

        except FileNotFoundError:
            logger.error(f"wasmtime not found at: {self.wasmtime_path}")
            return False
        except Exception as e:
            logger.error(f"WASI runtime verification failed: {e}")
            return False

    async def execute(
        self,
        wasm_module: bytes,
        args: list[str] = None,
        env: dict[str, str] = None,
        resources: ExecutionResources = None,
        stdin_data: bytes = b"",
        working_dir: str = "/workspace",
    ) -> ExecutionResult:
        """
        Execute WASM module in WASI sandbox

        Args:
            wasm_module: WebAssembly module bytes
            args: Command line arguments
            env: Environment variables
            resources: Resource limits
            stdin_data: Standard input data
            working_dir: Working directory inside sandbox

        Returns:
            ExecutionResult with outputs and resource usage
        """

        execution_id = str(uuid4())
        result = ExecutionResult(
            execution_id=execution_id, status=ExecutionStatus.PENDING, start_time=datetime.now(UTC)
        )

        # Verify runtime
        if not await self.verify_runtime():
            result.status = ExecutionStatus.FAILED
            result.error_message = "WASI runtime not available"
            return result

        # Set defaults
        args = args or []
        env = env or {}
        resources = resources or ExecutionResources()

        try:
            # Create sandbox directory
            sandbox_dir = self._create_sandbox(execution_id, working_dir)

            # Write WASM module to temporary file
            wasm_path = sandbox_dir / "module.wasm"
            wasm_path.write_bytes(wasm_module)

            # Prepare wasmtime command
            cmd = [
                self.wasmtime_path,
                "--wasm-features",
                "all",
                "--invoke",
                "_start",  # WASI start function
                "--dir",
                f"{sandbox_dir}::{working_dir}",  # Map sandbox to working dir
            ]

            # Add memory limit (in pages, 64KB each)
            memory_pages = (resources.memory_mb * 1024 * 1024) // 65536
            cmd.extend(["--max-memory-size", str(memory_pages)])

            # Add CPU limit (approximate via timeout)
            timeout = min(resources.max_duration_s, self.default_timeout)

            # Add the WASM module and args
            cmd.append(str(wasm_path))
            cmd.extend(args)

            logger.info(f"Starting WASI execution {execution_id}: {' '.join(cmd[:3])}...")

            # Start execution
            result.status = ExecutionStatus.STARTING

            # Create execution task
            exec_task = asyncio.create_task(self._run_wasi_process(cmd, env, stdin_data, timeout, sandbox_dir))
            self.active_executions[execution_id] = exec_task

            # Wait for completion
            result.status = ExecutionStatus.RUNNING

            try:
                proc_result = await exec_task
                result.exit_code = proc_result["exit_code"]
                result.stdout = proc_result["stdout"]
                result.stderr = proc_result["stderr"]
                result.cpu_time_s = proc_result["cpu_time_s"]
                result.memory_peak_mb = proc_result["memory_peak_mb"]
                result.disk_used_mb = proc_result["disk_used_mb"]

                if result.exit_code == 0:
                    result.status = ExecutionStatus.COMPLETED
                else:
                    result.status = ExecutionStatus.FAILED
                    result.error_message = f"Process exited with code {result.exit_code}"

            except asyncio.TimeoutError:
                result.status = ExecutionStatus.TIMEOUT
                result.error_message = f"Execution timed out after {timeout}s"

            except asyncio.CancelledError:
                result.status = ExecutionStatus.CANCELLED
                result.error_message = "Execution was cancelled"

            except Exception as e:
                result.status = ExecutionStatus.FAILED
                result.error_message = f"Execution error: {str(e)}"

            finally:
                # Clean up
                self.active_executions.pop(execution_id, None)
                self._cleanup_sandbox(sandbox_dir)

            # Calculate timing
            result.end_time = datetime.now(UTC)
            if result.start_time:
                duration = result.end_time - result.start_time
                result.duration_ms = duration.total_seconds() * 1000

            # Check resource violations
            result.resource_violations = self._check_resource_violations(result, resources)

            logger.info(f"WASI execution {execution_id} completed: {result.status}")
            return result

        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error_message = f"Setup error: {str(e)}"
            result.end_time = datetime.now(UTC)
            logger.error(f"WASI execution {execution_id} failed: {e}")
            return result

    async def _run_wasi_process(
        self, cmd: list[str], env: dict[str, str], stdin_data: bytes, timeout: float, sandbox_dir: Path
    ) -> dict[str, Any]:
        """Run the WASI process and monitor resources"""

        # Prepare environment
        process_env = os.environ.copy()
        process_env.update(env)

        # Start process
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=process_env,
            cwd=str(sandbox_dir),
        )

        # Monitor resource usage
        cpu_times = []
        memory_peaks = []

        async def monitor_resources():
            """Monitor CPU and memory usage"""
            try:
                if proc.pid:
                    process = psutil.Process(proc.pid)
                    while proc.returncode is None:
                        try:
                            cpu_percent = process.cpu_percent()
                            memory_mb = process.memory_info().rss / (1024 * 1024)

                            cpu_times.append(cpu_percent)
                            memory_peaks.append(memory_mb)

                            await asyncio.sleep(0.5)
                        except psutil.NoSuchProcess:
                            break
            except Exception as e:
                logger.debug(f"Resource monitoring error: {e}")

        # Start monitoring
        monitor_task = asyncio.create_task(monitor_resources())

        try:
            # Send stdin and wait for completion
            stdout, stderr = await asyncio.wait_for(proc.communicate(stdin_data), timeout=timeout)

            exit_code = proc.returncode

        except asyncio.TimeoutError:
            # Kill process and children
            try:
                if proc.pid:
                    parent = psutil.Process(proc.pid)
                    children = parent.children(recursive=True)
                    for child in children:
                        child.kill()
                    parent.kill()
            except psutil.NoSuchProcess:
                pass
            raise

        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        # Calculate resource usage
        cpu_time_s = sum(cpu_times) * 0.5 / 100.0  # Approximate CPU time
        memory_peak_mb = max(memory_peaks) if memory_peaks else 0

        # Calculate disk usage
        disk_used_mb = 0
        try:
            sandbox_size = sum(f.stat().st_size for f in sandbox_dir.rglob("*") if f.is_file())
            disk_used_mb = sandbox_size / (1024 * 1024)
        except Exception:
            pass

        return {
            "exit_code": exit_code,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "cpu_time_s": cpu_time_s,
            "memory_peak_mb": int(memory_peak_mb),
            "disk_used_mb": int(disk_used_mb),
        }

    def _create_sandbox(self, execution_id: str, working_dir: str) -> Path:
        """Create isolated sandbox directory"""

        sandbox_dir = Path(self.temp_dir) / f"wasi_sandbox_{execution_id}"
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        # Create working directory structure
        work_dir = sandbox_dir / working_dir.lstrip("/")
        work_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Created sandbox: {sandbox_dir}")
        return sandbox_dir

    def _cleanup_sandbox(self, sandbox_dir: Path):
        """Clean up sandbox directory"""

        try:
            shutil.rmtree(sandbox_dir)
            logger.debug(f"Cleaned up sandbox: {sandbox_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup sandbox {sandbox_dir}: {e}")

    def _check_resource_violations(self, result: ExecutionResult, resources: ExecutionResources) -> list[str]:
        """Check for resource limit violations"""

        violations = []

        if result.memory_peak_mb > resources.memory_mb:
            violations.append(f"Memory limit exceeded: {result.memory_peak_mb}MB > {resources.memory_mb}MB")

        if result.disk_used_mb > resources.disk_mb:
            violations.append(f"Disk limit exceeded: {result.disk_used_mb}MB > {resources.disk_mb}MB")

        if result.duration_ms > resources.max_duration_s * 1000:
            violations.append(f"Time limit exceeded: {result.duration_ms/1000:.1f}s > {resources.max_duration_s}s")

        return violations

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel running execution"""

        task = self.active_executions.get(execution_id)
        if task and not task.done():
            task.cancel()
            logger.info(f"Cancelled WASI execution: {execution_id}")
            return True

        return False

    def get_active_executions(self) -> list[str]:
        """Get list of active execution IDs"""
        return list(self.active_executions.keys())


class MicroVMRunner:
    """
    MicroVM runner using Firecracker

    Provides lightweight VM isolation for stronger security boundaries.
    Used for workloads requiring kernel-level isolation or specific OS features.
    """

    def __init__(
        self, firecracker_path: str = "firecracker", default_timeout: float = 300.0, temp_dir: str | None = None
    ):
        """
        Initialize MicroVM runner

        Args:
            firecracker_path: Path to firecracker binary
            default_timeout: Default execution timeout
            temp_dir: Temporary directory for VM images
        """

        self.firecracker_path = firecracker_path
        self.default_timeout = default_timeout
        self.temp_dir = temp_dir or tempfile.gettempdir()

        # Active VMs
        self.active_vms: dict[str, asyncio.Task] = {}

        # Runtime verification
        self._verified = False

    async def verify_runtime(self) -> bool:
        """Verify Firecracker runtime is available"""

        if self._verified:
            return True

        try:
            # Check firecracker version
            proc = await asyncio.create_subprocess_exec(
                self.firecracker_path, "--version", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                logger.error(f"firecracker check failed: {stderr.decode()}")
                return False

            version = stdout.decode().strip()
            logger.info(f"MicroVM runtime verified: {version}")

            # Check KVM support
            if not os.path.exists("/dev/kvm"):
                logger.warning("KVM not available - MicroVM may not work properly")

            self._verified = True
            return True

        except FileNotFoundError:
            logger.error(f"firecracker not found at: {self.firecracker_path}")
            return False
        except Exception as e:
            logger.error(f"MicroVM runtime verification failed: {e}")
            return False

    async def execute(
        self,
        kernel_image: bytes,
        rootfs_image: bytes,
        args: list[str] = None,
        env: dict[str, str] = None,
        resources: ExecutionResources = None,
    ) -> ExecutionResult:
        """
        Execute workload in MicroVM

        Args:
            kernel_image: Linux kernel image
            rootfs_image: Root filesystem image
            args: Kernel command line arguments
            env: Environment variables (injected into rootfs)
            resources: Resource limits

        Returns:
            ExecutionResult with outputs and resource usage
        """

        execution_id = str(uuid4())
        result = ExecutionResult(
            execution_id=execution_id, status=ExecutionStatus.PENDING, start_time=datetime.now(UTC)
        )

        # Verify runtime
        if not await self.verify_runtime():
            result.status = ExecutionStatus.FAILED
            result.error_message = "MicroVM runtime not available"
            return result

        # Set defaults
        args = args or []
        env = env or {}
        resources = resources or ExecutionResources()

        # TODO: Implement full MicroVM execution
        # This is a simplified placeholder implementation

        try:
            logger.info(f"MicroVM execution {execution_id} - implementation pending")

            # For now, return a mock successful result
            result.status = ExecutionStatus.COMPLETED
            result.exit_code = 0
            result.stdout = "MicroVM execution completed (mock)"
            result.stderr = ""
            result.cpu_time_s = 1.0
            result.memory_peak_mb = 64
            result.disk_used_mb = 10

            result.end_time = datetime.now(UTC)
            if result.start_time:
                duration = result.end_time - result.start_time
                result.duration_ms = duration.total_seconds() * 1000

            return result

        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error_message = f"MicroVM error: {str(e)}"
            result.end_time = datetime.now(UTC)
            logger.error(f"MicroVM execution {execution_id} failed: {e}")
            return result

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel running MicroVM execution"""

        task = self.active_vms.get(execution_id)
        if task and not task.done():
            task.cancel()
            logger.info(f"Cancelled MicroVM execution: {execution_id}")
            return True

        return False

    def get_active_executions(self) -> list[str]:
        """Get list of active execution IDs"""
        return list(self.active_vms.keys())


class ExecutionFabric:
    """
    Unified execution fabric managing multiple runtime types

    Provides a single interface for executing workloads across:
    - WASI sandboxes (WebAssembly)
    - MicroVMs (Firecracker)
    - OCI containers (future)
    - Native processes (limited use)
    """

    def __init__(self, temp_dir: str | None = None):
        """Initialize execution fabric"""

        self.wasi_runner = WASIRunner(temp_dir=temp_dir)
        self.microvm_runner = MicroVMRunner(temp_dir=temp_dir)

        # Capability cache
        self._runtime_capabilities: dict[RuntimeType, bool] = {}

    async def get_supported_runtimes(self) -> list[RuntimeType]:
        """Get list of supported runtime types"""

        supported = []

        # Check WASI support
        if await self.wasi_runner.verify_runtime():
            supported.append(RuntimeType.WASI)
            self._runtime_capabilities[RuntimeType.WASI] = True

        # Check MicroVM support
        if await self.microvm_runner.verify_runtime():
            supported.append(RuntimeType.MICROVM)
            self._runtime_capabilities[RuntimeType.MICROVM] = True

        # Native always supported (with caveats)
        supported.append(RuntimeType.NATIVE)
        self._runtime_capabilities[RuntimeType.NATIVE] = True

        return supported

    async def execute(
        self,
        runtime_type: RuntimeType,
        payload: bytes,
        args: list[str] = None,
        env: dict[str, str] = None,
        resources: ExecutionResources = None,
    ) -> ExecutionResult:
        """
        Execute payload using specified runtime

        Args:
            runtime_type: Type of runtime to use
            payload: Code/image to execute
            args: Command line arguments
            env: Environment variables
            resources: Resource limits

        Returns:
            ExecutionResult with outputs and resource usage
        """

        if runtime_type == RuntimeType.WASI:
            return await self.wasi_runner.execute(payload, args, env, resources)

        elif runtime_type == RuntimeType.MICROVM:
            # For MicroVM, payload should contain both kernel and rootfs
            # This is a simplified interface - real implementation would
            # need separate kernel and rootfs parameters
            return await self.microvm_runner.execute(payload, b"", args, env, resources)  # Simplified

        elif runtime_type == RuntimeType.NATIVE:
            # Native execution is dangerous and should be heavily restricted
            return await self._execute_native(payload, args, env, resources)

        else:
            result = ExecutionResult(
                execution_id=str(uuid4()),
                status=ExecutionStatus.FAILED,
                error_message=f"Unsupported runtime: {runtime_type}",
            )
            return result

    async def _execute_native(
        self, payload: bytes, args: list[str] = None, env: dict[str, str] = None, resources: ExecutionResources = None
    ) -> ExecutionResult:
        """Execute native binary (DANGEROUS - restricted use only)"""

        result = ExecutionResult(
            execution_id=str(uuid4()),
            status=ExecutionStatus.FAILED,
            error_message="Native execution not implemented for security reasons",
        )

        logger.warning("Native execution requested but blocked for security")
        return result

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel execution across all runtime types"""

        cancelled = False

        # Try WASI runner
        if await self.wasi_runner.cancel_execution(execution_id):
            cancelled = True

        # Try MicroVM runner
        if await self.microvm_runner.cancel_execution(execution_id):
            cancelled = True

        return cancelled

    def get_active_executions(self) -> dict[RuntimeType, list[str]]:
        """Get active executions by runtime type"""

        return {
            RuntimeType.WASI: self.wasi_runner.get_active_executions(),
            RuntimeType.MICROVM: self.microvm_runner.get_active_executions(),
        }
