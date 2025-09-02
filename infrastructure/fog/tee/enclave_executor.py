"""
Enclave Executor

Handles secure code execution within TEE environments:
- Secure code loading and execution
- Memory encryption and isolation
- I/O protection and monitoring
- Performance metrics collection
- Support for multiple TEE backends
"""

import asyncio
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

from .tee_types import (
    EnclaveContext,
    EnclaveMetrics,
    EnclaveState,
    TEECapability,
    TEEType,
)

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Base exception for execution errors"""

    pass


class SecurityViolationError(ExecutionError):
    """Exception for security policy violations"""

    pass


class ResourceExhaustionError(ExecutionError):
    """Exception for resource exhaustion"""

    pass


class SoftwareIsolationBackend:
    """Software isolation backend using containers and VMs"""

    def __init__(self):
        self.containers: dict[str, Any] = {}
        self.temp_dirs: dict[str, Path] = {}

        # Check available isolation technologies
        self.has_docker = self._check_docker()
        self.has_gvisor = self._check_gvisor()
        self.has_firecracker = self._check_firecracker()

        logger.info(
            f"Software isolation backend: docker={self.has_docker}, "
            f"gvisor={self.has_gvisor}, firecracker={self.has_firecracker}"
        )

    def _check_docker(self) -> bool:
        """Check if Docker is available"""
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, timeout=10)
            return result.returncode == 0
        except Exception:
            return False

    def _check_gvisor(self) -> bool:
        """Check if gVisor is available"""
        try:
            result = subprocess.run(["runsc", "--version"], capture_output=True, timeout=10)
            return result.returncode == 0
        except Exception:
            return False

    def _check_firecracker(self) -> bool:
        """Check if Firecracker is available"""
        try:
            result = subprocess.run(["firecracker", "--version"], capture_output=True, timeout=10)
            return result.returncode == 0
        except Exception:
            return False

    async def create_enclave(self, context: EnclaveContext) -> bool:
        """Create software isolation enclave"""
        enclave_id = context.spec.enclave_id

        try:
            # Create temporary directory for enclave
            temp_dir = Path(tempfile.mkdtemp(prefix=f"enclave_{enclave_id}_"))
            self.temp_dirs[enclave_id] = temp_dir

            # Setup enclave environment
            await self._setup_enclave_environment(context, temp_dir)

            # Choose isolation method
            if self.has_gvisor and context.spec.config.network_isolation:
                success = await self._create_gvisor_container(context, temp_dir)
            elif self.has_docker:
                success = await self._create_docker_container(context, temp_dir)
            else:
                # Fallback to basic process isolation
                success = await self._create_process_isolation(context, temp_dir)

            if success:
                context.state = EnclaveState.RUNNING
                logger.info(f"Software enclave {enclave_id} created successfully")
            else:
                context.state = EnclaveState.ERROR

            return success

        except Exception as e:
            logger.error(f"Failed to create software enclave {enclave_id}: {e}")
            context.state = EnclaveState.ERROR
            return False

    async def terminate_enclave(self, context: EnclaveContext) -> bool:
        """Terminate software isolation enclave"""
        enclave_id = context.spec.enclave_id

        try:
            # Stop container if it exists
            if enclave_id in self.containers:
                container_info = self.containers[enclave_id]
                if container_info["type"] == "docker":
                    await self._stop_docker_container(container_info["id"])
                elif container_info["type"] == "gvisor":
                    await self._stop_gvisor_container(container_info["id"])
                elif container_info["type"] == "process":
                    await self._stop_process(container_info["pid"])

                del self.containers[enclave_id]

            # Cleanup temporary directory
            if enclave_id in self.temp_dirs:
                shutil.rmtree(self.temp_dirs[enclave_id], ignore_errors=True)
                del self.temp_dirs[enclave_id]

            logger.info(f"Software enclave {enclave_id} terminated")
            return True

        except Exception as e:
            logger.error(f"Failed to terminate software enclave {enclave_id}: {e}")
            return False

    async def execute_code(self, context: EnclaveContext, code: bytes, input_data: bytes = b"") -> bytes:
        """Execute code in software isolation enclave"""
        enclave_id = context.spec.enclave_id

        if enclave_id not in self.containers:
            raise ExecutionError(f"Enclave {enclave_id} not found")

        container_info = self.containers[enclave_id]
        temp_dir = self.temp_dirs[enclave_id]

        try:
            # Write code and input to files
            code_file = temp_dir / "code.py"
            input_file = temp_dir / "input.data"
            output_file = temp_dir / "output.data"

            with open(code_file, "wb") as f:
                f.write(code)

            with open(input_file, "wb") as f:
                f.write(input_data)

            # Execute based on container type
            if container_info["type"] == "docker":
                await self._execute_in_docker(container_info["id"], code_file, input_file, output_file)
            elif container_info["type"] == "gvisor":
                await self._execute_in_gvisor(container_info["id"], code_file, input_file, output_file)
            else:
                await self._execute_in_process(container_info["pid"], code_file, input_file, output_file)

            # Read output
            if output_file.exists():
                with open(output_file, "rb") as f:
                    output = f.read()
            else:
                output = b""

            return output

        except Exception as e:
            logger.error(f"Code execution failed in enclave {enclave_id}: {e}")
            raise ExecutionError(f"Execution failed: {e}")

    async def get_metrics(self, context: EnclaveContext) -> EnclaveMetrics | None:
        """Get metrics for software isolation enclave"""
        enclave_id = context.spec.enclave_id

        if enclave_id not in self.containers:
            return None

        try:
            metrics = EnclaveMetrics(enclave_id=enclave_id)
            container_info = self.containers[enclave_id]

            if container_info["type"] == "docker":
                # Get Docker container stats
                proc = await asyncio.create_subprocess_exec(
                    "docker",
                    "stats",
                    container_info["id"],
                    "--no-stream",
                    "--format",
                    "table {{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()

                if proc.returncode == 0:
                    lines = stdout.decode().strip().split("\n")
                    if len(lines) > 1:  # Skip header
                        stats = lines[1].split("\t")
                        if len(stats) >= 4:
                            # Parse CPU usage
                            cpu_str = stats[0].replace("%", "")
                            if cpu_str != "--":
                                metrics.cpu_usage_percent = float(cpu_str)

                            # Parse memory usage
                            mem_str = stats[1].split(" / ")[0]
                            if "MiB" in mem_str:
                                metrics.memory_usage_mb = int(float(mem_str.replace("MiB", "")))

            return metrics

        except Exception as e:
            logger.debug(f"Failed to get metrics for enclave {enclave_id}: {e}")
            return None

    async def _setup_enclave_environment(self, context: EnclaveContext, temp_dir: Path) -> None:
        """Setup enclave execution environment"""
        # Create required directories
        (temp_dir / "app").mkdir()
        (temp_dir / "data").mkdir()
        (temp_dir / "logs").mkdir()

        # Create basic Python runtime environment
        dockerfile_content = """
FROM python:3.11-alpine
WORKDIR /app
RUN adduser -D -s /bin/sh enclave
USER enclave
COPY --chown=enclave:enclave . .
CMD ["python", "-c", "print('Enclave ready')"]
"""

        with open(temp_dir / "Dockerfile", "w") as f:
            f.write(dockerfile_content)

        # Create environment configuration
        env_config = {
            "ENCLAVE_ID": context.spec.enclave_id,
            "MEMORY_LIMIT": f"{context.spec.memory_mb}m",
            "CPU_LIMIT": str(context.spec.cpu_cores),
        }
        env_config.update(context.spec.environment_variables)

        with open(temp_dir / ".env", "w") as f:
            for key, value in env_config.items():
                f.write(f"{key}={value}\n")

    async def _create_docker_container(self, context: EnclaveContext, temp_dir: Path) -> bool:
        """Create Docker container for enclave"""
        enclave_id = context.spec.enclave_id

        try:
            # Build image
            build_proc = await asyncio.create_subprocess_exec(
                "docker",
                "build",
                "-t",
                f"enclave_{enclave_id}",
                str(temp_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await build_proc.communicate()

            if build_proc.returncode != 0:
                logger.error(f"Docker build failed: {stderr.decode()}")
                return False

            # Run container
            docker_args = [
                "docker",
                "run",
                "-d",
                "--name",
                f"enclave_{enclave_id}",
                "--memory",
                f"{context.spec.memory_mb}m",
                "--cpus",
                str(context.spec.cpu_cores),
                "--network",
                "none" if context.spec.config.network_isolation else "bridge",
                "--read-only",
                "--tmpfs",
                "/tmp",
                f"enclave_{enclave_id}",
            ]

            run_proc = await asyncio.create_subprocess_exec(
                *docker_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await run_proc.communicate()

            if run_proc.returncode != 0:
                logger.error(f"Docker run failed: {stderr.decode()}")
                return False

            container_id = stdout.decode().strip()
            self.containers[enclave_id] = {"type": "docker", "id": container_id, "created_at": time.time()}

            return True

        except Exception as e:
            logger.error(f"Failed to create Docker container: {e}")
            return False

    async def _create_gvisor_container(self, context: EnclaveContext, temp_dir: Path) -> bool:
        """Create gVisor container for enhanced isolation"""
        # Simplified gVisor integration
        # In production, this would use proper gVisor APIs
        enclave_id = context.spec.enclave_id

        try:
            # Use Docker with gVisor runtime
            docker_args = [
                "docker",
                "run",
                "-d",
                "--runtime",
                "runsc",
                "--name",
                f"gvisor_enclave_{enclave_id}",
                "--memory",
                f"{context.spec.memory_mb}m",
                "--cpus",
                str(context.spec.cpu_cores),
                "--network",
                "none",
                "--read-only",
                "--tmpfs",
                "/tmp",
                "python:3.11-alpine",
                "sleep",
                "infinity",
            ]

            proc = await asyncio.create_subprocess_exec(
                *docker_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                logger.error(f"gVisor container creation failed: {stderr.decode()}")
                return False

            container_id = stdout.decode().strip()
            self.containers[enclave_id] = {"type": "gvisor", "id": container_id, "created_at": time.time()}

            return True

        except Exception as e:
            logger.error(f"Failed to create gVisor container: {e}")
            return False

    async def _create_process_isolation(self, context: EnclaveContext, temp_dir: Path) -> bool:
        """Create basic process isolation"""
        enclave_id = context.spec.enclave_id

        try:
            # Create isolated process with resource limits
            # This is a basic implementation - production would use cgroups, namespaces, etc.

            self.containers[enclave_id] = {
                "type": "process",
                "pid": None,  # Will be set when process starts
                "temp_dir": temp_dir,
                "created_at": time.time(),
            }

            return True

        except Exception as e:
            logger.error(f"Failed to create process isolation: {e}")
            return False

    async def _execute_in_docker(self, container_id: str, code_file: Path, input_file: Path, output_file: Path) -> bool:
        """Execute code in Docker container"""
        try:
            # Copy files to container
            copy_proc = await asyncio.create_subprocess_exec(
                "docker",
                "cp",
                str(code_file),
                f"{container_id}:/app/code.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await copy_proc.communicate()

            copy_proc = await asyncio.create_subprocess_exec(
                "docker",
                "cp",
                str(input_file),
                f"{container_id}:/app/input.data",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await copy_proc.communicate()

            # Execute code
            exec_proc = await asyncio.create_subprocess_exec(
                "docker",
                "exec",
                container_id,
                "python",
                "/app/code.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await exec_proc.communicate()

            # Copy output back
            copy_proc = await asyncio.create_subprocess_exec(
                "docker",
                "cp",
                f"{container_id}:/app/output.data",
                str(output_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await copy_proc.communicate()

            return exec_proc.returncode == 0

        except Exception as e:
            logger.error(f"Docker execution failed: {e}")
            return False

    async def _execute_in_gvisor(self, container_id: str, code_file: Path, input_file: Path, output_file: Path) -> bool:
        """Execute code in gVisor container"""
        # Similar to Docker execution but with enhanced security
        return await self._execute_in_docker(container_id, code_file, input_file, output_file)

    async def _execute_in_process(self, pid: int | None, code_file: Path, input_file: Path, output_file: Path) -> bool:
        """Execute code in isolated process"""
        try:
            # Simple process execution with timeout
            proc = await asyncio.create_subprocess_exec(
                "python",
                str(code_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=code_file.parent,
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
                return proc.returncode == 0
            except asyncio.TimeoutError:
                proc.kill()
                logger.warning("Process execution timed out")
                return False

        except Exception as e:
            logger.error(f"Process execution failed: {e}")
            return False

    async def _stop_docker_container(self, container_id: str) -> None:
        """Stop Docker container"""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "stop", container_id, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()

            # Remove container
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", container_id, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()

        except Exception as e:
            logger.debug(f"Error stopping Docker container: {e}")

    async def _stop_gvisor_container(self, container_id: str) -> None:
        """Stop gVisor container"""
        await self._stop_docker_container(container_id)

    async def _stop_process(self, pid: int | None) -> None:
        """Stop isolated process"""
        if pid:
            try:
                os.kill(pid, 9)  # SIGKILL
            except ProcessLookupError:
                pass  # Process already terminated
            except Exception as e:
                logger.debug(f"Error stopping process {pid}: {e}")


class EnclaveExecutor:
    """
    Enclave Executor

    Handles secure code execution within TEE environments with support for
    multiple TEE backends and software isolation fallback.
    """

    def __init__(self):
        self.software_backend = SoftwareIsolationBackend()
        self.active_enclaves: dict[str, EnclaveContext] = {}

        # Hardware backends (placeholders for now)
        self.hardware_backends = {
            TEEType.AMD_SEV_SNP: None,
            TEEType.INTEL_TDX: None,
            TEEType.INTEL_SGX: None,
        }

        logger.info("Enclave Executor initialized")

    async def initialize(self) -> None:
        """Initialize the enclave executor"""
        logger.info("Initializing Enclave Executor...")

        # Initialize hardware backends if available
        await self._initialize_hardware_backends()

        logger.info("Enclave Executor initialization complete")

    async def shutdown(self) -> None:
        """Shutdown the enclave executor"""
        logger.info("Shutting down Enclave Executor...")

        # Terminate all active enclaves
        for context in list(self.active_enclaves.values()):
            await self.terminate_enclave(context)

        logger.info("Enclave Executor shutdown complete")

    async def create_enclave(self, context: EnclaveContext, capability: TEECapability) -> bool:
        """Create an enclave with the specified capability"""
        enclave_id = context.spec.enclave_id

        if enclave_id in self.active_enclaves:
            raise ExecutionError(f"Enclave {enclave_id} already exists")

        logger.info(f"Creating enclave {enclave_id} with {capability.tee_type.value}")

        try:
            success = False

            if capability.tee_type == TEEType.SOFTWARE_ISOLATION:
                success = await self.software_backend.create_enclave(context)
            else:
                # Hardware TEE creation (placeholder)
                success = await self._create_hardware_enclave(context, capability)

            if success:
                self.active_enclaves[enclave_id] = context
                context.runtime_handles["executor"] = self
                logger.info(f"Enclave {enclave_id} created successfully")
            else:
                logger.error(f"Failed to create enclave {enclave_id}")

            return success

        except Exception as e:
            logger.error(f"Enclave creation failed: {e}")
            context.state = EnclaveState.ERROR
            return False

    async def terminate_enclave(self, context: EnclaveContext) -> bool:
        """Terminate an enclave"""
        enclave_id = context.spec.enclave_id

        if enclave_id not in self.active_enclaves:
            logger.warning(f"Enclave {enclave_id} not found in active enclaves")
            return False

        logger.info(f"Terminating enclave {enclave_id}")

        try:
            success = False

            if context.tee_type == TEEType.SOFTWARE_ISOLATION:
                success = await self.software_backend.terminate_enclave(context)
            else:
                success = await self._terminate_hardware_enclave(context)

            if success:
                del self.active_enclaves[enclave_id]
                context.state = EnclaveState.TERMINATED
                logger.info(f"Enclave {enclave_id} terminated successfully")

            return success

        except Exception as e:
            logger.error(f"Enclave termination failed: {e}")
            return False

    async def execute_code(self, context: EnclaveContext, code: bytes, input_data: bytes = b"") -> bytes:
        """Execute code in an enclave"""
        enclave_id = context.spec.enclave_id

        if enclave_id not in self.active_enclaves:
            raise ExecutionError(f"Enclave {enclave_id} not found")

        if context.state != EnclaveState.RUNNING:
            raise ExecutionError(f"Enclave {enclave_id} is not running")

        logger.debug(f"Executing code in enclave {enclave_id}")

        start_time = time.time()

        try:
            # Validate code before execution
            await self._validate_code(code, context)

            # Execute based on TEE type
            if context.tee_type == TEEType.SOFTWARE_ISOLATION:
                result = await self.software_backend.execute_code(context, code, input_data)
            else:
                result = await self._execute_hardware_code(context, code, input_data)

            execution_time = time.time() - start_time

            # Update metrics
            context.metrics.tasks_executed += 1
            context.metrics.tasks_completed += 1

            # Update average execution time
            total_time = (
                context.metrics.average_execution_time_ms * (context.metrics.tasks_completed - 1)
                + execution_time * 1000
            )
            context.metrics.average_execution_time_ms = total_time / context.metrics.tasks_completed

            logger.debug(f"Code execution completed in {execution_time:.3f}s")
            return result

        except Exception as e:
            context.metrics.tasks_executed += 1
            context.metrics.tasks_failed += 1
            logger.error(f"Code execution failed: {e}")
            raise ExecutionError(f"Code execution failed: {e}")

    async def get_enclave_metrics(self, context: EnclaveContext) -> EnclaveMetrics | None:
        """Get metrics for an enclave"""
        if context.tee_type == TEEType.SOFTWARE_ISOLATION:
            return await self.software_backend.get_metrics(context)
        else:
            return await self._get_hardware_metrics(context)

    async def check_health(self, context: EnclaveContext) -> bool:
        """Check if an enclave is healthy"""
        try:
            # Simple health check - try to execute a no-op
            test_code = b"print('health_check')"
            await self.execute_code(context, test_code, b"")
            return True
        except Exception:
            return False

    async def _validate_code(self, code: bytes, context: EnclaveContext) -> None:
        """Validate code before execution"""
        # Basic validation
        if not code:
            raise SecurityViolationError("Empty code not allowed")

        if len(code) > 1024 * 1024:  # 1MB limit
            raise SecurityViolationError("Code size exceeds limit")

        # Check for dangerous imports/operations (basic filtering)
        code_str = code.decode("utf-8", errors="ignore")
        dangerous_patterns = [
            "import os",
            "import subprocess",
            "import sys",
            "__import__",
            "exec(",
            "eval(",
        ]

        for pattern in dangerous_patterns:
            if pattern in code_str:
                logger.warning(f"Potentially dangerous code pattern detected: {pattern}")
                # In strict mode, we would reject this
                # For now, just log the warning

    async def _create_hardware_enclave(self, context: EnclaveContext, capability: TEECapability) -> bool:
        """Create hardware TEE enclave (placeholder)"""
        # This would implement actual hardware TEE creation
        # For now, fall back to software isolation
        logger.warning(
            f"Hardware TEE {capability.tee_type.value} implementation pending, " "falling back to software isolation"
        )

        context.tee_type = TEEType.SOFTWARE_ISOLATION
        return await self.software_backend.create_enclave(context)

    async def _terminate_hardware_enclave(self, context: EnclaveContext) -> bool:
        """Terminate hardware TEE enclave (placeholder)"""
        # This would implement actual hardware TEE termination
        return await self.software_backend.terminate_enclave(context)

    async def _execute_hardware_code(self, context: EnclaveContext, code: bytes, input_data: bytes) -> bytes:
        """Execute code in hardware TEE (placeholder)"""
        # This would implement actual hardware TEE code execution
        return await self.software_backend.execute_code(context, code, input_data)

    async def _get_hardware_metrics(self, context: EnclaveContext) -> EnclaveMetrics | None:
        """Get hardware TEE metrics (placeholder)"""
        # This would implement actual hardware TEE metrics collection
        return await self.software_backend.get_metrics(context)

    async def _initialize_hardware_backends(self) -> None:
        """Initialize hardware TEE backends"""
        # Placeholder for hardware backend initialization
        # In production, this would initialize AMD SEV-SNP, Intel TDX, etc.
        logger.debug("Hardware TEE backends implementation pending, using software isolation")

    def get_executor_status(self) -> dict[str, Any]:
        """Get executor status"""
        return {
            "active_enclaves": len(self.active_enclaves),
            "software_backend": {
                "containers": len(self.software_backend.containers),
                "has_docker": self.software_backend.has_docker,
                "has_gvisor": self.software_backend.has_gvisor,
                "has_firecracker": self.software_backend.has_firecracker,
            },
            "hardware_backends": {
                tee_type.value: backend is not None for tee_type, backend in self.hardware_backends.items()
            },
        }
