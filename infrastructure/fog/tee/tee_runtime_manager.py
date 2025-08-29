"""
TEE Runtime Manager

Manages Trusted Execution Environment runtime with support for:
- AMD SEV-SNP and Intel TDX hardware enclaves
- Software isolation fallback (gVisor/Firecracker)
- Attestation service integration
- Enclave lifecycle management
- Performance monitoring and metrics
"""

import asyncio
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import logging
import os
from typing import Any

from .attestation_service import AttestationService
from .enclave_executor import EnclaveExecutor
from .tee_types import (
    EnclaveContext,
    EnclaveSpec,
    EnclaveState,
    TEECapability,
    TEEConfiguration,
    TEEType,
)

logger = logging.getLogger(__name__)


class TEERuntimeError(Exception):
    """Base exception for TEE runtime errors"""

    pass


class TEECapabilityDetector:
    """Detects available TEE capabilities on the system"""

    @staticmethod
    async def detect_capabilities() -> list[TEECapability]:
        """Detect available TEE technologies"""
        capabilities = []

        # Check for AMD SEV-SNP
        sev_capability = await TEECapabilityDetector._check_amd_sev_snp()
        if sev_capability:
            capabilities.append(sev_capability)

        # Check for Intel TDX
        tdx_capability = await TEECapabilityDetector._check_intel_tdx()
        if tdx_capability:
            capabilities.append(tdx_capability)

        # Check for Intel SGX
        sgx_capability = await TEECapabilityDetector._check_intel_sgx()
        if sgx_capability:
            capabilities.append(sgx_capability)

        # Software isolation is always available as fallback
        software_capability = TEECapability(
            tee_type=TEEType.SOFTWARE_ISOLATION,
            available=True,
            version="1.0",
            max_memory_mb=8192,
            max_enclaves=64,
            supports_migration=True,
            supports_remote_attestation=True,
            supports_sealed_storage=False,
            secure_boot=False,
            memory_encryption=False,
            io_protection=True,
            debug_disabled=False,
        )
        capabilities.append(software_capability)

        return capabilities

    @staticmethod
    async def _check_amd_sev_snp() -> TEECapability | None:
        """Check for AMD SEV-SNP support"""
        try:
            # Check for SEV device node
            if not os.path.exists("/dev/sev"):
                return None

            # Check kernel support
            proc = await asyncio.create_subprocess_exec(
                "dmesg", "|", "grep", "-i", "sev", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            if b"SEV-SNP" in stdout:
                return TEECapability(
                    tee_type=TEEType.AMD_SEV_SNP,
                    available=True,
                    version="1.0",
                    max_memory_mb=16384,
                    max_enclaves=32,
                    supports_migration=True,
                    supports_remote_attestation=True,
                    supports_sealed_storage=True,
                    secure_boot=True,
                    memory_encryption=True,
                    io_protection=True,
                    debug_disabled=True,
                )

        except Exception as e:
            logger.debug(f"Error checking AMD SEV-SNP: {e}")

        return None

    @staticmethod
    async def _check_intel_tdx() -> TEECapability | None:
        """Check for Intel TDX support"""
        try:
            # Check for TDX support in CPUID
            proc = await asyncio.create_subprocess_exec(
                "cpuid", "-1", "-l", "0x21", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0 and b"TDX" in stdout:
                return TEECapability(
                    tee_type=TEEType.INTEL_TDX,
                    available=True,
                    version="1.0",
                    max_memory_mb=32768,
                    max_enclaves=16,
                    supports_migration=False,
                    supports_remote_attestation=True,
                    supports_sealed_storage=True,
                    secure_boot=True,
                    memory_encryption=True,
                    io_protection=True,
                    debug_disabled=True,
                )

        except Exception as e:
            logger.debug(f"Error checking Intel TDX: {e}")

        return None

    @staticmethod
    async def _check_intel_sgx() -> TEECapability | None:
        """Check for Intel SGX support"""
        try:
            # Check for SGX device
            sgx_devices = ["/dev/sgx_enclave", "/dev/sgx/enclave"]
            has_sgx = any(os.path.exists(dev) for dev in sgx_devices)

            if has_sgx:
                return TEECapability(
                    tee_type=TEEType.INTEL_SGX,
                    available=True,
                    version="2.0",
                    max_memory_mb=256,  # SGX has limited memory
                    max_enclaves=128,
                    supports_migration=False,
                    supports_remote_attestation=True,
                    supports_sealed_storage=True,
                    secure_boot=True,
                    memory_encryption=True,
                    io_protection=False,
                    debug_disabled=True,
                )

        except Exception as e:
            logger.debug(f"Error checking Intel SGX: {e}")

        return None


class TEERuntimeManager:
    """
    TEE Runtime Manager

    Manages the lifecycle of Trusted Execution Environments with support for
    multiple TEE technologies and software isolation fallback.
    """

    def __init__(self, config: TEEConfiguration | None = None):
        self.config = config or TEEConfiguration()
        self.capabilities: list[TEECapability] = []
        self.enclaves: dict[str, EnclaveContext] = {}

        # Service components
        self.attestation_service = AttestationService()
        self.enclave_executor = EnclaveExecutor()

        # Runtime management
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.metrics_collector = None
        self.monitoring_tasks: dict[str, asyncio.Task] = {}

        # State management
        self._initialized = False
        self._shutdown = False

        logger.info("TEE Runtime Manager initialized")

    async def initialize(self) -> None:
        """Initialize the TEE runtime manager"""
        if self._initialized:
            return

        logger.info("Initializing TEE Runtime Manager...")

        # Detect available TEE capabilities
        self.capabilities = await TEECapabilityDetector.detect_capabilities()
        logger.info(f"Detected {len(self.capabilities)} TEE capabilities")

        for capability in self.capabilities:
            logger.info(
                f"  - {capability.tee_type.value}: available={capability.available}, "
                f"max_memory={capability.max_memory_mb}MB, max_enclaves={capability.max_enclaves}"
            )

        # Initialize services
        await self.attestation_service.initialize()
        await self.enclave_executor.initialize()

        # Start metrics collection
        self.metrics_collector = asyncio.create_task(self._collect_metrics_loop())

        self._initialized = True
        logger.info("TEE Runtime Manager initialization complete")

    async def shutdown(self) -> None:
        """Shutdown the TEE runtime manager"""
        if self._shutdown:
            return

        logger.info("Shutting down TEE Runtime Manager...")
        self._shutdown = True

        # Terminate all enclaves
        enclave_ids = list(self.enclaves.keys())
        for enclave_id in enclave_ids:
            await self.terminate_enclave(enclave_id)

        # Stop monitoring tasks
        for task in self.monitoring_tasks.values():
            task.cancel()

        # Stop metrics collection
        if self.metrics_collector:
            self.metrics_collector.cancel()

        # Shutdown services
        await self.enclave_executor.shutdown()
        await self.attestation_service.shutdown()

        # Shutdown thread pool
        self.executor.shutdown(wait=True)

        logger.info("TEE Runtime Manager shutdown complete")

    def get_capabilities(self) -> list[TEECapability]:
        """Get available TEE capabilities"""
        return self.capabilities.copy()

    def select_best_tee(self, spec: EnclaveSpec) -> TEECapability | None:
        """Select the best available TEE for the given specification"""
        if not self.capabilities:
            return None

        # Filter by requirements
        suitable_tees = []
        for capability in self.capabilities:
            if not capability.available:
                continue

            # Check resource requirements
            if capability.max_memory_mb < spec.memory_mb or len(self.enclaves) >= capability.max_enclaves:
                continue

            # Check preferred TEE type
            if spec.config.preferred_tee_type and capability.tee_type != spec.config.preferred_tee_type:
                continue

            # Check attestation requirement
            if spec.config.require_attestation and not capability.supports_remote_attestation:
                continue

            suitable_tees.append(capability)

        if not suitable_tees:
            return None

        # Prefer hardware TEEs over software isolation
        hardware_tees = [tee for tee in suitable_tees if tee.is_hardware_tee]
        if hardware_tees:
            # Prefer in order: TDX > SEV-SNP > SGX
            preference_order = [TEEType.INTEL_TDX, TEEType.AMD_SEV_SNP, TEEType.INTEL_SGX]
            for preferred_type in preference_order:
                for tee in hardware_tees:
                    if tee.tee_type == preferred_type:
                        return tee
            return hardware_tees[0]

        # Fall back to software isolation
        return suitable_tees[0]

    async def create_enclave(self, spec: EnclaveSpec) -> EnclaveContext:
        """Create and initialize a new enclave"""
        if not self._initialized:
            raise TEERuntimeError("TEE Runtime Manager not initialized")

        # Validate specification
        validation_errors = spec.validate()
        if validation_errors:
            raise TEERuntimeError(f"Invalid enclave specification: {validation_errors}")

        # Check if enclave already exists
        if spec.enclave_id in self.enclaves:
            raise TEERuntimeError(f"Enclave {spec.enclave_id} already exists")

        # Select appropriate TEE
        selected_tee = self.select_best_tee(spec)
        if not selected_tee:
            raise TEERuntimeError("No suitable TEE available for enclave requirements")

        logger.info(f"Creating enclave {spec.enclave_id} using {selected_tee.tee_type.value}")

        # Create enclave context
        context = EnclaveContext(spec=spec, state=EnclaveState.INITIALIZING, tee_type=selected_tee.tee_type)

        try:
            # Initialize enclave with executor
            await self.enclave_executor.create_enclave(context, selected_tee)

            # Update state
            context.state = EnclaveState.CREATED
            context.started_at = datetime.now(UTC)

            # Store enclave
            self.enclaves[spec.enclave_id] = context

            # Start monitoring
            monitor_task = asyncio.create_task(self._monitor_enclave(context))
            self.monitoring_tasks[spec.enclave_id] = monitor_task

            # Perform attestation if required
            if spec.config.require_attestation:
                await self._attest_enclave(context)

            logger.info(f"Enclave {spec.enclave_id} created successfully")
            return context

        except Exception as e:
            logger.error(f"Failed to create enclave {spec.enclave_id}: {e}")
            context.state = EnclaveState.ERROR
            raise TEERuntimeError(f"Failed to create enclave: {e}")

    async def terminate_enclave(self, enclave_id: str) -> bool:
        """Terminate an enclave"""
        context = self.enclaves.get(enclave_id)
        if not context:
            logger.warning(f"Enclave {enclave_id} not found")
            return False

        logger.info(f"Terminating enclave {enclave_id}")

        try:
            # Stop monitoring
            if enclave_id in self.monitoring_tasks:
                self.monitoring_tasks[enclave_id].cancel()
                del self.monitoring_tasks[enclave_id]

            # Terminate enclave
            await self.enclave_executor.terminate_enclave(context)

            # Update state
            context.state = EnclaveState.TERMINATED
            context.terminated_at = datetime.now(UTC)

            # Remove from active enclaves
            del self.enclaves[enclave_id]

            logger.info(f"Enclave {enclave_id} terminated successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to terminate enclave {enclave_id}: {e}")
            context.state = EnclaveState.ERROR
            return False

    async def get_enclave(self, enclave_id: str) -> EnclaveContext | None:
        """Get enclave context by ID"""
        return self.enclaves.get(enclave_id)

    async def list_enclaves(self) -> list[EnclaveContext]:
        """List all active enclaves"""
        return list(self.enclaves.values())

    async def execute_in_enclave(self, enclave_id: str, code: bytes, input_data: bytes = b"") -> bytes:
        """Execute code in a specific enclave"""
        context = self.enclaves.get(enclave_id)
        if not context:
            raise TEERuntimeError(f"Enclave {enclave_id} not found")

        if not context.is_running:
            raise TEERuntimeError(f"Enclave {enclave_id} is not running (state: {context.state.value})")

        # Check attestation if required
        if context.spec.config.require_attestation and not context.is_attested:
            raise TEERuntimeError(f"Enclave {enclave_id} is not properly attested")

        logger.debug(f"Executing code in enclave {enclave_id}")

        try:
            # Execute through enclave executor
            result = await self.enclave_executor.execute_code(context, code, input_data)

            # Update metrics
            context.metrics.tasks_executed += 1
            context.metrics.tasks_completed += 1

            return result

        except Exception as e:
            logger.error(f"Failed to execute code in enclave {enclave_id}: {e}")
            context.metrics.tasks_executed += 1
            context.metrics.tasks_failed += 1
            raise TEERuntimeError(f"Code execution failed: {e}")

    async def _attest_enclave(self, context: EnclaveContext) -> None:
        """Perform remote attestation for an enclave"""
        logger.info(f"Performing attestation for enclave {context.spec.enclave_id}")

        try:
            # Generate attestation report
            report = await self.attestation_service.generate_attestation_report(context)

            # Verify attestation
            verification_result = await self.attestation_service.verify_attestation(
                report, context.spec.expected_measurements
            )

            if verification_result:
                context.attestation_report = report
                context.last_attestation = datetime.now(UTC)
                context.state = EnclaveState.ATTESTED
                logger.info(f"Enclave {context.spec.enclave_id} attestation successful")
            else:
                context.state = EnclaveState.ERROR
                raise TEERuntimeError("Attestation verification failed")

        except Exception as e:
            logger.error(f"Attestation failed for enclave {context.spec.enclave_id}: {e}")
            context.state = EnclaveState.ERROR
            raise

    async def _monitor_enclave(self, context: EnclaveContext) -> None:
        """Monitor enclave health and metrics"""
        enclave_id = context.spec.enclave_id
        logger.debug(f"Starting monitoring for enclave {enclave_id}")

        while not self._shutdown and enclave_id in self.enclaves:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds

                # Update metrics
                await self._update_enclave_metrics(context)

                # Check health
                await self._check_enclave_health(context)

                # Re-attest if needed
                if context.spec.config.require_attestation:
                    await self._check_attestation_validity(context)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error monitoring enclave {enclave_id}: {e}")
                await asyncio.sleep(60)  # Back off on error

        logger.debug(f"Stopped monitoring for enclave {enclave_id}")

    async def _update_enclave_metrics(self, context: EnclaveContext) -> None:
        """Update metrics for an enclave"""
        try:
            # Get metrics from executor
            metrics = await self.enclave_executor.get_enclave_metrics(context)
            if metrics:
                context.metrics = metrics
                context.metrics.uptime_seconds = context.uptime_seconds

        except Exception as e:
            logger.debug(f"Failed to update metrics for enclave {context.spec.enclave_id}: {e}")

    async def _check_enclave_health(self, context: EnclaveContext) -> None:
        """Check enclave health and handle issues"""
        try:
            # Check if enclave is still responsive
            is_healthy = await self.enclave_executor.check_health(context)

            if not is_healthy:
                logger.warning(f"Enclave {context.spec.enclave_id} appears unhealthy")
                context.state = EnclaveState.ERROR

        except Exception as e:
            logger.debug(f"Health check failed for enclave {context.spec.enclave_id}: {e}")

    async def _check_attestation_validity(self, context: EnclaveContext) -> None:
        """Check if attestation needs renewal"""
        if not context.attestation_report or not context.last_attestation:
            return

        # Check if attestation is expired
        attestation_age = datetime.now(UTC) - context.last_attestation
        max_age = timedelta(seconds=context.spec.config.attestation_timeout_seconds)

        if attestation_age > max_age:
            logger.info(f"Re-attesting enclave {context.spec.enclave_id}")
            await self._attest_enclave(context)

    async def _collect_metrics_loop(self) -> None:
        """Collect and aggregate system metrics"""
        logger.debug("Starting metrics collection loop")

        while not self._shutdown:
            try:
                await asyncio.sleep(60)  # Collect every minute

                # Aggregate enclave metrics
                total_enclaves = len(self.enclaves)
                running_enclaves = len([ctx for ctx in self.enclaves.values() if ctx.state == EnclaveState.RUNNING])
                attested_enclaves = len([ctx for ctx in self.enclaves.values() if ctx.is_attested])

                # Log system status
                logger.debug(
                    f"TEE System Status: {total_enclaves} total enclaves, "
                    f"{running_enclaves} running, {attested_enclaves} attested"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in metrics collection: {e}")
                await asyncio.sleep(120)

        logger.debug("Stopped metrics collection loop")

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status"""
        enclave_states = {}
        for state in EnclaveState:
            count = len([ctx for ctx in self.enclaves.values() if ctx.state == state])
            enclave_states[state.value] = count

        tee_usage = {}
        for tee_type in TEEType:
            count = len([ctx for ctx in self.enclaves.values() if ctx.tee_type == tee_type])
            tee_usage[tee_type.value] = count

        return {
            "initialized": self._initialized,
            "capabilities": len(self.capabilities),
            "enclaves": {"total": len(self.enclaves), "by_state": enclave_states, "by_tee_type": tee_usage},
            "services": {
                "attestation_service": "active" if self.attestation_service else "inactive",
                "enclave_executor": "active" if self.enclave_executor else "inactive",
            },
        }

    @asynccontextmanager
    async def enclave_session(self, spec: EnclaveSpec) -> AsyncGenerator[EnclaveContext, None]:
        """Context manager for temporary enclave sessions"""
        context = await self.create_enclave(spec)
        try:
            yield context
        finally:
            await self.terminate_enclave(spec.enclave_id)
