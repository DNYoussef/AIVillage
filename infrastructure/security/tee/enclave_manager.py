"""
TEE Enclave Manager for Constitutional Workload Execution

Manages secure enclaves for constitutional AI workloads with:
- Intel SGX enclave lifecycle management
- AMD SEV-SNP virtual machine isolation
- Secure workload deployment and execution
- Memory encryption and sealing
- Attestation-based workload validation
- Cross-enclave secure communication
- Constitutional safety monitoring

Integrates with fog computing infrastructure for distributed constitutional AI execution.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import hashlib
import json
import logging
import os
import secrets
import subprocess
from typing import Any, Dict, List, Optional
import uuid

from .attestation import TEEType, ConstitutionalTier, AttestationResult, get_attestation_manager

logger = logging.getLogger(__name__)


class EnclaveStatus(Enum):
    """Enclave lifecycle status."""

    INITIALIZING = "initializing"
    READY = "ready"
    EXECUTING = "executing"
    SUSPENDED = "suspended"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    ERROR = "error"


class WorkloadType(Enum):
    """Types of constitutional workloads."""

    INFERENCE = "inference"  # Constitutional AI inference
    TRAINING = "training"  # Federated learning with constitutional constraints
    MODERATION = "moderation"  # Content moderation
    SAFETY_CHECK = "safety_check"  # Safety verification
    AUDIT = "audit"  # Constitutional compliance audit


class SecurityLevel(Enum):
    """Security isolation levels."""

    PROCESS = "process"  # Process-level isolation
    CONTAINER = "container"  # Container isolation
    VM = "vm"  # Virtual machine isolation
    ENCLAVE = "enclave"  # Hardware enclave isolation


@dataclass
class EnclaveConfiguration:
    """Configuration for secure enclave."""

    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Basic configuration
    name: str = ""
    description: str = ""
    tee_type: TEEType = TEEType.SOFTWARE_TEE
    security_level: SecurityLevel = SecurityLevel.ENCLAVE

    # Resource allocation
    memory_size_mb: int = 512
    max_threads: int = 4
    storage_size_mb: int = 1024

    # Security settings
    constitutional_tier: ConstitutionalTier = ConstitutionalTier.SILVER
    enable_memory_encryption: bool = True
    enable_attestation: bool = True
    enable_sealing: bool = True

    # Constitutional constraints
    allowed_workload_types: List[WorkloadType] = field(default_factory=list)
    harm_categories_monitored: List[str] = field(
        default_factory=lambda: ["violence", "hate_speech", "harassment", "privacy_violations"]
    )
    max_execution_time_seconds: int = 3600  # 1 hour

    # Communication settings
    enable_network_access: bool = False
    allowed_endpoints: List[str] = field(default_factory=list)
    enable_inter_enclave_communication: bool = True

    # Audit and monitoring
    enable_audit_logging: bool = True
    enable_performance_monitoring: bool = True
    enable_safety_monitoring: bool = True

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class WorkloadManifest:
    """Manifest for constitutional workload execution."""

    manifest_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Workload identification
    name: str = ""
    version: str = "1.0"
    workload_type: WorkloadType = WorkloadType.INFERENCE

    # Code and data
    code_hash: str = ""  # SHA-256 hash of workload code
    data_hash: str = ""  # SHA-256 hash of input data
    model_hash: str = ""  # SHA-256 hash of AI model

    # Constitutional requirements
    constitutional_tier: ConstitutionalTier = ConstitutionalTier.SILVER
    harm_categories: List[str] = field(default_factory=list)
    safety_constraints: Dict[str, Any] = field(default_factory=dict)

    # Resource requirements
    min_memory_mb: int = 256
    min_cpu_cores: int = 1
    estimated_runtime_seconds: int = 300

    # Inputs and outputs
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)

    # Security requirements
    requires_network_access: bool = False
    requires_persistent_storage: bool = False
    requires_external_apis: bool = False

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    created_by: str = ""


@dataclass
class EnclaveInstance:
    """Running enclave instance."""

    instance_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Configuration
    config: EnclaveConfiguration = field(default_factory=EnclaveConfiguration)
    node_id: str = ""

    # State
    status: EnclaveStatus = EnclaveStatus.INITIALIZING
    tee_type: TEEType = TEEType.SOFTWARE_TEE

    # Security context
    attestation_result: Optional[AttestationResult] = None
    measurement_hash: str = ""
    sealed_secrets: Dict[str, bytes] = field(default_factory=dict)

    # Runtime information
    pid: Optional[int] = None
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Workload tracking
    current_workload: Optional[WorkloadManifest] = None
    workload_history: List[str] = field(default_factory=list)

    # Monitoring data
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_received: int = 0

    # Safety monitoring
    constitutional_violations: int = 0
    safety_alerts: List[str] = field(default_factory=list)

    def is_healthy(self) -> bool:
        """Check if enclave is healthy."""
        if self.status in [EnclaveStatus.ERROR, EnclaveStatus.TERMINATED]:
            return False

        # Check heartbeat (should be within last 5 minutes)
        heartbeat_age = datetime.now(UTC) - self.last_heartbeat
        if heartbeat_age.total_seconds() > 300:
            return False

        return True

    def update_heartbeat(self):
        """Update heartbeat timestamp."""
        self.last_heartbeat = datetime.now(UTC)


class IntelSGXEnclaveManager:
    """Intel SGX-specific enclave management."""

    def __init__(self):
        self.sgx_sdk_path = os.getenv("SGX_SDK", "/opt/intel/sgxsdk")
        self.running_enclaves: Dict[str, subprocess.Popen] = {}

    async def create_enclave(self, config: EnclaveConfiguration) -> EnclaveInstance:
        """Create Intel SGX enclave."""
        instance = EnclaveInstance(
            config=config, tee_type=TEEType.INTEL_SGX, measurement_hash=self._calculate_enclave_measurement(config)
        )

        try:
            # Simulate SGX enclave creation
            await self._initialize_sgx_enclave(instance)
            instance.status = EnclaveStatus.READY

            logger.info(f"Created Intel SGX enclave: {instance.instance_id}")

        except Exception as e:
            logger.error(f"Failed to create SGX enclave: {e}")
            instance.status = EnclaveStatus.ERROR

        return instance

    async def deploy_workload(self, instance: EnclaveInstance, manifest: WorkloadManifest) -> bool:
        """Deploy workload to SGX enclave."""
        try:
            # Validate workload against enclave configuration
            if not self._validate_workload_compatibility(instance.config, manifest):
                logger.error("Workload not compatible with enclave configuration")
                return False

            # Load workload code into enclave
            await self._load_workload_code(instance, manifest)

            # Set up sealed storage for secrets
            await self._setup_sealed_storage(instance, manifest)

            instance.current_workload = manifest
            instance.workload_history.append(manifest.manifest_id)
            instance.status = EnclaveStatus.EXECUTING

            logger.info(f"Deployed workload {manifest.name} to SGX enclave {instance.instance_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to deploy workload to SGX enclave: {e}")
            return False

    async def terminate_enclave(self, instance: EnclaveInstance):
        """Terminate Intel SGX enclave."""
        try:
            instance.status = EnclaveStatus.TERMINATING

            # Clean up SGX enclave
            if instance.instance_id in self.running_enclaves:
                process = self.running_enclaves[instance.instance_id]
                process.terminate()
                await asyncio.sleep(1)
                if process.poll() is None:
                    process.kill()
                del self.running_enclaves[instance.instance_id]

            instance.status = EnclaveStatus.TERMINATED
            logger.info(f"Terminated Intel SGX enclave: {instance.instance_id}")

        except Exception as e:
            logger.error(f"Failed to terminate SGX enclave: {e}")
            instance.status = EnclaveStatus.ERROR

    async def _initialize_sgx_enclave(self, instance: EnclaveInstance):
        """Initialize SGX enclave (simulated)."""
        # In real implementation, this would:
        # 1. Load enclave binary
        # 2. Create enclave using SGX SDK
        # 3. Initialize enclave memory
        # 4. Set up secure communication channels

        # Simulate enclave initialization
        await asyncio.sleep(0.1)  # Simulate initialization time

        # Create mock process for demonstration
        process = subprocess.Popen(["sleep", "3600"], stdout=subprocess.PIPE)
        self.running_enclaves[instance.instance_id] = process
        instance.pid = process.pid

    async def _load_workload_code(self, instance: EnclaveInstance, manifest: WorkloadManifest):
        """Load workload code into enclave."""
        # Verify code hash
        if not manifest.code_hash:
            raise ValueError("Workload code hash not provided")

        # In real implementation, this would load and verify the actual code
        logger.debug(f"Loading workload code with hash {manifest.code_hash}")

    async def _setup_sealed_storage(self, instance: EnclaveInstance, manifest: WorkloadManifest):
        """Set up sealed storage for workload secrets."""
        if manifest.requires_persistent_storage:
            # Simulate sealing data to enclave
            sealed_key = secrets.token_bytes(32)
            instance.sealed_secrets["workload_key"] = sealed_key
            logger.debug("Set up sealed storage for workload")

    def _calculate_enclave_measurement(self, config: EnclaveConfiguration) -> str:
        """Calculate enclave measurement hash."""
        # In real SGX, this would be MRENCLAVE value
        measurement_data = f"{config.name}:{config.memory_size_mb}:{config.constitutional_tier.value}"
        return hashlib.sha256(measurement_data.encode()).hexdigest()

    def _validate_workload_compatibility(self, config: EnclaveConfiguration, manifest: WorkloadManifest) -> bool:
        """Validate workload is compatible with enclave."""
        # Check resource requirements
        if manifest.min_memory_mb > config.memory_size_mb:
            return False

        # Check workload type is allowed
        if config.allowed_workload_types and manifest.workload_type not in config.allowed_workload_types:
            return False

        # Check constitutional tier compatibility
        tier_hierarchy = {ConstitutionalTier.BRONZE: 1, ConstitutionalTier.SILVER: 2, ConstitutionalTier.GOLD: 3}

        config_level = tier_hierarchy.get(config.constitutional_tier, 2)
        manifest_level = tier_hierarchy.get(manifest.constitutional_tier, 2)

        if config_level < manifest_level:
            return False

        return True


class AMDSEVEnclaveManager:
    """AMD SEV-SNP-specific enclave management."""

    def __init__(self):
        self.sev_tool_path = "/usr/sbin/sevctl"
        self.running_vms: Dict[str, str] = {}  # instance_id -> vm_id

    async def create_enclave(self, config: EnclaveConfiguration) -> EnclaveInstance:
        """Create AMD SEV-SNP protected VM."""
        instance = EnclaveInstance(
            config=config, tee_type=TEEType.AMD_SEV_SNP, measurement_hash=self._calculate_vm_measurement(config)
        )

        try:
            # Create SEV-protected VM
            vm_id = await self._create_sev_vm(instance)
            self.running_vms[instance.instance_id] = vm_id

            instance.status = EnclaveStatus.READY
            logger.info(f"Created AMD SEV-SNP VM: {instance.instance_id}")

        except Exception as e:
            logger.error(f"Failed to create SEV VM: {e}")
            instance.status = EnclaveStatus.ERROR

        return instance

    async def deploy_workload(self, instance: EnclaveInstance, manifest: WorkloadManifest) -> bool:
        """Deploy workload to SEV-protected VM."""
        try:
            if instance.instance_id not in self.running_vms:
                logger.error("SEV VM not found for instance")
                return False

            # Deploy workload to VM
            await self._deploy_to_sev_vm(instance, manifest)

            instance.current_workload = manifest
            instance.workload_history.append(manifest.manifest_id)
            instance.status = EnclaveStatus.EXECUTING

            logger.info(f"Deployed workload {manifest.name} to SEV VM {instance.instance_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to deploy workload to SEV VM: {e}")
            return False

    async def terminate_enclave(self, instance: EnclaveInstance):
        """Terminate AMD SEV-SNP VM."""
        try:
            instance.status = EnclaveStatus.TERMINATING

            if instance.instance_id in self.running_vms:
                vm_id = self.running_vms[instance.instance_id]
                await self._destroy_sev_vm(vm_id)
                del self.running_vms[instance.instance_id]

            instance.status = EnclaveStatus.TERMINATED
            logger.info(f"Terminated AMD SEV VM: {instance.instance_id}")

        except Exception as e:
            logger.error(f"Failed to terminate SEV VM: {e}")
            instance.status = EnclaveStatus.ERROR

    async def _create_sev_vm(self, instance: EnclaveInstance) -> str:
        """Create SEV-protected VM."""
        # Simulate VM creation
        vm_id = f"sev-vm-{secrets.token_hex(8)}"

        # In real implementation, this would:
        # 1. Create VM with SEV-SNP protection
        # 2. Configure memory encryption
        # 3. Set up attestation
        # 4. Start VM

        await asyncio.sleep(0.2)  # Simulate VM creation time
        return vm_id

    async def _deploy_to_sev_vm(self, instance: EnclaveInstance, manifest: WorkloadManifest):
        """Deploy workload to SEV VM."""
        # Simulate workload deployment
        logger.debug(f"Deploying workload to SEV VM {self.running_vms[instance.instance_id]}")
        await asyncio.sleep(0.1)

    async def _destroy_sev_vm(self, vm_id: str):
        """Destroy SEV VM."""
        # Simulate VM destruction
        logger.debug(f"Destroying SEV VM {vm_id}")
        await asyncio.sleep(0.1)

    def _calculate_vm_measurement(self, config: EnclaveConfiguration) -> str:
        """Calculate VM measurement hash."""
        measurement_data = f"sev:{config.name}:{config.memory_size_mb}"
        return hashlib.sha256(measurement_data.encode()).hexdigest()


class SoftwareEnclaveManager:
    """Software-based enclave simulation for Bronze tier."""

    def __init__(self):
        self.running_processes: Dict[str, subprocess.Popen] = {}

    async def create_enclave(self, config: EnclaveConfiguration) -> EnclaveInstance:
        """Create software-simulated enclave."""
        instance = EnclaveInstance(
            config=config, tee_type=TEEType.SOFTWARE_TEE, measurement_hash=self._calculate_process_measurement(config)
        )

        try:
            # Create isolated process
            process = await self._create_isolated_process(instance)
            self.running_processes[instance.instance_id] = process
            instance.pid = process.pid

            instance.status = EnclaveStatus.READY
            logger.info(f"Created software enclave: {instance.instance_id}")

        except Exception as e:
            logger.error(f"Failed to create software enclave: {e}")
            instance.status = EnclaveStatus.ERROR

        return instance

    async def deploy_workload(self, instance: EnclaveInstance, manifest: WorkloadManifest) -> bool:
        """Deploy workload to software enclave."""
        try:
            # Simple workload deployment
            instance.current_workload = manifest
            instance.workload_history.append(manifest.manifest_id)
            instance.status = EnclaveStatus.EXECUTING

            logger.info(f"Deployed workload {manifest.name} to software enclave {instance.instance_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to deploy workload to software enclave: {e}")
            return False

    async def terminate_enclave(self, instance: EnclaveInstance):
        """Terminate software enclave."""
        try:
            instance.status = EnclaveStatus.TERMINATING

            if instance.instance_id in self.running_processes:
                process = self.running_processes[instance.instance_id]
                process.terminate()
                await asyncio.sleep(0.5)
                if process.poll() is None:
                    process.kill()
                del self.running_processes[instance.instance_id]

            instance.status = EnclaveStatus.TERMINATED
            logger.info(f"Terminated software enclave: {instance.instance_id}")

        except Exception as e:
            logger.error(f"Failed to terminate software enclave: {e}")
            instance.status = EnclaveStatus.ERROR

    async def _create_isolated_process(self, instance: EnclaveInstance) -> subprocess.Popen:
        """Create isolated process for workload execution."""
        # In real implementation, this would create a properly isolated process
        # with namespace isolation, resource limits, etc.
        process = subprocess.Popen(["sleep", "3600"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return process

    def _calculate_process_measurement(self, config: EnclaveConfiguration) -> str:
        """Calculate process measurement hash."""
        measurement_data = f"software:{config.name}:{config.memory_size_mb}"
        return hashlib.sha256(measurement_data.encode()).hexdigest()


class TEEEnclaveManager:
    """
    Main TEE Enclave Manager for Constitutional Workload Execution

    Coordinates secure enclave lifecycle management across different TEE types
    for constitutional AI workload execution in fog computing environments.
    """

    def __init__(self):
        # TEE-specific managers
        self.sgx_manager = IntelSGXEnclaveManager()
        self.sev_manager = AMDSEVEnclaveManager()
        self.software_manager = SoftwareEnclaveManager()

        # State management
        self.active_enclaves: Dict[str, EnclaveInstance] = {}
        self.workload_manifests: Dict[str, WorkloadManifest] = {}
        self.enclave_configurations: Dict[str, EnclaveConfiguration] = {}

        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False

        logger.info("TEE Enclave Manager initialized")

    async def start(self):
        """Start enclave manager services."""
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("TEE Enclave Manager started")

    async def stop(self):
        """Stop enclave manager and cleanup resources."""
        self.running = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        # Terminate all active enclaves
        for instance_id in list(self.active_enclaves.keys()):
            await self.terminate_enclave(instance_id)

        logger.info("TEE Enclave Manager stopped")

    async def create_enclave(
        self, node_id: str, config: EnclaveConfiguration, attestation_result: AttestationResult
    ) -> str:
        """Create secure enclave on specified node."""
        if not attestation_result.is_constitutional_compliant(config.constitutional_tier):
            raise ValueError("Node attestation does not meet constitutional requirements")

        # Select appropriate manager based on TEE type
        if config.tee_type == TEEType.INTEL_SGX:
            manager = self.sgx_manager
        elif config.tee_type == TEEType.AMD_SEV_SNP:
            manager = self.sev_manager
        else:
            manager = self.software_manager

        # Create enclave
        instance = await manager.create_enclave(config)
        instance.node_id = node_id
        instance.attestation_result = attestation_result

        if instance.status != EnclaveStatus.ERROR:
            self.active_enclaves[instance.instance_id] = instance
            self.enclave_configurations[config.config_id] = config

            logger.info(f"Created enclave {instance.instance_id} on node {node_id}")
            return instance.instance_id
        else:
            raise RuntimeError("Failed to create enclave")

    async def deploy_workload(self, instance_id: str, manifest: WorkloadManifest) -> bool:
        """Deploy constitutional workload to enclave."""
        if instance_id not in self.active_enclaves:
            logger.error(f"Enclave {instance_id} not found")
            return False

        instance = self.active_enclaves[instance_id]

        # Validate constitutional compliance
        if not self._validate_constitutional_compliance(instance, manifest):
            logger.error("Workload does not meet constitutional requirements")
            return False

        # Select appropriate manager
        if instance.tee_type == TEEType.INTEL_SGX:
            manager = self.sgx_manager
        elif instance.tee_type == TEEType.AMD_SEV_SNP:
            manager = self.sev_manager
        else:
            manager = self.software_manager

        # Deploy workload
        success = await manager.deploy_workload(instance, manifest)

        if success:
            self.workload_manifests[manifest.manifest_id] = manifest
            logger.info(f"Deployed workload {manifest.name} to enclave {instance_id}")

        return success

    async def terminate_enclave(self, instance_id: str) -> bool:
        """Terminate enclave and cleanup resources."""
        if instance_id not in self.active_enclaves:
            logger.warning(f"Enclave {instance_id} not found")
            return False

        instance = self.active_enclaves[instance_id]

        # Select appropriate manager
        if instance.tee_type == TEEType.INTEL_SGX:
            manager = self.sgx_manager
        elif instance.tee_type == TEEType.AMD_SEV_SNP:
            manager = self.sev_manager
        else:
            manager = self.software_manager

        # Terminate enclave
        await manager.terminate_enclave(instance)

        # Remove from active enclaves
        del self.active_enclaves[instance_id]

        logger.info(f"Terminated enclave {instance_id}")
        return True

    async def execute_constitutional_workload(
        self,
        node_id: str,
        workload_manifest: WorkloadManifest,
        input_data: Dict[str, Any],
        constitutional_tier: ConstitutionalTier = ConstitutionalTier.SILVER,
    ) -> Dict[str, Any]:
        """Execute constitutional workload with full security lifecycle."""

        try:
            # 1. Verify node attestation
            attestation_manager = await get_attestation_manager()
            attestation = attestation_manager.get_attestation_status(node_id)

            if not attestation or not attestation.is_constitutional_compliant(constitutional_tier):
                raise ValueError("Node does not meet constitutional requirements")

            # 2. Create enclave configuration
            config = EnclaveConfiguration(
                name=f"constitutional-{workload_manifest.workload_type.value}",
                tee_type=attestation.tee_type,
                constitutional_tier=constitutional_tier,
                memory_size_mb=max(512, workload_manifest.min_memory_mb),
                allowed_workload_types=[workload_manifest.workload_type],
                harm_categories_monitored=workload_manifest.harm_categories,
            )

            # 3. Create enclave
            instance_id = await self.create_enclave(node_id, config, attestation)

            try:
                # 4. Deploy workload
                if not await self.deploy_workload(instance_id, workload_manifest):
                    raise RuntimeError("Failed to deploy workload")

                # 5. Execute workload
                result = await self._execute_workload(instance_id, input_data)

                return {
                    "success": True,
                    "result": result,
                    "instance_id": instance_id,
                    "execution_time": datetime.now(UTC).isoformat(),
                }

            finally:
                # 6. Cleanup enclave
                await self.terminate_enclave(instance_id)

        except Exception as e:
            logger.error(f"Constitutional workload execution failed: {e}")
            return {"success": False, "error": str(e), "execution_time": datetime.now(UTC).isoformat()}

    def get_enclave_status(self, instance_id: str) -> Optional[EnclaveInstance]:
        """Get status of specific enclave."""
        return self.active_enclaves.get(instance_id)

    def get_active_enclaves_summary(self) -> Dict[str, Any]:
        """Get summary of all active enclaves."""
        summary = {
            "total_enclaves": len(self.active_enclaves),
            "by_tee_type": {},
            "by_status": {},
            "by_constitutional_tier": {},
            "total_workloads": 0,
        }

        for instance in self.active_enclaves.values():
            # Count by TEE type
            tee_type = instance.tee_type.value
            summary["by_tee_type"][tee_type] = summary["by_tee_type"].get(tee_type, 0) + 1

            # Count by status
            status = instance.status.value
            summary["by_status"][status] = summary["by_status"].get(status, 0) + 1

            # Count by constitutional tier
            tier = instance.config.constitutional_tier.value
            summary["by_constitutional_tier"][tier] = summary["by_constitutional_tier"].get(tier, 0) + 1

            # Count workloads
            summary["total_workloads"] += len(instance.workload_history)

        return summary

    # Private methods

    def _validate_constitutional_compliance(self, instance: EnclaveInstance, manifest: WorkloadManifest) -> bool:
        """Validate workload meets constitutional requirements."""
        config = instance.config

        # Check constitutional tier compatibility
        tier_hierarchy = {ConstitutionalTier.BRONZE: 1, ConstitutionalTier.SILVER: 2, ConstitutionalTier.GOLD: 3}

        config_level = tier_hierarchy.get(config.constitutional_tier, 2)
        manifest_level = tier_hierarchy.get(manifest.constitutional_tier, 2)

        if config_level < manifest_level:
            return False

        # Check harm categories are monitored
        for category in manifest.harm_categories:
            if category not in config.harm_categories_monitored:
                logger.warning(f"Harm category {category} not monitored by enclave")

        # Check workload type is allowed
        if config.allowed_workload_types and manifest.workload_type not in config.allowed_workload_types:
            return False

        return True

    async def _execute_workload(self, instance_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workload within enclave."""
        instance = self.active_enclaves[instance_id]

        # Update instance status
        instance.status = EnclaveStatus.EXECUTING
        instance.update_heartbeat()

        try:
            # Simulate workload execution
            await asyncio.sleep(0.5)  # Simulate processing time

            # Mock result generation
            result = {
                "status": "completed",
                "output": f"Processed {len(input_data)} input parameters",
                "constitutional_score": 0.95,  # High constitutional compliance
                "processing_time_ms": 500,
                "enclave_measurement": instance.measurement_hash,
            }

            instance.status = EnclaveStatus.READY
            return result

        except Exception as e:
            instance.status = EnclaveStatus.ERROR
            raise RuntimeError(f"Workload execution failed: {e}")

    async def _monitoring_loop(self):
        """Monitor enclave health and performance."""
        while self.running:
            try:
                current_time = datetime.now(UTC)

                for instance_id, instance in list(self.active_enclaves.items()):
                    # Check enclave health
                    if not instance.is_healthy():
                        logger.warning(f"Enclave {instance_id} health check failed")

                        # Try to recover or terminate unhealthy enclaves
                        if instance.status == EnclaveStatus.ERROR:
                            await self.terminate_enclave(instance_id)

                    # Update heartbeat for active enclaves
                    if instance.status in [EnclaveStatus.READY, EnclaveStatus.EXECUTING]:
                        instance.update_heartbeat()

                    # Check for execution timeouts
                    if instance.current_workload:
                        execution_time = (current_time - instance.start_time).total_seconds()
                        max_time = instance.config.max_execution_time_seconds

                        if execution_time > max_time:
                            logger.warning(f"Enclave {instance_id} execution timeout")
                            await self.terminate_enclave(instance_id)

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)


# Global enclave manager instance
_enclave_manager: Optional[TEEEnclaveManager] = None


async def get_enclave_manager() -> TEEEnclaveManager:
    """Get global enclave manager instance."""
    global _enclave_manager

    if _enclave_manager is None:
        _enclave_manager = TEEEnclaveManager()
        await _enclave_manager.start()

    return _enclave_manager


# Convenience functions for fog compute integration


async def execute_constitutional_inference(
    node_id: str,
    model_name: str,
    input_data: Dict[str, Any],
    constitutional_tier: ConstitutionalTier = ConstitutionalTier.SILVER,
) -> Dict[str, Any]:
    """Execute constitutional AI inference in secure enclave."""

    manifest = WorkloadManifest(
        name=f"constitutional-inference-{model_name}",
        workload_type=WorkloadType.INFERENCE,
        constitutional_tier=constitutional_tier,
        harm_categories=["hate_speech", "violence", "privacy_violations"],
        min_memory_mb=512,
        code_hash=hashlib.sha256(f"inference-{model_name}".encode()).hexdigest(),
    )

    manager = await get_enclave_manager()
    return await manager.execute_constitutional_workload(node_id, manifest, input_data, constitutional_tier)


async def create_constitutional_training_enclave(node_id: str, training_config: Dict[str, Any]) -> str:
    """Create enclave for constitutional federated learning."""

    attestation_manager = await get_attestation_manager()
    attestation = attestation_manager.get_attestation_status(node_id)

    if not attestation:
        raise ValueError("Node not attested")

    config = EnclaveConfiguration(
        name="constitutional-training",
        tee_type=attestation.tee_type,
        constitutional_tier=ConstitutionalTier.GOLD,  # High security for training
        memory_size_mb=training_config.get("memory_mb", 2048),
        allowed_workload_types=[WorkloadType.TRAINING],
        harm_categories_monitored=["hate_speech", "violence", "harassment", "privacy_violations"],
    )

    manager = await get_enclave_manager()
    return await manager.create_enclave(node_id, config, attestation)


if __name__ == "__main__":

    async def test_enclave_manager():
        """Test enclave management system."""
        manager = await get_enclave_manager()

        # Test constitutional inference
        result = await execute_constitutional_inference(
            "test_node_001", "constitutional-gpt", {"prompt": "What is the capital of France?"}
        )
        print(f"Inference result: {json.dumps(result, indent=2)}")

        # Get enclaves summary
        summary = manager.get_active_enclaves_summary()
        print(f"Active enclaves: {json.dumps(summary, indent=2)}")

        await manager.stop()

    asyncio.run(test_enclave_manager())
