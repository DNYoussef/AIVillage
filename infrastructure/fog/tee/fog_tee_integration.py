"""
Fog TEE Integration

Integrates TEE capabilities with the existing fog computing infrastructure,
providing confidential computing support for fog nodes and tasks.
"""

import asyncio
from datetime import datetime
import hashlib
import logging
from typing import Any

from ..edge.fog_compute.fog_coordinator import (
    ComputeCapacity,
    FogCoordinator,
    FogTask,
    TaskPriority,
    TaskType,
)
from .tee_runtime_manager import TEERuntimeManager
from .tee_types import (
    EnclaveSpec,
    TEEConfiguration,
    TEEType,
)

logger = logging.getLogger(__name__)


class TEEAwareComputeCapacity(ComputeCapacity):
    """Extended ComputeCapacity with TEE capabilities"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TEE capabilities
        self.tee_available: bool = kwargs.get("tee_available", False)
        self.tee_types: list[TEEType] = kwargs.get("tee_types", [])
        self.tee_memory_mb: int = kwargs.get("tee_memory_mb", 0)
        self.tee_attestation: bool = kwargs.get("tee_attestation", False)
        self.secure_enclaves_supported: int = kwargs.get("secure_enclaves_supported", 0)

    @property
    def has_hardware_tee(self) -> bool:
        """Check if hardware TEE is available"""
        return any(tee_type != TEEType.SOFTWARE_ISOLATION for tee_type in self.tee_types)

    @property
    def best_tee_type(self) -> TEEType | None:
        """Get the best available TEE type"""
        if not self.tee_types:
            return None

        # Preference order: TDX > SEV-SNP > SGX > Software
        preference_order = [TEEType.INTEL_TDX, TEEType.AMD_SEV_SNP, TEEType.INTEL_SGX, TEEType.SOFTWARE_ISOLATION]

        for preferred in preference_order:
            if preferred in self.tee_types:
                return preferred

        return self.tee_types[0]


class TEEAwareFogTask(FogTask):
    """Extended FogTask with TEE requirements"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TEE requirements
        self.requires_tee: bool = kwargs.get("requires_tee", False)
        self.preferred_tee_type: TEEType | None = kwargs.get("preferred_tee_type")
        self.requires_attestation: bool = kwargs.get("requires_attestation", False)
        self.confidentiality_level: str = kwargs.get("confidentiality_level", "public")
        self.enclave_spec: EnclaveSpec | None = kwargs.get("enclave_spec")

    def create_enclave_spec(self) -> EnclaveSpec:
        """Create enclave specification for this task"""
        if self.enclave_spec:
            return self.enclave_spec

        spec = EnclaveSpec(
            name=f"enclave_for_task_{self.task_id}",
            description=f"TEE enclave for {self.task_type.value} task",
            memory_mb=int(self.memory_mb_required * 1.2),  # 20% overhead
            cpu_cores=int(self.cpu_cores_required),
            code_hash=hashlib.sha256(self.input_data).hexdigest(),
            config=TEEConfiguration(
                preferred_tee_type=self.preferred_tee_type,
                memory_mb=self.memory_mb_required,
                cpu_cores=int(self.cpu_cores_required),
                require_attestation=self.requires_attestation,
                measurement_policy="strict" if self.confidentiality_level == "secret" else "permissive",
            ),
        )

        self.enclave_spec = spec
        return spec


class TEEFogCoordinator:
    """
    TEE-Aware Fog Coordinator

    Wraps the existing FogCoordinator with TEE capabilities for confidential
    computing in fog environments.
    """

    def __init__(self, coordinator_id: str | None = None, enable_tee: bool = True):
        self.coordinator_id = coordinator_id or f"tee_fog_coord_{asyncio.get_event_loop().time()}"
        self.enable_tee = enable_tee

        # Base fog coordinator
        self.fog_coordinator = FogCoordinator(coordinator_id, enable_tee=False)

        # TEE components
        self.tee_runtime: TEERuntimeManager | None = None
        self.tee_enclaves: dict[str, str] = {}  # task_id -> enclave_id
        self.tee_policies = {
            "prefer_hardware_tee": True,
            "require_attestation_for_confidential": True,
            "tee_memory_overhead_percent": 20,
            "max_enclaves_per_node": 4,
            "enclave_timeout_seconds": 1800,  # 30 minutes
        }

        # Enhanced statistics
        self.tee_stats = {
            "tee_tasks_scheduled": 0,
            "tee_tasks_completed": 0,
            "attestations_performed": 0,
            "enclaves_created": 0,
            "hardware_tee_usage": 0,
            "software_tee_usage": 0,
        }

        logger.info(f"TEE-aware Fog Coordinator {self.coordinator_id} initialized")

    async def initialize(self) -> bool:
        """Initialize the TEE-aware fog coordinator"""
        logger.info("Initializing TEE-aware fog coordinator...")

        # Initialize base fog coordinator (will be modified to not auto-start TEE)
        # We'll handle TEE initialization ourselves

        if self.enable_tee:
            await self._initialize_tee_runtime()

        logger.info("TEE-aware fog coordinator initialization complete")
        return True

    async def _initialize_tee_runtime(self) -> None:
        """Initialize TEE runtime manager"""
        if self.tee_runtime:
            return

        logger.info("Initializing TEE runtime manager for fog computing...")

        try:
            tee_config = TEEConfiguration(
                memory_mb=2048,  # Larger for fog workloads
                cpu_cores=4,
                allow_debug=False,
                require_attestation=self.tee_policies["require_attestation_for_confidential"],
                attestation_timeout_seconds=300,
                network_isolation=True,
            )

            self.tee_runtime = TEERuntimeManager(tee_config)
            await self.tee_runtime.initialize()

            logger.info("TEE runtime manager initialized for fog computing")

        except Exception as e:
            logger.error(f"Failed to initialize TEE runtime: {e}")
            self.tee_runtime = None

    async def register_node(
        self, node_id: str, capacity: TEEAwareComputeCapacity, node_metadata: dict[str, Any] | None = None
    ) -> bool:
        """Register a fog node with TEE capabilities"""

        # Convert to base ComputeCapacity for fog coordinator
        base_capacity = ComputeCapacity(
            cpu_cores=capacity.cpu_cores,
            cpu_utilization=capacity.cpu_utilization,
            memory_mb=capacity.memory_mb,
            memory_used_mb=capacity.memory_used_mb,
            gpu_available=capacity.gpu_available,
            gpu_memory_mb=capacity.gpu_memory_mb,
            battery_powered=capacity.battery_powered,
            battery_percent=capacity.battery_percent,
            is_charging=capacity.is_charging,
            thermal_state=capacity.thermal_state,
            power_budget_watts=capacity.power_budget_watts,
            network_bandwidth_mbps=capacity.network_bandwidth_mbps,
            network_latency_ms=capacity.network_latency_ms,
        )

        # Store TEE capabilities separately
        enhanced_metadata = node_metadata or {}
        enhanced_metadata.update(
            {
                "tee_available": capacity.tee_available,
                "tee_types": [t.value for t in capacity.tee_types],
                "tee_memory_mb": capacity.tee_memory_mb,
                "tee_attestation": capacity.tee_attestation,
                "secure_enclaves_supported": capacity.secure_enclaves_supported,
            }
        )

        success = await self.fog_coordinator.register_node(node_id, base_capacity, enhanced_metadata)

        if success and capacity.tee_available and self.enable_tee and not self.tee_runtime:
            await self._initialize_tee_runtime()

        return success

    async def submit_task(
        self,
        task_type: TaskType,
        priority: TaskPriority = TaskPriority.NORMAL,
        cpu_cores: float = 1.0,
        memory_mb: int = 512,
        estimated_duration: float = 60.0,
        input_data: bytes = b"",
        deadline: datetime | None = None,
        requires_gpu: bool = False,
        # TEE-specific parameters
        requires_tee: bool = False,
        preferred_tee_type: TEEType | None = None,
        requires_attestation: bool = False,
        confidentiality_level: str = "public",
    ) -> str:
        """Submit a task with optional TEE requirements"""

        task_id = f"task_{asyncio.get_event_loop().time()}"

        # Create enhanced task
        task = TEEAwareFogTask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            cpu_cores_required=cpu_cores,
            memory_mb_required=memory_mb,
            estimated_duration_seconds=estimated_duration,
            input_data=input_data,
            input_size_mb=len(input_data) / (1024 * 1024),
            deadline=deadline,
            requires_gpu=requires_gpu,
            requires_tee=requires_tee,
            preferred_tee_type=preferred_tee_type,
            requires_attestation=requires_attestation,
            confidentiality_level=confidentiality_level,
        )

        # Add to fog coordinator's pending tasks
        self.fog_coordinator.pending_tasks.append(task)
        self.fog_coordinator.stats["tasks_scheduled"] += 1

        if requires_tee:
            self.tee_stats["tee_tasks_scheduled"] += 1

        logger.info(
            f"Submitted task {task_id} ({task_type.value}) requiring {cpu_cores} cores, "
            f"{memory_mb}MB, TEE: {requires_tee}, attestation: {requires_attestation}"
        )

        return task_id

    async def submit_confidential_task(
        self,
        task_type: TaskType,
        code: bytes,
        input_data: bytes = b"",
        priority: TaskPriority = TaskPriority.HIGH,
        cpu_cores: float = 1.0,
        memory_mb: int = 512,
        estimated_duration: float = 60.0,
        preferred_tee_type: TEEType | None = None,
        confidentiality_level: str = "confidential",
    ) -> str:
        """Submit a confidential computing task"""

        return await self.submit_task(
            task_type=task_type,
            priority=priority,
            cpu_cores=cpu_cores,
            memory_mb=memory_mb,
            estimated_duration=estimated_duration,
            input_data=input_data,
            requires_tee=True,
            preferred_tee_type=preferred_tee_type,
            requires_attestation=True,
            confidentiality_level=confidentiality_level,
        )

    async def execute_in_enclave(self, task_id: str, code: bytes) -> bytes:
        """Execute code directly in a task's enclave"""
        if task_id not in self.tee_enclaves or not self.tee_runtime:
            raise ValueError(f"No enclave found for task {task_id}")

        enclave_id = self.tee_enclaves[task_id]
        return await self.tee_runtime.execute_in_enclave(enclave_id, code)

    def _find_suitable_tee_node(self, task: TEEAwareFogTask) -> str | None:
        """Find a node suitable for TEE task execution"""
        suitable_nodes = []

        for node_id, capacity in self.fog_coordinator.node_capacities.items():
            node_metadata = self.fog_coordinator.nodes[node_id].metadata

            # Check basic resource requirements
            if (
                capacity.available_cpu_cores < task.cpu_cores_required
                or capacity.available_memory_mb < task.memory_mb_required
            ):
                continue

            # Check TEE requirements
            if task.requires_tee:
                if not node_metadata.get("tee_available", False):
                    continue

                node_tee_types = [TEEType(t) for t in node_metadata.get("tee_types", [])]
                if task.preferred_tee_type and task.preferred_tee_type not in node_tee_types:
                    continue

                # Check TEE memory capacity
                tee_memory = node_metadata.get("tee_memory_mb", 0)
                required_tee_memory = task.memory_mb_required * (
                    1 + self.tee_policies["tee_memory_overhead_percent"] / 100
                )
                if tee_memory < required_tee_memory:
                    continue

                # Check enclave limit
                active_enclaves = len(
                    [tid for tid, eid in self.tee_enclaves.items() if eid and self._get_task_node(tid) == node_id]
                )
                if active_enclaves >= self.tee_policies["max_enclaves_per_node"]:
                    continue

            # Calculate suitability score
            score = self._calculate_tee_node_score(capacity, task, node_metadata)
            suitable_nodes.append((node_id, score))

        if not suitable_nodes:
            return None

        # Sort by score and return best
        suitable_nodes.sort(key=lambda x: x[1], reverse=True)
        return suitable_nodes[0][0]

    def _calculate_tee_node_score(
        self, capacity: ComputeCapacity, task: TEEAwareFogTask, node_metadata: dict[str, Any]
    ) -> float:
        """Calculate suitability score for TEE task"""
        # Base score from regular fog coordinator logic
        score = capacity.compute_score

        # TEE-specific bonuses
        if task.requires_tee:
            if node_metadata.get("tee_available", False):
                score *= 1.5

                # Hardware TEE bonus
                node_tee_types = [TEEType(t) for t in node_metadata.get("tee_types", [])]
                has_hardware_tee = any(t != TEEType.SOFTWARE_ISOLATION for t in node_tee_types)

                if self.tee_policies["prefer_hardware_tee"] and has_hardware_tee:
                    score *= 1.3

                # Preferred TEE type bonus
                if task.preferred_tee_type and task.preferred_tee_type in node_tee_types:
                    score *= 1.2

                # Attestation capability bonus
                if task.requires_attestation and node_metadata.get("tee_attestation", False):
                    score *= 1.1

        return score

    async def _handle_tee_task_assignment(self, task: TEEAwareFogTask, node_id: str) -> bool:
        """Handle TEE-specific task assignment logic"""
        if not task.requires_tee or not self.tee_runtime:
            return True  # No TEE required or available

        try:
            # Create enclave for the task
            enclave_spec = task.create_enclave_spec()
            enclave_context = await self.tee_runtime.create_enclave(enclave_spec)

            self.tee_enclaves[task.task_id] = enclave_context.spec.enclave_id
            self.tee_stats["enclaves_created"] += 1

            # Track hardware vs software TEE usage
            if enclave_context.tee_type == TEEType.SOFTWARE_ISOLATION:
                self.tee_stats["software_tee_usage"] += 1
            else:
                self.tee_stats["hardware_tee_usage"] += 1

            # Perform attestation if required
            if task.requires_attestation and enclave_context.attestation_report:
                self.tee_stats["attestations_performed"] += 1

            logger.info(f"Created enclave {enclave_context.spec.enclave_id} for task {task.task_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create enclave for task {task.task_id}: {e}")
            task.status = "failed"
            task.error_message = f"TEE enclave creation failed: {e}"
            return False

    async def _handle_tee_task_completion(self, task: TEEAwareFogTask) -> None:
        """Handle TEE-specific task completion logic"""
        if task.requires_tee:
            self.tee_stats["tee_tasks_completed"] += 1

        # Cleanup TEE enclave if task had one
        if task.task_id in self.tee_enclaves and self.tee_runtime:
            try:
                enclave_id = self.tee_enclaves[task.task_id]
                await self.tee_runtime.terminate_enclave(enclave_id)
                del self.tee_enclaves[task.task_id]
                logger.info(f"Cleaned up enclave {enclave_id} for completed task {task.task_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup enclave for task {task.task_id}: {e}")

    def _get_task_node(self, task_id: str) -> str | None:
        """Get the node assigned to a task"""
        for task in self.fog_coordinator.active_tasks.values():
            if task.task_id == task_id:
                return task.assigned_node
        return None

    def get_tee_capabilities(self) -> list[TEEType]:
        """Get available TEE capabilities across all nodes"""
        capabilities = set()
        for node in self.fog_coordinator.nodes.values():
            node_tee_types = node.metadata.get("tee_types", [])
            capabilities.update(TEEType(t) for t in node_tee_types)
        return list(capabilities)

    def get_tee_nodes(self) -> dict[str, list[TEEType]]:
        """Get nodes with TEE capabilities"""
        tee_nodes = {}
        for node_id, node in self.fog_coordinator.nodes.items():
            if node.metadata.get("tee_available", False):
                node_tee_types = [TEEType(t) for t in node.metadata.get("tee_types", [])]
                tee_nodes[node_id] = node_tee_types
        return tee_nodes

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status including TEE"""
        base_status = self.fog_coordinator.get_system_status()

        # Add TEE-specific status
        tee_status = {
            "tee_enabled": self.enable_tee,
            "tee_runtime_active": self.tee_runtime is not None,
            "active_enclaves": len(self.tee_enclaves),
            "tee_capable_nodes": len(
                [
                    node_id
                    for node_id, node in self.fog_coordinator.nodes.items()
                    if node.metadata.get("tee_available", False)
                ]
            ),
            "tee_capabilities": [t.value for t in self.get_tee_capabilities()],
            "tee_statistics": self.tee_stats.copy(),
            "tee_policies": self.tee_policies.copy(),
        }

        base_status["tee_status"] = tee_status
        return base_status

    async def shutdown(self) -> None:
        """Shutdown the TEE-aware fog coordinator"""
        logger.info("Shutting down TEE-aware fog coordinator...")

        # Shutdown TEE runtime
        if self.tee_runtime:
            await self.tee_runtime.shutdown()

        # Shutdown base fog coordinator
        await self.fog_coordinator.shutdown()

        logger.info("TEE-aware fog coordinator shutdown complete")


# Factory functions for easy instantiation


def create_tee_aware_capacity(
    cpu_cores: int, memory_mb: int, tee_types: list[TEEType] = None, **kwargs
) -> TEEAwareComputeCapacity:
    """Create a TEE-aware compute capacity"""
    tee_types = tee_types or [TEEType.SOFTWARE_ISOLATION]

    return TEEAwareComputeCapacity(
        cpu_cores=cpu_cores,
        cpu_utilization=kwargs.get("cpu_utilization", 0.0),
        memory_mb=memory_mb,
        memory_used_mb=kwargs.get("memory_used_mb", 0),
        gpu_available=kwargs.get("gpu_available", False),
        gpu_memory_mb=kwargs.get("gpu_memory_mb", 0),
        battery_powered=kwargs.get("battery_powered", False),
        battery_percent=kwargs.get("battery_percent"),
        is_charging=kwargs.get("is_charging", False),
        thermal_state=kwargs.get("thermal_state", "normal"),
        power_budget_watts=kwargs.get("power_budget_watts", 100.0),
        network_bandwidth_mbps=kwargs.get("network_bandwidth_mbps", 10.0),
        network_latency_ms=kwargs.get("network_latency_ms", 50.0),
        tee_available=kwargs.get("tee_available", len(tee_types) > 0),
        tee_types=tee_types,
        tee_memory_mb=kwargs.get("tee_memory_mb", memory_mb // 2),
        tee_attestation=kwargs.get("tee_attestation", True),
        secure_enclaves_supported=kwargs.get("secure_enclaves_supported", 4),
    )


async def create_tee_fog_coordinator(coordinator_id: str | None = None) -> TEEFogCoordinator:
    """Create and initialize a TEE-aware fog coordinator"""
    coordinator = TEEFogCoordinator(coordinator_id, enable_tee=True)
    await coordinator.initialize()
    return coordinator
