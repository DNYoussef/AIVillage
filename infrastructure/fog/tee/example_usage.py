"""
TEE Fog Computing Integration Example

Demonstrates how to use the TEE system with fog computing infrastructure
for confidential computing tasks.
"""

import asyncio
from datetime import datetime, timedelta
import logging

from ..edge.fog_compute.fog_coordinator import TaskPriority, TaskType
from .device_tee_capabilities import detect_local_device_tee
from .fog_tee_integration import create_tee_aware_capacity
from .tee_types import TEEType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_tee_fog_example():
    """Basic example of TEE fog computing integration"""

    logger.info("=== Basic TEE Fog Computing Example ===")

    # 1. Detect local device TEE capabilities
    local_profile = await detect_local_device_tee()
    logger.info(f"Local device has {len(local_profile.available_tee_types)} TEE types:")
    for tee_cap in local_profile.available_tee_types:
        logger.info(
            f"  - {tee_cap.tee_type.value}: {tee_cap.max_memory_mb}MB, "
            f"attestation: {tee_cap.supports_remote_attestation}"
        )

    # 2. Create TEE-aware fog coordinator
    coordinator = await create_tee_fog_coordinator("basic_example")

    # 3. Register the local device as a fog node
    tee_capacity = create_tee_aware_capacity(
        cpu_cores=4,
        memory_mb=8192,
        tee_types=local_profile.available_tee_types,
        tee_memory_mb=local_profile.max_tee_memory_mb,
        tee_attestation=local_profile.attestation_support,
        secure_enclaves_supported=local_profile.max_concurrent_enclaves,
    )

    await coordinator.register_node("local_tee_node", tee_capacity)

    # 4. Submit regular fog task
    regular_task_id = await coordinator.submit_task(
        TaskType.INFERENCE,
        priority=TaskPriority.NORMAL,
        cpu_cores=1.0,
        memory_mb=512,
        input_data=b"regular computation data",
    )

    logger.info(f"Submitted regular task: {regular_task_id}")

    # 5. Submit confidential task
    confidential_data = b"sensitive medical data for AI analysis"
    confidential_task_id = await coordinator.submit_confidential_task(
        TaskType.INFERENCE,
        code=b"""
# Confidential medical data analysis
import json
data = json.loads(input_data.decode())
result = {"analysis": "confidential_results", "privacy": "protected"}
output_data = json.dumps(result).encode()
        """,
        input_data=confidential_data,
        confidentiality_level="confidential",
        preferred_tee_type=local_profile.primary_tee.tee_type if local_profile.primary_tee else None,
    )

    logger.info(f"Submitted confidential task: {confidential_task_id}")

    # 6. Check system status
    status = coordinator.get_system_status()
    logger.info("TEE fog system status:")
    logger.info(f"  - TEE enabled: {status['tee_status']['tee_enabled']}")
    logger.info(f"  - Active enclaves: {status['tee_status']['active_enclaves']}")
    logger.info(f"  - TEE capable nodes: {status['tee_status']['tee_capable_nodes']}")
    logger.info(f"  - Available TEE types: {status['tee_status']['tee_capabilities']}")

    await coordinator.shutdown()

    logger.info("Basic example completed successfully!")


async def advanced_multi_node_example():
    """Advanced example with multiple TEE nodes and different confidentiality levels"""

    logger.info("\n=== Advanced Multi-Node TEE Example ===")

    # 1. Create advanced TEE fog coordinator
    coordinator = await create_tee_fog_coordinator("advanced_example")

    # 2. Simulate multiple nodes with different TEE capabilities
    nodes = [
        {
            "id": "high_security_server",
            "capacity": create_tee_aware_capacity(
                cpu_cores=16,
                memory_mb=32768,
                tee_types=[TEEType.INTEL_TDX, TEEType.SOFTWARE_ISOLATION],
                tee_memory_mb=8192,
                tee_attestation=True,
                secure_enclaves_supported=32,
                battery_powered=False,
            ),
        },
        {
            "id": "edge_gateway",
            "capacity": create_tee_aware_capacity(
                cpu_cores=8,
                memory_mb=16384,
                tee_types=[TEEType.AMD_SEV_SNP, TEEType.SOFTWARE_ISOLATION],
                tee_memory_mb=4096,
                tee_attestation=True,
                secure_enclaves_supported=16,
                battery_powered=False,
            ),
        },
        {
            "id": "mobile_device",
            "capacity": create_tee_aware_capacity(
                cpu_cores=4,
                memory_mb=6144,
                tee_types=[TEEType.ARM_TRUSTZONE, TEEType.SOFTWARE_ISOLATION],
                tee_memory_mb=1024,
                tee_attestation=False,
                secure_enclaves_supported=4,
                battery_powered=True,
                battery_percent=85,
                is_charging=True,
            ),
        },
    ]

    # Register all nodes
    for node in nodes:
        await coordinator.register_node(node["id"], node["capacity"])
        logger.info(f"Registered node: {node['id']}")

    # 3. Submit tasks with different confidentiality requirements
    tasks = []

    # Public task - can run anywhere
    public_task = await coordinator.submit_task(
        TaskType.PREPROCESSING,
        priority=TaskPriority.NORMAL,
        cpu_cores=2.0,
        memory_mb=1024,
        input_data=b"public dataset preprocessing",
        requires_tee=False,
    )
    tasks.append(("public", public_task))

    # Confidential task - requires TEE with attestation
    confidential_task = await coordinator.submit_task(
        TaskType.TRAINING,
        priority=TaskPriority.HIGH,
        cpu_cores=4.0,
        memory_mb=2048,
        input_data=b"confidential training data",
        requires_tee=True,
        requires_attestation=True,
        confidentiality_level="confidential",
        preferred_tee_type=TEEType.INTEL_TDX,
    )
    tasks.append(("confidential", confidential_task))

    # Secret task - requires hardware TEE with strong attestation
    secret_task = await coordinator.submit_task(
        TaskType.INFERENCE,
        priority=TaskPriority.CRITICAL,
        cpu_cores=2.0,
        memory_mb=1024,
        input_data=b"top secret intelligence data",
        requires_tee=True,
        requires_attestation=True,
        confidentiality_level="secret",
        preferred_tee_type=TEEType.INTEL_TDX,
        deadline=datetime.now() + timedelta(minutes=5),
    )
    tasks.append(("secret", secret_task))

    # Medical task - requires specific compliance
    medical_task = await coordinator.submit_confidential_task(
        TaskType.INFERENCE,
        code=b"""
# HIPAA-compliant medical data processing
import hashlib
# Process medical data with privacy preservation
anonymized_data = hashlib.sha256(input_data).hexdigest()
result = {"status": "processed", "hash": anonymized_data}
        """,
        input_data=b"patient medical records",
        confidentiality_level="confidential",
        preferred_tee_type=TEEType.AMD_SEV_SNP,
    )
    tasks.append(("medical", medical_task))

    logger.info(f"Submitted {len(tasks)} tasks with different confidentiality levels")

    # 4. Monitor system performance
    await asyncio.sleep(2)  # Let tasks get scheduled

    status = coordinator.get_system_status()
    tee_stats = status["tee_status"]["tee_statistics"]

    logger.info("System performance:")
    logger.info(f"  - TEE tasks scheduled: {tee_stats['tee_tasks_scheduled']}")
    logger.info(f"  - Enclaves created: {tee_stats['enclaves_created']}")
    logger.info(f"  - Hardware TEE usage: {tee_stats['hardware_tee_usage']}")
    logger.info(f"  - Software TEE usage: {tee_stats['software_tee_usage']}")

    # 5. Demonstrate direct enclave execution
    if tee_stats["enclaves_created"] > 0:
        logger.info("\nDemonstrating direct enclave execution...")

        # Find a confidential task with an enclave
        for task_type, task_id in tasks:
            if task_type in ["confidential", "secret", "medical"]:
                try:
                    # Execute additional code in the enclave
                    await coordinator.execute_in_enclave(task_id, b'print("Additional secure computation in enclave")')
                    logger.info(f"Direct enclave execution successful for {task_type} task")
                    break
                except Exception as e:
                    logger.debug(f"Direct execution failed for {task_id}: {e}")

    await coordinator.shutdown()

    logger.info("Advanced example completed successfully!")


async def performance_benchmark_example():
    """Benchmark TEE system performance"""

    logger.info("\n=== TEE System Performance Benchmark ===")

    coordinator = await create_tee_fog_coordinator("benchmark_example")

    # Create high-performance node
    capacity = create_tee_aware_capacity(
        cpu_cores=8,
        memory_mb=16384,
        tee_types=[TEEType.SOFTWARE_ISOLATION, TEEType.INTEL_TDX],
        tee_memory_mb=4096,
        secure_enclaves_supported=16,
    )

    await coordinator.register_node("benchmark_node", capacity)

    # Benchmark different task types
    benchmarks = {"regular_tasks": [], "tee_tasks": [], "attested_tasks": []}

    num_tasks = 5

    # Regular tasks
    start_time = asyncio.get_event_loop().time()
    for i in range(num_tasks):
        task_id = await coordinator.submit_task(
            TaskType.INFERENCE, cpu_cores=1.0, memory_mb=256, input_data=f"regular_task_{i}".encode()
        )
        benchmarks["regular_tasks"].append(task_id)
    regular_time = asyncio.get_event_loop().time() - start_time

    # TEE tasks (no attestation)
    start_time = asyncio.get_event_loop().time()
    for i in range(num_tasks):
        task_id = await coordinator.submit_task(
            TaskType.INFERENCE,
            cpu_cores=1.0,
            memory_mb=256,
            input_data=f"tee_task_{i}".encode(),
            requires_tee=True,
            requires_attestation=False,
        )
        benchmarks["tee_tasks"].append(task_id)
    tee_time = asyncio.get_event_loop().time() - start_time

    # Attested TEE tasks
    start_time = asyncio.get_event_loop().time()
    for i in range(num_tasks):
        task_id = await coordinator.submit_task(
            TaskType.INFERENCE,
            cpu_cores=1.0,
            memory_mb=256,
            input_data=f"attested_task_{i}".encode(),
            requires_tee=True,
            requires_attestation=True,
            confidentiality_level="confidential",
        )
        benchmarks["attested_tasks"].append(task_id)
    attested_time = asyncio.get_event_loop().time() - start_time

    # Report results
    logger.info("Performance benchmark results:")
    logger.info(f"  - Regular tasks ({num_tasks}): {regular_time:.3f}s ({regular_time/num_tasks:.3f}s/task)")
    logger.info(f"  - TEE tasks ({num_tasks}): {tee_time:.3f}s ({tee_time/num_tasks:.3f}s/task)")
    logger.info(f"  - Attested tasks ({num_tasks}): {attested_time:.3f}s ({attested_time/num_tasks:.3f}s/task)")

    tee_overhead = ((tee_time - regular_time) / regular_time) * 100 if regular_time > 0 else 0
    attestation_overhead = ((attested_time - tee_time) / tee_time) * 100 if tee_time > 0 else 0

    logger.info(f"  - TEE overhead: {tee_overhead:.1f}%")
    logger.info(f"  - Attestation overhead: {attestation_overhead:.1f}%")

    # Get final system stats
    status = coordinator.get_system_status()
    stats = status["tee_status"]["tee_statistics"]

    logger.info("Final system statistics:")
    logger.info(f"  - Total tasks scheduled: {stats['tee_tasks_scheduled'] + len(benchmarks['regular_tasks'])}")
    logger.info(f"  - TEE tasks scheduled: {stats['tee_tasks_scheduled']}")
    logger.info(f"  - Enclaves created: {stats['enclaves_created']}")
    logger.info(f"  - Attestations performed: {stats['attestations_performed']}")

    await coordinator.shutdown()

    logger.info("Benchmark completed successfully!")


async def main():
    """Run all TEE fog computing examples"""

    logger.info("TEE Fog Computing Integration Examples")
    logger.info("=" * 50)

    try:
        # Run examples in sequence
        await basic_tee_fog_example()
        await advanced_multi_node_example()
        await performance_benchmark_example()

        logger.info("\n" + "=" * 50)
        logger.info("🎉 All examples completed successfully!")
        logger.info("\nThe TEE fog computing system is ready for:")
        logger.info("✅ Confidential computing in fog environments")
        logger.info("✅ Multi-level security with attestation")
        logger.info("✅ Hardware and software TEE support")
        logger.info("✅ Dynamic task scheduling and resource management")
        logger.info("✅ Performance optimization for edge computing")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
