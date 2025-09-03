import pytest
from datetime import datetime

from infrastructure.distributed_inference.core.cross_node_optimizer import (
    CrossNodeOptimizer,
    OptimizationObjective,
    OptimizationAction,
    PerformanceMetrics,
    NodeResourceState,
    FaultType,
)


@pytest.mark.asyncio
async def test_standard_recommendations_balance_load():
    optimizer = CrossNodeOptimizer()
    await optimizer.register_node("node1", {"cpu_utilization": 0.9})
    await optimizer.register_node("node2", {"cpu_utilization": 0.1})

    recs = await optimizer.get_optimization_recommendations(
        OptimizationObjective.BALANCE_LOAD
    )
    assert recs, "Expected at least one recommendation"
    rec = recs[0]
    assert rec.action == OptimizationAction.MIGRATE_WORKLOAD
    assert set(rec.target_nodes) == {"node1", "node2"}


@pytest.mark.asyncio
async def test_performance_anomaly_detection():
    optimizer = CrossNodeOptimizer()
    optimizer.node_states["n1"] = NodeResourceState(
        node_id="n1",
        cpu_utilization=0.9,
        memory_utilization=0.9,
        disk_utilization=0.1,
        network_utilization=0.1,
        active_workloads=5,
        queue_length=5,
        response_time_ms=1500,
        error_rate=0.2,
        last_heartbeat=datetime.now(),
    )

    metrics = PerformanceMetrics(
        timestamp=datetime.now(),
        total_latency_ms=1200,
        network_latency_ms=100,
        compute_latency_ms=1100,
        throughput_ops_per_sec=10,
        success_rate=0.8,
        resource_efficiency=0.4,
        fault_count=0,
        optimization_score=0.3,
    )

    faults = await optimizer._check_performance_anomalies(metrics)
    assert len(faults) >= 2
    fault_types = {f.fault_type for f in faults}
    assert FaultType.SLOW_RESPONSE in fault_types
    assert FaultType.COMPUTE_OVERLOAD in fault_types
    assert len(optimizer.fault_history) == len(faults)


@pytest.mark.asyncio
async def test_network_topology_update_and_optimize():
    optimizer = CrossNodeOptimizer()
    await optimizer.register_node("n1", {"network_utilization": 0.2})
    await optimizer.register_node("n2", {"network_utilization": 0.8})

    await optimizer._update_network_topology()
    topo = optimizer.network_topology
    assert ("n1", "n2") in topo.latencies
    assert "n2" in topo.connections["n1"]

    topo.latencies[("n1", "n2")] = 250
    topo.latencies[("n2", "n1")] = 250
    topo.reliability_scores[("n1", "n2")] = 0.9
    topo.reliability_scores[("n2", "n1")] = 0.9

    await optimizer._optimize_network_topology()
    assert "n2" not in topo.connections.get("n1", set())
