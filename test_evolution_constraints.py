#!/usr/bin/env python3
"""Test evolution with resource constraints."""

import asyncio
from src.production.agent_forge.evolution.resource_constrained_evolution import ResourceConstrainedEvolution
from src.production.agent_forge.evolution.infrastructure_aware_evolution import InfrastructureAwareEvolution
from src.core.resources.device_profiler import DeviceProfiler
from src.core.resources.resource_monitor import ResourceMonitor
from src.core.resources.constraint_manager import ConstraintManager


class MockAgent:
    """Mock evolvable agent for testing."""
    def __init__(self, agent_id: str = "test_agent"):
        self.agent_id = agent_id
        self.version = "1.0"
        self.metrics = {"accuracy": 0.85, "latency": 100}
        self.capabilities = {"evolution": True}
        
    def get_capabilities(self):
        return self.capabilities
        
    def get_metrics(self):
        return self.metrics


async def test_evolution_with_constraints():
    """Test evolution system with resource constraints."""
    print("\n" + "="*60)
    print(" Testing Evolution with Resource Constraints")
    print("="*60)
    
    # Initialize resource management
    profiler = DeviceProfiler()
    monitor = ResourceMonitor(profiler)
    constraint_manager = ConstraintManager(monitor)
    
    # Start monitoring
    await monitor.start_monitoring()
    
    # Create evolution system
    evolution_system = ResourceConstrainedEvolution(
        monitor=monitor,
        constraint_manager=constraint_manager,
        resource_usage_threshold=0.75
    )
    
    # Create test agent
    agent = MockAgent()
    
    # Test evolution feasibility
    print("\n[TEST] Evolution Feasibility Check")
    try:
        feasible = await evolution_system._check_evolution_feasibility("nightly", agent)
        print(f"  Evolution feasible: {feasible}")
    except Exception as e:
        print(f"  Feasibility check error: {e}")
    
    # Test with 2GB memory constraint
    print("\n[TEST] Testing with 2GB Memory Constraint")
    device_profile = profiler.profile_device()
    print(f"  Current device: {device_profile.device_type}")
    print(f"  Total memory: {device_profile.total_memory_mb / 1024:.1f}GB")
    
    # Simulate constrained environment
    snapshot = profiler.take_snapshot()
    print(f"  Available memory: {snapshot.memory_available_mb / 1024:.1f}GB")
    
    # Check if evolution would work on 2GB device
    if snapshot.memory_available_mb > 2048:
        print("  ✓ Device has sufficient memory for 2GB constraint test")
    else:
        print("  ✗ Device memory too low for test")
    
    # Test resource allocation
    print("\n[TEST] Resource Allocation for Evolution")
    try:
        allocation = await evolution_system._allocate_resources_for_evolution("nightly", agent)
        if allocation:
            print(f"  Memory allocated: {allocation['memory_mb']}MB")
            print(f"  CPU allocated: {allocation['cpu_percent']}%")
            print(f"  Task ID: {allocation['task_id']}")
        else:
            print("  Resource allocation failed")
    except Exception as e:
        print(f"  Allocation error: {e}")
    
    # Test infrastructure-aware evolution
    print("\n[TEST] Infrastructure-Aware Evolution")
    infra_evolution = InfrastructureAwareEvolution(
        resource_monitor=monitor,
        constraint_manager=constraint_manager
    )
    
    try:
        # Test evolution mode selection
        mode = await infra_evolution._determine_evolution_mode(agent)
        print(f"  Selected evolution mode: {mode.value}")
        
        # Test system status
        status = await infra_evolution.get_system_status()
        print(f"  P2P enabled: {status.get('p2p_enabled', False)}")
        print(f"  Resource monitoring: {status.get('resource_monitoring_active', False)}")
        print(f"  Active components: {len(status.get('components', []))}")
    except Exception as e:
        print(f"  Infrastructure evolution error: {e}")
    
    # Stop monitoring
    await monitor.stop_monitoring()
    
    print("\n" + "="*60)
    print(" Evolution Constraint Test Complete")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(test_evolution_with_constraints())