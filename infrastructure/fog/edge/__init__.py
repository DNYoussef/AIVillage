"""
Edge Computing Layer for AIVillage Fog

Provides the "peripheral nervous system" layer that handles:
- Edge device capability discovery and advertisement
- WASI/microVM execution runtimes
- Resource monitoring and health reporting
- BetaNet integration for secure edge communication

Architecture:
- Beacon: Capability advertisement and discovery
- Runner: WASI/microVM execution environment
- Monitor: Resource utilization tracking
- Bridge: Integration with BetaNet transport layer
"""

from .aivillage_integration import AIVillageEdgeIntegration, create_integrated_edge_node
from .beacon import CapabilityBeacon, EdgeCapability
from .fabric import EdgeExecutionNode, JobRequest, JobStatus
from .monitor import HealthStatus, ResourceMonitor
from .runner import ExecutionResult, MicroVMRunner, WASIRunner

__version__ = "1.0.0"
__all__ = [
    "CapabilityBeacon",
    "EdgeCapability",
    "WASIRunner",
    "MicroVMRunner",
    "ExecutionResult",
    "ResourceMonitor",
    "HealthStatus",
    "EdgeExecutionNode",
    "JobRequest",
    "JobStatus",
    "AIVillageEdgeIntegration",
    "create_integrated_edge_node",
]
