"""
Fog Computing Infrastructure

Implements distributed edge computing using idle device resources:
- Fog node coordination and task distribution
- Idle charging compute utilization
- Distributed inference and training workloads
- Resource pooling across edge devices
- Mobile-aware fog computing policies
"""

from .fog_coordinator import FogCoordinator
from .fog_node import FogNode
from .task_distributor import TaskDistributor

__all__ = [
    "FogCoordinator",
    "FogNode",
    "TaskDistributor",
]
