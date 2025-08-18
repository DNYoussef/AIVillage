"""
AIVillage Architecture Controller

Manages the integration between hardware and software layers:
- Task delegation from digital twin to meta-agents
- King agent coordination decisions
- Hardware/software communication bridging
- Resource allocation between edge devices and fog cloud
- System-wide orchestration
"""

from .orchestrator import ArchitecturalOrchestrator
from .resource_manager import SystemResourceManager
from .task_router import TaskRouter

__all__ = ["ArchitecturalOrchestrator", "TaskRouter", "SystemResourceManager"]
