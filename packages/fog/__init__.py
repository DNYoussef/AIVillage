"""
Fog Computing Platform for AIVillage

Provides cloud-like APIs over BitChat/BetaNet infrastructure, enabling:
- Secure fog computing using idle edge devices
- WebAssembly sandboxes for safe code execution
- Rentable compute resources like AWS but BetaNet-secured
- Integration with AIVillage agents and RAG systems

Architecture:
- Gateway: Cloud-like control plane with OpenAPI
- Scheduler: Multi-objective placement optimization
- Execution: WASI/microVM sandboxes on edge devices
- Security: RBAC, compliance, and sandbox isolation
- Marketplace: Spot/on-demand fog compute rentals
"""

__version__ = "1.0.0"

# Core exports
from .gateway import AdminAPI, JobsAPI, SandboxAPI, UsageAPI
from .sdk import FogClient

__all__ = [
    "JobsAPI",
    "SandboxAPI",
    "UsageAPI",
    "AdminAPI",
    "FogClient",
]
