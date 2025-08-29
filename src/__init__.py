"""
Fog Computing Platform for AIVillage

Provides cloud-like APIs over BitChat/BetaNet infrastructure, enabling:
- Secure fog computing using idle edge devices
- WebAssembly sandboxes for safe code execution
- Rentable compute resources like AWS but BetaNet-secured
- Integration with AIVillage agents and RAG systems
- BetaNet bounty integration for advanced transport protocols

Architecture:
- Gateway: Cloud-like control plane with OpenAPI
- Scheduler: Multi-objective placement optimization
- Execution: WASI/microVM sandboxes on edge devices
- Security: RBAC, compliance, and sandbox isolation
- Marketplace: Spot/on-demand fog compute rentals
- BetaNet Integration: Covert channels, mixnet privacy, mobile optimization

BetaNet Integration:
The fog computing platform integrates with the separate BetaNet bounty implementation
to provide advanced transport capabilities while keeping the bounty code untouched
for verification purposes. This enables secure, privacy-preserving job distribution
across fog compute nodes.
"""

__version__ = "1.0.0"

# Core exports
# from .gateway import AdminAPI, JobsAPI, SandboxAPI, UsageAPI  # Commented during reorganization
# from .sdk import FogClient  # Commented during reorganization

# BetaNet integration exports (optional - graceful fallback if not available)
try:
    BETANET_INTEGRATION_AVAILABLE = True
except ImportError:
    BETANET_INTEGRATION_AVAILABLE = False

__all__ = [
    "JobsAPI",
    "SandboxAPI",
    "UsageAPI",
    "AdminAPI",
    "FogClient",
]

# Add BetaNet exports if available
if BETANET_INTEGRATION_AVAILABLE:
    __all__.extend(
        [
            "BetaNetFogTransport",
            "FogComputeBetaNetService",
            "create_betanet_transport",
            "is_betanet_available",
            "get_betanet_capabilities",
        ]
    )
