"""
Fog Computing Integration Bridges

This module provides bridge adapters that connect fog computing infrastructure
with external systems while maintaining separation of concerns.

Current Bridges:
- BetaNet Integration: Secure transport integration with BetaNet bounty
- (Future) Other transport protocols and external systems

Design Pattern:
- Import external components as dependencies
- Provide adapter/wrapper interfaces for fog compute
- Maintain compatibility and graceful degradation
- Keep external code untouched for verification
"""

from .betanet_integration import (
    BetaNetFogTransport,
    FogComputeBetaNetService,
    create_betanet_transport,
    get_betanet_capabilities,
    is_betanet_available,
)

__all__ = [
    "BetaNetFogTransport",
    "FogComputeBetaNetService",
    "create_betanet_transport",
    "is_betanet_available",
    "get_betanet_capabilities",
]
