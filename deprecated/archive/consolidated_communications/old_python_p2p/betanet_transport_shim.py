"""
Betanet Transport Compatibility Shim

Provides backward compatibility for v1 imports while redirecting to the
production-ready v2 implementation.
"""

import warnings

from .betanet_transport_v2 import BetanetMessageV2 as BetanetMessage
from .betanet_transport_v2 import BetanetTransportV2 as BetanetTransport

warnings.warn(
    "betanet_transport.py v1 is deprecated. "
    "Use betanet_transport_v2.BetanetTransportV2 directly for new code. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Export the v2 classes with v1 names for compatibility
__all__ = ["BetanetTransport", "BetanetMessage"]
