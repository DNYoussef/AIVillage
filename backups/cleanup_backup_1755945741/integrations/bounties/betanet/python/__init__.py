"""
BetaNet encrypted internet transport implementation - CONSOLIDATED VERSION.

Based on the production-ready betanet bounty implementation, providing:
- HTX v1.1 compliant transport protocol with enhanced covert channels
- Noise XK encryption with forward secrecy
- Access ticket authentication system
- uTLS fingerprinting for traffic analysis resistance
- Mobile-optimized performance and battery management
- Advanced covert channels (HTTP/2, HTTP/3, WebSocket)
- Mixnet privacy integration with VRF-based routing
- Complete mobile optimization with thermal/battery awareness

CONSOLIDATION STATUS: All scattered BetaNet implementations unified
This includes features from deprecated directories with advanced capabilities.
"""

try:
    from .access_tickets import AccessTicket, TicketManager
except ImportError:
    AccessTicket = None
    TicketManager = None

try:
    from .htx_transport import HtxClient, HtxFrame, HtxServer
except ImportError:
    HtxClient = None
    HtxFrame = None
    HtxServer = None

try:
    from .mixnode_client import MixnodeClient
except ImportError:
    MixnodeClient = None

try:
    from .noise_protocol import NoiseXKHandshake
except ImportError:
    NoiseXKHandshake = None

# CONSOLIDATED ADVANCED FEATURES
try:
    from .covert_channels import (
        BetaNetCovertTransport,
        BetaNetMixnetIntegration,
        CovertChannelConfig,
        CovertChannelType,
        CoverTrafficPattern,
        HTTP2CovertChannel,
        HTTP3CovertChannel,
        create_advanced_betanet_transport,
    )
except ImportError:
    BetaNetCovertTransport = None
    HTTP2CovertChannel = None
    HTTP3CovertChannel = None
    CovertChannelType = None
    CovertChannelConfig = None
    CoverTrafficPattern = None
    BetaNetMixnetIntegration = None
    create_advanced_betanet_transport = None

try:
    from .mixnet_privacy import (
        ConsolidatedBetaNetMixnet,
        ConstantRatePadding,
        MixnetConfig,
        PrivacyMode,
        VRFSelector,
        create_privacy_enhanced_transport,
    )
except ImportError:
    ConsolidatedBetaNetMixnet = None
    VRFSelector = None
    ConstantRatePadding = None
    PrivacyMode = None
    MixnetConfig = None
    create_privacy_enhanced_transport = None

try:
    from .mobile_optimization import (
        AdaptiveChunkingPolicy,
        BatteryState,
        DataBudgetManager,
        MobileBetaNetOptimizer,
        NetworkType,
        ThermalState,
        create_mobile_optimized_transport,
    )
except ImportError:
    MobileBetaNetOptimizer = None
    BatteryState = None
    ThermalState = None
    NetworkType = None
    AdaptiveChunkingPolicy = None
    DataBudgetManager = None
    create_mobile_optimized_transport = None

__all__ = [
    # Core HTX Transport
    "HtxClient",
    "HtxServer",
    "HtxFrame",
    "NoiseXKHandshake",
    "AccessTicket",
    "TicketManager",
    "MixnodeClient",
    # CONSOLIDATED ADVANCED FEATURES
    # Covert Channels
    "BetaNetCovertTransport",
    "HTTP2CovertChannel",
    "HTTP3CovertChannel",
    "CovertChannelType",
    "CovertChannelConfig",
    "CoverTrafficPattern",
    "BetaNetMixnetIntegration",
    "create_advanced_betanet_transport",
    # Mixnet Privacy
    "ConsolidatedBetaNetMixnet",
    "VRFSelector",
    "ConstantRatePadding",
    "PrivacyMode",
    "MixnetConfig",
    "create_privacy_enhanced_transport",
    # Mobile Optimization
    "MobileBetaNetOptimizer",
    "BatteryState",
    "ThermalState",
    "NetworkType",
    "AdaptiveChunkingPolicy",
    "DataBudgetManager",
    "create_mobile_optimized_transport",
]
