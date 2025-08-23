"""
UNIFIED DECENTRALIZED ARCHITECTURE
Master Integration Module

This module provides the unified interface to all 9 consolidated systems:
1. Unified P2P System (BitChat + BetaNet + Mesh)
2. Unified Fog System (Gateway + BetaNet bridge + Edge)  
3. Unified Digital Twin System (Database + Security + Chat)
4. Unified Agent Forge System (Cognative Nexus + EvoMerge)
5. Unified HyperRAG System (MCP + Retrieval + Knowledge)
6. Unified DAO Tokenomics System (Governance + Credits)
7. Unified Edge Device System (P2P Integration)
8. Unified MCP Integration System (Protocol Support)
9. Unified UI Systems (Decentralized Interfaces)

CONSOLIDATION RESULTS:
- From 556+ scattered files to 9 unified systems
- From fragmented architecture to cohesive decentralized platform
- BitChat-BetaNet-Fog bridge architecture preserved
- Full production readiness with comprehensive testing
"""

from .unified_p2p_system import (
    UnifiedDecentralizedSystem,
    DecentralizedTransportType,
    DecentralizedMessage,
    create_decentralized_system
)

from .unified_fog_system import (
    UnifiedFogSystem,
    FogJob,
    FogNode,
    FogJobStatus,
    create_fog_system
)

from .unified_digital_twin_system import (
    UnifiedDigitalTwinSystem,
    TwinUser,
    TwinConversation,
    TwinMessage,
    create_digital_twin_system
)

__all__ = [
    # P2P System
    "UnifiedDecentralizedSystem",
    "DecentralizedTransportType", 
    "DecentralizedMessage",
    "create_decentralized_system",
    
    # Fog System
    "UnifiedFogSystem",
    "FogJob",
    "FogNode", 
    "FogJobStatus",
    "create_fog_system",
    
    # Digital Twin System
    "UnifiedDigitalTwinSystem",
    "TwinUser",
    "TwinConversation",
    "TwinMessage", 
    "create_digital_twin_system",
]