"""
Constitutional Fog Computing Infrastructure

Phase 1 constitutional transformation of the fog computing system to implement
AI safety and governance at the architectural level.

Key Components:
- Tier mapping from 4-tier to constitutional Bronze/Silver/Gold system
- Constitutional governance engine with harm taxonomy enforcement
- Machine-only moderation integration points
- Constitutional workload router with isolation
- Transparency logging and audit trail system
- Viewpoint firewall integration architecture

This module provides the core constitutional architecture that can operate
independently while preparing for TEE integration.
"""

from .tier_mapping import (
    ConstitutionalTier,
    TierMapping,
    ConstitutionalTierManager,
    map_legacy_tier_to_constitutional,
    get_tier_requirements,
)

from .governance_engine import (
    ConstitutionalGovernanceEngine,
    PolicyDecision,
    GovernanceAction,
    HarmTaxonomy,
    ConstitutionalConstraint,
    ViewpointFirewall,
)

from .workload_router import (
    ConstitutionalWorkloadRouter,
    WorkloadClassification,
    IsolationLevel,
    ConstitutionalRouting,
    TransparencyLogger,
)

__all__ = [
    # Tier mapping
    "ConstitutionalTier",
    "TierMapping",
    "ConstitutionalTierManager",
    "map_legacy_tier_to_constitutional",
    "get_tier_requirements",
    # Governance engine
    "ConstitutionalGovernanceEngine",
    "PolicyDecision",
    "GovernanceAction",
    "HarmTaxonomy",
    "ConstitutionalConstraint",
    "ViewpointFirewall",
    # Workload router
    "ConstitutionalWorkloadRouter",
    "WorkloadClassification",
    "IsolationLevel",
    "ConstitutionalRouting",
    "TransparencyLogger",
]

# Version info
__version__ = "1.0.0"
__constitutional_phase__ = "phase_1_transformation"
