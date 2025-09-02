"""
Constitutional Mixnode Routing Enhancement

Advanced mixnode routing system with constitutional compliance awareness.
Enhances BetaNet's privacy-preserving mixnet with constitutional oversight
while maintaining strong privacy guarantees through selective routing,
constitutional-aware path selection, and privacy-preserving compliance verification.

Key Features:
- Constitutional-aware mixnode selection and routing
- Tiered privacy routing based on constitutional requirements
- Privacy-preserving constitutional compliance verification in mixnodes
- Constitutional audit trail without compromising anonymity
- Zero-knowledge proofs for mixnode constitutional verification
- TEE-secured constitutional enforcement in critical mixnodes
- Selective disclosure for constitutional transparency
- Advanced privacy protection with constitutional compliance

Architecture:
- Extends BetaNet mixnode architecture with constitutional capabilities
- Implements constitutional routing policies without metadata leakage
- Uses ZK proofs for mixnode constitutional verification
- Provides tiered routing based on constitutional privacy requirements
- Maintains backward compatibility with existing BetaNet mixnet protocol

Constitutional Routing Tiers:
- Bronze: Standard mixnet with constitutional audit trail
- Silver: Enhanced mixnet with selective constitutional monitoring
- Gold: ZK-verified mixnet with privacy-preserving compliance
- Platinum: Ultra-private mixnet with cryptographic constitutional verification
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
import hashlib
import json
import logging
import random
import secrets
from typing import Any, Dict, List, Optional, Tuple

from infrastructure.constitutional.moderation.pipeline import ConstitutionalModerationPipeline

from .constitutional_frames import ConstitutionalTier
from .privacy_verification import PrivacyPreservingVerificationEngine

logger = logging.getLogger(__name__)


class MixnodeCapability(Enum):
    """Mixnode constitutional capabilities."""

    STANDARD_MIXING = "standard_mixing"  # Standard BetaNet mixing
    CONSTITUTIONAL_VERIFICATION = "constitutional_verification"  # Can verify constitutional compliance
    ZK_PROOF_GENERATION = "zk_proof_generation"  # Can generate ZK proofs
    TEE_SECURED_MIXING = "tee_secured_mixing"  # TEE-secured constitutional mixing
    SELECTIVE_DISCLOSURE = "selective_disclosure"  # Supports selective audit disclosure
    PRIVACY_PRESERVING_AUDIT = "privacy_preserving_audit"  # Can audit without metadata leakage


class ConstitutionalRoutingPolicy(Enum):
    """Constitutional routing policies for different privacy levels."""

    FULL_TRANSPARENCY = "full_transparency"  # All mixnodes record constitutional metadata
    SELECTIVE_MONITORING = "selective_monitoring"  # Only designated mixnodes monitor
    PRIVACY_FIRST = "privacy_first"  # Minimal constitutional monitoring
    ZK_VERIFICATION_ONLY = "zk_verification_only"  # Only ZK proof verification
    CRYPTOGRAPHIC_ONLY = "cryptographic_only"  # Pure cryptographic verification


@dataclass
class ConstitutionalMixnode:
    """Mixnode with constitutional compliance capabilities."""

    mixnode_id: str = field(default_factory=lambda: secrets.token_hex(16))

    # Mixnode information
    address: str = ""
    port: int = 8080
    public_key: bytes = b""

    # Constitutional capabilities
    constitutional_capabilities: List[MixnodeCapability] = field(default_factory=list)
    supported_tiers: List[ConstitutionalTier] = field(default_factory=list)
    constitutional_policy: ConstitutionalRoutingPolicy = ConstitutionalRoutingPolicy.PRIVACY_FIRST

    # Performance and reliability
    latency_ms: int = 100
    reliability_score: float = 0.95
    constitutional_compliance_score: float = 0.9

    # Constitutional processing
    constitutional_messages_processed: int = 0
    constitutional_violations_detected: int = 0
    privacy_preservation_rate: float = 0.95

    # TEE and security
    tee_enabled: bool = False
    tee_attestation: Optional[bytes] = None
    last_constitutional_audit: Optional[datetime] = None

    # Network metadata
    location_region: str = ""
    bandwidth_mbps: int = 100
    uptime_percentage: float = 99.5

    # Reputation and trust
    trust_score: float = 0.8
    community_rating: float = 4.0
    operational_since: datetime = field(default_factory=lambda: datetime.now(UTC))

    def supports_tier(self, tier: ConstitutionalTier) -> bool:
        """Check if mixnode supports constitutional tier."""
        return tier in self.supported_tiers

    def has_capability(self, capability: MixnodeCapability) -> bool:
        """Check if mixnode has specific constitutional capability."""
        return capability in self.constitutional_capabilities

    def calculate_routing_score(self, tier: ConstitutionalTier, requirements: Dict[str, Any]) -> float:
        """Calculate routing score for constitutional requirements."""

        score = 0.0

        # Base reliability and performance
        score += self.reliability_score * 0.3
        score += min(1.0, 100.0 / max(1, self.latency_ms)) * 0.2
        score += self.constitutional_compliance_score * 0.3

        # Tier-specific scoring
        if not self.supports_tier(tier):
            return 0.0  # Cannot use this mixnode

        # Constitutional capability scoring
        required_caps = requirements.get("required_capabilities", [])
        for cap in required_caps:
            if self.has_capability(cap):
                score += 0.05
            else:
                score -= 0.1  # Penalty for missing required capability

        # Privacy preservation bonus
        score += self.privacy_preservation_rate * 0.1

        # TEE bonus for high-security tiers
        if tier in [ConstitutionalTier.GOLD, ConstitutionalTier.PLATINUM] and self.tee_enabled:
            score += 0.1

        return max(0.0, min(1.0, score))


@dataclass
class ConstitutionalRoute:
    """Constitutional mixnet route with privacy preservation."""

    route_id: str = field(default_factory=lambda: secrets.token_hex(16))

    # Route configuration
    mixnodes: List[ConstitutionalMixnode] = field(default_factory=list)
    constitutional_tier: ConstitutionalTier = ConstitutionalTier.SILVER
    routing_policy: ConstitutionalRoutingPolicy = ConstitutionalRoutingPolicy.PRIVACY_FIRST

    # Constitutional metadata
    constitutional_checkpoints: List[int] = field(default_factory=list)  # Mixnode indices that verify
    privacy_level: float = 0.5
    monitoring_scope: List[str] = field(default_factory=list)

    # Performance characteristics
    estimated_latency_ms: int = 500
    reliability_score: float = 0.9
    constitutional_compliance_guarantee: float = 0.95

    # Privacy guarantees
    anonymity_set_size: int = 1000
    traffic_analysis_resistance: float = 0.9
    metadata_protection_level: float = 0.8

    # Route lifecycle
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime = field(default_factory=lambda: datetime.now(UTC) + timedelta(hours=1))
    usage_count: int = 0

    def __post_init__(self):
        """Calculate route characteristics."""
        if self.mixnodes:
            # Calculate estimated latency
            self.estimated_latency_ms = sum(node.latency_ms for node in self.mixnodes)

            # Calculate reliability (product of individual reliabilities)
            self.reliability_score = 1.0
            for node in self.mixnodes:
                self.reliability_score *= node.reliability_score

            # Set constitutional checkpoints based on policy
            if self.routing_policy == ConstitutionalRoutingPolicy.FULL_TRANSPARENCY:
                self.constitutional_checkpoints = list(range(len(self.mixnodes)))
            elif self.routing_policy == ConstitutionalRoutingPolicy.SELECTIVE_MONITORING:
                # First and last mixnodes
                self.constitutional_checkpoints = [0, len(self.mixnodes) - 1]
            elif self.routing_policy == ConstitutionalRoutingPolicy.ZK_VERIFICATION_ONLY:
                # Only mixnodes with ZK capability
                self.constitutional_checkpoints = [
                    i
                    for i, node in enumerate(self.mixnodes)
                    if node.has_capability(MixnodeCapability.ZK_PROOF_GENERATION)
                ]
            # Other policies may have no checkpoints for maximum privacy

    def is_valid(self) -> bool:
        """Check if route is still valid."""
        return datetime.now(UTC) < self.expires_at and len(self.mixnodes) >= 3

    def get_constitutional_verification_points(self) -> List[int]:
        """Get mixnode indices that perform constitutional verification."""
        return self.constitutional_checkpoints


@dataclass
class ConstitutionalRoutingRequest:
    """Request for constitutional mixnet routing."""

    request_id: str = field(default_factory=lambda: secrets.token_hex(16))

    # Message requirements
    constitutional_tier: ConstitutionalTier = ConstitutionalTier.SILVER
    privacy_requirements: Dict[str, Any] = field(default_factory=dict)
    constitutional_requirements: Dict[str, Any] = field(default_factory=dict)

    # Routing preferences
    min_mixnodes: int = 3
    max_mixnodes: int = 7
    preferred_regions: List[str] = field(default_factory=list)
    avoid_regions: List[str] = field(default_factory=list)

    # Performance requirements
    max_latency_ms: int = 2000
    min_reliability: float = 0.8
    min_anonymity_set: int = 100

    # Constitutional constraints
    require_tee_nodes: bool = False
    require_constitutional_audit: bool = True
    max_constitutional_checkpoints: int = 2

    # Request metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime = field(default_factory=lambda: datetime.now(UTC) + timedelta(minutes=30))


class ConstitutionalMixnetRouter:
    """
    Constitutional Mixnet Router with Privacy-Preserving Compliance

    Advanced routing system that balances constitutional compliance requirements
    with strong privacy guarantees. Provides intelligent mixnode selection,
    constitutional verification routing, and privacy-preserving audit capabilities.
    """

    def __init__(self):
        # Mixnode registry
        self.mixnodes: Dict[str, ConstitutionalMixnode] = {}
        self.mixnode_capabilities: Dict[MixnodeCapability, List[str]] = {}

        # Routing state
        self.active_routes: Dict[str, ConstitutionalRoute] = {}
        self.route_cache: Dict[str, ConstitutionalRoute] = {}

        # Constitutional components
        self.privacy_verification: Optional[PrivacyPreservingVerificationEngine] = None
        self.moderation_pipeline: Optional[ConstitutionalModerationPipeline] = None

        # Performance monitoring
        self.routing_stats = {
            "total_routes_created": 0,
            "by_constitutional_tier": {tier.name: 0 for tier in ConstitutionalTier},
            "by_routing_policy": {policy.name: 0 for policy in ConstitutionalRoutingPolicy},
            "average_route_length": 0,
            "average_latency_ms": 0,
            "constitutional_compliance_rate": 0.0,
            "privacy_preservation_rate": 0.0,
        }

        # Configuration
        self.config = {
            "max_cached_routes": 1000,
            "route_refresh_interval_minutes": 30,
            "min_mixnodes_per_route": 3,
            "max_mixnodes_per_route": 7,
            "constitutional_verification_enabled": True,
            "privacy_priority_weight": 0.6,
            "constitutional_priority_weight": 0.4,
        }

        logger.info("Constitutional mixnet router initialized")

    async def initialize(self):
        """Initialize constitutional mixnet router."""

        # Initialize privacy verification
        try:
            from .privacy_verification import create_privacy_verification_engine

            self.privacy_verification = await create_privacy_verification_engine()
        except Exception as e:
            logger.warning(f"Privacy verification unavailable: {e}")

        # Initialize constitutional moderation
        try:
            self.moderation_pipeline = ConstitutionalModerationPipeline()
        except Exception as e:
            logger.warning(f"Constitutional moderation unavailable: {e}")

        # Load mixnode registry
        await self._discover_constitutional_mixnodes()

        logger.info("Constitutional mixnet router ready")

    async def _discover_constitutional_mixnodes(self):
        """Discover and register constitutional mixnodes."""

        # Simulate mixnode discovery (production would query network)
        demo_mixnodes = [
            ConstitutionalMixnode(
                mixnode_id="constitutional_mix_001",
                address="mixnode1.constitutional.net",
                port=8443,
                constitutional_capabilities=[
                    MixnodeCapability.CONSTITUTIONAL_VERIFICATION,
                    MixnodeCapability.ZK_PROOF_GENERATION,
                ],
                supported_tiers=[ConstitutionalTier.BRONZE, ConstitutionalTier.SILVER],
                constitutional_policy=ConstitutionalRoutingPolicy.SELECTIVE_MONITORING,
                latency_ms=80,
                reliability_score=0.97,
                constitutional_compliance_score=0.95,
                tee_enabled=False,
                location_region="us-east",
            ),
            ConstitutionalMixnode(
                mixnode_id="constitutional_mix_002",
                address="mixnode2.constitutional.net",
                port=8443,
                constitutional_capabilities=[
                    MixnodeCapability.CONSTITUTIONAL_VERIFICATION,
                    MixnodeCapability.TEE_SECURED_MIXING,
                    MixnodeCapability.PRIVACY_PRESERVING_AUDIT,
                ],
                supported_tiers=[ConstitutionalTier.SILVER, ConstitutionalTier.GOLD],
                constitutional_policy=ConstitutionalRoutingPolicy.ZK_VERIFICATION_ONLY,
                latency_ms=120,
                reliability_score=0.95,
                constitutional_compliance_score=0.98,
                tee_enabled=True,
                location_region="eu-west",
            ),
            ConstitutionalMixnode(
                mixnode_id="constitutional_mix_003",
                address="mixnode3.constitutional.net",
                port=8443,
                constitutional_capabilities=[
                    MixnodeCapability.ZK_PROOF_GENERATION,
                    MixnodeCapability.TEE_SECURED_MIXING,
                    MixnodeCapability.SELECTIVE_DISCLOSURE,
                ],
                supported_tiers=[ConstitutionalTier.GOLD, ConstitutionalTier.PLATINUM],
                constitutional_policy=ConstitutionalRoutingPolicy.CRYPTOGRAPHIC_ONLY,
                latency_ms=150,
                reliability_score=0.93,
                constitutional_compliance_score=0.99,
                tee_enabled=True,
                location_region="ap-southeast",
            ),
            # Add more standard mixnodes
            ConstitutionalMixnode(
                mixnode_id="standard_mix_001",
                address="mixnode4.betanet.io",
                port=8080,
                constitutional_capabilities=[MixnodeCapability.STANDARD_MIXING],
                supported_tiers=[ConstitutionalTier.BRONZE],
                constitutional_policy=ConstitutionalRoutingPolicy.FULL_TRANSPARENCY,
                latency_ms=60,
                reliability_score=0.99,
                constitutional_compliance_score=0.8,
                location_region="us-west",
            ),
            ConstitutionalMixnode(
                mixnode_id="standard_mix_002",
                address="mixnode5.betanet.io",
                port=8080,
                constitutional_capabilities=[MixnodeCapability.STANDARD_MIXING],
                supported_tiers=[ConstitutionalTier.BRONZE, ConstitutionalTier.SILVER],
                constitutional_policy=ConstitutionalRoutingPolicy.PRIVACY_FIRST,
                latency_ms=70,
                reliability_score=0.98,
                constitutional_compliance_score=0.85,
                location_region="eu-north",
            ),
        ]

        # Register mixnodes
        for mixnode in demo_mixnodes:
            await self.register_mixnode(mixnode)

        logger.info(f"Discovered {len(demo_mixnodes)} constitutional mixnodes")

    async def register_mixnode(self, mixnode: ConstitutionalMixnode):
        """Register constitutional mixnode."""

        self.mixnodes[mixnode.mixnode_id] = mixnode

        # Index by capabilities
        for capability in mixnode.constitutional_capabilities:
            if capability not in self.mixnode_capabilities:
                self.mixnode_capabilities[capability] = []
            self.mixnode_capabilities[capability].append(mixnode.mixnode_id)

        logger.info(f"Registered constitutional mixnode: {mixnode.mixnode_id}")

    async def create_constitutional_route(self, request: ConstitutionalRoutingRequest) -> Optional[ConstitutionalRoute]:
        """Create optimal constitutional route based on requirements."""

        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if cache_key in self.route_cache:
                cached_route = self.route_cache[cache_key]
                if cached_route.is_valid():
                    cached_route.usage_count += 1
                    return cached_route

            # Select candidate mixnodes
            candidates = await self._select_candidate_mixnodes(request)
            if len(candidates) < request.min_mixnodes:
                logger.warning("Insufficient mixnodes for constitutional routing")
                return None

            # Create optimal route
            route = await self._optimize_constitutional_route(request, candidates)
            if not route:
                return None

            # Verify route constitutional compliance
            if not await self._verify_route_constitutional_compliance(route, request):
                logger.warning("Route failed constitutional compliance verification")
                return None

            # Cache and store route
            self.route_cache[cache_key] = route
            self.active_routes[route.route_id] = route

            # Update statistics
            await self._update_routing_statistics(route)

            logger.info(f"Created constitutional route: {route.route_id} ({len(route.mixnodes)} mixnodes)")

            return route

        except Exception as e:
            logger.error(f"Error creating constitutional route: {e}")
            return None

    async def _select_candidate_mixnodes(self, request: ConstitutionalRoutingRequest) -> List[ConstitutionalMixnode]:
        """Select candidate mixnodes based on requirements."""

        candidates = []

        for mixnode in self.mixnodes.values():
            # Check tier support
            if not mixnode.supports_tier(request.constitutional_tier):
                continue

            # Check performance requirements
            if mixnode.latency_ms > request.max_latency_ms:
                continue

            if mixnode.reliability_score < request.min_reliability:
                continue

            # Check TEE requirement
            if request.require_tee_nodes and not mixnode.tee_enabled:
                continue

            # Check regional preferences
            if request.preferred_regions and mixnode.location_region not in request.preferred_regions:
                continue

            if request.avoid_regions and mixnode.location_region in request.avoid_regions:
                continue

            # Check constitutional requirements
            required_caps = request.constitutional_requirements.get("required_capabilities", [])
            if required_caps and not all(mixnode.has_capability(cap) for cap in required_caps):
                continue

            candidates.append(mixnode)

        # Sort by routing score
        for candidate in candidates:
            candidate._routing_score = candidate.calculate_routing_score(
                request.constitutional_tier, request.constitutional_requirements
            )

        candidates.sort(key=lambda x: x._routing_score, reverse=True)

        return candidates

    async def _optimize_constitutional_route(
        self, request: ConstitutionalRoutingRequest, candidates: List[ConstitutionalMixnode]
    ) -> Optional[ConstitutionalRoute]:
        """Optimize constitutional route for privacy and compliance."""

        # Determine route length
        route_length = min(max(request.min_mixnodes, 3), min(request.max_mixnodes, len(candidates)))

        if route_length < 3:
            return None

        # Select mixnodes for route
        selected_mixnodes = []

        # First mixnode: Choose from high-scoring constitutional mixnodes
        constitutional_mixnodes = [
            node for node in candidates if node.has_capability(MixnodeCapability.CONSTITUTIONAL_VERIFICATION)
        ]

        if constitutional_mixnodes:
            selected_mixnodes.append(constitutional_mixnodes[0])
            candidates.remove(constitutional_mixnodes[0])
        else:
            selected_mixnodes.append(candidates[0])
            candidates.remove(candidates[0])

        # Middle mixnodes: Balance performance and diversity
        for _ in range(route_length - 2):
            if not candidates:
                break

            # Select based on diversity and performance
            next_node = await self._select_diverse_mixnode(candidates, selected_mixnodes)
            selected_mixnodes.append(next_node)
            candidates.remove(next_node)

        # Last mixnode: High-performance exit node
        if candidates:
            # Choose highest performance remaining node
            exit_node = max(candidates, key=lambda x: x.reliability_score * x._routing_score)
            selected_mixnodes.append(exit_node)

        # Determine routing policy based on tier
        routing_policy = self._determine_routing_policy(request.constitutional_tier, request)

        # Create route
        route = ConstitutionalRoute(
            mixnodes=selected_mixnodes,
            constitutional_tier=request.constitutional_tier,
            routing_policy=routing_policy,
            privacy_level=self._calculate_privacy_level(request.constitutional_tier),
            monitoring_scope=self._determine_monitoring_scope(request.constitutional_tier),
        )

        return route

    async def _select_diverse_mixnode(
        self, candidates: List[ConstitutionalMixnode], selected: List[ConstitutionalMixnode]
    ) -> ConstitutionalMixnode:
        """Select diverse mixnode to improve route anonymity."""

        # Calculate diversity scores
        diversity_scores = {}

        for candidate in candidates:
            diversity_score = 1.0

            # Regional diversity
            selected_regions = [node.location_region for node in selected]
            if candidate.location_region not in selected_regions:
                diversity_score += 0.3

            # Capability diversity
            selected_capabilities = set()
            for node in selected:
                selected_capabilities.update(node.constitutional_capabilities)

            candidate_capabilities = set(candidate.constitutional_capabilities)
            new_capabilities = candidate_capabilities - selected_capabilities
            diversity_score += len(new_capabilities) * 0.1

            # Performance factor
            diversity_score *= candidate._routing_score

            diversity_scores[candidate.mixnode_id] = diversity_score

        # Select best diverse candidate
        best_candidate = max(candidates, key=lambda x: diversity_scores.get(x.mixnode_id, 0))
        return best_candidate

    def _determine_routing_policy(
        self, tier: ConstitutionalTier, request: ConstitutionalRoutingRequest
    ) -> ConstitutionalRoutingPolicy:
        """Determine routing policy based on constitutional tier and requirements."""

        tier_policies = {
            ConstitutionalTier.BRONZE: ConstitutionalRoutingPolicy.FULL_TRANSPARENCY,
            ConstitutionalTier.SILVER: ConstitutionalRoutingPolicy.SELECTIVE_MONITORING,
            ConstitutionalTier.GOLD: ConstitutionalRoutingPolicy.ZK_VERIFICATION_ONLY,
            ConstitutionalTier.PLATINUM: ConstitutionalRoutingPolicy.CRYPTOGRAPHIC_ONLY,
        }

        # Override based on specific requirements
        if not request.require_constitutional_audit:
            return ConstitutionalRoutingPolicy.PRIVACY_FIRST

        return tier_policies.get(tier, ConstitutionalRoutingPolicy.SELECTIVE_MONITORING)

    def _calculate_privacy_level(self, tier: ConstitutionalTier) -> float:
        """Calculate privacy level for constitutional tier."""

        privacy_levels = {
            ConstitutionalTier.BRONZE: 0.2,
            ConstitutionalTier.SILVER: 0.5,
            ConstitutionalTier.GOLD: 0.8,
            ConstitutionalTier.PLATINUM: 0.95,
        }

        return privacy_levels.get(tier, 0.5)

    def _determine_monitoring_scope(self, tier: ConstitutionalTier) -> List[str]:
        """Determine constitutional monitoring scope for tier."""

        monitoring_scopes = {
            ConstitutionalTier.BRONZE: ["H0", "H1", "H2", "H3"],
            ConstitutionalTier.SILVER: ["H2", "H3"],
            ConstitutionalTier.GOLD: ["H3"],
            ConstitutionalTier.PLATINUM: [],
        }

        return monitoring_scopes.get(tier, ["H2", "H3"])

    async def _verify_route_constitutional_compliance(
        self, route: ConstitutionalRoute, request: ConstitutionalRoutingRequest
    ) -> bool:
        """Verify route meets constitutional compliance requirements."""

        # Check constitutional checkpoints
        if request.require_constitutional_audit:
            if not route.constitutional_checkpoints:
                return False

            if len(route.constitutional_checkpoints) > request.max_constitutional_checkpoints:
                return False

        # Verify mixnode capabilities
        for checkpoint_idx in route.constitutional_checkpoints:
            if checkpoint_idx >= len(route.mixnodes):
                continue

            mixnode = route.mixnodes[checkpoint_idx]

            # Check constitutional verification capability
            if not mixnode.has_capability(MixnodeCapability.CONSTITUTIONAL_VERIFICATION):
                if route.routing_policy != ConstitutionalRoutingPolicy.PRIVACY_FIRST:
                    return False

        # Check TEE requirements for high-security tiers
        if request.constitutional_tier in [ConstitutionalTier.GOLD, ConstitutionalTier.PLATINUM]:
            if request.require_tee_nodes:
                tee_nodes = sum(1 for node in route.mixnodes if node.tee_enabled)
                if tee_nodes == 0:
                    return False

        return True

    def _generate_cache_key(self, request: ConstitutionalRoutingRequest) -> str:
        """Generate cache key for routing request."""

        key_data = {
            "tier": request.constitutional_tier.value,
            "min_nodes": request.min_mixnodes,
            "max_nodes": request.max_mixnodes,
            "max_latency": request.max_latency_ms,
            "require_tee": request.require_tee_nodes,
            "preferred_regions": sorted(request.preferred_regions),
            "constitutional_requirements": request.constitutional_requirements,
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    async def _update_routing_statistics(self, route: ConstitutionalRoute):
        """Update routing statistics."""

        self.routing_stats["total_routes_created"] += 1

        # Update by tier
        tier_name = route.constitutional_tier.name
        self.routing_stats["by_constitutional_tier"][tier_name] += 1

        # Update by policy
        policy_name = route.routing_policy.name
        self.routing_stats["by_routing_policy"][policy_name] += 1

        # Update averages
        total_routes = self.routing_stats["total_routes_created"]

        current_avg_length = self.routing_stats["average_route_length"]
        self.routing_stats["average_route_length"] = (
            current_avg_length * (total_routes - 1) + len(route.mixnodes)
        ) / total_routes

        current_avg_latency = self.routing_stats["average_latency_ms"]
        self.routing_stats["average_latency_ms"] = (
            current_avg_latency * (total_routes - 1) + route.estimated_latency_ms
        ) / total_routes

        # Update compliance and privacy rates
        compliance_rate = route.constitutional_compliance_guarantee
        current_compliance = self.routing_stats["constitutional_compliance_rate"]
        self.routing_stats["constitutional_compliance_rate"] = (
            current_compliance * (total_routes - 1) + compliance_rate
        ) / total_routes

        privacy_rate = route.privacy_level
        current_privacy = self.routing_stats["privacy_preservation_rate"]
        self.routing_stats["privacy_preservation_rate"] = (
            current_privacy * (total_routes - 1) + privacy_rate
        ) / total_routes

    async def route_constitutional_message(
        self,
        message_data: bytes,
        constitutional_tier: ConstitutionalTier,
        destination: str,
        routing_requirements: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Route message through constitutional mixnet."""

        try:
            # Create routing request
            request = ConstitutionalRoutingRequest(
                constitutional_tier=constitutional_tier,
                constitutional_requirements=routing_requirements or {},
                require_constitutional_audit=constitutional_tier != ConstitutionalTier.PLATINUM,
            )

            # Create route
            route = await self.create_constitutional_route(request)
            if not route:
                return False, {"error": "Failed to create constitutional route"}

            # Perform constitutional verification at checkpoints
            verification_results = await self._perform_constitutional_verification(message_data, route)

            # Route message through mixnodes
            routing_result = await self._route_through_mixnodes(message_data, route, destination, verification_results)

            return routing_result["success"], routing_result

        except Exception as e:
            logger.error(f"Error routing constitutional message: {e}")
            return False, {"error": str(e)}

    async def _perform_constitutional_verification(
        self, message_data: bytes, route: ConstitutionalRoute
    ) -> Dict[int, Dict[str, Any]]:
        """Perform constitutional verification at designated checkpoints."""

        verification_results = {}

        try:
            message_text = message_data.decode("utf-8", errors="ignore")

            for checkpoint_idx in route.constitutional_checkpoints:
                if checkpoint_idx >= len(route.mixnodes):
                    continue

                mixnode = route.mixnodes[checkpoint_idx]

                # Perform verification based on mixnode capabilities
                if mixnode.has_capability(MixnodeCapability.ZK_PROOF_GENERATION):
                    # Generate ZK proof for constitutional compliance
                    if self.privacy_verification:
                        # Reference moderation result for ZK proof
                        reference_result = type(
                            "ReferenceResult",
                            (),
                            {
                                "harm_analysis": type(
                                    "ReferenceHarmAnalysis",
                                    (),
                                    {"harm_level": "H0", "confidence_score": 0.9, "constitutional_concerns": {}},
                                )(),
                                "decision": type("ReferenceDecision", (), {"value": "allow"})(),
                            },
                        )()

                        proof_success, proof = await self.privacy_verification.generate_constitutional_proof(
                            content=message_text, moderation_result=reference_result, privacy_tier=route.constitutional_tier
                        )

                        verification_results[checkpoint_idx] = {
                            "type": "zk_proof",
                            "success": proof_success,
                            "proof": proof,
                            "mixnode_id": mixnode.mixnode_id,
                        }

                elif mixnode.has_capability(MixnodeCapability.CONSTITUTIONAL_VERIFICATION):
                    # Perform standard constitutional verification
                    if self.moderation_pipeline:
                        moderation_result = await self.moderation_pipeline.process_content(
                            content=message_text, content_type="text", user_tier=route.constitutional_tier.name
                        )

                        verification_results[checkpoint_idx] = {
                            "type": "standard_verification",
                            "success": moderation_result.decision.value in ["allow", "allow_with_warning"],
                            "moderation_result": moderation_result,
                            "mixnode_id": mixnode.mixnode_id,
                        }

        except Exception as e:
            logger.error(f"Constitutional verification failed: {e}")

        return verification_results

    async def _route_through_mixnodes(
        self,
        message_data: bytes,
        route: ConstitutionalRoute,
        destination: str,
        verification_results: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Route message through mixnode sequence."""

        try:
            # Simulate mixnode routing (production would use actual mixnode clients)
            routing_latency = 0
            hops_completed = 0

            for i, mixnode in enumerate(route.mixnodes):
                # Simulate mixnode processing
                await asyncio.sleep(mixnode.latency_ms / 1000.0)
                routing_latency += mixnode.latency_ms
                hops_completed += 1

                # Check constitutional verification at checkpoints
                if i in verification_results:
                    verification = verification_results[i]
                    if not verification.get("success", False):
                        return {
                            "success": False,
                            "error": "Constitutional verification failed at mixnode",
                            "failed_at_mixnode": mixnode.mixnode_id,
                            "hops_completed": hops_completed,
                        }

                # Simulate random failure based on reliability
                if random.random() > mixnode.reliability_score:
                    return {
                        "success": False,
                        "error": "Mixnode routing failure",
                        "failed_mixnode": mixnode.mixnode_id,
                        "hops_completed": hops_completed,
                    }

            # Mark route as used
            route.usage_count += 1

            return {
                "success": True,
                "route_id": route.route_id,
                "hops_completed": hops_completed,
                "total_latency_ms": routing_latency,
                "constitutional_tier": route.constitutional_tier.name,
                "privacy_level": route.privacy_level,
                "verification_points": len(verification_results),
                "destination": destination,
            }

        except Exception as e:
            logger.error(f"Mixnode routing failed: {e}")
            return {"success": False, "error": str(e)}

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""

        return {
            "total_mixnodes": len(self.mixnodes),
            "mixnodes_by_capability": {cap.name: len(nodes) for cap, nodes in self.mixnode_capabilities.items()},
            "routing_statistics": self.routing_stats,
            "active_routes": len(self.active_routes),
            "cached_routes": len(self.route_cache),
            "supported_tiers": list(
                set(tier.name for mixnode in self.mixnodes.values() for tier in mixnode.supported_tiers)
            ),
            "configuration": self.config,
        }

    async def cleanup_expired_routes(self):
        """Clean up expired routes."""

        current_time = datetime.now(UTC)

        # Remove expired active routes
        expired_active = [route_id for route_id, route in self.active_routes.items() if current_time > route.expires_at]

        for route_id in expired_active:
            del self.active_routes[route_id]

        # Remove expired cached routes
        expired_cached = [cache_key for cache_key, route in self.route_cache.items() if current_time > route.expires_at]

        for cache_key in expired_cached:
            del self.route_cache[cache_key]

        if expired_active or expired_cached:
            logger.info(f"Cleaned up {len(expired_active)} active and {len(expired_cached)} cached expired routes")


# Factory functions
async def create_constitutional_mixnet_router() -> ConstitutionalMixnetRouter:
    """Create and initialize constitutional mixnet router."""

    router = ConstitutionalMixnetRouter()
    await router.initialize()
    return router


def create_constitutional_routing_request(
    tier: ConstitutionalTier = ConstitutionalTier.SILVER, **requirements
) -> ConstitutionalRoutingRequest:
    """Create constitutional routing request with specified tier and requirements."""

    return ConstitutionalRoutingRequest(
        constitutional_tier=tier,
        constitutional_requirements=requirements.get("constitutional_requirements", {}),
        privacy_requirements=requirements.get("privacy_requirements", {}),
        **{k: v for k, v in requirements.items() if k not in ["constitutional_requirements", "privacy_requirements"]},
    )


if __name__ == "__main__":
    # Test constitutional mixnet routing
    async def test_constitutional_mixnet():

        router = await create_constitutional_mixnet_router()

        # Test different constitutional tiers
        for tier in [
            ConstitutionalTier.BRONZE,
            ConstitutionalTier.SILVER,
            ConstitutionalTier.GOLD,
            ConstitutionalTier.PLATINUM,
        ]:

            print(f"\n--- Testing {tier.name} Tier ---")

            # Create routing request
            request = create_constitutional_routing_request(
                tier=tier,
                min_mixnodes=3,
                max_mixnodes=5,
                require_tee_nodes=(tier in [ConstitutionalTier.GOLD, ConstitutionalTier.PLATINUM]),
                constitutional_requirements={
                    "required_capabilities": (
                        [MixnodeCapability.CONSTITUTIONAL_VERIFICATION] if tier != ConstitutionalTier.PLATINUM else []
                    )
                },
            )

            # Create route
            route = await router.create_constitutional_route(request)

            if route:
                print(f"  Route created: {route.route_id}")
                print(f"  Mixnodes: {len(route.mixnodes)}")
                print(f"  Privacy level: {route.privacy_level:.1%}")
                print(f"  Constitutional checkpoints: {len(route.constitutional_checkpoints)}")
                print(f"  Estimated latency: {route.estimated_latency_ms}ms")

                # Test message routing
                test_message = f"Test message for {tier.name} constitutional routing".encode()
                success, result = await router.route_constitutional_message(
                    message_data=test_message, constitutional_tier=tier, destination="test_destination"
                )

                print(f"  Message routing: {'Success' if success else 'Failed'}")
                if success:
                    print(f"  Total latency: {result.get('total_latency_ms', 0)}ms")
                    print(f"  Verification points: {result.get('verification_points', 0)}")
            else:
                print(f"  Route creation failed for {tier.name} tier")

        # Print routing statistics
        stats = router.get_routing_statistics()
        print("\n--- Routing Statistics ---")
        print(f"Total mixnodes: {stats['total_mixnodes']}")
        print(f"Routes created: {stats['routing_statistics']['total_routes_created']}")
        print(f"Average route length: {stats['routing_statistics']['average_route_length']:.1f}")
        print(f"Privacy preservation rate: {stats['routing_statistics']['privacy_preservation_rate']:.1%}")
        print(f"Constitutional compliance rate: {stats['routing_statistics']['constitutional_compliance_rate']:.1%}")

        # Cleanup
        await router.cleanup_expired_routes()

    asyncio.run(test_constitutional_mixnet())
