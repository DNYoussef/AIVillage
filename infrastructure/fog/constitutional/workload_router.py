"""
Constitutional Workload Router

Implements intelligent workload routing with constitutional AI safety integration.
Routes workloads based on constitutional tier requirements, isolation levels,
harm prevention, and transparency logging.

Key Features:
- Constitutional tier-aware routing decisions
- Workload classification and risk assessment
- Multi-level isolation (process, container, VM, physical)
- Real-time transparency logging and audit trails
- Integration with governance engine for policy enforcement
- Resource allocation based on constitutional constraints
- Federated workload coordination with safety guarantees

Architecture Decision Record:
- Route sensitive workloads to higher-tier nodes with enhanced isolation
- Apply constitutional constraints at routing level before execution
- Maintain transparency logs for all routing decisions
- Integrate with existing SLA classes while adding constitutional requirements
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid
import json

from .tier_mapping import ConstitutionalTier, ConstitutionalTierManager
from .governance_engine import ConstitutionalGovernanceEngine, PolicyDecision, PolicyDecisionType, HarmCategory

logger = logging.getLogger(__name__)


class WorkloadClassification(str, Enum):
    """Classification of workloads for routing purposes"""
    
    BASIC = "basic"                    # Standard computational workloads
    SENSITIVE = "sensitive"            # Contains potentially sensitive content
    HIGH_RISK = "high_risk"           # Requires enhanced safety measures
    FEDERATED_INFERENCE = "federated_inference"  # Federated inference workloads
    FEDERATED_TRAINING = "federated_training"    # Federated training workloads
    CONSTITUTIONAL_REVIEW = "constitutional_review"  # Requires constitutional oversight


class IsolationLevel(str, Enum):
    """Levels of workload isolation"""
    
    PROCESS = "process"       # Process-level isolation
    CONTAINER = "container"   # Container isolation (Docker/Podman)
    VM = "vm"                # Virtual machine isolation
    PHYSICAL = "physical"     # Physical hardware isolation
    SECURE_ENCLAVE = "secure_enclave"  # TEE/secure enclave (future)


class RoutingPriority(str, Enum):
    """Priority levels for workload routing"""
    
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"
    CONSTITUTIONAL_EMERGENCY = "constitutional_emergency"


@dataclass
class WorkloadRequirements:
    """Requirements for workload execution"""
    
    # Basic resource requirements
    cpu_cores: Decimal = Decimal("1")
    memory_gb: Decimal = Decimal("1")
    storage_gb: Decimal = Decimal("1")
    duration_hours: Decimal = Decimal("1")
    
    # Constitutional requirements
    constitutional_tier: ConstitutionalTier = ConstitutionalTier.BRONZE
    classification: WorkloadClassification = WorkloadClassification.BASIC
    isolation_level: IsolationLevel = IsolationLevel.PROCESS
    
    # Safety and governance requirements
    harm_categories_to_monitor: Set[HarmCategory] = field(default_factory=set)
    requires_human_oversight: bool = False
    requires_constitutional_review: bool = False
    max_allowed_risk_score: Decimal = Decimal("0.5")
    
    # Federated workload specifics
    participants_needed: int = 1
    coordination_complexity: str = "simple"  # simple, moderate, complex
    inter_node_communication: bool = False
    
    # Quality requirements
    min_trust_score: Decimal = Decimal("0.3")
    max_latency_ms: Decimal = Decimal("1000")
    required_uptime_percentage: Decimal = Decimal("95")
    
    # Compliance requirements
    audit_logging_required: bool = True
    transparency_level: str = "basic"  # basic, detailed, comprehensive
    data_residency_requirements: List[str] = field(default_factory=list)


@dataclass
class NodeCapabilities:
    """Capabilities of a fog node for constitutional workloads"""
    
    node_id: str
    
    # Basic capabilities
    available_cpu_cores: Decimal
    available_memory_gb: Decimal
    available_storage_gb: Decimal
    
    # Constitutional capabilities
    supported_tiers: Set[ConstitutionalTier]
    max_isolation_level: IsolationLevel
    constitutional_compliance_rating: Decimal
    
    # Safety and governance capabilities
    governance_engine_integrated: bool = False
    harm_detection_enabled: bool = False
    viewpoint_firewall_available: bool = False
    human_oversight_available: bool = False
    
    # Quality metrics
    trust_score: Decimal = Decimal("0.5")
    reputation_score: Decimal = Decimal("0.5")
    average_latency_ms: Decimal = Decimal("100")
    uptime_percentage: Decimal = Decimal("95")
    
    # Federated capabilities
    federated_coordination_support: bool = False
    secure_multi_party_computation: bool = False
    differential_privacy_support: bool = False
    
    # Location and compliance
    geographic_region: str = ""
    regulatory_compliance: List[str] = field(default_factory=list)
    
    # Current load
    current_workloads: int = 0
    max_concurrent_workloads: int = 10


@dataclass
class RoutingDecision:
    """Decision made by the constitutional workload router"""
    
    routing_id: str
    workload_id: str
    selected_nodes: List[str]
    
    # Decision details
    routing_rationale: str
    confidence_score: Decimal
    estimated_completion_time: datetime
    
    # Constitutional aspects
    constitutional_compliance: bool = True
    policy_decisions_applied: List[str] = field(default_factory=list)
    isolation_level_assigned: IsolationLevel = IsolationLevel.PROCESS
    
    # Quality guarantees
    sla_commitments: Dict[str, Any] = field(default_factory=dict)
    fallback_nodes: List[str] = field(default_factory=list)
    
    # Monitoring and transparency
    transparency_log_id: str = ""
    audit_trail_id: str = ""
    monitoring_required: bool = True
    
    # Timestamps
    decision_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    execution_deadline: Optional[datetime] = None


@dataclass
class TransparencyLogEntry:
    """Entry in the transparency log for routing decisions"""
    
    log_id: str
    timestamp: datetime
    
    # Routing context
    workload_id: str
    user_id: str
    routing_decision_id: str
    
    # Decision details
    action: str  # "route", "reject", "escalate", "monitor"
    reasoning: str
    constitutional_basis: List[str] = field(default_factory=list)
    
    # Privacy-preserving details
    anonymized_content_hash: str = ""
    risk_assessment_summary: Dict[str, Any] = field(default_factory=dict)
    governance_actions_taken: List[str] = field(default_factory=list)
    
    # Compliance tracking
    tier: ConstitutionalTier = ConstitutionalTier.BRONZE
    isolation_level: IsolationLevel = IsolationLevel.PROCESS
    transparency_level: str = "basic"
    
    # Audit requirements
    requires_public_reporting: bool = False
    retention_period_days: int = 90


class TransparencyLogger:
    """
    Transparency logging system for constitutional AI safety
    
    Provides privacy-preserving transparency logs that can be used for:
    - Public accountability reporting
    - Constitutional compliance auditing  
    - Governance oversight and appeals
    - Research and safety analysis
    """
    
    def __init__(self):
        self.log_entries: List[TransparencyLogEntry] = []
        self.config = {
            "public_reporting_enabled": True,
            "anonymization_level": "high",
            "retention_policy_days": 90,
            "real_time_monitoring": True,
        }
        
        logger.info("Transparency logger initialized")
    
    async def log_routing_decision(
        self, 
        routing_decision: RoutingDecision, 
        workload_requirements: WorkloadRequirements,
        user_id: str,
        content_summary: Dict[str, Any] = None
    ) -> str:
        """Log routing decision with appropriate transparency level"""
        
        log_id = f"tlog_{uuid.uuid4().hex[:8]}"
        
        # Anonymize content based on transparency requirements
        anonymized_hash = ""
        if content_summary:
            content_str = json.dumps(content_summary, sort_keys=True)
            anonymized_hash = f"hash_{hash(content_str) % 1000000:06d}"
        
        # Create risk assessment summary (anonymized)
        risk_summary = {
            "classification": workload_requirements.classification.value,
            "isolation_level": workload_requirements.isolation_level.value,
            "tier": workload_requirements.constitutional_tier.value,
            "max_allowed_risk": float(workload_requirements.max_allowed_risk_score),
        }
        
        entry = TransparencyLogEntry(
            log_id=log_id,
            timestamp=datetime.now(UTC),
            workload_id=routing_decision.workload_id,
            user_id=self._anonymize_user_id(user_id),
            routing_decision_id=routing_decision.routing_id,
            action="route",
            reasoning=routing_decision.routing_rationale,
            constitutional_basis=routing_decision.policy_decisions_applied,
            anonymized_content_hash=anonymized_hash,
            risk_assessment_summary=risk_summary,
            tier=workload_requirements.constitutional_tier,
            isolation_level=routing_decision.isolation_level_assigned,
            transparency_level=workload_requirements.transparency_level,
            requires_public_reporting=(workload_requirements.constitutional_tier == ConstitutionalTier.GOLD),
        )
        
        self.log_entries.append(entry)
        
        logger.info(f"Logged routing decision {routing_decision.routing_id} to transparency log {log_id}")
        return log_id
    
    async def log_governance_decision(self, policy_decision: PolicyDecision):
        """Log governance engine decisions"""
        
        log_id = f"tlog_{uuid.uuid4().hex[:8]}"
        
        entry = TransparencyLogEntry(
            log_id=log_id,
            timestamp=datetime.now(UTC),
            workload_id=policy_decision.workload_id,
            user_id=self._anonymize_user_id(policy_decision.user_id),
            routing_decision_id="",  # Not applicable for governance decisions
            action=policy_decision.decision_type.value,
            reasoning=policy_decision.reasoning,
            constitutional_basis=policy_decision.constitutional_basis,
            governance_actions_taken=[action.value for action in policy_decision.actions],
            tier=policy_decision.tier,
            requires_public_reporting=(policy_decision.tier == ConstitutionalTier.GOLD),
        )
        
        self.log_entries.append(entry)
        
        logger.info(f"Logged governance decision {policy_decision.decision_id} to transparency log {log_id}")
        return log_id
    
    def _anonymize_user_id(self, user_id: str) -> str:
        """Anonymize user ID for transparency logging"""
        if self.config["anonymization_level"] == "high":
            return f"user_{hash(user_id) % 10000:04d}"
        else:
            return user_id[:8] + "***"
    
    async def generate_transparency_report(self, days_back: int = 30) -> Dict[str, Any]:
        """Generate public transparency report"""
        
        cutoff_date = datetime.now(UTC) - timedelta(days=days_back)
        recent_entries = [entry for entry in self.log_entries if entry.timestamp > cutoff_date]
        
        # Aggregate statistics
        total_decisions = len(recent_entries)
        decisions_by_tier = {}
        decisions_by_action = {}
        governance_actions_taken = {}
        
        for entry in recent_entries:
            # Tier distribution
            tier = entry.tier.value
            decisions_by_tier[tier] = decisions_by_tier.get(tier, 0) + 1
            
            # Action distribution
            action = entry.action
            decisions_by_action[action] = decisions_by_action.get(action, 0) + 1
            
            # Governance actions
            for gov_action in entry.governance_actions_taken:
                governance_actions_taken[gov_action] = governance_actions_taken.get(gov_action, 0) + 1
        
        report = {
            "report_period": {
                "start_date": cutoff_date.isoformat(),
                "end_date": datetime.now(UTC).isoformat(),
                "days": days_back,
            },
            "summary_statistics": {
                "total_decisions": total_decisions,
                "decisions_by_tier": decisions_by_tier,
                "decisions_by_action": decisions_by_action,
                "governance_actions_taken": governance_actions_taken,
            },
            "constitutional_compliance": {
                "total_constitutional_decisions": len([e for e in recent_entries if e.constitutional_basis]),
                "escalations_to_human_review": len([e for e in recent_entries if "escalate" in e.action]),
                "transparency_level_distribution": self._get_transparency_distribution(recent_entries),
            },
            "safety_metrics": {
                "high_risk_workloads_processed": len([e for e in recent_entries 
                                                     if e.risk_assessment_summary.get("max_allowed_risk", 0) > 0.7]),
                "isolation_level_usage": self._get_isolation_distribution(recent_entries),
            },
            "metadata": {
                "report_generated_at": datetime.now(UTC).isoformat(),
                "anonymization_level": self.config["anonymization_level"],
                "data_retention_days": self.config["retention_policy_days"],
            },
        }
        
        return report
    
    def _get_transparency_distribution(self, entries: List[TransparencyLogEntry]) -> Dict[str, int]:
        """Get distribution of transparency levels"""
        distribution = {}
        for entry in entries:
            level = entry.transparency_level
            distribution[level] = distribution.get(level, 0) + 1
        return distribution
    
    def _get_isolation_distribution(self, entries: List[TransparencyLogEntry]) -> Dict[str, int]:
        """Get distribution of isolation levels used"""
        distribution = {}
        for entry in entries:
            level = entry.isolation_level.value
            distribution[level] = distribution.get(level, 0) + 1
        return distribution


class ConstitutionalWorkloadRouter:
    """
    Constitutional workload router with integrated AI safety
    
    Routes workloads to appropriate nodes based on:
    - Constitutional tier requirements and constraints
    - Workload classification and risk assessment  
    - Node capabilities and constitutional compliance
    - Isolation requirements and safety guarantees
    - Transparency and audit requirements
    
    Integrates with:
    - ConstitutionalTierManager for tier-based routing
    - ConstitutionalGovernanceEngine for policy enforcement
    - TransparencyLogger for accountability
    - Existing SLA classes and fog infrastructure
    """
    
    def __init__(
        self, 
        tier_manager: Optional[ConstitutionalTierManager] = None,
        governance_engine: Optional[ConstitutionalGovernanceEngine] = None
    ):
        self.tier_manager = tier_manager or ConstitutionalTierManager()
        self.governance_engine = governance_engine
        self.transparency_logger = TransparencyLogger()
        
        # Node registry
        self.available_nodes: Dict[str, NodeCapabilities] = {}
        self.node_load_tracking: Dict[str, List[datetime]] = {}
        
        # Routing state
        self.active_routes: Dict[str, RoutingDecision] = {}
        self.routing_history: List[RoutingDecision] = []
        
        # Configuration
        self.routing_config = {
            "max_routing_attempts": 3,
            "load_balancing_strategy": "constitutional_aware",
            "fallback_enabled": True,
            "real_time_monitoring": True,
            "transparency_logging": True,
        }
        
        # Integration points
        self.sla_manager = None  # Will be injected from existing infrastructure
        self.scheduler = None    # Will be injected from existing infrastructure
        
        logger.info("Constitutional workload router initialized")
    
    async def register_node(self, node_capabilities: NodeCapabilities):
        """Register a node with its constitutional capabilities"""
        
        node_id = node_capabilities.node_id
        self.available_nodes[node_id] = node_capabilities
        self.node_load_tracking[node_id] = []
        
        logger.info(
            f"Registered constitutional node {node_id}: "
            f"tiers={[t.value for t in node_capabilities.supported_tiers]}, "
            f"max_isolation={node_capabilities.max_isolation_level.value}, "
            f"trust_score={float(node_capabilities.trust_score):.2f}"
        )
    
    async def route_workload(
        self, 
        workload_id: str,
        user_id: str, 
        requirements: WorkloadRequirements,
        content_data: Dict[str, Any] = None
    ) -> RoutingDecision:
        """Route workload with constitutional safety integration"""
        
        routing_id = f"route_{uuid.uuid4().hex[:8]}"
        
        # Step 1: Get user's constitutional tier and validate requirements
        user_tier = self.tier_manager.get_constitutional_tier(user_id)
        requirements.constitutional_tier = user_tier
        
        # Step 2: Validate tier compliance  
        compliance_result = await self.tier_manager.validate_tier_compliance(
            user_id, requirements.classification.value, content_data or {}
        )
        
        if not compliance_result["compliant"]:
            # Create rejection decision
            return RoutingDecision(
                routing_id=routing_id,
                workload_id=workload_id,
                selected_nodes=[],
                routing_rationale=f"Workload not compliant with {user_tier.value} tier: {compliance_result['violations']}",
                confidence_score=Decimal("0"),
                estimated_completion_time=datetime.now(UTC),
                constitutional_compliance=False,
            )
        
        # Step 3: Apply governance engine evaluation if available
        policy_decisions = []
        if self.governance_engine and content_data:
            policy_decision = await self.governance_engine.evaluate_workload(
                workload_id, user_id, content_data
            )
            policy_decisions.append(policy_decision.decision_id)
            
            # Handle governance decision
            if policy_decision.decision_type == PolicyDecisionType.BLOCK:
                return RoutingDecision(
                    routing_id=routing_id,
                    workload_id=workload_id,
                    selected_nodes=[],
                    routing_rationale=f"Workload blocked by governance engine: {policy_decision.reasoning}",
                    confidence_score=Decimal("0"),
                    estimated_completion_time=datetime.now(UTC),
                    constitutional_compliance=False,
                    policy_decisions_applied=policy_decisions,
                )
            
            elif policy_decision.decision_type == PolicyDecisionType.ESCALATE:
                # For escalated decisions, route to Gold-tier nodes with human oversight
                requirements.requires_human_oversight = True
                requirements.constitutional_tier = ConstitutionalTier.GOLD
        
        # Step 4: Find suitable nodes
        candidate_nodes = await self._find_candidate_nodes(requirements)
        
        if not candidate_nodes:
            return RoutingDecision(
                routing_id=routing_id,
                workload_id=workload_id,
                selected_nodes=[],
                routing_rationale="No suitable nodes found matching constitutional requirements",
                confidence_score=Decimal("0"),
                estimated_completion_time=datetime.now(UTC),
                constitutional_compliance=True,
            )
        
        # Step 5: Select optimal nodes
        selected_nodes, fallback_nodes = await self._select_optimal_nodes(
            candidate_nodes, requirements
        )
        
        # Step 6: Determine isolation level
        isolation_level = self._determine_isolation_level(requirements, selected_nodes)
        
        # Step 7: Create routing decision
        routing_decision = RoutingDecision(
            routing_id=routing_id,
            workload_id=workload_id,
            selected_nodes=[node.node_id for node in selected_nodes],
            routing_rationale=self._generate_routing_rationale(requirements, selected_nodes),
            confidence_score=self._calculate_routing_confidence(selected_nodes, requirements),
            estimated_completion_time=self._estimate_completion_time(requirements, selected_nodes),
            constitutional_compliance=True,
            policy_decisions_applied=policy_decisions,
            isolation_level_assigned=isolation_level,
            fallback_nodes=[node.node_id for node in fallback_nodes],
            monitoring_required=(requirements.classification != WorkloadClassification.BASIC),
        )
        
        # Step 8: Create SLA commitments
        routing_decision.sla_commitments = await self._create_sla_commitments(
            requirements, selected_nodes
        )
        
        # Step 9: Log to transparency system
        if self.routing_config["transparency_logging"]:
            routing_decision.transparency_log_id = await self.transparency_logger.log_routing_decision(
                routing_decision, requirements, user_id, content_data
            )
        
        # Step 10: Store routing decision
        self.active_routes[routing_id] = routing_decision
        
        # Step 11: Update node load tracking
        for node in selected_nodes:
            self.node_load_tracking[node.node_id].append(datetime.now(UTC))
            node.current_workloads += 1
        
        logger.info(
            f"Routed workload {workload_id} to {len(selected_nodes)} nodes: "
            f"{[n.node_id for n in selected_nodes]} "
            f"(tier: {user_tier.value}, isolation: {isolation_level.value})"
        )
        
        return routing_decision
    
    async def _find_candidate_nodes(self, requirements: WorkloadRequirements) -> List[NodeCapabilities]:
        """Find nodes that can satisfy workload requirements"""
        
        candidates = []
        
        for node_id, node in self.available_nodes.items():
            # Check basic resource availability
            if (
                node.available_cpu_cores < requirements.cpu_cores or
                node.available_memory_gb < requirements.memory_gb or
                node.available_storage_gb < requirements.storage_gb
            ):
                continue
            
            # Check constitutional tier support
            if requirements.constitutional_tier not in node.supported_tiers:
                continue
            
            # Check isolation level capability
            if not self._node_supports_isolation(node, requirements.isolation_level):
                continue
            
            # Check trust score requirements
            if node.trust_score < requirements.min_trust_score:
                continue
            
            # Check latency requirements
            if node.average_latency_ms > requirements.max_latency_ms:
                continue
            
            # Check uptime requirements
            if node.uptime_percentage < requirements.required_uptime_percentage:
                continue
            
            # Check governance capabilities if required
            if requirements.requires_human_oversight and not node.human_oversight_available:
                continue
            
            # Check federated capabilities if needed
            if requirements.classification in [WorkloadClassification.FEDERATED_INFERENCE, WorkloadClassification.FEDERATED_TRAINING]:
                if not node.federated_coordination_support:
                    continue
            
            # Check current load
            if node.current_workloads >= node.max_concurrent_workloads:
                continue
            
            candidates.append(node)
        
        logger.debug(f"Found {len(candidates)} candidate nodes for workload requirements")
        return candidates
    
    def _node_supports_isolation(self, node: NodeCapabilities, required_level: IsolationLevel) -> bool:
        """Check if node supports required isolation level"""
        
        isolation_hierarchy = [
            IsolationLevel.PROCESS,
            IsolationLevel.CONTAINER,
            IsolationLevel.VM,
            IsolationLevel.PHYSICAL,
            IsolationLevel.SECURE_ENCLAVE,
        ]
        
        required_index = isolation_hierarchy.index(required_level)
        max_index = isolation_hierarchy.index(node.max_isolation_level)
        
        return max_index >= required_index
    
    async def _select_optimal_nodes(
        self, 
        candidates: List[NodeCapabilities], 
        requirements: WorkloadRequirements
    ) -> Tuple[List[NodeCapabilities], List[NodeCapabilities]]:
        """Select optimal nodes from candidates"""
        
        if not candidates:
            return [], []
        
        # Score each candidate
        scored_candidates = []
        for node in candidates:
            score = await self._calculate_node_score(node, requirements)
            scored_candidates.append((node, score))
        
        # Sort by score (highest first)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select primary nodes (top performers)
        num_needed = max(1, requirements.participants_needed)
        selected = [candidate[0] for candidate in scored_candidates[:num_needed]]
        
        # Select fallback nodes (next best)
        num_fallbacks = min(2, len(scored_candidates) - num_needed)
        fallbacks = [candidate[0] for candidate in scored_candidates[num_needed:num_needed + num_fallbacks]]
        
        return selected, fallbacks
    
    async def _calculate_node_score(self, node: NodeCapabilities, requirements: WorkloadRequirements) -> Decimal:
        """Calculate comprehensive score for node selection"""
        
        # Base capability score (30%)
        resource_score = min(
            node.available_cpu_cores / requirements.cpu_cores,
            node.available_memory_gb / requirements.memory_gb,
            node.available_storage_gb / requirements.storage_gb,
        )
        capability_score = min(Decimal("1.0"), resource_score) * Decimal("0.3")
        
        # Quality score (25%)
        quality_score = (
            node.trust_score * Decimal("0.4") +
            node.reputation_score * Decimal("0.3") +
            (node.uptime_percentage / Decimal("100")) * Decimal("0.3")
        ) * Decimal("0.25")
        
        # Performance score (20%)
        latency_score = max(Decimal("0"), Decimal("1") - (node.average_latency_ms / Decimal("1000")))
        load_factor = Decimal("1") - (Decimal(str(node.current_workloads)) / Decimal(str(node.max_concurrent_workloads)))
        performance_score = (latency_score * Decimal("0.6") + load_factor * Decimal("0.4")) * Decimal("0.2")
        
        # Constitutional compliance score (15%)
        compliance_score = node.constitutional_compliance_rating * Decimal("0.15")
        
        # Special capability bonuses (10%)
        bonus_score = Decimal("0")
        
        if requirements.requires_human_oversight and node.human_oversight_available:
            bonus_score += Decimal("0.03")
        
        if node.governance_engine_integrated:
            bonus_score += Decimal("0.02")
        
        if node.viewpoint_firewall_available and requirements.constitutional_tier in [ConstitutionalTier.SILVER, ConstitutionalTier.GOLD]:
            bonus_score += Decimal("0.02")
        
        if requirements.classification in [WorkloadClassification.FEDERATED_INFERENCE, WorkloadClassification.FEDERATED_TRAINING]:
            if node.federated_coordination_support:
                bonus_score += Decimal("0.03")
        
        total_score = capability_score + quality_score + performance_score + compliance_score + bonus_score
        
        return min(Decimal("1.0"), total_score)
    
    def _determine_isolation_level(
        self, 
        requirements: WorkloadRequirements, 
        selected_nodes: List[NodeCapabilities]
    ) -> IsolationLevel:
        """Determine appropriate isolation level for workload"""
        
        # Start with requirement minimum
        required_level = requirements.isolation_level
        
        # Upgrade based on workload classification
        if requirements.classification == WorkloadClassification.HIGH_RISK:
            required_level = max(required_level, IsolationLevel.CONTAINER)
        
        elif requirements.classification == WorkloadClassification.CONSTITUTIONAL_REVIEW:
            required_level = max(required_level, IsolationLevel.VM)
        
        elif requirements.classification in [WorkloadClassification.FEDERATED_TRAINING]:
            # Federated training needs strong isolation
            required_level = max(required_level, IsolationLevel.VM)
        
        # Upgrade based on tier
        if requirements.constitutional_tier == ConstitutionalTier.GOLD:
            required_level = max(required_level, IsolationLevel.CONTAINER)
        
        # Check if selected nodes support the level
        if selected_nodes:
            max_supported = min(node.max_isolation_level for node in selected_nodes)
            required_level = min(required_level, max_supported)
        
        return required_level
    
    def _generate_routing_rationale(
        self, 
        requirements: WorkloadRequirements, 
        selected_nodes: List[NodeCapabilities]
    ) -> str:
        """Generate human-readable rationale for routing decision"""
        
        if not selected_nodes:
            return "No suitable nodes available"
        
        rationale_parts = [
            f"Selected {len(selected_nodes)} node(s) for {requirements.constitutional_tier.value} tier workload",
        ]
        
        if requirements.classification != WorkloadClassification.BASIC:
            rationale_parts.append(f"Workload classified as {requirements.classification.value}")
        
        if requirements.isolation_level != IsolationLevel.PROCESS:
            rationale_parts.append(f"Using {requirements.isolation_level.value} isolation")
        
        # Add node quality summary
        avg_trust = sum(node.trust_score for node in selected_nodes) / len(selected_nodes)
        rationale_parts.append(f"Average node trust score: {float(avg_trust):.2f}")
        
        if requirements.requires_human_oversight:
            rationale_parts.append("Human oversight available")
        
        return ". ".join(rationale_parts)
    
    def _calculate_routing_confidence(
        self, 
        selected_nodes: List[NodeCapabilities], 
        requirements: WorkloadRequirements
    ) -> Decimal:
        """Calculate confidence in routing decision"""
        
        if not selected_nodes:
            return Decimal("0")
        
        # Base confidence from node quality
        node_quality_scores = [
            (node.trust_score + node.reputation_score + node.uptime_percentage / Decimal("100")) / Decimal("3")
            for node in selected_nodes
        ]
        avg_quality = sum(node_quality_scores) / len(node_quality_scores)
        
        # Adjust for requirements match
        requirements_match = Decimal("1.0")
        
        # Check if we have enough nodes for federated workloads
        if requirements.participants_needed > len(selected_nodes):
            requirements_match *= Decimal("0.7")
        
        # Check governance capabilities
        if requirements.requires_human_oversight:
            oversight_available = any(node.human_oversight_available for node in selected_nodes)
            if not oversight_available:
                requirements_match *= Decimal("0.5")
        
        # Check constitutional compliance
        avg_compliance = sum(node.constitutional_compliance_rating for node in selected_nodes) / len(selected_nodes)
        
        confidence = (avg_quality + requirements_match + avg_compliance) / Decimal("3")
        
        return min(Decimal("1.0"), confidence)
    
    def _estimate_completion_time(
        self, 
        requirements: WorkloadRequirements, 
        selected_nodes: List[NodeCapabilities]
    ) -> datetime:
        """Estimate workload completion time"""
        
        if not selected_nodes:
            return datetime.now(UTC) + timedelta(hours=1)  # Default fallback
        
        # Base execution time
        base_duration = requirements.duration_hours
        
        # Adjust for node performance
        avg_performance = sum(
            Decimal("1") / max(node.average_latency_ms, Decimal("1")) for node in selected_nodes
        ) / len(selected_nodes)
        
        performance_multiplier = Decimal("1") / (avg_performance * Decimal("1000"))
        
        # Adjust for isolation overhead
        isolation_overhead = {
            IsolationLevel.PROCESS: Decimal("1.0"),
            IsolationLevel.CONTAINER: Decimal("1.1"),
            IsolationLevel.VM: Decimal("1.3"),
            IsolationLevel.PHYSICAL: Decimal("1.0"),
            IsolationLevel.SECURE_ENCLAVE: Decimal("1.5"),
        }
        
        isolation_level = self._determine_isolation_level(requirements, selected_nodes)
        overhead_multiplier = isolation_overhead.get(isolation_level, Decimal("1.2"))
        
        # Constitutional overhead
        constitutional_overhead = {
            ConstitutionalTier.BRONZE: Decimal("1.05"),
            ConstitutionalTier.SILVER: Decimal("1.15"),
            ConstitutionalTier.GOLD: Decimal("1.3"),
        }
        
        const_multiplier = constitutional_overhead.get(requirements.constitutional_tier, Decimal("1.1"))
        
        # Calculate total estimated time
        estimated_hours = base_duration * performance_multiplier * overhead_multiplier * const_multiplier
        
        return datetime.now(UTC) + timedelta(hours=float(estimated_hours))
    
    async def _create_sla_commitments(
        self, 
        requirements: WorkloadRequirements, 
        selected_nodes: List[NodeCapabilities]
    ) -> Dict[str, Any]:
        """Create SLA commitments for routing decision"""
        
        if not selected_nodes:
            return {}
        
        # Calculate guarantees based on selected nodes
        min_uptime = min(node.uptime_percentage for node in selected_nodes)
        max_latency = max(node.average_latency_ms for node in selected_nodes)
        avg_trust = sum(node.trust_score for node in selected_nodes) / len(selected_nodes)
        
        commitments = {
            "guaranteed_uptime_percentage": float(min_uptime),
            "max_latency_ms": float(max_latency),
            "isolation_level": self._determine_isolation_level(requirements, selected_nodes).value,
            "constitutional_compliance": True,
            "transparency_logging": True,
            "audit_trail_available": True,
            "average_node_trust_score": float(avg_trust),
        }
        
        # Add tier-specific commitments
        tier_capabilities = self.tier_manager.get_tier_capabilities(requirements.constitutional_tier)
        commitments.update({
            "constitutional_compliance_sla": float(tier_capabilities.constitutional_compliance_sla),
            "safety_check_frequency": tier_capabilities.constitutional_constraints.safety_check_frequency,
            "harm_detection_threshold": float(tier_capabilities.constitutional_constraints.harm_detection_threshold),
        })
        
        # Add federated-specific commitments
        if requirements.classification in [WorkloadClassification.FEDERATED_INFERENCE, WorkloadClassification.FEDERATED_TRAINING]:
            federated_capable_nodes = sum(1 for node in selected_nodes if node.federated_coordination_support)
            commitments.update({
                "federated_coordination_available": federated_capable_nodes >= requirements.participants_needed,
                "participants_allocated": len(selected_nodes),
                "secure_multi_party_computation": any(node.secure_multi_party_computation for node in selected_nodes),
                "differential_privacy_support": any(node.differential_privacy_support for node in selected_nodes),
            })
        
        return commitments
    
    async def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        
        total_routes = len(self.routing_history) + len(self.active_routes)
        successful_routes = len([r for r in self.routing_history if r.selected_nodes])
        
        # Tier distribution
        tier_distribution = {}
        all_decisions = list(self.active_routes.values()) + self.routing_history
        for decision in all_decisions:
            # Extract tier from routing rationale (simplified)
            for tier in ConstitutionalTier:
                if tier.value in decision.routing_rationale.lower():
                    tier_distribution[tier.value] = tier_distribution.get(tier.value, 0) + 1
                    break
        
        # Node utilization
        node_utilization = {}
        for node_id, node in self.available_nodes.items():
            utilization_rate = node.current_workloads / max(node.max_concurrent_workloads, 1)
            node_utilization[node_id] = {
                "current_workloads": node.current_workloads,
                "max_workloads": node.max_concurrent_workloads,
                "utilization_rate": float(utilization_rate),
                "trust_score": float(node.trust_score),
                "supported_tiers": [t.value for t in node.supported_tiers],
            }
        
        return {
            "routing_overview": {
                "total_routing_decisions": total_routes,
                "successful_routes": successful_routes,
                "success_rate": successful_routes / max(total_routes, 1),
                "active_routes": len(self.active_routes),
            },
            "constitutional_distribution": {
                "tier_distribution": tier_distribution,
                "transparency_logs_generated": len(self.transparency_logger.log_entries),
            },
            "node_utilization": node_utilization,
            "routing_performance": {
                "average_confidence": float(
                    sum(d.confidence_score for d in all_decisions) / max(len(all_decisions), 1)
                ),
                "nodes_registered": len(self.available_nodes),
                "average_nodes_per_route": sum(len(d.selected_nodes) for d in all_decisions) / max(len(all_decisions), 1),
            },
        }


# Integration functions

async def create_constitutional_workload_router(
    tier_manager: Optional[ConstitutionalTierManager] = None,
    governance_engine: Optional[ConstitutionalGovernanceEngine] = None,
) -> ConstitutionalWorkloadRouter:
    """Create constitutional workload router with dependencies"""
    
    if not tier_manager:
        tier_manager = ConstitutionalTierManager()
    
    router = ConstitutionalWorkloadRouter(tier_manager, governance_engine)
    
    logger.info("Constitutional workload router created and ready")
    return router


def create_workload_requirements(
    cpu_cores: float = 1.0,
    memory_gb: float = 1.0,
    classification: str = "basic",
    constitutional_tier: str = "bronze",
    **kwargs
) -> WorkloadRequirements:
    """Create workload requirements from basic parameters"""
    
    return WorkloadRequirements(
        cpu_cores=Decimal(str(cpu_cores)),
        memory_gb=Decimal(str(memory_gb)),
        classification=WorkloadClassification(classification),
        constitutional_tier=ConstitutionalTier(constitutional_tier),
        **kwargs
    )


if __name__ == "__main__":
    # Demo constitutional workload routing
    import json
    
    async def demo_constitutional_routing():
        # Create router
        tier_manager = ConstitutionalTierManager()
        router = ConstitutionalWorkloadRouter(tier_manager)
        
        # Register some demo nodes
        bronze_node = NodeCapabilities(
            node_id="node_bronze_001",
            available_cpu_cores=Decimal("8"),
            available_memory_gb=Decimal("16"),
            available_storage_gb=Decimal("100"),
            supported_tiers={ConstitutionalTier.BRONZE},
            max_isolation_level=IsolationLevel.CONTAINER,
            constitutional_compliance_rating=Decimal("0.8"),
            trust_score=Decimal("0.7"),
            reputation_score=Decimal("0.75"),
            governance_engine_integrated=False,
        )
        
        gold_node = NodeCapabilities(
            node_id="node_gold_001",
            available_cpu_cores=Decimal("32"),
            available_memory_gb=Decimal("64"),
            available_storage_gb=Decimal("500"),
            supported_tiers={ConstitutionalTier.BRONZE, ConstitutionalTier.SILVER, ConstitutionalTier.GOLD},
            max_isolation_level=IsolationLevel.VM,
            constitutional_compliance_rating=Decimal("0.95"),
            trust_score=Decimal("0.9"),
            reputation_score=Decimal("0.88"),
            governance_engine_integrated=True,
            human_oversight_available=True,
            viewpoint_firewall_available=True,
        )
        
        await router.register_node(bronze_node)
        await router.register_node(gold_node)
        
        # Create test workload requirements
        requirements = WorkloadRequirements(
            cpu_cores=Decimal("4"),
            memory_gb=Decimal("8"),
            classification=WorkloadClassification.SENSITIVE,
            constitutional_tier=ConstitutionalTier.SILVER,
            isolation_level=IsolationLevel.CONTAINER,
            requires_human_oversight=False,
        )
        
        # Route workload
        decision = await router.route_workload(
            workload_id="demo_workload_001",
            user_id="demo_user",
            requirements=requirements,
            content_data={"text": "Demo content for routing", "risk_score": 0.3}
        )
        
        print("Constitutional Routing Decision:")
        print(f"Selected Nodes: {decision.selected_nodes}")
        print(f"Rationale: {decision.routing_rationale}")
        print(f"Confidence: {float(decision.confidence_score):.2f}")
        print(f"Isolation Level: {decision.isolation_level_assigned.value}")
        print(f"SLA Commitments: {json.dumps(decision.sla_commitments, indent=2)}")
        
        # Get routing statistics
        stats = await router.get_routing_statistics()
        print(f"\nRouting Statistics:")
        print(json.dumps(stats, indent=2))
        
        # Generate transparency report
        transparency_report = await router.transparency_logger.generate_transparency_report()
        print(f"\nTransparency Report:")
        print(json.dumps(transparency_report, indent=2))
    
    asyncio.run(demo_constitutional_routing())