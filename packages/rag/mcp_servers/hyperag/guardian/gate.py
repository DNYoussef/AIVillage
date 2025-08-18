"""Guardian Gate validation layer for HypeRAG repairs and creative bridges."""

import asyncio
import copy
import datetime
import hashlib
import json
import pathlib
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

import yaml

from . import audit
from .metrics import get_guardian_metrics

Decision = Literal["APPLY", "QUARANTINE", "REJECT"]

# Type aliases for imports that may not exist yet
try:
    from ..gdc.specs import Violation
    from ..repair.innovator_agent import RepairOperation
except ImportError:
    # Fallback types for testing
    RepairOperation = Any
    Violation = Any


# CreativeBridge type - define locally since it's referenced in requirements
@dataclass
class CreativeBridge:
    """Creative bridge structure for validation."""

    id: str
    confidence: float = 0.7
    bridge_type: str = "semantic"
    source_nodes: list[str] = None
    target_nodes: list[str] = None

    def __post_init__(self):
        if self.source_nodes is None:
            self.source_nodes = []
        if self.target_nodes is None:
            self.target_nodes = []


class GuardianGate:
    """Core decision engine for validating repairs and creative bridges."""

    def __init__(self, policy_path: str | None = None) -> None:
        if policy_path is None:
            policy_path = pathlib.Path(__file__).parent / "policies.yaml"

        self.policy_path = pathlib.Path(policy_path)
        self.policies = self._load_policies()
        self.metrics = get_guardian_metrics()

        # Initialize policy info in metrics
        self.metrics.set_policy_info(
            {
                "version": "1.0",
                "confidence_threshold": self.policies.get("thresholds", {}).get("apply", 0.8),
                "quarantine_threshold": self.policies.get("thresholds", {}).get("quarantine", 0.4),
            }
        )

    def _load_policies(self) -> dict[str, Any]:
        """Load policies from YAML configuration file."""
        try:
            with open(self.policy_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {
                "weights": {
                    "structural_fix": 0.4,
                    "domain_veracity": 0.4,
                    "evidence_strength": 0.2,
                },
                "thresholds": {"apply": 0.80, "quarantine": 0.40},
                "domain_heuristics": {
                    "medical": {
                        "must_preserve_edges": ["TAKES", "DIAGNOSED_WITH"],
                        "forbidden_deletes": ["ALLERGIC_TO"],
                    }
                },
            }

    async def evaluate_repair(self, proposals: list[RepairOperation], violation: Violation) -> Decision:
        """Validate Innovator Repair set and return decision."""
        time.time()
        getattr(violation, "domain", "general")

        try:
            # 1. Calculate semantic utility by applying proposals in-memory
            structural_fix = await self._calculate_structural_fix(proposals, violation)
        except Exception as e:
            structural_fix = 0.0
            logger.warning(f"Failed to calculate structural fix: {e}")

        # 2. External fact check (stub implementation)
        domain_veracity = await self._external_fact_check(proposals, violation)

        # 3. Calculate evidence strength
        evidence_strength = self._calculate_evidence_strength(proposals)

        # 4. Calculate impact score
        weights = self.policies["weights"]
        score = (
            structural_fix * weights["structural_fix"]
            + domain_veracity * weights["domain_veracity"]
            + evidence_strength * weights["evidence_strength"]
        )

        # 5. Apply decision thresholds with severity consideration
        severity = getattr(violation, "severity", "medium")
        decision = self._apply_thresholds(score, severity)

        # 6. Generate rationale and log audit record
        rationale = self._generate_rationale(decision, score, structural_fix, domain_veracity, evidence_strength)

        # 7. Log audit record
        audit_record = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "decision": decision,
            "gdc_id": getattr(violation, "id", "UNKNOWN"),
            "score": score,
            "proposals": [self._serialize_proposal(p) for p in proposals],
            "rationale": rationale,
            "components": {
                "structural_fix": structural_fix,
                "domain_veracity": domain_veracity,
                "evidence_strength": evidence_strength,
            },
        }
        audit.log(audit_record)

        return decision

    async def evaluate_creative(self, bridge: CreativeBridge) -> Decision:
        """Validate creative bridge before exposing to agents."""
        # Simplified evaluation for creative bridges
        structural_fix = 0.7  # Assume moderate structural value
        domain_veracity = await self._check_bridge_plausibility(bridge)
        evidence_strength = 0.6  # Assume moderate evidence

        weights = self.policies["weights"]
        score = (
            structural_fix * weights["structural_fix"]
            + domain_veracity * weights["domain_veracity"]
            + evidence_strength * weights["evidence_strength"]
        )

        decision = self._apply_thresholds(score, "medium")
        rationale = f"Creative bridge plausibility: {domain_veracity:.2f}, score: {score:.2f}"

        # Log audit record
        audit_record = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "decision": decision,
            "bridge_id": getattr(bridge, "id", "UNKNOWN"),
            "score": score,
            "rationale": rationale,
            "components": {
                "structural_fix": structural_fix,
                "domain_veracity": domain_veracity,
                "evidence_strength": evidence_strength,
            },
        }
        audit.log(audit_record)

        return decision

    async def _calculate_structural_fix(self, proposals: list[Any], violation: Any) -> float:
        """Calculate how well proposals fix the structural issue."""
        # Apply proposals in-memory to a copy of the violating subgraph
        original_subgraph = getattr(violation, "subgraph", {"nodes": [], "edges": []})
        simulated_graph = copy.deepcopy(original_subgraph)

        # Check for forbidden operations based on domain heuristics
        domain = getattr(violation, "domain", "general")
        domain_rules = self.policies.get("domain_heuristics", {}).get(domain, {})
        forbidden_deletes = domain_rules.get("forbidden_deletes", [])
        domain_rules.get("must_preserve_edges", [])

        penalty = 0.0
        for proposal in proposals:
            # Check for forbidden edge deletions
            if (
                hasattr(proposal, "operation_type")
                and proposal.operation_type == "delete_edge"
                and hasattr(proposal, "edge_type")
                and proposal.edge_type in forbidden_deletes
            ):
                penalty += 0.5  # Heavy penalty for forbidden operations

            # Apply proposal to simulated graph
            self._simulate_proposal(simulated_graph, proposal)

        # Calculate edge preservation score
        original_edges = len(original_subgraph.get("edges", []))
        modified_edges = len(simulated_graph.get("edges", []))

        if original_edges == 0:
            preservation_score = 1.0
        else:
            preservation_score = modified_edges / original_edges

        # Domain-specific scoring
        domain_score = self._score_domain_heuristics(simulated_graph, domain)

        # Combine scores with penalty
        base_score = preservation_score * 0.6 + domain_score * 0.4
        final_score = max(0.0, base_score - penalty)

        return min(1.0, final_score)

    def _simulate_proposal(self, graph: dict[str, Any], proposal: Any) -> None:
        """Apply a single proposal to the simulated graph."""
        if not hasattr(proposal, "operation_type"):
            return

        operation = proposal.operation_type

        if operation == "delete_edge" and hasattr(proposal, "target_id"):
            edges = graph.get("edges", [])
            graph["edges"] = [e for e in edges if e.get("id") != proposal.target_id]

        elif operation == "delete_node" and hasattr(proposal, "target_id"):
            nodes = graph.get("nodes", [])
            graph["nodes"] = [n for n in nodes if n.get("id") != proposal.target_id]
            # Remove connected edges
            edges = graph.get("edges", [])
            graph["edges"] = [
                e
                for e in edges
                if (e.get("startNode") != proposal.target_id and e.get("endNode") != proposal.target_id)
            ]

        elif operation == "add_edge":
            new_edge = {
                "id": getattr(proposal, "target_id", str(uuid.uuid4())),
                "startNode": getattr(proposal, "source_id", None),
                "endNode": getattr(proposal, "dest_id", None),
                "type": getattr(proposal, "relationship_type", "RELATED"),
                "properties": getattr(proposal, "properties", {}),
            }
            graph.setdefault("edges", []).append(new_edge)

        elif operation == "add_node":
            new_node = {
                "id": getattr(proposal, "target_id", str(uuid.uuid4())),
                "labels": [getattr(proposal, "node_type", "Unknown")],
                "properties": getattr(proposal, "properties", {}),
            }
            graph.setdefault("nodes", []).append(new_node)

    def _score_domain_heuristics(self, graph: dict[str, Any], domain: str) -> float:
        """Score graph based on domain-specific heuristics."""
        if domain == "medical":
            return self._score_medical_heuristics(graph)
        return self._score_general_heuristics(graph)

    def _score_medical_heuristics(self, graph: dict[str, Any]) -> float:
        """Score medical domain coherence."""
        score = 0.5  # Base score
        factors = []

        # Check for critical medical relationships
        edges = graph.get("edges", [])
        medical_rels = ["PRESCRIBES", "ALLERGIC_TO", "TREATS", "CONTRAINDICATED_WITH"]

        for edge in edges:
            edge_type = edge.get("type", "")
            if edge_type in medical_rels:
                factors.append(0.8)  # Bonus for maintaining critical relationships

                # Check for required properties
                props = edge.get("properties", {})
                if edge_type == "PRESCRIBES" and "dosage" in props:
                    factors.append(0.2)
                if edge_type == "ALLERGIC_TO" and "severity" in props:
                    factors.append(0.2)

        if factors:
            score += sum(factors) / len(factors) * 0.5

        return min(1.0, score)

    def _score_general_heuristics(self, graph: dict[str, Any]) -> float:
        """Score general domain coherence."""
        score = 0.5  # Base score
        factors = []

        # Check for confidence and source information
        for node in graph.get("nodes", []):
            props = node.get("properties", {})
            if "confidence" in props and props["confidence"] > 0.7:
                factors.append(0.3)
            if "source" in props:
                factors.append(0.2)

        for edge in graph.get("edges", []):
            props = edge.get("properties", {})
            if "confidence" in props and props["confidence"] > 0.7:
                factors.append(0.3)
            if "timestamp" in props:
                factors.append(0.1)

        if factors:
            score += sum(factors) / len(factors) * 0.5

        return min(1.0, score)

    async def _external_fact_check(self, proposals: list[Any], violation: Any) -> float:
        """Call domain adapter for external verification."""
        domain = getattr(violation, "domain", "general")

        try:
            # Simulate API call with timeout
            await asyncio.sleep(0.001)  # Simulate network latency

            # Domain-specific fact checking
            if domain == "medical":
                return await self._check_medical_facts(proposals)
            if domain == "general":
                return 0.7  # Moderate confidence for general domain
            return 0.5  # Unknown domain

        except Exception:
            return 0.5  # API unreachable → domain_veracity = 0.5 ("unknown")

    async def _check_medical_facts(self, proposals: list[Any]) -> float:
        """Check medical facts using domain APIs (stub)."""
        # Mock medical fact checking
        confidence_sum = 0.0
        checks = 0

        for proposal in proposals:
            if hasattr(proposal, "relationship_type"):
                rel_type = proposal.relationship_type
                if rel_type == "ALLERGIC_TO":
                    confidence_sum += 0.9  # High confidence for allergy relationships
                    checks += 1
                elif rel_type == "PRESCRIBES":
                    confidence_sum += 0.8  # Good confidence for prescriptions
                    checks += 1
                elif rel_type == "CONTRAINDICATED_WITH":
                    confidence_sum += 0.85  # High confidence for contraindications
                    checks += 1

        if checks == 0:
            return 0.7  # Default medical confidence

        return confidence_sum / checks

    async def _check_bridge_plausibility(self, bridge: Any) -> float:
        """Check plausibility of creative bridge via external API."""
        try:
            # Simulate plausibility check
            await asyncio.sleep(0.001)

            # Mock plausibility based on bridge properties
            if hasattr(bridge, "confidence"):
                return min(0.9, bridge.confidence + 0.1)

            return 0.8  # Default high plausibility

        except Exception:
            return 0.3  # Low plausibility if check fails

    def _calculate_evidence_strength(self, proposals: list[Any]) -> float:
        """Calculate evidence strength of proposals."""
        if not proposals:
            return 0.0

        strength_factors = []

        for proposal in proposals:
            # Base confidence from proposal
            if hasattr(proposal, "confidence"):
                strength_factors.append(proposal.confidence)
            else:
                strength_factors.append(0.5)  # Default confidence

            # Rationale quality bonus
            if hasattr(proposal, "rationale") and proposal.rationale:
                rationale_bonus = min(0.3, len(proposal.rationale) / 100.0)
                strength_factors.append(rationale_bonus)

        return sum(strength_factors) / len(strength_factors) if strength_factors else 0.5

    def _apply_thresholds(self, score: float, severity: str) -> Decision:
        """Apply decision thresholds based on score and severity."""
        thresholds = self.policies["thresholds"]

        # High severity violations need higher confidence for APPLY
        if severity == "high" and score >= 0.8:
            return "APPLY"
        if severity == "high" and score >= thresholds["quarantine"]:
            return "QUARANTINE"
        if score >= thresholds["apply"]:
            return "APPLY"
        if score >= thresholds["quarantine"]:
            return "QUARANTINE"
        return "REJECT"

    def _generate_rationale(
        self,
        decision: Decision,
        score: float,
        structural_fix: float,
        domain_veracity: float,
        evidence_strength: float,
    ) -> str:
        """Generate human-readable rationale for the decision."""
        components = []

        if structural_fix < 0.3:
            components.append("poor structural fix")
        elif structural_fix > 0.8:
            components.append("excellent structural fix")
        else:
            components.append("moderate structural fix")

        if domain_veracity < 0.3:
            components.append("low domain confidence")
        elif domain_veracity == 0.5:
            components.append("external API unknown")
        elif domain_veracity > 0.8:
            components.append("high domain confidence")
        else:
            components.append("moderate domain confidence")

        if evidence_strength < 0.3:
            components.append("weak evidence")
        elif evidence_strength > 0.7:
            components.append("strong evidence")
        else:
            components.append("moderate evidence")

        rationale = "; ".join(components)
        return f"{decision.lower()}: {rationale} (score: {score:.2f})"

    def _serialize_proposal(self, proposal: Any) -> dict[str, Any]:
        """Serialize proposal for audit logging."""
        return {
            "type": type(proposal).__name__,
            "operation": getattr(proposal, "operation_type", "unknown"),
            "target_id": getattr(proposal, "target_id", "unknown"),
            "confidence": getattr(proposal, "confidence", 0.5),
            "details": str(proposal),
        }

    def validate_adapter(self, adapter_info: dict[str, Any]) -> dict[str, Any]:
        """Validate a LoRA adapter registration."""
        domain = adapter_info.get("domain", "general")
        metrics = adapter_info.get("metrics", {})

        # Check adapter quality metrics
        accuracy = metrics.get("accuracy", 0)
        perplexity = metrics.get("perplexity", float("inf"))

        # Domain-specific thresholds
        min_accuracy = self.policies.get("adapter_thresholds", {}).get(domain, {}).get("min_accuracy", 0.7)
        max_perplexity = self.policies.get("adapter_thresholds", {}).get(domain, {}).get("max_perplexity", 100)

        # Make decision
        if accuracy < min_accuracy or perplexity > max_perplexity:
            decision = "REJECT"
            reason = f"Metrics below threshold: accuracy={accuracy:.3f} (min={min_accuracy}), perplexity={perplexity:.1f} (max={max_perplexity})"
            confidence = 0.9
        elif accuracy < min_accuracy + 0.1:  # Close to threshold
            decision = "QUARANTINE"
            reason = "Metrics close to minimum thresholds"
            confidence = 0.6
        else:
            decision = "APPLY"
            reason = "Adapter meets quality standards"
            confidence = 0.95

        # Generate signature for approved adapters
        signature = None
        if decision == "APPLY":
            signature = self._generate_signature(adapter_info)

        # Update metrics
        self.metrics.record_validation(domain, decision, "adapter")

        return {
            "decision": decision,
            "confidence": confidence,
            "reason": reason,
            "signature": signature,
        }

    def _generate_signature(self, data: dict[str, Any]) -> str:
        """Generate a cryptographic signature for approved items."""
        # Simple signature using SHA256 - in production use proper signing
        content = json.dumps(data, sort_keys=True)
        signature = hashlib.sha256(content.encode()).hexdigest()
        return f"guardian_v1:{signature}"

    def verify_signature(self, data: dict[str, Any], signature: str) -> bool:
        """Verify a Guardian signature."""
        if not signature or not signature.startswith("guardian_v1:"):
            return False

        expected_signature = self._generate_signature(data)
        return signature == expected_signature

    async def validate_query_result(self, query_context: dict[str, Any], result: dict[str, Any]) -> Decision:
        """Validate query results before returning to user."""
        domain = query_context.get("domain", "general")
        confidence = result.get("confidence", 0.5)

        # High-risk domains require higher confidence
        high_risk_domains = self.policies.get("high_risk_domains", ["medical", "financial"])

        if domain in high_risk_domains and confidence < 0.7:
            decision = "QUARANTINE"
            reason = f"High-risk domain {domain} requires manual review for low confidence results"
        elif confidence < self.policies["thresholds"]["quarantine"]:
            decision = "REJECT"
            reason = f"Result confidence {confidence:.2f} below minimum threshold"
        else:
            decision = "APPLY"
            reason = "Result meets quality standards"

        # Update metrics
        self.metrics.record_validation(domain, decision, "query")

        # Log audit record
        audit_record = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "decision": decision,
            "type": "query_validation",
            "domain": domain,
            "confidence": confidence,
            "rationale": reason,
        }
        audit.log(audit_record)

        return decision
