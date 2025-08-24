"""Unit tests for Guardian Gate validation layer."""

import pathlib
import tempfile
from unittest.mock import patch

from mcp_servers.hyperag.guardian import audit
from mcp_servers.hyperag.guardian.gate import GuardianGate
import pytest


class MockProposal:
    """Mock repair proposal for testing."""

    def __init__(
        self,
        operation_type: str,
        target_id: str,
        confidence: float = 0.8,
        relationship_type: str | None = None,
        rationale: str = "Test rationale",
        edge_type: str | None = None,
    ):
        self.operation_type = operation_type
        self.target_id = target_id
        self.confidence = confidence
        self.relationship_type = relationship_type
        self.rationale = rationale
        self.edge_type = edge_type
        self.properties = {}

    def __str__(self):
        return f"MockProposal({self.operation_type}, {self.target_id})"


class MockViolation:
    """Mock GDC violation for testing."""

    def __init__(
        self,
        id: str = "GDC_TEST",
        severity: str = "medium",
        domain: str = "general",
        subgraph: dict | None = None,
    ):
        self.id = id
        self.severity = severity
        self.domain = domain
        self.subgraph = subgraph or {"nodes": [], "edges": []}


class MockCreativeBridge:
    """Mock creative bridge for testing."""

    def __init__(self, id: str = "BRIDGE_TEST", confidence: float = 0.7):
        self.id = id
        self.confidence = confidence


@pytest.fixture
def temp_policies():
    """Create temporary policies file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(
            """
weights:
  structural_fix: 0.4
  domain_veracity: 0.4
  evidence_strength: 0.2

thresholds:
  apply: 0.80
  quarantine: 0.40

domain_heuristics:
  medical:
    must_preserve_edges: ["TAKES", "DIAGNOSED_WITH"]
    forbidden_deletes: ["ALLERGIC_TO"]
  general:
    must_preserve_edges: ["IDENTITY"]
    forbidden_deletes: []
"""
        )
        f.flush()
        yield f.name

    # Cleanup
    pathlib.Path(f.name).unlink()


@pytest.fixture
def guardian_gate(temp_policies):
    """Create GuardianGate instance with test policies."""
    return GuardianGate(policy_path=temp_policies)


@pytest.fixture
def medical_subgraph():
    """Medical domain subgraph for testing."""
    return {
        "nodes": [
            {
                "id": "patient1",
                "labels": ["Patient"],
                "properties": {"name": "John Doe"},
            },
            {"id": "drug1", "labels": ["Drug"], "properties": {"name": "Aspirin"}},
            {"id": "allergy1", "labels": ["Allergy"], "properties": {"type": "Drug"}},
        ],
        "edges": [
            {
                "id": "edge1",
                "startNode": "patient1",
                "endNode": "drug1",
                "type": "TAKES",
                "properties": {"dosage": "100mg"},
            },
            {
                "id": "edge2",
                "startNode": "patient1",
                "endNode": "allergy1",
                "type": "ALLERGIC_TO",
                "properties": {"severity": "mild"},
            },
        ],
    }


class TestGuardianGate:
    """Test cases for GuardianGate validation logic."""

    @pytest.mark.asyncio
    async def test_apply_path_high_severity_strong_fix(self, guardian_gate):
        """Test APPLY decision for high-severity violation with strong structural fix."""
        # Create proposals with high confidence
        proposals = [
            MockProposal(
                "add_edge",
                "new_edge_1",
                confidence=0.9,
                relationship_type="DIAGNOSES",
                rationale="Well-documented diagnosis",
            )
        ]

        # High-severity violation
        violation = MockViolation(id="GDC_MED_ALLERGY", severity="high", domain="medical")

        # Mock external fact-checking to return high confidence
        with patch.object(guardian_gate, "_external_fact_check", return_value=1.0):
            decision = await guardian_gate.evaluate_repair(proposals, violation)

        assert decision == "APPLY"

    @pytest.mark.asyncio
    async def test_quarantine_path_medium_score(self, guardian_gate):
        """Test QUARANTINE decision for medium score (0.5)."""
        # Create proposals with medium confidence
        proposals = [
            MockProposal(
                "update_attr",
                "node1",
                confidence=0.6,
                rationale="Moderate confidence update",
            )
        ]

        violation = MockViolation(severity="medium")

        # Mock external fact-checking to return medium confidence
        with patch.object(guardian_gate, "_external_fact_check", return_value=0.5):
            decision = await guardian_gate.evaluate_repair(proposals, violation)

        assert decision == "QUARANTINE"

        # Verify audit file was created
        recent_records = audit.get_recent_records(hours=1, limit=10)
        assert len(recent_records) > 0
        assert recent_records[0]["decision"] == "QUARANTINE"

    @pytest.mark.asyncio
    async def test_reject_path_forbidden_delete(self, guardian_gate, medical_subgraph):
        """Test REJECT decision when proposal deletes forbidden edge."""
        # Create proposal that deletes ALLERGIC_TO edge (forbidden in medical domain)
        proposals = [
            MockProposal(
                "delete_edge",
                "edge2",
                confidence=0.8,
                edge_type="ALLERGIC_TO",
                rationale="Remove allergy",
            )
        ]

        violation = MockViolation(
            id="GDC_MED_ALLERGY",
            severity="high",
            domain="medical",
            subgraph=medical_subgraph,
        )

        decision = await guardian_gate.evaluate_repair(proposals, violation)

        # Should be REJECT due to forbidden operation penalty
        assert decision == "REJECT"

    @pytest.mark.asyncio
    async def test_creative_bridge_validation(self, guardian_gate):
        """Test creative bridge validation."""
        bridge = MockCreativeBridge(id="CREATIVE_BRIDGE_1", confidence=0.8)

        decision = await guardian_gate.evaluate_creative(bridge)

        # Should be APPLY or QUARANTINE based on scoring
        assert decision in ["APPLY", "QUARANTINE", "REJECT"]

        # Verify audit record was created
        recent_records = audit.get_recent_records(hours=1, limit=10)
        bridge_records = [r for r in recent_records if "bridge_id" in r]
        assert len(bridge_records) > 0

    @pytest.mark.asyncio
    async def test_creative_bridge_fails_plausibility(self, guardian_gate):
        """Test creative bridge that fails plausibility check."""
        bridge = MockCreativeBridge(id="IMPLAUSIBLE_BRIDGE", confidence=0.2)

        # Mock plausibility check to fail
        with patch.object(guardian_gate, "_check_bridge_plausibility", return_value=0.1):
            decision = await guardian_gate.evaluate_creative(bridge)

        assert decision == "REJECT"

    def test_simulation_edge_deletion(self, guardian_gate, medical_subgraph):
        """Test graph simulation for edge deletion."""
        original_edges = len(medical_subgraph["edges"])

        # Create copy and simulate deletion
        simulated = medical_subgraph.copy()
        simulated["edges"] = medical_subgraph["edges"].copy()

        proposal = MockProposal("delete_edge", "edge1")
        guardian_gate._simulate_proposal(simulated, proposal)

        # Should have one fewer edge
        assert len(simulated["edges"]) == original_edges - 1
        assert not any(e["id"] == "edge1" for e in simulated["edges"])

    def test_simulation_node_deletion(self, guardian_gate, medical_subgraph):
        """Test graph simulation for node deletion."""
        original_nodes = len(medical_subgraph["nodes"])
        len(medical_subgraph["edges"])

        # Simulate deletion of patient1 node
        simulated = medical_subgraph.copy()
        simulated["nodes"] = medical_subgraph["nodes"].copy()
        simulated["edges"] = medical_subgraph["edges"].copy()

        proposal = MockProposal("delete_node", "patient1")
        guardian_gate._simulate_proposal(simulated, proposal)

        # Should have one fewer node
        assert len(simulated["nodes"]) == original_nodes - 1
        assert not any(n["id"] == "patient1" for n in simulated["nodes"])

        # Should remove connected edges
        remaining_edges = [e for e in simulated["edges"] if e["startNode"] != "patient1" and e["endNode"] != "patient1"]
        assert len(simulated["edges"]) == len(remaining_edges)

    def test_simulation_edge_addition(self, guardian_gate, medical_subgraph):
        """Test graph simulation for edge addition."""
        original_edges = len(medical_subgraph["edges"])

        simulated = medical_subgraph.copy()
        simulated["edges"] = medical_subgraph["edges"].copy()

        proposal = MockProposal("add_edge", "new_edge", relationship_type="TREATS")
        proposal.source_id = "drug1"
        proposal.dest_id = "patient1"

        guardian_gate._simulate_proposal(simulated, proposal)

        # Should have one more edge
        assert len(simulated["edges"]) == original_edges + 1
        assert any(e["id"] == "new_edge" for e in simulated["edges"])

    def test_medical_heuristics_scoring(self, guardian_gate, medical_subgraph):
        """Test medical domain heuristics scoring."""
        score = guardian_gate._score_medical_heuristics(medical_subgraph)

        # Should get bonus points for TAKES and ALLERGIC_TO relationships
        assert score > 0.5  # Better than base score
        assert score <= 1.0

    def test_general_heuristics_scoring(self, guardian_gate):
        """Test general domain heuristics scoring."""
        general_graph = {
            "nodes": [
                {"id": "node1", "properties": {"confidence": 0.8, "source": "test"}},
                {"id": "node2", "properties": {"confidence": 0.6}},
            ],
            "edges": [
                {
                    "id": "edge1",
                    "properties": {"confidence": 0.9, "timestamp": "2023-01-01"},
                }
            ],
        }

        score = guardian_gate._score_general_heuristics(general_graph)

        # Should get bonus points for confidence and metadata
        assert score > 0.5
        assert score <= 1.0

    @pytest.mark.asyncio
    async def test_external_fact_check_medical(self, guardian_gate):
        """Test external fact checking for medical domain."""
        proposals = [
            MockProposal("add_edge", "allergy_edge", relationship_type="ALLERGIC_TO"),
            MockProposal("add_edge", "prescription", relationship_type="PRESCRIBES"),
        ]

        violation = MockViolation(domain="medical")

        confidence = await guardian_gate._external_fact_check(proposals, violation)

        # Should return high confidence for medical facts
        assert confidence > 0.7

    @pytest.mark.asyncio
    async def test_external_fact_check_api_failure(self, guardian_gate):
        """Test external fact checking when API fails."""
        proposals = [MockProposal("add_edge", "test")]
        violation = MockViolation()

        # Mock API failure
        with patch("asyncio.sleep", side_effect=Exception("API Error")):
            confidence = await guardian_gate._external_fact_check(proposals, violation)

        # Should return 0.5 for unknown when API fails
        assert confidence == 0.5

    def test_evidence_strength_calculation(self, guardian_gate):
        """Test evidence strength calculation."""
        # High confidence proposals with good rationales
        strong_proposals = [
            MockProposal(
                "add_node",
                "node1",
                confidence=0.9,
                rationale="Very detailed rationale with lots of supporting evidence",
            ),
            MockProposal(
                "add_edge",
                "edge1",
                confidence=0.85,
                rationale="Another well-documented change",
            ),
        ]

        strong_score = guardian_gate._calculate_evidence_strength(strong_proposals)

        # Weak proposals
        weak_proposals = [
            MockProposal("delete_node", "node2", confidence=0.3, rationale=""),
            MockProposal("update_attr", "node3", confidence=0.2, rationale="Short"),
        ]

        weak_score = guardian_gate._calculate_evidence_strength(weak_proposals)

        assert strong_score > weak_score
        assert strong_score > 0.7
        assert weak_score < 0.5

    def test_threshold_application_high_severity(self, guardian_gate):
        """Test threshold application for high severity violations."""
        # High severity with good score should APPLY
        decision = guardian_gate._apply_thresholds(0.85, "high")
        assert decision == "APPLY"

        # High severity with medium score should QUARANTINE
        decision = guardian_gate._apply_thresholds(0.65, "high")
        assert decision == "QUARANTINE"

        # High severity with low score should REJECT
        decision = guardian_gate._apply_thresholds(0.3, "high")
        assert decision == "REJECT"

    def test_threshold_application_medium_severity(self, guardian_gate):
        """Test threshold application for medium severity violations."""
        # Medium severity with good score should APPLY
        decision = guardian_gate._apply_thresholds(0.85, "medium")
        assert decision == "APPLY"

        # Medium severity with medium score should QUARANTINE
        decision = guardian_gate._apply_thresholds(0.6, "medium")
        assert decision == "QUARANTINE"

        # Medium severity with low score should REJECT
        decision = guardian_gate._apply_thresholds(0.2, "medium")
        assert decision == "REJECT"

    def test_rationale_generation(self, guardian_gate):
        """Test rationale generation for different score combinations."""
        # High scores should generate positive rationale
        rationale = guardian_gate._generate_rationale("APPLY", 0.85, 0.9, 0.8, 0.8)
        assert "excellent structural fix" in rationale
        assert "high domain confidence" in rationale
        assert "strong evidence" in rationale
        assert "APPLY" in rationale.lower()

        # Low scores should generate negative rationale
        rationale = guardian_gate._generate_rationale("REJECT", 0.2, 0.1, 0.2, 0.3)
        assert "poor structural fix" in rationale
        assert "low domain confidence" in rationale
        assert "weak evidence" in rationale
        assert "REJECT" in rationale.lower()

        # API unknown case
        rationale = guardian_gate._generate_rationale("QUARANTINE", 0.6, 0.7, 0.5, 0.6)
        assert "external API unknown" in rationale

    def test_proposal_serialization(self, guardian_gate):
        """Test proposal serialization for audit logging."""
        proposal = MockProposal("add_edge", "test_edge", confidence=0.8, relationship_type="RELATED")

        serialized = guardian_gate._serialize_proposal(proposal)

        assert serialized["type"] == "MockProposal"
        assert serialized["operation"] == "add_edge"
        assert serialized["target_id"] == "test_edge"
        assert serialized["confidence"] == 0.8
        assert "details" in serialized

    @pytest.mark.asyncio
    async def test_performance_target(self, guardian_gate):
        """Test that evaluation meets performance target of ≤ 20ms per proposal set."""
        import time

        proposals = [MockProposal("add_edge", f"edge_{i}", confidence=0.7) for i in range(5)]
        violation = MockViolation()

        start_time = time.time()
        await guardian_gate.evaluate_repair(proposals, violation)
        end_time = time.time()

        elapsed_ms = (end_time - start_time) * 1000

        # Should be under 20ms per proposal set
        assert elapsed_ms < 20, f"Evaluation took {elapsed_ms:.1f}ms, target is <20ms"


class TestAuditLogging:
    """Test cases for audit logging functionality."""

    def test_audit_log_creation(self):
        """Test that audit.log creates proper JSON files."""
        test_record = {
            "decision": "APPLY",
            "score": 0.85,
            "rationale": "Test rationale",
        }

        audit.log(test_record)

        # Check that record was logged
        recent_records = audit.get_recent_records(hours=1, limit=10)
        assert len(recent_records) > 0

        # Find our test record
        test_records = [r for r in recent_records if r.get("decision") == "APPLY"]
        assert len(test_records) > 0

        logged_record = test_records[0]
        assert logged_record["score"] == 0.85
        assert logged_record["rationale"] == "Test rationale"
        assert "id" in logged_record
        assert "timestamp" in logged_record

    def test_audit_statistics(self):
        """Test audit statistics calculation."""
        # Log some test records
        for i, decision in enumerate(["APPLY", "QUARANTINE", "REJECT", "APPLY"]):
            audit.log(
                {
                    "decision": decision,
                    "score": 0.5 + i * 0.1,
                    "test_batch": "stats_test",
                }
            )

        stats = audit.get_statistics(hours=1)

        assert stats["total_validations"] >= 4
        assert "decisions" in stats
        assert "decision_rates" in stats
        assert "average_score" in stats

        # Check decision counts
        decisions = stats["decisions"]
        assert decisions["APPLY"] >= 2
        assert decisions["QUARANTINE"] >= 1
        assert decisions["REJECT"] >= 1

    def test_audit_cleanup(self):
        """Test audit record cleanup functionality."""
        # This would need to be tested with mocked timestamps
        # for records older than the cutoff date
        deleted_count = audit.cleanup_old_records(days=30)
        assert isinstance(deleted_count, int)
        assert deleted_count >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
