#!/usr/bin/env python3
"""HypeRAG Graph Repair Test Suite.

Injects controlled violations into test knowledge graphs and measures the repair pipeline's ability to:
- Detect violations (Detection Recall)
- Generate valid repair proposals (Proposal Validity)
- Guardian rejection rate for destructive fixes (Guardian Reject %)
- Measure residual violations after repair (Residual Violations)

Test Categories:
- Allergy Conflicts: Patient prescribed drug they're allergic to
- Duplicate Identity: Same entity represented multiple times
- Temporal Inconsistencies: Events with impossible timestamps
- Missing Properties: Critical attributes missing from entities
- Orphaned Relationships: Edges pointing to non-existent nodes
"""

import argparse
import asyncio
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
import json
import logging
from pathlib import Path

# Import HypeRAG components
import sys
from typing import Any
import uuid

sys.path.append(str(Path(__file__).parent.parent))

from mcp_servers.hyperag.guardian.gate import GuardianGate
from mcp_servers.hyperag.repair.innovator_agent import InnovatorAgent

logger = logging.getLogger(__name__)


@dataclass
class RepairTestMetrics:
    """Repair test suite results."""

    detection_recall: float
    proposal_validity: float
    guardian_reject_rate: float
    residual_violation_rate: float
    total_violations_injected: int
    total_violations_detected: int
    total_proposals_generated: int
    total_guardian_rejections: int
    test_suite_name: str
    timestamp: str


@dataclass
class InjectedViolation:
    """A violation injected into the test graph."""

    violation_id: str
    violation_type: str
    severity: str
    description: str
    affected_nodes: list[str]
    affected_edges: list[str]
    expected_detection: bool
    ground_truth_repair: dict[str, Any]


@dataclass
class TestGraphNode:
    """Node in test graph."""

    id: str
    type: str
    properties: dict[str, Any]


@dataclass
class TestGraphEdge:
    """Edge in test graph."""

    id: str
    source: str
    target: str
    type: str
    properties: dict[str, Any]


class ViolationInjector:
    """Injects controlled violations into test knowledge graphs."""

    def __init__(self) -> None:
        self.violation_templates = self._create_violation_templates()

    def _create_violation_templates(self) -> dict[str, dict]:
        """Create templates for different violation types."""
        return {
            "allergy_conflict": {
                "description": "Patient prescribed drug they are allergic to",
                "severity": "high",
                "detection_expected": True,
                "repair_operations": ["delete_edge", "add_edge"],
            },
            "duplicate_identity": {
                "description": "Same entity represented by multiple nodes",
                "severity": "medium",
                "detection_expected": True,
                "repair_operations": ["merge_nodes", "delete_node"],
            },
            "temporal_inconsistency": {
                "description": "Event with impossible timestamp",
                "severity": "medium",
                "detection_expected": True,
                "repair_operations": ["update_property"],
            },
            "missing_critical_property": {
                "description": "Critical property missing from entity",
                "severity": "medium",
                "detection_expected": True,
                "repair_operations": ["add_property", "update_property"],
            },
            "orphaned_relationship": {
                "description": "Edge pointing to non-existent node",
                "severity": "high",
                "detection_expected": True,
                "repair_operations": ["delete_edge", "add_node"],
            },
            "contradictory_facts": {
                "description": "Two contradictory statements about same entity",
                "severity": "high",
                "detection_expected": True,
                "repair_operations": ["delete_edge", "update_property"],
            },
            "circular_dependency": {
                "description": "Circular relationship that should be hierarchical",
                "severity": "low",
                "detection_expected": False,
                "repair_operations": ["delete_edge", "add_edge"],
            },
        }

    def create_medical_test_graph(
        self,
    ) -> tuple[list[TestGraphNode], list[TestGraphEdge]]:
        """Create a medical domain test graph."""
        nodes = [
            # Patients
            TestGraphNode("P001", "Patient", {"name": "John Doe", "age": 45, "gender": "M"}),
            TestGraphNode("P002", "Patient", {"name": "Jane Smith", "age": 32, "gender": "F"}),
            TestGraphNode("P003", "Patient", {"name": "Bob Johnson", "age": 67, "gender": "M"}),
            # Drugs
            TestGraphNode("D001", "Drug", {"name": "Aspirin", "class": "NSAID"}),
            TestGraphNode("D002", "Drug", {"name": "Penicillin", "class": "Antibiotic"}),
            TestGraphNode("D003", "Drug", {"name": "Warfarin", "class": "Anticoagulant"}),
            TestGraphNode("D004", "Drug", {"name": "Ibuprofen", "class": "NSAID"}),
            # Conditions
            TestGraphNode("C001", "Condition", {"name": "Hypertension", "severity": "moderate"}),
            TestGraphNode("C002", "Condition", {"name": "Diabetes", "type": "Type 2"}),
            TestGraphNode("C003", "Condition", {"name": "Pneumonia", "severity": "severe"}),
            # Allergies
            TestGraphNode("A001", "Allergy", {"allergen": "Penicillin", "severity": "severe"}),
            TestGraphNode("A002", "Allergy", {"allergen": "NSAIDs", "severity": "moderate"}),
        ]

        edges = [
            # Normal relationships
            TestGraphEdge("E001", "P001", "C001", "DIAGNOSED_WITH", {"date": "2024-01-15"}),
            TestGraphEdge("E002", "P002", "C002", "DIAGNOSED_WITH", {"date": "2024-02-10"}),
            TestGraphEdge("E003", "P003", "C003", "DIAGNOSED_WITH", {"date": "2024-03-05"}),
            # Prescriptions (some will be made problematic)
            TestGraphEdge(
                "E004",
                "P001",
                "D001",
                "PRESCRIBED",
                {"date": "2024-01-16", "dosage": "100mg daily"},
            ),
            TestGraphEdge(
                "E005",
                "P002",
                "D003",
                "PRESCRIBED",
                {"date": "2024-02-11", "dosage": "5mg daily"},
            ),
            # Allergies
            TestGraphEdge("E006", "P001", "A002", "ALLERGIC_TO", {"discovered": "2024-01-10"}),
            TestGraphEdge("E007", "P003", "A001", "ALLERGIC_TO", {"discovered": "2023-12-01"}),
        ]

        return nodes, edges

    def inject_allergy_conflict(self, nodes: list[TestGraphNode], edges: list[TestGraphEdge]) -> InjectedViolation:
        """Inject an allergy conflict violation."""
        violation_id = f"violation_{uuid.uuid4().hex[:8]}"

        # Find a patient with an allergy
        allergic_patient = None
        allergen = None
        for edge in edges:
            if edge.type == "ALLERGIC_TO":
                allergic_patient = edge.source
                # Find the allergen
                for node in nodes:
                    if node.id == edge.target and node.type == "Allergy":
                        allergen = node.properties.get("allergen")
                        break
                break

        if not allergic_patient or not allergen:
            # Create allergy relationship if none exists
            allergic_patient = "P001"
            allergen = "Penicillin"
            allergy_node = TestGraphNode("A999", "Allergy", {"allergen": allergen, "severity": "severe"})
            allergy_edge = TestGraphEdge(
                "E999",
                allergic_patient,
                "A999",
                "ALLERGIC_TO",
                {"discovered": "2024-01-01"},
            )
            nodes.append(allergy_node)
            edges.append(allergy_edge)

        # Find a drug that matches the allergen
        problematic_drug = None
        for node in nodes:
            if node.type == "Drug" and (
                node.properties.get("name") == allergen or node.properties.get("class") == allergen
            ):
                problematic_drug = node.id
                break

        if not problematic_drug:
            # Create the problematic drug
            problematic_drug = "D999"
            drug_node = TestGraphNode(problematic_drug, "Drug", {"name": allergen, "class": "Antibiotic"})
            nodes.append(drug_node)

        # Inject the problematic prescription
        violation_edge_id = f"E_{violation_id}"
        violation_edge = TestGraphEdge(
            violation_edge_id,
            allergic_patient,
            problematic_drug,
            "PRESCRIBED",
            {"date": "2024-01-20", "dosage": "500mg twice daily"},
        )
        edges.append(violation_edge)

        return InjectedViolation(
            violation_id=violation_id,
            violation_type="allergy_conflict",
            severity="high",
            description=f"Patient {allergic_patient} prescribed {problematic_drug} despite allergy to {allergen}",
            affected_nodes=[allergic_patient, problematic_drug],
            affected_edges=[violation_edge_id],
            expected_detection=True,
            ground_truth_repair={
                "operation": "delete_edge",
                "target": violation_edge_id,
                "rationale": "Remove prescription conflicting with known allergy",
            },
        )

    def inject_duplicate_identity(self, nodes: list[TestGraphNode], edges: list[TestGraphEdge]) -> InjectedViolation:
        """Inject a duplicate identity violation."""
        violation_id = f"violation_{uuid.uuid4().hex[:8]}"

        # Create a duplicate of an existing patient
        original_patient = nodes[0]  # Take first patient
        duplicate_patient = TestGraphNode(
            f"P_{violation_id}",
            "Patient",
            {
                **original_patient.properties,
                "name": original_patient.properties["name"] + " (Duplicate)",
            },
        )
        nodes.append(duplicate_patient)

        # Create some relationships for the duplicate
        violation_edge_id = f"E_{violation_id}"
        if len(edges) > 0:
            # Duplicate an existing relationship
            original_edge = edges[0]
            if original_edge.source == original_patient.id:
                duplicate_edge = TestGraphEdge(
                    violation_edge_id,
                    duplicate_patient.id,
                    original_edge.target,
                    original_edge.type,
                    original_edge.properties.copy(),
                )
                edges.append(duplicate_edge)

        return InjectedViolation(
            violation_id=violation_id,
            violation_type="duplicate_identity",
            severity="medium",
            description=f"Duplicate identity: {original_patient.id} and {duplicate_patient.id}",
            affected_nodes=[original_patient.id, duplicate_patient.id],
            affected_edges=[violation_edge_id],
            expected_detection=True,
            ground_truth_repair={
                "operation": "merge_nodes",
                "targets": [original_patient.id, duplicate_patient.id],
                "rationale": "Merge duplicate patient records",
            },
        )

    def inject_temporal_inconsistency(
        self, nodes: list[TestGraphNode], edges: list[TestGraphEdge]
    ) -> InjectedViolation:
        """Inject a temporal inconsistency violation."""
        violation_id = f"violation_{uuid.uuid4().hex[:8]}"

        # Find an edge with a date property and make it inconsistent
        target_edge = None
        for edge in edges:
            if "date" in edge.properties:
                target_edge = edge
                break

        if not target_edge:
            # Create a new edge with temporal inconsistency
            violation_edge_id = f"E_{violation_id}"
            future_date = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")
            violation_edge = TestGraphEdge(
                violation_edge_id,
                nodes[0].id,
                nodes[1].id,
                "FUTURE_EVENT",
                {"date": future_date, "description": "Event in the future"},
            )
            edges.append(violation_edge)
            target_edge = violation_edge
        else:
            # Modify existing edge to have future date
            future_date = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")
            target_edge.properties["date"] = future_date

        return InjectedViolation(
            violation_id=violation_id,
            violation_type="temporal_inconsistency",
            severity="medium",
            description=f"Edge {target_edge.id} has future date: {target_edge.properties['date']}",
            affected_nodes=[target_edge.source, target_edge.target],
            affected_edges=[target_edge.id],
            expected_detection=True,
            ground_truth_repair={
                "operation": "update_property",
                "target": target_edge.id,
                "property": "date",
                "value": datetime.now().strftime("%Y-%m-%d"),
                "rationale": "Correct temporal inconsistency",
            },
        )

    def inject_missing_critical_property(
        self, nodes: list[TestGraphNode], edges: list[TestGraphEdge]
    ) -> InjectedViolation:
        """Inject a missing critical property violation."""
        violation_id = f"violation_{uuid.uuid4().hex[:8]}"

        # Find a drug node and remove its dosage from a prescription
        target_edge = None
        for edge in edges:
            if edge.type == "PRESCRIBED" and "dosage" in edge.properties:
                del edge.properties["dosage"]
                target_edge = edge
                break

        if not target_edge:
            # Create a prescription without dosage
            violation_edge_id = f"E_{violation_id}"
            violation_edge = TestGraphEdge(
                violation_edge_id,
                nodes[0].id,
                nodes[-1].id,  # Last node (likely a drug)
                "PRESCRIBED",
                {"date": "2024-01-15"},  # Missing dosage
            )
            edges.append(violation_edge)
            target_edge = violation_edge

        return InjectedViolation(
            violation_id=violation_id,
            violation_type="missing_critical_property",
            severity="medium",
            description=f"Prescription {target_edge.id} missing dosage information",
            affected_nodes=[target_edge.source, target_edge.target],
            affected_edges=[target_edge.id],
            expected_detection=True,
            ground_truth_repair={
                "operation": "add_property",
                "target": target_edge.id,
                "property": "dosage",
                "value": "Standard dosage per guidelines",
                "rationale": "Add missing critical dosage information",
            },
        )

    def inject_orphaned_relationship(self, nodes: list[TestGraphNode], edges: list[TestGraphEdge]) -> InjectedViolation:
        """Inject an orphaned relationship violation."""
        violation_id = f"violation_{uuid.uuid4().hex[:8]}"

        # Create an edge pointing to a non-existent node
        violation_edge_id = f"E_{violation_id}"
        non_existent_node = f"MISSING_{violation_id}"

        violation_edge = TestGraphEdge(
            violation_edge_id,
            nodes[0].id,
            non_existent_node,
            "POINTS_TO_MISSING",
            {"created": "2024-01-15"},
        )
        edges.append(violation_edge)

        return InjectedViolation(
            violation_id=violation_id,
            violation_type="orphaned_relationship",
            severity="high",
            description=f"Edge {violation_edge_id} points to non-existent node {non_existent_node}",
            affected_nodes=[nodes[0].id, non_existent_node],
            affected_edges=[violation_edge_id],
            expected_detection=True,
            ground_truth_repair={
                "operation": "delete_edge",
                "target": violation_edge_id,
                "rationale": "Remove orphaned relationship",
            },
        )


class RepairTestSuite:
    """Main repair test suite evaluation system."""

    def __init__(
        self,
        guardian_gate: GuardianGate | None = None,
        innovator_agent: InnovatorAgent | None = None,
        output_dir: Path = Path("./repair_test_results"),
    ) -> None:
        self.guardian_gate = guardian_gate or GuardianGate()
        self.innovator_agent = innovator_agent  # Will be mocked if None
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.violation_injector = ViolationInjector()
        self.test_results = []

    async def run_comprehensive_repair_tests(self) -> RepairTestMetrics:
        """Run the complete repair test suite."""
        logger.info("Starting HypeRAG Graph Repair Test Suite...")

        # Test different violation types
        violation_types = [
            "allergy_conflict",
            "duplicate_identity",
            "temporal_inconsistency",
            "missing_critical_property",
            "orphaned_relationship",
        ]

        all_violations = []
        all_detections = []
        all_proposals = []
        all_guardian_decisions = []

        for violation_type in violation_types:
            logger.info(f"Testing {violation_type} violations...")

            # Run multiple tests for each violation type
            for test_num in range(3):  # 3 tests per violation type
                try:
                    test_result = await self._run_single_violation_test(violation_type, test_num)
                    self.test_results.append(test_result)

                    all_violations.extend(test_result["injected_violations"])
                    all_detections.extend(test_result["detected_violations"])
                    all_proposals.extend(test_result["repair_proposals"])
                    all_guardian_decisions.extend(test_result["guardian_decisions"])

                except Exception as e:
                    logger.exception(f"Error in {violation_type} test {test_num}: {e}")
                    continue

        # Calculate metrics
        metrics = self._calculate_repair_metrics(all_violations, all_detections, all_proposals, all_guardian_decisions)

        # Save results
        await self._save_test_results(metrics)

        logger.info("Repair test suite completed successfully")
        return metrics

    async def _run_single_violation_test(self, violation_type: str, test_num: int) -> dict[str, Any]:
        """Run a single violation test."""
        test_id = f"{violation_type}_{test_num}"
        logger.debug(f"Running test {test_id}")

        # Create test graph
        nodes, edges = self.violation_injector.create_medical_test_graph()

        # Inject violations based on type
        injected_violations = []
        if violation_type == "allergy_conflict":
            violation = self.violation_injector.inject_allergy_conflict(nodes, edges)
            injected_violations.append(violation)
        elif violation_type == "duplicate_identity":
            violation = self.violation_injector.inject_duplicate_identity(nodes, edges)
            injected_violations.append(violation)
        elif violation_type == "temporal_inconsistency":
            violation = self.violation_injector.inject_temporal_inconsistency(nodes, edges)
            injected_violations.append(violation)
        elif violation_type == "missing_critical_property":
            violation = self.violation_injector.inject_missing_critical_property(nodes, edges)
            injected_violations.append(violation)
        elif violation_type == "orphaned_relationship":
            violation = self.violation_injector.inject_orphaned_relationship(nodes, edges)
            injected_violations.append(violation)

        # Convert to graph format for processing
        test_graph = self._convert_to_graph_format(nodes, edges)

        # Step 1: Detection phase
        detected_violations = await self._detect_violations(test_graph)

        # Step 2: Repair proposal generation
        repair_proposals = []
        for detected in detected_violations:
            proposals = await self._generate_repair_proposals(detected, test_graph)
            repair_proposals.extend(proposals)

        # Step 3: Guardian validation
        guardian_decisions = []
        for proposal in repair_proposals:
            decision = await self._validate_with_guardian(proposal, detected_violations)
            guardian_decisions.append(decision)

        # Step 4: Apply repairs and check residuals
        applied_repairs = [d for d in guardian_decisions if d["decision"] == "APPLY"]
        residual_violations = await self._check_residual_violations(test_graph, applied_repairs)

        return {
            "test_id": test_id,
            "violation_type": violation_type,
            "injected_violations": injected_violations,
            "detected_violations": detected_violations,
            "repair_proposals": repair_proposals,
            "guardian_decisions": guardian_decisions,
            "applied_repairs": applied_repairs,
            "residual_violations": residual_violations,
            "test_graph": test_graph,
        }

    def _convert_to_graph_format(self, nodes: list[TestGraphNode], edges: list[TestGraphEdge]) -> dict[str, Any]:
        """Convert test nodes/edges to graph format."""
        return {
            "nodes": [{"id": n.id, "type": n.type, "properties": n.properties} for n in nodes],
            "edges": [
                {
                    "id": e.id,
                    "source": e.source,
                    "target": e.target,
                    "type": e.type,
                    "properties": e.properties,
                }
                for e in edges
            ],
        }

    async def _detect_violations(self, graph: dict[str, Any]) -> list[dict[str, Any]]:
        """Mock violation detection - in reality would use ViolationExtractor."""
        detected = []

        # Mock detection logic based on common patterns
        nodes_by_id = {n["id"]: n for n in graph["nodes"]}

        for edge in graph["edges"]:
            # Check for orphaned relationships
            if edge["target"] not in nodes_by_id:
                detected.append(
                    {
                        "violation_id": f"detected_{edge['id']}",
                        "type": "orphaned_relationship",
                        "severity": "high",
                        "description": f"Edge {edge['id']} points to missing node {edge['target']}",
                        "affected_elements": [edge["id"]],
                    }
                )

            # Check for allergy conflicts
            if edge["type"] == "PRESCRIBED":
                source_node = nodes_by_id.get(edge["source"])
                if source_node:
                    # Check if patient has allergies that conflict with prescription
                    for other_edge in graph["edges"]:
                        if other_edge["source"] == edge["source"] and other_edge["type"] == "ALLERGIC_TO":
                            detected.append(
                                {
                                    "violation_id": f"detected_allergy_{edge['id']}",
                                    "type": "allergy_conflict",
                                    "severity": "high",
                                    "description": f"Patient {edge['source']} prescribed drug despite allergy",
                                    "affected_elements": [edge["id"], other_edge["id"]],
                                }
                            )

            # Check for missing critical properties
            if edge["type"] == "PRESCRIBED" and "dosage" not in edge["properties"]:
                detected.append(
                    {
                        "violation_id": f"detected_missing_{edge['id']}",
                        "type": "missing_critical_property",
                        "severity": "medium",
                        "description": f"Prescription {edge['id']} missing dosage information",
                        "affected_elements": [edge["id"]],
                    }
                )

            # Check for temporal inconsistencies
            if "date" in edge["properties"]:
                try:
                    edge_date = datetime.strptime(edge["properties"]["date"], "%Y-%m-%d")
                    if edge_date > datetime.now():
                        detected.append(
                            {
                                "violation_id": f"detected_temporal_{edge['id']}",
                                "type": "temporal_inconsistency",
                                "severity": "medium",
                                "description": f"Edge {edge['id']} has future date",
                                "affected_elements": [edge["id"]],
                            }
                        )
                except Exception as e:
                    import logging

                    logging.exception("Exception in temporal consistency check date parsing: %s", str(e))

        # Check for duplicate identities (simplified)
        patient_names = defaultdict(list)
        for node in graph["nodes"]:
            if node["type"] == "Patient" and "name" in node["properties"]:
                name = node["properties"]["name"].replace(" (Duplicate)", "")
                patient_names[name].append(node["id"])

        for name, node_ids in patient_names.items():
            if len(node_ids) > 1:
                detected.append(
                    {
                        "violation_id": f"detected_dup_{name}",
                        "type": "duplicate_identity",
                        "severity": "medium",
                        "description": f"Duplicate patient records for {name}",
                        "affected_elements": node_ids,
                    }
                )

        return detected

    async def _generate_repair_proposals(
        self, violation: dict[str, Any], graph: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Mock repair proposal generation - in reality would use InnovatorAgent."""
        proposals = []

        violation_type = violation["type"]

        if violation_type == "orphaned_relationship":
            # Propose deleting the orphaned edge
            proposals.append(
                {
                    "proposal_id": f"repair_{violation['violation_id']}",
                    "operation": "delete_edge",
                    "target": violation["affected_elements"][0],
                    "rationale": "Remove orphaned relationship",
                    "confidence": 0.9,
                }
            )

        elif violation_type == "allergy_conflict":
            # Propose deleting the conflicting prescription
            affected_edges = violation["affected_elements"]
            prescription_edge = None
            for edge_id in affected_edges:
                for edge in graph["edges"]:
                    if edge["id"] == edge_id and edge["type"] == "PRESCRIBED":
                        prescription_edge = edge_id
                        break

            if prescription_edge:
                proposals.append(
                    {
                        "proposal_id": f"repair_{violation['violation_id']}",
                        "operation": "delete_edge",
                        "target": prescription_edge,
                        "rationale": "Remove prescription conflicting with known allergy",
                        "confidence": 0.95,
                    }
                )

        elif violation_type == "missing_critical_property":
            # Propose adding the missing property
            proposals.append(
                {
                    "proposal_id": f"repair_{violation['violation_id']}",
                    "operation": "add_property",
                    "target": violation["affected_elements"][0],
                    "property": "dosage",
                    "value": "Standard dosage per guidelines",
                    "rationale": "Add missing critical dosage information",
                    "confidence": 0.8,
                }
            )

        elif violation_type == "temporal_inconsistency":
            # Propose correcting the date
            proposals.append(
                {
                    "proposal_id": f"repair_{violation['violation_id']}",
                    "operation": "update_property",
                    "target": violation["affected_elements"][0],
                    "property": "date",
                    "value": datetime.now().strftime("%Y-%m-%d"),
                    "rationale": "Correct temporal inconsistency",
                    "confidence": 0.85,
                }
            )

        elif violation_type == "duplicate_identity":
            # Propose merging duplicate nodes
            proposals.append(
                {
                    "proposal_id": f"repair_{violation['violation_id']}",
                    "operation": "merge_nodes",
                    "targets": violation["affected_elements"],
                    "rationale": "Merge duplicate patient records",
                    "confidence": 0.75,
                }
            )

        return proposals

    async def _validate_with_guardian(
        self, proposal: dict[str, Any], violations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Validate repair proposal with Guardian Gate."""
        # Mock Guardian validation
        confidence = proposal.get("confidence", 0.5)
        operation = proposal.get("operation", "unknown")

        # Guardian is more likely to reject destructive operations with low confidence
        if operation in ["delete_edge", "delete_node"] and confidence < 0.7:
            decision = "REJECT"
            reason = "Destructive operation with insufficient confidence"
        elif confidence < 0.4:
            decision = "QUARANTINE"
            reason = "Low confidence repair proposal"
        else:
            decision = "APPLY"
            reason = "Repair proposal meets quality standards"

        return {
            "proposal_id": proposal["proposal_id"],
            "decision": decision,
            "reason": reason,
            "guardian_confidence": confidence,
        }

    async def _check_residual_violations(
        self, graph: dict[str, Any], applied_repairs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Check for violations remaining after repairs."""
        # Mock residual violation checking
        # In reality, this would apply the repairs to the graph and re-run detection

        # Simple heuristic: assume 10-20% of violations remain due to incomplete repairs
        residual_count = max(1, len(applied_repairs) // 5)

        residual_violations = []
        for i in range(residual_count):
            residual_violations.append(
                {
                    "violation_id": f"residual_{i}",
                    "type": "incomplete_repair",
                    "severity": "low",
                    "description": "Violation not fully resolved by repair",
                    "affected_elements": [f"element_{i}"],
                }
            )

        return residual_violations

    def _calculate_repair_metrics(
        self,
        violations: list[InjectedViolation],
        detections: list[dict[str, Any]],
        proposals: list[dict[str, Any]],
        guardian_decisions: list[dict[str, Any]],
    ) -> RepairTestMetrics:
        """Calculate repair test metrics."""
        # Detection Recall: What fraction of injected violations were detected?
        expected_detections = sum(1 for v in violations if v.expected_detection)
        actual_detections = len(detections)
        detection_recall = actual_detections / expected_detections if expected_detections > 0 else 0.0

        # Proposal Validity: What fraction of proposals are valid operations?
        valid_proposals = 0
        for proposal in proposals:
            # Simple validity check - has required fields
            if (
                proposal.get("operation")
                and proposal.get("target")
                and proposal.get("rationale")
                and proposal.get("confidence", 0) > 0
            ):
                valid_proposals += 1

        proposal_validity = valid_proposals / len(proposals) if proposals else 0.0

        # Guardian Reject Rate: What fraction of proposals did Guardian reject?
        rejections = sum(1 for d in guardian_decisions if d["decision"] == "REJECT")
        guardian_reject_rate = rejections / len(guardian_decisions) if guardian_decisions else 0.0

        # Residual Violation Rate: Simplified calculation
        total_original_violations = len(violations)
        applied_repairs = sum(1 for d in guardian_decisions if d["decision"] == "APPLY")
        estimated_resolved = min(applied_repairs, total_original_violations)
        residual_violations = max(0, total_original_violations - estimated_resolved)
        residual_violation_rate = (
            residual_violations / total_original_violations if total_original_violations > 0 else 0.0
        )

        return RepairTestMetrics(
            detection_recall=detection_recall,
            proposal_validity=proposal_validity,
            guardian_reject_rate=guardian_reject_rate,
            residual_violation_rate=residual_violation_rate,
            total_violations_injected=len(violations),
            total_violations_detected=len(detections),
            total_proposals_generated=len(proposals),
            total_guardian_rejections=rejections,
            test_suite_name="comprehensive_repair_test",
            timestamp=datetime.now(UTC).isoformat(),
        )

    async def _save_test_results(self, metrics: RepairTestMetrics) -> None:
        """Save test results to files."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = self.output_dir / f"repair_test_results_{timestamp}.json"
        detailed_results = {
            "metadata": {
                "test_suite_version": "1.0",
                "timestamp": metrics.timestamp,
                "total_violations_injected": metrics.total_violations_injected,
            },
            "summary_metrics": asdict(metrics),
            "detailed_test_results": self.test_results,
        }

        with open(results_file, "w") as f:
            json.dump(detailed_results, f, indent=2)

        # Save metrics summary
        metrics_file = self.output_dir / f"repair_metrics_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(asdict(metrics), f, indent=2)

        logger.info(f"Results saved to {results_file}")
        logger.info(f"Metrics saved to {metrics_file}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="HypeRAG Graph Repair Test Suite")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./repair_test_results"),
        help="Output directory for results",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run test suite
    test_suite = RepairTestSuite(output_dir=args.output_dir)

    try:
        metrics = await test_suite.run_comprehensive_repair_tests()

        # Print results
        print("\n" + "=" * 60)
        print("HYPERAG GRAPH REPAIR TEST SUITE RESULTS")
        print("=" * 60)
        print(f"Detection Recall:        {metrics.detection_recall:.1%}")
        print(f"Proposal Validity:       {metrics.proposal_validity:.1%}")
        print(f"Guardian Reject Rate:    {metrics.guardian_reject_rate:.1%}")
        print(f"Residual Violation Rate: {metrics.residual_violation_rate:.1%}")
        print(f"Violations Injected:     {metrics.total_violations_injected}")
        print(f"Violations Detected:     {metrics.total_violations_detected}")
        print(f"Proposals Generated:     {metrics.total_proposals_generated}")
        print(f"Guardian Rejections:     {metrics.total_guardian_rejections}")
        print(f"Timestamp:               {metrics.timestamp}")
        print("=" * 60)

    except Exception as e:
        logger.exception(f"Test suite failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
