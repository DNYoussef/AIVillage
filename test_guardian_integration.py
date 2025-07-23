#!/usr/bin/env python3
"""
Simple integration test for Guardian Gate without importing the full HypeRAG module.
"""

import sys
import asyncio
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import Guardian components directly
from mcp_servers.hyperag.guardian.gate import GuardianGate, CreativeBridge
from mcp_servers.hyperag.guardian import audit


class MockProposal:
    """Mock repair proposal for testing."""

    def __init__(self, operation_type: str, target_id: str, confidence: float = 0.8):
        self.operation_type = operation_type
        self.target_id = target_id
        self.confidence = confidence
        self.rationale = f"Mock {operation_type} operation"
        self.relationship_type = "RELATED"
        self.properties = {}

    def __str__(self):
        return f"MockProposal({self.operation_type}, {self.target_id})"


class MockViolation:
    """Mock GDC violation for testing."""

    def __init__(self, id: str = "GDC_TEST", severity: str = "medium", domain: str = "general"):
        self.id = id
        self.severity = severity
        self.domain = domain
        self.subgraph = {
            "nodes": [
                {"id": "node1", "labels": ["Entity"], "properties": {"name": "Test"}},
                {"id": "node2", "labels": ["Entity"], "properties": {"name": "Test2"}}
            ],
            "edges": [
                {"id": "edge1", "startNode": "node1", "endNode": "node2", "type": "RELATED"}
            ]
        }


async def test_guardian_gate():
    """Test Guardian Gate functionality."""
    print("Testing Guardian Gate...")

    # Create Guardian Gate instance
    gate = GuardianGate()
    print("+ GuardianGate created successfully")

    # Test repair evaluation - APPLY case
    high_confidence_proposals = [
        MockProposal("add_edge", "new_edge", confidence=0.9)
    ]
    high_severity_violation = MockViolation(id="GDC_HIGH", severity="high", domain="medical")

    decision = await gate.evaluate_repair(high_confidence_proposals, high_severity_violation)
    print(f"+ High severity repair evaluation: {decision}")

    # Test repair evaluation - QUARANTINE case
    medium_confidence_proposals = [
        MockProposal("update_attr", "node1", confidence=0.6)
    ]
    medium_violation = MockViolation(severity="medium")

    decision = await gate.evaluate_repair(medium_confidence_proposals, medium_violation)
    print(f"+ Medium confidence repair evaluation: {decision}")

    # Test repair evaluation - REJECT case
    forbidden_proposals = [
        MockProposal("delete_edge", "edge1", confidence=0.5)
    ]
    forbidden_proposals[0].edge_type = "ALLERGIC_TO"  # Forbidden in medical domain

    medical_violation = MockViolation(domain="medical")
    decision = await gate.evaluate_repair(forbidden_proposals, medical_violation)
    print(f"+ Forbidden operation evaluation: {decision}")

    # Test creative bridge evaluation
    bridge = CreativeBridge(id="test_bridge", confidence=0.8)
    decision = await gate.evaluate_creative(bridge)
    print(f"+ Creative bridge evaluation: {decision}")

    # Test audit logging
    recent_records = audit.get_recent_records(hours=1, limit=10)
    print(f"+ Audit records created: {len(recent_records)} records")

    # Test audit statistics
    stats = audit.get_statistics(hours=1)
    print(f"+ Audit statistics: {stats['total_validations']} validations")

    print("\nAll Guardian Gate tests passed!")


if __name__ == "__main__":
    # Patch the import issue by creating empty module paths
    import types
    sys.modules['mcp_servers.hyperag.repair.innovator_agent'] = types.ModuleType('mock_repair')
    sys.modules['mcp_servers.hyperag.gdc.specs'] = types.ModuleType('mock_gdc')

    # Add mock classes to avoid import errors
    class MockRepairOperation:
        pass
    class MockViolationClass:
        pass

    sys.modules['mcp_servers.hyperag.repair.innovator_agent'].RepairOperation = MockRepairOperation
    sys.modules['mcp_servers.hyperag.gdc.specs'].Violation = MockViolationClass

    # Run the test
    asyncio.run(test_guardian_gate())
