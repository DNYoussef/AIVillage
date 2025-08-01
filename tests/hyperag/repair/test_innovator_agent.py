"""
Unit tests for Innovator Repair Agent
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_servers.hyperag.repair.innovator_agent import (
    InnovatorAgent,
    PromptComposer,
    RepairOperation,
    RepairOperationType,
    RepairProposal,
)
from mcp_servers.hyperag.repair.llm_driver import (
    GenerationResponse,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestRepairOperation:
    """Test suite for RepairOperation"""

    def test_operation_creation(self):
        """Test repair operation creation"""
        op = RepairOperation(
            operation_type=RepairOperationType.DELETE_EDGE,
            target_id="edge_123",
            rationale="Remove unsafe relationship",
            confidence=0.9,
            source_id="node_1",
            relationship_type="PRESCRIBES",
        )

        assert op.operation_type == RepairOperationType.DELETE_EDGE
        assert op.target_id == "edge_123"
        assert op.rationale == "Remove unsafe relationship"
        assert op.confidence == 0.9
        assert op.source_id == "node_1"
        assert op.relationship_type == "PRESCRIBES"
        assert op.operation_id is not None

    def test_operation_to_dict(self):
        """Test operation serialization to dictionary"""
        op = RepairOperation(
            operation_type=RepairOperationType.UPDATE_ATTR,
            target_id="node_456",
            rationale="Fix dosage",
            confidence=0.8,
            property_name="dosage",
            property_value="500mg",
        )

        op_dict = op.to_dict()

        assert op_dict["op"] == "update_attr"
        assert op_dict["target_id"] == "node_456"
        assert op_dict["rationale"] == "Fix dosage"
        assert op_dict["confidence"] == 0.8
        assert op_dict["property_name"] == "dosage"
        assert op_dict["property_value"] == "500mg"

    def test_operation_from_dict(self):
        """Test operation deserialization from dictionary"""
        op_data = {
            "op": "add_edge",
            "target_id": "new_edge_1",
            "rationale": "Add missing relationship",
            "confidence": 0.7,
            "source_id": "node_A",
            "relationship_type": "RELATES_TO",
        }

        op = RepairOperation.from_dict(op_data)

        assert op.operation_type == RepairOperationType.ADD_EDGE
        assert op.target_id == "new_edge_1"
        assert op.rationale == "Add missing relationship"
        assert op.confidence == 0.7
        assert op.source_id == "node_A"
        assert op.relationship_type == "RELATES_TO"

    def test_operation_to_jsonl(self):
        """Test operation JSONL serialization"""
        op = RepairOperation(
            operation_type=RepairOperationType.MERGE_NODES,
            target_id="node_1",
            rationale="Merge duplicates",
            confidence=0.6,
            merge_target_id="node_2",
        )

        jsonl = op.to_jsonl()
        parsed = json.loads(jsonl)

        assert parsed["op"] == "merge_nodes"
        assert parsed["target_id"] == "node_1"
        assert parsed["merge_target_id"] == "node_2"


class TestRepairProposal:
    """Test suite for RepairProposal"""

    def test_proposal_creation(self):
        """Test repair proposal creation"""
        operations = [
            RepairOperation(
                operation_type=RepairOperationType.DELETE_EDGE,
                target_id="edge_1",
                rationale="Remove bad edge",
                confidence=0.9,
            ),
            RepairOperation(
                operation_type=RepairOperationType.ADD_EDGE,
                target_id="edge_2",
                rationale="Add good edge",
                confidence=0.8,
            ),
        ]

        proposal = RepairProposal(
            violation_id="VIO_001", gdc_rule="test_rule", operations=operations
        )

        assert proposal.violation_id == "VIO_001"
        assert proposal.gdc_rule == "test_rule"
        assert len(proposal.operations) == 2
        assert proposal.overall_confidence == 0.85  # Average of 0.9 and 0.8
        assert proposal.proposal_id is not None

    def test_proposal_to_jsonl(self):
        """Test proposal JSONL serialization"""
        operations = [
            RepairOperation(
                operation_type=RepairOperationType.UPDATE_ATTR,
                target_id="node_1",
                rationale="Update property",
                confidence=0.7,
            )
        ]

        proposal = RepairProposal(
            violation_id="VIO_002", gdc_rule="attr_rule", operations=operations
        )

        jsonl = proposal.to_jsonl()
        lines = jsonl.strip().split("\n")

        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["op"] == "update_attr"
        assert parsed["target_id"] == "node_1"

    def test_proposal_quality_metrics(self):
        """Test proposal quality metric calculations"""
        operations = [
            RepairOperation(
                operation_type=RepairOperationType.DELETE_EDGE,
                target_id="edge_1",
                rationale="Safety critical",
                confidence=0.9,
                safety_critical=True,
            ),
            RepairOperation(
                operation_type=RepairOperationType.UPDATE_ATTR,
                target_id="node_1",
                rationale="Minor fix",
                confidence=0.8,
                safety_critical=False,
            ),
        ]

        proposal = RepairProposal(
            violation_id="VIO_003", gdc_rule="quality_rule", operations=operations
        )

        assert proposal.overall_confidence == 0.85
        assert proposal.safety_score == 0.5  # 1 - (1 safety critical / 2 total)
        assert proposal.completeness_score == (2 / 3)  # 2 operations / 3 expected

    def test_high_confidence_operations(self):
        """Test filtering high confidence operations"""
        operations = [
            RepairOperation(RepairOperationType.DELETE_EDGE, "e1", "high conf", 0.9),
            RepairOperation(RepairOperationType.ADD_EDGE, "e2", "medium conf", 0.7),
            RepairOperation(RepairOperationType.UPDATE_ATTR, "n1", "low conf", 0.5),
        ]

        proposal = RepairProposal("VIO_004", "test_rule", operations)
        high_conf_ops = proposal.get_high_confidence_operations(threshold=0.8)

        assert len(high_conf_ops) == 1
        assert high_conf_ops[0].confidence == 0.9


class TestPromptComposer:
    """Test suite for PromptComposer"""

    def test_composer_initialization(self):
        """Test prompt composer initialization"""
        composer = PromptComposer(domain="medical")
        assert composer.domain == "medical"
        assert composer.prompts is not None

    def test_system_prompt_selection(self):
        """Test system prompt selection by domain"""
        general_composer = PromptComposer(domain="general")
        medical_composer = PromptComposer(domain="medical")

        general_prompt = general_composer.get_system_prompt()
        medical_prompt = medical_composer.get_system_prompt()

        assert isinstance(general_prompt, str)
        assert isinstance(medical_prompt, str)
        # Medical prompt should be different (or same if fallback)

    @patch("mcp_servers.hyperag.repair.templates.TemplateEncoder")
    def test_repair_prompt_composition(self, mock_encoder):
        """Test repair prompt composition"""
        # Mock violation template
        mock_violation = MagicMock()
        mock_violation.to_description.return_value = "Test violation description"
        mock_violation.violation_id = "VIO_005"
        mock_violation.gdc_rule = "test_rule"

        composer = PromptComposer(domain="general")

        context = {
            "entity_analysis": {"Patient": [{"id": "P1"}]},
            "relationship_analysis": {"PRESCRIBES": [{"id": "R1"}]},
        }

        prompt = composer.compose_repair_prompt(mock_violation, context)

        assert "Test violation description" in prompt
        assert "Entity Analysis" in prompt
        assert "Relationship Analysis" in prompt
        assert "JSONL format" in prompt


class TestInnovatorAgent:
    """Test suite for InnovatorAgent"""

    @pytest.fixture
    def mock_llm_driver(self):
        """Mock LLM driver"""
        driver = MagicMock()
        driver.config = MagicMock()
        driver.config.model_name = "test_model"
        return driver

    @pytest.fixture
    def mock_template_encoder(self):
        """Mock template encoder"""
        encoder = MagicMock()
        return encoder

    @pytest.fixture
    def mock_prompt_composer(self):
        """Mock prompt composer"""
        composer = MagicMock()
        composer.get_system_prompt.return_value = "Test system prompt"
        composer.compose_repair_prompt.return_value = "Test repair prompt"
        return composer

    @pytest.fixture
    def agent(self, mock_llm_driver, mock_template_encoder, mock_prompt_composer):
        """Create InnovatorAgent with mocks"""
        return InnovatorAgent(
            llm_driver=mock_llm_driver,
            template_encoder=mock_template_encoder,
            prompt_composer=mock_prompt_composer,
            domain="general",
        )

    @pytest.mark.asyncio
    async def test_analyze_violation(self, agent, mock_template_encoder):
        """Test violation analysis"""
        violation_data = {"violation_id": "VIO_006"}
        mock_violation = MagicMock()
        mock_template_encoder.encode_violation.return_value = mock_violation

        result = await agent.analyze_violation(violation_data)

        mock_template_encoder.encode_violation.assert_called_once_with(violation_data)
        assert result == mock_violation

    @pytest.mark.asyncio
    async def test_generate_repair_proposals_success(
        self, agent, mock_llm_driver, mock_template_encoder, mock_prompt_composer
    ):
        """Test successful repair proposal generation"""
        # Mock violation data
        violation_data = {"violation_id": "VIO_007", "rule_name": "test_rule"}

        # Mock template encoder
        mock_violation = MagicMock()
        mock_violation.violation_id = "VIO_007"
        mock_violation.gdc_rule = "test_rule"
        mock_template_encoder.encode_violation.return_value = mock_violation
        mock_template_encoder.create_repair_context.return_value = {}

        # Mock LLM response with valid JSONL
        mock_response = GenerationResponse(
            text='{"op":"delete_edge","target_id":"edge_1","rationale":"Test rationale","confidence":0.8}\n',
            finish_reason="completed",
            usage={"total_tokens": 50},
            model="test_model",
            latency_ms=100.0,
        )
        mock_llm_driver.generate.return_value = mock_response

        # Generate proposals
        proposal = await agent.generate_repair_proposals(violation_data)

        assert proposal.violation_id == "VIO_007"
        assert proposal.gdc_rule == "test_rule"
        assert len(proposal.operations) == 1
        assert proposal.operations[0].operation_type == RepairOperationType.DELETE_EDGE
        assert proposal.operations[0].confidence == 0.8

    @pytest.mark.asyncio
    async def test_generate_repair_proposals_parsing_error(
        self, agent, mock_llm_driver, mock_template_encoder, mock_prompt_composer
    ):
        """Test handling of LLM response parsing errors"""
        violation_data = {"violation_id": "VIO_008", "rule_name": "test_rule"}

        # Mock template encoder
        mock_violation = MagicMock()
        mock_violation.violation_id = "VIO_008"
        mock_violation.gdc_rule = "test_rule"
        mock_template_encoder.encode_violation.return_value = mock_violation
        mock_template_encoder.create_repair_context.return_value = {}

        # Mock LLM response with invalid JSON
        mock_response = GenerationResponse(
            text='This is not valid JSON\n{"invalid": json}',
            finish_reason="completed",
            usage={"total_tokens": 20},
            model="test_model",
            latency_ms=100.0,
        )
        mock_llm_driver.generate.return_value = mock_response

        # Generate proposals
        proposal = await agent.generate_repair_proposals(violation_data)

        assert proposal.violation_id == "VIO_008"
        assert len(proposal.operations) == 0  # No valid operations parsed

    def test_parse_repair_operations_valid(self, agent):
        """Test parsing valid repair operations"""
        response_text = """{"op":"delete_edge","target_id":"edge_1","rationale":"Remove bad edge","confidence":0.9}
{"op":"add_edge","target_id":"edge_2","source_id":"node_1","rationale":"Add good edge","confidence":0.8}
{"op":"update_attr","target_id":"node_1","property_name":"status","property_value":"active","rationale":"Update status","confidence":0.7}"""

        operations = agent._parse_repair_operations(response_text)

        assert len(operations) == 3
        assert operations[0].operation_type == RepairOperationType.DELETE_EDGE
        assert operations[1].operation_type == RepairOperationType.ADD_EDGE
        assert operations[2].operation_type == RepairOperationType.UPDATE_ATTR
        assert operations[0].confidence == 0.9
        assert operations[1].source_id == "node_1"
        assert operations[2].property_name == "status"

    def test_parse_repair_operations_mixed_content(self, agent):
        """Test parsing operations with mixed valid/invalid content"""
        response_text = """Here are the repair operations:
{"op":"delete_edge","target_id":"edge_1","rationale":"Remove bad edge","confidence":0.9}
This is explanatory text that should be ignored.
{"invalid": "json"}
{"op":"update_attr","target_id":"node_1","property_name":"status","property_value":"fixed","rationale":"Fix it","confidence":0.8}
Final explanatory text."""

        operations = agent._parse_repair_operations(response_text)

        assert len(operations) == 2  # Only valid operations
        assert operations[0].operation_type == RepairOperationType.DELETE_EDGE
        assert operations[1].operation_type == RepairOperationType.UPDATE_ATTR

    def test_is_safety_critical_medical(self, agent):
        """Test safety critical detection for medical domain"""
        agent.domain = "medical"

        # Safety critical: delete prescription edge
        op1 = RepairOperation(
            RepairOperationType.DELETE_EDGE,
            "edge_1",
            "Remove prescription",
            0.9,
            relationship_type="PRESCRIBES",
        )

        # Safety critical: allergy-related operation
        op2 = RepairOperation(
            RepairOperationType.UPDATE_ATTR,
            "node_1",
            "Patient has allergy to penicillin",
            0.8,
        )

        # Not safety critical: general update
        op3 = RepairOperation(
            RepairOperationType.UPDATE_ATTR, "node_2", "Update contact information", 0.7
        )

        assert agent._is_safety_critical(op1)
        assert agent._is_safety_critical(op2)
        assert agent._is_safety_critical(op3) == False

    def test_estimate_impact(self, agent):
        """Test impact estimation"""
        # High impact: node operations
        op1 = RepairOperation(RepairOperationType.DELETE_NODE, "n1", "Delete", 0.9)
        op2 = RepairOperation(RepairOperationType.MERGE_NODES, "n1", "Merge", 0.8)

        # Medium impact: edge operations
        op3 = RepairOperation(RepairOperationType.DELETE_EDGE, "e1", "Delete edge", 0.9)
        op4 = RepairOperation(RepairOperationType.ADD_EDGE, "e2", "Add edge", 0.8)

        # Low impact: attribute updates
        op5 = RepairOperation(RepairOperationType.UPDATE_ATTR, "n1", "Update", 0.7)

        assert agent._estimate_impact(op1) == "high"
        assert agent._estimate_impact(op2) == "high"
        assert agent._estimate_impact(op3) == "medium"
        assert agent._estimate_impact(op4) == "medium"
        assert agent._estimate_impact(op5) == "low"

    @pytest.mark.asyncio
    async def test_validate_proposal(self, agent):
        """Test proposal validation"""
        # Valid proposal
        operations = [
            RepairOperation(
                RepairOperationType.UPDATE_ATTR, "node_1", "Good rationale", 0.8
            )
        ]
        proposal = RepairProposal("VIO_009", "test_rule", operations)

        validation = await agent.validate_proposal(proposal)

        assert validation["is_valid"]
        assert len(validation["errors"]) == 0

        # Invalid proposal - missing target_id
        bad_operations = [
            RepairOperation(
                RepairOperationType.UPDATE_ATTR,
                "",  # Empty target_id
                "Bad operation",
                0.5,
            )
        ]
        bad_proposal = RepairProposal("VIO_010", "test_rule", bad_operations)

        bad_validation = await agent.validate_proposal(bad_proposal)

        assert bad_validation["is_valid"] == False
        assert len(bad_validation["errors"]) > 0

    def test_performance_stats(self, agent):
        """Test performance statistics"""
        # Empty history
        stats = agent.get_performance_stats()
        assert "message" in stats

        # Add some history
        agent.repair_history = [
            {
                "timestamp": datetime.now(),
                "violation_id": "VIO_011",
                "gdc_rule": "rule_1",
                "operations_count": 2,
                "overall_confidence": 0.8,
                "safety_score": 0.9,
                "generation_time_ms": 150.0,
            },
            {
                "timestamp": datetime.now(),
                "violation_id": "VIO_012",
                "gdc_rule": "rule_2",
                "operations_count": 1,
                "overall_confidence": 0.7,
                "safety_score": 1.0,
                "generation_time_ms": 100.0,
            },
        ]

        stats = agent.get_performance_stats()

        assert stats["total_repairs_attempted"] == 2
        assert stats["average_confidence"] == 0.75
        assert stats["average_generation_time_ms"] == 125.0
        assert stats["average_operations_per_repair"] == 1.5
        assert stats["domain"] == "general"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
