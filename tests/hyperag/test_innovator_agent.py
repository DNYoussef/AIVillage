"""
Unit tests for Enhanced Innovator Repair Agent
"""

import asyncio
import json
from pathlib import Path
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_servers.hyperag.repair.innovator_agent import (
    InnovatorAgent,
    RepairOperation,
    RepairOperationType,
    RepairProposalSet,
)
from mcp_servers.hyperag.repair.llm_driver import (
    GenerationResponse,
    LLMDriver,
    ModelBackend,
    ModelConfig,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestRepairProposalSet:
    """Test suite for RepairProposalSet"""

    def test_proposal_set_creation(self):
        """Test repair proposal set creation"""
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

        proposal_set = RepairProposalSet(
            proposals=operations, violation_id="VIO_001", gdc_rule="test_rule"
        )

        assert proposal_set.violation_id == "VIO_001"
        assert proposal_set.gdc_rule == "test_rule"
        assert len(proposal_set.proposals) == 2
        assert proposal_set.overall_confidence == 0.85  # Average of 0.9 and 0.8
        assert proposal_set.proposal_set_id is not None
        assert proposal_set.is_valid is True  # Initially valid

    def test_proposal_set_validation(self):
        """Test proposal set validation"""
        # Valid proposals
        valid_ops = [
            RepairOperation(
                RepairOperationType.UPDATE_ATTR, "node_1", "Good rationale", 0.8
            )
        ]
        valid_set = RepairProposalSet(valid_ops, "VIO_002", "test_rule")

        assert valid_set.validate() is True
        assert valid_set.is_valid is True
        assert len(valid_set.validation_errors) == 0

        # Invalid proposals - missing target_id
        invalid_ops = [
            RepairOperation(
                RepairOperationType.UPDATE_ATTR,
                "",  # Empty target_id
                "Bad operation",
                0.5,
            )
        ]
        invalid_set = RepairProposalSet(invalid_ops, "VIO_003", "test_rule")

        assert invalid_set.validate() is False
        assert invalid_set.is_valid is False
        assert len(invalid_set.validation_errors) > 0

    def test_proposal_set_to_json_array(self):
        """Test JSON array serialization"""
        operations = [
            RepairOperation(
                RepairOperationType.DELETE_EDGE, "edge_1", "Delete operation", 0.9
            )
        ]

        proposal_set = RepairProposalSet(operations, "VIO_004", "test_rule")
        json_str = proposal_set.to_json_array()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["op"] == "delete_edge"
        assert parsed[0]["target_id"] == "edge_1"

    def test_high_confidence_filtering(self):
        """Test filtering high confidence proposals"""
        operations = [
            RepairOperation(RepairOperationType.DELETE_EDGE, "e1", "high conf", 0.9),
            RepairOperation(RepairOperationType.ADD_EDGE, "e2", "medium conf", 0.7),
            RepairOperation(RepairOperationType.UPDATE_ATTR, "n1", "low conf", 0.5),
        ]

        proposal_set = RepairProposalSet(operations, "VIO_005", "test_rule")
        high_conf_ops = proposal_set.get_high_confidence_proposals(threshold=0.8)

        assert len(high_conf_ops) == 1
        assert high_conf_ops[0].confidence == 0.9


class TestEnhancedLLMDriver:
    """Test suite for enhanced LLM Driver features"""

    @pytest.fixture
    def mock_config(self):
        """Mock model configuration"""
        return ModelConfig(
            model_name="test_model",
            backend=ModelBackend.OLLAMA,
            requests_per_minute=60,
            max_concurrent_requests=3,
        )

    def test_lmstudio_backend_creation(self, mock_config):
        """Test LMStudio backend creation"""
        mock_config.backend = ModelBackend.LMSTUDIO
        driver = LLMDriver(mock_config)

        assert driver.config.backend == ModelBackend.LMSTUDIO
        from mcp_servers.hyperag.repair.llm_driver import LMStudioBackend

        assert isinstance(driver.backend, LMStudioBackend)

    @pytest.mark.asyncio
    async def test_rate_limiting(self, mock_config):
        """Test rate limiting functionality"""
        mock_config.requests_per_minute = 2  # Very low limit for testing

        with patch(
            "mcp_servers.hyperag.repair.llm_driver.OllamaBackend"
        ) as MockBackend:
            mock_backend = AsyncMock()
            mock_response = GenerationResponse(
                text="test",
                finish_reason="completed",
                usage={"total_tokens": 10},
                model="test",
                latency_ms=100,
            )
            mock_backend.generate.return_value = mock_response
            MockBackend.return_value = mock_backend

            driver = LLMDriver(mock_config)

            # First two requests should succeed
            await driver.generate("test1")
            await driver.generate("test2")

            # Third request should be rate limited (will wait)
            start_time = asyncio.get_event_loop().time()
            await driver.generate("test3")
            end_time = asyncio.get_event_loop().time()

            # Should have been delayed by rate limiting
            assert len(driver._request_times) <= 2

    def test_audit_logging(self, mock_config):
        """Test audit logging functionality"""
        driver = LLMDriver(mock_config)

        # Create mock response
        mock_response = GenerationResponse(
            text="test response",
            finish_reason="completed",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            model="test_model",
            latency_ms=100.0,
        )

        # Log a request
        driver._log_request("test prompt", "system prompt", mock_response)

        # Check audit log
        audit_log = driver.get_audit_log()
        assert len(audit_log) == 1

        log_entry = audit_log[0]
        assert log_entry["model"] == "test_model"
        assert log_entry["prompt_length"] == len("test prompt")
        assert log_entry["usage"]["total_tokens"] == 15
        assert log_entry["latency_ms"] == 100.0

    def test_usage_stats(self, mock_config):
        """Test usage statistics"""
        driver = LLMDriver(mock_config)

        # Initially no stats
        stats = driver.get_usage_stats()
        assert "message" in stats

        # Add some audit entries
        for i in range(3):
            mock_response = GenerationResponse(
                text=f"response {i}",
                finish_reason="completed",
                usage={"total_tokens": 10 + i},
                model="test_model",
                latency_ms=100.0 + i * 10,
            )
            driver._log_request(f"prompt {i}", None, mock_response)

        stats = driver.get_usage_stats()
        assert stats["total_requests"] == 3
        assert stats["total_tokens_used"] == 33  # 10 + 11 + 12
        assert stats["average_latency_ms"] == 110.0  # (100 + 110 + 120) / 3


class TestEnhancedInnovatorAgent:
    """Test suite for enhanced Innovator Agent"""

    @pytest.fixture
    def mock_llm_driver(self):
        """Mock LLM driver with enhanced features"""
        driver = MagicMock()
        driver.config = MagicMock()
        driver.config.model_name = "test_model"
        return driver

    @pytest.fixture
    def agent(self, mock_llm_driver):
        """Create enhanced InnovatorAgent"""
        from mcp_servers.hyperag.repair.innovator_agent import PromptComposer
        from mcp_servers.hyperag.repair.templates import TemplateEncoder

        return InnovatorAgent(
            llm_driver=mock_llm_driver,
            template_encoder=TemplateEncoder(),
            prompt_composer=PromptComposer(),
            domain="general",
        )

    def test_enhanced_json_parsing_array_format(self, agent):
        """Test enhanced JSON parsing with array format"""
        # Test JSON array format (preferred)
        response_text = """Here are the repair operations:
[
  {"op":"delete_edge","target":"edge_1","rationale":"Remove bad edge","confidence":0.9},
  {"op":"add_edge","target":"edge_2","source":"node_1","type":"RELATES_TO","rationale":"Add good edge","confidence":0.8}
]
That's the complete set of operations."""

        operations = agent._parse_repair_operations_enhanced(response_text)

        assert len(operations) == 2
        assert operations[0].operation_type == RepairOperationType.DELETE_EDGE
        assert operations[0].target_id == "edge_1"
        assert operations[0].confidence == 0.9
        assert operations[1].operation_type == RepairOperationType.ADD_EDGE
        assert operations[1].source_id == "node_1"
        assert operations[1].relationship_type == "RELATES_TO"

    def test_enhanced_json_parsing_fallback_jsonl(self, agent):
        """Test fallback to JSONL parsing"""
        # Test JSONL format fallback
        response_text = """{"op":"delete_edge","target_id":"edge_1","rationale":"Remove bad edge","confidence":0.9}
{"op":"update_attr","target_id":"node_1","property_name":"status","property_value":"fixed","rationale":"Fix it","confidence":0.8}"""

        operations = agent._parse_repair_operations_enhanced(response_text)

        assert len(operations) == 2
        assert operations[0].operation_type == RepairOperationType.DELETE_EDGE
        assert operations[1].operation_type == RepairOperationType.UPDATE_ATTR

    def test_confidence_validation_in_parsing(self, agent):
        """Test confidence validation during parsing"""
        # Test with invalid confidence values
        response_text = """[
  {"op":"delete_edge","target":"edge_1","rationale":"test","confidence":"invalid"},
  {"op":"add_edge","target":"edge_2","rationale":"test","confidence":1.5},
  {"op":"update_attr","target":"node_1","rationale":"test","confidence":-0.1}
]"""

        operations = agent._parse_repair_operations_enhanced(response_text)

        assert len(operations) == 3
        # All should have normalized confidence values
        for op in operations:
            assert 0.0 <= op.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_generate_repair_proposal_set(self, agent, mock_llm_driver):
        """Test generating RepairProposalSet instead of RepairProposal"""
        # Mock violation data
        violation_data = {
            "violation_id": "VIO_007",
            "rule_name": "test_rule",
            "subgraph": {"nodes": [], "edges": []},
        }

        # Mock LLM response with JSON array
        mock_response = GenerationResponse(
            text='[{"op":"delete_edge","target":"edge_1","rationale":"Test rationale","confidence":0.8}]',
            finish_reason="completed",
            usage={"total_tokens": 50},
            model="test_model",
            latency_ms=100.0,
        )
        mock_llm_driver.generate.return_value = mock_response

        # Generate proposals
        proposal_set = await agent.generate_repair_proposals(violation_data)

        # Should return RepairProposalSet
        assert isinstance(proposal_set, RepairProposalSet)
        assert proposal_set.violation_id == "VIO_007"
        assert proposal_set.gdc_rule == "test_rule"
        assert len(proposal_set.proposals) == 1
        assert proposal_set.is_valid is True
        assert (
            proposal_set.proposals[0].operation_type == RepairOperationType.DELETE_EDGE
        )

    @pytest.mark.asyncio
    async def test_proposal_set_validation_integration(self, agent, mock_llm_driver):
        """Test that proposal sets are automatically validated"""
        violation_data = {
            "violation_id": "VIO_008",
            "rule_name": "test_rule",
            "subgraph": {"nodes": [], "edges": []},
        }

        # Mock LLM response with invalid operation
        mock_response = GenerationResponse(
            text='[{"op":"delete_edge","target":"","rationale":"Missing target","confidence":0.8}]',
            finish_reason="completed",
            usage={"total_tokens": 30},
            model="test_model",
            latency_ms=100.0,
        )
        mock_llm_driver.generate.return_value = mock_response

        proposal_set = await agent.generate_repair_proposals(violation_data)

        # Should be marked as invalid due to empty target_id
        assert proposal_set.is_valid is False
        assert len(proposal_set.validation_errors) > 0

    def test_lmstudio_default_creation(self):
        """Test creating agent with LMStudio backend"""
        config = ModelConfig(
            model_name="test_model",
            backend=ModelBackend.LMSTUDIO,
            api_endpoint="http://localhost:1234",
        )

        driver = LLMDriver(config)
        assert driver.config.backend == ModelBackend.LMSTUDIO

        # Test that the backend was created properly
        from mcp_servers.hyperag.repair.llm_driver import LMStudioBackend

        assert isinstance(driver.backend, LMStudioBackend)
        assert driver.backend.base_url == "http://localhost:1234"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
