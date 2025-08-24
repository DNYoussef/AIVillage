"""
Tests for ReasoningInterface

Behavioral tests ensuring interface compliance and contract validation.
"""

from abc import ABC
from datetime import datetime

import pytest

from core.rag.interfaces.reasoning_interface import (
    InferenceChainType,
    InferenceStep,
    ReasoningContext,
    ReasoningInterface,
    ReasoningMode,
    ReasoningResult,
)


class MockReasoningInterface(ReasoningInterface):
    """Mock implementation for testing interface compliance"""

    async def reason(self, query, premises, mode=None, context=None):
        return self._reason_mock

    async def build_inference_chain(self, premises, target_conclusion, chain_type=None, max_steps=10):
        return self._build_inference_chain_mock

    async def validate_reasoning(self, inference_chain, logical_rules=None):
        return self._validate_reasoning_mock

    async def explain_reasoning(self, reasoning_result, explanation_depth="summary"):
        return self._explain_reasoning_mock

    async def identify_assumptions(self, query, premises):
        return self._identify_assumptions_mock

    async def assess_uncertainty(self, inference_chain):
        return self._assess_uncertainty_mock

    async def get_reasoning_stats(self):
        return self._get_reasoning_stats_mock

    def __init__(self):
        self._reason_mock = None
        self._build_inference_chain_mock = None
        self._validate_reasoning_mock = None
        self._explain_reasoning_mock = None
        self._identify_assumptions_mock = None
        self._assess_uncertainty_mock = None
        self._get_reasoning_stats_mock = None


@pytest.fixture
def reasoning_interface():
    """Fixture providing mock reasoning interface"""
    return MockReasoningInterface()


@pytest.fixture
def sample_context():
    """Sample reasoning context for testing"""
    return ReasoningContext(
        user_id="test_user",
        session_id="test_session",
        domain="mathematics",
        confidence_threshold=0.8,
        max_inference_depth=3,
    )


@pytest.fixture
def sample_inference_step():
    """Sample inference step for testing"""
    return InferenceStep(
        step_id="step_1",
        premise="All humans are mortal",
        conclusion="Socrates is mortal",
        confidence=0.9,
        evidence=["Socrates is human"],
        rule_applied="universal_instantiation",
        metadata={"timestamp": datetime.now().isoformat()},
    )


class TestReasoningInterface:
    """Test reasoning interface contract and behavior"""

    def test_is_abstract_base_class(self):
        """Test that ReasoningInterface is properly abstract"""
        assert issubclass(ReasoningInterface, ABC)

        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            ReasoningInterface()

    @pytest.mark.asyncio
    async def test_reason_contract(self, reasoning_interface, sample_context):
        """Test reason method contract"""
        # Setup mock return
        expected_result = ReasoningResult(
            conclusion="Test conclusion",
            confidence_score=0.8,
            reasoning_mode=ReasoningMode.DEDUCTIVE,
            inference_chain=[],
            alternative_conclusions=["Alt 1"],
            uncertainty_factors=["Limited data"],
            metadata={},
        )
        reasoning_interface.reason.return_value = expected_result

        # Test method call
        result = await reasoning_interface.reason(
            query="Test query",
            premises=["Premise 1", "Premise 2"],
            mode=ReasoningMode.DEDUCTIVE,
            context=sample_context,
        )

        # Verify call and result
        reasoning_interface.reason.assert_called_once_with(
            "Test query", ["Premise 1", "Premise 2"], ReasoningMode.DEDUCTIVE, sample_context
        )
        assert result == expected_result
        assert isinstance(result, ReasoningResult)

    @pytest.mark.asyncio
    async def test_build_inference_chain_contract(self, reasoning_interface, sample_inference_step):
        """Test build_inference_chain method contract"""
        expected_chain = [sample_inference_step]
        reasoning_interface.build_inference_chain.return_value = expected_chain

        result = await reasoning_interface.build_inference_chain(
            premises=["All humans are mortal", "Socrates is human"],
            target_conclusion="Socrates is mortal",
            chain_type=InferenceChainType.LINEAR,
            max_steps=5,
        )

        reasoning_interface.build_inference_chain.assert_called_once()
        assert result == expected_chain
        assert isinstance(result, list)
        assert all(isinstance(step, InferenceStep) for step in result)

    @pytest.mark.asyncio
    async def test_validate_reasoning_contract(self, reasoning_interface, sample_inference_step):
        """Test validate_reasoning method contract"""
        expected_validation = {"is_valid": True, "consistency_score": 0.95, "logical_errors": [], "confidence": 0.9}
        reasoning_interface.validate_reasoning.return_value = expected_validation

        result = await reasoning_interface.validate_reasoning(
            inference_chain=[sample_inference_step], logical_rules=["modus_ponens", "universal_instantiation"]
        )

        reasoning_interface.validate_reasoning.assert_called_once()
        assert result == expected_validation
        assert isinstance(result, dict)
        assert "is_valid" in result

    @pytest.mark.asyncio
    async def test_explain_reasoning_contract(self, reasoning_interface):
        """Test explain_reasoning method contract"""
        reasoning_result = ReasoningResult(
            conclusion="Test conclusion",
            confidence_score=0.8,
            reasoning_mode=ReasoningMode.DEDUCTIVE,
            inference_chain=[],
            alternative_conclusions=[],
            uncertainty_factors=[],
            metadata={},
        )

        expected_explanation = "This is a test explanation of the reasoning process."
        reasoning_interface.explain_reasoning.return_value = expected_explanation

        result = await reasoning_interface.explain_reasoning(
            reasoning_result=reasoning_result, explanation_depth="summary"
        )

        reasoning_interface.explain_reasoning.assert_called_once()
        assert result == expected_explanation
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_identify_assumptions_contract(self, reasoning_interface):
        """Test identify_assumptions method contract"""
        expected_assumptions = ["Assumption 1", "Assumption 2"]
        reasoning_interface.identify_assumptions.return_value = expected_assumptions

        result = await reasoning_interface.identify_assumptions(query="Test query", premises=["Premise 1", "Premise 2"])

        reasoning_interface.identify_assumptions.assert_called_once()
        assert result == expected_assumptions
        assert isinstance(result, list)
        assert all(isinstance(assumption, str) for assumption in result)

    @pytest.mark.asyncio
    async def test_assess_uncertainty_contract(self, reasoning_interface, sample_inference_step):
        """Test assess_uncertainty method contract"""
        expected_uncertainty = {"data_quality": 0.8, "logical_consistency": 0.9, "premise_reliability": 0.7}
        reasoning_interface.assess_uncertainty.return_value = expected_uncertainty

        result = await reasoning_interface.assess_uncertainty(inference_chain=[sample_inference_step])

        reasoning_interface.assess_uncertainty.assert_called_once()
        assert result == expected_uncertainty
        assert isinstance(result, dict)
        assert all(isinstance(score, float) for score in result.values())

    @pytest.mark.asyncio
    async def test_get_reasoning_stats_contract(self, reasoning_interface):
        """Test get_reasoning_stats method contract"""
        expected_stats = {
            "total_reasoning_operations": 100,
            "average_confidence": 0.85,
            "success_rate": 0.92,
            "system_health": "healthy",
        }
        reasoning_interface.get_reasoning_stats.return_value = expected_stats

        result = await reasoning_interface.get_reasoning_stats()

        reasoning_interface.get_reasoning_stats.assert_called_once()
        assert result == expected_stats
        assert isinstance(result, dict)


class TestReasoningDataClasses:
    """Test reasoning data classes and enums"""

    def test_reasoning_mode_enum(self):
        """Test ReasoningMode enum values"""
        assert ReasoningMode.DEDUCTIVE.value == "deductive"
        assert ReasoningMode.INDUCTIVE.value == "inductive"
        assert ReasoningMode.ABDUCTIVE.value == "abductive"
        assert ReasoningMode.ANALOGICAL.value == "analogical"
        assert ReasoningMode.CAUSAL.value == "causal"

    def test_inference_chain_type_enum(self):
        """Test InferenceChainType enum values"""
        assert InferenceChainType.LINEAR.value == "linear"
        assert InferenceChainType.BRANCHING.value == "branching"
        assert InferenceChainType.HIERARCHICAL.value == "hierarchical"
        assert InferenceChainType.ITERATIVE.value == "iterative"

    def test_reasoning_context_creation(self):
        """Test ReasoningContext dataclass creation"""
        context = ReasoningContext(user_id="test", confidence_threshold=0.7, max_inference_depth=5)

        assert context.user_id == "test"
        assert context.confidence_threshold == 0.7
        assert context.max_inference_depth == 5
        assert context.session_id is None  # Default value

    def test_inference_step_creation(self, sample_inference_step):
        """Test InferenceStep dataclass creation"""
        step = sample_inference_step

        assert step.step_id == "step_1"
        assert step.premise == "All humans are mortal"
        assert step.conclusion == "Socrates is mortal"
        assert step.confidence == 0.9
        assert step.evidence == ["Socrates is human"]
        assert step.rule_applied == "universal_instantiation"
        assert isinstance(step.metadata, dict)

    def test_reasoning_result_creation(self):
        """Test ReasoningResult dataclass creation"""
        result = ReasoningResult(
            conclusion="Test conclusion",
            confidence_score=0.8,
            reasoning_mode=ReasoningMode.DEDUCTIVE,
            inference_chain=[],
            alternative_conclusions=["Alt 1"],
            uncertainty_factors=["Factor 1"],
            metadata={"key": "value"},
        )

        assert result.conclusion == "Test conclusion"
        assert result.confidence_score == 0.8
        assert result.reasoning_mode == ReasoningMode.DEDUCTIVE
        assert isinstance(result.inference_chain, list)
        assert result.alternative_conclusions == ["Alt 1"]
        assert result.uncertainty_factors == ["Factor 1"]
        assert result.metadata == {"key": "value"}


@pytest.mark.parametrize(
    "mode",
    [
        ReasoningMode.DEDUCTIVE,
        ReasoningMode.INDUCTIVE,
        ReasoningMode.ABDUCTIVE,
        ReasoningMode.ANALOGICAL,
        ReasoningMode.CAUSAL,
    ],
)
def test_reasoning_modes_parametrized(mode):
    """Parametrized test for all reasoning modes"""
    assert isinstance(mode, ReasoningMode)
    assert isinstance(mode.value, str)


@pytest.mark.parametrize(
    "chain_type",
    [
        InferenceChainType.LINEAR,
        InferenceChainType.BRANCHING,
        InferenceChainType.HIERARCHICAL,
        InferenceChainType.ITERATIVE,
    ],
)
def test_inference_chain_types_parametrized(chain_type):
    """Parametrized test for all inference chain types"""
    assert isinstance(chain_type, InferenceChainType)
    assert isinstance(chain_type.value, str)
