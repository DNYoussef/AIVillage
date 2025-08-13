"""
Unit tests for Query Planning System
"""

from pathlib import Path
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_servers.hyperag.planning.plan_structures import (
    ExecutionStatus,
    QueryPlan,
    QueryType,
    ReasoningStrategy,
    RetrievalConstraints,
)
from mcp_servers.hyperag.planning.query_planner import AgentReasoningModel, QueryPlanner

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestAgentReasoningModel:
    """Test suite for AgentReasoningModel"""

    def test_agent_model_creation(self):
        """Test creation of agent reasoning model"""
        model = AgentReasoningModel(
            model_name="test_model",
            capabilities=["reasoning", "retrieval"],
            performance_profile={
                "max_complexity": 0.8,
                "reasoning_speed": 1.2,
                "accuracy": 0.9,
                "memory_limit_mb": 1000,
            },
        )

        assert model.model_name == "test_model"
        assert "reasoning" in model.capabilities
        assert model.max_complexity == 0.8
        assert model.reasoning_speed == 1.2
        assert model.accuracy == 0.9
        assert model.memory_limit_mb == 1000

    def test_agent_model_defaults(self):
        """Test default values for agent model"""
        model = AgentReasoningModel(
            model_name="test_model", capabilities=[], performance_profile={}
        )

        assert model.max_complexity == 0.8  # Default
        assert model.reasoning_speed == 1.0  # Default
        assert model.accuracy == 0.85  # Default
        assert model.memory_limit_mb == 1000  # Default


class TestQueryPlanner:
    """Test suite for QueryPlanner"""

    @pytest.fixture
    def mock_classifier(self):
        """Mock query classifier"""
        classifier = MagicMock()
        classifier.classify_query.return_value = (
            QueryType.SIMPLE_FACT,
            0.8,
            {"complexity_score": 0.3},
        )
        return classifier

    @pytest.fixture
    def mock_strategy_selector(self):
        """Mock strategy selector"""
        selector = MagicMock()
        selector.select_strategy.return_value = ReasoningStrategy.DIRECT_RETRIEVAL
        selector.create_strategy_instance.return_value = MagicMock()
        return selector

    @pytest.fixture
    def planner(self, mock_classifier, mock_strategy_selector):
        """Create QueryPlanner with mocked dependencies"""
        return QueryPlanner(
            classifier=mock_classifier, strategy_selector=mock_strategy_selector
        )

    @pytest.fixture
    def agent_model(self):
        """Create test agent model"""
        return AgentReasoningModel(
            model_name="test_agent",
            capabilities=["reasoning", "retrieval"],
            performance_profile={
                "max_complexity": 0.8,
                "reasoning_speed": 1.0,
                "accuracy": 0.85,
                "memory_limit_mb": 1000,
            },
        )

    def test_planner_initialization(self):
        """Test planner initialization"""
        planner = QueryPlanner()

        assert planner.classifier is not None
        assert planner.strategy_selector is not None
        assert planner.default_constraints is not None
        assert planner.max_replan_attempts == 3
        assert planner.confidence_threshold_for_replan == 0.5
        assert planner.performance_metrics["plans_created"] == 0

    @pytest.mark.asyncio
    async def test_create_plan_basic(
        self, planner, agent_model, mock_strategy_selector
    ):
        """Test basic plan creation"""

        # Mock strategy instance
        mock_strategy = AsyncMock()
        mock_plan = QueryPlan(
            original_query="test query",
            query_type=QueryType.SIMPLE_FACT,
            reasoning_strategy=ReasoningStrategy.DIRECT_RETRIEVAL,
        )
        mock_strategy.create_plan.return_value = mock_plan
        mock_strategy_selector.create_strategy_instance.return_value = mock_strategy

        # Create plan
        plan = await planner.create_plan("test query", agent_model)

        assert plan is not None
        assert plan.original_query == "test query"
        assert plan.agent_model == "test_agent"
        assert plan.complexity_score > 0
        assert plan.overall_confidence > 0

    @pytest.mark.asyncio
    async def test_create_plan_with_constraints(self, planner, agent_model):
        """Test plan creation with custom constraints"""

        constraints = RetrievalConstraints(
            max_depth=5, max_nodes=200, confidence_threshold=0.8, time_budget_ms=10000
        )

        # Mock strategy
        mock_strategy = AsyncMock()
        mock_plan = QueryPlan(
            original_query="test query",
            query_type=QueryType.SIMPLE_FACT,
            reasoning_strategy=ReasoningStrategy.DIRECT_RETRIEVAL,
            retrieval_constraints=constraints,
        )
        mock_strategy.create_plan.return_value = mock_plan
        planner.strategy_selector.create_strategy_instance.return_value = mock_strategy

        plan = await planner.create_plan(
            "test query", agent_model, constraints=constraints
        )

        assert plan.retrieval_constraints.max_depth <= 5
        assert plan.retrieval_constraints.time_budget_ms <= 10000

    @pytest.mark.asyncio
    async def test_create_plan_agent_limits(self, planner, mock_strategy_selector):
        """Test plan creation respects agent limits"""

        # Create agent with low limits
        limited_agent = AgentReasoningModel(
            model_name="limited_agent",
            capabilities=["basic_retrieval"],
            performance_profile={
                "max_complexity": 0.3,  # Very low
                "reasoning_speed": 0.5,  # Slow
                "accuracy": 0.7,  # Lower accuracy
                "memory_limit_mb": 100,  # Low memory
            },
        )

        # Mock high complexity classification
        planner.classifier.classify_query.return_value = (
            QueryType.CAUSAL_CHAIN,
            0.9,
            {"complexity_score": 0.9},
        )

        mock_strategy = AsyncMock()
        mock_plan = QueryPlan(original_query="complex query")
        mock_strategy.create_plan.return_value = mock_plan
        mock_strategy_selector.create_strategy_instance.return_value = mock_strategy

        plan = await planner.create_plan("complex query", limited_agent)

        # Should adjust constraints for limited agent
        assert plan.retrieval_constraints.max_nodes <= 100  # Adjusted for memory
        assert plan.retrieval_constraints.time_budget_ms <= 2500  # Adjusted for speed

    @pytest.mark.asyncio
    async def test_create_plan_error_handling(self, planner, agent_model):
        """Test plan creation error handling"""

        # Force classifier to raise exception
        planner.classifier.classify_query.side_effect = Exception(
            "Classification failed"
        )

        # Should create fallback plan
        plan = await planner.create_plan("test query", agent_model)

        assert plan is not None
        assert plan.original_query == "test query"
        assert plan.complexity_score > 0

        # Should record failure in metrics
        assert planner.performance_metrics["plans_created"] > 0

    @pytest.mark.asyncio
    async def test_replan_basic(self, planner, agent_model):
        """Test basic replanning functionality"""

        # Create original plan
        original_plan = QueryPlan(
            original_query="test query",
            query_type=QueryType.CAUSAL_CHAIN,
            reasoning_strategy=ReasoningStrategy.CAUSAL_REASONING,
            replan_count=0,
        )

        # Mock strategy for replan
        mock_strategy = AsyncMock()
        new_plan = QueryPlan(
            original_query="test query",
            query_type=QueryType.CAUSAL_CHAIN,
            reasoning_strategy=ReasoningStrategy.DIRECT_RETRIEVAL,
        )
        mock_strategy.create_plan.return_value = new_plan
        planner.strategy_selector.create_strategy_instance.return_value = mock_strategy

        # Replan due to low confidence
        replanned = await planner.replan(
            original_plan,
            intermediate_results={},
            current_confidence=0.3,
            failure_reason="low_confidence",
        )

        assert replanned is not None
        assert replanned.replan_count == 1
        assert replanned.parent_plan_id == original_plan.plan_id
        assert replanned.adaptation_reason == "low_confidence"

    @pytest.mark.asyncio
    async def test_replan_max_attempts(self, planner, agent_model):
        """Test replan max attempts limit"""

        # Create plan with max replans already
        original_plan = QueryPlan(
            original_query="test query",
            replan_count=3,  # At max limit
        )

        # Should not replan further
        result = await planner.replan(
            original_plan, intermediate_results={}, current_confidence=0.1
        )

        assert result == original_plan  # Returns original plan
        assert result.replan_count == 3  # Unchanged

    @pytest.mark.asyncio
    async def test_constraint_adjustment(self, planner):
        """Test constraint adjustment for agent capabilities"""

        base_constraints = RetrievalConstraints(
            max_depth=10, max_nodes=1000, confidence_threshold=0.5, time_budget_ms=5000
        )

        # Low-capability agent
        low_agent = AgentReasoningModel(
            model_name="low_agent",
            capabilities=[],
            performance_profile={
                "max_complexity": 0.4,
                "reasoning_speed": 0.5,
                "accuracy": 0.7,
                "memory_limit_mb": 200,
            },
        )

        adjusted = planner._adjust_constraints_for_agent(
            base_constraints, low_agent, complexity_score=0.8
        )

        # Should reduce limits
        assert adjusted.max_depth <= base_constraints.max_depth
        assert adjusted.max_nodes <= base_constraints.max_nodes
        assert adjusted.confidence_threshold >= base_constraints.confidence_threshold
        assert adjusted.time_budget_ms <= base_constraints.time_budget_ms

    @pytest.mark.asyncio
    async def test_plan_validation(self, planner, agent_model):
        """Test plan validation and optimization"""

        # Create plan with high time requirements
        plan = QueryPlan(
            original_query="test query",
            retrieval_constraints=RetrievalConstraints(time_budget_ms=1000),
        )

        # Add steps with high timeouts
        from mcp_servers.hyperag.planning.plan_structures import ExecutionStep

        step1 = ExecutionStep(
            step_type="retrieval",
            description="Step 1",
            operation="test_op",
            timeout_ms=800,
        )
        step2 = ExecutionStep(
            step_type="reasoning",
            description="Step 2",
            operation="test_op",
            timeout_ms=800,
        )
        plan.add_step(step1)
        plan.add_step(step2)

        # Validate plan
        validated_plan = await planner._validate_and_optimize_plan(plan, agent_model)

        # Should adjust timeouts to fit budget
        total_timeout = sum(step.timeout_ms for step in validated_plan.execution_steps)
        assert total_timeout <= plan.retrieval_constraints.time_budget_ms

    def test_planning_metrics(self, planner):
        """Test planning metrics recording"""

        plan = QueryPlan(original_query="test query", overall_confidence=0.8)

        # Record successful planning
        planner._record_planning_metrics(plan, 100.0, True)

        assert planner.performance_metrics["plans_created"] == 1
        assert planner.performance_metrics["successful_plans"] == 1
        assert planner.performance_metrics["avg_plan_confidence"] > 0

        # Record failed planning
        failed_plan = QueryPlan(overall_confidence=0.3)
        planner._record_planning_metrics(failed_plan, 200.0, False)

        assert planner.performance_metrics["plans_created"] == 2
        assert planner.performance_metrics["successful_plans"] == 1  # Still 1

    def test_planning_stats(self, planner):
        """Test planning statistics generation"""

        # Initially empty stats
        stats = planner.get_planning_stats()
        assert "message" in stats

        # Add some planning history
        planner.planning_history = [
            {
                "query": "test1",
                "query_type": "simple_fact",
                "strategy": "direct_retrieval",
                "complexity": 0.3,
                "confidence": 0.8,
                "planning_time_ms": 100,
                "success": True,
                "step_count": 1,
            },
            {
                "query": "test2",
                "query_type": "causal_chain",
                "strategy": "causal_reasoning",
                "complexity": 0.7,
                "confidence": 0.6,
                "planning_time_ms": 200,
                "success": False,
                "step_count": 3,
            },
        ]
        planner.performance_metrics["plans_created"] = 2
        planner.performance_metrics["successful_plans"] = 1

        stats = planner.get_planning_stats()

        assert stats["total_plans"] == 2
        assert stats["success_rate"] == 0.5
        assert stats["recent_avg_complexity"] > 0
        assert stats["recent_avg_steps"] > 0
        assert "strategy_distribution" in stats

    @pytest.mark.asyncio
    async def test_fallback_plan_creation(self, planner):
        """Test fallback plan creation"""

        constraints = RetrievalConstraints()

        with patch(
            "mcp_servers.hyperag.planning.strategies.SimpleFactStrategy"
        ) as mock_strategy_class:
            mock_strategy = AsyncMock()
            mock_plan = QueryPlan(
                original_query="fallback query",
                complexity_score=0.1,
                overall_confidence=0.6,
            )
            mock_strategy.create_plan.return_value = mock_plan
            mock_strategy_class.return_value = mock_strategy

            fallback_plan = await planner._create_fallback_plan(
                "fallback query", constraints
            )

            assert fallback_plan.original_query == "fallback query"
            assert fallback_plan.complexity_score == 0.1
            assert fallback_plan.overall_confidence == 0.6

    @pytest.mark.asyncio
    async def test_replan_analysis(self, planner):
        """Test replan needs analysis"""

        # Create plan with failed steps
        plan = QueryPlan(
            original_query="test query",
            reasoning_strategy=ReasoningStrategy.CAUSAL_REASONING,
            overall_confidence=0.8,
        )

        from mcp_servers.hyperag.planning.plan_structures import ExecutionStep

        failed_step = ExecutionStep(
            step_type="retrieval",
            description="Failed step",
            operation="test_op",
            status=ExecutionStatus.FAILED,
            error_message="Network error",
        )
        success_step = ExecutionStep(
            step_type="reasoning",
            description="Success step",
            operation="test_op",
            status=ExecutionStatus.COMPLETED,
            confidence_score=0.9,
        )
        plan.add_step(failed_step)
        plan.add_step(success_step)

        analysis = await planner._analyze_replan_needs(
            plan,
            intermediate_results={"key": "value"},
            current_confidence=0.3,
            failure_reason="step_failure",
        )

        assert analysis["original_strategy"] == ReasoningStrategy.CAUSAL_REASONING
        assert len(analysis["failed_steps"]) == 1
        assert len(analysis["successful_steps"]) == 1
        assert analysis["confidence_drop"] == 0.5  # 0.8 - 0.3
        assert analysis["failure_reason"] == "step_failure"
        assert "replan_type" in analysis


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
