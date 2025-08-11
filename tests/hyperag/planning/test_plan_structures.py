"""
Unit tests for Planning Data Structures
"""

from pathlib import Path
import sys

import pytest

from mcp_servers.hyperag.planning.plan_structures import (
    ExecutionStatus,
    ExecutionStep,
    PlanCheckpoint,
    PlanDSL,
    QueryPlan,
    QueryType,
    ReasoningStrategy,
    RetrievalConstraints,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestRetrievalConstraints:
    """Test suite for RetrievalConstraints"""

    def test_default_constraints(self):
        """Test default constraint values"""
        constraints = RetrievalConstraints()

        assert constraints.max_depth == 3
        assert constraints.max_nodes == 100
        assert constraints.confidence_threshold == 0.7
        assert constraints.time_budget_ms == 5000
        assert constraints.include_explanations is True
        assert constraints.prefer_recent is False
        assert constraints.domain_filter is None
        assert constraints.exclude_uncertainty is False

    def test_custom_constraints(self):
        """Test custom constraint values"""
        constraints = RetrievalConstraints(
            max_depth=5,
            max_nodes=200,
            confidence_threshold=0.8,
            time_budget_ms=10000,
            include_explanations=False,
            prefer_recent=True,
            domain_filter="science",
            exclude_uncertainty=True,
        )

        assert constraints.max_depth == 5
        assert constraints.max_nodes == 200
        assert constraints.confidence_threshold == 0.8
        assert constraints.time_budget_ms == 10000
        assert constraints.include_explanations is False
        assert constraints.prefer_recent is True
        assert constraints.domain_filter == "science"
        assert constraints.exclude_uncertainty is True


class TestExecutionStep:
    """Test suite for ExecutionStep"""

    def test_step_creation(self):
        """Test execution step creation"""
        step = ExecutionStep(
            step_type="retrieval",
            description="Test retrieval step",
            operation="semantic_search",
            parameters={"query": "test", "limit": 10},
            dependencies=["step1", "step2"],
            confidence_threshold=0.8,
            timeout_ms=2000,
        )

        assert step.step_type == "retrieval"
        assert step.description == "Test retrieval step"
        assert step.operation == "semantic_search"
        assert step.parameters["query"] == "test"
        assert step.dependencies == ["step1", "step2"]
        assert step.confidence_threshold == 0.8
        assert step.timeout_ms == 2000
        assert step.status == ExecutionStatus.PENDING
        assert step.step_id is not None

    def test_step_dependency_check(self):
        """Test step dependency checking"""
        step = ExecutionStep(dependencies=["step1", "step2"])

        # Not ready - missing dependencies
        assert not step.is_ready_to_execute(set())
        assert not step.is_ready_to_execute({"step1"})

        # Ready - all dependencies satisfied
        assert step.is_ready_to_execute({"step1", "step2"})
        assert step.is_ready_to_execute({"step1", "step2", "step3"})

    def test_step_status_transitions(self):
        """Test step status transitions"""
        step = ExecutionStep()

        # Initial state
        assert step.status == ExecutionStatus.PENDING
        assert step.started_at is None
        assert step.completed_at is None

        # Mark started
        step.mark_started()
        assert step.status == ExecutionStatus.IN_PROGRESS
        assert step.started_at is not None

        # Mark completed
        step.mark_completed("test_output", 0.85)
        assert step.status == ExecutionStatus.COMPLETED
        assert step.actual_output == "test_output"
        assert step.confidence_score == 0.85
        assert step.completed_at is not None
        assert step.execution_time_ms > 0

    def test_step_failure(self):
        """Test step failure handling"""
        step = ExecutionStep()

        step.mark_failed("Network timeout")

        assert step.status == ExecutionStatus.FAILED
        assert step.error_message == "Network timeout"
        assert step.completed_at is not None


class TestPlanCheckpoint:
    """Test suite for PlanCheckpoint"""

    def test_checkpoint_creation(self):
        """Test checkpoint creation"""
        checkpoint = PlanCheckpoint(
            step_index=2,
            completed_steps={"step1", "step2"},
            intermediate_results={"key": "value"},
            aggregate_confidence=0.75,
            execution_time_ms=1500.0,
        )

        assert checkpoint.step_index == 2
        assert checkpoint.completed_steps == {"step1", "step2"}
        assert checkpoint.intermediate_results["key"] == "value"
        assert checkpoint.aggregate_confidence == 0.75
        assert checkpoint.execution_time_ms == 1500.0
        assert checkpoint.checkpoint_id is not None
        assert checkpoint.created_at is not None

    def test_checkpoint_rollback(self):
        """Test checkpoint rollback capability"""
        checkpoint = PlanCheckpoint(completed_steps={"step1", "step2", "step3"})

        assert checkpoint.can_rollback_to("step1")
        assert checkpoint.can_rollback_to("step2")
        assert not checkpoint.can_rollback_to("step4")


class TestQueryPlan:
    """Test suite for QueryPlan"""

    def test_plan_creation(self):
        """Test query plan creation"""
        constraints = RetrievalConstraints(max_depth=5)

        plan = QueryPlan(
            original_query="test query",
            query_type=QueryType.CAUSAL_CHAIN,
            reasoning_strategy=ReasoningStrategy.CAUSAL_REASONING,
            retrieval_constraints=constraints,
        )

        assert plan.original_query == "test query"
        assert plan.query_type == QueryType.CAUSAL_CHAIN
        assert plan.reasoning_strategy == ReasoningStrategy.CAUSAL_REASONING
        assert plan.retrieval_constraints.max_depth == 5
        assert plan.plan_id is not None
        assert plan.created_at is not None
        assert plan.status == ExecutionStatus.PENDING

    def test_plan_step_management(self):
        """Test plan step management"""
        plan = QueryPlan()

        # Add steps
        step1 = ExecutionStep(step_type="retrieval", description="Step 1")
        step2 = ExecutionStep(step_type="reasoning", description="Step 2")

        plan.add_step(step1)
        plan.add_step(step2)

        assert len(plan.execution_steps) == 2
        assert plan.expected_steps == 2

        # Find step by ID
        found_step = plan.get_step_by_id(step1.step_id)
        assert found_step == step1

        # Non-existent step
        assert plan.get_step_by_id("nonexistent") is None

    def test_plan_completion_status(self):
        """Test plan completion status tracking"""
        plan = QueryPlan()

        step1 = ExecutionStep(status=ExecutionStatus.COMPLETED)
        step2 = ExecutionStep(status=ExecutionStatus.COMPLETED)
        step3 = ExecutionStep(status=ExecutionStatus.PENDING)

        plan.add_step(step1)
        plan.add_step(step2)
        plan.add_step(step3)

        # Not complete with pending step
        assert not plan.is_complete()
        assert not plan.has_failed_steps()

        # Complete when all steps done
        step3.status = ExecutionStatus.COMPLETED
        assert plan.is_complete()

        # Has failed steps
        step3.status = ExecutionStatus.FAILED
        assert plan.has_failed_steps()
        assert not plan.is_complete()

    def test_plan_ready_steps(self):
        """Test finding ready-to-execute steps"""
        plan = QueryPlan()

        step1 = ExecutionStep(step_id="step1", dependencies=[])
        step2 = ExecutionStep(step_id="step2", dependencies=["step1"])
        step3 = ExecutionStep(step_id="step3", dependencies=["step1", "step2"])

        plan.add_step(step1)
        plan.add_step(step2)
        plan.add_step(step3)

        # Initially only step1 is ready
        ready_step = plan.get_next_ready_step(set())
        assert ready_step == step1

        # After step1 completes, step2 is ready
        ready_step = plan.get_next_ready_step({"step1"})
        assert ready_step == step2

        # After both complete, step3 is ready
        ready_step = plan.get_next_ready_step({"step1", "step2"})
        assert ready_step == step3

        # No more ready steps
        ready_step = plan.get_next_ready_step({"step1", "step2", "step3"})
        assert ready_step is None

    def test_plan_completed_steps(self):
        """Test getting completed steps"""
        plan = QueryPlan()

        step1 = ExecutionStep(step_id="step1", status=ExecutionStatus.COMPLETED)
        step2 = ExecutionStep(step_id="step2", status=ExecutionStatus.PENDING)
        step3 = ExecutionStep(step_id="step3", status=ExecutionStatus.COMPLETED)

        plan.add_step(step1)
        plan.add_step(step2)
        plan.add_step(step3)

        completed = plan.get_completed_steps()
        assert completed == {"step1", "step3"}

    def test_plan_checkpoint_creation(self):
        """Test plan checkpoint creation"""
        plan = QueryPlan(current_step_index=2, overall_confidence=0.8)

        completed_steps = {"step1", "step2"}
        intermediate_results = {"result1": "value1"}

        checkpoint = plan.create_checkpoint(completed_steps, intermediate_results)

        assert checkpoint.step_index == 2
        assert checkpoint.completed_steps == completed_steps
        assert checkpoint.intermediate_results == intermediate_results
        assert checkpoint.aggregate_confidence == 0.8
        assert len(plan.checkpoints) == 1

    def test_plan_confidence_calculation(self):
        """Test overall confidence calculation"""
        plan = QueryPlan()

        step1 = ExecutionStep(status=ExecutionStatus.COMPLETED, confidence_score=0.8)
        step2 = ExecutionStep(status=ExecutionStatus.COMPLETED, confidence_score=0.9)
        step3 = ExecutionStep(status=ExecutionStatus.PENDING)

        plan.add_step(step1)
        plan.add_step(step2)
        plan.add_step(step3)

        confidence = plan.calculate_overall_confidence()
        expected = (0.8 + 0.9) / 2  # Average of completed steps
        assert abs(confidence - expected) < 0.01

    def test_plan_serialization(self):
        """Test plan serialization to dictionary"""
        plan = QueryPlan(
            original_query="test query",
            query_type=QueryType.TEMPORAL_ANALYSIS,
            reasoning_strategy=ReasoningStrategy.TEMPORAL_REASONING,
            overall_confidence=0.85,
            complexity_score=0.6,
        )

        step = ExecutionStep(
            step_type="retrieval",
            description="Test step",
            operation="test_op",
            confidence_score=0.8,
        )
        plan.add_step(step)

        plan_dict = plan.to_dict()

        assert plan_dict["original_query"] == "test query"
        assert plan_dict["query_type"] == "temporal_analysis"
        assert plan_dict["reasoning_strategy"] == "temporal_reasoning"
        assert plan_dict["overall_confidence"] == 0.85
        assert plan_dict["complexity_score"] == 0.6
        assert len(plan_dict["execution_steps"]) == 1
        assert plan_dict["execution_steps"][0]["step_type"] == "retrieval"

    def test_plan_deserialization(self):
        """Test plan deserialization from dictionary"""
        plan_data = {
            "plan_id": "test-plan-123",
            "original_query": "test query",
            "query_type": "causal_chain",
            "reasoning_strategy": "causal_reasoning",
            "status": "pending",
            "overall_confidence": 0.75,
            "complexity_score": 0.7,
            "replan_count": 1,
            "created_at": "2025-01-01T12:00:00",
            "execution_steps": [
                {
                    "step_id": "step1",
                    "step_type": "retrieval",
                    "description": "Test step",
                    "operation": "test_op",
                    "parameters": {"key": "value"},
                    "dependencies": [],
                    "status": "completed",
                    "confidence_score": 0.8,
                }
            ],
        }

        plan = QueryPlan.from_dict(plan_data)

        assert plan.plan_id == "test-plan-123"
        assert plan.original_query == "test query"
        assert plan.query_type == QueryType.CAUSAL_CHAIN
        assert plan.reasoning_strategy == ReasoningStrategy.CAUSAL_REASONING
        assert plan.overall_confidence == 0.75
        assert plan.complexity_score == 0.7
        assert plan.replan_count == 1
        assert len(plan.execution_steps) == 1
        assert plan.execution_steps[0].step_id == "step1"


class TestPlanDSL:
    """Test suite for Plan DSL"""

    def test_plan_serialization_to_dsl(self):
        """Test plan serialization to DSL format"""
        constraints = RetrievalConstraints(max_depth=3, max_nodes=50, confidence_threshold=0.8, time_budget_ms=3000)

        plan = QueryPlan(
            plan_id="test-plan-123",
            original_query="What causes climate change?",
            query_type=QueryType.CAUSAL_CHAIN,
            reasoning_strategy=ReasoningStrategy.CAUSAL_REASONING,
            complexity_score=0.7,
            retrieval_constraints=constraints,
        )

        step1 = ExecutionStep(
            description="Identify causal entities",
            operation="causal_extraction",
            dependencies=[],
            confidence_threshold=0.8,
        )
        step2 = ExecutionStep(
            description="Analyze causal relationships",
            operation="causal_analysis",
            dependencies=[step1.step_id],
            confidence_threshold=0.7,
        )

        plan.add_step(step1)
        plan.add_step(step2)

        dsl_text = PlanDSL.serialize_plan(plan)

        # Check that DSL contains expected elements
        assert "PLAN test-plan-123" in dsl_text
        assert "QUERY: What causes climate change?" in dsl_text
        assert "TYPE: causal_chain" in dsl_text
        assert "STRATEGY: causal_reasoning" in dsl_text
        assert "COMPLEXITY: 0.7" in dsl_text
        assert "CONSTRAINTS:" in dsl_text
        assert "max_depth: 3" in dsl_text
        assert "STEPS:" in dsl_text
        assert "Identify causal entities" in dsl_text
        assert "Analyze causal relationships" in dsl_text

    def test_plan_parsing_from_dsl(self):
        """Test plan parsing from DSL format"""
        dsl_text = """
        PLAN test-plan-456
        QUERY: How do ecosystems adapt to climate change?
        TYPE: temporal_analysis
        STRATEGY: temporal_reasoning
        COMPLEXITY: 0.6
        """

        plan = PlanDSL.parse_plan(dsl_text)

        assert plan.plan_id == "test-plan-456"
        assert plan.original_query == "How do ecosystems adapt to climate change?"
        # Note: This is a simplified parser, full implementation would set all fields


class TestEnumValues:
    """Test suite for enum values"""

    def test_query_type_values(self):
        """Test QueryType enum values"""
        assert QueryType.SIMPLE_FACT.value == "simple_fact"
        assert QueryType.TEMPORAL_ANALYSIS.value == "temporal_analysis"
        assert QueryType.CAUSAL_CHAIN.value == "causal_chain"
        assert QueryType.COMPARATIVE.value == "comparative"
        assert QueryType.META_KNOWLEDGE.value == "meta_knowledge"
        assert QueryType.MULTI_HOP.value == "multi_hop"
        assert QueryType.AGGREGATION.value == "aggregation"
        assert QueryType.HYPOTHETICAL.value == "hypothetical"

    def test_reasoning_strategy_values(self):
        """Test ReasoningStrategy enum values"""
        assert ReasoningStrategy.DIRECT_RETRIEVAL.value == "direct_retrieval"
        assert ReasoningStrategy.STEP_BY_STEP.value == "step_by_step"
        assert ReasoningStrategy.GRAPH_TRAVERSAL.value == "graph_traversal"
        assert ReasoningStrategy.TEMPORAL_REASONING.value == "temporal_reasoning"
        assert ReasoningStrategy.CAUSAL_REASONING.value == "causal_reasoning"
        assert ReasoningStrategy.COMPARATIVE_ANALYSIS.value == "comparative_analysis"
        assert ReasoningStrategy.META_REASONING.value == "meta_reasoning"
        assert ReasoningStrategy.HYBRID.value == "hybrid"

    def test_execution_status_values(self):
        """Test ExecutionStatus enum values"""
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.IN_PROGRESS.value == "in_progress"
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert ExecutionStatus.FAILED.value == "failed"
        assert ExecutionStatus.REPLANNED.value == "replanned"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
