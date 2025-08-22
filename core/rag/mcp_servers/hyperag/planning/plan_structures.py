"""Planning Data Structures.

Core data structures for representing query plans, execution steps, and checkpoints.
"""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np


class QueryType(Enum):
    """Types of queries classified by reasoning requirements."""

    SIMPLE_FACT = "simple_fact"  # Basic factual lookup
    TEMPORAL_ANALYSIS = "temporal_analysis"  # Time-based reasoning
    CAUSAL_CHAIN = "causal_chain"  # Cause-effect relationships
    COMPARATIVE = "comparative"  # Comparison and contrast
    META_KNOWLEDGE = "meta_knowledge"  # Questions about knowledge itself
    MULTI_HOP = "multi_hop"  # Complex multi-step reasoning
    AGGREGATION = "aggregation"  # Statistical/aggregative queries
    HYPOTHETICAL = "hypothetical"  # What-if scenarios


class ReasoningStrategy(Enum):
    """Available reasoning strategies."""

    DIRECT_RETRIEVAL = "direct_retrieval"
    STEP_BY_STEP = "step_by_step"
    GRAPH_TRAVERSAL = "graph_traversal"
    TEMPORAL_REASONING = "temporal_reasoning"
    CAUSAL_REASONING = "causal_reasoning"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    META_REASONING = "meta_reasoning"
    HYBRID = "hybrid"


class ExecutionStatus(Enum):
    """Status of plan execution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REPLANNED = "replanned"


@dataclass
class RetrievalConstraints:
    """Constraints for knowledge retrieval during planning."""

    max_depth: int = 3  # Maximum reasoning depth
    max_nodes: int = 100  # Maximum nodes to retrieve
    confidence_threshold: float = 0.7  # Minimum confidence required
    time_budget_ms: int = 5000  # Time budget for execution
    include_explanations: bool = True  # Include reasoning explanations
    prefer_recent: bool = False  # Prefer recent information
    domain_filter: str | None = None  # Domain-specific filtering
    exclude_uncertainty: bool = False  # Exclude uncertain information


@dataclass
class ExecutionStep:
    """Individual step in query execution plan."""

    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_type: str = ""  # Type of operation (retrieve, reason, verify)
    description: str = ""  # Human-readable description
    operation: str = ""  # Specific operation to perform
    parameters: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)  # Step IDs this depends on
    expected_output: str = ""  # Expected output description
    confidence_threshold: float = 0.7  # Required confidence for success
    timeout_ms: int = 1000  # Step timeout
    retry_count: int = 0  # Number of retries allowed
    status: ExecutionStatus = ExecutionStatus.PENDING

    # Execution results
    actual_output: Any | None = None
    confidence_score: float = 0.0
    execution_time_ms: float = 0.0
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def is_ready_to_execute(self, completed_steps: set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep_id in completed_steps for dep_id in self.dependencies)

    def mark_started(self) -> None:
        """Mark step as started."""
        self.status = ExecutionStatus.IN_PROGRESS
        self.started_at = datetime.now(UTC)

    def mark_completed(self, output: Any, confidence: float) -> None:
        """Mark step as completed with results."""
        self.status = ExecutionStatus.COMPLETED
        self.actual_output = output
        self.confidence_score = confidence
        self.completed_at = datetime.now(UTC)

        if self.started_at:
            delta = self.completed_at - self.started_at
            self.execution_time_ms = delta.total_seconds() * 1000

    def mark_failed(self, error: str) -> None:
        """Mark step as failed."""
        self.status = ExecutionStatus.FAILED
        self.error_message = error
        self.completed_at = datetime.now(UTC)


@dataclass
class PlanCheckpoint:
    """Checkpoint for plan execution state."""

    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_index: int = 0  # Index of current step
    completed_steps: set[str] = field(default_factory=set)
    intermediate_results: dict[str, Any] = field(default_factory=dict)
    aggregate_confidence: float = 1.0  # Confidence so far
    execution_time_ms: float = 0.0  # Time elapsed
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # State for rollback
    knowledge_graph_state: dict[str, Any] | None = None
    retrieval_context: dict[str, Any] = field(default_factory=dict)

    def can_rollback_to(self, target_step: str) -> bool:
        """Check if we can rollback to a specific step."""
        return target_step in self.completed_steps


@dataclass
class QueryPlan:
    """Complete execution plan for a query."""

    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_query: str = ""
    query_type: QueryType = QueryType.SIMPLE_FACT
    reasoning_strategy: ReasoningStrategy = ReasoningStrategy.DIRECT_RETRIEVAL

    # Plan structure
    execution_steps: list[ExecutionStep] = field(default_factory=list)
    checkpoints: list[PlanCheckpoint] = field(default_factory=list)
    retrieval_constraints: RetrievalConstraints = field(default_factory=RetrievalConstraints)

    # Execution state
    current_step_index: int = 0
    status: ExecutionStatus = ExecutionStatus.PENDING
    overall_confidence: float = 0.0
    total_execution_time_ms: float = 0.0

    # Planning metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    agent_model: str | None = None  # Which agent model created this
    complexity_score: float = 0.5  # Estimated complexity [0,1]
    expected_steps: int = 1  # Expected number of steps

    # Results
    final_result: Any | None = None
    explanation: str = ""
    evidence_nodes: list[str] = field(default_factory=list)

    # Adaptation
    replan_count: int = 0
    parent_plan_id: str | None = None  # If this is a replan
    adaptation_reason: str | None = None

    def add_step(self, step: ExecutionStep) -> None:
        """Add execution step to plan."""
        self.execution_steps.append(step)
        self.expected_steps = len(self.execution_steps)

    def create_checkpoint(self, completed_steps: set[str], intermediate_results: dict[str, Any]) -> PlanCheckpoint:
        """Create checkpoint at current execution state."""
        checkpoint = PlanCheckpoint(
            step_index=self.current_step_index,
            completed_steps=completed_steps.copy(),
            intermediate_results=intermediate_results.copy(),
            aggregate_confidence=self.overall_confidence,
            execution_time_ms=self.total_execution_time_ms,
        )
        self.checkpoints.append(checkpoint)
        return checkpoint

    def get_next_ready_step(self, completed_steps: set[str]) -> ExecutionStep | None:
        """Get next step that's ready to execute."""
        for step in self.execution_steps:
            if step.status == ExecutionStatus.PENDING and step.is_ready_to_execute(completed_steps):
                return step
        return None

    def get_step_by_id(self, step_id: str) -> ExecutionStep | None:
        """Find step by ID."""
        for step in self.execution_steps:
            if step.step_id == step_id:
                return step
        return None

    def is_complete(self) -> bool:
        """Check if all steps are completed."""
        return all(step.status == ExecutionStatus.COMPLETED for step in self.execution_steps)

    def has_failed_steps(self) -> bool:
        """Check if any steps have failed."""
        return any(step.status == ExecutionStatus.FAILED for step in self.execution_steps)

    def get_completed_steps(self) -> set[str]:
        """Get IDs of completed steps."""
        return {step.step_id for step in self.execution_steps if step.status == ExecutionStatus.COMPLETED}

    def calculate_overall_confidence(self) -> float:
        """Calculate aggregate confidence from completed steps."""
        completed = [step for step in self.execution_steps if step.status == ExecutionStatus.COMPLETED]

        if not completed:
            return 0.0

        # Weighted average by step importance (could be enhanced)
        confidences = [step.confidence_score for step in completed]
        return np.mean(confidences)

    def to_dict(self) -> dict[str, Any]:
        """Serialize plan to dictionary."""
        return {
            "plan_id": self.plan_id,
            "original_query": self.original_query,
            "query_type": self.query_type.value,
            "reasoning_strategy": self.reasoning_strategy.value,
            "execution_steps": [
                {
                    "step_id": step.step_id,
                    "step_type": step.step_type,
                    "description": step.description,
                    "operation": step.operation,
                    "parameters": step.parameters,
                    "dependencies": step.dependencies,
                    "status": step.status.value,
                    "confidence_score": step.confidence_score,
                }
                for step in self.execution_steps
            ],
            "status": self.status.value,
            "overall_confidence": self.overall_confidence,
            "complexity_score": self.complexity_score,
            "replan_count": self.replan_count,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryPlan":
        """Deserialize plan from dictionary."""
        plan = cls(
            plan_id=data["plan_id"],
            original_query=data["original_query"],
            query_type=QueryType(data["query_type"]),
            reasoning_strategy=ReasoningStrategy(data["reasoning_strategy"]),
            status=ExecutionStatus(data["status"]),
            overall_confidence=data["overall_confidence"],
            complexity_score=data["complexity_score"],
            replan_count=data["replan_count"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )

        # Reconstruct steps
        for step_data in data["execution_steps"]:
            step = ExecutionStep(
                step_id=step_data["step_id"],
                step_type=step_data["step_type"],
                description=step_data["description"],
                operation=step_data["operation"],
                parameters=step_data["parameters"],
                dependencies=step_data["dependencies"],
                status=ExecutionStatus(step_data["status"]),
                confidence_score=step_data["confidence_score"],
            )
            plan.execution_steps.append(step)

        return plan


@dataclass
class PlanDSL:
    """Domain-Specific Language for plan serialization."""

    @staticmethod
    def serialize_plan(plan: QueryPlan) -> str:
        """Serialize plan to DSL format."""
        lines = [
            f"PLAN {plan.plan_id}",
            f"QUERY: {plan.original_query}",
            f"TYPE: {plan.query_type.value}",
            f"STRATEGY: {plan.reasoning_strategy.value}",
            f"COMPLEXITY: {plan.complexity_score}",
            "",
        ]

        # Add constraints
        c = plan.retrieval_constraints
        lines.extend(
            [
                "CONSTRAINTS:",
                f"  max_depth: {c.max_depth}",
                f"  max_nodes: {c.max_nodes}",
                f"  confidence_threshold: {c.confidence_threshold}",
                f"  time_budget_ms: {c.time_budget_ms}",
                "",
            ]
        )

        # Add steps
        lines.append("STEPS:")
        for i, step in enumerate(plan.execution_steps):
            deps = ", ".join(step.dependencies) if step.dependencies else "none"
            lines.extend(
                [
                    f"  {i + 1}. {step.description}",
                    f"     operation: {step.operation}",
                    f"     depends_on: {deps}",
                    f"     confidence_threshold: {step.confidence_threshold}",
                    "",
                ]
            )

        return "\n".join(lines)

    @staticmethod
    def parse_plan(dsl_text: str) -> QueryPlan:
        """Parse plan from DSL format (simplified implementation)."""
        lines = dsl_text.strip().split("\n")

        # Extract basic info
        plan_id = lines[0].split()[1] if lines[0].startswith("PLAN") else str(uuid.uuid4())
        query = lines[1].split("QUERY: ")[1] if "QUERY:" in lines[1] else ""

        plan = QueryPlan(plan_id=plan_id, original_query=query)

        # This would be expanded for full DSL parsing
        return plan
