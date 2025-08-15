"""Query Planner.

Strategic planning system for complex queries inspired by PlanRAG research.
Provides adaptive planning, strategy selection, and intelligent re-planning capabilities.
"""

import logging
import time
from typing import Any

from ..guardian.gate import GuardianGate
from .plan_structures import (
    ExecutionStatus,
    QueryPlan,
    QueryType,
    ReasoningStrategy,
    RetrievalConstraints,
)
from .query_classifier import QueryClassifier
from .strategy_selector import StrategySelector

logger = logging.getLogger(__name__)


class AgentReasoningModel:
    """Model representing agent reasoning capabilities."""

    def __init__(
        self,
        model_name: str,
        capabilities: list[str],
        performance_profile: dict[str, float],
    ) -> None:
        self.model_name = model_name
        self.capabilities = capabilities
        self.performance_profile = performance_profile

        # Default performance metrics
        self.max_complexity = performance_profile.get("max_complexity", 0.8)
        self.reasoning_speed = performance_profile.get("reasoning_speed", 1.0)
        self.accuracy = performance_profile.get("accuracy", 0.85)
        self.memory_limit_mb = performance_profile.get("memory_limit_mb", 1000)


class QueryPlanner:
    """Strategic planning system for complex queries.

    Analyzes query structure and intent, determines retrieval strategy,
    plans reasoning approach, and enables adaptive re-planning.
    """

    def __init__(
        self,
        classifier: QueryClassifier | None = None,
        strategy_selector: StrategySelector | None = None,
        guardian_gate: GuardianGate | None = None,
    ) -> None:
        self.classifier = classifier or QueryClassifier()
        self.strategy_selector = strategy_selector or StrategySelector()
        self.guardian_gate = guardian_gate or GuardianGate()

        # Planning configuration
        self.default_constraints = RetrievalConstraints()
        self.max_replan_attempts = 3
        self.confidence_threshold_for_replan = 0.5

        # Guardian integration settings
        self.guardian_confidence_threshold = 0.7
        self.guardian_high_risk_domains = {"medical", "financial", "legal"}

        # Planning history for learning
        self.planning_history = []
        self.performance_metrics = {
            "plans_created": 0,
            "successful_plans": 0,
            "replans_triggered": 0,
            "avg_plan_confidence": 0.0,
            "guardian_blocks": 0,
            "guardian_approvals": 0,
        }

    async def create_plan(
        self,
        query: str,
        agent_model: AgentReasoningModel,
        constraints: RetrievalConstraints | None = None,
        context: dict[str, Any] | None = None,
    ) -> QueryPlan:
        """Create a strategic execution plan for a query.

        Args:
            query: Input query string
            agent_model: Agent reasoning model and capabilities
            constraints: Execution constraints (optional)
            context: Additional context for planning (optional)

        Returns:
            Complete query execution plan
        """
        start_time = time.time()
        constraints = constraints or self.default_constraints
        context = context or {}

        logger.info(f"Creating plan for query: '{query[:100]}...'")

        try:
            # Step 1: Classify query complexity and intent
            (
                query_type,
                classification_confidence,
                analysis,
            ) = await self._classify_query(query)

            # Step 2: Determine retrieval strategy
            strategy = await self._select_strategy(
                query_type,
                analysis["complexity_score"],
                constraints,
                agent_model,
                context,
            )

            # Step 3: Adjust constraints based on agent capabilities
            adjusted_constraints = self._adjust_constraints_for_agent(
                constraints, agent_model, analysis["complexity_score"]
            )

            # Step 4: Create strategy-specific plan
            plan = await self._create_strategy_plan(
                query, query_type, strategy, adjusted_constraints, analysis
            )

            # Step 5: Set plan metadata
            plan.agent_model = agent_model.model_name
            plan.complexity_score = analysis["complexity_score"]
            plan.overall_confidence = classification_confidence

            # Step 6: Validate and optimize plan
            plan = await self._validate_and_optimize_plan(plan, agent_model)

            planning_time = (time.time() - start_time) * 1000

            # Record planning metrics
            self._record_planning_metrics(plan, planning_time, True)

            logger.info(
                f"Created {strategy.value} plan with {len(plan.execution_steps)} steps "
                f"(complexity: {plan.complexity_score:.2f}, time: {planning_time:.1f}ms)"
            )

            return plan

        except Exception as e:
            planning_time = (time.time() - start_time) * 1000
            logger.exception(f"Failed to create plan for query '{query[:50]}...': {e}")

            # Create fallback simple plan
            fallback_plan = await self._create_fallback_plan(query, constraints)
            self._record_planning_metrics(fallback_plan, planning_time, False)

            return fallback_plan

    async def replan(
        self,
        original_plan: QueryPlan,
        intermediate_results: dict[str, Any],
        current_confidence: float,
        failure_reason: str | None = None,
    ) -> QueryPlan:
        """Adaptive replanning when confidence is low or execution fails.

        Args:
            original_plan: Original execution plan
            intermediate_results: Results from partially executed plan
            current_confidence: Current aggregate confidence
            failure_reason: Reason for replanning (optional)

        Returns:
            New execution plan
        """
        logger.info(
            f"Replanning query '{original_plan.original_query[:50]}...' "
            f"(confidence: {current_confidence:.3f}, reason: {failure_reason})"
        )

        # Check replan limits
        if original_plan.replan_count >= self.max_replan_attempts:
            logger.warning(
                f"Maximum replan attempts ({self.max_replan_attempts}) reached"
            )
            return original_plan

        try:
            # Analyze what went wrong
            replan_analysis = await self._analyze_replan_needs(
                original_plan, intermediate_results, current_confidence, failure_reason
            )

            # Determine new strategy
            new_strategy = await self._select_replan_strategy(
                original_plan, replan_analysis
            )

            # Create new plan
            new_plan = await self._create_replan(
                original_plan, new_strategy, replan_analysis
            )

            # Set replan metadata
            new_plan.replan_count = original_plan.replan_count + 1
            new_plan.parent_plan_id = original_plan.plan_id
            new_plan.adaptation_reason = (
                failure_reason or f"low_confidence_{current_confidence:.2f}"
            )

            # Transfer useful intermediate results
            await self._transfer_intermediate_results(
                original_plan, new_plan, intermediate_results
            )

            self.performance_metrics["replans_triggered"] += 1

            logger.info(
                f"Created replan with {new_strategy.value} strategy "
                f"({len(new_plan.execution_steps)} steps)"
            )

            return new_plan

        except Exception as e:
            logger.exception(f"Failed to create replan: {e}")
            # Return original plan as fallback
            return original_plan

    async def _classify_query(
        self, query: str
    ) -> tuple[QueryType, float, dict[str, Any]]:
        """Classify query type and complexity."""
        return self.classifier.classify_query(query)

    async def _select_strategy(
        self,
        query_type: QueryType,
        complexity_score: float,
        constraints: RetrievalConstraints,
        agent_model: AgentReasoningModel,
        context: dict[str, Any],
    ) -> ReasoningStrategy:
        """Select appropriate reasoning strategy."""
        # Check agent capabilities
        if complexity_score > agent_model.max_complexity:
            logger.warning(
                f"Query complexity ({complexity_score:.2f}) exceeds agent limit "
                f"({agent_model.max_complexity:.2f})"
            )
            # Use simpler strategy
            if query_type in [QueryType.CAUSAL_CHAIN, QueryType.MULTI_HOP]:
                query_type = QueryType.TEMPORAL_ANALYSIS
            elif query_type == QueryType.HYPOTHETICAL:
                query_type = QueryType.COMPARATIVE

        # Add agent preferences to context
        context["agent_capabilities"] = agent_model.capabilities
        context["prefer_fast"] = agent_model.reasoning_speed < 0.5
        context["prefer_accurate"] = agent_model.accuracy > 0.9

        return self.strategy_selector.select_strategy(
            query_type, complexity_score, constraints, context
        )

    def _adjust_constraints_for_agent(
        self,
        constraints: RetrievalConstraints,
        agent_model: AgentReasoningModel,
        complexity_score: float,
    ) -> RetrievalConstraints:
        """Adjust constraints based on agent capabilities."""
        adjusted = RetrievalConstraints(
            max_depth=min(constraints.max_depth, int(agent_model.max_complexity * 10)),
            max_nodes=min(constraints.max_nodes, int(agent_model.memory_limit_mb / 10)),
            confidence_threshold=max(
                constraints.confidence_threshold, 1.0 - agent_model.accuracy
            ),
            time_budget_ms=int(
                constraints.time_budget_ms * agent_model.reasoning_speed
            ),
            include_explanations=constraints.include_explanations,
            prefer_recent=constraints.prefer_recent,
            domain_filter=constraints.domain_filter,
            exclude_uncertainty=constraints.exclude_uncertainty,
        )

        # Adjust for complexity
        if complexity_score > 0.8:
            adjusted.time_budget_ms = int(adjusted.time_budget_ms * 1.5)
            adjusted.confidence_threshold = max(
                adjusted.confidence_threshold - 0.1, 0.5
            )

        return adjusted

    async def _create_strategy_plan(
        self,
        query: str,
        query_type: QueryType,
        strategy: ReasoningStrategy,
        constraints: RetrievalConstraints,
        analysis: dict[str, Any],
    ) -> QueryPlan:
        """Create plan using selected strategy."""
        # Get strategy instance
        strategy_instance = self.strategy_selector.create_strategy_instance(strategy)

        # Create plan
        plan = await strategy_instance.create_plan(
            query, query_type, constraints, analysis
        )

        # Add reasoning hints
        if hasattr(self.classifier, "get_reasoning_hints"):
            hints = self.classifier.get_reasoning_hints(query, query_type)
            for step in plan.execution_steps:
                step.parameters.setdefault("reasoning_hints", hints)

        return plan

    async def _validate_and_optimize_plan(
        self, plan: QueryPlan, agent_model: AgentReasoningModel
    ) -> QueryPlan:
        """Validate plan feasibility and optimize."""
        # Check time budget
        total_estimated_time = sum(step.timeout_ms for step in plan.execution_steps)
        if total_estimated_time > plan.retrieval_constraints.time_budget_ms:
            logger.warning(
                f"Plan estimated time ({total_estimated_time}ms) exceeds budget "
                f"({plan.retrieval_constraints.time_budget_ms}ms)"
            )

            # Reduce timeouts proportionally
            scale_factor = (
                plan.retrieval_constraints.time_budget_ms / total_estimated_time
            )
            for step in plan.execution_steps:
                step.timeout_ms = int(step.timeout_ms * scale_factor)

        # Check memory requirements
        strategy_requirements = self.strategy_selector.get_strategy_requirements(
            plan.reasoning_strategy
        )
        required_memory = strategy_requirements.get("min_memory_mb", 100)

        if required_memory > agent_model.memory_limit_mb:
            logger.warning(
                f"Plan requires {required_memory}MB but agent limit is "
                f"{agent_model.memory_limit_mb}MB"
            )

            # Reduce node limits
            scale_factor = agent_model.memory_limit_mb / required_memory
            plan.retrieval_constraints.max_nodes = int(
                plan.retrieval_constraints.max_nodes * scale_factor
            )

        # Optimize step order for better parallelization
        plan = self._optimize_step_dependencies(plan)

        return plan

    def _optimize_step_dependencies(self, plan: QueryPlan) -> QueryPlan:
        """Optimize step dependencies for better execution."""
        # Simple optimization: ensure retrieval steps can run in parallel
        # where possible
        retrieval_steps = [
            s for s in plan.execution_steps if s.step_type == "retrieval"
        ]

        for step in retrieval_steps:
            # Remove unnecessary dependencies between retrieval steps
            step.dependencies = [
                dep
                for dep in step.dependencies
                if any(
                    s.step_id == dep and s.step_type != "retrieval"
                    for s in plan.execution_steps
                )
            ]

        return plan

    async def _create_fallback_plan(
        self, query: str, constraints: RetrievalConstraints
    ) -> QueryPlan:
        """Create simple fallback plan when planning fails."""
        from .strategies import SimpleFactStrategy

        strategy = SimpleFactStrategy()
        plan = await strategy.create_plan(query, QueryType.SIMPLE_FACT, constraints, {})

        plan.complexity_score = 0.1
        plan.overall_confidence = 0.6

        return plan

    async def _analyze_replan_needs(
        self,
        original_plan: QueryPlan,
        intermediate_results: dict[str, Any],
        current_confidence: float,
        failure_reason: str | None,
    ) -> dict[str, Any]:
        """Analyze what went wrong and what needs to change."""
        analysis = {
            "original_strategy": original_plan.reasoning_strategy,
            "failed_steps": [],
            "successful_steps": [],
            "confidence_drop": original_plan.overall_confidence - current_confidence,
            "failure_reason": failure_reason,
            "intermediate_results": intermediate_results,
        }

        # Analyze step performance
        for step in original_plan.execution_steps:
            if step.status == ExecutionStatus.FAILED:
                analysis["failed_steps"].append(
                    {
                        "step_id": step.step_id,
                        "step_type": step.step_type,
                        "error": step.error_message,
                    }
                )
            elif step.status == ExecutionStatus.COMPLETED:
                analysis["successful_steps"].append(
                    {"step_id": step.step_id, "confidence": step.confidence_score}
                )

        # Determine replan strategy
        if len(analysis["failed_steps"]) > len(analysis["successful_steps"]):
            analysis["replan_type"] = "complete_restart"
        elif current_confidence < 0.3:
            analysis["replan_type"] = "strategy_change"
        else:
            analysis["replan_type"] = "incremental_improvement"

        return analysis

    async def _select_replan_strategy(
        self, original_plan: QueryPlan, analysis: dict[str, Any]
    ) -> ReasoningStrategy:
        """Select new strategy for replanning."""
        original_strategy = original_plan.reasoning_strategy

        # Try fallback strategy first
        fallback = self.strategy_selector.recommend_fallback_strategy(
            original_strategy, original_plan.query_type
        )

        if fallback and fallback != original_strategy:
            return fallback

        # If no fallback, use simpler strategy
        if original_strategy == ReasoningStrategy.HYBRID:
            return ReasoningStrategy.STEP_BY_STEP
        if original_strategy == ReasoningStrategy.STEP_BY_STEP:
            return ReasoningStrategy.GRAPH_TRAVERSAL
        if original_strategy in [
            ReasoningStrategy.CAUSAL_REASONING,
            ReasoningStrategy.TEMPORAL_REASONING,
        ]:
            return ReasoningStrategy.COMPARATIVE_ANALYSIS
        return ReasoningStrategy.DIRECT_RETRIEVAL

    async def _create_replan(
        self,
        original_plan: QueryPlan,
        new_strategy: ReasoningStrategy,
        analysis: dict[str, Any],
    ) -> QueryPlan:
        """Create new plan with different strategy."""
        # Create new plan with updated strategy
        new_plan = await self._create_strategy_plan(
            original_plan.original_query,
            original_plan.query_type,
            new_strategy,
            original_plan.retrieval_constraints,
            {"complexity_score": max(original_plan.complexity_score - 0.2, 0.1)},
        )

        # Inherit successful parts if possible
        if analysis["replan_type"] == "incremental_improvement":
            # Try to preserve successful steps
            [s["step_id"] for s in analysis["successful_steps"]]
            # This would require more sophisticated merging logic

        return new_plan

    async def _transfer_intermediate_results(
        self,
        original_plan: QueryPlan,
        new_plan: QueryPlan,
        intermediate_results: dict[str, Any],
    ) -> None:
        """Transfer useful intermediate results to new plan."""
        # Simple transfer of any successful retrieval results
        if intermediate_results:
            # Add intermediate results as context for first step
            if new_plan.execution_steps:
                first_step = new_plan.execution_steps[0]
                first_step.parameters.setdefault("prior_results", intermediate_results)

    def _record_planning_metrics(
        self, plan: QueryPlan, planning_time_ms: float, success: bool
    ) -> None:
        """Record planning performance metrics."""
        self.performance_metrics["plans_created"] += 1
        if success:
            self.performance_metrics["successful_plans"] += 1

        # Update average confidence
        alpha = 0.1
        current_avg = self.performance_metrics["avg_plan_confidence"]
        self.performance_metrics["avg_plan_confidence"] = (
            1 - alpha
        ) * current_avg + alpha * plan.overall_confidence

        # Record in history
        self.planning_history.append(
            {
                "query": plan.original_query[:100],
                "query_type": plan.query_type.value,
                "strategy": plan.reasoning_strategy.value,
                "complexity": plan.complexity_score,
                "confidence": plan.overall_confidence,
                "planning_time_ms": planning_time_ms,
                "success": success,
                "step_count": len(plan.execution_steps),
            }
        )

        # Keep only recent history
        if len(self.planning_history) > 1000:
            self.planning_history = self.planning_history[-1000:]

    def get_planning_stats(self) -> dict[str, Any]:
        """Get planning performance statistics."""
        total_plans = self.performance_metrics["plans_created"]
        if total_plans == 0:
            return {"message": "No plans created yet"}

        success_rate = self.performance_metrics["successful_plans"] / total_plans
        replan_rate = self.performance_metrics["replans_triggered"] / total_plans

        recent_history = self.planning_history[-100:] if self.planning_history else []

        stats = {
            "total_plans": total_plans,
            "success_rate": success_rate,
            "replan_rate": replan_rate,
            "avg_confidence": self.performance_metrics["avg_plan_confidence"],
            "recent_avg_complexity": (
                sum(h["complexity"] for h in recent_history) / len(recent_history)
                if recent_history
                else 0
            ),
            "recent_avg_steps": (
                sum(h["step_count"] for h in recent_history) / len(recent_history)
                if recent_history
                else 0
            ),
            "strategy_distribution": {},
        }

        # Strategy usage distribution
        if recent_history:
            strategy_counts = {}
            for h in recent_history:
                strategy = h["strategy"]
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

            total_recent = len(recent_history)
            stats["strategy_distribution"] = {
                strategy: count / total_recent
                for strategy, count in strategy_counts.items()
            }

        return stats

    async def validate_final_answer(
        self, answer: str, confidence: float, context: dict[str, Any]
    ) -> tuple[bool, str]:
        """Validate final answer through Guardian Gate if confidence is low or domain is high-risk.

        Args:
            answer: The final answer to validate
            confidence: Confidence score from reasoning
            context: Query context including domain information

        Returns:
            Tuple of (should_proceed, rationale)
        """
        try:
            # Check if Guardian validation is needed
            domain = context.get("domain", "general")
            needs_validation = (
                confidence < self.guardian_confidence_threshold
                or domain in self.guardian_high_risk_domains
            )

            if not needs_validation:
                return (
                    True,
                    "Answer approved: sufficient confidence and low-risk domain",
                )

            logger.info(
                f"Triggering Guardian validation for domain '{domain}' with confidence {confidence:.3f}"
            )

            # Create a mock creative bridge for validation
            # In a real implementation, this would be more sophisticated
            from ..guardian.gate import CreativeBridge

            bridge = CreativeBridge(
                id=f"query_answer_{hash(answer) % 10000}",
                confidence=confidence,
                bridge_type="query_answer",
            )

            # Add answer context to bridge metadata if available
            if hasattr(bridge, "metadata"):
                bridge.metadata = {
                    "answer": answer[:200],  # Truncate for safety
                    "domain": domain,
                    "query_context": context.get("query_type", "unknown"),
                }

            # Validate through Guardian
            decision = await self.guardian_gate.evaluate_creative(bridge)

            # Update metrics
            if decision == "APPLY":
                self.performance_metrics["guardian_approvals"] += 1
                return True, f"Guardian approved answer (decision: {decision})"
            self.performance_metrics["guardian_blocks"] += 1
            return False, f"Guardian blocked answer (decision: {decision})"

        except Exception as e:
            logger.exception(f"Guardian validation failed: {e}")
            # Default to allowing on error to avoid blocking legitimate queries
            return True, f"Guardian validation error, defaulting to allow: {e!s}"
