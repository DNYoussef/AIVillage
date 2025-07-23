"""
Execution Strategies

Concrete implementations of reasoning strategies for different query types.
Each strategy defines how to break down and execute specific types of queries.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .plan_structures import (
    QueryPlan, ExecutionStep, RetrievalConstraints, QueryType,
    ReasoningStrategy, ExecutionStatus
)

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Base class for all reasoning strategies"""

    description = "Base reasoning strategy"

    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    async def create_plan(self,
                         query: str,
                         query_type: QueryType,
                         constraints: RetrievalConstraints,
                         context: Dict[str, Any]) -> QueryPlan:
        """Create execution plan for the query"""
        pass

    def _create_base_plan(self,
                         query: str,
                         query_type: QueryType,
                         strategy: ReasoningStrategy,
                         constraints: RetrievalConstraints) -> QueryPlan:
        """Create base plan structure"""
        return QueryPlan(
            original_query=query,
            query_type=query_type,
            reasoning_strategy=strategy,
            retrieval_constraints=constraints
        )

    def _add_retrieval_step(self,
                           plan: QueryPlan,
                           description: str,
                           operation: str,
                           parameters: Dict[str, Any],
                           dependencies: List[str] = None) -> str:
        """Add a retrieval step to the plan"""
        step = ExecutionStep(
            step_type="retrieval",
            description=description,
            operation=operation,
            parameters=parameters,
            dependencies=dependencies or [],
            confidence_threshold=plan.retrieval_constraints.confidence_threshold,
            timeout_ms=min(plan.retrieval_constraints.time_budget_ms // 2, 2000)
        )
        plan.add_step(step)
        return step.step_id

    def _add_reasoning_step(self,
                           plan: QueryPlan,
                           description: str,
                           operation: str,
                           parameters: Dict[str, Any],
                           dependencies: List[str] = None) -> str:
        """Add a reasoning step to the plan"""
        step = ExecutionStep(
            step_type="reasoning",
            description=description,
            operation=operation,
            parameters=parameters,
            dependencies=dependencies or [],
            confidence_threshold=0.6,  # Lower threshold for reasoning steps
            timeout_ms=min(plan.retrieval_constraints.time_budget_ms // 3, 3000)
        )
        plan.add_step(step)
        return step.step_id

    def _add_verification_step(self,
                              plan: QueryPlan,
                              description: str,
                              operation: str,
                              parameters: Dict[str, Any],
                              dependencies: List[str] = None) -> str:
        """Add a verification step to the plan"""
        step = ExecutionStep(
            step_type="verification",
            description=description,
            operation=operation,
            parameters=parameters,
            dependencies=dependencies or [],
            confidence_threshold=0.8,  # Higher threshold for verification
            timeout_ms=1000
        )
        plan.add_step(step)
        return step.step_id


class SimpleFactStrategy(BaseStrategy):
    """Strategy for simple factual queries requiring direct retrieval"""

    description = "Direct retrieval for simple factual queries"

    async def create_plan(self,
                         query: str,
                         query_type: QueryType,
                         constraints: RetrievalConstraints,
                         context: Dict[str, Any]) -> QueryPlan:

        plan = self._create_base_plan(query, query_type, ReasoningStrategy.DIRECT_RETRIEVAL, constraints)
        plan.complexity_score = 0.2

        # Single retrieval step for simple facts
        self._add_retrieval_step(
            plan,
            description="Retrieve factual information",
            operation="semantic_search",
            parameters={
                "query": query,
                "limit": min(constraints.max_nodes, 20),
                "confidence_threshold": constraints.confidence_threshold,
                "include_explanations": constraints.include_explanations
            }
        )

        return plan


class TemporalStrategy(BaseStrategy):
    """Strategy for temporal reasoning queries"""

    description = "Temporal reasoning with chronological analysis"

    async def create_plan(self,
                         query: str,
                         query_type: QueryType,
                         constraints: RetrievalConstraints,
                         context: Dict[str, Any]) -> QueryPlan:

        plan = self._create_base_plan(query, query_type, ReasoningStrategy.TEMPORAL_REASONING, constraints)
        plan.complexity_score = 0.6

        # Step 1: Extract temporal entities and events
        temporal_step = self._add_retrieval_step(
            plan,
            description="Extract temporal entities and events",
            operation="temporal_extraction",
            parameters={
                "query": query,
                "extract_dates": True,
                "extract_events": True,
                "temporal_relations": True
            }
        )

        # Step 2: Build temporal graph
        graph_step = self._add_reasoning_step(
            plan,
            description="Build temporal relationship graph",
            operation="temporal_graph_construction",
            parameters={
                "temporal_entities": "from_previous_step",
                "ordering_rules": ["chronological", "causal"]
            },
            dependencies=[temporal_step]
        )

        # Step 3: Temporal reasoning
        self._add_reasoning_step(
            plan,
            description="Perform temporal reasoning",
            operation="temporal_reasoning",
            parameters={
                "temporal_graph": "from_previous_step",
                "query": query,
                "reasoning_type": "chronological_analysis"
            },
            dependencies=[graph_step]
        )

        return plan


class CausalStrategy(BaseStrategy):
    """Strategy for causal reasoning queries"""

    description = "Causal reasoning and cause-effect analysis"

    async def create_plan(self,
                         query: str,
                         query_type: QueryType,
                         constraints: RetrievalConstraints,
                         context: Dict[str, Any]) -> QueryPlan:

        plan = self._create_base_plan(query, query_type, ReasoningStrategy.CAUSAL_REASONING, constraints)
        plan.complexity_score = 0.7

        # Step 1: Identify causal entities
        entity_step = self._add_retrieval_step(
            plan,
            description="Identify potential causes and effects",
            operation="causal_entity_extraction",
            parameters={
                "query": query,
                "extract_causes": True,
                "extract_effects": True,
                "causal_indicators": ["because", "due to", "leads to", "causes"]
            }
        )

        # Step 2: Retrieve causal relationships
        causal_step = self._add_retrieval_step(
            plan,
            description="Retrieve causal relationships",
            operation="causal_graph_retrieval",
            parameters={
                "entities": "from_previous_step",
                "relation_types": ["causes", "prevents", "enables", "triggers"],
                "max_depth": 3
            },
            dependencies=[entity_step]
        )

        # Step 3: Causal analysis
        analysis_step = self._add_reasoning_step(
            plan,
            description="Analyze causal chains",
            operation="causal_analysis",
            parameters={
                "causal_graph": "from_previous_step",
                "query": query,
                "analysis_type": "chain_reasoning"
            },
            dependencies=[causal_step]
        )

        # Step 4: Verify causal relationships
        self._add_verification_step(
            plan,
            description="Verify causal relationships",
            operation="causal_verification",
            parameters={
                "causal_chains": "from_previous_step",
                "evidence_threshold": 0.7
            },
            dependencies=[analysis_step]
        )

        return plan


class ComparativeStrategy(BaseStrategy):
    """Strategy for comparative analysis queries"""

    description = "Comparative analysis and contrast reasoning"

    async def create_plan(self,
                         query: str,
                         query_type: QueryType,
                         constraints: RetrievalConstraints,
                         context: Dict[str, Any]) -> QueryPlan:

        plan = self._create_base_plan(query, query_type, ReasoningStrategy.COMPARATIVE_ANALYSIS, constraints)
        plan.complexity_score = 0.6

        # Step 1: Extract comparison entities
        entity_step = self._add_retrieval_step(
            plan,
            description="Extract entities for comparison",
            operation="comparative_entity_extraction",
            parameters={
                "query": query,
                "comparison_indicators": ["vs", "versus", "compared to", "between"],
                "extract_attributes": True
            }
        )

        # Step 2: Retrieve comparison data
        data_step = self._add_retrieval_step(
            plan,
            description="Retrieve comparison data",
            operation="comparative_data_retrieval",
            parameters={
                "entities": "from_previous_step",
                "comparison_dimensions": "auto_detect",
                "include_attributes": True,
                "include_relationships": True
            },
            dependencies=[entity_step]
        )

        # Step 3: Comparative analysis
        self._add_reasoning_step(
            plan,
            description="Perform comparative analysis",
            operation="comparative_reasoning",
            parameters={
                "comparison_data": "from_previous_step",
                "query": query,
                "analysis_type": "similarity_contrast"
            },
            dependencies=[data_step]
        )

        return plan


class MetaQueryStrategy(BaseStrategy):
    """Strategy for meta-knowledge queries"""

    description = "Meta-reasoning about knowledge and information"

    async def create_plan(self,
                         query: str,
                         query_type: QueryType,
                         constraints: RetrievalConstraints,
                         context: Dict[str, Any]) -> QueryPlan:

        plan = self._create_base_plan(query, query_type, ReasoningStrategy.META_REASONING, constraints)
        plan.complexity_score = 0.4

        # Step 1: Knowledge source analysis
        source_step = self._add_retrieval_step(
            plan,
            description="Analyze knowledge sources",
            operation="knowledge_source_analysis",
            parameters={
                "query": query,
                "analyze_confidence": True,
                "analyze_coverage": True,
                "analyze_recency": True
            }
        )

        # Step 2: Meta-information retrieval
        meta_step = self._add_retrieval_step(
            plan,
            description="Retrieve meta-information",
            operation="meta_information_retrieval",
            parameters={
                "query": query,
                "include_provenance": True,
                "include_confidence_scores": True,
                "include_source_quality": True
            },
            dependencies=[source_step]
        )

        # Step 3: Meta-reasoning
        self._add_reasoning_step(
            plan,
            description="Perform meta-reasoning",
            operation="meta_reasoning",
            parameters={
                "meta_information": "from_previous_step",
                "query": query,
                "reasoning_type": "knowledge_assessment"
            },
            dependencies=[meta_step]
        )

        return plan


class MultiHopStrategy(BaseStrategy):
    """Strategy for multi-hop reasoning queries"""

    description = "Step-by-step multi-hop reasoning"

    async def create_plan(self,
                         query: str,
                         query_type: QueryType,
                         constraints: RetrievalConstraints,
                         context: Dict[str, Any]) -> QueryPlan:

        plan = self._create_base_plan(query, query_type, ReasoningStrategy.STEP_BY_STEP, constraints)
        plan.complexity_score = 0.8

        # Step 1: Query decomposition
        decomp_step = self._add_reasoning_step(
            plan,
            description="Decompose complex query into sub-questions",
            operation="query_decomposition",
            parameters={
                "query": query,
                "max_subqueries": 5,
                "dependency_analysis": True
            }
        )

        # Step 2: Initial retrieval
        initial_step = self._add_retrieval_step(
            plan,
            description="Initial knowledge retrieval",
            operation="initial_retrieval",
            parameters={
                "subqueries": "from_previous_step",
                "max_depth": 1,
                "confidence_threshold": constraints.confidence_threshold
            },
            dependencies=[decomp_step]
        )

        # Step 3: Iterative reasoning
        reasoning_step = self._add_reasoning_step(
            plan,
            description="Iterative multi-hop reasoning",
            operation="iterative_reasoning",
            parameters={
                "initial_results": "from_previous_step",
                "subqueries": "from_decomposition_step",
                "max_hops": constraints.max_depth,
                "convergence_threshold": 0.1
            },
            dependencies=[initial_step]
        )

        # Step 4: Result synthesis
        self._add_reasoning_step(
            plan,
            description="Synthesize multi-hop results",
            operation="result_synthesis",
            parameters={
                "reasoning_results": "from_previous_step",
                "original_query": query,
                "synthesis_method": "evidence_aggregation"
            },
            dependencies=[reasoning_step]
        )

        return plan


class AggregationStrategy(BaseStrategy):
    """Strategy for aggregation and statistical queries"""

    description = "Graph traversal for aggregation queries"

    async def create_plan(self,
                         query: str,
                         query_type: QueryType,
                         constraints: RetrievalConstraints,
                         context: Dict[str, Any]) -> QueryPlan:

        plan = self._create_base_plan(query, query_type, ReasoningStrategy.GRAPH_TRAVERSAL, constraints)
        plan.complexity_score = 0.5

        # Step 1: Identify aggregation target
        target_step = self._add_retrieval_step(
            plan,
            description="Identify aggregation targets",
            operation="aggregation_target_identification",
            parameters={
                "query": query,
                "aggregation_types": ["count", "sum", "average", "list"],
                "target_entities": "auto_detect"
            }
        )

        # Step 2: Graph traversal
        traversal_step = self._add_retrieval_step(
            plan,
            description="Traverse graph for aggregation data",
            operation="graph_traversal",
            parameters={
                "start_entities": "from_previous_step",
                "traversal_rules": "aggregation_focused",
                "max_depth": constraints.max_depth,
                "collect_all_paths": True
            },
            dependencies=[target_step]
        )

        # Step 3: Aggregation computation
        self._add_reasoning_step(
            plan,
            description="Compute aggregation results",
            operation="aggregation_computation",
            parameters={
                "traversal_results": "from_previous_step",
                "query": query,
                "aggregation_method": "auto_detect"
            },
            dependencies=[traversal_step]
        )

        return plan


class HypotheticalStrategy(BaseStrategy):
    """Strategy for hypothetical and what-if queries"""

    description = "Hypothetical reasoning and scenario analysis"

    async def create_plan(self,
                         query: str,
                         query_type: QueryType,
                         constraints: RetrievalConstraints,
                         context: Dict[str, Any]) -> QueryPlan:

        plan = self._create_base_plan(query, query_type, ReasoningStrategy.STEP_BY_STEP, constraints)
        plan.complexity_score = 0.7

        # Step 1: Extract scenario conditions
        scenario_step = self._add_reasoning_step(
            plan,
            description="Extract hypothetical scenario conditions",
            operation="scenario_extraction",
            parameters={
                "query": query,
                "hypothetical_indicators": ["what if", "suppose", "imagine"],
                "condition_analysis": True
            }
        )

        # Step 2: Baseline knowledge retrieval
        baseline_step = self._add_retrieval_step(
            plan,
            description="Retrieve baseline knowledge",
            operation="baseline_retrieval",
            parameters={
                "scenario_entities": "from_previous_step",
                "include_current_state": True,
                "include_historical_data": True
            },
            dependencies=[scenario_step]
        )

        # Step 3: Scenario simulation
        simulation_step = self._add_reasoning_step(
            plan,
            description="Simulate hypothetical scenario",
            operation="scenario_simulation",
            parameters={
                "baseline_knowledge": "from_previous_step",
                "scenario_conditions": "from_scenario_step",
                "simulation_method": "rule_based_projection"
            },
            dependencies=[baseline_step]
        )

        # Step 4: Impact analysis
        self._add_reasoning_step(
            plan,
            description="Analyze scenario impacts",
            operation="impact_analysis",
            parameters={
                "simulation_results": "from_previous_step",
                "query": query,
                "analysis_scope": "comprehensive"
            },
            dependencies=[simulation_step]
        )

        return plan


class HybridStrategy(BaseStrategy):
    """Hybrid strategy combining multiple approaches"""

    description = "Hybrid approach combining multiple reasoning strategies"

    async def create_plan(self,
                         query: str,
                         query_type: QueryType,
                         constraints: RetrievalConstraints,
                         context: Dict[str, Any]) -> QueryPlan:

        plan = self._create_base_plan(query, query_type, ReasoningStrategy.HYBRID, constraints)
        plan.complexity_score = 0.9

        # Step 1: Multi-strategy analysis
        analysis_step = self._add_reasoning_step(
            plan,
            description="Analyze query for multiple reasoning approaches",
            operation="multi_strategy_analysis",
            parameters={
                "query": query,
                "detect_temporal": True,
                "detect_causal": True,
                "detect_comparative": True,
                "strategy_weights": True
            }
        )

        # Step 2: Parallel strategy execution
        parallel_step = self._add_reasoning_step(
            plan,
            description="Execute multiple strategies in parallel",
            operation="parallel_strategy_execution",
            parameters={
                "strategy_analysis": "from_previous_step",
                "max_parallel_strategies": 3,
                "timeout_per_strategy": constraints.time_budget_ms // 3
            },
            dependencies=[analysis_step]
        )

        # Step 3: Result integration
        integration_step = self._add_reasoning_step(
            plan,
            description="Integrate results from multiple strategies",
            operation="result_integration",
            parameters={
                "parallel_results": "from_previous_step",
                "integration_method": "weighted_consensus",
                "confidence_weighting": True
            },
            dependencies=[parallel_step]
        )

        # Step 4: Consistency verification
        self._add_verification_step(
            plan,
            description="Verify result consistency",
            operation="consistency_verification",
            parameters={
                "integrated_results": "from_previous_step",
                "consistency_threshold": 0.8,
                "conflict_resolution": "evidence_based"
            },
            dependencies=[integration_step]
        )

        return plan
