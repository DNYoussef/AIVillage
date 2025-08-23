"""
Reasoning Interface

Defines the contract for reasoning systems that process queries,
integrate context, and execute inference chains.
Built upon the established KnowledgeRetrievalInterface patterns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ReasoningMode(Enum):
    """Reasoning processing modes"""

    DEDUCTIVE = "deductive"  # Logical deduction from premises
    INDUCTIVE = "inductive"  # Pattern recognition and generalization
    ABDUCTIVE = "abductive"  # Best explanation inference
    ANALOGICAL = "analogical"  # Reasoning by analogy
    CAUSAL = "causal"  # Cause and effect reasoning


class InferenceChainType(Enum):
    """Types of inference chains"""

    LINEAR = "linear"  # Sequential step-by-step reasoning
    BRANCHING = "branching"  # Multiple parallel reasoning paths
    HIERARCHICAL = "hierarchical"  # Nested reasoning levels
    ITERATIVE = "iterative"  # Refinement through iterations


@dataclass
class ReasoningContext:
    """Context information for reasoning operations"""

    user_id: str | None = None
    session_id: str | None = None
    domain: str | None = None
    confidence_threshold: float = 0.7
    max_inference_depth: int = 5
    reasoning_constraints: dict[str, Any] = None


@dataclass
class InferenceStep:
    """Individual step in reasoning chain"""

    step_id: str
    premise: str
    conclusion: str
    confidence: float
    evidence: list[str]
    rule_applied: str
    metadata: dict[str, Any]


@dataclass
class ReasoningResult:
    """Result of reasoning operation"""

    conclusion: str
    confidence_score: float
    reasoning_mode: ReasoningMode
    inference_chain: list[InferenceStep]
    alternative_conclusions: list[str]
    uncertainty_factors: list[str]
    metadata: dict[str, Any]


class ReasoningInterface(ABC):
    """
    Abstract interface for reasoning systems

    Defines the contract for systems that perform logical reasoning,
    inference, and conclusion generation from premises and context.
    Follows the established patterns from KnowledgeRetrievalInterface.
    """

    @abstractmethod
    async def reason(
        self,
        query: str,
        premises: list[str],
        mode: ReasoningMode = ReasoningMode.DEDUCTIVE,
        context: ReasoningContext | None = None,
    ) -> ReasoningResult:
        """
        Execute reasoning process to reach conclusions

        Args:
            query: Question or problem to reason about
            premises: Known facts and assumptions
            mode: Type of reasoning to apply
            context: Additional reasoning context

        Returns:
            Reasoning result with conclusion and inference chain
        """
        pass

    @abstractmethod
    async def build_inference_chain(
        self,
        premises: list[str],
        target_conclusion: str,
        chain_type: InferenceChainType = InferenceChainType.LINEAR,
        max_steps: int = 10,
    ) -> list[InferenceStep]:
        """
        Build logical inference chain from premises to conclusion

        Args:
            premises: Starting facts and assumptions
            target_conclusion: Desired conclusion to reach
            chain_type: Structure of inference chain
            max_steps: Maximum reasoning steps allowed

        Returns:
            Ordered list of inference steps
        """
        pass

    @abstractmethod
    async def validate_reasoning(
        self, inference_chain: list[InferenceStep], logical_rules: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Validate logical consistency of reasoning chain

        Args:
            inference_chain: Chain of reasoning steps to validate
            logical_rules: Additional logical rules to apply

        Returns:
            Validation results with consistency metrics
        """
        pass

    @abstractmethod
    async def explain_reasoning(self, reasoning_result: ReasoningResult, explanation_depth: str = "summary") -> str:
        """
        Generate human-readable explanation of reasoning process

        Args:
            reasoning_result: Result to explain
            explanation_depth: Level of detail (summary, detailed, technical)

        Returns:
            Natural language explanation of reasoning
        """
        pass

    @abstractmethod
    async def identify_assumptions(self, query: str, premises: list[str]) -> list[str]:
        """
        Identify implicit assumptions in reasoning problem

        Args:
            query: Question being reasoned about
            premises: Explicit premises provided

        Returns:
            List of implicit assumptions
        """
        pass

    @abstractmethod
    async def assess_uncertainty(self, inference_chain: list[InferenceStep]) -> dict[str, float]:
        """
        Assess uncertainty factors in reasoning chain

        Args:
            inference_chain: Reasoning chain to analyze

        Returns:
            Dictionary mapping uncertainty sources to confidence scores
        """
        pass

    @abstractmethod
    async def get_reasoning_stats(self) -> dict[str, Any]:
        """
        Get reasoning system statistics and performance metrics

        Returns:
            Dictionary with system metrics and health info
        """
        pass
