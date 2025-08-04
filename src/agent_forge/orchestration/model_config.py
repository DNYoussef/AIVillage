"""Model routing configuration for multi-model orchestration.

Maps task types to optimal models with fallback strategies and cost tiers.
"""

from enum import Enum
from typing import Any


class TaskType(Enum):
    """Types of tasks that can be routed to different models."""

    PROBLEM_GENERATION = "problem_generation"
    EVALUATION_GRADING = "evaluation_grading"
    CONTENT_VARIATION = "content_variation"
    RESEARCH_DOCUMENTATION = "research_documentation"
    CODE_GENERATION = "code_generation"
    MATHEMATICAL_REASONING = "mathematical_reasoning"


class CostTier(Enum):
    """Cost tiers for model selection."""

    PREMIUM = "premium"  # Highest quality, highest cost
    BALANCED = "balanced"  # Good quality, moderate cost
    BUDGET = "budget"  # Acceptable quality, lowest cost


# Model routing configuration
MODEL_ROUTING_CONFIG: dict[TaskType, dict[str, Any]] = {
    TaskType.PROBLEM_GENERATION: {
        "primary": "anthropic/claude-3-opus-20240229",
        "fallback": ["google/gemini-pro-1.5", "anthropic/claude-3-sonnet-20240229"],
        "cost_tier": CostTier.PREMIUM,
        "max_tokens": 2000,
        "temperature": 0.8,
        "description": "Complex reasoning tasks and high-quality problem generation",
    },
    TaskType.EVALUATION_GRADING: {
        "primary": "openai/gpt-4o-mini",
        "fallback": ["anthropic/claude-3-haiku-20240307", "google/gemini-flash-1.5"],
        "cost_tier": CostTier.BUDGET,
        "max_tokens": 1000,
        "temperature": 0.3,
        "description": "Efficient evaluation and grading of responses",
    },
    TaskType.CONTENT_VARIATION: {
        "primary": "openai/gpt-4o-mini",
        "fallback": [
            "anthropic/claude-3-haiku-20240307",
            "meta-llama/llama-3.1-70b-instruct",
        ],
        "cost_tier": CostTier.BUDGET,
        "max_tokens": 1500,
        "temperature": 0.7,
        "description": "Creating variations of existing content",
    },
    TaskType.RESEARCH_DOCUMENTATION: {
        "primary": "google/gemini-pro-1.5",
        "fallback": ["anthropic/claude-3-opus-20240229", "openai/gpt-4-turbo"],
        "cost_tier": CostTier.BALANCED,
        "max_tokens": 4000,
        "temperature": 0.5,
        "description": "Long-context analysis and documentation",
    },
    TaskType.CODE_GENERATION: {
        "primary": "anthropic/claude-3-opus-20240229",
        "fallback": ["openai/gpt-4-turbo", "deepseek/deepseek-coder-v2-instruct"],
        "cost_tier": CostTier.PREMIUM,
        "max_tokens": 3000,
        "temperature": 0.2,
        "description": "High-quality code generation for Magi specialization",
    },
    TaskType.MATHEMATICAL_REASONING: {
        "primary": "anthropic/claude-3-opus-20240229",
        "fallback": ["openai/gpt-4-turbo", "google/gemini-pro-1.5"],
        "cost_tier": CostTier.PREMIUM,
        "max_tokens": 2500,
        "temperature": 0.1,
        "description": "Mathematical proofs and complex reasoning",
    },
}


# Cost limits per task type (in USD)
COST_LIMITS = {
    TaskType.PROBLEM_GENERATION: 0.10,
    TaskType.EVALUATION_GRADING: 0.01,
    TaskType.CONTENT_VARIATION: 0.02,
    TaskType.RESEARCH_DOCUMENTATION: 0.05,
    TaskType.CODE_GENERATION: 0.10,
    TaskType.MATHEMATICAL_REASONING: 0.10,
}


# Rate limits per model (requests per minute)
RATE_LIMITS = {
    "anthropic/claude-3-opus-20240229": 10,
    "anthropic/claude-3-sonnet-20240229": 20,
    "anthropic/claude-3-haiku-20240307": 50,
    "openai/gpt-4-turbo": 10,
    "openai/gpt-4o-mini": 100,
    "google/gemini-pro-1.5": 20,
    "google/gemini-flash-1.5": 50,
    "meta-llama/llama-3.1-70b-instruct": 30,
    "deepseek/deepseek-coder-v2-instruct": 30,
}
