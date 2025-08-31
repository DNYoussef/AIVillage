"""Performance and incentive model constants."""

from dataclasses import dataclass
from enum import Enum
from typing import Final


class PerformanceLevel(Enum):
    """Performance level enumeration."""
    POOR = 0.5
    BELOW_AVERAGE = 0.7
    AVERAGE = 1.0
    ABOVE_AVERAGE = 1.3
    EXCELLENT = 2.0


class RewardFactors(Enum):
    """Reward calculation factors."""
    BASE_SUCCESS_REWARD = 10
    INNOVATION_BONUS = 0.5
    COLLABORATION_BONUS = 0.3
    QUALITY_WEIGHT = 0.5
    TIME_WEIGHT = 1.0
    COST_WEIGHT = 1.0


@dataclass(frozen=True)
class IncentiveDefaults:
    """Default values for incentive model calculations."""
    
    # Learning parameters
    DEFAULT_LEARNING_RATE: Final[float] = 0.01
    DEFAULT_HISTORY_LENGTH: Final[int] = 1000
    
    # Performance factors
    PERFORMANCE_INCREASE_FACTOR: Final[float] = 1.1
    PERFORMANCE_DECREASE_FACTOR: Final[float] = 0.9
    MAX_PERFORMANCE_MULTIPLIER: Final[float] = 2.0
    MIN_PERFORMANCE_MULTIPLIER: Final[float] = 0.5
    
    # Specialization and collaboration updates
    SPECIALIZATION_UPDATE_RATE: Final[float] = 0.1
    COLLABORATION_UPDATE_RATE: Final[float] = 0.1
    INNOVATION_UPDATE_RATE: Final[float] = 0.1
    
    # PCA components for analysis
    DEFAULT_PCA_COMPONENTS: Final[int] = 5
    
    # Trend calculation limits
    MIN_TREND_FACTOR: Final[float] = 0.5
    MAX_TREND_FACTOR: Final[float] = 1.5
    NEUTRAL_TREND_FACTOR: Final[float] = 1.0
    
    # Task difficulty normalization
    MAX_NORMALIZED_DIFFICULTY: Final[float] = 1.0
    MIN_NORMALIZED_DIFFICULTY: Final[float] = 0.0
    DIFFICULTY_NORMALIZATION_DIVISOR: Final[int] = 100
    SKILL_FACTOR_MULTIPLIER: Final[float] = 0.1


@dataclass(frozen=True)
class PerformanceConstants:
    """Core performance tracking constants."""
    
    # Learning rate and history
    LEARNING_RATE: Final[float] = IncentiveDefaults.DEFAULT_LEARNING_RATE
    HISTORY_LENGTH: Final[int] = IncentiveDefaults.DEFAULT_HISTORY_LENGTH
    
    # Performance multipliers
    PERFORMANCE_BOOST_FACTOR: Final[float] = IncentiveDefaults.PERFORMANCE_INCREASE_FACTOR
    PERFORMANCE_PENALTY_FACTOR: Final[float] = IncentiveDefaults.PERFORMANCE_DECREASE_FACTOR
    MAX_PERFORMANCE: Final[float] = IncentiveDefaults.MAX_PERFORMANCE_MULTIPLIER
    MIN_PERFORMANCE: Final[float] = IncentiveDefaults.MIN_PERFORMANCE_MULTIPLIER
    
    # Update rates for different metrics
    SPECIALIZATION_RATE: Final[float] = IncentiveDefaults.SPECIALIZATION_UPDATE_RATE
    COLLABORATION_RATE: Final[float] = IncentiveDefaults.COLLABORATION_UPDATE_RATE
    INNOVATION_RATE: Final[float] = IncentiveDefaults.INNOVATION_UPDATE_RATE
    
    # Analysis parameters
    PCA_COMPONENTS: Final[int] = IncentiveDefaults.DEFAULT_PCA_COMPONENTS
    
    # Trend factors
    MIN_TREND: Final[float] = IncentiveDefaults.MIN_TREND_FACTOR
    MAX_TREND: Final[float] = IncentiveDefaults.MAX_TREND_FACTOR
    NEUTRAL_TREND: Final[float] = IncentiveDefaults.NEUTRAL_TREND_FACTOR


@dataclass(frozen=True)
class RewardConstants:
    """Constants for reward calculation."""
    
    # Base reward values
    BASE_SUCCESS_REWARD: Final[int] = RewardFactors.BASE_SUCCESS_REWARD.value
    INNOVATION_BONUS: Final[float] = RewardFactors.INNOVATION_BONUS.value
    COLLABORATION_BONUS: Final[float] = RewardFactors.COLLABORATION_BONUS.value
    
    # Weight factors for reward calculation
    QUALITY_WEIGHT: Final[float] = RewardFactors.QUALITY_WEIGHT.value
    TIME_WEIGHT: Final[float] = RewardFactors.TIME_WEIGHT.value
    COST_WEIGHT: Final[float] = RewardFactors.COST_WEIGHT.value
    
    # Reward calculation divisor for normalization
    REWARD_NORMALIZATION_DIVISOR: Final[int] = 3
    
    # Default values for missing reward components
    DEFAULT_QUALITY_SCORE: Final[float] = 0.5
    DEFAULT_EXPECTED_TIME: Final[int] = 1
    DEFAULT_BUDGET: Final[int] = 1
    DEFAULT_TIME_TAKEN: Final[int] = 0
    DEFAULT_COST: Final[int] = 0


@dataclass(frozen=True)
class TaskDifficultyConstants:
    """Constants for task difficulty calculation."""
    
    # Default values for missing difficulty components
    DEFAULT_COMPLEXITY: Final[int] = 1
    DEFAULT_ESTIMATED_TIME: Final[int] = 1
    DEFAULT_PRIORITY: Final[int] = 1
    
    # Normalization parameters
    DIFFICULTY_MAX: Final[float] = IncentiveDefaults.MAX_NORMALIZED_DIFFICULTY
    DIFFICULTY_MIN: Final[float] = IncentiveDefaults.MIN_NORMALIZED_DIFFICULTY
    DIFFICULTY_DIVISOR: Final[int] = IncentiveDefaults.DIFFICULTY_NORMALIZATION_DIVISOR
    SKILL_MULTIPLIER: Final[float] = IncentiveDefaults.SKILL_FACTOR_MULTIPLIER


# Dictionary keys for consistent access
@dataclass(frozen=True)
class PerformanceFieldNames:
    """Field names for performance data structures."""
    
    # Result fields
    SUCCESS_FIELD: Final[str] = "success"
    TIME_TAKEN_FIELD: Final[str] = "time_taken"
    EXPECTED_TIME_FIELD: Final[str] = "expected_time"
    QUALITY_FIELD: Final[str] = "quality"
    COST_FIELD: Final[str] = "cost"
    BUDGET_FIELD: Final[str] = "budget"
    INNOVATIVE_SOLUTION_FIELD: Final[str] = "innovative_solution"
    COLLABORATORS_FIELD: Final[str] = "collaborators"
    
    # Task fields
    ASSIGNED_AGENT_FIELD: Final[str] = "assigned_agent"
    TASK_ID_FIELD: Final[str] = "task_id"
    TASK_TYPE_FIELD: Final[str] = "type"
    DESCRIPTION_FIELD: Final[str] = "description"
    COMPLEXITY_FIELD: Final[str] = "complexity"
    PRIORITY_FIELD: Final[str] = "priority"
    ESTIMATED_TIME_FIELD: Final[str] = "estimated_time"
    REQUIRED_SKILLS_FIELD: Final[str] = "required_skills"
    
    # Analysis result fields
    AVERAGE_FIELD: Final[str] = "average"
    TREND_FIELD: Final[str] = "trend"
    LONG_TERM_FIELD: Final[str] = "long_term"
    SPECIALIZATION_FIELD: Final[str] = "specialization"
    COLLABORATION_FIELD: Final[str] = "collaboration"
    INNOVATION_FIELD: Final[str] = "innovation"
    PCA_COMPONENTS_FIELD: Final[str] = "pca_components"