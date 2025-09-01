"""
SLO Recovery Router - Intelligent Problem Classification and Remedy Selection
Following SLO Breach Recovery Loop ROUTE TO REMEDIES phase

This module provides intelligent problem classification and remedy selection
with 30min MTTR target and 92.8%+ success rate through:

1. Breach Classification: Priority-based routing with adaptive thresholds
2. Strategy Selection: Condition-based routing to optimal remedies  
3. Parallel Coordination: Multi-agent execution with conflict resolution
4. Escalation Management: Human intervention procedures with cost optimization
5. Integration Adapters: Real-time data feeds from Flake Detector and GitHub Orchestrator
6. Validation Optimizer: DSPy-based success rate optimization

Target Performance:
- Success Rate: 92.8%+
- MTTR: 30 minutes maximum
- Confidence Threshold: 75%+
- Escalation Rate: <15%
"""

from .breach_classifier import BreachClassifier, BreachClassification, BreachSeverity, FailureCategory

from .strategy_selector import StrategySelector, StrategySelection, RecoveryStrategy, AgentType

from .parallel_coordinator import ParallelCoordinator, CoordinationPlan, AgentExecution, CoordinationStatus

from .escalation_manager import EscalationManager, EscalationEvent, EscalationLevel, EscalationTrigger

from .slo_recovery_router import SLORecoveryRouter, RoutingDecision

from .integration_adapter import (
    IntegrationCoordinator,
    FlakeDetectorAdapter,
    GitHubOrchestratorAdapter,
    create_integration_coordinator,
)

from .validation_optimizer import ValidationOptimizer, ValidationMetrics, OptimizationResult

__version__ = "1.0.0"

__all__ = [
    # Core routing components
    "SLORecoveryRouter",
    "RoutingDecision",
    # Breach classification
    "BreachClassifier",
    "BreachClassification",
    "BreachSeverity",
    "FailureCategory",
    # Strategy selection
    "StrategySelector",
    "StrategySelection",
    "RecoveryStrategy",
    "AgentType",
    # Parallel coordination
    "ParallelCoordinator",
    "CoordinationPlan",
    "AgentExecution",
    "CoordinationStatus",
    # Escalation management
    "EscalationManager",
    "EscalationEvent",
    "EscalationLevel",
    "EscalationTrigger",
    # Integration adapters
    "IntegrationCoordinator",
    "FlakeDetectorAdapter",
    "GitHubOrchestratorAdapter",
    "create_integration_coordinator",
    # Validation and optimization
    "ValidationOptimizer",
    "ValidationMetrics",
    "OptimizationResult",
]

# Configuration defaults
DEFAULT_CONFIG = {
    "target_success_rate": 0.928,
    "target_mttr_minutes": 30,
    "confidence_threshold": 0.75,
    "escalation_rate_threshold": 0.15,
    "polling_interval_seconds": 30,
    "optimization_window_days": 7,
}


def create_slo_recovery_system(
    flake_detector_config=None, github_orchestrator_config=None, database_path="slo_routing_metrics.db"
):
    """
    Factory function to create complete SLO Recovery Router system

    Args:
        flake_detector_config: Configuration for Flake Detector integration
        github_orchestrator_config: Configuration for GitHub Orchestrator integration
        database_path: Path for metrics database

    Returns:
        Tuple of (router, coordinator, optimizer) components
    """

    # Create core router
    router = SLORecoveryRouter()

    # Create integration coordinator
    coordinator = create_integration_coordinator(router, flake_detector_config, github_orchestrator_config)

    # Create validation optimizer
    optimizer = ValidationOptimizer(database_path)

    return router, coordinator, optimizer
