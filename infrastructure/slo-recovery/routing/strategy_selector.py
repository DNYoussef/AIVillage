"""
SLO Recovery Router - Strategy Selection Engine
Condition-based routing and remedy selection with agent coordination
Target: 30min MTTR with optimal remedy selection
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from .breach_classifier import BreachClassification, BreachSeverity, FailureCategory


class AgentType(Enum):
    SECURITY_MANAGER = "security-manager"
    PRODUCTION_VALIDATOR = "production-validator"
    DEPENDENCY_RESOLVER = "dependency-resolver"
    BUILD_SPECIALIST = "build-specialist"
    CONFIG_MANAGER = "config-manager"
    PATH_VALIDATOR = "path-validator"
    DOC_FORMATTER = "doc-formatter"
    GENERAL_FIXER = "general-fixer"


@dataclass
class RecoveryStrategy:
    strategy_id: str
    name: str
    conditions: List[str]
    route: str
    priority: str
    agents: List[AgentType]
    estimated_duration: int  # minutes
    success_rate: float
    prerequisites: List[str]
    rollback_strategy: Optional[str]
    parallel_execution: bool


@dataclass
class StrategySelection:
    selection_id: str
    breach_classification: BreachClassification
    selected_strategy: RecoveryStrategy
    alternative_strategies: List[RecoveryStrategy]
    confidence_score: float
    execution_plan: Dict
    resource_requirements: Dict
    timestamp: datetime


class StrategySelector:
    """
    DSPy-optimized recovery strategy selection with condition-based routing
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.recovery_strategies = self._initialize_strategies()
        self.strategy_performance_history = {}

    def _initialize_strategies(self) -> List[RecoveryStrategy]:
        """Initialize recovery strategies with condition-based routing"""
        return [
            # Immediate Security Remediation
            RecoveryStrategy(
                strategy_id="ISR001",
                name="immediate_security_remediation",
                conditions=[
                    "security_baseline_failure AND deployment_blocking",
                    "security_baseline_failure AND critical_severity",
                    "deployment_blocking AND security_context",
                ],
                route="immediate_security_remediation",
                priority="critical",
                agents=[AgentType.SECURITY_MANAGER, AgentType.PRODUCTION_VALIDATOR],
                estimated_duration=15,
                success_rate=0.92,
                prerequisites=["security_scan_complete", "baseline_identified"],
                rollback_strategy="security_rollback",
                parallel_execution=True,
            ),
            # Dependency Resolution Workflow
            RecoveryStrategy(
                strategy_id="DRW001",
                name="dependency_resolution_workflow",
                conditions=[
                    "dependency_conflicts AND tool_failures",
                    "dependency_conflicts AND high_severity",
                    "tool_installation_failure AND dependency_context",
                ],
                route="dependency_resolution_workflow",
                priority="high",
                agents=[AgentType.DEPENDENCY_RESOLVER, AgentType.BUILD_SPECIALIST],
                estimated_duration=25,
                success_rate=0.88,
                prerequisites=["dependency_tree_analysis", "conflict_identification"],
                rollback_strategy="dependency_rollback",
                parallel_execution=True,
            ),
            # Configuration Standardization
            RecoveryStrategy(
                strategy_id="CS001",
                name="configuration_standardization",
                conditions=[
                    "configuration_drift AND path_errors",
                    "configuration_drift AND medium_severity",
                    "path_issues AND configuration_context",
                ],
                route="configuration_standardization",
                priority="medium",
                agents=[AgentType.CONFIG_MANAGER, AgentType.PATH_VALIDATOR],
                estimated_duration=15,
                success_rate=0.85,
                prerequisites=["config_baseline", "path_validation"],
                rollback_strategy="config_rollback",
                parallel_execution=False,
            ),
            # Documentation Improvement
            RecoveryStrategy(
                strategy_id="DI001",
                name="documentation_improvement",
                conditions=["documentation_formatting AND low_severity", "documentation_missing AND low_priority"],
                route="documentation_improvement",
                priority="low",
                agents=[AgentType.DOC_FORMATTER],
                estimated_duration=5,
                success_rate=0.95,
                prerequisites=["style_guide_available"],
                rollback_strategy=None,
                parallel_execution=False,
            ),
            # Multi-Vector Recovery (Complex scenarios)
            RecoveryStrategy(
                strategy_id="MVR001",
                name="multi_vector_recovery",
                conditions=[
                    "multiple_failure_categories",
                    "high_complexity AND multiple_systems",
                    "cascading_failures",
                ],
                route="multi_vector_recovery",
                priority="critical",
                agents=[
                    AgentType.SECURITY_MANAGER,
                    AgentType.DEPENDENCY_RESOLVER,
                    AgentType.CONFIG_MANAGER,
                    AgentType.PRODUCTION_VALIDATOR,
                ],
                estimated_duration=30,
                success_rate=0.82,
                prerequisites=["full_system_analysis", "impact_assessment"],
                rollback_strategy="full_system_rollback",
                parallel_execution=True,
            ),
            # General Remediation (Fallback)
            RecoveryStrategy(
                strategy_id="GR001",
                name="general_remediation",
                conditions=["unknown_failure_pattern", "low_confidence_classification", "fallback_required"],
                route="general_remediation",
                priority="medium",
                agents=[AgentType.GENERAL_FIXER, AgentType.BUILD_SPECIALIST],
                estimated_duration=20,
                success_rate=0.75,
                prerequisites=["basic_diagnostics"],
                rollback_strategy="general_rollback",
                parallel_execution=False,
            ),
        ]

    def select_strategy(self, classification: BreachClassification) -> StrategySelection:
        """
        Select optimal recovery strategy based on breach classification
        """
        # Score all strategies based on condition matching
        strategy_scores = []

        for strategy in self.recovery_strategies:
            score = self._calculate_strategy_score(strategy, classification)
            if score > 0:
                strategy_scores.append((strategy, score))

        # Sort by score (highest first)
        strategy_scores.sort(key=lambda x: x[1], reverse=True)

        if not strategy_scores:
            # Fallback to general remediation
            general_strategy = next(s for s in self.recovery_strategies if s.strategy_id == "GR001")
            selected_strategy = general_strategy
            alternatives = []
            confidence = 0.5
        else:
            selected_strategy = strategy_scores[0][0]
            alternatives = [s[0] for s in strategy_scores[1:3]]  # Top 2 alternatives
            confidence = strategy_scores[0][1]

        # Generate execution plan
        execution_plan = self._generate_execution_plan(selected_strategy, classification)

        # Calculate resource requirements
        resource_requirements = self._calculate_resource_requirements(selected_strategy)

        # Create strategy selection
        selection_id = f"SEL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        selection = StrategySelection(
            selection_id=selection_id,
            breach_classification=classification,
            selected_strategy=selected_strategy,
            alternative_strategies=alternatives,
            confidence_score=confidence,
            execution_plan=execution_plan,
            resource_requirements=resource_requirements,
            timestamp=datetime.now(),
        )

        return selection

    def _calculate_strategy_score(self, strategy: RecoveryStrategy, classification: BreachClassification) -> float:
        """Calculate strategy score based on condition matching and historical performance"""

        # Base score from condition matching
        condition_score = self._evaluate_conditions(strategy.conditions, classification)

        if condition_score == 0:
            return 0

        # Historical performance adjustment
        historical_performance = self.strategy_performance_history.get(strategy.strategy_id, strategy.success_rate)

        # Priority alignment score
        priority_score = self._calculate_priority_alignment(strategy.priority, classification.severity)

        # Duration penalty (prefer faster strategies)
        duration_penalty = max(0, 1 - (strategy.estimated_duration / 60))  # Normalize to 0-1

        # Combine scores with weights
        total_score = (
            condition_score * 0.4 + historical_performance * 0.3 + priority_score * 0.2 + duration_penalty * 0.1
        )

        return min(1.0, total_score)

    def _evaluate_conditions(self, conditions: List[str], classification: BreachClassification) -> float:
        """Evaluate strategy conditions against breach classification"""

        # Create context for condition evaluation
        context = {
            "security_baseline_failure": classification.category == FailureCategory.SECURITY_BASELINE,
            "deployment_blocking": classification.category == FailureCategory.DEPLOYMENT_BLOCKING,
            "dependency_conflicts": classification.category == FailureCategory.DEPENDENCY_CONFLICTS,
            "tool_failures": classification.category == FailureCategory.TOOL_INSTALLATION,
            "tool_installation_failure": classification.category == FailureCategory.TOOL_INSTALLATION,
            "configuration_drift": classification.category == FailureCategory.CONFIGURATION_DRIFT,
            "path_errors": classification.category == FailureCategory.PATH_ISSUES,
            "path_issues": classification.category == FailureCategory.PATH_ISSUES,
            "documentation_formatting": classification.category == FailureCategory.DOCUMENTATION,
            "documentation_missing": classification.category == FailureCategory.DOCUMENTATION,
            "critical_severity": classification.severity == BreachSeverity.CRITICAL,
            "high_severity": classification.severity == BreachSeverity.HIGH,
            "medium_severity": classification.severity == BreachSeverity.MEDIUM,
            "low_severity": classification.severity == BreachSeverity.LOW,
            "security_context": "security" in " ".join(classification.indicators_matched),
            "dependency_context": "dependency" in " ".join(classification.indicators_matched),
            "configuration_context": "config" in " ".join(classification.indicators_matched),
            "multiple_failure_categories": len(classification.indicators_matched) > 3,
            "high_complexity": classification.priority_score > 80,
            "multiple_systems": len(classification.indicators_matched) > 5,
            "cascading_failures": "cascade" in " ".join(classification.indicators_matched),
            "unknown_failure_pattern": classification.confidence_score < 0.7,
            "low_confidence_classification": classification.confidence_score < 0.65,
            "fallback_required": classification.confidence_score < 0.6,
            "low_priority": classification.priority_score < 40,
        }

        # Evaluate each condition
        matches = 0
        for condition in conditions:
            if self._evaluate_single_condition(condition, context):
                matches += 1

        return matches / len(conditions) if conditions else 0

    def _evaluate_single_condition(self, condition: str, context: Dict) -> bool:
        """Evaluate a single condition string against context"""
        try:
            # Replace context variables in condition
            for var, value in context.items():
                condition = condition.replace(var, str(value))

            # Evaluate boolean expression
            return eval(condition)
        except Exception as e:
            self.logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return False

    def _calculate_priority_alignment(self, strategy_priority: str, breach_severity: BreachSeverity) -> float:
        """Calculate alignment between strategy priority and breach severity"""
        priority_map = {"critical": 4, "high": 3, "medium": 2, "low": 1}

        severity_map = {
            BreachSeverity.CRITICAL: 4,
            BreachSeverity.HIGH: 3,
            BreachSeverity.MEDIUM: 2,
            BreachSeverity.LOW: 1,
        }

        strategy_level = priority_map.get(strategy_priority, 2)
        breach_level = severity_map.get(breach_severity, 2)

        # Perfect match = 1.0, one level off = 0.7, etc.
        diff = abs(strategy_level - breach_level)
        return max(0.3, 1.0 - (diff * 0.3))

    def _generate_execution_plan(self, strategy: RecoveryStrategy, classification: BreachClassification) -> Dict:
        """Generate detailed execution plan for selected strategy"""

        plan = {
            "strategy_id": strategy.strategy_id,
            "strategy_name": strategy.name,
            "execution_mode": "parallel" if strategy.parallel_execution else "sequential",
            "estimated_duration": strategy.estimated_duration,
            "phases": [],
            "agent_assignments": [],
            "checkpoints": [],
            "rollback_triggers": [],
        }

        # Generate phases based on strategy type
        if strategy.name == "immediate_security_remediation":
            plan["phases"] = [
                {"phase": "security_assessment", "duration": 5, "agents": ["security-manager"]},
                {
                    "phase": "vulnerability_remediation",
                    "duration": 8,
                    "agents": ["security-manager", "production-validator"],
                },
                {"phase": "validation_testing", "duration": 2, "agents": ["production-validator"]},
            ]
        elif strategy.name == "dependency_resolution_workflow":
            plan["phases"] = [
                {"phase": "dependency_analysis", "duration": 8, "agents": ["dependency-resolver"]},
                {"phase": "conflict_resolution", "duration": 12, "agents": ["dependency-resolver", "build-specialist"]},
                {"phase": "build_validation", "duration": 5, "agents": ["build-specialist"]},
            ]
        elif strategy.name == "configuration_standardization":
            plan["phases"] = [
                {"phase": "config_audit", "duration": 5, "agents": ["config-manager"]},
                {"phase": "path_validation", "duration": 5, "agents": ["path-validator"]},
                {"phase": "standardization", "duration": 5, "agents": ["config-manager"]},
            ]

        # Agent assignments
        for i, agent in enumerate(strategy.agents):
            plan["agent_assignments"].append(
                {
                    "agent_id": f"agent_{i+1}",
                    "agent_type": agent.value,
                    "primary_responsibility": self._get_agent_responsibility(agent, classification),
                    "parallel_execution": strategy.parallel_execution,
                }
            )

        # Checkpoints
        plan["checkpoints"] = [
            {"checkpoint": "prerequisites_verified", "time": 0},
            {"checkpoint": "remediation_started", "time": 2},
            {"checkpoint": "progress_assessment", "time": strategy.estimated_duration // 2},
            {"checkpoint": "remediation_complete", "time": strategy.estimated_duration},
        ]

        # Rollback triggers
        if strategy.rollback_strategy:
            plan["rollback_triggers"] = [
                {"condition": "remediation_failure", "threshold": 0.3},
                {"condition": "time_exceeded", "threshold": strategy.estimated_duration * 1.5},
                {"condition": "system_instability", "threshold": 0.2},
            ]

        return plan

    def _get_agent_responsibility(self, agent: AgentType, classification: BreachClassification) -> str:
        """Get primary responsibility for agent based on classification"""
        responsibilities = {
            AgentType.SECURITY_MANAGER: "security_remediation_and_validation",
            AgentType.PRODUCTION_VALIDATOR: "production_readiness_validation",
            AgentType.DEPENDENCY_RESOLVER: "dependency_conflict_resolution",
            AgentType.BUILD_SPECIALIST: "build_system_optimization",
            AgentType.CONFIG_MANAGER: "configuration_standardization",
            AgentType.PATH_VALIDATOR: "path_resolution_and_validation",
            AgentType.DOC_FORMATTER: "documentation_formatting",
            AgentType.GENERAL_FIXER: "general_issue_resolution",
        }
        return responsibilities.get(agent, "general_support")

    def _calculate_resource_requirements(self, strategy: RecoveryStrategy) -> Dict:
        """Calculate resource requirements for strategy execution"""

        base_requirements = {
            "agents": len(strategy.agents),
            "estimated_duration": strategy.estimated_duration,
            "parallel_slots": len(strategy.agents) if strategy.parallel_execution else 1,
            "memory_mb": 512 * len(strategy.agents),
            "cpu_cores": 2 if strategy.parallel_execution else 1,
        }

        # Adjust based on strategy complexity
        if strategy.name == "multi_vector_recovery":
            base_requirements["memory_mb"] *= 2
            base_requirements["cpu_cores"] += 1

        return base_requirements

    def generate_strategy_selection_map(self) -> Dict:
        """Generate recovery strategy selection mapping for output"""

        strategy_map = {
            "strategies": [],
            "condition_evaluation": {},
            "agent_capabilities": {},
            "priority_matrix": {},
            "performance_metrics": {},
        }

        # Strategies
        for strategy in self.recovery_strategies:
            strategy_map["strategies"].append(
                {
                    "strategy_id": strategy.strategy_id,
                    "name": strategy.name,
                    "conditions": strategy.conditions,
                    "route": strategy.route,
                    "priority": strategy.priority,
                    "agents": [agent.value for agent in strategy.agents],
                    "estimated_duration": strategy.estimated_duration,
                    "success_rate": strategy.success_rate,
                    "prerequisites": strategy.prerequisites,
                    "parallel_execution": strategy.parallel_execution,
                }
            )

        # Condition evaluation logic
        strategy_map["condition_evaluation"] = {
            "supported_operators": ["AND", "OR", "NOT"],
            "context_variables": [
                "security_baseline_failure",
                "deployment_blocking",
                "dependency_conflicts",
                "tool_failures",
                "configuration_drift",
                "path_errors",
                "critical_severity",
                "high_severity",
                "medium_severity",
                "low_severity",
            ],
        }

        # Agent capabilities
        for agent in AgentType:
            strategy_map["agent_capabilities"][agent.value] = self._get_agent_responsibility(agent, None)

        # Priority matrix
        strategy_map["priority_matrix"] = {
            "critical": {"score_range": [85, 100], "max_duration": 20},
            "high": {"score_range": [70, 84], "max_duration": 25},
            "medium": {"score_range": [45, 69], "max_duration": 15},
            "low": {"score_range": [0, 44], "max_duration": 10},
        }

        # Performance metrics
        strategy_map["performance_metrics"] = {
            "target_success_rate": 0.928,
            "target_mttr": 30,
            "confidence_threshold": 0.75,
        }

        return strategy_map


# Export for use by other components
__all__ = ["StrategySelector", "StrategySelection", "RecoveryStrategy", "AgentType"]
