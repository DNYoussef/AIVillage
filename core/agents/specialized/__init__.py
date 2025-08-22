"""Specialized Agents Module - Advanced Sub-Agent Implementations

This module provides 8 specialized sub-agents designed to complement the existing
18 primary agents in the AIVillage ecosystem:

1. DataScienceAgent - Statistical analysis, ML training, data insights
2. DevOpsAgent - CI/CD, infrastructure management, deployment automation
3. FinancialAgent - Portfolio optimization, risk analysis, economic modeling
4. CreativeAgent - Story generation, visual design, music composition
5. SocialAgent - Community management, conflict resolution, engagement
6. TranslatorAgent - Multi-language translation, localization, cultural adaptation
7. ArchitectAgent - System architecture, technology selection, scalability planning
8. TesterAgent - Test automation, QA strategies, performance/security testing

These agents are designed to work together through the SpecializedAgentRegistry
for complex multi-agent tasks and specialized domain expertise.
"""

from packages.core.legacy.agents.specialist_agent_registry import SpecialistAgentRegistry, get_specialist_registry

from packages.agents.core.agent_interface import AgentCapability

from .architect_agent import ArchitectAgent
from .creative_agent import CreativeAgent, CreativeRequest
from .data_science_agent import DataAnalysisRequest, DataScienceAgent
from .devops_agent import DeploymentRequest, DevOpsAgent
from .financial_agent import FinancialAgent, FinancialAnalysisRequest
from .social_agent import SocialAgent, SocialInteraction
from .tester_agent import TesterAgent, TestRequest
from .translator_agent import TranslationRequest, TranslatorAgent

SpecializedAgentRegistry = SpecialistAgentRegistry
get_global_registry = get_specialist_registry


def shutdown_global_registry() -> None:  # pragma: no cover - backward compatibility
    """Compatibility no-op for legacy API."""
    return None


__all__ = [
    # Agent Classes
    "DataScienceAgent",
    "DevOpsAgent",
    "FinancialAgent",
    "CreativeAgent",
    "SocialAgent",
    "TranslatorAgent",
    "ArchitectAgent",
    "TesterAgent",
    # Request/Data Classes
    "DataAnalysisRequest",
    "DeploymentRequest",
    "FinancialAnalysisRequest",
    "CreativeRequest",
    "SocialInteraction",
    "TranslationRequest",
    "TestRequest",
    # Registry Classes
    "SpecializedAgentRegistry",
    "AgentCapability",
    "get_global_registry",
    "shutdown_global_registry",
]

# Module metadata
__version__ = "1.0.0"
__author__ = "AIVillage Development Team"
__description__ = "Specialized sub-agents for domain-specific AI tasks"

# Agent capabilities summary
AGENT_CAPABILITIES = {
    "DataScienceAgent": [
        "statistical_analysis",
        "ml_model_training",
        "data_preprocessing",
        "feature_engineering",
        "visualization",
        "ab_testing",
        "time_series_analysis",
        "anomaly_detection",
    ],
    "DevOpsAgent": [
        "ci_cd_management",
        "infrastructure_provisioning",
        "container_orchestration",
        "deployment_automation",
        "monitoring_alerting",
        "service_mesh_management",
        "configuration_management",
        "security_scanning",
    ],
    "FinancialAgent": [
        "portfolio_optimization",
        "risk_analysis",
        "financial_modeling",
        "market_forecasting",
        "sentiment_analysis",
        "trading_strategies",
        "economic_indicators",
        "regulatory_compliance",
    ],
    "CreativeAgent": [
        "story_generation",
        "poetry_writing",
        "visual_design",
        "music_composition",
        "art_direction",
        "brand_creative",
        "character_development",
        "world_building",
    ],
    "SocialAgent": [
        "community_moderation",
        "conflict_resolution",
        "engagement_strategies",
        "relationship_building",
        "cultural_sensitivity",
        "sentiment_monitoring",
        "crisis_communication",
        "inclusive_practices",
    ],
    "TranslatorAgent": [
        "real_time_translation",
        "language_detection",
        "cultural_localization",
        "domain_specialization",
        "multilingual_seo",
        "linguistic_analysis",
        "conversation_translation",
        "document_translation",
    ],
    "ArchitectAgent": [
        "system_architecture",
        "microservices_design",
        "database_modeling",
        "api_design",
        "performance_optimization",
        "scalability_planning",
        "integration_patterns",
        "technology_selection",
    ],
    "TesterAgent": [
        "test_automation",
        "test_strategy_planning",
        "performance_testing",
        "security_testing",
        "coverage_analysis",
        "regression_testing",
        "api_testing",
        "ui_testing",
    ],
}


def get_agent_summary() -> dict:
    """Get summary of all specialized agents and their capabilities"""
    return {
        "total_agents": len(__all__) - 4,  # Exclude non-agent classes
        "total_capabilities": sum(len(caps) for caps in AGENT_CAPABILITIES.values()),
        "agent_types": list(AGENT_CAPABILITIES.keys()),
        "capabilities_by_agent": AGENT_CAPABILITIES,
        "module_version": __version__,
    }
