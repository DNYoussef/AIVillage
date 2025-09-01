"""
Configuration for SageAgent services.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ServiceConfig:
    """Base configuration for services."""

    enabled: bool = True
    lazy_load: bool = True
    cache_size: int = 1000
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    config_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveServiceConfig(ServiceConfig):
    """Configuration for cognitive services."""

    cognitive_nexus_enabled: bool = True
    latent_space_activation: bool = True
    foundational_layer_cache: int = 500
    continuous_learning_rate: float = 0.1
    evolution_threshold: float = 0.8


@dataclass
class ProcessingServiceConfig(ServiceConfig):
    """Configuration for processing services."""

    query_processor_enabled: bool = True
    task_executor_enabled: bool = True
    response_generator_enabled: bool = True
    parallel_processing: bool = True
    max_concurrent_tasks: int = 10


@dataclass
class KnowledgeServiceConfig(ServiceConfig):
    """Configuration for knowledge services."""

    vector_store_enabled: bool = True
    rag_system_enabled: bool = True
    knowledge_tracker_enabled: bool = True
    exploration_mode: bool = True
    embedding_cache_size: int = 2000


@dataclass
class LearningServiceConfig(ServiceConfig):
    """Configuration for learning services."""

    continuous_learning: bool = True
    adaptive_error_control: bool = True
    confidence_estimation: bool = True
    performance_tracking: bool = True
    learning_rate: float = 0.01


@dataclass
class CollaborationServiceConfig(ServiceConfig):
    """Configuration for collaboration services."""

    knowledge_sharing: bool = True
    task_delegation: bool = True
    joint_reasoning: bool = True
    max_collaborators: int = 5


@dataclass
class ResearchServiceConfig(ServiceConfig):
    """Configuration for research services."""

    web_scraping: bool = True
    web_search: bool = True
    data_analysis: bool = True
    info_synthesis: bool = True
    request_timeout: float = 10.0


@dataclass
class SageAgentConfig:
    """Main configuration for SageAgent services."""

    # Service configurations
    cognitive: CognitiveServiceConfig = field(default_factory=CognitiveServiceConfig)
    processing: ProcessingServiceConfig = field(default_factory=ProcessingServiceConfig)
    knowledge: KnowledgeServiceConfig = field(default_factory=KnowledgeServiceConfig)
    learning: LearningServiceConfig = field(default_factory=LearningServiceConfig)
    collaboration: CollaborationServiceConfig = field(default_factory=CollaborationServiceConfig)
    research: ResearchServiceConfig = field(default_factory=ResearchServiceConfig)

    # Global settings
    enable_lazy_loading: bool = True
    enable_caching: bool = True
    enable_performance_monitoring: bool = True
    service_timeout_seconds: float = 30.0
    max_memory_usage_mb: int = 1000

    # Research capabilities
    research_capabilities: List[str] = field(
        default_factory=lambda: ["web_scraping", "web_search", "data_analysis", "info_synthesis"]
    )

    # Performance settings
    performance_metrics_enabled: bool = True
    metrics_retention_days: int = 30

    @classmethod
    def from_unified_config(cls, unified_config: Any) -> "SageAgentConfig":
        """Create SageAgentConfig from UnifiedConfig."""
        config = cls()

        # Extract research capabilities
        if hasattr(unified_config, "get"):
            config.research_capabilities = unified_config.get("research_capabilities", config.research_capabilities)

            # Extract other relevant configurations
            if unified_config.get("enable_lazy_loading") is not None:
                config.enable_lazy_loading = unified_config.get("enable_lazy_loading")

            if unified_config.get("enable_caching") is not None:
                config.enable_caching = unified_config.get("enable_caching")

            # Configure services based on unified config
            if unified_config.get("cognitive_services"):
                cognitive_conf = unified_config.get("cognitive_services", {})
                config.cognitive = CognitiveServiceConfig(
                    enabled=cognitive_conf.get("enabled", True),
                    lazy_load=cognitive_conf.get("lazy_load", True),
                    cognitive_nexus_enabled=cognitive_conf.get("cognitive_nexus", True),
                    latent_space_activation=cognitive_conf.get("latent_space", True),
                )

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "cognitive": self.cognitive.__dict__,
            "processing": self.processing.__dict__,
            "knowledge": self.knowledge.__dict__,
            "learning": self.learning.__dict__,
            "collaboration": self.collaboration.__dict__,
            "research": self.research.__dict__,
            "enable_lazy_loading": self.enable_lazy_loading,
            "enable_caching": self.enable_caching,
            "enable_performance_monitoring": self.enable_performance_monitoring,
            "research_capabilities": self.research_capabilities,
        }

    def validate(self) -> List[str]:
        """Validate configuration and return any errors."""
        errors = []

        if self.max_memory_usage_mb < 100:
            errors.append("max_memory_usage_mb must be at least 100 MB")

        if self.service_timeout_seconds < 1.0:
            errors.append("service_timeout_seconds must be at least 1.0 seconds")

        if not self.research_capabilities:
            errors.append("research_capabilities cannot be empty")

        return errors
