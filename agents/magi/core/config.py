"""Configuration for MAGI agent system."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

@dataclass
class TechniqueConfig:
    """Configuration for reasoning techniques."""
    enabled_techniques: List[str] = field(default_factory=lambda: [
        "chain_of_thought",
        "tree_of_thoughts",
        "program_of_thoughts",
        "self_ask",
        "least_to_most",
        "contrastive_chain",
        "memory_of_thought",
        "choice_annealing",
        "prompt_chaining",
        "self_consistency",
        "evolutionary_tournament"
    ])
    technique_weights: Dict[str, float] = field(default_factory=lambda: {
        "chain_of_thought": 1.0,
        "tree_of_thoughts": 1.0,
        "program_of_thoughts": 1.0,
        "self_ask": 1.0,
        "least_to_most": 1.0,
        "contrastive_chain": 1.0,
        "memory_of_thought": 1.0,
        "choice_annealing": 1.0,
        "prompt_chaining": 1.0,
        "self_consistency": 1.0,
        "evolutionary_tournament": 1.0
    })

@dataclass
class ToolConfig:
    """Configuration for tool management."""
    tool_storage_path: Path = Path("tools_storage")
    max_tools: int = 100
    tool_creation_timeout: int = 30  # seconds
    tool_execution_timeout: int = 60  # seconds
    allowed_imports: List[str] = field(default_factory=lambda: [
        "os", "sys", "json", "yaml", "numpy", "pandas", 
        "requests", "aiohttp", "asyncio", "datetime"
    ])

@dataclass
class LearningConfig:
    """Configuration for continuous learning."""
    learning_rate: float = 0.01
    min_learning_rate: float = 0.001
    max_learning_rate: float = 0.1
    performance_window: int = 50
    evolution_interval: int = 100
    min_samples_for_evolution: int = 50
    max_history_size: int = 1000

@dataclass
class QAConfig:
    """Configuration for quality assurance."""
    safety_threshold: float = 0.7
    performance_threshold: float = 0.6
    uncertainty_threshold: float = 0.3
    max_retries: int = 3
    validation_rules: List[str] = field(default_factory=lambda: [
        "Move all living things towards eudaimonia",
        "Embrace and encourage curiosity",
        "Protect the AI village and its inhabitants",
        "Maintain self-preservation unless it interferes with the other rules"
    ])

@dataclass
class ResourceConfig:
    """Configuration for resource management."""
    max_memory_usage: int = 1024 * 1024 * 1024  # 1GB
    max_cpu_percent: float = 80.0
    max_concurrent_tasks: int = 10
    task_timeout: int = 300  # seconds
    max_retries_per_task: int = 3

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_level: str = "INFO"
    log_file: Optional[Path] = Path("logs/magi.log")
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_performance_logging: bool = True
    enable_debug_logging: bool = False

@dataclass
class MAGIConfig:
    """Main configuration for MAGI agent system."""
    techniques: TechniqueConfig = field(default_factory=TechniqueConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    qa: QAConfig = field(default_factory=QAConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Agent-specific settings
    agent_name: str = "MAGI"
    agent_version: str = "1.0.0"
    agent_description: str = "Multi-Agent General Intelligence"
    
    # Model settings
    model_name: str = "gpt-4"
    model_temperature: float = 0.7
    max_tokens: int = 2000
    
    # System paths
    base_path: Path = Path("agents/magi")
    data_path: Path = Path("data")
    cache_path: Path = Path("cache")
    
    def __post_init__(self):
        """Ensure all paths exist."""
        for path in [self.base_path, self.data_path, self.cache_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        if self.logging.log_file:
            self.logging.log_file.parent.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "techniques": {
                "enabled_techniques": self.techniques.enabled_techniques,
                "technique_weights": self.techniques.technique_weights
            },
            "tools": {
                "tool_storage_path": str(self.tools.tool_storage_path),
                "max_tools": self.tools.max_tools,
                "tool_creation_timeout": self.tools.tool_creation_timeout,
                "tool_execution_timeout": self.tools.tool_execution_timeout,
                "allowed_imports": self.tools.allowed_imports
            },
            "learning": {
                "learning_rate": self.learning.learning_rate,
                "min_learning_rate": self.learning.min_learning_rate,
                "max_learning_rate": self.learning.max_learning_rate,
                "performance_window": self.learning.performance_window,
                "evolution_interval": self.learning.evolution_interval,
                "min_samples_for_evolution": self.learning.min_samples_for_evolution,
                "max_history_size": self.learning.max_history_size
            },
            "qa": {
                "safety_threshold": self.qa.safety_threshold,
                "performance_threshold": self.qa.performance_threshold,
                "uncertainty_threshold": self.qa.uncertainty_threshold,
                "max_retries": self.qa.max_retries,
                "validation_rules": self.qa.validation_rules
            },
            "resources": {
                "max_memory_usage": self.resources.max_memory_usage,
                "max_cpu_percent": self.resources.max_cpu_percent,
                "max_concurrent_tasks": self.resources.max_concurrent_tasks,
                "task_timeout": self.resources.task_timeout,
                "max_retries_per_task": self.resources.max_retries_per_task
            },
            "logging": {
                "log_level": self.logging.log_level,
                "log_file": str(self.logging.log_file) if self.logging.log_file else None,
                "max_log_size": self.logging.max_log_size,
                "backup_count": self.logging.backup_count,
                "log_format": self.logging.log_format,
                "enable_performance_logging": self.logging.enable_performance_logging,
                "enable_debug_logging": self.logging.enable_debug_logging
            },
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "agent_description": self.agent_description,
            "model_name": self.model_name,
            "model_temperature": self.model_temperature,
            "max_tokens": self.max_tokens,
            "base_path": str(self.base_path),
            "data_path": str(self.data_path),
            "cache_path": str(self.cache_path)
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MAGIConfig':
        """Create config from dictionary."""
        techniques = TechniqueConfig(
            enabled_techniques=config_dict["techniques"]["enabled_techniques"],
            technique_weights=config_dict["techniques"]["technique_weights"]
        )
        
        tools = ToolConfig(
            tool_storage_path=Path(config_dict["tools"]["tool_storage_path"]),
            max_tools=config_dict["tools"]["max_tools"],
            tool_creation_timeout=config_dict["tools"]["tool_creation_timeout"],
            tool_execution_timeout=config_dict["tools"]["tool_execution_timeout"],
            allowed_imports=config_dict["tools"]["allowed_imports"]
        )
        
        learning = LearningConfig(
            learning_rate=config_dict["learning"]["learning_rate"],
            min_learning_rate=config_dict["learning"]["min_learning_rate"],
            max_learning_rate=config_dict["learning"]["max_learning_rate"],
            performance_window=config_dict["learning"]["performance_window"],
            evolution_interval=config_dict["learning"]["evolution_interval"],
            min_samples_for_evolution=config_dict["learning"]["min_samples_for_evolution"],
            max_history_size=config_dict["learning"]["max_history_size"]
        )
        
        qa = QAConfig(
            safety_threshold=config_dict["qa"]["safety_threshold"],
            performance_threshold=config_dict["qa"]["performance_threshold"],
            uncertainty_threshold=config_dict["qa"]["uncertainty_threshold"],
            max_retries=config_dict["qa"]["max_retries"],
            validation_rules=config_dict["qa"]["validation_rules"]
        )
        
        resources = ResourceConfig(
            max_memory_usage=config_dict["resources"]["max_memory_usage"],
            max_cpu_percent=config_dict["resources"]["max_cpu_percent"],
            max_concurrent_tasks=config_dict["resources"]["max_concurrent_tasks"],
            task_timeout=config_dict["resources"]["task_timeout"],
            max_retries_per_task=config_dict["resources"]["max_retries_per_task"]
        )
        
        logging_config = LoggingConfig(
            log_level=config_dict["logging"]["log_level"],
            log_file=Path(config_dict["logging"]["log_file"]) if config_dict["logging"]["log_file"] else None,
            max_log_size=config_dict["logging"]["max_log_size"],
            backup_count=config_dict["logging"]["backup_count"],
            log_format=config_dict["logging"]["log_format"],
            enable_performance_logging=config_dict["logging"]["enable_performance_logging"],
            enable_debug_logging=config_dict["logging"]["enable_debug_logging"]
        )
        
        return cls(
            techniques=techniques,
            tools=tools,
            learning=learning,
            qa=qa,
            resources=resources,
            logging=logging_config,
            agent_name=config_dict["agent_name"],
            agent_version=config_dict["agent_version"],
            agent_description=config_dict["agent_description"],
            model_name=config_dict["model_name"],
            model_temperature=config_dict["model_temperature"],
            max_tokens=config_dict["max_tokens"],
            base_path=Path(config_dict["base_path"]),
            data_path=Path(config_dict["data_path"]),
            cache_path=Path(config_dict["cache_path"])
        )
