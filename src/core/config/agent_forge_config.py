"""
Agent Forge Configuration System

Structured configuration management for the Agent Forge pipeline
with validation and environment variable support.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from .base_config import BaseConfig
from ..exceptions import ConfigurationError


@dataclass
class ModelConfig(BaseConfig):
    """Configuration for model parameters."""
    
    model_path: str = "microsoft/DialoGPT-medium"
    tokenizer_path: Optional[str] = None
    max_length: int = 512
    batch_size: int = 16
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    precision: str = "fp16"  # "fp32", "fp16", "bf16"
    gradient_checkpointing: bool = False
    
    def validate(self) -> None:
        if not self.model_path:
            raise ConfigurationError("model_path cannot be empty")
        
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        if self.device not in valid_devices:
            raise ConfigurationError(f"device must be one of {valid_devices}")
        
        valid_precisions = ["fp32", "fp16", "bf16"]
        if self.precision not in valid_precisions:
            raise ConfigurationError(f"precision must be one of {valid_precisions}")
        
        if self.max_length <= 0:
            raise ConfigurationError("max_length must be positive")
        
        if self.batch_size <= 0:
            raise ConfigurationError("batch_size must be positive")


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for training parameters."""
    
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_steps: int = 500
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    optimizer: str = "adamw"  # "adamw", "adam", "sgd"
    scheduler: str = "linear"  # "linear", "cosine", "constant"
    
    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ConfigurationError("learning_rate must be positive")
        
        if self.weight_decay < 0:
            raise ConfigurationError("weight_decay must be non-negative")
        
        if self.num_epochs <= 0:
            raise ConfigurationError("num_epochs must be positive")
        
        valid_optimizers = ["adamw", "adam", "sgd"]
        if self.optimizer not in valid_optimizers:
            raise ConfigurationError(f"optimizer must be one of {valid_optimizers}")
        
        valid_schedulers = ["linear", "cosine", "constant"]
        if self.scheduler not in valid_schedulers:
            raise ConfigurationError(f"scheduler must be one of {valid_schedulers}")


@dataclass
class PhaseConfig(BaseConfig):
    """Base configuration for pipeline phases."""
    
    enabled: bool = True
    timeout_seconds: Optional[int] = None
    retry_attempts: int = 0
    checkpoint_path: Optional[str] = None
    
    def validate(self) -> None:
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ConfigurationError("timeout_seconds must be positive if specified")
        
        if self.retry_attempts < 0:
            raise ConfigurationError("retry_attempts must be non-negative")


@dataclass
class EvoMergeConfig(PhaseConfig):
    """Configuration for EvoMerge phase."""
    
    base_models: List[str] = field(default_factory=lambda: [
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-medium"
    ])
    population_size: int = 10
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    selection_method: str = "tournament"  # "tournament", "roulette", "rank"
    merge_methods: List[str] = field(default_factory=lambda: ["linear", "slerp"])
    
    def validate(self) -> None:
        super().validate()
        
        if not self.base_models:
            raise ConfigurationError("base_models cannot be empty")
        
        if self.population_size <= 0:
            raise ConfigurationError("population_size must be positive")
        
        if self.generations <= 0:
            raise ConfigurationError("generations must be positive")
        
        if not 0 <= self.mutation_rate <= 1:
            raise ConfigurationError("mutation_rate must be between 0 and 1")
        
        if not 0 <= self.crossover_rate <= 1:
            raise ConfigurationError("crossover_rate must be between 0 and 1")
        
        valid_selection = ["tournament", "roulette", "rank"]
        if self.selection_method not in valid_selection:
            raise ConfigurationError(f"selection_method must be one of {valid_selection}")


@dataclass
class ToolPersonaBakingConfig(PhaseConfig):
    """Configuration for Tool & Persona Baking phase."""
    
    tools_config_path: Optional[str] = None
    persona_templates: List[str] = field(default_factory=list)
    baking_iterations: int = 100
    persona_weight: float = 0.5
    tool_weight: float = 0.5
    temperature: float = 0.7
    top_p: float = 0.9
    
    def validate(self) -> None:
        super().validate()
        
        if self.baking_iterations <= 0:
            raise ConfigurationError("baking_iterations must be positive")
        
        if not 0 <= self.persona_weight <= 1:
            raise ConfigurationError("persona_weight must be between 0 and 1")
        
        if not 0 <= self.tool_weight <= 1:
            raise ConfigurationError("tool_weight must be between 0 and 1")
        
        if not 0 < self.temperature <= 2:
            raise ConfigurationError("temperature must be between 0 and 2")
        
        if not 0 < self.top_p <= 1:
            raise ConfigurationError("top_p must be between 0 and 1")


@dataclass
class CompressionConfig(PhaseConfig):
    """Configuration for compression phase."""
    
    compression_method: str = "quantization"  # "quantization", "pruning", "distillation"
    target_size_mb: Optional[float] = None
    quantization_bits: int = 8
    pruning_ratio: float = 0.5
    distillation_temperature: float = 4.0
    
    def validate(self) -> None:
        super().validate()
        
        valid_methods = ["quantization", "pruning", "distillation"]
        if self.compression_method not in valid_methods:
            raise ConfigurationError(f"compression_method must be one of {valid_methods}")
        
        if self.target_size_mb is not None and self.target_size_mb <= 0:
            raise ConfigurationError("target_size_mb must be positive if specified")
        
        if self.quantization_bits not in [4, 8, 16]:
            raise ConfigurationError("quantization_bits must be 4, 8, or 16")
        
        if not 0 < self.pruning_ratio < 1:
            raise ConfigurationError("pruning_ratio must be between 0 and 1")
        
        if self.distillation_temperature <= 0:
            raise ConfigurationError("distillation_temperature must be positive")


@dataclass
class EvaluationConfig(BaseConfig):
    """Configuration for model evaluation."""
    
    eval_datasets: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=lambda: ["perplexity", "bleu"])
    batch_size: int = 16
    max_eval_samples: Optional[int] = None
    output_dir: str = "eval_results"
    
    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ConfigurationError("batch_size must be positive")
        
        if self.max_eval_samples is not None and self.max_eval_samples <= 0:
            raise ConfigurationError("max_eval_samples must be positive if specified")


@dataclass
class AgentForgeConfig(BaseConfig):
    """Main configuration for the Agent Forge pipeline."""
    
    # Core configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Phase configurations
    evomerge: EvoMergeConfig = field(default_factory=EvoMergeConfig)
    tool_persona_baking: ToolPersonaBakingConfig = field(default_factory=ToolPersonaBakingConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    
    # Pipeline settings
    pipeline_phases: List[str] = field(default_factory=lambda: [
        "evomerge", "tool_persona_baking", "compression"
    ])
    output_dir: str = "output"
    experiment_name: Optional[str] = None
    seed: int = 42
    
    # Resource settings
    max_memory_gb: float = 16.0
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    
    def validate(self) -> None:
        """Validate the entire configuration."""
        # Validate sub-configurations
        self.model.validate()
        self.training.validate()
        self.evaluation.validate()
        self.evomerge.validate()
        self.tool_persona_baking.validate()
        self.compression.validate()
        
        # Validate pipeline settings
        if not self.pipeline_phases:
            raise ConfigurationError("pipeline_phases cannot be empty")
        
        valid_phases = [
            "evomerge", "tool_persona_baking", "compression", 
            "quietstar", "adas", "forge_training"
        ]
        for phase in self.pipeline_phases:
            if phase not in valid_phases:
                raise ConfigurationError(f"Invalid phase '{phase}'. Valid phases: {valid_phases}")
        
        # Validate resource settings
        if self.max_memory_gb <= 0:
            raise ConfigurationError("max_memory_gb must be positive")
        
        if not 0 < self.gpu_memory_fraction <= 1:
            raise ConfigurationError("gpu_memory_fraction must be between 0 and 1")
        
        if self.seed < 0:
            raise ConfigurationError("seed must be non-negative")
    
    def get_phase_config(self, phase_name: str) -> PhaseConfig:
        """Get configuration for a specific phase."""
        phase_configs = {
            "evomerge": self.evomerge,
            "tool_persona_baking": self.tool_persona_baking,
            "compression": self.compression,
        }
        
        if phase_name not in phase_configs:
            raise ConfigurationError(f"No configuration found for phase '{phase_name}'")
        
        return phase_configs[phase_name]
    
    def get_enabled_phases(self) -> List[str]:
        """Get list of enabled phases in order."""
        enabled_phases = []
        
        for phase_name in self.pipeline_phases:
            try:
                phase_config = self.get_phase_config(phase_name)
                if phase_config.enabled:
                    enabled_phases.append(phase_name)
            except ConfigurationError:
                # Skip unknown phases
                continue
        
        return enabled_phases
    
    @classmethod
    def create_default(cls) -> 'AgentForgeConfig':
        """Create a default configuration."""
        return cls()
    
    @classmethod
    def create_minimal(cls) -> 'AgentForgeConfig':
        """Create a minimal configuration for testing."""
        config = cls()
        config.model.batch_size = 4
        config.training.num_epochs = 1
        config.evomerge.generations = 5
        config.tool_persona_baking.baking_iterations = 10
        return config
    
    @classmethod
    def create_production(cls) -> 'AgentForgeConfig':
        """Create a production-ready configuration."""
        config = cls()
        config.model.batch_size = 32
        config.training.num_epochs = 10
        config.evomerge.generations = 100
        config.tool_persona_baking.baking_iterations = 500
        config.max_memory_gb = 32.0
        return config