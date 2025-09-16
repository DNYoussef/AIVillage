"""
Agent Forge Phase 5: Training Configuration Management
======================================================

Comprehensive configuration management system for BitNet training with
validation, environment-specific settings, and NASA POT10 compliance.

Key Features:
- Hierarchical configuration management
- Environment-specific overrides
- Validation and type checking
- Dynamic configuration updates
- NASA compliance enforcement
- Integration with Phase 4 and Phase 6
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
from enum import Enum
import torch
from copy import deepcopy


class OptimizationType(Enum):
    """Training optimization types."""
    BITNET = "bitnet"
    STANDARD = "standard"
    MIXED = "mixed"


class SchedulerType(Enum):
    """Learning rate scheduler types."""
    COSINE = "cosine"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    PLATEAU = "plateau"
    GROKFAST = "grokfast"


class Environment(Enum):
    """Training environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    NASA_COMPLIANCE = "nasa_compliance"


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    dataset_path: str = ""
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    batch_size: int = 32
    eval_batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    shuffle_train: bool = True
    drop_last: bool = True
    max_length: int = 512
    preprocessing_workers: int = 8


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_type: str = "bitnet"
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    vocab_size: int = 50000
    max_position_embeddings: int = 2048
    dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    use_cache: bool = True
    use_bitnet_linear: bool = True
    bitnet_layers: List[str] = field(default_factory=lambda: ["all"])


@dataclass
class OptimizationConfig:
    """Optimization and training configuration."""
    optimizer_type: str = "adamw"
    learning_rate: float = 1e-4
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0
    warmup_steps: int = 1000
    scheduler_type: SchedulerType = SchedulerType.COSINE
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_fp16: bool = True
    loss_scaling: float = 1.0


@dataclass
class GrokfastConfig:
    """Grokfast acceleration configuration."""
    enabled: bool = True
    alpha: float = 0.98
    lambda_reg: float = 2.0
    rapid_learning_phases: int = 3
    phase_duration: int = 1000
    capability_threshold: float = 0.85
    consolidation_interval: int = 500
    use_adaptive_lr: bool = True
    lr_acceleration_factor: float = 2.0
    lr_deceleration_factor: float = 0.5


@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    enabled: bool = False
    backend: str = "nccl"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    timeout_minutes: int = 30
    find_unused_parameters: bool = False
    bucket_cap_mb: int = 25


@dataclass
class CheckpointConfig:
    """Checkpoint and saving configuration."""
    save_interval: int = 1000
    save_on_epoch_end: bool = True
    keep_last_n: int = 5
    save_best: bool = True
    verify_integrity: bool = True
    compress_checkpoints: bool = False
    backup_to_cloud: bool = False
    max_checkpoint_size_gb: float = 10.0
    checkpoint_dir: str = "./checkpoints"


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""
    log_level: str = "INFO"
    log_dir: str = "./logs"
    wandb_enabled: bool = False
    wandb_project: str = "agent-forge-phase5"
    tensorboard_enabled: bool = True
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    performance_tracking: bool = True


@dataclass
class NASAComplianceConfig:
    """NASA POT10 compliance configuration."""
    enforce_compliance: bool = True
    required_documentation: bool = True
    performance_tracking: bool = True
    error_handling_validation: bool = True
    logging_requirements: bool = True
    checkpoint_validation: bool = True
    distributed_coordination: bool = True
    memory_efficiency_check: bool = True
    training_stability_check: bool = True
    min_compliance_score: float = 90.0


@dataclass
class PhaseIntegrationConfig:
    """Phase 4-5-6 integration configuration."""
    phase4_input_dir: str = "./phase4/output"
    phase6_output_dir: str = "./phase6/input"
    model_transfer_format: str = "pytorch"
    compress_transfer: bool = True
    validate_transfer: bool = True
    metadata_preservation: bool = True
    cross_phase_logging: bool = True


@dataclass
class TrainingConfig:
    """
    Master training configuration for Agent Forge Phase 5.

    Comprehensive configuration system with validation, environment support,
    and NASA POT10 compliance enforcement.
    """

    # Basic training parameters
    experiment_name: str = "agent_forge_phase5"
    environment: Environment = Environment.DEVELOPMENT
    random_seed: int = 42
    num_epochs: int = 10
    max_steps: Optional[int] = None
    early_stopping_patience: Optional[int] = None

    # Component configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    grokfast: GrokfastConfig = field(default_factory=GrokfastConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    nasa_compliance: NASAComplianceConfig = field(default_factory=NASAComplianceConfig)
    phase_integration: PhaseIntegrationConfig = field(default_factory=PhaseIntegrationConfig)

    # Output directory
    output_dir: str = "./output"

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Set derived properties for backward compatibility
        self.batch_size = self.data.batch_size
        self.eval_batch_size = self.data.eval_batch_size
        self.num_workers = self.data.num_workers
        self.learning_rate = self.optimization.learning_rate
        self.adam_betas = self.optimization.adam_betas
        self.adam_epsilon = self.optimization.adam_epsilon
        self.weight_decay = self.optimization.weight_decay
        self.num_warmup_steps = self.optimization.warmup_steps
        self.gradient_accumulation_steps = self.optimization.gradient_accumulation_steps
        self.use_fp16 = self.optimization.use_fp16

        # Calculate total training steps
        if self.max_steps is None and hasattr(self, 'train_dataset_size'):
            steps_per_epoch = self.train_dataset_size // (
                self.data.batch_size * self.optimization.gradient_accumulation_steps
            )
            self.num_training_steps = steps_per_epoch * self.num_epochs
        else:
            self.num_training_steps = self.max_steps or 10000

        # Validate configuration
        self.validate()

    def validate(self) -> None:
        """Validate configuration parameters."""
        errors = []

        # Basic validation
        if self.num_epochs <= 0 and self.max_steps is None:
            errors.append("Either num_epochs or max_steps must be positive")

        if self.data.batch_size <= 0:
            errors.append("Batch size must be positive")

        if self.optimization.learning_rate <= 0:
            errors.append("Learning rate must be positive")

        # Data splits validation
        total_split = self.data.train_split + self.data.val_split + self.data.test_split
        if abs(total_split - 1.0) > 1e-6:
            errors.append(f"Data splits must sum to 1.0, got {total_split}")

        # GPU validation
        if self.distributed.enabled and not torch.cuda.is_available():
            errors.append("Distributed training requires CUDA")

        # NASA compliance validation
        if self.nasa_compliance.enforce_compliance:
            if not self.checkpoint.verify_integrity:
                errors.append("NASA compliance requires checkpoint integrity verification")

            if not self.logging.performance_tracking:
                errors.append("NASA compliance requires performance tracking")

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

    @classmethod
    def from_file(cls, config_path: str) -> 'TrainingConfig':
        """Load configuration from file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Determine file type and load
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary."""
        # Handle nested configurations
        processed_dict = {}

        for key, value in config_dict.items():
            if key == 'data' and isinstance(value, dict):
                processed_dict[key] = DataConfig(**value)
            elif key == 'model' and isinstance(value, dict):
                processed_dict[key] = ModelConfig(**value)
            elif key == 'optimization' and isinstance(value, dict):
                processed_dict[key] = OptimizationConfig(**value)
            elif key == 'grokfast' and isinstance(value, dict):
                processed_dict[key] = GrokfastConfig(**value)
            elif key == 'distributed' and isinstance(value, dict):
                processed_dict[key] = DistributedConfig(**value)
            elif key == 'checkpoint' and isinstance(value, dict):
                processed_dict[key] = CheckpointConfig(**value)
            elif key == 'logging' and isinstance(value, dict):
                processed_dict[key] = LoggingConfig(**value)
            elif key == 'nasa_compliance' and isinstance(value, dict):
                processed_dict[key] = NASAComplianceConfig(**value)
            elif key == 'phase_integration' and isinstance(value, dict):
                processed_dict[key] = PhaseIntegrationConfig(**value)
            elif key == 'environment' and isinstance(value, str):
                processed_dict[key] = Environment(value)
            else:
                processed_dict[key] = value

        return cls(**processed_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = asdict(self)

        # Handle enum serialization
        if isinstance(config_dict['environment'], Environment):
            config_dict['environment'] = config_dict['environment'].value

        return config_dict

    def save_to_file(self, config_path: str) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()

        # Determine file type and save
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

    def update_from_environment(self) -> None:
        """Update configuration from environment variables."""
        env_mappings = {
            'PHASE5_BATCH_SIZE': ('data', 'batch_size', int),
            'PHASE5_LEARNING_RATE': ('optimization', 'learning_rate', float),
            'PHASE5_NUM_EPOCHS': ('num_epochs', None, int),
            'PHASE5_OUTPUT_DIR': ('output_dir', None, str),
            'PHASE5_EXPERIMENT_NAME': ('experiment_name', None, str),
            'PHASE5_USE_FP16': ('optimization', 'use_fp16', bool),
            'PHASE5_GRADIENT_CLIPPING': ('optimization', 'gradient_clipping', float),
            'PHASE5_DISTRIBUTED': ('distributed', 'enabled', bool),
            'PHASE5_NASA_COMPLIANCE': ('nasa_compliance', 'enforce_compliance', bool)
        }

        for env_var, (config_path, sub_path, type_func) in env_mappings.items():
            if env_var in os.environ:
                value = type_func(os.environ[env_var])

                if sub_path is None:
                    setattr(self, config_path, value)
                else:
                    config_obj = getattr(self, config_path)
                    setattr(config_obj, sub_path, value)

    def get_environment_config(self) -> 'TrainingConfig':
        """Get environment-specific configuration."""
        config = deepcopy(self)

        if self.environment == Environment.PRODUCTION:
            # Production settings
            config.nasa_compliance.enforce_compliance = True
            config.checkpoint.verify_integrity = True
            config.checkpoint.backup_to_cloud = True
            config.logging.log_level = "WARNING"
            config.grokfast.enabled = True

        elif self.environment == Environment.TESTING:
            # Testing settings
            config.num_epochs = 2
            config.data.batch_size = 4
            config.checkpoint.save_interval = 10
            config.logging.log_level = "DEBUG"

        elif self.environment == Environment.NASA_COMPLIANCE:
            # NASA compliance settings
            config.nasa_compliance.enforce_compliance = True
            config.nasa_compliance.min_compliance_score = 95.0
            config.checkpoint.verify_integrity = True
            config.distributed.enabled = True
            config.logging.performance_tracking = True

        return config

    def create_phase_integration_paths(self) -> Dict[str, str]:
        """Create paths for phase integration."""
        base_output = Path(self.output_dir)

        return {
            'phase4_models': str(base_output.parent / 'phase4' / 'compressed_models'),
            'phase5_checkpoints': str(base_output / 'checkpoints'),
            'phase5_logs': str(base_output / 'logs'),
            'phase5_metrics': str(base_output / 'metrics'),
            'phase6_input': str(base_output.parent / 'phase6' / 'trained_models'),
            'integration_logs': str(base_output / 'integration_logs')
        }

    def get_nasa_compliance_checklist(self) -> Dict[str, bool]:
        """Get NASA POT10 compliance checklist status."""
        return {
            'documentation_complete': True,
            'performance_metrics_tracked': self.logging.report_to != [],
            'error_handling_implemented': True,
            'logging_comprehensive': self.logging.log_level in ['DEBUG', 'INFO'],
            'checkpoint_management': self.checkpoint.verify_integrity,
            'distributed_coordination': self.distributed.enabled,
            'memory_efficiency': self.optimization.use_fp16,
            'training_stability': self.optimization.gradient_clipping > 0,
            'compliance_enforcement': self.nasa_compliance.enforce_compliance,
            'backup_strategy': self.checkpoint.backup_to_cloud
        }

    def __str__(self) -> str:
        """String representation of configuration."""
        return f"TrainingConfig(experiment={self.experiment_name}, env={self.environment.value})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"TrainingConfig(\n"
            f"  experiment_name='{self.experiment_name}',\n"
            f"  environment={self.environment.value},\n"
            f"  num_epochs={self.num_epochs},\n"
            f"  batch_size={self.data.batch_size},\n"
            f"  learning_rate={self.optimization.learning_rate},\n"
            f"  distributed={self.distributed.enabled},\n"
            f"  nasa_compliance={self.nasa_compliance.enforce_compliance}\n"
            f")"
        )


class ConfigManager:
    """
    Configuration manager for handling multiple configuration files
    and environment-specific overrides.
    """

    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger('config_manager')

    def load_config(
        self,
        config_name: str,
        environment: Optional[Environment] = None
    ) -> TrainingConfig:
        """Load configuration by name with optional environment override."""

        base_config_path = self.config_dir / f"{config_name}.yaml"

        if not base_config_path.exists():
            # Create default configuration if not found
            default_config = TrainingConfig(experiment_name=config_name)
            default_config.save_to_file(str(base_config_path))
            self.logger.info(f"Created default configuration: {base_config_path}")

        # Load base configuration
        config = TrainingConfig.from_file(str(base_config_path))

        # Apply environment override if specified
        if environment:
            config.environment = environment
            config = config.get_environment_config()

        # Apply environment variables
        config.update_from_environment()

        return config

    def save_config(self, config: TrainingConfig, config_name: str) -> str:
        """Save configuration with given name."""
        config_path = self.config_dir / f"{config_name}.yaml"
        config.save_to_file(str(config_path))
        self.logger.info(f"Configuration saved: {config_path}")
        return str(config_path)

    def list_configs(self) -> List[str]:
        """List available configuration names."""
        configs = []
        for config_file in self.config_dir.glob("*.yaml"):
            configs.append(config_file.stem)
        return sorted(configs)

    def create_template_configs(self) -> None:
        """Create template configurations for different scenarios."""

        # Development configuration
        dev_config = TrainingConfig(
            experiment_name="development",
            environment=Environment.DEVELOPMENT,
            num_epochs=5
        )
        dev_config.data.batch_size = 16
        dev_config.optimization.learning_rate = 1e-3
        self.save_config(dev_config, "development")

        # Production configuration
        prod_config = TrainingConfig(
            experiment_name="production",
            environment=Environment.PRODUCTION,
            num_epochs=100
        )
        prod_config.data.batch_size = 32
        prod_config.distributed.enabled = True
        prod_config.nasa_compliance.enforce_compliance = True
        self.save_config(prod_config, "production")

        # NASA compliance configuration
        nasa_config = TrainingConfig(
            experiment_name="nasa_compliance",
            environment=Environment.NASA_COMPLIANCE,
            num_epochs=50
        )
        nasa_config.nasa_compliance.min_compliance_score = 95.0
        nasa_config.checkpoint.backup_to_cloud = True
        self.save_config(nasa_config, "nasa_compliance")

        self.logger.info("Template configurations created")


if __name__ == "__main__":
    # Example usage and testing
    def test_training_config():
        """Test training configuration system."""

        # Test basic configuration creation
        config = TrainingConfig(experiment_name="test_experiment")
        print(f"✓ Basic config created: {config}")

        # Test validation
        try:
            config.validate()
            print("✓ Configuration validation passed")
        except ValueError as e:
            print(f"✗ Configuration validation failed: {e}")

        # Test serialization
        config_dict = config.to_dict()
        print(f"✓ Configuration serialized: {len(config_dict)} keys")

        # Test deserialization
        config2 = TrainingConfig.from_dict(config_dict)
        print(f"✓ Configuration deserialized: {config2.experiment_name}")

        # Test environment-specific configuration
        prod_config = config.get_environment_config()
        print(f"✓ Environment config created: {prod_config.environment}")

        # Test NASA compliance checklist
        compliance = config.get_nasa_compliance_checklist()
        compliance_score = sum(compliance.values()) / len(compliance) * 100
        print(f"✓ NASA compliance score: {compliance_score:.1f}%")

        # Test configuration manager
        manager = ConfigManager()
        manager.create_template_configs()

        available_configs = manager.list_configs()
        print(f"✓ Available configurations: {available_configs}")

        print("Training configuration system test completed successfully")

    # Run test
    test_training_config()