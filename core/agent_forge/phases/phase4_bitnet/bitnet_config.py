"""
BitNet Configuration and Hyperparameters for Agent Forge Phase 4

Configuration Management System
===============================

This module provides comprehensive configuration management for BitNet models,
including hyperparameter optimization, NASA POT10 compliance settings,
and integration parameters for Agent Forge phases.

Features:
1. Hierarchical configuration management
2. Environment-specific configurations
3. Hyperparameter validation and optimization
4. NASA POT10 compliance presets
5. Phase integration configurations
6. Performance tuning profiles

Author: Agent Forge Phase 4 - Configuration Specialist
License: NASA POT10 Compliant
"""

import json
import logging
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import warnings
from enum import Enum
import torch

# Configure logging for NASA POT10 compliance
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelSize(Enum):
    """Predefined model size configurations."""
    TINY = "tiny"
    SMALL = "small"
    BASE = "base"
    LARGE = "large"
    XLARGE = "xlarge"

class OptimizationProfile(Enum):
    """Optimization profiles for different use cases."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    INFERENCE = "inference"
    TRAINING = "training"
    EDGE_DEPLOYMENT = "edge_deployment"

class ComplianceLevel(Enum):
    """NASA POT10 compliance levels."""
    STANDARD = "standard"
    ENHANCED = "enhanced"
    DEFENSE_GRADE = "defense_grade"

@dataclass
class ModelArchitectureConfig:
    """Model architecture configuration."""
    # Core architecture parameters
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    vocab_size: int = 50257
    
    # BitNet specific parameters
    use_1bit_quantization: bool = True
    activation_quantization_bits: int = 8
    weight_quantization_bits: int = 1
    quantization_method: str = "sign_function"
    
    # Attention configuration
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    use_bias: bool = False  # BitNet typically doesn't use bias
    
    # Advanced features
    use_flash_attention: bool = True
    use_rotary_embeddings: bool = False
    use_alibi_bias: bool = False
    
    def validate(self) -> List[str]:
        """Validate architecture configuration."""
        issues = []
        
        # Check divisibility constraints
        if self.hidden_size % self.num_attention_heads != 0:
            issues.append(f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})")
        
        # Check size constraints
        if self.hidden_size < 64:
            issues.append("hidden_size should be at least 64 for stable training")
        
        if self.num_attention_heads < 1:
            issues.append("num_attention_heads must be at least 1")
        
        # Check quantization parameters
        if self.weight_quantization_bits not in [1, 2, 4, 8]:
            issues.append(f"weight_quantization_bits ({self.weight_quantization_bits}) should be 1, 2, 4, or 8")
        
        if self.activation_quantization_bits not in [4, 8, 16]:
            issues.append(f"activation_quantization_bits ({self.activation_quantization_bits}) should be 4, 8, or 16")
        
        return issues

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    # Basic training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_steps: int = 100000
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    
    # Learning rate scheduling
    lr_scheduler_type: str = "cosine_with_warmup"
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Optimization parameters
    optimizer_type: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    gradient_clipping: float = 1.0
    
    # BitNet specific training
    quantization_aware_training: bool = True
    straight_through_estimator: bool = True
    temperature_annealing: bool = True
    initial_temperature: float = 1.0
    final_temperature: float = 0.1
    
    # Regularization
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    label_smoothing: float = 0.1
    
    # Memory optimization
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    dataloader_num_workers: int = 4
    
    # Validation and checkpointing
    eval_steps: int = 1000
    save_steps: int = 5000
    logging_steps: int = 100
    max_checkpoints_to_keep: int = 5
    
    def validate(self) -> List[str]:
        """Validate training configuration."""
        issues = []
        
        # Learning rate checks
        if self.learning_rate <= 0:
            issues.append("learning_rate must be positive")
        
        if self.learning_rate > 1e-2:
            issues.append("learning_rate seems too high, consider values < 1e-2")
        
        # Batch size checks
        if self.batch_size < 1:
            issues.append("batch_size must be at least 1")
        
        # Steps validation
        if self.warmup_steps >= self.max_steps:
            issues.append("warmup_steps should be less than max_steps")
        
        # Dropout validation
        if not (0 <= self.dropout_rate <= 1):
            issues.append("dropout_rate must be between 0 and 1")
        
        return issues

@dataclass
class InferenceConfig:
    """Inference-specific configuration."""
    # Generation parameters
    max_length: int = 512
    min_length: int = 1
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    
    # Batch processing
    batch_size: int = 1
    max_batch_size: int = 32
    dynamic_batching: bool = True
    padding_side: str = "left"
    
    # Performance optimization
    use_kv_cache: bool = True
    use_torch_compile: bool = True
    enable_cpu_offload: bool = False
    memory_fraction: float = 0.8
    
    # Precision settings
    inference_precision: str = "float16"  # float16, bfloat16, int8
    quantization_enabled: bool = True
    
    def validate(self) -> List[str]:
        """Validate inference configuration."""
        issues = []
        
        # Length validation
        if self.min_length > self.max_length:
            issues.append("min_length cannot be greater than max_length")
        
        # Temperature validation
        if self.temperature <= 0:
            issues.append("temperature must be positive")
        
        # Top-k validation
        if self.top_k < 0:
            issues.append("top_k must be non-negative")
        
        # Top-p validation
        if not (0 < self.top_p <= 1):
            issues.append("top_p must be between 0 and 1")
        
        # Batch size validation
        if self.batch_size > self.max_batch_size:
            issues.append("batch_size cannot exceed max_batch_size")
        
        return issues

@dataclass
class PhaseIntegrationConfig:
    """Configuration for Agent Forge phase integration."""
    # Phase 2 integration (EvoMerge)
    evomerge_integration: bool = True
    evomerge_model_path: str = "phase2_outputs/evomerge_model.pt"
    preserve_evomerge_optimizations: bool = True
    
    # Phase 3 integration (Quiet-STaR)
    quiet_star_integration: bool = True
    quiet_star_model_path: str = "phase3_outputs/quiet_star_model.pt"
    thought_vector_dimensions: int = 768
    enable_reasoning_enhancement: bool = True
    
    # Phase 5 preparation (Training Pipeline)
    phase5_pipeline_ready: bool = True
    export_format: str = "pytorch"  # pytorch, onnx, tensorrt
    deployment_targets: List[str] = field(default_factory=lambda: ["cuda", "cpu"])
    
    # Cross-phase validation
    validate_phase_compatibility: bool = True
    ensure_state_continuity: bool = True
    maintain_performance_metrics: bool = True
    
    def validate(self) -> List[str]:
        """Validate phase integration configuration."""
        issues = []
        
        # Path validation
        if self.evomerge_integration and not self.evomerge_model_path:
            issues.append("evomerge_model_path must be specified when evomerge_integration is enabled")
        
        if self.quiet_star_integration and not self.quiet_star_model_path:
            issues.append("quiet_star_model_path must be specified when quiet_star_integration is enabled")
        
        # Dimension validation
        if self.thought_vector_dimensions <= 0:
            issues.append("thought_vector_dimensions must be positive")
        
        # Export format validation
        valid_formats = ["pytorch", "onnx", "tensorrt", "tflite"]
        if self.export_format not in valid_formats:
            issues.append(f"export_format must be one of {valid_formats}")
        
        return issues

@dataclass
class NASAComplianceConfig:
    """NASA POT10 compliance configuration."""
    # Compliance level
    compliance_level: ComplianceLevel = ComplianceLevel.ENHANCED
    
    # Audit and traceability
    enable_audit_trail: bool = True
    audit_log_path: str = "audit_logs/bitnet_audit.log"
    model_provenance_tracking: bool = True
    
    # Security requirements
    security_validation: bool = True
    input_sanitization: bool = True
    output_validation: bool = True
    secure_model_storage: bool = True
    
    # Performance monitoring
    performance_monitoring: bool = True
    resource_usage_tracking: bool = True
    error_monitoring: bool = True
    
    # Code quality
    enforce_code_standards: bool = True
    require_documentation: bool = True
    require_unit_tests: bool = True
    
    # Deployment validation
    pre_deployment_validation: bool = True
    runtime_verification: bool = True
    failsafe_mechanisms: bool = True
    
    def get_compliance_requirements(self) -> Dict[str, Any]:
        """Get requirements based on compliance level."""
        requirements = {
            ComplianceLevel.STANDARD: {
                "min_test_coverage": 80,
                "max_complexity_score": 10,
                "require_peer_review": True,
                "security_scan_required": False
            },
            ComplianceLevel.ENHANCED: {
                "min_test_coverage": 90,
                "max_complexity_score": 8,
                "require_peer_review": True,
                "security_scan_required": True,
                "performance_benchmarks": True
            },
            ComplianceLevel.DEFENSE_GRADE: {
                "min_test_coverage": 95,
                "max_complexity_score": 6,
                "require_peer_review": True,
                "security_scan_required": True,
                "performance_benchmarks": True,
                "formal_verification": True,
                "supply_chain_validation": True
            }
        }
        
        return requirements.get(self.compliance_level, requirements[ComplianceLevel.STANDARD])

@dataclass
class BitNetConfig:
    """Comprehensive BitNet configuration."""
    # Component configurations
    architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    phase_integration: PhaseIntegrationConfig = field(default_factory=PhaseIntegrationConfig)
    nasa_compliance: NASAComplianceConfig = field(default_factory=NASAComplianceConfig)
    
    # Global settings
    model_size: ModelSize = ModelSize.BASE
    optimization_profile: OptimizationProfile = OptimizationProfile.PRODUCTION
    random_seed: int = 42
    device: str = "auto"  # auto, cuda, cpu, mps
    
    # Version and metadata
    config_version: str = "1.0.0"
    created_timestamp: Optional[str] = None
    last_modified: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        import time
        current_time = str(int(time.time()))
        
        if self.created_timestamp is None:
            self.created_timestamp = current_time
        self.last_modified = current_time
        
        # Apply model size presets
        self._apply_model_size_preset()
        
        # Apply optimization profile
        self._apply_optimization_profile()
    
    def _apply_model_size_preset(self):
        """Apply predefined model size configurations."""
        size_configs = {
            ModelSize.TINY: {
                "hidden_size": 256,
                "num_attention_heads": 4,
                "num_hidden_layers": 6,
                "intermediate_size": 1024,
            },
            ModelSize.SMALL: {
                "hidden_size": 512,
                "num_attention_heads": 8,
                "num_hidden_layers": 8,
                "intermediate_size": 2048,
            },
            ModelSize.BASE: {
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
                "intermediate_size": 3072,
            },
            ModelSize.LARGE: {
                "hidden_size": 1024,
                "num_attention_heads": 16,
                "num_hidden_layers": 16,
                "intermediate_size": 4096,
            },
            ModelSize.XLARGE: {
                "hidden_size": 1536,
                "num_attention_heads": 24,
                "num_hidden_layers": 24,
                "intermediate_size": 6144,
            }
        }
        
        if self.model_size in size_configs:
            config = size_configs[self.model_size]
            for key, value in config.items():
                setattr(self.architecture, key, value)
    
    def _apply_optimization_profile(self):
        """Apply optimization profile settings."""
        profile_configs = {
            OptimizationProfile.DEVELOPMENT: {
                "training": {
                    "gradient_checkpointing": False,
                    "mixed_precision": False,
                    "batch_size": 8
                },
                "inference": {
                    "use_torch_compile": False,
                    "batch_size": 1
                },
                "nasa_compliance": {
                    "compliance_level": ComplianceLevel.STANDARD
                }
            },
            OptimizationProfile.PRODUCTION: {
                "training": {
                    "gradient_checkpointing": True,
                    "mixed_precision": True,
                    "batch_size": 32
                },
                "inference": {
                    "use_torch_compile": True,
                    "batch_size": 16
                },
                "nasa_compliance": {
                    "compliance_level": ComplianceLevel.ENHANCED
                }
            },
            OptimizationProfile.INFERENCE: {
                "inference": {
                    "use_kv_cache": True,
                    "use_torch_compile": True,
                    "dynamic_batching": True,
                    "batch_size": 1
                }
            },
            OptimizationProfile.TRAINING: {
                "training": {
                    "gradient_checkpointing": True,
                    "mixed_precision": True,
                    "batch_size": 64,
                    "gradient_clipping": 1.0
                }
            },
            OptimizationProfile.EDGE_DEPLOYMENT: {
                "architecture": {
                    "use_flash_attention": False
                },
                "inference": {
                    "inference_precision": "int8",
                    "quantization_enabled": True,
                    "batch_size": 1
                }
            }
        }
        
        if self.optimization_profile in profile_configs:
            profile = profile_configs[self.optimization_profile]
            
            for component_name, component_config in profile.items():
                component = getattr(self, component_name)
                for key, value in component_config.items():
                    setattr(component, key, value)
    
    def validate(self) -> Dict[str, List[str]]:
        """Validate all configuration components."""
        validation_results = {
            "architecture": self.architecture.validate(),
            "training": self.training.validate(),
            "inference": self.inference.validate(),
            "phase_integration": self.phase_integration.validate()
        }
        
        # Cross-component validation
        cross_validation_issues = []
        
        # Architecture-training consistency
        if self.architecture.hidden_dropout != self.training.dropout_rate:
            cross_validation_issues.append("architecture.hidden_dropout should match training.dropout_rate")
        
        # Training-inference consistency
        if self.training.batch_size > self.inference.max_batch_size:
            cross_validation_issues.append("training.batch_size should not exceed inference.max_batch_size")
        
        if cross_validation_issues:
            validation_results["cross_component"] = cross_validation_issues
        
        return validation_results
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BitNetConfig':
        """Load configuration from file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct nested dataclasses
        if 'architecture' in config_dict:
            config_dict['architecture'] = ModelArchitectureConfig(**config_dict['architecture'])
        if 'training' in config_dict:
            config_dict['training'] = TrainingConfig(**config_dict['training'])
        if 'inference' in config_dict:
            config_dict['inference'] = InferenceConfig(**config_dict['inference'])
        if 'phase_integration' in config_dict:
            config_dict['phase_integration'] = PhaseIntegrationConfig(**config_dict['phase_integration'])
        if 'nasa_compliance' in config_dict:
            config_dict['nasa_compliance'] = NASAComplianceConfig(**config_dict['nasa_compliance'])
        
        # Handle enums
        if 'model_size' in config_dict and isinstance(config_dict['model_size'], str):
            config_dict['model_size'] = ModelSize(config_dict['model_size'])
        if 'optimization_profile' in config_dict and isinstance(config_dict['optimization_profile'], str):
            config_dict['optimization_profile'] = OptimizationProfile(config_dict['optimization_profile'])
        
        logger.info(f"Configuration loaded from {path}")
        return cls(**config_dict)
    
    def get_model_parameters_count(self) -> int:
        """Estimate number of model parameters."""
        arch = self.architecture
        
        # Embedding parameters
        token_embeddings = arch.vocab_size * arch.hidden_size
        position_embeddings = arch.max_position_embeddings * arch.hidden_size
        
        # Transformer block parameters
        attention_params = 4 * arch.hidden_size * arch.hidden_size  # Q, K, V, O projections
        mlp_params = arch.hidden_size * arch.intermediate_size + arch.intermediate_size * arch.hidden_size
        layer_norm_params = 2 * arch.hidden_size  # Two layer norms per block
        
        block_params = attention_params + mlp_params + layer_norm_params
        total_block_params = block_params * arch.num_hidden_layers
        
        # Output layer
        output_params = arch.hidden_size * arch.vocab_size
        
        # Final layer norm
        final_layer_norm_params = arch.hidden_size
        
        total_params = (token_embeddings + position_embeddings + 
                       total_block_params + output_params + final_layer_norm_params)
        
        return total_params
    
    def get_memory_estimate(self) -> Dict[str, float]:
        """Estimate memory requirements."""
        total_params = self.get_model_parameters_count()
        
        # Model weights (1-bit for quantized layers, full precision for others)
        quantized_ratio = 0.8  # Assume 80% of parameters are quantized
        quantized_params = int(total_params * quantized_ratio)
        full_precision_params = total_params - quantized_params
        
        # Memory calculations (in MB)
        quantized_memory_mb = quantized_params * 0.125 / 1024**2  # 1 bit per parameter
        full_precision_memory_mb = full_precision_params * 4 / 1024**2  # 4 bytes per float32
        model_memory_mb = quantized_memory_mb + full_precision_memory_mb
        
        # Training memory (gradients, optimizer states, activations)
        gradient_memory_mb = total_params * 4 / 1024**2  # Full precision gradients
        optimizer_memory_mb = total_params * 8 / 1024**2  # AdamW states (2x parameters)
        
        # Activation memory (rough estimate)
        batch_size = self.training.batch_size
        seq_len = self.architecture.max_position_embeddings
        activation_memory_mb = (batch_size * seq_len * self.architecture.hidden_size * 
                               self.architecture.num_hidden_layers * 4) / 1024**2
        
        return {
            "model_memory_mb": model_memory_mb,
            "gradient_memory_mb": gradient_memory_mb,
            "optimizer_memory_mb": optimizer_memory_mb,
            "activation_memory_mb": activation_memory_mb,
            "total_training_memory_mb": model_memory_mb + gradient_memory_mb + optimizer_memory_mb + activation_memory_mb,
            "inference_memory_mb": model_memory_mb + activation_memory_mb * 0.1  # Much smaller activation memory for inference
        }

def create_development_config() -> BitNetConfig:
    """Create a development-optimized configuration."""
    return BitNetConfig(
        model_size=ModelSize.SMALL,
        optimization_profile=OptimizationProfile.DEVELOPMENT
    )

def create_production_config() -> BitNetConfig:
    """Create a production-ready configuration."""
    return BitNetConfig(
        model_size=ModelSize.BASE,
        optimization_profile=OptimizationProfile.PRODUCTION
    )

def create_inference_config() -> BitNetConfig:
    """Create an inference-optimized configuration."""
    return BitNetConfig(
        model_size=ModelSize.BASE,
        optimization_profile=OptimizationProfile.INFERENCE
    )

def create_defense_grade_config() -> BitNetConfig:
    """Create a defense-grade configuration with maximum compliance."""
    config = BitNetConfig(
        model_size=ModelSize.BASE,
        optimization_profile=OptimizationProfile.PRODUCTION
    )
    
    # Override NASA compliance for defense grade
    config.nasa_compliance.compliance_level = ComplianceLevel.DEFENSE_GRADE
    config.nasa_compliance.security_validation = True
    config.nasa_compliance.formal_verification = True
    
    return config

def main():
    """
    Demonstration of BitNet configuration system.
    """
    print("BitNet Configuration System - Agent Forge Phase 4")
    print("=" * 55)
    
    # Create different configuration profiles
    configs = {
        "development": create_development_config(),
        "production": create_production_config(),
        "inference": create_inference_config(),
        "defense_grade": create_defense_grade_config()
    }
    
    for config_name, config in configs.items():
        print(f"\n{config_name.title()} Configuration:")
        print(f"  Model Size: {config.model_size.value}")
        print(f"  Optimization Profile: {config.optimization_profile.value}")
        print(f"  Hidden Size: {config.architecture.hidden_size}")
        print(f"  Number of Layers: {config.architecture.num_hidden_layers}")
        print(f"  Parameters: {config.get_model_parameters_count() / 1e6:.1f}M")
        
        # Memory estimates
        memory = config.get_memory_estimate()
        print(f"  Model Memory: {memory['model_memory_mb']:.1f} MB")
        print(f"  Training Memory: {memory['total_training_memory_mb']:.1f} MB")
        print(f"  Inference Memory: {memory['inference_memory_mb']:.1f} MB")
        
        # Validation
        validation_results = config.validate()
        total_issues = sum(len(issues) for issues in validation_results.values())
        print(f"  Validation Issues: {total_issues}")
    
    # Demonstrate saving and loading
    print(f"\nConfiguration Management:")
    test_config = configs["production"]
    config_path = "test_bitnet_config.json"
    
    test_config.save(config_path)
    print(f"  Saved configuration to: {config_path}")
    
    loaded_config = BitNetConfig.load(config_path)
    print(f"  Loaded configuration successfully")
    print(f"  Config version: {loaded_config.config_version}")
    
    # Cleanup
    if os.path.exists(config_path):
        os.remove(config_path)
        print(f"  Cleaned up test file")
    
    print(f"\nBitNet configuration system demonstrated successfully!")

if __name__ == "__main__":
    main()