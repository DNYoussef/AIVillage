#!/usr/bin/env python3
"""
Agent Forge Phase 6: Baking Configuration
=========================================

Comprehensive configuration management system for the model baking pipeline,
providing centralized configuration, validation, and environment-specific
settings for optimal model optimization.
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
import logging
from enum import Enum

class OptimizationLevel(Enum):
    """Optimization levels for model baking"""
    CONSERVATIVE = 0
    BASIC = 1
    STANDARD = 2
    AGGRESSIVE = 3
    MAXIMUM = 4

class DeviceType(Enum):
    """Supported device types"""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

class ExportFormat(Enum):
    """Supported export formats"""
    PYTORCH = "pytorch"
    TORCHSCRIPT = "torchscript"
    ONNX = "onnx"
    TENSORRT = "tensorrt"

@dataclass
class OptimizationPassConfig:
    """Configuration for individual optimization passes"""
    enabled: bool = True
    priority: int = 1
    parameters: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HardwareConfig:
    """Hardware-specific configuration"""
    target_device: DeviceType = DeviceType.AUTO
    cuda_memory_fraction: float = 0.9
    cpu_threads: Optional[int] = None
    enable_mixed_precision: bool = True
    enable_tensorrt: bool = True
    enable_onednn: bool = True
    batch_optimization: bool = True

@dataclass
class QualityConfig:
    """Quality assurance configuration"""
    accuracy_threshold: float = 0.95
    latency_threshold_ms: float = 100.0
    memory_threshold_mb: float = 4096.0
    enable_theater_detection: bool = True
    enable_nasa_compliance: bool = True
    required_nasa_score: float = 0.90
    validation_samples: int = 1000

@dataclass
class ProfilingConfig:
    """Performance profiling configuration"""
    enable_profiling: bool = True
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    enable_detailed_profiling: bool = True
    enable_memory_profiling: bool = True
    profile_batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])

@dataclass
class ExportConfig:
    """Model export configuration"""
    export_formats: List[ExportFormat] = field(default_factory=lambda: [ExportFormat.PYTORCH, ExportFormat.TORCHSCRIPT])
    output_directory: str = "./baked_models"
    include_metadata: bool = True
    include_benchmarks: bool = True
    compress_models: bool = False

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    enable_tensorboard: bool = False
    tensorboard_log_dir: str = "./logs/tensorboard"

class BakingConfiguration:
    """
    Comprehensive configuration management for the model baking pipeline.

    Provides centralized configuration with validation, environment-specific
    settings, and configuration persistence.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self.logger = self._setup_logger()

        # Core configuration sections
        self.optimization_level = OptimizationLevel.STANDARD
        self.hardware = HardwareConfig()
        self.quality = QualityConfig()
        self.profiling = ProfilingConfig()
        self.export = ExportConfig()
        self.logging = LoggingConfig()

        # Optimization passes configuration
        self.optimization_passes = self._initialize_default_passes()

        # Environment-specific overrides
        self.environment_overrides: Dict[str, Any] = {}

        # Load configuration if path provided
        if self.config_path and self.config_path.exists():
            self.load_configuration(self.config_path)

        self.logger.info("BakingConfiguration initialized")

    def _setup_logger(self) -> logging.Logger:
        """Setup configuration logger"""
        logger = logging.getLogger("BakingConfiguration")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _initialize_default_passes(self) -> Dict[str, OptimizationPassConfig]:
        """Initialize default optimization passes configuration"""
        return {
            # Pruning passes
            "magnitude_pruning": OptimizationPassConfig(
                enabled=True,
                priority=1,
                parameters={
                    "sparsity": 0.1,
                    "structured": False,
                    "global_pruning": False
                },
                conditions={
                    "min_optimization_level": OptimizationLevel.BASIC.value
                }
            ),

            "structured_pruning": OptimizationPassConfig(
                enabled=True,
                priority=2,
                parameters={
                    "channel_sparsity": 0.05,
                    "layer_sparsity": 0.1
                },
                conditions={
                    "min_optimization_level": OptimizationLevel.STANDARD.value
                }
            ),

            # Quantization passes
            "bitnet_quantization": OptimizationPassConfig(
                enabled=False,  # Enabled only for BitNet models
                priority=3,
                parameters={
                    "bits": 1,
                    "activation_quantization": True,
                    "weight_clipping": True
                },
                conditions={
                    "model_type": "bitnet"
                }
            ),

            "int8_quantization": OptimizationPassConfig(
                enabled=True,
                priority=3,
                parameters={
                    "calibration_samples": 100,
                    "quantization_scheme": "symmetric"
                },
                conditions={
                    "min_optimization_level": OptimizationLevel.STANDARD.value
                }
            ),

            # Graph optimization passes
            "operator_fusion": OptimizationPassConfig(
                enabled=True,
                priority=4,
                parameters={
                    "conv_bn_fusion": True,
                    "linear_activation_fusion": True,
                    "attention_fusion": True
                }
            ),

            "constant_folding": OptimizationPassConfig(
                enabled=True,
                priority=5,
                parameters={
                    "fold_batch_norms": True,
                    "eliminate_dead_code": True
                }
            ),

            # Advanced optimization passes
            "knowledge_distillation": OptimizationPassConfig(
                enabled=False,  # Enabled only when teacher model available
                priority=6,
                parameters={
                    "temperature": 4.0,
                    "alpha": 0.7,
                    "epochs": 5
                },
                conditions={
                    "min_optimization_level": OptimizationLevel.AGGRESSIVE.value,
                    "requires_teacher_model": True
                }
            ),

            "neural_architecture_search": OptimizationPassConfig(
                enabled=False,
                priority=7,
                parameters={
                    "search_space": "micro",
                    "search_iterations": 10
                },
                conditions={
                    "min_optimization_level": OptimizationLevel.MAXIMUM.value
                }
            )
        }

    def get_enabled_passes(self, model_info: Optional[Dict[str, Any]] = None) -> Dict[str, OptimizationPassConfig]:
        """Get optimization passes that should be enabled based on current configuration"""
        enabled_passes = {}
        model_info = model_info or {}

        for pass_name, pass_config in self.optimization_passes.items():
            if not pass_config.enabled:
                continue

            # Check optimization level condition
            min_level = pass_config.conditions.get("min_optimization_level", 0)
            if self.optimization_level.value < min_level:
                continue

            # Check model type condition
            required_model_type = pass_config.conditions.get("model_type")
            if required_model_type and model_info.get("type") != required_model_type:
                continue

            # Check teacher model requirement
            if (pass_config.conditions.get("requires_teacher_model", False) and
                not model_info.get("has_teacher_model", False)):
                continue

            enabled_passes[pass_name] = pass_config

        return enabled_passes

    def update_for_hardware(self, device_capabilities: Dict[str, Any]):
        """Update configuration based on detected hardware capabilities"""
        self.logger.info("Updating configuration for hardware capabilities")

        # Update hardware configuration
        if device_capabilities.get("device_type") == "cuda":
            self.hardware.enable_mixed_precision = device_capabilities.get("supports_mixed_precision", False)
            self.hardware.enable_tensorrt = device_capabilities.get("supports_tensorrt", False)

            # Adjust memory settings based on available memory
            available_memory = device_capabilities.get("memory_available_mb", 0)
            if available_memory > 0:
                self.hardware.cuda_memory_fraction = min(0.9, (available_memory - 1000) / available_memory)

        elif device_capabilities.get("device_type") == "cpu":
            self.hardware.enable_onednn = device_capabilities.get("supports_onednn", False)
            self.hardware.cpu_threads = device_capabilities.get("core_count", os.cpu_count())

        # Adjust optimization passes based on hardware
        if device_capabilities.get("supports_tensorrt", False):
            # Enable TensorRT-specific optimizations
            self.optimization_passes["operator_fusion"].parameters["tensorrt_optimization"] = True

        if device_capabilities.get("memory_limited", False):
            # Enable more aggressive memory optimizations
            self.optimization_passes["magnitude_pruning"].parameters["sparsity"] = 0.2
            self.quality.memory_threshold_mb = min(2048, self.quality.memory_threshold_mb)

    def update_for_model(self, model_info: Dict[str, Any]):
        """Update configuration based on model characteristics"""
        self.logger.info(f"Updating configuration for model: {model_info.get('name', 'unknown')}")

        model_type = model_info.get("type", "unknown")
        model_size = model_info.get("parameter_count", 0)

        # BitNet-specific configuration
        if model_type == "bitnet":
            self.optimization_passes["bitnet_quantization"].enabled = True
            self.optimization_passes["int8_quantization"].enabled = False

        # Large model optimizations
        if model_size > 100_000_000:  # 100M parameters
            self.optimization_level = OptimizationLevel.AGGRESSIVE
            self.optimization_passes["structured_pruning"].parameters["channel_sparsity"] = 0.1
            self.quality.memory_threshold_mb = min(8192, self.quality.memory_threshold_mb)

        # Small model optimizations
        elif model_size < 1_000_000:  # 1M parameters
            self.optimization_level = OptimizationLevel.CONSERVATIVE
            self.optimization_passes["magnitude_pruning"].parameters["sparsity"] = 0.05

        # Vision model optimizations
        if model_info.get("domain") == "vision":
            self.optimization_passes["operator_fusion"].parameters["conv_bn_fusion"] = True

        # NLP model optimizations
        elif model_info.get("domain") == "nlp":
            self.optimization_passes["operator_fusion"].parameters["attention_fusion"] = True

    def validate_configuration(self) -> List[str]:
        """Validate current configuration and return list of issues"""
        issues = []

        # Validate optimization level consistency
        if self.optimization_level == OptimizationLevel.MAXIMUM:
            if not self.profiling.enable_detailed_profiling:
                issues.append("Detailed profiling should be enabled for maximum optimization level")

        # Validate hardware configuration
        if self.hardware.cuda_memory_fraction > 1.0 or self.hardware.cuda_memory_fraction < 0.1:
            issues.append("CUDA memory fraction must be between 0.1 and 1.0")

        if self.hardware.cpu_threads is not None and self.hardware.cpu_threads < 1:
            issues.append("CPU threads must be positive")

        # Validate quality thresholds
        if self.quality.accuracy_threshold < 0.0 or self.quality.accuracy_threshold > 1.0:
            issues.append("Accuracy threshold must be between 0.0 and 1.0")

        if self.quality.latency_threshold_ms <= 0:
            issues.append("Latency threshold must be positive")

        # Validate export configuration
        if not self.export.export_formats:
            issues.append("At least one export format must be specified")

        # Validate optimization passes
        enabled_passes = sum(1 for p in self.optimization_passes.values() if p.enabled)
        if enabled_passes == 0:
            issues.append("At least one optimization pass should be enabled")

        return issues

    def save_configuration(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Save current configuration to file"""
        save_path = Path(path) if path else self.config_path
        if not save_path:
            save_path = Path("baking_config.yaml")

        config_dict = {
            "optimization_level": self.optimization_level.name,
            "hardware": asdict(self.hardware),
            "quality": asdict(self.quality),
            "profiling": asdict(self.profiling),
            "export": asdict(self.export),
            "logging": asdict(self.logging),
            "optimization_passes": {
                name: asdict(config) for name, config in self.optimization_passes.items()
            },
            "environment_overrides": self.environment_overrides
        }

        # Convert enums to strings for serialization
        config_dict["hardware"]["target_device"] = self.hardware.target_device.value
        config_dict["export"]["export_formats"] = [fmt.value for fmt in self.export.export_formats]

        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        self.logger.info(f"Configuration saved to {save_path}")
        return save_path

    def load_configuration(self, path: Union[str, Path]):
        """Load configuration from file"""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {load_path}")

        self.logger.info(f"Loading configuration from {load_path}")

        with open(load_path, 'r') as f:
            if load_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                config_dict = yaml.safe_load(f)

        # Load optimization level
        if "optimization_level" in config_dict:
            level_name = config_dict["optimization_level"]
            self.optimization_level = OptimizationLevel[level_name.upper()]

        # Load hardware configuration
        if "hardware" in config_dict:
            hardware_config = config_dict["hardware"]
            self.hardware = HardwareConfig(**hardware_config)
            if "target_device" in hardware_config:
                self.hardware.target_device = DeviceType(hardware_config["target_device"])

        # Load other configurations
        if "quality" in config_dict:
            self.quality = QualityConfig(**config_dict["quality"])

        if "profiling" in config_dict:
            self.profiling = ProfilingConfig(**config_dict["profiling"])

        if "export" in config_dict:
            export_config = config_dict["export"]
            self.export = ExportConfig(**export_config)
            if "export_formats" in export_config:
                self.export.export_formats = [ExportFormat(fmt) for fmt in export_config["export_formats"]]

        if "logging" in config_dict:
            self.logging = LoggingConfig(**config_dict["logging"])

        # Load optimization passes
        if "optimization_passes" in config_dict:
            passes_config = config_dict["optimization_passes"]
            for pass_name, pass_data in passes_config.items():
                if pass_name in self.optimization_passes:
                    self.optimization_passes[pass_name] = OptimizationPassConfig(**pass_data)

        # Load environment overrides
        if "environment_overrides" in config_dict:
            self.environment_overrides = config_dict["environment_overrides"]

        self.config_path = load_path

    def apply_environment_overrides(self, environment: str = "production"):
        """Apply environment-specific configuration overrides"""
        if environment not in self.environment_overrides:
            return

        overrides = self.environment_overrides[environment]
        self.logger.info(f"Applying {environment} environment overrides")

        # Apply overrides to various configuration sections
        if "quality" in overrides:
            for key, value in overrides["quality"].items():
                if hasattr(self.quality, key):
                    setattr(self.quality, key, value)

        if "hardware" in overrides:
            for key, value in overrides["hardware"].items():
                if hasattr(self.hardware, key):
                    setattr(self.hardware, key, value)

        if "optimization_passes" in overrides:
            for pass_name, pass_overrides in overrides["optimization_passes"].items():
                if pass_name in self.optimization_passes:
                    for key, value in pass_overrides.items():
                        if hasattr(self.optimization_passes[pass_name], key):
                            setattr(self.optimization_passes[pass_name], key, value)

    def get_phase5_integration_config(self) -> Dict[str, Any]:
        """Get configuration for Phase 5 integration"""
        return {
            "model_input_dir": "./phase5_models",
            "expected_model_formats": ["pytorch", "torchscript"],
            "model_validation": {
                "check_architecture": True,
                "verify_weights": True,
                "validate_forward_pass": True
            },
            "metadata_requirements": [
                "training_config",
                "model_architecture",
                "performance_metrics"
            ]
        }

    def get_phase7_integration_config(self) -> Dict[str, Any]:
        """Get configuration for Phase 7 integration"""
        return {
            "output_dir": "./phase7_ready",
            "adas_compatibility": {
                "inference_only": True,
                "deterministic_output": True,
                "memory_constraints": True,
                "latency_constraints": True
            },
            "export_metadata": {
                "optimization_applied": True,
                "performance_benchmarks": True,
                "quality_metrics": True,
                "hardware_requirements": True
            },
            "validation_requirements": {
                "accuracy_preservation": self.quality.accuracy_threshold,
                "latency_compliance": self.quality.latency_threshold_ms,
                "memory_compliance": self.quality.memory_threshold_mb
            }
        }

    def create_preset_configurations(self) -> Dict[str, 'BakingConfiguration']:
        """Create preset configurations for common scenarios"""
        presets = {}

        # Development preset - fast iteration
        dev_config = BakingConfiguration()
        dev_config.optimization_level = OptimizationLevel.BASIC
        dev_config.profiling.benchmark_iterations = 50
        dev_config.quality.accuracy_threshold = 0.90
        dev_config.export.export_formats = [ExportFormat.PYTORCH]
        presets["development"] = dev_config

        # Production preset - maximum quality
        prod_config = BakingConfiguration()
        prod_config.optimization_level = OptimizationLevel.AGGRESSIVE
        prod_config.quality.enable_nasa_compliance = True
        prod_config.quality.required_nasa_score = 0.95
        prod_config.export.include_benchmarks = True
        presets["production"] = prod_config

        # Edge deployment preset - size and speed optimized
        edge_config = BakingConfiguration()
        edge_config.optimization_level = OptimizationLevel.MAXIMUM
        edge_config.optimization_passes["magnitude_pruning"].parameters["sparsity"] = 0.3
        edge_config.quality.memory_threshold_mb = 512
        edge_config.quality.latency_threshold_ms = 50
        presets["edge"] = edge_config

        # Research preset - detailed analysis
        research_config = BakingConfiguration()
        research_config.optimization_level = OptimizationLevel.STANDARD
        research_config.profiling.enable_detailed_profiling = True
        research_config.profiling.profile_batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        research_config.export.include_metadata = True
        presets["research"] = research_config

        return presets

    def __str__(self) -> str:
        """String representation of configuration"""
        return f"BakingConfiguration(level={self.optimization_level.name}, device={self.hardware.target_device.value})"

    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"BakingConfiguration(optimization_level={self.optimization_level}, "
                f"enabled_passes={len([p for p in self.optimization_passes.values() if p.enabled])}, "
                f"target_device={self.hardware.target_device.value})")


def create_default_config(config_path: Optional[Union[str, Path]] = None) -> BakingConfiguration:
    """Create a default baking configuration"""
    config = BakingConfiguration(config_path)

    # Set sensible defaults
    config.optimization_level = OptimizationLevel.STANDARD
    config.hardware.target_device = DeviceType.AUTO
    config.quality.accuracy_threshold = 0.95
    config.export.export_formats = [ExportFormat.PYTORCH, ExportFormat.TORCHSCRIPT]

    return config


def main():
    """Example usage of BakingConfiguration"""
    # Create default configuration
    config = create_default_config()

    print(f"Default configuration: {config}")
    print(f"Enabled passes: {len(config.get_enabled_passes())}")

    # Validate configuration
    issues = config.validate_configuration()
    if issues:
        print(f"Configuration issues: {issues}")
    else:
        print("Configuration is valid")

    # Save configuration
    config_path = config.save_configuration("example_baking_config.yaml")
    print(f"Configuration saved to: {config_path}")

    # Create presets
    presets = config.create_preset_configurations()
    print(f"Available presets: {list(presets.keys())}")


if __name__ == "__main__":
    main()