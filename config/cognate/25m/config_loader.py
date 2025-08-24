#!/usr/bin/env python3
"""
Cognate 25M Configuration Loader
Dynamic configuration loading and validation system for the 25M Cognate Refiner
"""

from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConfigMetadata:
    """Configuration metadata tracking."""
    config_name: str
    version: str
    loaded_at: str
    environment: str
    source_files: list[str]
    validation_status: str = "pending"
    parameter_count: int | None = None
    

class ConfigurationError(Exception):
    """Configuration-specific exceptions."""
    pass


class CognateConfigLoader:
    """
    Comprehensive configuration loader for Cognate 25M system.
    
    Supports:
    - Multi-file configuration loading
    - Environment-specific overrides
    - Configuration validation
    - Parameter count verification
    - Integration with Agent Forge pipeline
    """
    
    def __init__(self, config_dir: str | None = None):
        """Initialize configuration loader."""
        if config_dir is None:
            # Default to this directory
            config_dir = Path(__file__).parent
        
        self.config_dir = Path(config_dir)
        self.environment = self._detect_environment()
        self.loaded_configs: dict[str, Any] = {}
        self.metadata: ConfigMetadata | None = None
        
        logger.info(f"Initialized CognateConfigLoader for environment: {self.environment}")
        logger.info(f"Configuration directory: {self.config_dir}")
    
    def _detect_environment(self) -> str:
        """Detect current environment from various sources."""
        # Check environment variable
        env = os.getenv("COGNATE_ENVIRONMENT")
        if env:
            return env.lower()
            
        # Check command line arguments
        import sys
        for arg in sys.argv:
            if arg.startswith("--environment="):
                return arg.split("=", 1)[1].lower()
            if arg == "--dev":
                return "development"
            if arg == "--staging":
                return "staging"
            if arg == "--prod":
                return "production"
            if arg == "--test":
                return "testing"
                
        # Check for CI/testing environment
        if os.getenv("CI") or os.getenv("PYTEST_CURRENT_TEST"):
            return "testing"
            
        # Default to development
        return "development"
    
    def load_config(self, config_name: str = "complete") -> dict[str, Any]:
        """
        Load complete Cognate 25M configuration.
        
        Args:
            config_name: Type of configuration to load
                        - "complete": All configuration files
                        - "model": Only model configuration
                        - "training": Only training configuration
                        - "deployment": Only deployment configuration
                        
        Returns:
            Complete configuration dictionary
        """
        logger.info(f"Loading {config_name} configuration for environment: {self.environment}")
        
        # Define configuration file mapping
        config_files = {
            "complete": [
                "cognate_25m_config.yaml",
                "training_config.yaml", 
                "hyperparameter_config.yaml",
                "dataset_config.yaml",
                "validation_config.yaml"
            ],
            "model": ["cognate_25m_config.yaml"],
            "training": ["training_config.yaml", "hyperparameter_config.yaml", "dataset_config.yaml"],
            "deployment": ["deployment_config.yaml"],
            "validation": ["validation_config.yaml"],
            "pipeline": ["../agent_forge/cognate/pipeline_integration.yaml"],
            "environments": ["environment_configs.yaml"]
        }
        
        if config_name not in config_files:
            raise ConfigurationError(f"Unknown configuration type: {config_name}")
            
        # Load base configuration files
        config = {}
        source_files = []
        
        for config_file in config_files[config_name]:
            file_path = self.config_dir / config_file
            if file_path.exists():
                logger.info(f"Loading configuration file: {config_file}")
                file_config = self._load_yaml_file(file_path)
                config = self._merge_configs(config, file_config)
                source_files.append(str(file_path))
            else:
                logger.warning(f"Configuration file not found: {file_path}")
        
        # Apply environment-specific overrides
        env_config = self._load_environment_config()
        if env_config:
            config = self._apply_environment_overrides(config, env_config)
            
        # Apply final validation and processing
        config = self._post_process_config(config)
        
        # Store loaded configuration
        self.loaded_configs[config_name] = config
        
        # Create metadata
        self.metadata = ConfigMetadata(
            config_name=config_name,
            version="1.0.0",
            loaded_at=datetime.now().isoformat(),
            environment=self.environment,
            source_files=source_files
        )
        
        logger.info(f"Successfully loaded {config_name} configuration")
        return config
    
    def _load_yaml_file(self, file_path: Path) -> dict[str, Any]:
        """Load a single YAML configuration file."""
        try:
            with open(file_path, encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigurationError(f"Error loading {file_path}: {e}")
    
    def _load_environment_config(self) -> dict[str, Any] | None:
        """Load environment-specific configuration overrides."""
        env_config_file = self.config_dir / "environment_configs.yaml"
        if not env_config_file.exists():
            return None
            
        try:
            env_configs = self._load_yaml_file(env_config_file)
            return env_configs.get("environments", {}).get(self.environment)
        except Exception as e:
            logger.warning(f"Error loading environment config: {e}")
            return None
    
    def _merge_configs(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _apply_environment_overrides(self, config: dict[str, Any], env_overrides: dict[str, Any]) -> dict[str, Any]:
        """Apply environment-specific overrides to configuration."""
        if not env_overrides:
            return config
            
        logger.info(f"Applying {self.environment} environment overrides")
        
        # Apply dot-notation overrides
        for key, value in env_overrides.items():
            if isinstance(key, str) and '.' in key:
                self._set_nested_value(config, key, value)
            else:
                # Direct key override
                if key in config:
                    if isinstance(config[key], dict) and isinstance(value, dict):
                        config[key] = self._merge_configs(config[key], value)
                    else:
                        config[key] = value
                        
        return config
    
    def _set_nested_value(self, config: dict[str, Any], key_path: str, value: Any):
        """Set value in nested dictionary using dot notation."""
        keys = key_path.split('.')
        current = config
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
            
        # Set final value
        current[keys[-1]] = value
    
    def _post_process_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Post-process configuration for final validation and computed values."""
        # Compute derived values
        if "model" in config:
            model_config = config["model"]
            if "d_model" in model_config and "n_heads" in model_config:
                model_config["head_dim"] = model_config["d_model"] // model_config["n_heads"]
            if "d_model" in model_config and "ffn_mult" in model_config:
                model_config["intermediate_size"] = model_config["d_model"] * model_config["ffn_mult"]
        
        # Compute effective batch size
        if "training" in config and "batching" in config["training"]:
            batching = config["training"]["batching"]
            if "batch_size" in batching and "gradient_accumulation_steps" in batching:
                batching["effective_batch_size"] = batching["batch_size"] * batching["gradient_accumulation_steps"]
        
        # Validate memory dimension matches model dimension
        if "model" in config and "memory" in config:
            if config["model"].get("d_model") != config["memory"].get("d_mem"):
                logger.warning("Memory dimension doesn't match model dimension, adjusting...")
                config["memory"]["d_mem"] = config["model"]["d_model"]
        
        return config
    
    def validate_config(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Validate configuration for consistency and correctness.
        
        Returns:
            Validation report with status and any issues found
        """
        if config is None:
            if "complete" in self.loaded_configs:
                config = self.loaded_configs["complete"]
            else:
                raise ConfigurationError("No configuration loaded for validation")
                
        validation_report = {
            "status": "passed",
            "errors": [],
            "warnings": [],
            "checks": {}
        }
        
        try:
            # Validate model architecture
            self._validate_model_architecture(config, validation_report)
            
            # Validate training configuration
            self._validate_training_config(config, validation_report)
            
            # Validate memory configuration
            self._validate_memory_config(config, validation_report)
            
            # Validate ACT configuration
            self._validate_act_config(config, validation_report)
            
            # Overall validation status
            if validation_report["errors"]:
                validation_report["status"] = "failed"
            elif validation_report["warnings"]:
                validation_report["status"] = "warnings"
                
            # Update metadata
            if self.metadata:
                self.metadata.validation_status = validation_report["status"]
                
        except Exception as e:
            validation_report["status"] = "error"
            validation_report["errors"].append(f"Validation error: {e}")
            
        logger.info(f"Configuration validation: {validation_report['status']}")
        return validation_report
    
    def _validate_model_architecture(self, config: dict[str, Any], report: dict[str, Any]):
        """Validate model architecture configuration."""
        if "model" not in config:
            report["errors"].append("Missing model configuration")
            return
            
        model = config["model"]
        
        # Check required fields
        required_fields = ["d_model", "n_layers", "n_heads", "vocab_size"]
        for field in required_fields:
            if field not in model:
                report["errors"].append(f"Missing required model field: {field}")
            else:
                report["checks"][f"has_{field}"] = True
                
        # Check dimension consistency
        if "d_model" in model and "n_heads" in model:
            if model["d_model"] % model["n_heads"] != 0:
                report["errors"].append("d_model must be divisible by n_heads")
            else:
                report["checks"]["d_model_head_divisibility"] = True
                
        # Check reasonable values
        if "d_model" in model:
            if not (64 <= model["d_model"] <= 8192):
                report["warnings"].append("d_model outside reasonable range (64-8192)")
            else:
                report["checks"]["d_model_reasonable"] = True
                
        if "n_layers" in model:
            if not (1 <= model["n_layers"] <= 48):
                report["warnings"].append("n_layers outside reasonable range (1-48)")
            else:
                report["checks"]["n_layers_reasonable"] = True
    
    def _validate_training_config(self, config: dict[str, Any], report: dict[str, Any]):
        """Validate training configuration."""
        if "training" not in config:
            report["warnings"].append("No training configuration found")
            return
            
        training = config["training"]
        
        # Check learning rate - handle nested structure
        lr = None
        if "learning_rate" in training:
            lr = training["learning_rate"]
        elif "optimization" in training and "learning_rate" in training["optimization"]:
            lr = training["optimization"]["learning_rate"]
        elif "training_hyperparameters" in config:
            train_hyper = config["training_hyperparameters"]
            if "optimization" in train_hyper and "learning_rate" in train_hyper["optimization"]:
                lr = train_hyper["optimization"]["learning_rate"]
                
        if lr is not None and isinstance(lr, int | float):
            if not (1e-6 <= lr <= 1e-2):
                report["warnings"].append("Learning rate outside reasonable range")
            else:
                report["checks"]["learning_rate_reasonable"] = True
                
        # Check batch configuration - handle nested structure
        batch_size = None
        if "batch_size" in training:
            batch_size = training["batch_size"]
        elif "batching" in training and "batch_size" in training["batching"]:
            batch_size = training["batching"]["batch_size"]
        elif "training_hyperparameters" in config:
            train_hyper = config["training_hyperparameters"]
            if "batching" in train_hyper and "batch_size" in train_hyper["batching"]:
                batch_size = train_hyper["batching"]["batch_size"]
                
        if batch_size is not None and isinstance(batch_size, int | float):
            if batch_size <= 0:
                report["errors"].append("Batch size must be positive")
            else:
                report["checks"]["batch_size_positive"] = True
    
    def _validate_memory_config(self, config: dict[str, Any], report: dict[str, Any]):
        """Validate memory system configuration."""
        if "memory" not in config:
            report["warnings"].append("No memory configuration found")
            return
            
        memory = config["memory"]
        model = config.get("model", {})
        
        # Check memory dimension matches model dimension
        if "d_mem" in memory and "d_model" in model:
            if memory["d_mem"] != model["d_model"]:
                report["warnings"].append("Memory dimension should match model dimension")
            else:
                report["checks"]["memory_model_dim_match"] = True
                
        # Check memory capacity
        if "mem_capacity" in memory:
            if not (64 <= memory["mem_capacity"] <= 16384):
                report["warnings"].append("Memory capacity outside reasonable range")
            else:
                report["checks"]["memory_capacity_reasonable"] = True
    
    def _validate_act_config(self, config: dict[str, Any], report: dict[str, Any]):
        """Validate ACT configuration."""
        if "act" not in config:
            report["warnings"].append("No ACT configuration found")
            return
            
        act = config["act"]
        
        # Check ACT threshold - handle nested structure
        threshold = None
        if "act_threshold" in act:
            threshold = act["act_threshold"]
        elif "core" in act and "act_threshold" in act["core"]:
            threshold = act["core"]["act_threshold"]
        elif "act_hyperparameters" in config:
            act_hyper = config["act_hyperparameters"]
            if "core" in act_hyper and "act_threshold" in act_hyper["core"]:
                threshold = act_hyper["core"]["act_threshold"]
                
        if threshold is not None and isinstance(threshold, int | float):
            if not (0.5 <= threshold <= 0.999):
                report["warnings"].append("ACT threshold outside reasonable range")
            else:
                report["checks"]["act_threshold_reasonable"] = True
    
    def estimate_parameter_count(self, config: dict[str, Any] | None = None) -> int:
        """Estimate parameter count from configuration."""
        if config is None:
            config = self.loaded_configs.get("complete", {})
            
        if "model" not in config:
            raise ConfigurationError("Model configuration required for parameter estimation")
            
        model = config["model"]
        
        # Extract parameters
        vocab_size = model.get("vocab_size", 32000)
        d_model = model.get("d_model", 216)
        n_layers = model.get("n_layers", 11)
        model.get("n_heads", 4)
        ffn_mult = model.get("ffn_mult", 4)
        
        # Calculate parameter counts
        # Token embeddings
        embedding_params = vocab_size * d_model
        
        # Transformer layers
        layer_params = 0
        for _ in range(n_layers):
            # Self-attention
            qkv_params = d_model * d_model * 3  # Q, K, V projections
            o_params = d_model * d_model        # Output projection
            attn_params = qkv_params + o_params
            
            # FFN
            ffn_dim = d_model * ffn_mult
            ffn_params = d_model * ffn_dim * 2 + ffn_dim * d_model  # up, gate, down
            
            # Layer norms
            norm_params = d_model * 2  # input and post-attention norms
            
            layer_params += attn_params + ffn_params + norm_params
        
        # Final norm
        final_norm_params = d_model
        
        # Language modeling head
        lm_head_params = d_model * vocab_size
        
        # ACT halting head
        halting_head_params = d_model + 1  # Linear layer + bias
        
        # Memory controllers (rough estimate)
        memory_params = d_model * d_model * 2  # Read + write controllers
        
        total_params = (
            embedding_params +
            layer_params + 
            final_norm_params +
            lm_head_params +
            halting_head_params +
            memory_params
        )
        
        logger.info(f"Estimated parameter count: {total_params:,}")
        
        # Update metadata
        if self.metadata:
            self.metadata.parameter_count = total_params
            
        return total_params
    
    def save_config(self, config: dict[str, Any], output_path: str, format: str = "yaml"):
        """Save configuration to file."""
        output_path = Path(output_path)
        
        try:
            if format.lower() == "yaml":
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
            else:
                raise ConfigurationError(f"Unsupported format: {format}")
                
            logger.info(f"Configuration saved to: {output_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration: {e}")
    
    def get_metadata(self) -> ConfigMetadata | None:
        """Get configuration metadata."""
        return self.metadata
    
    def create_cognate_config_object(self, config: dict[str, Any] | None = None) -> 'CognateConfig':
        """
        Create CognateConfig object from loaded configuration.
        
        Returns:
            CognateConfig instance ready for model creation
        """
        if config is None:
            config = self.loaded_configs.get("complete", {})
            
        # Import here to avoid circular imports
        try:
            from pathlib import Path
            import sys
            sys.path.append(str(Path(__file__).parent.parent.parent.parent / "packages"))
            from agent_forge.models.cognate.refiner_core import CognateConfig
        except ImportError:
            raise ConfigurationError("Could not import CognateConfig. Ensure the package is installed.")
            
        # Extract model configuration
        model_config = config.get("model", {})
        memory_config = config.get("memory", {})
        act_config = config.get("act", {})
        system_config = config.get("system", {})
        
        # Create CognateConfig instance
        cognate_config = CognateConfig(
            # Model architecture
            vocab_size=model_config.get("vocab_size", 32000),
            d_model=model_config.get("d_model", 216),
            n_layers=model_config.get("n_layers", 11),
            n_heads=model_config.get("n_heads", 4),
            ffn_mult=model_config.get("ffn_mult", 4),
            max_seq_len=model_config.get("max_seq_len", 2048),
            
            # Memory configuration
            d_mem=memory_config.get("d_mem", model_config.get("d_model", 216)),
            mem_capacity=memory_config.get("mem_capacity", 4096),
            mem_topk=memory_config.get("mem_topk", 4),
            
            # Memory policies
            read_policy=memory_config.get("read_policy", "entropy_gated"),
            write_policy=memory_config.get("write_policy", "surprise_novelty"),
            entropy_threshold=memory_config.get("entropy_threshold", 0.8),
            surprise_threshold=memory_config.get("surprise_threshold", 0.6),
            novelty_threshold=memory_config.get("novelty_threshold", 0.7),
            
            # ACT configuration
            act_threshold=act_config.get("act_threshold", 0.99),
            act_epsilon=act_config.get("act_epsilon", 0.01),
            max_act_steps=act_config.get("max_act_steps", 16),
            
            # System configuration
            device=system_config.get("device", "auto"),
            torch_dtype=system_config.get("torch_dtype", "float32"),
            
            # Training hyperparameters
            dropout=config.get("model_hyperparameters", {}).get("regularization", {}).get("dropout", 0.1),
            layer_norm_eps=config.get("model_hyperparameters", {}).get("regularization", {}).get("layer_norm_eps", 1e-5),
        )
        
        return cognate_config


# Utility functions for easy usage
def load_cognate_config(environment: str | None = None, config_type: str = "complete") -> dict[str, Any]:
    """
    Convenient function to load Cognate configuration.
    
    Args:
        environment: Environment to load (development, staging, production, testing)
        config_type: Type of configuration (complete, model, training, etc.)
        
    Returns:
        Loaded configuration dictionary
    """
    if environment:
        os.environ["COGNATE_ENVIRONMENT"] = environment
        
    loader = CognateConfigLoader()
    return loader.load_config(config_type)


def validate_cognate_config(config: dict[str, Any] | None = None, environment: str | None = None) -> dict[str, Any]:
    """
    Convenient function to validate Cognate configuration.
    
    Args:
        config: Configuration to validate (loads default if None)
        environment: Environment context for validation
        
    Returns:
        Validation report
    """
    if environment:
        os.environ["COGNATE_ENVIRONMENT"] = environment
        
    loader = CognateConfigLoader()
    
    if config is None:
        config = loader.load_config("complete")
    
    return loader.validate_config(config)


def create_cognate_model_from_config(config_path: str | None = None, environment: str | None = None):
    """
    Create CognateRefiner model from configuration.
    
    Args:
        config_path: Path to configuration file (uses default if None)
        environment: Environment to use
        
    Returns:
        Initialized CognateRefiner model
    """
    if environment:
        os.environ["COGNATE_ENVIRONMENT"] = environment
        
    loader = CognateConfigLoader(config_path)
    config = loader.load_config("complete")
    
    # Validate configuration
    validation_report = loader.validate_config(config)
    if validation_report["status"] == "failed":
        raise ConfigurationError(f"Configuration validation failed: {validation_report['errors']}")
    
    # Create CognateConfig object
    cognate_config = loader.create_cognate_config_object(config)
    
    # Create and return model
    try:
        from pathlib import Path
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent.parent / "packages"))
        from agent_forge.models.cognate.refiner_core import CognateRefiner
        
        model = CognateRefiner(cognate_config)
        logger.info(f"Created CognateRefiner with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model
        
    except ImportError as e:
        raise ConfigurationError(f"Could not import CognateRefiner: {e}")


if __name__ == "__main__":
    # Example usage and testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Cognate 25M Configuration Loader")
    parser.add_argument("--config-type", default="complete", 
                       choices=["complete", "model", "training", "deployment"],
                       help="Type of configuration to load")
    parser.add_argument("--environment", choices=["development", "staging", "production", "testing"],
                       help="Environment to use")
    parser.add_argument("--validate", action="store_true", help="Validate configuration")
    parser.add_argument("--estimate-params", action="store_true", help="Estimate parameter count")
    parser.add_argument("--output", help="Save configuration to file")
    parser.add_argument("--format", choices=["yaml", "json"], default="yaml", help="Output format")
    
    args = parser.parse_args()
    
    try:
        # Set environment if specified
        if args.environment:
            os.environ["COGNATE_ENVIRONMENT"] = args.environment
            
        # Create loader and load configuration
        loader = CognateConfigLoader()
        config = loader.load_config(args.config_type)
        
        print(f"Loaded {args.config_type} configuration for environment: {loader.environment}")
        
        # Validate if requested
        if args.validate:
            validation_report = loader.validate_config(config)
            print(f"Validation status: {validation_report['status']}")
            if validation_report['errors']:
                print("Errors:")
                for error in validation_report['errors']:
                    print(f"  - {error}")
            if validation_report['warnings']:
                print("Warnings:")
                for warning in validation_report['warnings']:
                    print(f"  - {warning}")
        
        # Estimate parameters if requested
        if args.estimate_params:
            param_count = loader.estimate_parameter_count(config)
            print(f"Estimated parameter count: {param_count:,}")
            
            # Check if close to 25M target
            target = 25_000_000
            diff = abs(param_count - target)
            percent_diff = (diff / target) * 100
            print(f"Difference from 25M target: {diff:,} ({percent_diff:.2f}%)")
            
            if diff <= 2_000_000:
                print("✅ Parameter count within acceptable range (±2M)")
            else:
                print("⚠️ Parameter count outside acceptable range")
        
        # Save if requested
        if args.output:
            loader.save_config(config, args.output, args.format)
            
        # Print metadata
        metadata = loader.get_metadata()
        if metadata:
            print("\nConfiguration Metadata:")
            print(f"  Name: {metadata.config_name}")
            print(f"  Version: {metadata.version}")
            print(f"  Environment: {metadata.environment}")
            print(f"  Loaded at: {metadata.loaded_at}")
            print(f"  Source files: {len(metadata.source_files)}")
            if metadata.parameter_count:
                print(f"  Parameter count: {metadata.parameter_count:,}")
                
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)