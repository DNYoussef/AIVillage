"""
Cogment Configuration Loader.

Unified configuration loading and management system for all Cogment components.
Provides parameter budget validation and stage-specific configuration loading.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class CogmentCompleteConfig:
    """Complete Cogment configuration combining all components."""

    # Core configurations
    model_config: dict[str, Any]
    training_config: dict[str, Any]
    grokfast_config: dict[str, Any]
    deployment_config: dict[str, Any]

    # Stage-specific configurations
    stage_configs: dict[int, dict[str, Any]] = field(default_factory=dict)

    # Metadata
    config_version: str = "1.0"
    loaded_from: str | None = None
    parameter_budget: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageConfig:
    """Individual stage configuration."""

    stage_id: int
    name: str
    description: str
    training: dict[str, Any]
    model: dict[str, Any]
    data: dict[str, Any]
    loss: dict[str, Any]
    grokfast: dict[str, Any]
    convergence: dict[str, Any]

    # Optional sections
    memory: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    resources: dict[str, Any] | None = None
    advancement: dict[str, Any] | None = None


@dataclass
class TrainingConfig:
    """Training configuration for Agent 4 compatibility."""

    curriculum: dict[str, Any]
    optimizers: dict[str, Any]
    schedulers: dict[str, Any]
    training: dict[str, Any]
    memory: dict[str, Any]
    loss: dict[str, Any]


class CogmentConfigLoader:
    """
    Unified configuration loader for Cogment system.

    Loads and validates all configuration files, ensures parameter budget compliance,
    and provides convenient access to stage-specific configurations.
    """

    def __init__(self, config_dir: str | Path | None = None):
        """
        Initialize configuration loader.

        Args:
            config_dir: Path to configuration directory (defaults to current package)
        """
        if config_dir is None:
            config_dir = Path(__file__).parent

        self.config_dir = Path(config_dir)
        self.stage_configs_dir = self.config_dir / "stage_configs"

        # Validate configuration directory structure
        self._validate_config_structure()

        logger.info(f"Initialized CogmentConfigLoader with config_dir: {self.config_dir}")

    def _validate_config_structure(self):
        """Validate that all required configuration files exist."""
        required_files = [
            "cogment_config.yaml",
            "training_config.yaml",
            "grokfast_config.yaml",
            "deployment_config.yaml",
        ]

        required_stage_files = [
            "stage_0_sanity.yaml",
            "stage_1_arc.yaml",
            "stage_2_puzzles.yaml",
            "stage_3_reasoning.yaml",
            "stage_4_longcontext.yaml",
        ]

        # Check main config files
        for file_name in required_files:
            file_path = self.config_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Required configuration file not found: {file_path}")

        # Check stage config files
        for file_name in required_stage_files:
            file_path = self.stage_configs_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Required stage configuration file not found: {file_path}")

        logger.info("Configuration directory structure validated successfully")

    def load_complete_config(self) -> CogmentCompleteConfig:
        """
        Load complete Cogment configuration from all files.

        Returns:
            Complete configuration object with all components
        """
        logger.info("Loading complete Cogment configuration")

        # Load main configuration files
        model_config = self._load_yaml("cogment_config.yaml")
        training_config = self._load_yaml("training_config.yaml")
        grokfast_config = self._load_yaml("grokfast_config.yaml")
        deployment_config = self._load_yaml("deployment_config.yaml")

        # Load all stage configurations
        stage_configs = {}
        for stage_id in range(5):  # Stages 0-4
            stage_config = self.load_stage_config(stage_id)
            stage_configs[stage_id] = stage_config.__dict__

        # Create complete configuration
        complete_config = CogmentCompleteConfig(
            model_config=model_config,
            training_config=training_config,
            grokfast_config=grokfast_config,
            deployment_config=deployment_config,
            stage_configs=stage_configs,
            loaded_from=str(self.config_dir),
        )

        # Validate parameter budget
        if model_config.get("parameter_budget", {}).get("validate_on_load", True):
            self.validate_parameter_budget(complete_config)

        logger.info("Complete Cogment configuration loaded successfully")
        return complete_config

    def load_stage_config(self, stage: int) -> StageConfig:
        """
        Load configuration for a specific curriculum stage.

        Args:
            stage: Stage number (0-4)

        Returns:
            Stage-specific configuration
        """
        stage_file_map = {
            0: "stage_0_sanity.yaml",
            1: "stage_1_arc.yaml",
            2: "stage_2_puzzles.yaml",
            3: "stage_3_reasoning.yaml",
            4: "stage_4_longcontext.yaml",
        }

        if stage not in stage_file_map:
            raise ValueError(f"Invalid stage number: {stage}. Must be 0-4.")

        stage_file = stage_file_map[stage]
        stage_data = self._load_yaml(stage_file, subdirectory="stage_configs")

        # Extract required sections
        stage_config = StageConfig(
            stage_id=stage_data["stage"]["stage_id"],
            name=stage_data["stage"]["name"],
            description=stage_data["stage"]["description"],
            training=stage_data["training"],
            model=stage_data["model"],
            data=stage_data["data"],
            loss=stage_data["loss"],
            grokfast=stage_data["grokfast"],
            convergence=stage_data["convergence"],
            memory=stage_data.get("memory"),
            validation=stage_data.get("validation"),
            resources=stage_data.get("resources"),
            advancement=stage_data.get("advancement"),
        )

        logger.debug(f"Loaded stage {stage} configuration: {stage_config.name}")
        return stage_config

    def load_training_config(self) -> TrainingConfig:
        """
        Load training configuration in format compatible with Agent 4's TrainingConfig.

        Returns:
            Training configuration object
        """
        training_data = self._load_yaml("training_config.yaml")

        training_config = TrainingConfig(
            curriculum=training_data["curriculum"],
            optimizers=training_data["optimizers"],
            schedulers=training_data["schedulers"],
            training=training_data["training"],
            memory=training_data["memory"],
            loss=training_data["loss"],
        )

        logger.debug("Training configuration loaded for Agent 4 compatibility")
        return training_config

    def override_with_args(self, config: CogmentCompleteConfig, args: dict[str, Any]) -> CogmentCompleteConfig:
        """
        Override configuration values with command-line arguments or runtime parameters.

        Args:
            config: Base configuration
            args: Override arguments

        Returns:
            Updated configuration
        """
        logger.info(f"Applying {len(args)} configuration overrides")

        # Apply overrides to model config
        if "model" in args:
            for key, value in args["model"].items():
                if isinstance(value, dict) and key in config.model_config:
                    config.model_config[key].update(value)
                else:
                    config.model_config[key] = value

        # Apply overrides to training config
        if "training" in args:
            config.training_config.update(args["training"])

        # Apply overrides to specific stages
        if "stage_overrides" in args:
            for stage_id, overrides in args["stage_overrides"].items():
                if stage_id in config.stage_configs:
                    config.stage_configs[stage_id].update(overrides)

        # Re-validate parameter budget if model config changed
        if "model" in args:
            self.validate_parameter_budget(config)

        logger.debug("Configuration overrides applied successfully")
        return config

    def validate_parameter_budget(self, config: CogmentCompleteConfig) -> bool:
        """
        Validate that configuration stays within parameter budget.

        Args:
            config: Configuration to validate

        Returns:
            True if within budget, raises ValueError if not
        """
        model_config = config.model_config
        parameter_budget = model_config.get("parameter_budget", {})

        target_params = parameter_budget.get("target_params", 25_000_000)
        tolerance = parameter_budget.get("tolerance", 0.05)

        # Calculate parameter count based on current config
        estimated_params = self._estimate_parameter_count(model_config)

        # Calculate acceptable range
        min_params = target_params * (1 - tolerance)
        max_params = target_params * (1 + tolerance)

        # Validate
        if estimated_params < min_params:
            raise ValueError(
                f"Parameter count too low: {estimated_params:,} < {min_params:,} "
                f"(target: {target_params:,}, tolerance: {tolerance:.1%})"
            )

        if estimated_params > max_params:
            raise ValueError(
                f"Parameter count too high: {estimated_params:,} > {max_params:,} "
                f"(target: {target_params:,}, tolerance: {tolerance:.1%})"
            )

        logger.info(
            f"Parameter budget validation passed: {estimated_params:,} parameters "
            f"(target: {target_params:,}, range: {min_params:,}-{max_params:,})"
        )

        # Update config with actual estimate
        config.parameter_budget = {
            "target_params": target_params,
            "estimated_params": estimated_params,
            "within_budget": True,
            "utilization": estimated_params / target_params,
        }

        return True

    def _estimate_parameter_count(self, model_config: dict[str, Any]) -> int:
        """
        Estimate parameter count based on model configuration.

        Args:
            model_config: Model configuration

        Returns:
            Estimated parameter count
        """
        # Extract key parameters
        d_model = model_config["model"]["d_model"]
        n_layers = model_config["model"]["n_layers"]
        vocab_size = model_config["model"]["vocab_size"]
        d_ff = model_config["model"]["d_ff"]
        model_config["model"]["n_head"]

        # Memory parameters
        mem_slots = model_config["gated_ltm"]["mem_slots"]
        ltm_dim = model_config["gated_ltm"]["ltm_dim"]

        # Estimate components (based on Agent 1-4 analysis)

        # 1. Embedding parameters
        embedding_params = vocab_size * d_model

        # 2. Transformer backbone
        # Each layer: attention (4 * d_model^2) + FF (2 * d_model * d_ff) + norms
        attention_params_per_layer = 4 * d_model * d_model  # QKV + output projection
        ff_params_per_layer = 2 * d_model * d_ff
        norm_params_per_layer = 4 * d_model  # 2 layer norms per layer

        backbone_params = n_layers * (attention_params_per_layer + ff_params_per_layer + norm_params_per_layer)

        # 3. Memory parameters (main driver for Option A)
        memory_params = mem_slots * ltm_dim  # Main memory storage
        memory_gate_params = d_model * ltm_dim  # Memory gating
        memory_total = memory_params + memory_gate_params

        # 4. Refinement core (from Agent 1)
        refinement_params = model_config.get("estimated_params", {}).get("refinement_core", 2_400_000)

        # 5. Output head (if not tied)
        if not model_config["heads"]["tie_embeddings"]:
            output_head_params = vocab_size * d_model
        else:
            output_head_params = 0  # Tied with embeddings

        # Total estimate
        total_params = embedding_params + backbone_params + memory_total + refinement_params + output_head_params

        logger.debug(
            f"Parameter breakdown: embedding={embedding_params:,}, "
            f"backbone={backbone_params:,}, memory={memory_total:,}, "
            f"refinement={refinement_params:,}, output={output_head_params:,}"
        )

        return total_params

    def _load_yaml(self, filename: str, subdirectory: str | None = None) -> dict[str, Any]:
        """
        Load YAML configuration file.

        Args:
            filename: Name of YAML file
            subdirectory: Optional subdirectory within config_dir

        Returns:
            Loaded configuration data
        """
        if subdirectory:
            file_path = self.config_dir / subdirectory / filename
        else:
            file_path = self.config_dir / filename

        try:
            with open(file_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            logger.debug(f"Loaded configuration from {file_path}")
            return data

        except FileNotFoundError:
            logger.error(f"Configuration file not found: {file_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading {file_path}: {e}")
            raise

    def get_available_stages(self) -> list[int]:
        """Get list of available curriculum stages."""
        return list(range(5))  # Stages 0-4

    def get_stage_names(self) -> dict[int, str]:
        """Get mapping of stage IDs to names."""
        stage_names = {}
        for stage_id in self.get_available_stages():
            try:
                stage_config = self.load_stage_config(stage_id)
                stage_names[stage_id] = stage_config.name
            except Exception:
                stage_names[stage_id] = f"stage_{stage_id}"

        return stage_names

    def export_config_summary(self, config: CogmentCompleteConfig, output_path: Path | None = None) -> dict[str, Any]:
        """
        Export a summary of the complete configuration.

        Args:
            config: Configuration to summarize
            output_path: Optional path to save summary

        Returns:
            Configuration summary
        """
        summary = {
            "config_version": config.config_version,
            "loaded_from": config.loaded_from,
            "parameter_budget": config.parameter_budget,
            "model_summary": {
                "d_model": config.model_config["model"]["d_model"],
                "n_layers": config.model_config["model"]["n_layers"],
                "vocab_size": config.model_config["model"]["vocab_size"],
                "mem_slots": config.model_config["gated_ltm"]["mem_slots"],
            },
            "training_summary": {
                "curriculum_stages": len(config.stage_configs),
                "stage_names": [stage["name"] for stage in config.stage_configs.values()],
                "multi_optimizer": True,
                "grokfast_enabled": config.grokfast_config.get("refinement_core", {}).get("enabled", False),
            },
            "stage_summary": {},
        }

        # Add stage summaries
        for stage_id, stage_data in config.stage_configs.items():
            summary["stage_summary"][stage_id] = {
                "name": stage_data["name"],
                "max_steps": stage_data["training"]["max_steps"],
                "max_refinement_steps": stage_data["model"]["max_refinement_steps"],
                "batch_size": stage_data["training"]["batch_size"],
                "sequence_length": stage_data["training"]["sequence_length"],
            }

        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(summary, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration summary exported to {output_path}")

        return summary
