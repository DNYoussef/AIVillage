#!/usr/bin/env python3
"""
Agent Forge Pipeline Integration for Canonical Cognate Model

This module provides the integration layer between the canonical Cognate model
and the Agent Forge pipeline, ensuring full compatibility with EvoMerge and
other pipeline components.
"""

import json
import logging
from pathlib import Path
from typing import Any

import torch

from .cognate_production import (
    CognateProductionModel,
    create_three_production_cognate_models,
)

logger = logging.getLogger(__name__)


class AgentForgeCompatibilityError(Exception):
    """Exception raised when Agent Forge compatibility issues are detected."""

    pass


def validate_agent_forge_compatibility(models: list[CognateProductionModel]) -> dict[str, Any]:
    """
    Validate that models are compatible with Agent Forge pipeline.

    Args:
        models: List of Cognate models to validate

    Returns:
        Validation report dictionary

    Raises:
        AgentForgeCompatibilityError: If models are not compatible
    """
    if len(models) != 3:
        raise AgentForgeCompatibilityError(f"Expected 3 models, got {len(models)}")

    # Check parameter counts
    param_counts = [model.count_parameters() for model in models]
    target_params = 25_069_534

    validation_report = {
        "model_count": len(models),
        "parameter_counts": param_counts,
        "target_parameters": target_params,
        "parameter_consistency": all(count == param_counts[0] for count in param_counts),
        "parameter_accuracy": [],
        "state_dict_compatibility": True,
        "variant_names": [],
        "architecture_validation": True,
    }

    # Validate parameter accuracy
    for i, count in enumerate(param_counts):
        error_pct = abs(count - target_params) / target_params * 100
        validation_report["parameter_accuracy"].append(
            {
                "model": i + 1,
                "count": count,
                "error_percent": error_pct,
                "within_tolerance": error_pct <= 5.0,  # 5% tolerance
            }
        )

        if error_pct > 10.0:  # 10% is hard limit
            raise AgentForgeCompatibilityError(f"Model {i+1} parameter count {count:,} is {error_pct:.1f}% off target")

    # Check variant names
    variant_names = [getattr(model, "variant_name", f"model-{i+1}") for i, model in enumerate(models)]
    validation_report["variant_names"] = variant_names

    if len(set(variant_names)) != 3:
        raise AgentForgeCompatibilityError("Models must have unique variant names")

    # Check state dict compatibility (for EvoMerge)
    state_dicts = [model.state_dict() for model in models]
    key_sets = [set(sd.keys()) for sd in state_dicts]

    # All models must have identical keys
    if not all(keys == key_sets[0] for keys in key_sets[1:]):
        validation_report["state_dict_compatibility"] = False
        raise AgentForgeCompatibilityError("Models have incompatible state dict keys")

    # Check tensor shapes
    for key in key_sets[0]:
        shapes = [sd[key].shape for sd in state_dicts]
        if not all(shape == shapes[0] for shape in shapes[1:]):
            validation_report["state_dict_compatibility"] = False
            raise AgentForgeCompatibilityError(f"Incompatible shapes for parameter {key}")

    # Basic forward pass test
    try:
        test_input = torch.randint(0, 32000, (1, 32))

        for model in models:
            model.eval()
            with torch.no_grad():
                outputs = model(test_input, return_dict=True)

                # Check required outputs
                required_keys = ["logits", "act_steps", "halt_probs", "memory_stats"]
                for key in required_keys:
                    if key not in outputs:
                        raise AgentForgeCompatibilityError(f"Missing required output: {key}")

    except Exception as e:
        validation_report["architecture_validation"] = False
        raise AgentForgeCompatibilityError(f"Forward pass validation failed: {e}")

    logger.info("Agent Forge compatibility validation passed")
    return validation_report


def save_agent_forge_models(
    models: list[CognateProductionModel], output_dir: str | Path, include_metadata: bool = True
) -> dict[str, str]:
    """
    Save models in Agent Forge expected format.

    Args:
        models: List of models to save
        output_dir: Directory to save models
        include_metadata: Whether to include Agent Forge metadata

    Returns:
        Dictionary mapping model names to saved paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    for i, model in enumerate(models):
        model_name = getattr(model, "variant_name", f"cognate-model-{i+1}")
        model_dir = output_dir / model_name

        # Save model
        model.save_pretrained(str(model_dir))
        saved_paths[model_name] = str(model_dir)

        if include_metadata:
            # Add Agent Forge specific metadata
            metadata = {
                "agent_forge": {
                    "phase": "cognate",
                    "model_type": "cognate_production",
                    "version": "1.0.0",
                    "parameter_count": model.count_parameters(),
                    "evomerge_compatible": True,
                    "architecture": {
                        "d_model": model.config.d_model,
                        "n_layers": model.config.n_layers,
                        "n_heads": model.config.n_heads,
                        "vocab_size": model.config.vocab_size,
                    },
                    "features": [
                        "act_halting",
                        "memory_system",
                        "train_many_infer_few",
                        "rope_positional_encoding",
                        "swiglu_activation",
                    ],
                },
                "performance": {
                    "target_parameters": 25_069_534,
                    "actual_parameters": model.count_parameters(),
                    "memory_capacity": model.config.mem_capacity,
                    "max_sequence_length": model.config.max_seq_len,
                },
                "training": {
                    "train_max_steps": model.config.train_max_steps,
                    "infer_max_steps": model.config.infer_max_steps,
                    "act_threshold": model.config.act_threshold,
                    "supports_grokfast": True,
                },
            }

            with open(model_dir / "agent_forge_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

    # Create summary file
    summary = {
        "models": list(saved_paths.keys()),
        "total_models": len(models),
        "total_parameters": sum(model.count_parameters() for model in models),
        "created_by": "Canonical Cognate Implementation v1.0.0",
        "compatible_with": [
            "Agent Forge Pipeline",
            "EvoMerge",
            "HuggingFace Transformers",
        ],
    }

    with open(output_dir / "models_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved {len(models)} models to {output_dir}")
    return saved_paths


def load_agent_forge_models(model_dir: str | Path) -> list[CognateProductionModel]:
    """
    Load models saved in Agent Forge format.

    Args:
        model_dir: Directory containing saved models

    Returns:
        List of loaded models
    """
    model_dir = Path(model_dir)

    # Read summary
    summary_path = model_dir / "models_summary.json"
    if not summary_path.exists():
        raise AgentForgeCompatibilityError(f"No models summary found in {model_dir}")

    with open(summary_path) as f:
        summary = json.load(f)

    models = []
    for model_name in summary["models"]:
        model_path = model_dir / model_name
        if not model_path.exists():
            raise AgentForgeCompatibilityError(f"Model directory not found: {model_path}")

        model = CognateProductionModel.from_pretrained(str(model_path))
        model.variant_name = model_name
        models.append(model)

    logger.info(f"Loaded {len(models)} models from {model_dir}")
    return models


# Main Agent Forge integration function
def create_agent_forge_cognate_models(
    output_dir: str | Path | None = None, validate_compatibility: bool = True, **config_overrides
) -> dict[str, Any]:
    """
    Create and optionally save Cognate models for Agent Forge pipeline.

    This is the main entry point used by Agent Forge to create Cognate models.

    Args:
        output_dir: Directory to save models (optional)
        validate_compatibility: Whether to run compatibility validation
        **config_overrides: Configuration overrides

    Returns:
        Dictionary with models, validation report, and metadata
    """
    logger.info("Creating Cognate models for Agent Forge pipeline...")

    # Create three models
    models = create_three_production_cognate_models(**config_overrides)

    # Validate compatibility if requested
    validation_report = None
    if validate_compatibility:
        validation_report = validate_agent_forge_compatibility(models)

    # Save models if output directory provided
    saved_paths = None
    if output_dir is not None:
        saved_paths = save_agent_forge_models(models, output_dir)

    # Prepare result
    result = {
        "models": models,
        "model_count": len(models),
        "parameter_counts": [model.count_parameters() for model in models],
        "validation_report": validation_report,
        "saved_paths": saved_paths,
        "success": True,
    }

    logger.info(f"Successfully created {len(models)} Cognate models for Agent Forge")
    return result


# Compatibility with existing scattered implementations
def create_three_cognate_models(**kwargs) -> list[CognateProductionModel]:
    """
    Legacy compatibility function for scattered implementations.

    This function provides backwards compatibility with any existing code
    that calls create_three_cognate_models.
    """
    logger.info("Legacy create_three_cognate_models called - redirecting to production implementation")
    return create_three_production_cognate_models(**kwargs)


if __name__ == "__main__":
    # Test Agent Forge integration
    logging.basicConfig(level=logging.INFO)

    print("Testing Agent Forge Integration...")

    # Create models
    result = create_agent_forge_cognate_models(output_dir="./test_agent_forge_models", validate_compatibility=True)

    print("\n=== Agent Forge Integration Test Results ===")
    print(f"Models created: {result['model_count']}")
    print(f"Parameter counts: {result['parameter_counts']}")
    print(f"Validation passed: {result['validation_report']['parameter_consistency']}")
    print(f"Saved to: {result['saved_paths']}")

    # Test loading
    if result["saved_paths"]:
        loaded_models = load_agent_forge_models("./test_agent_forge_models")
        print(f"Successfully loaded {len(loaded_models)} models")

    print("\nAgent Forge Integration test completed successfully!")
