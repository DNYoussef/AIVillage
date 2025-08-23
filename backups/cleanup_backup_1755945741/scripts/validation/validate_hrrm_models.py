#!/usr/bin/env python3
"""Validation script for HRRM models."""

import json
import logging
import os
import sys

import torch

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath("."))

from packages.hrrm.memory.model import MemoryAsContextTiny, MemoryConfig
from packages.hrrm.planner.heads import PlannerConfig
from packages.hrrm.planner.model import HRMPlanner
from packages.hrrm.reasoner.model import HRMReasoner, ReasonerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_and_validate_model(model_path, config_path, model_class, config_class, model_name):
    """Load and validate a single model."""
    logger.info(f"Validating {model_name}...")

    # Load config
    with open(config_path) as f:
        config_dict = json.load(f)

    # Create config object
    if model_name == "HRMPlanner":
        # Handle the control_tokens list properly
        config = config_class(**{k: v for k, v in config_dict.items() if k != "control_tokens"})
        config.control_tokens = config_dict["control_tokens"]
    else:
        config = config_class(**config_dict)

    # Create model
    model = model_class(config)

    # Load weights
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # Count parameters
    param_count = count_parameters(model)
    param_count_m = param_count / 1e6

    # Test forward pass
    model.eval()
    with torch.no_grad():
        test_input = torch.randint(0, min(1000, config.vocab_size - 1), (2, 10))

        try:
            if model_name == "MemoryAsContextTiny":
                output = model(test_input)
                output_shape = output.logits.shape
            else:
                output = model(test_input)
                output_shape = output.logits.shape

            logger.info(f"  ✓ {model_name} loaded successfully")
            logger.info(f"  ✓ Parameters: {param_count:,} ({param_count_m:.1f}M)")
            logger.info(f"  ✓ Forward pass successful, output shape: {output_shape}")

            return {
                "status": "success",
                "param_count": param_count,
                "param_count_m": param_count_m,
                "output_shape": list(output_shape),
                "config": config_dict,
            }

        except Exception as e:
            logger.error(f"  ✗ Forward pass failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "param_count": param_count,
                "param_count_m": param_count_m,
                "config": config_dict,
            }


def main():
    """Main validation function."""
    logger.info("🤖 HRRM Model Validation Starting...")
    logger.info("=" * 60)

    models_to_validate = [
        {
            "name": "HRMPlanner",
            "model_path": "artifacts/checkpoints/planner/model.pt",
            "config_path": "artifacts/checkpoints/planner/config.json",
            "model_class": HRMPlanner,
            "config_class": PlannerConfig,
        },
        {
            "name": "HRMReasoner",
            "model_path": "artifacts/checkpoints/reasoner/model.pt",
            "config_path": "artifacts/checkpoints/reasoner/config.json",
            "model_class": HRMReasoner,
            "config_class": ReasonerConfig,
        },
        {
            "name": "MemoryAsContextTiny",
            "model_path": "artifacts/checkpoints/memory/model.pt",
            "config_path": "artifacts/checkpoints/memory/config.json",
            "model_class": MemoryAsContextTiny,
            "config_class": MemoryConfig,
        },
    ]

    validation_results = {}
    total_params = 0
    successful_models = 0

    for model_info in models_to_validate:
        result = load_and_validate_model(
            model_info["model_path"],
            model_info["config_path"],
            model_info["model_class"],
            model_info["config_class"],
            model_info["name"],
        )

        validation_results[model_info["name"]] = result

        if result["status"] == "success":
            successful_models += 1
            total_params += result["param_count"]

        logger.info("")

    # Generate final report
    logger.info("=" * 60)
    logger.info("🏆 HRRM VALIDATION SUMMARY")
    logger.info("=" * 60)

    logger.info(f"Models Successfully Validated: {successful_models}/3")
    logger.info(f"Total Parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    if successful_models == 3:
        logger.info("✅ ALL HRRM MODELS VALIDATION PASSED!")
        logger.info("")
        logger.info("Individual Model Details:")
        for name, result in validation_results.items():
            if result["status"] == "success":
                logger.info(f"  • {name}: {result['param_count_m']:.1f}M parameters")

        logger.info("")
        logger.info("🚀 HRRM Bootstrap System Ready for Agent Forge Integration!")
        logger.info("Next steps:")
        logger.info("  1. Export to HuggingFace format")
        logger.info("  2. Integrate with EvoMerge for 30x speedup")
        logger.info("  3. Begin Agent Forge pipeline training")

        # Create validation report
        validation_summary = {
            "validation_status": "SUCCESS",
            "models_validated": successful_models,
            "total_models": 3,
            "total_parameters": total_params,
            "total_parameters_m": total_params / 1e6,
            "models": validation_results,
            "ready_for_production": True,
        }

    else:
        logger.info(f"❌ {3 - successful_models} MODEL(S) FAILED VALIDATION")
        validation_summary = {
            "validation_status": "FAILED",
            "models_validated": successful_models,
            "total_models": 3,
            "models": validation_results,
            "ready_for_production": False,
        }

    # Save validation report
    report_path = "artifacts/hrrm_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(validation_summary, f, indent=2)

    logger.info(f"📋 Validation report saved to {report_path}")


if __name__ == "__main__":
    main()
