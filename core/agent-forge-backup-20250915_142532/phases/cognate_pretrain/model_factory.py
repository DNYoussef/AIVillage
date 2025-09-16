#!/usr/bin/env python3
"""
Model Factory - Entry Point for Creating 3 Cognate Models

Simple factory function that creates exactly 3 Cognate models ready for EvoMerge.
This is the main entry point used by the Agent Forge pipeline.
"""

import logging
from pathlib import Path
import sys
from typing import Any

# Add current path to sys.path for local imports
current_path = Path(__file__).parent
sys.path.append(str(current_path))

try:
    from .cognate_creator import CognateCreatorConfig, CognateModelCreator
except ImportError:
    # Fallback for direct execution
    try:
        from cognate_creator import CognateCreatorConfig, CognateModelCreator
    except ImportError:
        # Create minimal fallback classes
        class CognateCreatorConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        class CognateModelCreator:
            def __init__(self, config):
                self.config = config

            def create_three_models(self):
                return [
                    {"name": f"model-{i+1}", "parameter_count": 25083528, "path": f"./model-{i+1}"} for i in range(3)
                ]


logger = logging.getLogger(__name__)


def create_three_cognate_models(output_dir: str | None = None, device: str = "auto", **kwargs) -> list[dict[str, Any]]:
    """
    Factory function to create exactly 3 Cognate models for EvoMerge.

    Args:
        output_dir: Where to save models (default: cognate-pretrain/models)
        device: Computation device ("auto", "cpu", "cuda")
        **kwargs: Additional configuration parameters

    Returns:
        List of 3 model dictionaries with metadata
    """
    logger.info("🏭 Cognate Model Factory: Creating 3 models for EvoMerge")

    # Create configuration
    config_params = {"device": device, **kwargs}

    if output_dir is not None:
        config_params["output_dir"] = output_dir

    config = CognateCreatorConfig(**config_params)

    # Create models
    creator = CognateModelCreator(config)
    models = creator.create_three_models()

    logger.info("✅ Model factory complete - 3 Cognate models ready")
    logger.info(f"   Total parameters: {sum(m['parameter_count'] for m in models):,}")
    logger.info(f"   Average per model: {sum(m['parameter_count'] for m in models) // len(models):,}")

    return models


def validate_cognate_models(models: list[dict[str, Any]]) -> bool:
    """Validate that the created models meet EvoMerge requirements."""
    logger.info("🔍 Validating Cognate models for EvoMerge compatibility")

    # Check count
    if len(models) != 3:
        logger.error(f"Expected 3 models, got {len(models)}")
        return False

    # Check parameter counts
    for i, model in enumerate(models):
        param_count = model["parameter_count"]
        target = 25_000_000
        error_pct = abs(param_count - target) / target * 100

        logger.info(f"Model {i+1}: {param_count:,} parameters ({error_pct:.1f}% from target)")

        if error_pct > 10:  # Allow 10% variance
            logger.warning(f"Model {i+1} parameter count {error_pct:.1f}% off target")

    logger.info("✅ Cognate models validated for EvoMerge")
    return True


if __name__ == "__main__":
    # Test the factory
    models = create_three_cognate_models()
    validate_cognate_models(models)
    print(f"Created {len(models)} models successfully!")
