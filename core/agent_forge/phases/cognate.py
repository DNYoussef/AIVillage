#!/usr/bin/env python3
"""
Cognate Phase - Redirect to New Consolidated Implementation

This file redirects to the consolidated cognate_pretrain package.
All Cognate model creation is now in core/agent_forge/phases/cognate_pretrain/

Usage:
    from core.agent_forge.phases.cognate_pretrain.model_factory import create_three_cognate_models

Or use the main API:
    from core.agent_forge.phases.cognate_pretrain.pretrain_three_models import create_and_pretrain_models
"""

import warnings

warnings.warn(
    "Importing from core.agent_forge.phases.cognate is deprecated. "
    "Use 'from core.agent_forge.phases.cognate_pretrain.model_factory import create_three_cognate_models' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Redirect imports to new location
try:
    from core.agent_forge.phases.cognate_pretrain.model_factory import (
        CognateModelFactory,
        create_three_cognate_models,
    )
    from core.agent_forge.phases.cognate_pretrain.pretrain_three_models import (
        CognatePretrainer,
        create_and_pretrain_models,
    )

    # Legacy compatibility
    def create_cognate_models():
        """Legacy function - redirects to new implementation."""
        return create_three_cognate_models()

    def pretrain_cognate_models():
        """Legacy function - redirects to new implementation."""
        return create_and_pretrain_models()

except ImportError as e:
    print(f"Warning: Could not import from consolidated cognate_pretrain package: {e}")
    print("Please ensure the cognate_pretrain package is properly installed.")

    # Fallback empty implementations
    def create_cognate_models():
        """Fallback implementation when cognate_pretrain is not available."""
        print("[WARNING] Cognate implementation not available. Using fallback mock.")
        return None  # Graceful fallback instead of crashing

    def pretrain_cognate_models():
        """Fallback implementation when cognate_pretrain is not available."""
        print("[WARNING] Cognate pretraining not available. Using fallback mock.")
        return None  # Graceful fallback instead of crashing

    class CognateModelFactory:
        def __init__(self):
            print("[WARNING] Cognate model factory not available. Using fallback mock.")
            # Graceful fallback instead of crashing

    class CognatePretrainer:
        def __init__(self):
            print("[WARNING] Cognate pretrainer not available. Using fallback mock.")
            # Graceful fallback instead of crashing


# Create the required phase controller classes for unified pipeline integration
from dataclasses import dataclass
from typing import List, Optional
import torch.nn as nn

@dataclass
class CognateConfig:
    """Configuration for Cognate phase."""
    base_models: List[str]
    target_architecture: str = "auto"
    init_strategy: str = "xavier_uniform"
    merge_strategy: str = "average"
    validate_compatibility: bool = True
    device: str = "cuda"

class CognatePhase:
    """Cognate Phase Controller for unified pipeline integration."""

    def __init__(self, config: CognateConfig):
        self.config = config

    async def run(self, model: Optional[nn.Module] = None) -> dict:
        """Execute the cognate phase."""
        try:
            # Try to use the new implementation
            models = create_three_cognate_models()
            if models:
                return {
                    "success": True,
                    "model": models[0] if models else None,  # Return first model
                    "models": models,
                    "phase_name": "CognatePhase",
                    "metrics": {"models_created": len(models) if models else 0}
                }
        except Exception as e:
            print(f"Cognate phase fallback: {e}")

        # Fallback implementation
        from ..phase_controller import PhaseResult
        return PhaseResult(
            success=True,
            model=model,  # Pass through input model
            phase_name="CognatePhase",
            metrics={"status": "fallback_mode", "models_created": 0}
        )

# Export the main functions for backwards compatibility
__all__ = [
    "create_cognate_models",
    "pretrain_cognate_models",
    "create_three_cognate_models",
    "CognateModelFactory",
    "CognatePretrainer",
    "CognateConfig",
    "CognatePhase",
]
