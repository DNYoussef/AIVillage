#!/usr/bin/env python3
"""
Cognate Phase - Redirect to New Consolidated Implementation

This file redirects to the consolidated cognate_pretrain package.
All Cognate model creation is now in core/agent-forge/phases/cognate_pretrain/

Usage:
    from core.agent_forge.phases.cognate_pretrain.model_factory import create_three_cognate_models

Or use the main API:
    from core.agent_forge.phases.cognate_pretrain.pretrain_three_models import create_and_pretrain_models
"""

import warnings
from pathlib import Path

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
        validate_models,
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
        raise NotImplementedError("Cognate implementation not available. Check cognate_pretrain package.")

    def pretrain_cognate_models():
        raise NotImplementedError("Cognate implementation not available. Check cognate_pretrain package.")

    class CognateModelFactory:
        def __init__(self):
            raise NotImplementedError("Cognate implementation not available.")

    class CognatePretrainer:
        def __init__(self):
            raise NotImplementedError("Cognate implementation not available.")


# Export the main functions for backwards compatibility
__all__ = [
    "create_cognate_models",
    "pretrain_cognate_models",
    "create_three_cognate_models",
    "CognateModelFactory",
    "CognatePretrainer",
]
