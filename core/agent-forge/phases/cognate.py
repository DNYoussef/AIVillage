#!/usr/bin/env python3
"""
Cognate Phase - REDIRECT to New Cognate Pre-training Package

This file now redirects to the consolidated cognate-pretrain package.
All Cognate model creation has been reorganized into a dedicated folder structure.

NEW LOCATION: core/agent-forge/phases/cognate-pretrain/
"""

import logging
from typing import Any
import warnings

# Issue deprecation warning
warnings.warn(
    "Direct import from cognate.py is deprecated. "
    "Use 'from core.agent_forge.phases.cognate_pretrain import create_three_cognate_models' instead.",
    DeprecationWarning,
    stacklevel=2
)

logger = logging.getLogger(__name__)

# Import from new organized structure
try:
    from .cognate_pretrain import (
        CognateCreatorConfig,
        CognateModelCreator,
        create_three_cognate_models,
        validate_cognate_models,
    )
    NEW_STRUCTURE_AVAILABLE = True
    logger.info("✅ Using new cognate-pretrain package structure")
except ImportError:
    NEW_STRUCTURE_AVAILABLE = False
    logger.error("❌ New cognate-pretrain structure not available")


def create_cognate_models(num_models: int = 3, **kwargs) -> list[dict[str, Any]]:
    """
    DEPRECATED: Create Cognate models (redirects to new structure).
    
    Use create_three_cognate_models() from cognate-pretrain package instead.
    """
    if not NEW_STRUCTURE_AVAILABLE:
        raise ImportError("New cognate-pretrain package not available")
    
    logger.warning("create_cognate_models() is deprecated. Use create_three_cognate_models() instead.")
    
    if num_models != 3:
        logger.warning(f"Cognate system requires exactly 3 models, got {num_models}. Using 3.")
    
    return create_three_cognate_models(**kwargs)


class CognatePhase:
    """DEPRECATED: Use cognate-pretrain package directly."""
    
    def __init__(self, **kwargs):
        logger.warning("CognatePhase class is deprecated. Use cognate-pretrain package directly.")
        self.config = kwargs
    
    def execute(self) -> dict[str, Any]:
        """Execute cognate phase (redirects to new structure)."""
        if not NEW_STRUCTURE_AVAILABLE:
            raise ImportError("New cognate-pretrain package not available")
        
        logger.info("🔄 Redirecting to new cognate-pretrain structure...")
        models = create_three_cognate_models(**self.config)
        
        return {
            "phase": "cognate",
            "status": "complete",
            "models_created": len(models),
            "models": models,
            "next_phase": "evomerge",
            "message": "Successfully created 3 Cognate models using new structure"
        }


# Legacy compatibility exports
__all__ = [
    'create_cognate_models',  # DEPRECATED
    'CognatePhase',          # DEPRECATED 
    'create_three_cognate_models',  # NEW
    'validate_cognate_models',      # NEW
    'CognateModelCreator',          # NEW
    'CognateCreatorConfig'          # NEW
]

if __name__ == "__main__":
    print("=" * 60)
    print("COGNATE PHASE - PACKAGE REORGANIZATION")
    print("=" * 60)
    print()
    print("❗ This file is now a redirect to the new structure:")
    print("   📁 core/agent-forge/phases/cognate-pretrain/")
    print()
    print("✅ NEW USAGE:")
    print("   from core.agent_forge.phases.cognate_pretrain import create_three_cognate_models")
    print("   models = create_three_cognate_models()")
    print()
    print("❌ DEPRECATED USAGE:")  
    print("   from core.agent_forge.phases.cognate import create_cognate_models")
    print()
    print("📚 See cognate-pretrain/README.md for full documentation")
    print("=" * 60)
    
    if NEW_STRUCTURE_AVAILABLE:
        print("✅ New structure is available and working")
        try:
            models = create_three_cognate_models()
            print(f"✅ Successfully created {len(models)} models via redirect")
        except Exception as e:
            print(f"❌ Error testing new structure: {e}")
    else:
        print("❌ New structure not available - check imports")