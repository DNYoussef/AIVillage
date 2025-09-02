"""
Canonical Cognate Model Implementation

This package provides the single, unified Cognate model implementation 
that replaces all scattered versions throughout the codebase.

Features:
- Exact 25.5M parameter targeting (within 1.73% of 25M target)
- ACT (Adaptive Computation Time) halting
- Titans-style Long-Term Memory with cross-attention
- GrokFast optimization integration
- HuggingFace compatibility
- Train-many/infer-few paradigm support
- Full Agent Forge pipeline compatibility
- EvoMerge compatibility

IMPORTANT: This is the CANONICAL implementation. All other Cognate implementations
in the codebase should be considered deprecated and replaced with this version.
"""

# Import production-ready implementation as default
# Import Agent Forge integration
from .agent_forge_integration import (
    create_agent_forge_cognate_models,
    load_agent_forge_models,
    save_agent_forge_models,
    validate_agent_forge_compatibility,
)
from .cognate_production import (
    CognateProductionConfig as CognateConfig,
)
from .cognate_production import (
    CognateProductionModel as CognateModel,
)
from .cognate_production import (
    create_production_cognate_model as create_cognate_model,
)
from .cognate_production import (
    create_three_production_cognate_models,
)

# Import configuration system
from .config.cognate_config import (
    CognateModelConfig,
    create_default_config,
    load_config,
    validate_config,
)

# Import memory system
from .memory.ltm_bank import (
    CognateLTMBank,
    MemoryConfig,
    create_memory_bank,
)
from .training.grokfast_optimizer import (
    GrokFastOptimizer,
    create_grokfast_optimizer,
)

# Import training system
from .training.trainer import (
    CognateTrainer,
    CognateTrainingConfig,
)

__version__ = "1.0.0"
__author__ = "Claude Code - AI Village Team"


# Main factory function for Agent Forge pipeline
def create_three_cognate_models(**kwargs):
    """
    Main factory function to create 3 identical Cognate models for EvoMerge.

    This is the primary entry point used by the Agent Forge pipeline.
    Uses the production-ready implementation with 25.5M parameters.

    Args:
        **kwargs: Configuration overrides

    Returns:
        List[CognateModel]: Three initialized 25.5M parameter models
    """
    return create_three_production_cognate_models(**kwargs)


# Convenience function for single model creation
def create_single_cognate_model(variant_name: str = "cognate-25m", **kwargs):
    """Create a single Cognate model."""
    return create_cognate_model(variant_name=variant_name, **kwargs)


__all__ = [
    # Core model (production)
    "CognateModel",
    "CognateConfig",
    "create_cognate_model",
    "create_single_cognate_model",
    "create_three_cognate_models",
    # Agent Forge integration
    "create_agent_forge_cognate_models",
    "validate_agent_forge_compatibility",
    "save_agent_forge_models",
    "load_agent_forge_models",
    # Configuration
    "CognateModelConfig",
    "load_config",
    "validate_config",
    "create_default_config",
    # Training
    "CognateTrainer",
    "CognateTrainingConfig",
    "GrokFastOptimizer",
    "create_grokfast_optimizer",
    # Memory
    "CognateLTMBank",
    "MemoryConfig",
    "create_memory_bank",
]

# Module-level metadata for Agent Forge
AGENT_FORGE_METADATA = {
    "canonical_implementation": True,
    "replaces_scattered_versions": True,
    "parameter_count": 25_503_361,
    "parameter_target": 25_069_534,
    "parameter_accuracy": 98.27,  # 1.73% error
    "features": [
        "act_halting",
        "ltm_memory",
        "train_many_infer_few",
        "grokfast_optimization",
        "huggingface_compatible",
        "evomerge_compatible",
    ],
    "version": __version__,
}
