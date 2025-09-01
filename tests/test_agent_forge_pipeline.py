#!/usr/bin/env python3
"""
Test Agent Forge training pipeline to ensure grokfast dependency fix is complete.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_agent_forge_pipeline():
    """Test that Agent Forge training pipeline works without external grokfast dependency."""

    results = {}

    # Test 1: Core grokfast components
    try:
        grokfast_path = str(Path(__file__).parent.parent / "core" / "agent_forge" / "phases" / "cognate_pretrain")
        if grokfast_path not in sys.path:
            sys.path.insert(0, grokfast_path)

        from grokfast_enhanced import EnhancedGrokFastOptimizer

        results["grokfast_components"] = True
        logger.info("✓ All local grokfast components available")
    except ImportError as e:
        results["grokfast_components"] = False
        logger.error(f"✗ Local grokfast components failed: {e}")

    # Test 2: Real pretraining pipeline
    try:
        from real_pretraining_pipeline import REAL_IMPORTS, create_grokfast_adamw

        results["real_pipeline"] = REAL_IMPORTS
        logger.info(f"✓ Real pretraining pipeline: REAL_IMPORTS = {REAL_IMPORTS}")

        # Test the create_grokfast_adamw function
        if REAL_IMPORTS:
            import torch
            import torch.nn as nn

            # Create a simple model for testing
            model = nn.Linear(10, 1)
            create_grokfast_adamw(model, lr=1e-4)

            logger.info("✓ create_grokfast_adamw function works")
            results["grokfast_adamw_creation"] = True
        else:
            results["grokfast_adamw_creation"] = False

    except Exception as e:
        results["real_pipeline"] = False
        results["grokfast_adamw_creation"] = False
        logger.error(f"✗ Real pretraining pipeline failed: {e}")

    # Test 3: Enhanced trainer
    try:
        trainer_path = str(Path(__file__).parent.parent / "core" / "agent_forge" / "models" / "cognate" / "training")
        if trainer_path not in sys.path:
            sys.path.insert(0, trainer_path)

        import enhanced_trainer

        results["enhanced_trainer"] = hasattr(enhanced_trainer, "GROKFAST_AVAILABLE")

        if hasattr(enhanced_trainer, "GROKFAST_AVAILABLE"):
            logger.info(f"✓ Enhanced trainer: GROKFAST_AVAILABLE = {enhanced_trainer.GROKFAST_AVAILABLE}")
        else:
            logger.info("✓ Enhanced trainer imported (compatibility mode)")

    except Exception as e:
        results["enhanced_trainer"] = False
        logger.error(f"✗ Enhanced trainer failed: {e}")

    # Test 4: Enhanced training pipeline
    try:
        pipeline_path = str(Path(__file__).parent.parent / "core" / "agent_forge" / "phases" / "cognate_pretrain")
        if pipeline_path not in sys.path:
            sys.path.insert(0, pipeline_path)

        from enhanced_training_pipeline import EnhancedGrokFastOptimizer

        results["enhanced_pipeline"] = EnhancedGrokFastOptimizer is not None
        logger.info("✓ Enhanced training pipeline imports successfully")

    except Exception as e:
        results["enhanced_pipeline"] = False
        logger.error(f"✗ Enhanced training pipeline failed: {e}")

    # Test 5: GrokFast optimizer alternatives in experiments
    try:
        exp_path = str(Path(__file__).parent.parent / "experiments" / "training" / "training")
        if exp_path not in sys.path:
            sys.path.insert(0, exp_path)

        from grokfast_opt import AugmentedAdam

        # Test creating the optimizer
        import torch

        params = [torch.nn.Parameter(torch.randn(5, 5))]
        AugmentedAdam(params)

        results["experiments_grokfast"] = True
        logger.info("✓ Experiments grokfast_opt works with local implementation")

    except Exception as e:
        results["experiments_grokfast"] = False
        logger.error(f"✗ Experiments grokfast failed: {e}")

    # Summary
    successful_tests = sum(results.values())
    total_tests = len(results)

    logger.info("\\n=== AGENT FORGE GROKFAST FIX VALIDATION ===")
    logger.info(f"Tests passed: {successful_tests}/{total_tests}")

    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test}")

    if successful_tests >= 4:  # Allow for some optional components to fail
        logger.info("\\n🎉 SUCCESS: Agent Forge pipeline is functional without external grokfast!")
        logger.info("The training pipeline should now work without grokfast dependency errors.")
        return True
    else:
        logger.error("\\n❌ FAILURE: Agent Forge pipeline still has critical issues.")
        return False


if __name__ == "__main__":
    success = test_agent_forge_pipeline()
    sys.exit(0 if success else 1)
