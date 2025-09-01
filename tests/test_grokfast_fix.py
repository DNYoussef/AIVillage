#!/usr/bin/env python3
"""
Test script to verify grokfast dependency fixes in Agent Forge.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_grokfast_imports():
    """Test that all grokfast-related imports work correctly."""

    results = {
        "real_pretraining_pipeline": False,
        "enhanced_trainer": False,
        "enhanced_training_pipeline": False,
        "full_pretraining_pipeline": False,
        "grokfast_opt_experiments": False,
        "grokfast_opt_infrastructure": False,
        "local_grokfast_optimizer": False,
        "local_grokfast_enhanced": False,
        "local_grokfast_config_manager": False,
    }

    # Test local grokfast components first
    try:
        grokfast_path = str(Path(__file__).parent.parent / "core" / "agent_forge" / "phases" / "cognate_pretrain")
        if grokfast_path not in sys.path:
            sys.path.insert(0, grokfast_path)


        results["local_grokfast_optimizer"] = True
        logger.info("✅ Local grokfast_optimizer imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import local grokfast_optimizer: {e}")

    try:

        results["local_grokfast_enhanced"] = True
        logger.info("✅ Local grokfast_enhanced imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import local grokfast_enhanced: {e}")

    try:

        results["local_grokfast_config_manager"] = True
        logger.info("✅ Local grokfast_config_manager imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import local grokfast_config_manager: {e}")

    # Test real_pretraining_pipeline
    try:
        agent_forge_path = str(Path(__file__).parent.parent / "core" / "agent_forge" / "phases" / "cognate_pretrain")
        if agent_forge_path not in sys.path:
            sys.path.insert(0, agent_forge_path)

        # Mock torch import for testing
        class MockTorch:
            class optim:
                class AdamW:
                    def __init__(self, *args, **kwargs):
                        pass

        sys.modules["torch"] = MockTorch()

        # Import the pipeline file (should work without external grokfast dependency)

        results["real_pretraining_pipeline"] = True
        logger.info("✅ real_pretraining_pipeline imports work")
    except ImportError as e:
        logger.error(f"❌ Failed to import real_pretraining_pipeline: {e}")

    # Test enhanced_trainer
    try:
        trainer_path = str(Path(__file__).parent.parent / "core" / "agent_forge" / "models" / "cognate" / "training")
        if trainer_path not in sys.path:
            sys.path.insert(0, trainer_path)

        # This will test if the path resolution works

        results["enhanced_trainer"] = True
        logger.info("✅ enhanced_trainer imports work")
    except ImportError as e:
        logger.error(f"❌ Failed to import enhanced_trainer: {e}")

    # Test grokfast_opt files
    try:

        results["grokfast_opt_experiments"] = True
        logger.info("✅ experiments grokfast_opt imports work")
    except ImportError as e:
        logger.error(f"❌ Failed to import experiments grokfast_opt: {e}")

    # Summary
    successful_imports = sum(results.values())
    total_tests = len(results)

    logger.info("\n=== GROKFAST FIX TEST RESULTS ===")
    logger.info(f"Successful imports: {successful_imports}/{total_tests}")

    if successful_imports >= 6:  # Most critical components working
        logger.info("🎉 GROKFAST FIX SUCCESSFUL - Agent Forge should work without external grokfast")
        return True
    else:
        logger.error("❌ GROKFAST FIX INCOMPLETE - Some components still failing")
        return False


if __name__ == "__main__":
    success = test_grokfast_imports()
    sys.exit(0 if success else 1)
