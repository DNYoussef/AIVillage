#!/usr/bin/env python3
"""
Test Real Cognate Imports and Infrastructure
"""

import sys
import logging
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "core" / "agent_forge"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_cognate_imports():
    """Test if CognateRefiner imports work now"""
    logger.info("Testing Cognate imports...")

    try:
        from phases.cognate_pretrain.cognate_creator import COGNATE_AVAILABLE, CognateModelCreator, CognateCreatorConfig

        if COGNATE_AVAILABLE:
            logger.info("✅ SUCCESS: CognateRefiner imports working!")
            logger.info("✅ Real pretraining will be used")
            return True
        else:
            logger.warning("⚠️  WARNING: CognateRefiner not available, will use mock models")
            return False

    except Exception as e:
        logger.error(f"❌ FAILED: Import error: {e}")
        return False

def test_evomerge_imports():
    """Test if EvoMerge imports work"""
    logger.info("Testing EvoMerge imports...")

    try:
        from phases.evomerge import EvoMergePhase, EvoMergeConfig
        logger.info("✅ SUCCESS: EvoMerge imports working!")
        return True
    except Exception as e:
        logger.error(f"❌ FAILED: EvoMerge import error: {e}")
        return False

def main():
    logger.info("=== Testing Real Cognate + EvoMerge Infrastructure ===")

    cognate_ok = test_cognate_imports()
    evomerge_ok = test_evomerge_imports()

    if cognate_ok and evomerge_ok:
        logger.info("🎉 ALL IMPORTS WORKING - Ready for real training!")
        return True
    else:
        logger.error("💥 IMPORT ISSUES - Check paths and dependencies")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)