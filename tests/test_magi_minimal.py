#!/usr/bin/env python3
"""
Minimal Magi Specialization Test

This script tests the core Magi specialization functionality
with minimal parameters to verify system stability.
"""

import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_magi_minimal():
    """Test minimal Magi functionality"""
    logger.info("🧙 Starting Minimal Magi Specialization Test")

    try:
        # Import components
        from agent_forge.training.magi_specialization import (
            MagiConfig,
            MagiSpecializationPipeline,
        )

        # Create minimal configuration
        config = MagiConfig(
            curriculum_levels=2,
            questions_per_level=10,
            total_questions=20,
            enable_self_modification=False,  # Disable self-mod for stability
            enable_geometric_awareness=True,
            output_dir="D:/AgentForge/magi_production/minimal_test",
        )

        logger.info(
            f"Configuration: {config.curriculum_levels} levels, {config.questions_per_level} questions per level"
        )

        # Initialize pipeline
        pipeline = MagiSpecializationPipeline(config)

        # Run the complete minimal training pipeline
        logger.info("Starting minimal Magi specialization run...")
        result = await pipeline.run_magi_specialization()
        logger.info(f"Magi specialization completed with success: {result.get('success', False)}")

        success = result.get("success", False)
        if success:
            logger.info("SUCCESS: Minimal Magi test completed successfully!")

        return success

    except Exception as e:
        logger.exception(f"FAILED: Minimal Magi test failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_magi_minimal())
    if success:
        print("SUCCESS: Minimal Magi test completed")
    else:
        print("FAILED: Minimal Magi test failed")
