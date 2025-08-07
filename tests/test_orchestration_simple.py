#!/usr/bin/env python3
"""
Simple test to verify the orchestration system is working.
"""

import asyncio
import logging
import os
import sys

from agent_forge.orchestration.curriculum_integration import MultiModelOrchestrator
from agent_forge.training.magi_specialization import MagiConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_orchestration():
    """Test basic orchestration functionality."""

    # Check API key
    api_key_available = bool(os.getenv("OPENROUTER_API_KEY"))
    logger.info(f"OpenRouter API key available: {api_key_available}")

    # Create minimal config
    config = MagiConfig(curriculum_levels=1, questions_per_level=1, total_questions=1)

    # Test orchestrator
    orchestrator = MultiModelOrchestrator(config, enable_openrouter=api_key_available)

    try:
        # Test question generation
        logger.info("Testing question generation...")
        questions = orchestrator.question_generator.generate_curriculum_questions()
        logger.info(f"Generated {len(questions)} questions")

        if questions:
            q = questions[0]
            logger.info(
                f"Sample question - Domain: {q.domain}, Difficulty: {q.difficulty}"
            )
            logger.info(f"Text: {q.text[:100]}...")

        # Test cost summary
        cost_summary = orchestrator.get_cost_summary()
        logger.info(f"Cost tracking enabled: {cost_summary['enabled']}")

        logger.info("SUCCESS: Basic orchestration test passed!")
        return True

    except Exception as e:
        logger.exception(f"Test failed: {e}")
        return False

    finally:
        await orchestrator.close()


if __name__ == "__main__":
    success = asyncio.run(test_basic_orchestration())
    if success:
        print("ORCHESTRATION TEST: PASSED")
    else:
        print("ORCHESTRATION TEST: FAILED")
    sys.exit(0 if success else 1)
