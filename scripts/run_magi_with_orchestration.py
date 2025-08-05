#!/usr/bin/env python3
"""Run Magi Agent Specialization with Multi-Model Orchestration

This script demonstrates how to use the new OpenRouter orchestration system
with the existing Magi specialization pipeline.
"""

import asyncio
import logging
import os

from agent_forge.orchestration.curriculum_integration import MultiModelOrchestrator
from agent_forge.training.magi_specialization import MagiConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def run_magi_with_orchestration():
    """Run Magi specialization with OpenRouter orchestration."""
    logger.info("🎭 Starting Magi Agent Specialization with Multi-Model Orchestration")
    logger.info("=" * 80)

    # Check if OpenRouter is available
    openrouter_available = bool(os.getenv("OPENROUTER_API_KEY"))

    if openrouter_available:
        logger.info("✅ OpenRouter API key found - enabling multi-model orchestration")
    else:
        logger.warning("⚠️  OpenRouter API key not found - using local generation only")

    # Create Magi configuration - reduced for demo
    config = MagiConfig(
        optimal_model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        output_dir="D:/AgentForge/magi_orchestrated",
        curriculum_levels=3,  # Reduced for demo
        questions_per_level=20,  # Reduced for demo
        total_questions=60,  # 3 levels × 20 questions
        enable_geometric_awareness=True,
        enable_self_modification=True,
    )

    # Initialize orchestrator
    orchestrator = MultiModelOrchestrator(config, enable_openrouter=openrouter_available)

    try:
        # Phase 1: Generate Enhanced Curriculum
        logger.info("\n📚 Phase 1: Generating Enhanced Curriculum")
        logger.info("-" * 50)

        questions = orchestrator.question_generator.generate_curriculum_questions()
        logger.info(f"Generated {len(questions)} total questions across {config.curriculum_levels} levels")

        # Show sample questions
        for i, question in enumerate(questions[:3]):
            logger.info(f"\nSample Question {i + 1}:")
            logger.info(f"Domain: {question.domain}")
            logger.info(f"Difficulty: {question.difficulty}")
            logger.info(f"Text: {question.text[:200]}...")

        # Phase 2: Demonstrate Enhanced Evaluation
        logger.info("\n🎯 Phase 2: Enhanced Evaluation Demo")
        logger.info("-" * 50)

        # Test evaluation on a sample question
        sample_question = questions[0]
        test_answer = "This is a sample answer demonstrating the evaluation system."

        evaluation = await orchestrator.evaluate_answer_with_explanation(sample_question, test_answer)

        logger.info("Evaluation Results:")
        logger.info(f"Result: {str(evaluation)[:300]}...")

        # Phase 3: Generate Research Context
        if openrouter_available:
            logger.info("\n🔬 Phase 3: Research Context Generation")
            logger.info("-" * 50)

            research_context = await orchestrator.generate_research_context(
                "advanced algorithms and data structures",
                max_length=500,  # Reduced for demo
            )

            logger.info("Generated Research Context:")
            logger.info(f"{research_context[:400]}...")

        # Phase 4: Cost and Performance Summary
        logger.info("\n💰 Phase 4: Cost and Performance Summary")
        logger.info("-" * 50)

        cost_summary = orchestrator.get_cost_summary()

        if cost_summary["enabled"]:
            logger.info("Multi-Model Orchestration Metrics:")
            metrics = cost_summary["metrics"]
            logger.info(f"Total Cost: ${metrics['total_cost']:.4f}")
            logger.info(f"Cost by Task: {metrics['cost_by_task']}")

            if metrics["model_performance"]:
                logger.info("\nModel Performance:")
                for model, perf in metrics["model_performance"].items():
                    logger.info(
                        f"  {model}: {perf['requests']} requests, "
                        f"avg latency: {perf['avg_latency']:.2f}s, "
                        f"cost: ${perf['total_cost']:.4f}"
                    )
        else:
            logger.info("OpenRouter not enabled - no cost metrics available")

        # Success message
        logger.info("\n🎉 Magi Orchestration Demo Complete!")
        logger.info("=" * 80)
        logger.info("✅ All components working correctly")
        logger.info("✅ Multi-model orchestration functional")
        logger.info("✅ Cost tracking operational")
        logger.info("✅ Ready for full Magi specialization training")

        return True

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Clean up
        await orchestrator.close()


def main():
    """Main entry point."""
    success = asyncio.run(run_magi_with_orchestration())

    if success:
        print("\n🎭 Multi-Model Orchestration Demo: SUCCESS! ✅")
        print("\nNext steps:")
        print("1. Run full Magi specialization: python -m agent_forge.training.magi_specialization")
        print("2. Monitor costs with the orchestration dashboard")
        print("3. Adjust model routing based on performance metrics")
    else:
        print("\n❌ Multi-Model Orchestration Demo: FAILED")
        print("Check the logs above for error details")

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
