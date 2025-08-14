"""Demonstration of Multi-Model Frontier Curriculum Engine.

Shows how the curriculum alternates between GPT-4o, Claude Opus, and Gemini Pro
for question generation, grading, and hint generation.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def demo_multi_model_curriculum():
    """Demonstrate multi-model curriculum system."""

    print("MULTI-MODEL FRONTIER CURRICULUM ENGINE DEMO")
    print("=" * 50)
    print("This demo shows how the curriculum alternates between:")
    print("• OpenAI GPT-4o")
    print("• Anthropic Claude 3.5 Sonnet")
    print("• Google Gemini Pro 1.5")
    print()

    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("⚠️  OPENROUTER_API_KEY not set - using mock mode")
        api_key = "mock-key"
        use_mock = True
    else:
        print("✅ OpenRouter API key found - using live API")
        use_mock = False

    try:
        # Import curriculum components
        from agent_forge.curriculum import (
            EdgeWindow,
            Grader,
            HintGenerator,
            OpenRouterLLM,
            Problem,
            ProblemGenerator,
            TopicMix,
        )

        # Initialize multi-model client
        multi_model_pool = [
            "openai/gpt-4o",  # GPT-4o (closest to GPT-5)
            "anthropic/claude-3-5-sonnet-20241022",  # Claude 3.5 Sonnet
            "google/gemini-pro-1.5",  # Gemini Pro 1.5
        ]

        if use_mock:
            # Use mock client for demo
            from tests.curriculum.test_integration_comprehensive import (
                MockOpenRouterLLM,
            )

            llm_client = MockOpenRouterLLM(api_key)
            print("🤖 Using mock LLM client for demonstration")
        else:
            llm_client = OpenRouterLLM(
                api_key=api_key, model_pool=multi_model_pool, rpm_limit=60
            )
            print(f"🌐 Using live OpenRouter with {len(multi_model_pool)} models")

        async with llm_client:
            print("\n" + "=" * 60)
            print("PHASE 1: MULTI-MODEL PROBLEM GENERATION")
            print("=" * 60)

            # Create problem generator
            problem_gen = ProblemGenerator(llm_client)
            edge = EdgeWindow(low=0.55, high=0.75)
            topic_mix = [
                TopicMix(topic="string_manipulation", weight=0.4),
                TopicMix(topic="algorithms", weight=0.6),
            ]

            # Generate problems (will use random model selection)
            print("🔄 Generating problems with random model selection...")
            result = await problem_gen.generate_problems(
                domain="coding-python",
                edge=edge,
                topic_mix=topic_mix,
                n=6,  # Generate 6 problems to see model diversity
                batch_size=2,
            )

            if result.ok:
                print(f"✅ Generated {len(result.problems)} problems")
                for i, problem in enumerate(result.problems):
                    print(f"   Problem {i + 1}: {problem.statement[:60]}...")
            else:
                print(f"❌ Problem generation failed: {result.msg}")

            print("\n" + "=" * 60)
            print("PHASE 2: MULTI-MODEL GRADING")
            print("=" * 60)

            # Create grader
            grader = Grader(llm_client, enable_code_execution=False)

            # Test different solutions
            test_problem = Problem(
                id="demo_001",
                topic="algorithms",
                difficulty=0.6,
                statement="Write a function to find the maximum value in a list",
                canonical_answer="def find_max(lst): return max(lst) if lst else None",
                rubric="Function finds maximum correctly and handles empty list",
                unit_tests=["assert find_max([1,3,2]) == 3"],
            )

            test_solutions = [
                "def find_max(lst): return max(lst) if lst else None",  # Correct
                "def find_max(lst): return max(lst)",  # Missing empty case
                "def find_max(lst): return min(lst)",  # Wrong function
            ]

            print("🔄 Grading solutions with random model selection...")
            for i, solution in enumerate(test_solutions):
                result = await grader.grade_solution(
                    problem=test_problem, model_answer=solution
                )

                if result.ok:
                    print(
                        f"   Solution {i + 1}: {'✅ CORRECT' if result.correct else '❌ INCORRECT'}"
                    )
                    if result.error_tags:
                        print(f"      Error tags: {result.error_tags}")
                else:
                    print(f"   Solution {i + 1}: ❌ Grading failed")

            print("\n" + "=" * 60)
            print("PHASE 3: MULTI-MODEL HINT GENERATION")
            print("=" * 60)

            # Create hint generator
            hint_gen = HintGenerator(llm_client)

            wrong_solutions = [
                "def find_max(lst): return lst[0]",  # Only returns first element
                "def find_max(lst): return sorted(lst)[-1]",  # Inefficient but works
                "print(max(lst))",  # Print instead of return
            ]

            print("🔄 Generating hints with random model selection...")
            for i, wrong_solution in enumerate(wrong_solutions):
                result = await hint_gen.generate_hint(
                    problem=test_problem, wrong_answer=wrong_solution
                )

                if result.ok:
                    print(f"   Wrong solution {i + 1}:")
                    print(f"      Code: {wrong_solution}")
                    print(f"      Hint: {result.hint}")
                    print(f"      Type: {result.hint_type}")
                else:
                    print(f"   Hint generation {i + 1}: ❌ Failed")

            print("\n" + "=" * 60)
            print("MODEL USAGE STATISTICS")
            print("=" * 60)

            # Show model usage statistics if available
            if hasattr(llm_client, "get_model_stats"):
                stats = llm_client.get_model_stats()

                if "error" not in stats:
                    print(f"📊 Total requests: {stats['total_requests']}")
                    print(f"🤖 Model pool enabled: {stats['model_pool_enabled']}")
                    print(f"🎯 Diversity score: {stats['diversity_score']:.2f}")

                    if stats["model_usage"]:
                        print("📈 Model usage breakdown:")
                        for model, count in stats["model_usage"].items():
                            percentage = (count / stats["total_requests"]) * 100
                            print(f"   {model}: {count} requests ({percentage:.1f}%)")

                    print(f"\n🌟 Model pool: {len(stats['model_pool'])} models")
                    for model in stats["model_pool"]:
                        print(f"   • {model}")
                else:
                    print(f"❌ Could not retrieve model stats: {stats['error']}")

            print("\n" + "=" * 60)
            print("CURRICULUM BENEFITS OF MULTI-MODEL APPROACH")
            print("=" * 60)

            print("🧠 Cognitive Diversity Benefits:")
            print("   • Different reasoning styles for problem generation")
            print("   • Varied grading perspectives reduce bias")
            print("   • Diverse hint strategies for different learning styles")
            print("   • Reduced overfitting to single model's patterns")

            print("\n🎯 Curriculum Quality Benefits:")
            print("   • More robust problem difficulty assessment")
            print("   • Better coverage of edge cases in grading")
            print("   • Richer hint generation with multiple approaches")
            print("   • Enhanced edge-of-chaos maintenance through consensus")

            print("\n⚖️ Fairness & Robustness Benefits:")
            print("   • No single point of failure if one model is down")
            print("   • Reduced bias from any single model's training")
            print("   • Better generalization to diverse student populations")
            print("   • Natural A/B testing of different AI capabilities")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n🎉 MULTI-MODEL CURRICULUM DEMO COMPLETE!")
    print("\nThe Frontier Curriculum Engine now uses model diversity to:")
    print("1. Generate more varied and robust problems")
    print("2. Provide more balanced grading across different AI perspectives")
    print("3. Offer diverse hint strategies for different learning styles")
    print("4. Maintain the edge-of-chaos more effectively through consensus")

    return True


def main():
    """Main demo function."""

    print("Multi-Model Frontier Curriculum Engine")
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        success = asyncio.run(demo_multi_model_curriculum())

        if success:
            print("\n✅ Demo completed successfully!")
            print("\nTo use in production:")
            print("1. Set OPENROUTER_API_KEY environment variable")
            print("2. Initialize OpenRouterLLM with model_pool parameter")
            print("3. The curriculum will automatically alternate between models")
        else:
            print("\n❌ Demo encountered errors")

        return success

    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
