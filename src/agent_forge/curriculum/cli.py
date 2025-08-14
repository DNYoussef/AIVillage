"""CLI interface for Frontier Curriculum Engine.

Provides command-line tools for curriculum design, edge detection,
problem generation, and integration with Agent Forge training loop.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import click
import yaml

from .openrouter import OpenRouterLLM
from .schemas import (
    DifficultyScale,
    EdgeAssessmentRequest,
    EdgeAssessmentResponse,
    EdgeConstraints,
    TelemetryEntry,
)

logger = logging.getLogger(__name__)


@click.group()
def curriculum_cli():
    """Frontier Curriculum Engine - Edge-of-chaos curriculum design."""
    pass


@curriculum_cli.command()
@click.option("--config", type=click.Path(exists=True), help="Config file path")
@click.option("--api-key", help="OpenRouter API key (or use OPENROUTER_API_KEY env var)")
@click.option(
    "--model",
    default="anthropic/claude-3-5-sonnet-20241022",
    help="Model for edge finding",
)
@click.option("--domain", default="coding-python", help="Problem domain")
@click.option("--target-low", default=0.55, help="Target accuracy lower bound")
@click.option("--target-high", default=0.75, help="Target accuracy upper bound")
@click.option("--budget", default=1000, help="Problem generation budget")
@click.option("--telemetry-file", type=click.Path(exists=True), help="Telemetry data JSON file")
@click.option("--output-file", default="edge_assessment.json", help="Output file for results")
def find_edge(
    config: str | None,
    api_key: str | None,
    model: str,
    domain: str,
    target_low: float,
    target_high: float,
    budget: int,
    telemetry_file: str | None,
    output_file: str,
):
    """Find model's edge-of-chaos band from telemetry data."""

    click.echo("🎯 Finding Edge-of-Chaos Band")
    click.echo("=" * 40)

    # Load config if provided
    if config:
        with open(config) as f:
            config_data = yaml.safe_load(f)
        click.echo(f"✅ Loaded config from {config}")
    else:
        config_data = {}

    # Get API key
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            click.echo("❌ OpenRouter API key required (use --api-key or OPENROUTER_API_KEY env var)")
            sys.exit(1)

    # Load or generate sample telemetry data
    if telemetry_file:
        with open(telemetry_file) as f:
            telemetry_data = json.load(f)
        telemetry_entries = [TelemetryEntry(**entry) for entry in telemetry_data]
        click.echo(f"✅ Loaded {len(telemetry_entries)} telemetry entries from {telemetry_file}")
    else:
        # Generate sample telemetry for demo
        import random

        telemetry_entries = []
        for i in range(50):
            difficulty = random.uniform(0.3, 0.9)
            # Simulate accuracy curve - lower accuracy at extremes
            if difficulty < 0.4 or difficulty > 0.8:
                correct_prob = 0.3
            elif 0.5 <= difficulty <= 0.7:
                correct_prob = 0.65  # Edge-of-chaos band
            else:
                correct_prob = 0.45

            correct = random.random() < correct_prob
            telemetry_entries.append(TelemetryEntry(task_id=f"task_{i:03d}", difficulty=difficulty, correct=correct))
        click.echo(f"✅ Generated {len(telemetry_entries)} sample telemetry entries")

    # Create assessment request
    request = EdgeAssessmentRequest(
        domain=domain,
        telemetry=telemetry_entries,
        difficulty_scale=DifficultyScale(min=0.0, max=1.0),
        constraints=EdgeConstraints(target_low=target_low, target_high=target_high, problem_budget=budget),
    )

    # Run edge assessment
    async def run_assessment():
        async with OpenRouterLLM(api_key=api_key) as client:
            # Load template
            template_path = Path(__file__).parent / "templates" / "edge_finder.jinja"
            with open(template_path, encoding="utf-8") as f:
                template = f.read()

            # Render prompt
            prompt = client.render_template(
                template,
                domain=request.domain,
                telemetry=request.telemetry,
                difficulty_scale=request.difficulty_scale,
                constraints=request.constraints,
            )

            click.echo(f"🔍 Analyzing {len(request.telemetry)} telemetry entries...")

            # Invoke with schema validation
            response = await client.invoke_with_schema(
                prompt=prompt,
                schema_class=EdgeAssessmentResponse,
                model=model,
                temperature=0.3,
                max_tokens=4096,
            )

            return response

    try:
        response = asyncio.run(run_assessment())

        # Save results
        result_dict = response.dict()
        with open(output_file, "w") as f:
            json.dump(result_dict, f, indent=2)

        # Display results
        click.echo("\n📊 Edge Assessment Results:")
        click.echo(f"   Edge Band: {response.edge.low:.1%} - {response.edge.high:.1%}")
        click.echo(f"   Topic Mix: {len(response.topic_mix)} topics")
        for topic in response.topic_mix[:3]:
            click.echo(f"     • {topic.topic}: {topic.weight:.1%}")
        if len(response.topic_mix) > 3:
            click.echo(f"     ... and {len(response.topic_mix) - 3} more")
        click.echo(f"   Distribution: {len(response.distribution)} difficulty levels")
        click.echo(f"   Generation Plan: {response.generation_plan.n_total} problems total")
        click.echo(f"   Results saved to: {output_file}")

    except Exception as e:
        click.echo(f"❌ Edge assessment failed: {e}")
        logger.exception("Edge assessment error")
        sys.exit(1)


@curriculum_cli.command()
@click.option("--api-key", help="OpenRouter API key")
@click.option("--model", default="anthropic/claude-3-5-sonnet-20241022", help="Model to use")
@click.option(
    "--temperature-points",
    default="0.1,0.3,0.5,0.7,0.9",
    help="Temperature points to test",
)
@click.option("--prompt", help="Test prompt")
@click.option("--output-file", default="temperature_test.json", help="Output results file")
def test_temperatures(
    api_key: str | None,
    model: str,
    temperature_points: str,
    prompt: str | None,
    output_file: str,
):
    """Test temperature consistency across different settings."""

    click.echo("🌡️ Testing Temperature Consistency")
    click.echo("=" * 40)

    # Get API key
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            click.echo("❌ OpenRouter API key required")
            sys.exit(1)

    # Parse temperature points
    temps = [float(t.strip()) for t in temperature_points.split(",")]

    # Default test prompt
    if not prompt:
        prompt = """Write a Python function to solve this problem:

Given a list of integers, return the two numbers that sum to a target value.
Target: 9
Input: [2, 7, 11, 15]
Expected output: [2, 7] (or [0, 1] for indices)

Provide ONLY the function code, no explanation."""

    click.echo(f"Testing {len(temps)} temperature points: {temps}")
    click.echo(f"Test prompt: {prompt[:60]}...")

    async def run_temperature_test():
        results = {}

        async with OpenRouterLLM(api_key=api_key) as client:
            for temp in temps:
                click.echo(f"\n🌡️ Testing temperature {temp}")
                responses = []

                # Get 3 responses at each temperature
                for i in range(3):
                    try:
                        response = await client.invoke(prompt=prompt, model=model, temperature=temp, max_tokens=512)
                        responses.append(response)
                        click.echo(f"   Response {i + 1}: {len(response)} chars")
                    except Exception as e:
                        click.echo(f"   ❌ Response {i + 1} failed: {e}")
                        responses.append(f"ERROR: {e}")

                # Analyze consistency
                lengths = [len(r) for r in responses if not r.startswith("ERROR")]
                avg_length = sum(lengths) / len(lengths) if lengths else 0
                length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths) if lengths else 0

                results[str(temp)] = {
                    "responses": responses,
                    "avg_length": avg_length,
                    "length_variance": length_variance,
                    "success_count": len([r for r in responses if not r.startswith("ERROR")]),
                }

        return results

    try:
        results = asyncio.run(run_temperature_test())

        # Save results
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        # Display analysis
        click.echo("\n📊 Temperature Analysis:")
        for temp, data in results.items():
            success_rate = data["success_count"] / 3 * 100
            avg_len = data["avg_length"]
            variance = data["length_variance"]
            click.echo(f"   T={temp}: {success_rate:.0f}% success, {avg_len:.0f}±{variance:.0f} chars")

        click.echo(f"\n💾 Detailed results saved to: {output_file}")

    except Exception as e:
        click.echo(f"❌ Temperature test failed: {e}")
        sys.exit(1)


@curriculum_cli.command()
@click.option("--api-key", help="OpenRouter API key")
@click.option("--cache-dir", default=".forge/cache", help="Cache directory")
def cache_stats(api_key: str | None, cache_dir: str):
    """Show cache statistics and cost summary."""

    click.echo("💰 Curriculum Engine Cache & Cost Statistics")
    click.echo("=" * 50)

    # Get API key (for client initialization)
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY", "dummy")

    try:
        client = OpenRouterLLM(api_key=api_key, cache_dir=cache_dir)
        stats = client.get_cache_stats()

        if "error" in stats:
            click.echo(f"❌ Failed to get cache stats: {stats['error']}")
            return

        # Display cache stats
        click.echo("📦 Cache Statistics:")
        click.echo(f"   Total Entries: {stats['total_entries']:,}")

        if stats["by_model"]:
            click.echo("   By Model:")
            for model_stats in stats["by_model"]:
                model = model_stats["model"]
                count = model_stats["count"]
                tokens = model_stats["total_tokens"]
                cost = model_stats["total_cost_usd"]
                click.echo(f"     {model}: {count} requests, {tokens:,} tokens, ${cost:.3f}")

        if stats["recent_activity"]:
            click.echo("   Recent Activity:")
            for activity in stats["recent_activity"]:
                date = activity["date"]
                count = activity["count"]
                click.echo(f"     {date}: {count} requests")

        # Check cost log
        cost_log_path = Path(cache_dir) / "costs.jsonl"
        if cost_log_path.exists():
            click.echo(f"\n💸 Cost Log: {cost_log_path}")

            total_cost = 0.0
            total_requests = 0
            cached_requests = 0

            with open(cost_log_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("cost_usd"):
                            total_cost += entry["cost_usd"]
                        total_requests += 1
                        if entry.get("cached"):
                            cached_requests += 1
                    except:
                        continue

            cache_hit_rate = (cached_requests / total_requests * 100) if total_requests else 0

            click.echo(f"   Total Requests: {total_requests:,}")
            click.echo(f"   Cached Requests: {cached_requests:,} ({cache_hit_rate:.1f}%)")
            click.echo(f"   Total Cost: ${total_cost:.4f}")
            click.echo(f"   Average Cost/Request: ${total_cost / max(total_requests - cached_requests, 1):.5f}")
        else:
            click.echo(f"\n💸 No cost log found at {cost_log_path}")

        click.echo(f"\n📁 Cache Directory: {cache_dir}")

    except Exception as e:
        click.echo(f"❌ Failed to get statistics: {e}")
        sys.exit(1)


@curriculum_cli.command()
def demo():
    """Run curriculum engine demonstration."""

    click.echo("🎯 Frontier Curriculum Engine Demo")
    click.echo("=" * 45)

    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        click.echo("✅ OpenRouter API key found")
        click.echo("\nTo run full demo:")
        click.echo("  forge curriculum find-edge --telemetry-file your_data.json")
        click.echo("  forge curriculum test-temperatures --prompt 'Your test prompt'")
        click.echo("  forge curriculum cache-stats")
    else:
        click.echo("⚠️  No OpenRouter API key found")
        click.echo("   Set OPENROUTER_API_KEY environment variable to use live API")

    click.echo("\n🎯 System Components:")
    click.echo("   • Edge Finder: Identify optimal difficulty bands")
    click.echo("   • Problem Generator: Create targeted challenges")
    click.echo("   • Variant Synthesizer: Generate cosmetic variants")
    click.echo("   • Auto Grader: Score final answers only")
    click.echo("   • Hint System: Provide ≤25 token hints")
    click.echo("   • Mastery Tracker: 3-variant-pass rule")
    click.echo("   • Edge Controller: Maintain target accuracy")
    click.echo("   • Curriculum Conductor: Orchestrate full pipeline")

    click.echo("\n🔄 Integration Points:")
    click.echo("   • Agent Forge Training Loop")
    click.echo("   • OpenRouter API (60 RPM limit)")
    click.echo("   • SQLite caching for cost efficiency")
    click.echo("   • JSONL cost tracking")
    click.echo("   • Jinja2 prompt templates")

    click.echo("\n📈 Expected Outcomes:")
    click.echo("   • ~1000 problems at edge-of-chaos (55-75% accuracy)")
    click.echo("   • Productive struggle with adaptive difficulty")
    click.echo("   • Efficient mastery tracking and progression")
    click.echo("   • Cost-effective curriculum generation")

    # Show sample configuration
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        click.echo(f"\n⚙️  Configuration: {config_path}")
        click.echo("   Edit config.yaml to customize behavior")

    click.echo("\n🚀 Ready to revolutionize AI curriculum design!")


if __name__ == "__main__":
    curriculum_cli()
