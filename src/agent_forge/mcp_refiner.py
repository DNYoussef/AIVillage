"""MCP (Model Context Protocol) Refiner

Implements tool prompt refinement and optimization for MCP integration:
- Automatic tool prompt optimization
- Context-aware prompt generation
- Tool usage pattern analysis
- Performance-driven prompt evolution
- Integration with geometry feedback for guided refinement
"""

import asyncio
import json
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_forge.geometry_feedback import GeometryTracker

logger = logging.getLogger(__name__)


@dataclass
class ToolUsagePattern:
    """Represents a pattern of tool usage."""

    tool_name: str
    parameter_patterns: dict[str, Any]
    context_patterns: list[str]
    success_rate: float
    frequency: int
    avg_response_time: float
    error_patterns: list[str]


@dataclass
class PromptCandidate:
    """A candidate prompt for optimization."""

    prompt_text: str
    tool_name: str
    parameters: dict[str, str]
    performance_score: float
    usage_count: int
    success_count: int
    error_count: int
    avg_response_quality: float
    compass_alignment: str

    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.usage_count)


@dataclass
class MCPRefinementConfig:
    """Configuration for MCP refinement."""

    model_path: str
    output_dir: str
    max_prompt_length: int = 512
    num_optimization_rounds: int = 50
    population_size: int = 20
    mutation_rate: float = 0.15
    crossover_rate: float = 0.6
    evaluation_samples: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_project: str = "mcp-refinement"


class ToolAnalyzer:
    """Analyzes tool usage patterns and performance."""

    def __init__(self):
        self.usage_logs = []
        self.tool_patterns = defaultdict(list)
        self.performance_metrics = defaultdict(list)

    def log_tool_usage(
        self,
        tool_name: str,
        prompt: str,
        parameters: dict[str, Any],
        success: bool,
        response_time: float,
        response_quality: float,
        error_message: str | None = None,
    ):
        """Log a tool usage event."""
        usage_event = {
            "tool_name": tool_name,
            "prompt": prompt,
            "parameters": parameters,
            "success": success,
            "response_time": response_time,
            "response_quality": response_quality,
            "error_message": error_message,
            "timestamp": time.time(),
        }

        self.usage_logs.append(usage_event)
        self.tool_patterns[tool_name].append(usage_event)
        self.performance_metrics[tool_name].append(
            {
                "success": success,
                "response_time": response_time,
                "response_quality": response_quality,
            }
        )

    def analyze_tool_patterns(self, tool_name: str) -> ToolUsagePattern:
        """Analyze usage patterns for a specific tool."""
        if tool_name not in self.tool_patterns:
            return ToolUsagePattern(
                tool_name=tool_name,
                parameter_patterns={},
                context_patterns=[],
                success_rate=0.0,
                frequency=0,
                avg_response_time=0.0,
                error_patterns=[],
            )

        events = self.tool_patterns[tool_name]
        metrics = self.performance_metrics[tool_name]

        # Analyze parameter patterns
        param_patterns = defaultdict(Counter)
        for event in events:
            for param, value in event["parameters"].items():
                if isinstance(value, str) and len(value) < 100:
                    param_patterns[param][value] += 1

        # Extract context patterns
        context_patterns = []
        for event in events:
            # Extract patterns from prompts
            words = re.findall(r"\b\w+\b", event["prompt"].lower())
            if len(words) >= 3:
                for i in range(len(words) - 2):
                    trigram = " ".join(words[i: i + 3])
                    context_patterns.append(trigram)

        # Get most common context patterns
        context_counter = Counter(context_patterns)
        top_contexts = [pattern for pattern, count in context_counter.most_common(10)]

        # Analyze errors
        error_patterns = []
        for event in events:
            if not event["success"] and event["error_message"]:
                error_patterns.append(event["error_message"])

        # Calculate metrics
        success_rate = sum(1 for m in metrics if m["success"]) / len(metrics)
        avg_response_time = np.mean([m["response_time"] for m in metrics])

        return ToolUsagePattern(
            tool_name=tool_name,
            parameter_patterns=dict(param_patterns),
            context_patterns=top_contexts,
            success_rate=success_rate,
            frequency=len(events),
            avg_response_time=avg_response_time,
            error_patterns=error_patterns[:10],  # Top 10 error patterns
        )

    def get_underperforming_tools(self, min_usage: int = 10) -> list[str]:
        """Get list of tools that are underperforming."""
        underperforming = []

        for tool_name, metrics in self.performance_metrics.items():
            if len(metrics) >= min_usage:
                success_rate = sum(1 for m in metrics if m["success"]) / len(metrics)
                avg_quality = np.mean([m["response_quality"] for m in metrics])

                if success_rate < 0.7 or avg_quality < 0.6:
                    underperforming.append(tool_name)

        return underperforming


class PromptEvolver:
    """Evolves and optimizes tool prompts using genetic algorithms."""

    def __init__(self, config: MCPRefinementConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.geometry_tracker = None

    async def initialize(self):
        """Initialize model and tokenizer."""
        logger.info(f"Loading model from {self.config.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16
            if self.config.device == "cuda"
            else torch.float32,
            device_map="auto" if self.config.device == "cuda" else None,
        )

        if self.config.device == "cpu":
            self.model = self.model.to(self.config.device)

        # Initialize geometry tracker
        self.geometry_tracker = GeometryTracker(
            self.model,
            update_interval=10,
            output_dir=f"{self.config.output_dir}/geometry",
        )

    def generate_initial_population(
        self, base_prompt: str, tool_name: str, tool_pattern: ToolUsagePattern
    ) -> list[PromptCandidate]:
        """Generate initial population of prompt candidates."""
        population = []

        # Include original prompt
        population.append(
            PromptCandidate(
                prompt_text=base_prompt,
                tool_name=tool_name,
                parameters={},
                performance_score=0.5,
                usage_count=1,
                success_count=1,
                error_count=0,
                avg_response_quality=0.5,
                compass_alignment="Unknown",
            )
        )

        # Generate variations
        for _ in range(self.config.population_size - 1):
            variant = self._mutate_prompt(base_prompt, tool_pattern)
            population.append(
                PromptCandidate(
                    prompt_text=variant,
                    tool_name=tool_name,
                    parameters={},
                    performance_score=0.0,
                    usage_count=0,
                    success_count=0,
                    error_count=0,
                    avg_response_quality=0.0,
                    compass_alignment="Unknown",
                )
            )

        return population

    def _mutate_prompt(self, prompt: str, pattern: ToolUsagePattern) -> str:
        """Mutate a prompt to create a variant."""
        mutations = [
            self._add_context_clues,
            self._add_parameter_guidance,
            self._add_error_prevention,
            self._simplify_language,
            self._add_examples,
            self._restructure_format,
        ]

        # Apply random mutations
        mutated = prompt
        for mutation_func in np.random.choice(mutations, size=2, replace=False):
            try:
                mutated = mutation_func(mutated, pattern)
            except Exception as e:
                logger.warning(f"Mutation failed: {e}")

        return mutated

    def _add_context_clues(self, prompt: str, pattern: ToolUsagePattern) -> str:
        """Add context clues based on usage patterns."""
        if pattern.context_patterns:
            context = np.random.choice(pattern.context_patterns)
            return f"{prompt}\n\nContext hint: Consider {context} when using this tool."
        return prompt

    def _add_parameter_guidance(self, prompt: str, pattern: ToolUsagePattern) -> str:
        """Add parameter guidance based on successful patterns."""
        if pattern.parameter_patterns:
            param_name = np.random.choice(list(pattern.parameter_patterns.keys()))
            param_values = pattern.parameter_patterns[param_name]

            if param_values:
                common_value = max(param_values, key=param_values.get)
                return f"{prompt}\n\nTip: For {param_name}, consider using '{common_value}' for better results."
        return prompt

    def _add_error_prevention(self, prompt: str, pattern: ToolUsagePattern) -> str:
        """Add error prevention guidance."""
        if pattern.error_patterns:
            error = np.random.choice(pattern.error_patterns)
            return f"{prompt}\n\nNote: To avoid errors, ensure {error.lower()} is handled properly."
        return prompt

    def _simplify_language(self, prompt: str, pattern: ToolUsagePattern) -> str:
        """Simplify language for better clarity."""
        # Replace complex words with simpler alternatives
        replacements = {
            "utilize": "use",
            "facilitate": "help",
            "implement": "do",
            "demonstrate": "show",
            "establish": "set up",
            "determine": "find",
            "subsequent": "next",
            "aforementioned": "mentioned",
            "nevertheless": "however",
            "consequently": "so",
        }

        simplified = prompt
        for complex_word, simple_word in replacements.items():
            simplified = re.sub(
                r"\b" + complex_word + r"\b",
                simple_word,
                simplified,
                flags=re.IGNORECASE,
            )

        return simplified

    def _add_examples(self, prompt: str, pattern: ToolUsagePattern) -> str:
        """Add usage examples based on patterns."""
        if pattern.parameter_patterns:
            example = f"\n\nExample usage: {pattern.tool_name}("
            examples = []
            for param, values in pattern.parameter_patterns.items():
                if values:
                    common_value = max(values, key=values.get)
                    examples.append(f"{param}='{common_value}'")
            example += ", ".join(examples[:3]) + ")"
            return prompt + example
        return prompt

    def _restructure_format(self, prompt: str, pattern: ToolUsagePattern) -> str:
        """Restructure prompt format for better readability."""
        lines = prompt.split("\n")

        # Add headers and structure
        if len(lines) > 3:
            structured = ["# Tool Usage Guide", ""]

            # Add purpose section
            structured.extend(["## Purpose", lines[0], ""])

            # Add parameters section
            if len(lines) > 1:
                structured.extend(["## Parameters", lines[1], ""])

            # Add usage notes
            if len(lines) > 2:
                structured.extend(["## Usage Notes"] + lines[2:])

            return "\n".join(structured)

        return prompt

    def crossover_prompts(
        self, parent1: PromptCandidate, parent2: PromptCandidate
    ) -> str:
        """Create offspring prompt by crossing over two parents."""
        lines1 = parent1.prompt_text.split("\n")
        lines2 = parent2.prompt_text.split("\n")

        # Mix lines from both parents
        offspring_lines = []
        max_lines = max(len(lines1), len(lines2))

        for i in range(max_lines):
            if i < len(lines1) and i < len(lines2):
                # Choose randomly from either parent
                line = np.random.choice([lines1[i], lines2[i]])
            elif i < len(lines1):
                line = lines1[i]
            else:
                line = lines2[i]

            offspring_lines.append(line)

        return "\n".join(offspring_lines)

    async def evaluate_prompt(
        self, candidate: PromptCandidate, test_scenarios: list[dict[str, Any]]
    ) -> float:
        """Evaluate a prompt candidate on test scenarios."""
        if not test_scenarios:
            return 0.5

        total_score = 0.0
        successful_evaluations = 0

        for scenario in test_scenarios[:20]:  # Limit to 20 scenarios for efficiency
            try:
                score = await self._evaluate_single_scenario(candidate, scenario)
                total_score += score
                successful_evaluations += 1
            except Exception as e:
                logger.warning(f"Evaluation failed for scenario: {e}")

        if successful_evaluations == 0:
            return 0.0

        avg_score = total_score / successful_evaluations

        # Update geometry tracking
        if self.geometry_tracker and self.model:
            # Get hidden states from model
            inputs = self.tokenizer.encode(
                candidate.prompt_text, return_tensors="pt"
            ).to(self.config.device)
            with torch.no_grad():
                outputs = self.model(inputs, output_hidden_states=True)
                if hasattr(outputs, "hidden_states"):
                    hidden_states = outputs.hidden_states[-1]
                    metrics = self.geometry_tracker.update(hidden_states)

                    if metrics:
                        candidate.compass_alignment = metrics.compass_direction

        return avg_score

    async def _evaluate_single_scenario(
        self, candidate: PromptCandidate, scenario: dict[str, Any]
    ) -> float:
        """Evaluate prompt on a single test scenario."""
        # Simulate prompt evaluation
        prompt_with_scenario = (
            f"{candidate.prompt_text}\n\nScenario: {scenario.get('description', '')}"
        )

        # Tokenize and check length
        tokens = self.tokenizer.encode(prompt_with_scenario)
        if len(tokens) > self.config.max_prompt_length:
            return 0.2  # Penalize overly long prompts

        # Simulate tool execution quality
        base_score = 0.7

        # Bonus for good structure
        if "Purpose" in candidate.prompt_text or "Example" in candidate.prompt_text:
            base_score += 0.1

        # Bonus for appropriate length
        if 50 <= len(candidate.prompt_text) <= 300:
            base_score += 0.1

        # Penalty for too much complexity
        complex_words = ["utilize", "facilitate", "implement", "demonstrate"]
        complexity_penalty = sum(
            0.02 for word in complex_words if word in candidate.prompt_text.lower()
        )
        base_score -= complexity_penalty

        return max(0.0, min(1.0, base_score + np.random.normal(0, 0.1)))


class MCPRefiner:
    """Main MCP refinement orchestrator."""

    def __init__(self, config: MCPRefinementConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.tool_analyzer = ToolAnalyzer()
        self.prompt_evolver = PromptEvolver(config)

        # Tracking
        self.refinement_history = {}
        self.best_prompts = {}

        # W&B logging
        if wandb.run is None:
            wandb.init(
                project=config.wandb_project,
                config=config.__dict__,
                job_type="mcp_refinement",
            )

    async def initialize(self):
        """Initialize refiner components."""
        await self.prompt_evolver.initialize()
        logger.info("MCP Refiner initialized successfully")

    async def refine_tool_prompts(
        self,
        tool_definitions: dict[str, dict[str, Any]],
        usage_logs: list[dict[str, Any]] = None,
    ) -> dict[str, str]:
        """Refine prompts for multiple tools."""
        logger.info(f"Starting MCP refinement for {len(tool_definitions)} tools")

        # Process usage logs if provided
        if usage_logs:
            for log_entry in usage_logs:
                self.tool_analyzer.log_tool_usage(**log_entry)

        refined_prompts = {}

        for tool_name, tool_def in tool_definitions.items():
            logger.info(f"Refining prompts for tool: {tool_name}")

            try:
                # Analyze tool patterns
                pattern = self.tool_analyzer.analyze_tool_patterns(tool_name)

                # Get base prompt
                base_prompt = tool_def.get("description", f"Use the {tool_name} tool.")

                # Generate test scenarios
                test_scenarios = self._generate_test_scenarios(tool_def, pattern)

                # Evolve prompts
                best_prompt = await self._evolve_prompt(
                    base_prompt, tool_name, pattern, test_scenarios
                )

                refined_prompts[tool_name] = best_prompt

                # Log results
                wandb.log(
                    {
                        f"tools/{tool_name}/refinement_complete": True,
                        f"tools/{tool_name}/original_length": len(base_prompt),
                        f"tools/{tool_name}/refined_length": len(best_prompt),
                        f"tools/{tool_name}/success_rate": pattern.success_rate,
                    }
                )

            except Exception as e:
                logger.error(f"Failed to refine prompts for {tool_name}: {e}")
                refined_prompts[tool_name] = tool_def.get(
                    "description", f"Use the {tool_name} tool."
                )

        # Save results
        await self._save_refinement_results(refined_prompts)

        logger.info("MCP refinement completed successfully")
        return refined_prompts

    def _generate_test_scenarios(
        self, tool_def: dict[str, Any], pattern: ToolUsagePattern
    ) -> list[dict[str, Any]]:
        """Generate test scenarios for prompt evaluation."""
        scenarios = []

        # Base scenarios from tool definition
        if "examples" in tool_def:
            for example in tool_def["examples"]:
                scenarios.append(
                    {
                        "description": example.get("description", ""),
                        "parameters": example.get("parameters", {}),
                        "expected_outcome": example.get("expected_outcome", "success"),
                    }
                )

        # Scenarios from usage patterns
        for context in pattern.context_patterns[:5]:
            scenarios.append(
                {
                    "description": f"Context involving {context}",
                    "parameters": {},
                    "expected_outcome": "success",
                }
            )

        # Error scenarios
        for error in pattern.error_patterns[:3]:
            scenarios.append(
                {
                    "description": f"Scenario that might cause: {error}",
                    "parameters": {},
                    "expected_outcome": "error_prevention",
                }
            )

        # Generate additional synthetic scenarios
        for _ in range(10):
            scenarios.append(
                {
                    "description": f"Generic usage scenario {len(scenarios) + 1}",
                    "parameters": {},
                    "expected_outcome": "success",
                }
            )

        return scenarios

    async def _evolve_prompt(
        self,
        base_prompt: str,
        tool_name: str,
        pattern: ToolUsagePattern,
        test_scenarios: list[dict[str, Any]],
    ) -> str:
        """Evolve a prompt using genetic algorithm."""
        # Initialize population
        population = self.prompt_evolver.generate_initial_population(
            base_prompt, tool_name, pattern
        )

        best_candidate = None
        best_score = 0.0

        for generation in range(self.config.num_optimization_rounds):
            logger.info(
                f"Generation {generation + 1}/{self.config.num_optimization_rounds} for {tool_name}"
            )

            # Evaluate population
            for candidate in population:
                score = await self.prompt_evolver.evaluate_prompt(
                    candidate, test_scenarios
                )
                candidate.performance_score = score

                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            # Log generation results
            avg_score = np.mean([c.performance_score for c in population])
            wandb.log(
                {
                    f"evolution/{tool_name}/generation": generation,
                    f"evolution/{tool_name}/best_score": best_score,
                    f"evolution/{tool_name}/avg_score": avg_score,
                }
            )

            # Selection and reproduction
            if generation < self.config.num_optimization_rounds - 1:
                population = self._evolve_population(population, pattern)

        return best_candidate.prompt_text if best_candidate else base_prompt

    def _evolve_population(
        self, population: list[PromptCandidate], pattern: ToolUsagePattern
    ) -> list[PromptCandidate]:
        """Evolve population for next generation."""
        # Sort by performance
        sorted_pop = sorted(population, key=lambda x: x.performance_score, reverse=True)

        # Keep top performers (elitism)
        elite_size = max(2, self.config.population_size // 4)
        new_population = sorted_pop[:elite_size]

        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(sorted_pop)
            parent2 = self._tournament_selection(sorted_pop)

            # Crossover
            if np.random.random() < self.config.crossover_rate:
                offspring_text = self.prompt_evolver.crossover_prompts(parent1, parent2)
            else:
                offspring_text = parent1.prompt_text

            # Mutation
            if np.random.random() < self.config.mutation_rate:
                offspring_text = self.prompt_evolver._mutate_prompt(
                    offspring_text, pattern
                )

            # Create new candidate
            offspring = PromptCandidate(
                prompt_text=offspring_text,
                tool_name=parent1.tool_name,
                parameters={},
                performance_score=0.0,
                usage_count=0,
                success_count=0,
                error_count=0,
                avg_response_quality=0.0,
                compass_alignment="Unknown",
            )

            new_population.append(offspring)

        return new_population

    def _tournament_selection(
        self, population: list[PromptCandidate], tournament_size: int = 3
    ) -> PromptCandidate:
        """Tournament selection for parent selection."""
        tournament = np.random.choice(
            population, size=min(tournament_size, len(population)), replace=False
        )
        return max(tournament, key=lambda x: x.performance_score)

    async def _save_refinement_results(self, refined_prompts: dict[str, str]):
        """Save refinement results."""
        results = {
            "refined_prompts": refined_prompts,
            "refinement_config": self.config.__dict__,
            "timestamp": time.time(),
        }

        results_path = self.output_dir / "mcp_refined_prompts.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Refinement results saved: {results_path}")

        # Save individual prompt files
        prompts_dir = self.output_dir / "prompts"
        prompts_dir.mkdir(exist_ok=True)

        for tool_name, prompt in refined_prompts.items():
            prompt_file = prompts_dir / f"{tool_name}_refined.md"
            with open(prompt_file, "w") as f:
                f.write(f"# {tool_name} - Refined Prompt\n\n")
                f.write(prompt)

    def get_refinement_summary(self) -> dict[str, Any]:
        """Get refinement summary statistics."""
        underperforming = self.tool_analyzer.get_underperforming_tools()

        tool_stats = {}
        for tool_name in self.tool_analyzer.tool_patterns.keys():
            pattern = self.tool_analyzer.analyze_tool_patterns(tool_name)
            tool_stats[tool_name] = {
                "success_rate": pattern.success_rate,
                "frequency": pattern.frequency,
                "avg_response_time": pattern.avg_response_time,
                "error_count": len(pattern.error_patterns),
            }

        return {
            "total_tools_analyzed": len(self.tool_analyzer.tool_patterns),
            "underperforming_tools": underperforming,
            "tool_statistics": tool_stats,
            "total_usage_events": len(self.tool_analyzer.usage_logs),
        }


# CLI and usage
async def main():
    """Main MCP refinement entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Tool Prompt Refinement")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--tool-definitions", required=True, help="Path to tool definitions JSON"
    )
    parser.add_argument("--usage-logs", help="Path to usage logs JSON")
    parser.add_argument("--rounds", type=int, default=30, help="Optimization rounds")
    parser.add_argument("--population", type=int, default=15, help="Population size")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Load tool definitions
    with open(args.tool_definitions) as f:
        tool_definitions = json.load(f)

    # Load usage logs if provided
    usage_logs = None
    if args.usage_logs:
        with open(args.usage_logs) as f:
            usage_logs = json.load(f)

    # Create configuration
    config = MCPRefinementConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_optimization_rounds=args.rounds,
        population_size=args.population,
        device=args.device,
    )

    # Initialize refiner
    refiner = MCPRefiner(config)
    await refiner.initialize()

    # Run refinement
    refined_prompts = await refiner.refine_tool_prompts(tool_definitions, usage_logs)

    # Print summary
    summary = refiner.get_refinement_summary()
    print("\nRefinement Summary:")
    print(f"Tools analyzed: {summary['total_tools_analyzed']}")
    print(f"Underperforming tools: {len(summary['underperforming_tools'])}")
    print(f"Usage events processed: {summary['total_usage_events']}")

    print(
        f"\nRefined prompts for {len(refined_prompts)} tools saved to {args.output_dir}"
    )


if __name__ == "__main__":
    asyncio.run(main())
