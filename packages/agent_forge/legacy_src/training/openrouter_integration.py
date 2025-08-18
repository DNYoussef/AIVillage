"""
OpenRouter integration for temperature-alternating prompt templates.
Provides prompt suite templates optimized for different temperature ranges.
"""

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import aiohttp

from .temp_curriculum import TempBin

logger = logging.getLogger(__name__)


class PromptCategory(Enum):
    """Categories of prompts for different domains."""

    CODING_PYTHON = "coding-python"
    CODING_JAVASCRIPT = "coding-javascript"
    MATH = "math"
    LOGIC = "logic"
    REASONING = "reasoning"
    CREATIVE_WRITING = "creative-writing"
    ANALYSIS = "analysis"


class PromptComplexity(Enum):
    """Complexity levels for prompts."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class PromptTemplate:
    """Template for generating prompts with temperature awareness."""

    id: str
    category: PromptCategory
    complexity: PromptComplexity
    template: str
    variables: dict[str, list[str]] = field(default_factory=dict)
    optimal_temp_range: tuple[float, float] = (0.3, 0.7)
    expected_tokens: int = 150
    rubric: str = ""

    def generate(self, **kwargs) -> str:
        """Generate concrete prompt by filling template variables."""
        prompt = self.template

        # Fill template variables
        for var, choices in self.variables.items():
            if var in kwargs:
                value = kwargs[var]
            else:
                value = random.choice(choices)
            prompt = prompt.replace(f"{{{var}}}", str(value))

        return prompt

    def is_suitable_for_temp(self, temperature: float) -> bool:
        """Check if this template is suitable for given temperature."""
        return self.optimal_temp_range[0] <= temperature <= self.optimal_temp_range[1]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "category": self.category.value,
            "complexity": self.complexity.value,
            "template": self.template,
            "variables": self.variables,
            "optimal_temp_range": self.optimal_temp_range,
            "expected_tokens": self.expected_tokens,
            "rubric": self.rubric,
        }


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        default_model: str = "anthropic/claude-3-haiku",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://aivillage.dev",
                "X-Title": "AIVillage Temperature Alternation Training",
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def generate_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 200,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Generate completion using OpenRouter API."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use as async context manager.")

        if not self.api_key:
            # Return mock response for demo
            return await self._mock_completion(prompt, temperature, max_tokens)

        model = model or self.default_model

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

        try:
            async with self.session.post(f"{self.base_url}/chat/completions", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "completion": data["choices"][0]["message"]["content"],
                        "model": model,
                        "usage": data.get("usage", {}),
                        "temperature": temperature,
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}",
                        "temperature": temperature,
                    }

        except Exception as e:
            return {"success": False, "error": str(e), "temperature": temperature}

    async def _mock_completion(self, prompt: str, temperature: float, max_tokens: int) -> dict[str, Any]:
        """Generate mock completion for demo purposes."""

        # Simple mock based on prompt type
        if "python" in prompt.lower() or "code" in prompt.lower():
            if temperature < 0.3:
                completion = "def solution():\n    return 42"
            elif temperature < 0.7:
                completion = (
                    "def solution():\n    # Implementation here\n    result = process_data()\n    return result"
                )
            else:
                completion = "def solution():\n    # Creative approach\n    import random\n    magic = random.choice(['creativity', 'innovation'])\n    return f'Solution with {magic}!'"

        elif "math" in prompt.lower():
            if temperature < 0.3:
                completion = "The answer is 42."
            elif temperature < 0.7:
                completion = "Let me solve this step by step:\n1. Analyze the problem\n2. Apply relevant formulas\n3. The answer is 42."
            else:
                completion = "This is a fascinating mathematical puzzle! Let me explore multiple approaches and see where creativity takes us. The answer might be 42, but the journey is what matters!"

        else:
            base_responses = {
                "low": "Here is a direct answer:",
                "medium": "Let me think about this systematically and provide a comprehensive response:",
                "high": "What an intriguing question! Let me explore this from multiple angles and dive deep into the creative possibilities:",
            }

            temp_key = "low" if temperature < 0.3 else "medium" if temperature < 0.7 else "high"
            completion = base_responses[temp_key]

        # Simulate processing delay
        await asyncio.sleep(0.1)

        return {
            "success": True,
            "completion": completion,
            "model": "mock-model",
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(completion.split()),
            },
            "temperature": temperature,
        }


class PromptSuiteManager:
    """Manages prompt suite templates for temperature alternation training."""

    def __init__(self):
        self.templates = {}
        self.categories = {}
        self._initialize_templates()

    def _initialize_templates(self):
        """Initialize built-in prompt templates."""

        # Coding Python templates
        coding_templates = [
            PromptTemplate(
                id="python_list_comp",
                category=PromptCategory.CODING_PYTHON,
                complexity=PromptComplexity.SIMPLE,
                template="Write a Python list comprehension to {task} from a list of {data_type}.",
                variables={
                    "task": [
                        "filter even numbers",
                        "extract names",
                        "convert to uppercase",
                        "calculate squares",
                    ],
                    "data_type": ["integers", "strings", "dictionaries", "objects"],
                },
                optimal_temp_range=(0.1, 0.4),
                expected_tokens=50,
                rubric="Check for correctness, efficiency, and proper Python syntax.",
            ),
            PromptTemplate(
                id="python_function_design",
                category=PromptCategory.CODING_PYTHON,
                complexity=PromptComplexity.MEDIUM,
                template="Design a Python function that {functionality}. Include error handling and {requirement}.",
                variables={
                    "functionality": [
                        "processes user authentication",
                        "implements a caching system",
                        "parses configuration files",
                        "handles API rate limiting",
                    ],
                    "requirement": [
                        "comprehensive logging",
                        "type hints",
                        "docstring documentation",
                        "unit tests",
                    ],
                },
                optimal_temp_range=(0.3, 0.6),
                expected_tokens=200,
                rubric="Evaluate code structure, error handling, documentation, and adherence to best practices.",
            ),
            PromptTemplate(
                id="python_architecture",
                category=PromptCategory.CODING_PYTHON,
                complexity=PromptComplexity.EXPERT,
                template="Design a scalable Python architecture for {system_type} that handles {scale} and ensures {quality_attribute}. Consider microservices, databases, and deployment strategies.",
                variables={
                    "system_type": [
                        "real-time chat application",
                        "machine learning inference pipeline",
                        "financial trading system",
                        "content recommendation engine",
                    ],
                    "scale": [
                        "1M+ daily users",
                        "10TB+ data processing",
                        "sub-100ms latency",
                        "99.99% uptime",
                    ],
                    "quality_attribute": [
                        "high availability",
                        "security",
                        "performance",
                        "maintainability",
                    ],
                },
                optimal_temp_range=(0.6, 0.9),
                expected_tokens=400,
                rubric="Assess architectural decisions, scalability considerations, and technical depth.",
            ),
        ]

        # Math templates
        math_templates = [
            PromptTemplate(
                id="basic_arithmetic",
                category=PromptCategory.MATH,
                complexity=PromptComplexity.SIMPLE,
                template="Solve: {problem}. Show your work step by step.",
                variables={
                    "problem": [
                        "245 + 378 - 129",
                        "15 × 24 ÷ 3",
                        "√144 + 2³",
                        "3/4 + 2/3 - 1/6",
                    ]
                },
                optimal_temp_range=(0.0, 0.2),
                expected_tokens=100,
                rubric="Check calculation accuracy and clarity of steps.",
            ),
            PromptTemplate(
                id="algebra_word_problem",
                category=PromptCategory.MATH,
                complexity=PromptComplexity.MEDIUM,
                template="Solve this algebra word problem: {scenario}. Set up equations and solve step by step.",
                variables={
                    "scenario": [
                        "A train travels 180 miles in 3 hours. If it increases speed by 20 mph, how long will the same journey take?",
                        "The sum of two numbers is 45. One number is 3 more than twice the other. Find both numbers.",
                        "A rectangle's length is 5 cm more than its width. If the perimeter is 30 cm, find the dimensions.",
                        "An investment of $1000 grows to $1200 in 2 years with compound interest. What is the annual interest rate?",
                    ]
                },
                optimal_temp_range=(0.2, 0.5),
                expected_tokens=200,
                rubric="Evaluate problem setup, equation formulation, algebraic manipulation, and final verification.",
            ),
            PromptTemplate(
                id="advanced_calculus",
                category=PromptCategory.MATH,
                complexity=PromptComplexity.EXPERT,
                template="Analyze the {concept} of the function f(x) = {function}. Discuss {analysis_aspect} and provide geometric interpretation.",
                variables={
                    "concept": [
                        "convergence behavior",
                        "optimization properties",
                        "differential characteristics",
                        "integral applications",
                    ],
                    "function": ["xe^(-x²)", "ln(x)/x", "sin(x)/x", "x²e^(-x)"],
                    "analysis_aspect": [
                        "critical points and extrema",
                        "inflection points and concavity",
                        "asymptotic behavior",
                        "area under the curve",
                    ],
                },
                optimal_temp_range=(0.4, 0.8),
                expected_tokens=300,
                rubric="Assess mathematical rigor, conceptual understanding, and quality of explanations.",
            ),
        ]

        # Logic and reasoning templates
        logic_templates = [
            PromptTemplate(
                id="basic_logic",
                category=PromptCategory.LOGIC,
                complexity=PromptComplexity.SIMPLE,
                template="Evaluate the logical validity of this argument: {premise1}. {premise2}. Therefore, {conclusion}.",
                variables={
                    "premise1": [
                        "All cats are mammals",
                        "If it rains, the ground gets wet",
                        "All students in this class study math",
                        "Every bird can fly",
                    ],
                    "premise2": [
                        "Fluffy is a cat",
                        "It is raining outside",
                        "Sarah is in this class",
                        "A penguin is a bird",
                    ],
                    "conclusion": [
                        "Fluffy is a mammal",
                        "the ground is wet",
                        "Sarah studies math",
                        "penguins can fly",
                    ],
                },
                optimal_temp_range=(0.1, 0.3),
                expected_tokens=80,
                rubric="Check logical reasoning accuracy and identification of valid/invalid arguments.",
            ),
            PromptTemplate(
                id="puzzle_solving",
                category=PromptCategory.REASONING,
                complexity=PromptComplexity.COMPLEX,
                template="Solve this reasoning puzzle: {puzzle_scenario}. Explain your reasoning process.",
                variables={
                    "puzzle_scenario": [
                        "Three switches control three light bulbs in another room. You can only visit the room once. How do you determine which switch controls which bulb?",
                        "You have 12 balls, one of which is either heavier or lighter. Using a balance scale only 3 times, how do you find the odd ball?",
                        "A man lives on the 20th floor. Every morning he takes the elevator down to ground floor. When he comes home, he takes the elevator to the 10th floor and walks the rest, except on rainy days when he takes it all the way. Why?",
                        "Five pirates must divide 100 gold coins. They vote in order of seniority, and any proposal needs majority approval. What should the senior pirate propose?",
                    ]
                },
                optimal_temp_range=(0.5, 0.8),
                expected_tokens=250,
                rubric="Evaluate logical reasoning, creative problem-solving, and clarity of explanation.",
            ),
        ]

        # Creative writing templates
        creative_templates = [
            PromptTemplate(
                id="creative_story",
                category=PromptCategory.CREATIVE_WRITING,
                complexity=PromptComplexity.MEDIUM,
                template="Write a short story ({length}) about {character} who discovers {discovery} in a {setting}. Include {element}.",
                variables={
                    "length": ["200-300 words", "400-500 words", "one page"],
                    "character": [
                        "a time-traveling librarian",
                        "an AI that gained consciousness",
                        "a detective who can read memories",
                        "a child who speaks to animals",
                    ],
                    "discovery": [
                        "a hidden door in their basement",
                        "they can manipulate probability",
                        "their reflection acts independently",
                        "a message from their future self",
                    ],
                    "setting": [
                        "abandoned space station",
                        "underwater city",
                        "digital reality",
                        "parallel dimension",
                    ],
                    "element": [
                        "a moral dilemma",
                        "an unexpected ally",
                        "a race against time",
                        "a surprising plot twist",
                    ],
                },
                optimal_temp_range=(0.7, 1.2),
                expected_tokens=400,
                rubric="Assess creativity, narrative structure, character development, and engagement.",
            )
        ]

        # Store templates by category
        all_templates = coding_templates + math_templates + logic_templates + creative_templates

        for template in all_templates:
            self.templates[template.id] = template

            if template.category not in self.categories:
                self.categories[template.category] = []
            self.categories[template.category].append(template)

        logger.info(f"Initialized {len(all_templates)} prompt templates across {len(self.categories)} categories")

    def get_templates_for_temp_bin(self, temp_bin: TempBin) -> list[PromptTemplate]:
        """Get templates suitable for a given temperature bin."""
        suitable_templates = []

        for template in self.templates.values():
            if template.is_suitable_for_temp(temp_bin.center):
                suitable_templates.append(template)

        return suitable_templates

    def get_templates_by_category(self, category: PromptCategory) -> list[PromptTemplate]:
        """Get all templates for a specific category."""
        return self.categories.get(category, [])

    def get_template(self, template_id: str) -> PromptTemplate | None:
        """Get specific template by ID."""
        return self.templates.get(template_id)

    def generate_prompt_batch(
        self,
        temp_bin: TempBin,
        batch_size: int = 10,
        category_filter: PromptCategory | None = None,
    ) -> list[tuple[str, PromptTemplate]]:
        """Generate batch of prompts for temperature bin."""

        # Get suitable templates
        suitable_templates = self.get_templates_for_temp_bin(temp_bin)

        if category_filter:
            suitable_templates = [t for t in suitable_templates if t.category == category_filter]

        if not suitable_templates:
            logger.warning(f"No suitable templates found for temperature bin {temp_bin.center}")
            return []

        # Generate batch
        batch = []
        for _ in range(batch_size):
            template = random.choice(suitable_templates)
            prompt = template.generate()
            batch.append((prompt, template))

        return batch


class OpenRouterTempAltSystem:
    """Complete OpenRouter integration for temperature alternation system."""

    def __init__(
        self,
        api_key: str | None = None,
        prompt_suite_manager: PromptSuiteManager | None = None,
    ):
        self.api_key = api_key
        self.prompt_manager = prompt_suite_manager or PromptSuiteManager()
        self.results_cache = {}

    async def generate_training_data(
        self,
        temp_bins: list[TempBin],
        samples_per_bin: int = 20,
        categories: list[PromptCategory] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate training data across temperature bins using OpenRouter."""

        training_data = []

        async with OpenRouterClient(api_key=self.api_key) as client:
            for temp_bin in temp_bins:
                logger.info(f"Generating {samples_per_bin} samples for temp bin {temp_bin.center:.2f}")

                # Generate prompts for this temperature bin
                prompts_batch = self.prompt_manager.generate_prompt_batch(
                    temp_bin=temp_bin, batch_size=samples_per_bin, category_filter=None
                )

                for prompt, template in prompts_batch:
                    # Generate completion with OpenRouter
                    result = await client.generate_completion(
                        prompt=prompt,
                        temperature=temp_bin.center,
                        max_tokens=template.expected_tokens,
                    )

                    if result["success"]:
                        training_sample = {
                            "prompt": prompt,
                            "completion": result["completion"],
                            "temperature": temp_bin.center,
                            "temp_bin": temp_bin.to_dict(),
                            "template_id": template.id,
                            "category": template.category.value,
                            "complexity": template.complexity.value,
                            "rubric": template.rubric,
                            "usage": result.get("usage", {}),
                            "model": result.get("model", "unknown"),
                        }
                        training_data.append(training_sample)
                    else:
                        logger.warning(f"Failed to generate completion: {result['error']}")

        logger.info(f"Generated {len(training_data)} training samples")
        return training_data

    async def evaluate_temperature_consistency(
        self, prompt: str, temperature_points: list[float] = None
    ) -> dict[str, Any]:
        """Evaluate how completions vary across temperature points."""

        if temperature_points is None:
            temperature_points = [0.1, 0.3, 0.5, 0.7, 0.9, 1.2]

        results = []

        async with OpenRouterClient(api_key=self.api_key) as client:
            for temp in temperature_points:
                result = await client.generate_completion(prompt=prompt, temperature=temp, max_tokens=150)
                results.append(
                    {
                        "temperature": temp,
                        "success": result["success"],
                        "completion": result.get("completion", ""),
                        "error": result.get("error", ""),
                        "usage": result.get("usage", {}),
                    }
                )

        # Analyze consistency
        successful_results = [r for r in results if r["success"]]

        analysis = {
            "prompt": prompt,
            "total_attempts": len(temperature_points),
            "successful_attempts": len(successful_results),
            "success_rate": len(successful_results) / len(temperature_points),
            "results": results,
            "analysis": {
                "length_variance": self._analyze_length_variance(successful_results),
                "content_diversity": self._analyze_content_diversity(successful_results),
                "temperature_sensitivity": self._analyze_temperature_sensitivity(successful_results),
            },
        }

        return analysis

    def _analyze_length_variance(self, results: list[dict]) -> dict[str, float]:
        """Analyze variance in completion lengths across temperatures."""
        lengths = [len(r["completion"].split()) for r in results]

        if not lengths:
            return {"mean": 0, "variance": 0, "min": 0, "max": 0}

        return {
            "mean": np.mean(lengths),
            "variance": np.var(lengths),
            "min": min(lengths),
            "max": max(lengths),
        }

    def _analyze_content_diversity(self, results: list[dict]) -> dict[str, float]:
        """Analyze content diversity across temperatures."""
        completions = [r["completion"] for r in results]

        if len(completions) < 2:
            return {"average_similarity": 1.0, "unique_ratio": 1.0}

        # Simple similarity metric based on word overlap
        similarities = []
        unique_words = set()
        total_words = 0

        for i, comp1 in enumerate(completions):
            words1 = set(comp1.lower().split())
            unique_words.update(words1)
            total_words += len(words1)

            for j, comp2 in enumerate(completions[i + 1 :], i + 1):
                words2 = set(comp2.lower().split())

                if len(words1.union(words2)) > 0:
                    similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                    similarities.append(similarity)

        return {
            "average_similarity": np.mean(similarities) if similarities else 0.0,
            "unique_ratio": len(unique_words) / total_words if total_words > 0 else 0.0,
        }

    def _analyze_temperature_sensitivity(self, results: list[dict]) -> dict[str, Any]:
        """Analyze how sensitive completions are to temperature changes."""
        if len(results) < 2:
            return {"sensitivity_score": 0.0, "monotonic_trend": False}

        # Sort by temperature
        sorted_results = sorted(results, key=lambda x: x["temperature"])

        # Measure length trend
        temperatures = [r["temperature"] for r in sorted_results]
        lengths = [len(r["completion"].split()) for r in sorted_results]

        # Simple correlation
        if len(set(temperatures)) > 1 and len(set(lengths)) > 1:
            correlation = np.corrcoef(temperatures, lengths)[0, 1]
            sensitivity_score = abs(correlation)
        else:
            sensitivity_score = 0.0

        # Check for monotonic trend
        length_diffs = [lengths[i + 1] - lengths[i] for i in range(len(lengths) - 1)]
        monotonic_trend = all(d >= 0 for d in length_diffs) or all(d <= 0 for d in length_diffs)

        return {
            "sensitivity_score": sensitivity_score,
            "monotonic_trend": monotonic_trend,
            "correlation": correlation if "correlation" in locals() else 0.0,
        }

    def export_templates(self, filepath: str):
        """Export all templates to JSON file."""
        templates_data = {
            "templates": {tid: template.to_dict() for tid, template in self.prompt_manager.templates.items()},
            "categories": {cat.value: len(templates) for cat, templates in self.prompt_manager.categories.items()},
            "total_templates": len(self.prompt_manager.templates),
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(templates_data, f, indent=2)

        logger.info(f"Exported {len(templates_data['templates'])} templates to {filepath}")


# Factory functions


def create_openrouter_system(api_key: str | None = None) -> OpenRouterTempAltSystem:
    """Create OpenRouter temperature alternation system."""
    return OpenRouterTempAltSystem(api_key=api_key)


def create_prompt_suite_manager() -> PromptSuiteManager:
    """Create prompt suite manager with default templates."""
    return PromptSuiteManager()


if __name__ == "__main__":
    # Demo OpenRouter integration system
    print("🌐🌡️ OpenRouter Temperature Alternation Integration Demo")
    print("=" * 65)

    async def main():
        # Create system (without API key for demo)
        system = create_openrouter_system(api_key=None)  # Will use mock completions

        print(f"Created OpenRouter system with {len(system.prompt_manager.templates)} templates")
        print()

        # Show template categories
        print("📝 Available Template Categories:")
        for category, templates in system.prompt_manager.categories.items():
            print(f"   {category.value:20} | {len(templates):2} templates")
        print()

        # Create mock temperature bins
        from .temp_curriculum import TempBin, TempBinType

        temp_bins = [
            TempBin(0.0, 0.2, 0.1, TempBinType.LOW),
            TempBin(0.4, 0.6, 0.5, TempBinType.MID),
            TempBin(0.8, 1.0, 0.9, TempBinType.HIGH),
        ]

        print(f"🌡️ Testing with {len(temp_bins)} temperature bins:")
        for bin in temp_bins:
            print(f"   {bin.bin_type.value:10} | [{bin.low:.1f}, {bin.high:.1f}] center={bin.center:.1f}")
        print()

        # Generate sample prompts for each temperature bin
        print("📋 Sample Prompts by Temperature Bin:")

        for temp_bin in temp_bins:
            suitable_templates = system.prompt_manager.get_templates_for_temp_bin(temp_bin)
            print(f"\n   {temp_bin.bin_type.value.upper()} Temperature (τ={temp_bin.center:.1f}):")
            print(f"   Suitable templates: {len(suitable_templates)}")

            if suitable_templates:
                # Generate sample prompt
                template = suitable_templates[0]
                sample_prompt = template.generate()
                print(f"   Sample: {sample_prompt[:80]}...")
                print(f"   Category: {template.category.value}, Complexity: {template.complexity.value}")

        print()

        # Test temperature consistency evaluation
        print("🔬 Temperature Consistency Evaluation:")
        test_prompt = "Write a Python function to calculate the factorial of a number."

        consistency_analysis = await system.evaluate_temperature_consistency(
            prompt=test_prompt, temperature_points=[0.1, 0.5, 0.9]
        )

        print(f"   Test prompt: {test_prompt}")
        print(f"   Success rate: {consistency_analysis['success_rate']:.1%}")
        print(f"   Length variance: {consistency_analysis['analysis']['length_variance']['variance']:.2f}")
        print(f"   Content diversity: {consistency_analysis['analysis']['content_diversity']['unique_ratio']:.2f}")
        print(
            f"   Temperature sensitivity: {consistency_analysis['analysis']['temperature_sensitivity']['sensitivity_score']:.2f}"
        )

        print()

        # Show sample completions at different temperatures
        print("📊 Sample Completions at Different Temperatures:")
        for result in consistency_analysis["results"]:
            if result["success"]:
                temp = result["temperature"]
                completion = (
                    result["completion"][:60] + "..." if len(result["completion"]) > 60 else result["completion"]
                )
                print(f"   τ={temp:.1f}: {completion}")

        print()

        # Generate small training dataset
        print("🏗️ Training Data Generation:")

        training_data = await system.generate_training_data(
            temp_bins=temp_bins[:2],  # First 2 bins for demo
            samples_per_bin=3,
        )

        print(f"   Generated {len(training_data)} training samples")

        if training_data:
            sample = training_data[0]
            print("   Sample training record:")
            print(f"     Temperature: {sample['temperature']:.1f}")
            print(f"     Category: {sample['category']}")
            print(f"     Prompt: {sample['prompt'][:50]}...")
            print(f"     Completion: {sample['completion'][:50]}...")

        print()

        # Export templates
        print("💾 Template Export:")
        system.export_templates("temp_alt_templates.json")
        print("   Templates exported to temp_alt_templates.json")

        print()
        print("✅ OpenRouter Integration Demo Complete")
        print()
        print("Key Features Demonstrated:")
        print("  • Comprehensive prompt template library with temperature optimization")
        print("  • OpenRouter API integration with async support")
        print("  • Temperature consistency analysis and evaluation")
        print("  • Automated training data generation across temperature bins")
        print("  • Template categorization by domain, complexity, and optimal temperature")
        print("  • Content diversity and sensitivity analysis")
        print("  • JSON export/import for template management")
        print("  • Production-ready integration with temperature alternation system")

    # Run async demo
    asyncio.run(main())
