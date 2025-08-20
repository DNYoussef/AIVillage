"""
Thought-token A/B baking system for Quiet-STaR optimization.
Implements B4) A/B testing different reflection styles and prompt formats.
"""

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .config import QuietSTaRConfig
from .teacher import ReflectionPrompt, TeacherPromptGenerator, TrainingPair


class ReflectionStyle(Enum):
    """Different reflection styles for A/B testing."""

    STEP_BY_STEP = "step_by_step"
    CRITICAL_THINKING = "critical_thinking"
    ANALYTICAL = "analytical"
    EXPLORATORY = "exploratory"
    CONCISE = "concise"
    VERBOSE = "verbose"
    STRUCTURED = "structured"
    FREE_FORM = "free_form"


class PromptFormat(Enum):
    """Different prompt formatting approaches."""

    STANDARD = "standard"  # Question <SoT>reflection</SoT> Answer
    EXPLICIT = "explicit"  # Question Let me think: <SoT>reflection</SoT> Answer
    MINIMAL = "minimal"  # Question <SoT>reflection</SoT> Answer (shorter reflections)
    CHAIN_OF_THOUGHT = "cot"  # Question <SoT>Step 1: ... Step 2: ...</SoT> Answer
    SELF_CORRECTION = "correction"  # Question <SoT>Initial thought... Actually, ...</SoT> Answer


@dataclass
class ABVariant:
    """A single A/B test variant with specific configuration."""

    name: str
    reflection_style: ReflectionStyle
    prompt_format: PromptFormat
    max_reflection_tokens: int = 128
    temperature: float = 0.7
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set variant-specific parameters."""
        if self.reflection_style == ReflectionStyle.CONCISE:
            self.max_reflection_tokens = min(self.max_reflection_tokens, 64)
        elif self.reflection_style == ReflectionStyle.VERBOSE:
            self.max_reflection_tokens = max(self.max_reflection_tokens, 256)

        # Set temperature based on style
        if self.reflection_style == ReflectionStyle.CRITICAL_THINKING:
            self.temperature = 0.8  # More creativity for critical analysis
        elif self.reflection_style == ReflectionStyle.STRUCTURED:
            self.temperature = 0.5  # More deterministic for structured output


@dataclass
class ABTestResult:
    """Results from testing a single A/B variant."""

    variant: ABVariant
    metrics: dict[str, float]
    sample_outputs: list[dict[str, Any]]
    execution_time: float
    success_rate: float
    leak_rate: float
    quality_score: float

    def overall_score(self, weights: dict[str, float] | None = None) -> float:
        """Calculate weighted overall score for ranking variants."""
        if weights is None:
            weights = {
                "success_rate": 0.4,  # Most important: no leaks
                "quality_score": 0.3,  # Quality of reflections
                "response_coherence": 0.2,  # How coherent the final answer
                "efficiency": 0.1,  # Speed/token efficiency
            }

        # Calculate efficiency score (inverse of execution time, normalized)
        efficiency = 1.0 / max(self.execution_time, 0.1)
        normalized_efficiency = min(efficiency / 10.0, 1.0)  # Normalize to 0-1

        score_components = {
            "success_rate": self.success_rate,
            "quality_score": self.quality_score,
            "response_coherence": self.metrics.get("response_coherence", 0.5),
            "efficiency": normalized_efficiency,
        }

        weighted_score = sum(weights.get(component, 0) * value for component, value in score_components.items())

        return weighted_score


@dataclass
class ABTestSuite:
    """Complete A/B test suite configuration and results."""

    test_name: str
    variants: list[ABVariant]
    test_questions: list[str]
    results: list[ABTestResult] = field(default_factory=list)
    winner: ABVariant | None = None
    confidence_level: float = 0.0

    def get_winner(self, min_confidence: float = 0.8) -> ABVariant | None:
        """Get the winning variant if confidence is high enough."""
        if not self.results or len(self.results) < 2:
            return None

        # Sort by overall score
        sorted_results = sorted(self.results, key=lambda r: r.overall_score(), reverse=True)

        # Calculate confidence based on score differences
        best_score = sorted_results[0].overall_score()
        second_best_score = sorted_results[1].overall_score() if len(sorted_results) > 1 else 0

        # Confidence is based on the relative difference
        if second_best_score > 0:
            confidence = (best_score - second_best_score) / best_score
        else:
            confidence = 1.0

        self.confidence_level = confidence

        if confidence >= min_confidence:
            self.winner = sorted_results[0].variant
            return self.winner

        return None


class ThoughtTokenABBaker:
    """
    A/B testing system for optimizing thought-token prompts and reflection styles.
    Bakes the best performing variants into optimized configurations.
    """

    def __init__(self, config: QuietSTaRConfig, base_model_name: str = "microsoft/DialoGPT-small"):
        self.config = config
        self.base_model_name = base_model_name
        self.test_suites: list[ABTestSuite] = []
        self.baked_variants: dict[str, ABVariant] = {}

    def create_reflection_prompts_for_variant(self, variant: ABVariant) -> list[ReflectionPrompt]:
        """Create reflection prompts tailored to the A/B variant."""

        style_templates = {
            ReflectionStyle.STEP_BY_STEP: [
                "{question} {start_token}Let me break this down step by step. First, I'll analyze the question. Then, I'll work through each component systematically.{end_token}",
                "{question} {start_token}Step 1: Understand the problem. Step 2: Identify key components. Step 3: Work through the solution methodically.{end_token}",
            ],
            ReflectionStyle.CRITICAL_THINKING: [
                "{question} {start_token}I should examine this critically. What assumptions am I making? What evidence supports different viewpoints? Let me challenge my initial thinking.{end_token}",
                "{question} {start_token}Critical analysis required. What are the strengths and weaknesses of different approaches? What might I be missing?{end_token}",
            ],
            ReflectionStyle.ANALYTICAL: [
                "{question} {start_token}This requires systematic analysis. Let me identify the variables, relationships, and underlying principles involved.{end_token}",
                "{question} {start_token}Analytical approach: break down into components, examine relationships, synthesize findings into a coherent answer.{end_token}",
            ],
            ReflectionStyle.EXPLORATORY: [
                "{question} {start_token}Let me explore different angles. What if I approach this from another perspective? What alternative solutions exist?{end_token}",
                "{question} {start_token}Exploring possibilities: multiple approaches could work here. Let me consider the trade-offs and best path forward.{end_token}",
            ],
            ReflectionStyle.CONCISE: [
                "{question} {start_token}Quick analysis: key point is X, therefore Y.{end_token}",
                "{question} {start_token}Core issue: A leads to B, so C.{end_token}",
            ],
            ReflectionStyle.VERBOSE: [
                "{question} {start_token}This is a complex question that requires comprehensive analysis. Let me thoroughly examine each aspect, consider multiple perspectives, evaluate evidence, and work through potential objections before reaching a well-reasoned conclusion.{end_token}",
                "{question} {start_token}I need to carefully consider all dimensions of this question. The background context is important, as are the implications of different answers. Let me think through this comprehensively.{end_token}",
            ],
            ReflectionStyle.STRUCTURED: [
                "{question} {start_token}Analysis Framework: 1) Problem Definition 2) Key Variables 3) Solution Approach 4) Validation. Let me work through each systematically.{end_token}",
                "{question} {start_token}Structured thinking: Context -> Analysis -> Options -> Decision -> Rationale. Following this framework.{end_token}",
            ],
            ReflectionStyle.FREE_FORM: [
                "{question} {start_token}Hmm, this makes me think about... actually, there's an interesting connection to... let me follow this line of reasoning...{end_token}",
                "{question} {start_token}This question reminds me of related concepts. Let me explore the connections and see where the reasoning leads.{end_token}",
            ],
        }

        templates = style_templates.get(variant.reflection_style, style_templates[ReflectionStyle.STEP_BY_STEP])

        return [
            ReflectionPrompt(
                template=template,
                reflection_style=variant.reflection_style.value,
                max_reflection_tokens=variant.max_reflection_tokens,
            )
            for template in templates
        ]

    def generate_training_data_for_variant(
        self,
        variant: ABVariant,
        questions: list[str],
        num_samples_per_question: int = 1,
    ) -> list[TrainingPair]:
        """Generate training data using the specified variant configuration."""

        # Create teacher with variant-specific prompts
        teacher = TeacherPromptGenerator(self.config, self.base_model_name)
        teacher.reflection_prompts = self.create_reflection_prompts_for_variant(variant)

        training_pairs = []

        for question in questions:
            for _ in range(num_samples_per_question):
                try:
                    # Generate with variant-specific settings
                    pair = teacher.create_training_pair(
                        question=question,
                        style=variant.reflection_style.value,
                        max_reflection_tokens=variant.max_reflection_tokens,
                    )

                    # Apply prompt format transformation
                    formatted_pair = self._apply_prompt_format(pair, variant.prompt_format)
                    formatted_pair.metadata.update(
                        {
                            "variant_name": variant.name,
                            "reflection_style": variant.reflection_style.value,
                            "prompt_format": variant.prompt_format.value,
                            "temperature": variant.temperature,
                        }
                    )

                    training_pairs.append(formatted_pair)

                except Exception as e:
                    print(f"Warning: Failed to generate training pair for variant {variant.name}: {e}")
                    continue

        return training_pairs

    def _apply_prompt_format(self, pair: TrainingPair, prompt_format: PromptFormat) -> TrainingPair:
        """Apply specific prompt formatting to a training pair."""

        if prompt_format == PromptFormat.STANDARD:
            # No change needed - already in standard format
            return pair

        elif prompt_format == PromptFormat.EXPLICIT:
            # Add explicit thinking cue
            new_reflection = f"Let me think about this carefully. {pair.reflection}"
            return TrainingPair(
                question=pair.question,
                reflection=new_reflection,
                answer=pair.answer,
                metadata=pair.metadata,
            )

        elif prompt_format == PromptFormat.MINIMAL:
            # Shorten reflection while keeping key points
            words = pair.reflection.split()
            if len(words) > 20:
                shortened = " ".join(words[:15]) + "... Therefore, the answer is:"
            else:
                shortened = pair.reflection

            return TrainingPair(
                question=pair.question,
                reflection=shortened,
                answer=pair.answer,
                metadata=pair.metadata,
            )

        elif prompt_format == PromptFormat.CHAIN_OF_THOUGHT:
            # Structure as explicit steps
            structured_reflection = self._convert_to_chain_of_thought(pair.reflection)
            return TrainingPair(
                question=pair.question,
                reflection=structured_reflection,
                answer=pair.answer,
                metadata=pair.metadata,
            )

        elif prompt_format == PromptFormat.SELF_CORRECTION:
            # Add self-correction pattern
            corrected_reflection = f"Initially, I might think {pair.reflection[:50]}... But actually, let me reconsider. {pair.reflection[50:]}"
            return TrainingPair(
                question=pair.question,
                reflection=corrected_reflection,
                answer=pair.answer,
                metadata=pair.metadata,
            )

        return pair

    def _convert_to_chain_of_thought(self, reflection: str) -> str:
        """Convert free-form reflection to chain-of-thought format."""
        sentences = reflection.split(". ")
        if len(sentences) <= 1:
            return reflection

        # Structure as steps
        structured_steps = []
        for i, sentence in enumerate(sentences[:3], 1):  # Limit to 3 steps
            if sentence.strip():
                structured_steps.append(f"Step {i}: {sentence.strip()}.")

        return " ".join(structured_steps)

    def evaluate_variant(
        self,
        variant: ABVariant,
        test_questions: list[str],
        evaluation_samples: int = 10,
    ) -> ABTestResult:
        """Evaluate a single A/B variant and return detailed results."""

        start_time = time.time()

        # Generate training data for this variant
        training_pairs = self.generate_training_data_for_variant(variant, test_questions, num_samples_per_question=1)

        if not training_pairs:
            return ABTestResult(
                variant=variant,
                metrics={"error": 1.0},
                sample_outputs=[],
                execution_time=time.time() - start_time,
                success_rate=0.0,
                leak_rate=1.0,
                quality_score=0.0,
            )

        # Evaluate quality metrics
        metrics = self._calculate_variant_metrics(training_pairs)

        # Sample outputs for inspection
        sample_outputs = [
            {
                "question": pair.question,
                "reflection": pair.reflection,
                "answer": pair.answer,
                "training_format": pair.to_training_text(),
                "inference_format": pair.to_inference_text(),
            }
            for pair in training_pairs[: min(3, len(training_pairs))]
        ]

        execution_time = time.time() - start_time

        # Calculate success and leak rates
        leak_count = sum(
            1 for pair in training_pairs if "<SoT>" in pair.to_inference_text() or "</SoT>" in pair.to_inference_text()
        )
        leak_rate = leak_count / len(training_pairs) if training_pairs else 1.0
        success_rate = 1.0 - leak_rate

        # Calculate quality score based on multiple factors
        quality_score = self._calculate_quality_score(training_pairs, metrics)

        return ABTestResult(
            variant=variant,
            metrics=metrics,
            sample_outputs=sample_outputs,
            execution_time=execution_time,
            success_rate=success_rate,
            leak_rate=leak_rate,
            quality_score=quality_score,
        )

    def _calculate_variant_metrics(self, training_pairs: list[TrainingPair]) -> dict[str, float]:
        """Calculate detailed metrics for variant evaluation."""
        if not training_pairs:
            return {"error": 1.0}

        metrics = {}

        # Reflection length statistics
        reflection_lengths = [len(pair.reflection.split()) for pair in training_pairs]
        metrics["avg_reflection_length"] = statistics.mean(reflection_lengths)
        metrics["reflection_length_std"] = statistics.stdev(reflection_lengths) if len(reflection_lengths) > 1 else 0

        # Answer length statistics
        answer_lengths = [len(pair.answer.split()) for pair in training_pairs]
        metrics["avg_answer_length"] = statistics.mean(answer_lengths)

        # Quality indicators
        metrics["reflection_complexity"] = self._calculate_reflection_complexity(training_pairs)
        metrics["answer_coherence"] = self._calculate_answer_coherence(training_pairs)
        metrics["question_answer_relevance"] = self._calculate_relevance(training_pairs)

        # Format consistency
        training_texts = [pair.to_training_text() for pair in training_pairs]
        inference_texts = [pair.to_inference_text() for pair in training_pairs]
        metrics["format_consistency"] = self._calculate_format_consistency(training_texts, inference_texts)

        return metrics

    def _calculate_reflection_complexity(self, training_pairs: list[TrainingPair]) -> float:
        """Calculate complexity score for reflections (0-1 scale)."""
        complexity_indicators = [
            "because",
            "therefore",
            "however",
            "consider",
            "analyze",
            "approach",
            "method",
            "step",
            "first",
            "then",
            "finally",
        ]

        total_score = 0
        for pair in training_pairs:
            reflection_lower = pair.reflection.lower()
            indicator_count = sum(1 for indicator in complexity_indicators if indicator in reflection_lower)
            # Normalize by reflection length and max possible indicators
            length_factor = min(len(pair.reflection.split()) / 50, 1.0)
            indicator_factor = min(indicator_count / 5, 1.0)
            total_score += (length_factor + indicator_factor) / 2

        return total_score / len(training_pairs)

    def _calculate_answer_coherence(self, training_pairs: list[TrainingPair]) -> float:
        """Calculate coherence score for answers (0-1 scale)."""
        coherence_score = 0

        for pair in training_pairs:
            # Simple coherence heuristics
            sentences = pair.answer.split(".")
            if len(sentences) > 1:
                # Check for connecting words between sentences
                connectors = [
                    "this",
                    "these",
                    "therefore",
                    "however",
                    "also",
                    "additionally",
                ]
                connection_score = 0
                for i in range(1, len(sentences)):
                    sentence_lower = sentences[i].lower()
                    if any(conn in sentence_lower for conn in connectors):
                        connection_score += 1
                coherence_score += connection_score / max(len(sentences) - 1, 1)
            else:
                coherence_score += 0.5  # Single sentence gets medium score

        return coherence_score / len(training_pairs)

    def _calculate_relevance(self, training_pairs: list[TrainingPair]) -> float:
        """Calculate question-answer relevance score (0-1 scale)."""
        relevance_score = 0

        for pair in training_pairs:
            # Extract key words from question
            question_words = set(
                word.lower().strip("?.,")
                for word in pair.question.split()
                if len(word) > 3 and word.lower() not in {"what", "how", "why", "when", "where", "which"}
            )

            # Check overlap with answer
            answer_words = set(word.lower().strip(".,") for word in pair.answer.split())

            if question_words:
                overlap = len(question_words.intersection(answer_words))
                relevance_score += overlap / len(question_words)
            else:
                relevance_score += 0.5

        return relevance_score / len(training_pairs)

    def _calculate_format_consistency(self, training_texts: list[str], inference_texts: list[str]) -> float:
        """Calculate format consistency score (0-1 scale)."""
        consistency_score = 0

        for training_text, inference_text in zip(training_texts, inference_texts, strict=False):
            # Training should have thought tokens, inference should not
            has_training_tokens = "<SoT>" in training_text and "</SoT>" in training_text
            has_inference_tokens = "<SoT>" in inference_text or "</SoT>" in inference_text

            if has_training_tokens and not has_inference_tokens:
                consistency_score += 1.0
            elif has_training_tokens and has_inference_tokens:
                consistency_score += 0.0  # Critical failure - leak detected
            elif not has_training_tokens:
                consistency_score += 0.5  # Missing tokens in training

        return consistency_score / len(training_texts)

    def _calculate_quality_score(self, training_pairs: list[TrainingPair], metrics: dict[str, float]) -> float:
        """Calculate overall quality score combining multiple metrics."""
        quality_components = {
            "reflection_complexity": metrics.get("reflection_complexity", 0) * 0.3,
            "answer_coherence": metrics.get("answer_coherence", 0) * 0.3,
            "relevance": metrics.get("question_answer_relevance", 0) * 0.2,
            "format_consistency": metrics.get("format_consistency", 0) * 0.2,
        }

        return sum(quality_components.values())

    def run_ab_test_suite(
        self,
        test_name: str,
        variants: list[ABVariant],
        test_questions: list[str],
        parallel: bool = True,
        max_workers: int = 3,
    ) -> ABTestSuite:
        """Run a complete A/B test suite comparing multiple variants."""

        test_suite = ABTestSuite(test_name=test_name, variants=variants, test_questions=test_questions)

        print(f"🧪 Running A/B test suite: {test_name}")
        print(f"   Variants: {len(variants)}")
        print(f"   Test questions: {len(test_questions)}")

        if parallel and len(variants) > 1:
            # Run variants in parallel for faster testing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_variant = {
                    executor.submit(self.evaluate_variant, variant, test_questions): variant for variant in variants
                }

                for future in as_completed(future_to_variant):
                    variant = future_to_variant[future]
                    try:
                        result = future.result()
                        test_suite.results.append(result)
                        print(f"   ✓ Completed variant: {variant.name} (score: {result.overall_score():.3f})")
                    except Exception as e:
                        print(f"   ✗ Failed variant: {variant.name} - {e}")
        else:
            # Sequential execution
            for variant in variants:
                print(f"   Testing variant: {variant.name}")
                try:
                    result = self.evaluate_variant(variant, test_questions)
                    test_suite.results.append(result)
                    print(f"   ✓ Completed (score: {result.overall_score():.3f})")
                except Exception as e:
                    print(f"   ✗ Failed: {e}")

        # Determine winner
        winner = test_suite.get_winner()
        if winner:
            print(f"🏆 Winner: {winner.name} (confidence: {test_suite.confidence_level:.1%})")
        else:
            print("🤷 No clear winner - results too close")

        self.test_suites.append(test_suite)
        return test_suite

    def bake_winning_variants(self, min_confidence: float = 0.8) -> dict[str, ABVariant]:
        """Bake winning variants from all test suites into optimized configurations."""

        baked_variants = {}

        for test_suite in self.test_suites:
            winner = test_suite.get_winner(min_confidence)
            if winner:
                baked_variants[test_suite.test_name] = winner
                self.baked_variants[test_suite.test_name] = winner
                print(f"🔥 Baked winner from {test_suite.test_name}: {winner.name}")

        return baked_variants

    def save_test_results(self, output_path: Path):
        """Save all test results and baked variants to files."""
        output_path.mkdir(parents=True, exist_ok=True)

        # Save detailed test results
        results_data = {
            "test_suites": [
                {
                    "test_name": suite.test_name,
                    "variants": [
                        {
                            "name": variant.name,
                            "reflection_style": variant.reflection_style.value,
                            "prompt_format": variant.prompt_format.value,
                            "max_reflection_tokens": variant.max_reflection_tokens,
                            "temperature": variant.temperature,
                        }
                        for variant in suite.variants
                    ],
                    "results": [
                        {
                            "variant_name": result.variant.name,
                            "overall_score": result.overall_score(),
                            "success_rate": result.success_rate,
                            "leak_rate": result.leak_rate,
                            "quality_score": result.quality_score,
                            "execution_time": result.execution_time,
                            "metrics": result.metrics,
                            "sample_outputs": result.sample_outputs[:2],  # Limit samples
                        }
                        for result in suite.results
                    ],
                    "winner": (
                        {
                            "name": suite.winner.name,
                            "confidence": suite.confidence_level,
                        }
                        if suite.winner
                        else None
                    ),
                }
                for suite in self.test_suites
            ]
        }

        with open(output_path / "ab_test_results.json", "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        # Save baked variants configuration
        baked_config = {
            variant_name: {
                "name": variant.name,
                "reflection_style": variant.reflection_style.value,
                "prompt_format": variant.prompt_format.value,
                "max_reflection_tokens": variant.max_reflection_tokens,
                "temperature": variant.temperature,
                "metadata": variant.metadata,
            }
            for variant_name, variant in self.baked_variants.items()
        }

        with open(output_path / "baked_variants.json", "w") as f:
            json.dump(baked_config, f, indent=2)

        print(f"💾 Saved A/B test results to {output_path}")


def create_standard_ab_variants() -> list[ABVariant]:
    """Create a standard set of A/B test variants for comparison."""
    return [
        ABVariant(
            name="step_by_step_standard",
            reflection_style=ReflectionStyle.STEP_BY_STEP,
            prompt_format=PromptFormat.STANDARD,
            max_reflection_tokens=128,
        ),
        ABVariant(
            name="critical_thinking_explicit",
            reflection_style=ReflectionStyle.CRITICAL_THINKING,
            prompt_format=PromptFormat.EXPLICIT,
            max_reflection_tokens=150,
        ),
        ABVariant(
            name="analytical_structured",
            reflection_style=ReflectionStyle.ANALYTICAL,
            prompt_format=PromptFormat.CHAIN_OF_THOUGHT,
            max_reflection_tokens=160,
        ),
        ABVariant(
            name="concise_minimal",
            reflection_style=ReflectionStyle.CONCISE,
            prompt_format=PromptFormat.MINIMAL,
            max_reflection_tokens=64,
        ),
        ABVariant(
            name="verbose_correction",
            reflection_style=ReflectionStyle.VERBOSE,
            prompt_format=PromptFormat.SELF_CORRECTION,
            max_reflection_tokens=256,
        ),
        ABVariant(
            name="exploratory_standard",
            reflection_style=ReflectionStyle.EXPLORATORY,
            prompt_format=PromptFormat.STANDARD,
            max_reflection_tokens=140,
        ),
    ]


def create_optimization_test_questions() -> list[str]:
    """Create test questions for A/B optimization."""
    return [
        "What are the key differences between supervised and unsupervised machine learning?",
        "How does blockchain technology ensure security and transparency?",
        "Explain the process of photosynthesis in plants.",
        "What factors contribute to climate change and what are potential solutions?",
        "How do neural networks learn and make predictions?",
        "What are the principles of object-oriented programming?",
        "Describe the structure and function of DNA in living organisms.",
        "How does quantum computing differ from classical computing?",
        "What are the main causes and effects of economic inflation?",
        "Explain how the internet works from a technical perspective.",
        "What are the ethical considerations in artificial intelligence development?",
        "How do vaccines work to prevent diseases?",
        "What is the relationship between supply and demand in economics?",
        "Describe the process of protein synthesis in cells.",
        "How do search engines rank and retrieve information?",
    ]


if __name__ == "__main__":
    # Demo A/B testing system
    from .config import get_training_config

    print("🧪 Thought-Token A/B Baking System Demo")
    print("=" * 50)

    # Initialize A/B baker
    config = get_training_config()
    baker = ThoughtTokenABBaker(config)

    # Create test variants
    variants = create_standard_ab_variants()[:3]  # Test first 3 for demo
    test_questions = create_optimization_test_questions()[:5]  # Test with 5 questions

    print(f"Testing {len(variants)} variants with {len(test_questions)} questions:")
    for variant in variants:
        print(f"  • {variant.name}: {variant.reflection_style.value} + {variant.prompt_format.value}")
    print()

    # Run A/B test suite
    test_suite = baker.run_ab_test_suite(
        test_name="reflection_style_optimization",
        variants=variants,
        test_questions=test_questions,
        parallel=False,  # Sequential for demo
    )

    # Show results
    print("\n📊 A/B Test Results:")
    sorted_results = sorted(test_suite.results, key=lambda r: r.overall_score(), reverse=True)

    for i, result in enumerate(sorted_results, 1):
        print(f"{i}. {result.variant.name}:")
        print(f"   Overall Score: {result.overall_score():.3f}")
        print(f"   Success Rate: {result.success_rate:.1%}")
        print(f"   Quality Score: {result.quality_score:.3f}")
        print(f"   Leak Rate: {result.leak_rate:.1%}")
        print(f"   Execution Time: {result.execution_time:.2f}s")

    # Bake winners
    print("\n🔥 Baking Results:")
    baked_variants = baker.bake_winning_variants()

    if baked_variants:
        for test_name, winner in baked_variants.items():
            print(f"  🏆 {test_name}: {winner.name}")
    else:
        print("  No clear winners to bake (confidence too low)")

    print("\n✅ A/B Testing System Demo Complete")
