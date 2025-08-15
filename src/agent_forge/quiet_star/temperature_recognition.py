"""
Temperature self-recognition system for Quiet-STaR.
Implements D1-D2) Dynamic temperature adjustment based on confidence and context complexity.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import QuietSTaRConfig
from .sampler import ThoughtSampler


class ConfidenceLevel(Enum):
    """Model's confidence in its current reasoning."""

    VERY_LOW = "very_low"  # < 0.2
    LOW = "low"  # 0.2 - 0.4
    MODERATE = "moderate"  # 0.4 - 0.6
    HIGH = "high"  # 0.6 - 0.8
    VERY_HIGH = "very_high"  # 0.8+


class ContextComplexity(Enum):
    """Assessed complexity of the current context."""

    SIMPLE = "simple"  # Factual, straightforward
    MODERATE = "moderate"  # Some reasoning required
    COMPLEX = "complex"  # Multi-step reasoning
    VERY_COMPLEX = "very_complex"  # Deep analysis needed


class TemperatureStrategy(Enum):
    """Different temperature adjustment strategies."""

    CONSERVATIVE = "conservative"  # Lower temps, more deterministic
    BALANCED = "balanced"  # Standard adaptive approach
    EXPLORATORY = "exploratory"  # Higher temps, more creative
    CONTEXT_ADAPTIVE = "context_adaptive"  # Fully context-dependent


@dataclass
class ConfidenceSignals:
    """Signals used to assess model confidence."""

    entropy_score: float = 0.0  # Token prediction entropy
    consistency_score: float = 0.0  # Multiple sampling consistency
    uncertainty_markers: int = 0  # "maybe", "possibly", etc.
    hedging_language: int = 0  # "I think", "it seems", etc.
    reflection_confidence: float = 0.0  # Confidence in thought process
    token_probability_variance: float = 0.0  # Variance in token probabilities

    def overall_confidence(self) -> float:
        """Calculate overall confidence score (0-1)."""
        # Combine signals with weights
        entropy_factor = max(
            0, 1 - (self.entropy_score / 3.0)
        )  # Lower entropy = higher confidence
        consistency_factor = self.consistency_score
        language_penalty = min(
            (self.uncertainty_markers + self.hedging_language) * 0.1, 0.5
        )
        reflection_factor = self.reflection_confidence
        variance_penalty = min(self.token_probability_variance * 2, 0.3)

        confidence = (
            entropy_factor * 0.3
            + consistency_factor * 0.25
            + reflection_factor * 0.25
            + (1 - language_penalty) * 0.1
            + (1 - variance_penalty) * 0.1
        )

        return max(0.0, min(1.0, confidence))


@dataclass
class ComplexitySignals:
    """Signals used to assess context complexity."""

    question_length: int = 0  # Character count
    question_word_count: int = 0  # Word count
    nested_clauses: int = 0  # Grammatical complexity
    technical_terms: int = 0  # Domain-specific vocabulary
    multiple_concepts: int = 0  # Number of distinct concepts
    causal_relationships: int = 0  # "because", "therefore", etc.
    conditional_statements: int = 0  # "if", "when", "unless"
    comparison_requests: int = 0  # "compare", "contrast", etc.
    analysis_requests: int = 0  # "analyze", "evaluate", etc.

    def overall_complexity(self) -> float:
        """Calculate overall complexity score (0-1)."""
        # Length factors
        length_factor = min(self.question_length / 500, 1.0)
        word_factor = min(self.question_word_count / 50, 1.0)

        # Structural complexity
        structure_factor = min(
            (self.nested_clauses + self.conditional_statements) * 0.2, 1.0
        )

        # Content complexity
        content_factor = min(
            (
                self.technical_terms * 0.15
                + self.multiple_concepts * 0.1
                + self.causal_relationships * 0.1
                + self.comparison_requests * 0.2
                + self.analysis_requests * 0.25
            ),
            1.0,
        )

        complexity = (
            length_factor * 0.2
            + word_factor * 0.15
            + structure_factor * 0.25
            + content_factor * 0.4
        )

        return max(0.0, min(1.0, complexity))


@dataclass
class TemperatureRecommendation:
    """Temperature recommendation with reasoning."""

    base_temperature: float
    adjusted_temperature: float
    confidence_level: ConfidenceLevel
    complexity_level: ContextComplexity
    strategy: TemperatureStrategy
    reasoning: str
    confidence_signals: ConfidenceSignals
    complexity_signals: ComplexitySignals
    adjustment_factor: float = 1.0


class TemperatureSelfRecognition:
    """
    Self-recognition system for dynamic temperature adjustment.
    Analyzes confidence and context complexity to optimize sampling temperature.
    """

    def __init__(
        self,
        config: QuietSTaRConfig,
        strategy: TemperatureStrategy = TemperatureStrategy.BALANCED,
    ):
        self.config = config
        self.strategy = strategy
        self.base_temperature = getattr(config, "temperature", 0.7)

        # Temperature bounds
        self.min_temperature = 0.1
        self.max_temperature = 1.5

        # Calibration history for adaptive learning
        self.calibration_history: list[dict[str, Any]] = []

        # Pattern recognition for complexity assessment
        self.complexity_patterns = self._initialize_complexity_patterns()
        self.confidence_patterns = self._initialize_confidence_patterns()

    def _initialize_complexity_patterns(self) -> dict[str, list[str]]:
        """Initialize patterns for complexity detection."""
        return {
            "technical_terms": [
                "algorithm",
                "neural",
                "quantum",
                "molecule",
                "ecosystem",
                "methodology",
                "paradigm",
                "framework",
                "architecture",
                "implementation",
                "optimization",
                "synchronization",
            ],
            "analysis_requests": [
                "analyze",
                "evaluate",
                "assess",
                "critique",
                "examine",
                "investigate",
                "explore",
                "scrutinize",
                "dissect",
            ],
            "comparison_requests": [
                "compare",
                "contrast",
                "differentiate",
                "distinguish",
                "versus",
                "versus",
                "relative to",
                "in relation to",
            ],
            "causal_relationships": [
                "because",
                "therefore",
                "thus",
                "hence",
                "consequently",
                "as a result",
                "due to",
                "leads to",
                "causes",
                "results in",
            ],
            "conditional_statements": [
                "if",
                "when",
                "unless",
                "provided that",
                "assuming",
                "given that",
                "in case",
                "suppose",
                "imagine",
            ],
        }

    def _initialize_confidence_patterns(self) -> dict[str, list[str]]:
        """Initialize patterns for confidence detection."""
        return {
            "uncertainty_markers": [
                "maybe",
                "perhaps",
                "possibly",
                "might",
                "could be",
                "uncertain",
                "unclear",
                "ambiguous",
                "debatable",
            ],
            "hedging_language": [
                "i think",
                "it seems",
                "appears to",
                "suggests",
                "likely",
                "probably",
                "tends to",
                "generally",
            ],
            "confidence_markers": [
                "definitely",
                "certainly",
                "clearly",
                "obviously",
                "undoubtedly",
                "without question",
                "absolutely",
            ],
        }

    def assess_confidence(
        self,
        logits: torch.Tensor,
        generated_text: str = "",
        reflection_text: str = "",
        multiple_samples: list[str] | None = None,
    ) -> ConfidenceSignals:
        """Assess model confidence using multiple signals."""

        signals = ConfidenceSignals()

        # Entropy-based confidence
        if logits is not None and len(logits.shape) >= 2:
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            signals.entropy_score = entropy.mean().item()

            # Token probability variance
            max_probs = torch.max(probs, dim=-1)[0]
            signals.token_probability_variance = torch.var(max_probs).item()

        # Language-based confidence assessment
        combined_text = f"{generated_text} {reflection_text}".lower()

        signals.uncertainty_markers = sum(
            combined_text.count(marker)
            for marker in self.confidence_patterns["uncertainty_markers"]
        )

        signals.hedging_language = sum(
            combined_text.count(hedge)
            for hedge in self.confidence_patterns["hedging_language"]
        )

        confidence_markers = sum(
            combined_text.count(marker)
            for marker in self.confidence_patterns["confidence_markers"]
        )

        # Consistency assessment from multiple samples
        if multiple_samples and len(multiple_samples) > 1:
            signals.consistency_score = self._calculate_consistency(multiple_samples)
        else:
            signals.consistency_score = 0.5  # Default moderate consistency

        # Reflection confidence (if reflection available)
        if reflection_text:
            signals.reflection_confidence = self._assess_reflection_confidence(
                reflection_text
            )

        return signals

    def assess_context_complexity(
        self, question: str, context: str = ""
    ) -> ComplexitySignals:
        """Assess the complexity of the current context."""

        signals = ComplexitySignals()
        combined_text = f"{question} {context}".lower()

        # Basic metrics
        signals.question_length = len(question)
        signals.question_word_count = len(question.split())

        # Pattern-based complexity assessment
        signals.technical_terms = sum(
            combined_text.count(term)
            for term in self.complexity_patterns["technical_terms"]
        )

        signals.analysis_requests = sum(
            combined_text.count(req)
            for req in self.complexity_patterns["analysis_requests"]
        )

        signals.comparison_requests = sum(
            combined_text.count(req)
            for req in self.complexity_patterns["comparison_requests"]
        )

        signals.causal_relationships = sum(
            combined_text.count(rel)
            for rel in self.complexity_patterns["causal_relationships"]
        )

        signals.conditional_statements = sum(
            combined_text.count(cond)
            for cond in self.complexity_patterns["conditional_statements"]
        )

        # Structural complexity heuristics
        signals.nested_clauses = combined_text.count(",") + combined_text.count(";")
        signals.multiple_concepts = len(
            set(
                word
                for word in combined_text.split()
                if len(word) > 6 and word.isalpha()
            )
        )

        return signals

    def _calculate_consistency(self, samples: list[str]) -> float:
        """Calculate consistency between multiple samples."""
        if len(samples) < 2:
            return 1.0

        # Simple consistency measure based on word overlap
        word_sets = [set(sample.lower().split()) for sample in samples]

        # Calculate pairwise Jaccard similarity
        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                if union > 0:
                    similarities.append(intersection / union)

        return np.mean(similarities) if similarities else 0.0

    def _assess_reflection_confidence(self, reflection_text: str) -> float:
        """Assess confidence based on reflection content."""
        reflection_lower = reflection_text.lower()

        # Positive indicators
        structured_thinking = sum(
            reflection_lower.count(word)
            for word in ["step", "first", "then", "therefore", "because"]
        )
        definitive_language = sum(
            reflection_lower.count(word)
            for word in ["clear", "obvious", "definitely", "certain"]
        )

        # Negative indicators
        uncertainty = sum(
            reflection_lower.count(word)
            for word in ["unsure", "unclear", "complex", "difficult"]
        )

        # Calculate confidence score
        positive_score = min(
            structured_thinking * 0.1 + definitive_language * 0.15, 0.5
        )
        negative_score = min(uncertainty * 0.2, 0.3)

        return max(0.0, min(1.0, 0.5 + positive_score - negative_score))

    def recommend_temperature(
        self,
        question: str,
        context: str = "",
        logits: torch.Tensor | None = None,
        generated_text: str = "",
        reflection_text: str = "",
        multiple_samples: list[str] | None = None,
    ) -> TemperatureRecommendation:
        """Generate temperature recommendation based on current context."""

        # Assess confidence and complexity
        confidence_signals = self.assess_confidence(
            logits, generated_text, reflection_text, multiple_samples
        )
        complexity_signals = self.assess_context_complexity(question, context)

        # Determine confidence and complexity levels
        confidence_score = confidence_signals.overall_confidence()
        complexity_score = complexity_signals.overall_complexity()

        confidence_level = self._score_to_confidence_level(confidence_score)
        complexity_level = self._score_to_complexity_level(complexity_score)

        # Calculate temperature adjustment
        adjusted_temperature, adjustment_factor, reasoning = (
            self._calculate_temperature_adjustment(
                confidence_score, complexity_score, confidence_level, complexity_level
            )
        )

        return TemperatureRecommendation(
            base_temperature=self.base_temperature,
            adjusted_temperature=adjusted_temperature,
            confidence_level=confidence_level,
            complexity_level=complexity_level,
            strategy=self.strategy,
            reasoning=reasoning,
            confidence_signals=confidence_signals,
            complexity_signals=complexity_signals,
            adjustment_factor=adjustment_factor,
        )

    def _score_to_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert confidence score to categorical level."""
        if score < 0.2:
            return ConfidenceLevel.VERY_LOW
        elif score < 0.4:
            return ConfidenceLevel.LOW
        elif score < 0.6:
            return ConfidenceLevel.MODERATE
        elif score < 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH

    def _score_to_complexity_level(self, score: float) -> ContextComplexity:
        """Convert complexity score to categorical level."""
        if score < 0.25:
            return ContextComplexity.SIMPLE
        elif score < 0.5:
            return ContextComplexity.MODERATE
        elif score < 0.75:
            return ContextComplexity.COMPLEX
        else:
            return ContextComplexity.VERY_COMPLEX

    def _calculate_temperature_adjustment(
        self,
        confidence_score: float,
        complexity_score: float,
        confidence_level: ConfidenceLevel,
        complexity_level: ContextComplexity,
    ) -> tuple[float, float, str]:
        """Calculate temperature adjustment based on strategy."""

        base_temp = self.base_temperature

        if self.strategy == TemperatureStrategy.CONSERVATIVE:
            # Lower temperatures for more deterministic outputs
            confidence_factor = 0.7 + (confidence_score * 0.2)  # 0.7-0.9 range
            complexity_factor = 0.9 - (complexity_score * 0.3)  # 0.6-0.9 range
            adjustment_factor = (confidence_factor + complexity_factor) / 2
            reasoning = f"Conservative strategy: prioritizing deterministic output (conf={confidence_score:.2f}, comp={complexity_score:.2f})"

        elif self.strategy == TemperatureStrategy.EXPLORATORY:
            # Higher temperatures for more creative outputs
            confidence_factor = 1.1 + (1 - confidence_score) * 0.3  # 0.8-1.4 range
            complexity_factor = 1.0 + (complexity_score * 0.2)  # 1.0-1.2 range
            adjustment_factor = (confidence_factor + complexity_factor) / 2
            reasoning = f"Exploratory strategy: encouraging creative exploration (conf={confidence_score:.2f}, comp={complexity_score:.2f})"

        elif self.strategy == TemperatureStrategy.CONTEXT_ADAPTIVE:
            # Fully adaptive based on context
            if complexity_level in [
                ContextComplexity.COMPLEX,
                ContextComplexity.VERY_COMPLEX,
            ]:
                if confidence_level in [
                    ConfidenceLevel.HIGH,
                    ConfidenceLevel.VERY_HIGH,
                ]:
                    adjustment_factor = (
                        0.8  # High conf + complex = lower temp for precision
                    )
                    reasoning = "High confidence on complex topic: lowering temperature for precision"
                else:
                    adjustment_factor = (
                        1.3  # Low conf + complex = higher temp for exploration
                    )
                    reasoning = "Low confidence on complex topic: raising temperature for exploration"
            else:
                if confidence_level in [
                    ConfidenceLevel.HIGH,
                    ConfidenceLevel.VERY_HIGH,
                ]:
                    adjustment_factor = 0.9  # High conf + simple = slightly lower temp
                    reasoning = (
                        "High confidence on simple topic: slightly lower temperature"
                    )
                else:
                    adjustment_factor = 1.1  # Low conf + simple = slightly higher temp
                    reasoning = (
                        "Low confidence on simple topic: slightly higher temperature"
                    )

        else:  # BALANCED strategy (default)
            # Balanced approach: inverse relationship between confidence and temperature
            confidence_factor = 1.2 - (confidence_score * 0.4)  # 0.8-1.2 range
            complexity_factor = 0.9 + (complexity_score * 0.2)  # 0.9-1.1 range
            adjustment_factor = (confidence_factor + complexity_factor) / 2
            reasoning = f"Balanced strategy: adjusting based on confidence and complexity (conf={confidence_score:.2f}, comp={complexity_score:.2f})"

        # Apply bounds
        adjusted_temperature = max(
            self.min_temperature,
            min(self.max_temperature, base_temp * adjustment_factor),
        )

        return adjusted_temperature, adjustment_factor, reasoning

    def update_calibration(
        self, recommendation: TemperatureRecommendation, outcome_quality: float
    ):
        """Update calibration based on outcome quality feedback."""
        calibration_entry = {
            "confidence_score": recommendation.confidence_signals.overall_confidence(),
            "complexity_score": recommendation.complexity_signals.overall_complexity(),
            "recommended_temperature": recommendation.adjusted_temperature,
            "outcome_quality": outcome_quality,
            "strategy": recommendation.strategy.value,
        }

        self.calibration_history.append(calibration_entry)

        # Keep only recent history
        if len(self.calibration_history) > 1000:
            self.calibration_history = self.calibration_history[-1000:]

    def get_calibration_stats(self) -> dict[str, Any]:
        """Get statistics about calibration performance."""
        if not self.calibration_history:
            return {"message": "No calibration data available"}

        # Calculate average outcome quality by temperature ranges
        temp_ranges = {"low": [], "medium": [], "high": []}

        for entry in self.calibration_history:
            temp = entry["recommended_temperature"]
            quality = entry["outcome_quality"]

            if temp < 0.5:
                temp_ranges["low"].append(quality)
            elif temp < 1.0:
                temp_ranges["medium"].append(quality)
            else:
                temp_ranges["high"].append(quality)

        stats = {
            "total_samples": len(self.calibration_history),
            "temperature_performance": {
                range_name: {
                    "count": len(qualities),
                    "avg_quality": np.mean(qualities) if qualities else 0,
                    "std_quality": np.std(qualities) if qualities else 0,
                }
                for range_name, qualities in temp_ranges.items()
            },
            "strategy_performance": {},
        }

        # Performance by strategy
        strategy_groups = {}
        for entry in self.calibration_history:
            strategy = entry["strategy"]
            if strategy not in strategy_groups:
                strategy_groups[strategy] = []
            strategy_groups[strategy].append(entry["outcome_quality"])

        for strategy, qualities in strategy_groups.items():
            stats["strategy_performance"][strategy] = {
                "count": len(qualities),
                "avg_quality": np.mean(qualities),
                "std_quality": np.std(qualities),
            }

        return stats


class AdaptiveTemperatureSampler(ThoughtSampler):
    """
    Extended ThoughtSampler with adaptive temperature recognition.
    Dynamically adjusts temperature during generation.
    """

    def __init__(
        self,
        config: QuietSTaRConfig,
        tokenizer,
        strategy: TemperatureStrategy = TemperatureStrategy.BALANCED,
    ):
        super().__init__(config, tokenizer)
        self.temp_recognition = TemperatureSelfRecognition(config, strategy)
        self.adaptive_enabled = True

    def sample_with_adaptive_temperature(
        self,
        model,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        question: str = "",
        context: str = "",
        force_thoughts: bool = None,
        collect_calibration_data: bool = False,
    ):
        """Sample with adaptive temperature adjustment."""

        if not self.adaptive_enabled:
            # Fall back to standard sampling
            return self.sample_with_thoughts(
                model, input_ids, max_new_tokens, force_thoughts
            )

        # Get initial temperature recommendation
        initial_recommendation = self.temp_recognition.recommend_temperature(
            question=question, context=context
        )

        # Use recommended temperature for sampling
        original_temperature = getattr(self, "temperature", 0.7)
        self.temperature = initial_recommendation.adjusted_temperature

        # Sample with thoughts using adaptive temperature
        result = self.sample_with_thoughts(
            model, input_ids, max_new_tokens, force_thoughts
        )

        # Post-generation analysis for further calibration
        if hasattr(result, "generated_ids") and collect_calibration_data:
            generated_text = self.tokenizer.decode(
                result.generated_ids[0], skip_special_tokens=True
            )

            # Get final recommendation with generated text
            final_recommendation = self.temp_recognition.recommend_temperature(
                question=question,
                context=context,
                generated_text=generated_text,
                reflection_text=getattr(result, "thought_text", ""),
            )

            # Add recommendation to result for external evaluation
            result.temperature_recommendation = final_recommendation

        # Restore original temperature
        self.temperature = original_temperature

        return result

    def enable_adaptation(self, enabled: bool = True):
        """Enable or disable adaptive temperature."""
        self.adaptive_enabled = enabled

    def set_strategy(self, strategy: TemperatureStrategy):
        """Change temperature adaptation strategy."""
        self.temp_recognition.strategy = strategy

    def get_temperature_stats(self) -> dict[str, Any]:
        """Get temperature adaptation statistics."""
        return self.temp_recognition.get_calibration_stats()


# Preset temperature strategies for different use cases
TEMPERATURE_STRATEGIES = {
    "factual_qa": TemperatureStrategy.CONSERVATIVE,  # Factual Q&A needs deterministic answers
    "creative_writing": TemperatureStrategy.EXPLORATORY,  # Creative tasks benefit from exploration
    "code_generation": TemperatureStrategy.CONSERVATIVE,  # Code needs precision
    "analysis": TemperatureStrategy.CONTEXT_ADAPTIVE,  # Analysis varies by complexity
    "general": TemperatureStrategy.BALANCED,  # General-purpose balanced approach
    "research": TemperatureStrategy.CONTEXT_ADAPTIVE,  # Research adapts to topic complexity
}


def create_temperature_recognizer(
    config: QuietSTaRConfig, use_case: str = "general"
) -> TemperatureSelfRecognition:
    """Create temperature recognizer for specific use case."""
    strategy = TEMPERATURE_STRATEGIES.get(use_case, TemperatureStrategy.BALANCED)
    return TemperatureSelfRecognition(config, strategy)


if __name__ == "__main__":
    # Demo temperature self-recognition system
    from .config import get_training_config

    print("🌡️  Temperature Self-Recognition System Demo")
    print("=" * 50)

    # Initialize system
    config = get_training_config()
    temp_recognizer = TemperatureSelfRecognition(config, TemperatureStrategy.BALANCED)

    print(f"Initialized with base temperature: {temp_recognizer.base_temperature}")
    print(f"Strategy: {temp_recognizer.strategy.value}")
    print(
        f"Temperature bounds: {temp_recognizer.min_temperature} - {temp_recognizer.max_temperature}"
    )
    print()

    # Test scenarios
    test_scenarios = [
        {
            "question": "What is 2 + 2?",
            "context": "",
            "description": "Simple factual question",
        },
        {
            "question": "Analyze the long-term economic implications of artificial intelligence automation on global employment patterns and social structures.",
            "context": "Consider technological unemployment, wealth inequality, and societal adaptation mechanisms.",
            "description": "Complex analytical question",
        },
        {
            "question": "How do neural networks learn from data?",
            "context": "Focus on the mathematical foundations and training process.",
            "description": "Technical explanation request",
        },
        {
            "question": "Write a creative story about a robot who discovers emotions.",
            "context": "Make it engaging and original with character development.",
            "description": "Creative writing task",
        },
        {
            "question": "What might be the potential risks of quantum computing for current encryption methods?",
            "context": "Consider both theoretical and practical aspects.",
            "description": "Speculative technical analysis",
        },
    ]

    print("📊 Temperature Recommendations by Scenario:")
    print()

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"Scenario {i}: {scenario['description']}")
        print(
            f"  Question: {scenario['question'][:80]}{'...' if len(scenario['question']) > 80 else ''}"
        )

        # Get temperature recommendation
        recommendation = temp_recognizer.recommend_temperature(
            question=scenario["question"], context=scenario["context"]
        )

        print(
            f"  Confidence: {recommendation.confidence_level.value} "
            f"({recommendation.confidence_signals.overall_confidence():.2f})"
        )
        print(
            f"  Complexity: {recommendation.complexity_level.value} "
            f"({recommendation.complexity_signals.overall_complexity():.2f})"
        )
        print(
            f"  Temperature: {recommendation.base_temperature:.2f} → {recommendation.adjusted_temperature:.2f} "
            f"(×{recommendation.adjustment_factor:.2f})"
        )
        print(f"  Reasoning: {recommendation.reasoning}")
        print()

    # Test different strategies
    print("🎯 Strategy Comparison for Complex Question:")
    complex_question = test_scenarios[1]["question"]
    complex_context = test_scenarios[1]["context"]

    strategies = [
        TemperatureStrategy.CONSERVATIVE,
        TemperatureStrategy.BALANCED,
        TemperatureStrategy.EXPLORATORY,
        TemperatureStrategy.CONTEXT_ADAPTIVE,
    ]

    for strategy in strategies:
        recognizer = TemperatureSelfRecognition(config, strategy)
        recommendation = recognizer.recommend_temperature(
            complex_question, complex_context
        )

        print(
            f"  {strategy.value:15}: {recommendation.adjusted_temperature:.2f} - {recommendation.reasoning[:80]}..."
        )

    print()
    print("✅ Temperature Self-Recognition System Demo Complete")
    print()
    print("Key Features Demonstrated:")
    print("  • Confidence assessment using multiple signals")
    print("  • Context complexity analysis")
    print("  • Dynamic temperature adjustment strategies")
    print("  • Use-case specific optimization")
    print("  • Calibration tracking for continuous improvement")
