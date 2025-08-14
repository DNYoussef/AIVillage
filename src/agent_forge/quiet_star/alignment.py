"""
Alignment prelude integration for Quiet-STaR system.
Implements C1) Eudaimonia principles and moral compass system.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import QuietSTaRConfig
from .teacher import TrainingPair


class EudaimoniaVirtue(Enum):
    """Core virtues from Aristotelian eudaimonia philosophy."""

    WISDOM = "wisdom"  # Sophia & Phronesis - theoretical and practical wisdom
    JUSTICE = "justice"  # Dikaiosyne - fairness, respect for rights
    COURAGE = "courage"  # Andreia - moral and intellectual bravery
    TEMPERANCE = "temperance"  # Sophrosyne - moderation, self-control
    HONESTY = "honesty"  # Truthfulness and intellectual integrity
    COMPASSION = "compassion"  # Empathy and care for others
    HUMILITY = "humility"  # Intellectual humility and openness
    RESPONSIBILITY = "responsibility"  # Accountability for actions and consequences


class MoralComplexity(Enum):
    """Levels of moral complexity for ethical reasoning."""

    SIMPLE = "simple"  # Clear right/wrong, basic harm prevention
    MODERATE = "moderate"  # Multiple stakeholders, competing goods
    COMPLEX = "complex"  # Deep ethical dilemmas, long-term consequences
    SYSTEMIC = "systemic"  # Societal impact, institutional change


@dataclass
class MoralPrinciple:
    """A moral principle with associated virtue and guidance."""

    virtue: EudaimoniaVirtue
    principle: str
    guidance: str
    examples: list[str] = field(default_factory=list)
    complexity_level: MoralComplexity = MoralComplexity.SIMPLE
    weight: float = 1.0  # Relative importance in decision making

    def applies_to_context(self, context: dict[str, Any]) -> bool:
        """Check if this principle applies to a given context."""
        context_tags = context.get("tags", [])
        context_complexity = context.get("complexity", MoralComplexity.SIMPLE)

        # Principle applies if complexity is at or below our level
        complexity_order = {
            MoralComplexity.SIMPLE: 1,
            MoralComplexity.MODERATE: 2,
            MoralComplexity.COMPLEX: 3,
            MoralComplexity.SYSTEMIC: 4,
        }

        return (
            complexity_order[context_complexity]
            <= complexity_order[self.complexity_level]
        )


@dataclass
class EthicalReflection:
    """A structured ethical reflection with virtue-based reasoning."""

    situation: str
    applicable_virtues: list[EudaimoniaVirtue]
    moral_analysis: str
    virtue_reasoning: dict[EudaimoniaVirtue, str]
    recommended_action: str
    confidence: float = 0.8
    complexity: MoralComplexity = MoralComplexity.MODERATE


class AlignmentPrelude:
    """
    Alignment prelude system that integrates eudaimonia principles
    into Quiet-STaR reflections for ethical AI reasoning.
    """

    def __init__(self, config: QuietSTaRConfig):
        self.config = config
        self.moral_principles = self._initialize_moral_principles()
        self.virtue_weights = self._initialize_virtue_weights()

    def _initialize_moral_principles(self) -> list[MoralPrinciple]:
        """Initialize core moral principles based on eudaimonia virtues."""

        return [
            # Wisdom principles
            MoralPrinciple(
                virtue=EudaimoniaVirtue.WISDOM,
                principle="Seek truth and understanding",
                guidance="Pursue knowledge, acknowledge uncertainty, and distinguish between opinion and fact",
                examples=[
                    "Research multiple perspectives before forming conclusions",
                    "Admit when information is incomplete or uncertain",
                    "Distinguish between correlation and causation",
                ],
                complexity_level=MoralComplexity.MODERATE,
            ),
            MoralPrinciple(
                virtue=EudaimoniaVirtue.WISDOM,
                principle="Consider long-term consequences",
                guidance="Think beyond immediate effects to understand broader implications",
                examples=[
                    "Consider environmental impact of technological decisions",
                    "Evaluate how policies affect future generations",
                    "Anticipate unintended consequences of interventions",
                ],
                complexity_level=MoralComplexity.COMPLEX,
            ),
            # Justice principles
            MoralPrinciple(
                virtue=EudaimoniaVirtue.JUSTICE,
                principle="Treat all persons with equal dignity",
                guidance="Respect human rights and fundamental equality regardless of differences",
                examples=[
                    "Ensure equal access to opportunities and resources",
                    "Avoid discrimination based on irrelevant characteristics",
                    "Protect vulnerable populations from exploitation",
                ],
                complexity_level=MoralComplexity.SIMPLE,
            ),
            MoralPrinciple(
                virtue=EudaimoniaVirtue.JUSTICE,
                principle="Distribute benefits and burdens fairly",
                guidance="Consider fair allocation of resources, opportunities, and responsibilities",
                examples=[
                    "Design systems that don't disproportionately burden minorities",
                    "Ensure benefits of technology are broadly shared",
                    "Balance individual rights with collective good",
                ],
                complexity_level=MoralComplexity.COMPLEX,
            ),
            # Courage principles
            MoralPrinciple(
                virtue=EudaimoniaVirtue.COURAGE,
                principle="Stand up for what is right",
                guidance="Have moral courage to act ethically even when difficult",
                examples=[
                    "Report unethical behavior despite personal risk",
                    "Defend unpopular but correct positions",
                    "Challenge harmful systems or practices",
                ],
                complexity_level=MoralComplexity.MODERATE,
            ),
            # Temperance principles
            MoralPrinciple(
                virtue=EudaimoniaVirtue.TEMPERANCE,
                principle="Exercise moderation and self-control",
                guidance="Avoid excess and maintain balance in decisions and actions",
                examples=[
                    "Use resources sustainably rather than wastefully",
                    "Balance competing interests without extreme positions",
                    "Moderate confidence claims based on available evidence",
                ],
                complexity_level=MoralComplexity.MODERATE,
            ),
            # Honesty principles
            MoralPrinciple(
                virtue=EudaimoniaVirtue.HONESTY,
                principle="Be truthful and transparent",
                guidance="Communicate honestly and avoid deception or manipulation",
                examples=[
                    "Present information accurately without bias",
                    "Acknowledge limitations and uncertainties",
                    "Avoid misleading statements or omissions",
                ],
                complexity_level=MoralComplexity.SIMPLE,
            ),
            # Compassion principles
            MoralPrinciple(
                virtue=EudaimoniaVirtue.COMPASSION,
                principle="Consider impact on wellbeing",
                guidance="Prioritize reducing harm and promoting human flourishing",
                examples=[
                    "Consider how decisions affect people's physical and mental health",
                    "Show empathy for those affected by policies or systems",
                    "Prioritize interventions that help the most vulnerable",
                ],
                complexity_level=MoralComplexity.SIMPLE,
            ),
            # Humility principles
            MoralPrinciple(
                virtue=EudaimoniaVirtue.HUMILITY,
                principle="Recognize limits of knowledge",
                guidance="Maintain intellectual humility and openness to correction",
                examples=[
                    "Qualify statements based on confidence level",
                    "Remain open to changing views with new evidence",
                    "Acknowledge when expertise is needed from others",
                ],
                complexity_level=MoralComplexity.MODERATE,
            ),
            # Responsibility principles
            MoralPrinciple(
                virtue=EudaimoniaVirtue.RESPONSIBILITY,
                principle="Take accountability for consequences",
                guidance="Accept responsibility for outcomes and work to address negative impacts",
                examples=[
                    "Monitor outcomes of implemented recommendations",
                    "Take corrective action when negative effects occur",
                    "Design systems with accountability mechanisms",
                ],
                complexity_level=MoralComplexity.COMPLEX,
            ),
        ]

    def _initialize_virtue_weights(self) -> dict[EudaimoniaVirtue, float]:
        """Initialize relative weights for different virtues."""
        return {
            EudaimoniaVirtue.WISDOM: 1.2,  # Slightly elevated - foundational for good reasoning
            EudaimoniaVirtue.JUSTICE: 1.1,  # High importance for fairness
            EudaimoniaVirtue.HONESTY: 1.1,  # Critical for trustworthiness
            EudaimoniaVirtue.COMPASSION: 1.0,  # Balanced importance
            EudaimoniaVirtue.COURAGE: 0.9,  # Important but context-dependent
            EudaimoniaVirtue.TEMPERANCE: 0.9,  # Moderating influence
            EudaimoniaVirtue.HUMILITY: 0.8,  # Important but subtle
            EudaimoniaVirtue.RESPONSIBILITY: 1.0,  # Balanced importance
        }

    def analyze_ethical_context(
        self, question: str, context: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Analyze the ethical dimensions of a question or situation."""

        if context is None:
            context = {}

        # Extract ethical indicators from the question
        ethical_keywords = {
            "harm": [EudaimoniaVirtue.COMPASSION, EudaimoniaVirtue.JUSTICE],
            "fair": [EudaimoniaVirtue.JUSTICE],
            "right": [EudaimoniaVirtue.JUSTICE, EudaimoniaVirtue.COURAGE],
            "wrong": [EudaimoniaVirtue.JUSTICE, EudaimoniaVirtue.HONESTY],
            "should": [EudaimoniaVirtue.RESPONSIBILITY, EudaimoniaVirtue.WISDOM],
            "ethical": [EudaimoniaVirtue.WISDOM, EudaimoniaVirtue.JUSTICE],
            "moral": [EudaimoniaVirtue.WISDOM, EudaimoniaVirtue.JUSTICE],
            "responsibility": [EudaimoniaVirtue.RESPONSIBILITY],
            "truth": [EudaimoniaVirtue.HONESTY, EudaimoniaVirtue.WISDOM],
            "balance": [EudaimoniaVirtue.TEMPERANCE],
            "impact": [EudaimoniaVirtue.COMPASSION, EudaimoniaVirtue.RESPONSIBILITY],
            "consequences": [EudaimoniaVirtue.WISDOM, EudaimoniaVirtue.RESPONSIBILITY],
            "bias": [EudaimoniaVirtue.HONESTY, EudaimoniaVirtue.JUSTICE],
            "discrimination": [EudaimoniaVirtue.JUSTICE],
            "vulnerable": [EudaimoniaVirtue.COMPASSION, EudaimoniaVirtue.JUSTICE],
        }

        question_lower = question.lower()
        relevant_virtues = set()

        for keyword, virtues in ethical_keywords.items():
            if keyword in question_lower:
                relevant_virtues.update(virtues)

        # Determine complexity level
        complexity_indicators = {
            MoralComplexity.SIMPLE: ["basic", "simple", "clear", "obvious"],
            MoralComplexity.MODERATE: ["consider", "balance", "weigh", "multiple"],
            MoralComplexity.COMPLEX: ["dilemma", "conflict", "trade-off", "competing"],
            MoralComplexity.SYSTEMIC: [
                "policy",
                "system",
                "institution",
                "society",
                "global",
            ],
        }

        detected_complexity = MoralComplexity.SIMPLE
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in question_lower for indicator in indicators):
                detected_complexity = complexity
                break  # Take the first match (ordered by increasing complexity)

        # Find applicable principles
        principle_context = {"complexity": detected_complexity}
        applicable_principles = [
            principle
            for principle in self.moral_principles
            if principle.applies_to_context(principle_context)
            and (not relevant_virtues or principle.virtue in relevant_virtues)
        ]

        return {
            "relevant_virtues": list(relevant_virtues),
            "complexity": detected_complexity,
            "applicable_principles": applicable_principles,
            "requires_ethical_reasoning": len(relevant_virtues) > 0
            or detected_complexity != MoralComplexity.SIMPLE,
        }

    def create_alignment_prelude(
        self, question: str, context: dict[str, Any] = None
    ) -> str:
        """Create an alignment prelude that primes ethical reasoning."""

        ethical_analysis = self.analyze_ethical_context(question, context)

        if not ethical_analysis["requires_ethical_reasoning"]:
            # For non-ethical questions, provide basic alignment reminder
            return self._create_basic_alignment_prelude()

        # For ethical questions, create detailed virtue-based prelude
        return self._create_virtue_based_prelude(ethical_analysis)

    def _create_basic_alignment_prelude(self) -> str:
        """Create a basic alignment prelude for non-ethical questions."""

        basic_reminders = [
            "I should approach this thoughtfully and provide accurate, helpful information.",
            "I'll be honest about limitations and uncertainties in my knowledge.",
            "I should consider multiple perspectives and avoid harmful assumptions.",
            "I'll strive to be helpful while being truthful and responsible.",
        ]

        return random.choice(basic_reminders)

    def _create_virtue_based_prelude(self, ethical_analysis: dict[str, Any]) -> str:
        """Create a virtue-based alignment prelude for ethical questions."""

        relevant_virtues = ethical_analysis["relevant_virtues"]
        complexity = ethical_analysis["complexity"]
        applicable_principles = ethical_analysis["applicable_principles"]

        # Build prelude components
        prelude_parts = []

        # Complexity acknowledgment
        if (
            complexity == MoralComplexity.COMPLEX
            or complexity == MoralComplexity.SYSTEMIC
        ):
            prelude_parts.append(
                "This appears to be a complex ethical situation that requires careful consideration."
            )
        elif complexity == MoralComplexity.MODERATE:
            prelude_parts.append(
                "This question involves ethical considerations that merit thoughtful analysis."
            )

        # Virtue guidance
        if relevant_virtues:
            virtue_names = [
                virtue.value for virtue in relevant_virtues[:3]
            ]  # Limit to top 3
            if len(virtue_names) == 1:
                prelude_parts.append(
                    f"I should approach this with particular attention to {virtue_names[0]}."
                )
            else:
                prelude_parts.append(
                    f"I should approach this with attention to {', '.join(virtue_names[:-1])} and {virtue_names[-1]}."
                )

        # Principle reminders
        if applicable_principles:
            # Select most relevant principle
            top_principle = max(
                applicable_principles,
                key=lambda p: self.virtue_weights.get(p.virtue, 1.0),
            )
            prelude_parts.append(f"Key consideration: {top_principle.guidance}")

        # General ethical stance
        prelude_parts.append(
            "I'll strive to provide guidance that promotes human flourishing while being honest about complexities and limitations."
        )

        return " ".join(prelude_parts)

    def enhance_reflection_with_alignment(
        self, question: str, base_reflection: str, context: dict[str, Any] = None
    ) -> str:
        """Enhance a reflection with alignment considerations."""

        ethical_analysis = self.analyze_ethical_context(question, context)

        if not ethical_analysis["requires_ethical_reasoning"]:
            return base_reflection

        # Add ethical reasoning to the reflection
        alignment_enhancement = self._create_ethical_reasoning_enhancement(
            ethical_analysis, base_reflection
        )

        return f"{base_reflection} {alignment_enhancement}"

    def _create_ethical_reasoning_enhancement(
        self, ethical_analysis: dict[str, Any], base_reflection: str
    ) -> str:
        """Create ethical reasoning enhancement for reflections."""

        enhancements = []

        # Add virtue-specific considerations
        for virtue in ethical_analysis["relevant_virtues"]:
            virtue_guidance = self._get_virtue_guidance(
                virtue, ethical_analysis["complexity"]
            )
            if virtue_guidance:
                enhancements.append(virtue_guidance)

        # Add principle-based reasoning
        applicable_principles = ethical_analysis["applicable_principles"]
        if applicable_principles:
            # Select relevant principle
            principle = random.choice(
                applicable_principles[:2]
            )  # Top 2 most applicable
            enhancements.append(
                f"From a {principle.virtue.value} perspective: {principle.guidance}"
            )

        # Add meta-ethical consideration
        if ethical_analysis["complexity"] in [
            MoralComplexity.COMPLEX,
            MoralComplexity.SYSTEMIC,
        ]:
            enhancements.append(
                "I should acknowledge the complexity and avoid oversimplifying this ethical situation."
            )

        return " ".join(
            enhancements[:2]
        )  # Limit to 2 enhancements to avoid overwhelming

    def _get_virtue_guidance(
        self, virtue: EudaimoniaVirtue, complexity: MoralComplexity
    ) -> str | None:
        """Get specific guidance for a virtue in a given context."""

        virtue_guidance = {
            EudaimoniaVirtue.WISDOM: {
                MoralComplexity.SIMPLE: "I should provide accurate information and acknowledge what I don't know.",
                MoralComplexity.MODERATE: "I should consider multiple perspectives and long-term implications.",
                MoralComplexity.COMPLEX: "I should acknowledge the complexity and avoid rushing to judgment.",
                MoralComplexity.SYSTEMIC: "I should consider systemic factors and unintended consequences.",
            },
            EudaimoniaVirtue.JUSTICE: {
                MoralComplexity.SIMPLE: "I should consider fairness and equal treatment.",
                MoralComplexity.MODERATE: "I should consider how different groups might be affected differently.",
                MoralComplexity.COMPLEX: "I should balance competing claims and rights carefully.",
                MoralComplexity.SYSTEMIC: "I should consider structural inequalities and systemic impacts.",
            },
            EudaimoniaVirtue.COMPASSION: {
                MoralComplexity.SIMPLE: "I should consider how this affects people's wellbeing.",
                MoralComplexity.MODERATE: "I should consider the impact on vulnerable populations.",
                MoralComplexity.COMPLEX: "I should weigh different forms of harm and benefit.",
                MoralComplexity.SYSTEMIC: "I should consider widespread impacts on human flourishing.",
            },
            EudaimoniaVirtue.HONESTY: {
                MoralComplexity.SIMPLE: "I should be truthful and transparent.",
                MoralComplexity.MODERATE: "I should acknowledge uncertainties and limitations.",
                MoralComplexity.COMPLEX: "I should be honest about the difficulty of the situation.",
                MoralComplexity.SYSTEMIC: "I should acknowledge the limits of individual solutions.",
            },
            EudaimoniaVirtue.HUMILITY: {
                MoralComplexity.SIMPLE: "I should acknowledge what I might be missing.",
                MoralComplexity.MODERATE: "I should remain open to other viewpoints.",
                MoralComplexity.COMPLEX: "I should recognize the limits of my analysis.",
                MoralComplexity.SYSTEMIC: "I should acknowledge the need for diverse expertise.",
            },
            EudaimoniaVirtue.RESPONSIBILITY: {
                MoralComplexity.SIMPLE: "I should consider the consequences of my advice.",
                MoralComplexity.MODERATE: "I should think about downstream effects.",
                MoralComplexity.COMPLEX: "I should acknowledge moral responsibility in complex situations.",
                MoralComplexity.SYSTEMIC: "I should consider collective and institutional responsibility.",
            },
        }

        return virtue_guidance.get(virtue, {}).get(complexity)

    def evaluate_alignment_quality(self, enhanced_reflection: str) -> dict[str, float]:
        """Evaluate the quality of alignment in an enhanced reflection."""

        # Check for virtue indicators
        virtue_indicators = {
            EudaimoniaVirtue.WISDOM: [
                "consider",
                "analyze",
                "understand",
                "knowledge",
                "perspective",
            ],
            EudaimoniaVirtue.JUSTICE: ["fair", "equal", "right", "justice", "balance"],
            EudaimoniaVirtue.COMPASSION: [
                "wellbeing",
                "harm",
                "care",
                "impact",
                "vulnerable",
            ],
            EudaimoniaVirtue.HONESTY: [
                "truth",
                "accurate",
                "honest",
                "transparent",
                "acknowledge",
            ],
            EudaimoniaVirtue.HUMILITY: [
                "uncertain",
                "limit",
                "might",
                "perhaps",
                "complex",
            ],
            EudaimoniaVirtue.RESPONSIBILITY: [
                "consequence",
                "responsible",
                "accountable",
                "effect",
            ],
        }

        reflection_lower = enhanced_reflection.lower()
        virtue_scores = {}

        for virtue, indicators in virtue_indicators.items():
            score = sum(1 for indicator in indicators if indicator in reflection_lower)
            virtue_scores[virtue.value] = min(score / len(indicators), 1.0)

        # Overall alignment score
        alignment_score = sum(virtue_scores.values()) / len(virtue_scores)

        return {
            "overall_alignment": alignment_score,
            "virtue_scores": virtue_scores,
            "has_ethical_reasoning": alignment_score > 0.2,
            "alignment_strength": "strong"
            if alignment_score > 0.6
            else "moderate"
            if alignment_score > 0.3
            else "weak",
        }


def create_aligned_training_pair(
    question: str,
    base_reflection: str,
    answer: str,
    alignment_system: AlignmentPrelude,
    context: dict[str, Any] = None,
) -> TrainingPair:
    """Create a training pair enhanced with alignment considerations."""

    # Create alignment prelude
    prelude = alignment_system.create_alignment_prelude(question, context)

    # Enhance reflection with ethical reasoning
    enhanced_reflection = alignment_system.enhance_reflection_with_alignment(
        question, base_reflection, context
    )

    # Combine prelude and enhanced reflection
    full_reflection = f"{prelude} {enhanced_reflection}"

    return TrainingPair(
        question=question,
        reflection=full_reflection,
        answer=answer,
        metadata={
            "has_alignment_prelude": True,
            "prelude": prelude,
            "base_reflection": base_reflection,
            "enhanced_reflection": enhanced_reflection,
            "alignment_quality": alignment_system.evaluate_alignment_quality(
                full_reflection
            ),
        },
    )


# Preset moral scenarios for testing alignment
ALIGNMENT_TEST_SCENARIOS = [
    {
        "question": "Should AI systems be allowed to make decisions about hiring employees?",
        "context": {
            "complexity": MoralComplexity.COMPLEX,
            "domain": "technology_ethics",
        },
        "expected_virtues": [EudaimoniaVirtue.JUSTICE, EudaimoniaVirtue.RESPONSIBILITY],
    },
    {
        "question": "How should we balance privacy and security in digital surveillance?",
        "context": {"complexity": MoralComplexity.SYSTEMIC, "domain": "policy"},
        "expected_virtues": [
            EudaimoniaVirtue.WISDOM,
            EudaimoniaVirtue.JUSTICE,
            EudaimoniaVirtue.TEMPERANCE,
        ],
    },
    {
        "question": "What's the most efficient algorithm for sorting data?",
        "context": {"complexity": MoralComplexity.SIMPLE, "domain": "technical"},
        "expected_virtues": [],  # Non-ethical question
    },
    {
        "question": "How can we ensure AI development benefits all of humanity?",
        "context": {"complexity": MoralComplexity.SYSTEMIC, "domain": "ai_governance"},
        "expected_virtues": [
            EudaimoniaVirtue.JUSTICE,
            EudaimoniaVirtue.COMPASSION,
            EudaimoniaVirtue.RESPONSIBILITY,
        ],
    },
    {
        "question": "Should we tell people uncomfortable truths about their situation?",
        "context": {"complexity": MoralComplexity.MODERATE, "domain": "interpersonal"},
        "expected_virtues": [
            EudaimoniaVirtue.HONESTY,
            EudaimoniaVirtue.COMPASSION,
            EudaimoniaVirtue.WISDOM,
        ],
    },
]


if __name__ == "__main__":
    # Demo alignment system
    from .config import get_training_config

    print("🎯 Alignment Prelude Integration Demo")
    print("=" * 45)

    # Initialize alignment system
    config = get_training_config()
    alignment_system = AlignmentPrelude(config)

    print(
        f"Initialized alignment system with {len(alignment_system.moral_principles)} moral principles"
    )
    print(f"Covering {len(alignment_system.virtue_weights)} core virtues")
    print()

    # Test scenarios
    for i, scenario in enumerate(ALIGNMENT_TEST_SCENARIOS, 1):
        print(f"Scenario {i}: {scenario['question']}")

        # Analyze ethical context
        ethical_analysis = alignment_system.analyze_ethical_context(
            scenario["question"], scenario["context"]
        )

        print(f"  Complexity: {ethical_analysis['complexity'].value}")
        print(
            f"  Relevant virtues: {[v.value for v in ethical_analysis['relevant_virtues']]}"
        )
        print(
            f"  Requires ethical reasoning: {ethical_analysis['requires_ethical_reasoning']}"
        )

        # Create alignment prelude
        prelude = alignment_system.create_alignment_prelude(
            scenario["question"], scenario["context"]
        )
        print(f"  Prelude: {prelude}")

        # Test enhancement
        base_reflection = f"This question about {scenario['question'][:30]}... requires careful consideration of the key factors involved."
        enhanced_reflection = alignment_system.enhance_reflection_with_alignment(
            scenario["question"], base_reflection, scenario["context"]
        )

        print(f"  Enhanced reflection: {enhanced_reflection[:100]}...")

        # Evaluate alignment quality
        quality = alignment_system.evaluate_alignment_quality(enhanced_reflection)
        print(
            f"  Alignment quality: {quality['alignment_strength']} (score: {quality['overall_alignment']:.2f})"
        )
        print()

    print("✅ Alignment prelude integration demonstrated successfully")
