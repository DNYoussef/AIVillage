"""Culturally Aware ELI5 Explanation Chain
Sprint R-4+AF1: Education Core System - Task A.2.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import random
from typing import Any

# Import AI model clients
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

import wandb

logger = logging.getLogger(__name__)


@dataclass
class CulturalExample:
    """Cultural example for concept explanation."""

    example_id: str
    concept: str
    region: str
    language: str
    example_text: str
    context_type: str  # daily_life, tradition, food, sport, etc.
    age_appropriateness: int  # minimum age
    effectiveness_score: float = 0.0
    usage_count: int = 0
    cultural_sensitivity: float = 1.0  # 0-1 score


@dataclass
class ExplanationTemplate:
    """Template for generating explanations."""

    template_id: str
    concept_category: str
    age_range: tuple[int, int]
    language: str
    template_structure: str
    cultural_adaptability: float
    effectiveness_metrics: dict[str, float]


@dataclass
class ExplanationResult:
    """Result of ELI5 explanation generation."""

    concept: str
    age: int
    language: str
    region: str
    explanation: str
    cultural_examples_used: list[str]
    template_used: str
    readability_score: float
    cultural_relevance_score: float
    engagement_score: float
    generation_time: float
    timestamp: str


class CulturallyAwareELI5:
    """Generate culturally relevant, age-appropriate explanations."""

    def __init__(self, project_name: str = "aivillage-education") -> None:
        self.project_name = project_name
        self.cultural_examples = defaultdict(list)  # (region, concept) -> List[CulturalExample]
        self.explanation_templates = {}  # template_id -> ExplanationTemplate
        self.regional_contexts = {}  # region -> context data
        self.language_adaptations = {}  # language -> adaptation rules

        # AI model clients
        self.anthropic_client = AsyncAnthropic()
        self.openai_client = AsyncOpenAI()

        # Performance tracking
        self.explanation_history = []
        self.effectiveness_metrics = defaultdict(list)

        # Initialize W&B tracking
        self.initialize_wandb_tracking()

        # Load cultural and regional data
        asyncio.create_task(self.initialize_cultural_database())

    def initialize_wandb_tracking(self) -> None:
        """Initialize W&B tracking for ELI5 explanations."""
        try:
            wandb.init(
                project=self.project_name,
                job_type="eli5_cultural_explanations",
                config={
                    "eli5_version": "1.0.0",
                    "supported_ages": "3-18",
                    "cultural_regions": [
                        "north_america",
                        "latin_america",
                        "europe",
                        "south_asia",
                        "east_asia",
                        "southeast_asia",
                        "middle_east",
                        "africa",
                        "oceania",
                    ],
                    "supported_languages": [
                        "en",
                        "es",
                        "hi",
                        "fr",
                        "ar",
                        "pt",
                        "sw",
                        "de",
                        "it",
                        "zh",
                        "ja",
                        "ko",
                    ],
                    "explanation_types": [
                        "scientific",
                        "mathematical",
                        "historical",
                        "literary",
                        "social",
                    ],
                },
            )

            logger.info("W&B ELI5 tracking initialized")

        except Exception as e:
            logger.exception(f"Failed to initialize W&B tracking: {e}")

    async def initialize_cultural_database(self) -> None:
        """Initialize database of cultural examples and regional contexts."""
        # North American examples
        await self.add_cultural_examples_region("north_america")

        # Latin American examples
        await self.add_cultural_examples_region("latin_america")

        # South Asian examples
        await self.add_cultural_examples_region("south_asia")

        # Load explanation templates
        await self.initialize_explanation_templates()

        logger.info("Cultural database initialized")

    async def add_cultural_examples_region(self, region: str) -> None:
        """Add cultural examples for a specific region."""
        if region == "north_america":
            examples = [
                # Mathematics
                (
                    "fractions",
                    "en",
                    "Think of pizza slices! If you cut a pizza into 4 equal pieces and eat 2, you ate 2/4 or 1/2 of the pizza!",
                    "food",
                    6,
                ),
                (
                    "multiplication",
                    "en",
                    "If each football team has 11 players, and there are 2 teams on the field, that's 11 × 2 = 22 players total!",
                    "sport",
                    7,
                ),
                (
                    "geometry",
                    "en",
                    "A square is like a baseball diamond - 4 equal sides and 4 corners (bases)!",
                    "sport",
                    8,
                ),
                # Science
                (
                    "photosynthesis",
                    "en",
                    "Plants eat sunlight like we eat breakfast! They use sunlight, water, and air to make their own food and give us oxygen to breathe.",
                    "daily_life",
                    8,
                ),
                (
                    "gravity",
                    "en",
                    "Gravity is like an invisible friend that always pulls things down - that's why your basketball falls when you drop it!",
                    "daily_life",
                    6,
                ),
                (
                    "ecosystem",
                    "en",
                    "An ecosystem is like your neighborhood - different animals and plants live together and help each other, like how bees help flowers and flowers help bees!",
                    "daily_life",
                    9,
                ),
                # History
                (
                    "american_revolution",
                    "en",
                    "The American Revolution was like kids deciding they didn't want to follow their parents' rules anymore and wanted to make their own family rules!",
                    "daily_life",
                    10,
                ),
                (
                    "westward_expansion",
                    "en",
                    "Imagine your family decided to pack everything and move across the country to build a new house - that's what many families did in the 1800s!",
                    "daily_life",
                    9,
                ),
            ]

        elif region == "latin_america":
            examples = [
                # Mathematics
                (
                    "fractions",
                    "es",
                    "¡Piensa en empanadas! Si mamá hace 8 empanadas y tú comes 2, comiste 2/8 o 1/4 de las empanadas.",
                    "food",
                    6,
                ),
                (
                    "multiplication",
                    "es",
                    "Si cada equipo de fútbol tiene 11 jugadores y hay 2 equipos, son 11 × 2 = 22 jugadores en total!",
                    "sport",
                    7,
                ),
                # Science
                (
                    "photosynthesis",
                    "es",
                    "Las plantas comen luz solar como nosotros comemos tortillas! Usan el sol, agua y aire para hacer su comida y nos dan oxígeno.",
                    "food",
                    8,
                ),
                (
                    "rain_cycle",
                    "es",
                    "El ciclo del agua es como cuando mamá hierve frijoles - el vapor sube, se convierte en nubes, y luego llueve!",
                    "daily_life",
                    7,
                ),
                # History
                (
                    "aztec_empire",
                    "es",
                    "Los aztecas construyeron Tenochtitlan en un lago, como construir una ciudad flotante con canales en lugar de calles!",
                    "tradition",
                    10,
                ),
                (
                    "day_of_dead",
                    "es",
                    "El Día de los Muertos es cuando recordamos a nuestros abuelos con amor, decorando altares con sus comidas favoritas.",
                    "tradition",
                    8,
                ),
            ]

        elif region == "south_asia":
            examples = [
                # Mathematics
                (
                    "fractions",
                    "hi",
                    "रोटी के टुकड़ों की तरह सोचो! अगर रोटी को 4 बराबर हिस्सों में काटें और 2 खाएं, तो आपने 2/4 या 1/2 रोटी खाई!",
                    "food",
                    6,
                ),
                (
                    "place_value",
                    "hi",
                    "संख्याओं के स्थान बिल्कुल क्रिकेट टीम की तरह हैं - हर खिलाड़ी की अपनी जगह होती है!",
                    "sport",
                    8,
                ),
                # Science
                (
                    "monsoon",
                    "hi",
                    "मानसून बारिश आसमान का त्योहार है - जैसे होली में रंग बरसता है, वैसे ही बादल पानी बरसाते हैं!",
                    "tradition",
                    7,
                ),
                (
                    "digestion",
                    "hi",
                    "पेट में खाना पचना दाल बनाने जैसा है - सब कुछ मिलकर शरीर के लिए ताकत बनता है!",
                    "food",
                    8,
                ),
                # History
                (
                    "ancient_india",
                    "hi",
                    "प्राचीन भारत में महान राजा थे जो न्याय करते थे, जैसे पंचायत में बुजुर्ग फैसला करते हैं!",
                    "tradition",
                    10,
                ),
            ]

        # Store examples
        for concept, language, example_text, context_type, min_age in examples:
            example = CulturalExample(
                example_id=f"{region}_{concept}_{language}_{len(self.cultural_examples[(region, concept)])}",
                concept=concept,
                region=region,
                language=language,
                example_text=example_text,
                context_type=context_type,
                age_appropriateness=min_age,
                cultural_sensitivity=1.0,
            )

            self.cultural_examples[(region, concept)].append(example)

        # Log to W&B
        wandb.log(
            {
                f"cultural_examples/{region}": len(list(examples)),
                "cultural_database_updated": True,
                "region": region,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    async def initialize_explanation_templates(self) -> None:
        """Initialize explanation templates for different concepts and ages."""
        templates = [
            # Young children (3-6)
            ExplanationTemplate(
                template_id="young_simple",
                concept_category="general",
                age_range=(3, 6),
                language="en",
                template_structure="""
                {concept} is like {simple_analogy}!

                {cultural_example}

                Remember: {simple_rule}
                """,
                cultural_adaptability=0.9,
                effectiveness_metrics={"engagement": 0.8, "comprehension": 0.7},
            ),
            # Elementary (7-10)
            ExplanationTemplate(
                template_id="elementary_detailed",
                concept_category="general",
                age_range=(7, 10),
                language="en",
                template_structure="""
                Let me explain {concept} in a way that's easy to understand!

                {concept} is {basic_definition}.

                Here's a great example from your life: {cultural_example}

                Why does this matter? {practical_importance}

                Try this: {simple_activity}
                """,
                cultural_adaptability=0.85,
                effectiveness_metrics={"engagement": 0.85, "comprehension": 0.8},
            ),
            # Middle school (11-14)
            ExplanationTemplate(
                template_id="middle_comprehensive",
                concept_category="general",
                age_range=(11, 14),
                language="en",
                template_structure="""
                {concept} explained simply:

                **What it is:** {detailed_definition}

                **Real-world connection:** {cultural_example}

                **Why it matters:** {broader_context}

                **Step by step:**
                {step_by_step_breakdown}

                **Try it yourself:** {hands_on_activity}
                """,
                cultural_adaptability=0.8,
                effectiveness_metrics={"engagement": 0.82, "comprehension": 0.85},
            ),
            # High school (15-18)
            ExplanationTemplate(
                template_id="high_analytical",
                concept_category="general",
                age_range=(15, 18),
                language="en",
                template_structure="""
                Understanding {concept}:

                **Core Concept:** {sophisticated_definition}

                **Cultural Context:** {cultural_example}

                **Deeper Analysis:** {analytical_explanation}

                **Connections:** {interdisciplinary_links}

                **Application:** {real_world_applications}

                **Critical Thinking:** {discussion_questions}
                """,
                cultural_adaptability=0.75,
                effectiveness_metrics={"engagement": 0.78, "comprehension": 0.88},
            ),
        ]

        for template in templates:
            self.explanation_templates[template.template_id] = template

        # Create multilingual versions
        await self.create_multilingual_templates()

    async def create_multilingual_templates(self) -> None:
        """Create templates for different languages."""
        # Spanish templates
        spanish_templates = [
            ExplanationTemplate(
                template_id="young_simple_es",
                concept_category="general",
                age_range=(3, 6),
                language="es",
                template_structure="""
                ¡{concept} es como {simple_analogy}!

                {cultural_example}

                Recuerda: {simple_rule}
                """,
                cultural_adaptability=0.9,
                effectiveness_metrics={"engagement": 0.8, "comprehension": 0.7},
            ),
            ExplanationTemplate(
                template_id="elementary_detailed_es",
                concept_category="general",
                age_range=(7, 10),
                language="es",
                template_structure="""
                ¡Te voy a explicar {concept} de una manera fácil de entender!

                {concept} es {basic_definition}.

                Aquí tienes un ejemplo de tu vida: {cultural_example}

                ¿Por qué es importante? {practical_importance}

                Prueba esto: {simple_activity}
                """,
                cultural_adaptability=0.85,
                effectiveness_metrics={"engagement": 0.85, "comprehension": 0.8},
            ),
        ]

        # Hindi templates
        hindi_templates = [
            ExplanationTemplate(
                template_id="young_simple_hi",
                concept_category="general",
                age_range=(3, 6),
                language="hi",
                template_structure="""
                {concept} बिल्कुल {simple_analogy} की तरह है!

                {cultural_example}

                याद रखो: {simple_rule}
                """,
                cultural_adaptability=0.9,
                effectiveness_metrics={"engagement": 0.8, "comprehension": 0.7},
            )
        ]

        for template in spanish_templates + hindi_templates:
            self.explanation_templates[template.template_id] = template

    async def explain(
        self,
        concept: str,
        age: int,
        language: str,
        region: str,
        learning_style: str = "balanced",
        complexity_preference: str = "auto",
    ) -> ExplanationResult:
        """Generate culturally relevant, age-appropriate explanation."""
        start_time = asyncio.get_event_loop().time()

        try:
            # Get base explanation
            base_explanation = await self.get_base_explanation(concept, age, language)

            # Find appropriate cultural examples
            local_examples = self.get_cultural_examples(region, concept, age, language)

            # Select explanation template
            template = self.select_explanation_template(age, language, concept)

            # Generate culturally adapted explanation
            adapted_explanation = await self.generate_adapted_explanation(
                concept=concept,
                age=age,
                language=language,
                region=region,
                base_explanation=base_explanation,
                cultural_examples=local_examples,
                template=template,
                learning_style=learning_style,
            )

            # Calculate metrics
            generation_time = asyncio.get_event_loop().time() - start_time
            readability_score = self.calculate_readability_score(adapted_explanation, age)
            cultural_relevance_score = self.calculate_cultural_relevance(adapted_explanation, region)
            engagement_score = self.calculate_engagement_score(adapted_explanation, age)

            # Create result
            result = ExplanationResult(
                concept=concept,
                age=age,
                language=language,
                region=region,
                explanation=adapted_explanation,
                cultural_examples_used=[ex.example_id for ex in local_examples],
                template_used=template.template_id if template else "default",
                readability_score=readability_score,
                cultural_relevance_score=cultural_relevance_score,
                engagement_score=engagement_score,
                generation_time=generation_time,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            # Store result and update metrics
            self.explanation_history.append(result)
            await self.update_effectiveness_metrics(result)

            # Log to W&B
            wandb.log(
                {
                    "eli5/concept": concept,
                    "eli5/age": age,
                    "eli5/language": language,
                    "eli5/region": region,
                    "eli5/examples_used": len(local_examples),
                    "eli5/readability_score": readability_score,
                    "eli5/cultural_relevance": cultural_relevance_score,
                    "eli5/engagement_score": engagement_score,
                    "eli5/generation_time": generation_time,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            logger.info(f"Generated ELI5 explanation for '{concept}' (age {age}, {language}, {region})")

            return result

        except Exception as e:
            logger.exception(f"Error generating ELI5 explanation: {e}")

            # Return fallback explanation
            fallback_explanation = await self.get_fallback_explanation(concept, age, language)

            return ExplanationResult(
                concept=concept,
                age=age,
                language=language,
                region=region,
                explanation=fallback_explanation,
                cultural_examples_used=[],
                template_used="fallback",
                readability_score=0.5,
                cultural_relevance_score=0.3,
                engagement_score=0.4,
                generation_time=0.1,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    async def get_base_explanation(self, concept: str, age: int, language: str) -> str:
        """Get base explanation for concept."""
        # Create age-appropriate prompt
        age_descriptor = self.get_age_descriptor(age)

        prompt = f"""
        Explain the concept "{concept}" for {age_descriptor} in {language}.

        Requirements:
        - Use simple, clear language appropriate for age {age}
        - Be accurate but not overly technical
        - Focus on core understanding
        - Make it relatable to everyday life
        - Keep it concise (2-3 sentences for young children, 4-5 for older)

        Concept to explain: {concept}
        """

        try:
            # Use Anthropic Claude for explanation generation
            response = await self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

            return response.content[0].text.strip()

        except Exception as e:
            logger.exception(f"Error getting base explanation from Anthropic: {e}")

            # Fallback to OpenAI
            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )

                return response.choices[0].message.content.strip()

            except Exception as e2:
                logger.exception(f"Error getting base explanation from OpenAI: {e2}")
                return f"I'll explain {concept} in a simple way that's perfect for someone who is {age} years old."

    def get_age_descriptor(self, age: int) -> str:
        """Get age-appropriate descriptor."""
        if age <= 5:
            return "a very young child"
        if age <= 8:
            return "an elementary school child"
        if age <= 12:
            return "a middle school student"
        if age <= 16:
            return "a high school student"
        return "a young adult"

    def get_cultural_examples(self, region: str, concept: str, age: int, language: str) -> list[CulturalExample]:
        """Get relevant cultural examples for the concept."""
        # Direct match
        direct_examples = self.cultural_examples.get((region, concept), [])

        # Filter by age appropriateness and language
        suitable_examples = [ex for ex in direct_examples if ex.age_appropriateness <= age and ex.language == language]

        # If no direct examples, look for related concepts
        if not suitable_examples:
            # Look for examples from same region, any concept
            region_examples = []
            for (r, _c), examples in self.cultural_examples.items():
                if r == region:
                    region_examples.extend(
                        [ex for ex in examples if ex.age_appropriateness <= age and ex.language == language]
                    )

            # Select most relevant ones (simple heuristic)
            suitable_examples = region_examples[:2]

        # Sort by effectiveness score
        suitable_examples.sort(key=lambda x: x.effectiveness_score, reverse=True)

        return suitable_examples[:3]  # Return top 3 examples

    def select_explanation_template(self, age: int, language: str, concept: str) -> ExplanationTemplate | None:
        """Select most appropriate explanation template."""
        # Filter templates by age range and language
        suitable_templates = [
            template
            for template in self.explanation_templates.values()
            if (template.age_range[0] <= age <= template.age_range[1] and template.language == language)
        ]

        if not suitable_templates:
            # Fallback to English templates if no language match
            suitable_templates = [
                template
                for template in self.explanation_templates.values()
                if (template.age_range[0] <= age <= template.age_range[1] and template.language == "en")
            ]

        if not suitable_templates:
            return None

        # Select template with highest effectiveness for this type
        best_template = max(
            suitable_templates,
            key=lambda t: t.effectiveness_metrics.get("comprehension", 0.5),
        )

        return best_template

    async def generate_adapted_explanation(
        self,
        concept: str,
        age: int,
        language: str,
        region: str,
        base_explanation: str,
        cultural_examples: list[CulturalExample],
        template: ExplanationTemplate | None,
        learning_style: str,
    ) -> str:
        """Generate culturally adapted explanation using template and examples."""
        # Prepare cultural examples text
        examples_text = ""
        if cultural_examples:
            examples_text = "\n".join([ex.example_text for ex in cultural_examples[:2]])

        # Create adaptation prompt
        adaptation_prompt = f"""
        Adapt this explanation of "{concept}" to be culturally relevant for someone from {region}:

        Base explanation: {base_explanation}

        Cultural examples to incorporate:
        {examples_text}

        Requirements:
        - Age: {age} years old
        - Language: {language}
        - Region: {region}
        - Learning style: {learning_style}
        - Make it relatable to daily life in {region}
        - Use the cultural examples naturally
        - Keep the scientific/factual accuracy
        - Make it engaging and memorable

        {f"Use this template structure: {template.template_structure}" if template else ""}
        """

        try:
            response = await self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=800,
                messages=[{"role": "user", "content": adaptation_prompt}],
            )

            adapted_explanation = response.content[0].text.strip()

            # Post-process for learning style
            if learning_style == "visual":
                adapted_explanation = await self.add_visual_elements(adapted_explanation, concept)
            elif learning_style == "kinesthetic":
                adapted_explanation = await self.add_hands_on_activities(adapted_explanation, concept, age)

            return adapted_explanation

        except Exception as e:
            logger.exception(f"Error generating adapted explanation: {e}")

            # Simple fallback: combine base explanation with examples
            if cultural_examples:
                return f"{base_explanation}\n\n{cultural_examples[0].example_text}"
            return base_explanation

    async def add_visual_elements(self, explanation: str, concept: str) -> str:
        """Add visual learning elements to explanation."""
        visual_prompts = [
            "Picture this:",
            "Imagine you can see:",
            "Visualize:",
            "Draw in your mind:",
        ]

        visual_element = f"\n\n{random.choice(visual_prompts)} {explanation.split('.')[0].lower()}..."

        return explanation + visual_element

    async def add_hands_on_activities(self, explanation: str, concept: str, age: int) -> str:
        """Add kinesthetic learning activities."""
        if age <= 8:
            activity_starters = [
                "Try this with your hands:",
                "You can practice by:",
                "Move your body to show:",
            ]
        else:
            activity_starters = [
                "Hands-on activity:",
                "Try this experiment:",
                "Practice by doing:",
            ]

        activity_prompt = f"\n\n{random.choice(activity_starters)} [Simple activity related to {concept}]"

        return explanation + activity_prompt

    def calculate_readability_score(self, text: str, age: int) -> float:
        """Calculate age-appropriate readability score."""
        words = text.split()
        sentences = text.count(".") + text.count("!") + text.count("?")

        if sentences == 0:
            sentences = 1

        avg_words_per_sentence = len(words) / sentences

        # Simple readability heuristic
        if age <= 6:
            ideal_words_per_sentence = 8
        elif age <= 10:
            ideal_words_per_sentence = 12
        elif age <= 14:
            ideal_words_per_sentence = 16
        else:
            ideal_words_per_sentence = 20

        # Score based on how close to ideal
        score = max(
            0.0,
            1.0 - abs(avg_words_per_sentence - ideal_words_per_sentence) / ideal_words_per_sentence,
        )

        return score

    def calculate_cultural_relevance(self, text: str, region: str) -> float:
        """Calculate cultural relevance score."""
        # Simple heuristic based on presence of cultural indicators
        cultural_indicators = {
            "north_america": [
                "pizza",
                "football",
                "baseball",
                "neighborhood",
                "school",
            ],
            "latin_america": ["empanada", "fútbol", "familia", "mamá", "tortilla"],
            "south_asia": ["रोटी", "cricket", "family", "festival", "tradition"],
        }

        indicators = cultural_indicators.get(region, [])
        text_lower = text.lower()

        matches = sum(1 for indicator in indicators if indicator in text_lower)

        if not indicators:
            return 0.5  # Neutral if no indicators defined

        return min(1.0, matches / len(indicators))

    def calculate_engagement_score(self, text: str, age: int) -> float:
        """Calculate engagement score based on content."""
        engagement_factors = 0.0

        # Question presence (encourages thinking)
        if "?" in text:
            engagement_factors += 0.2

        # Exclamation marks (excitement)
        exclamation_count = text.count("!")
        engagement_factors += min(0.2, exclamation_count * 0.1)

        # Personal pronouns (relatability)
        personal_words = ["you", "your", "we", "us", "I"]
        personal_count = sum(1 for word in personal_words if word in text.lower())
        engagement_factors += min(0.2, personal_count * 0.05)

        # Action words for younger kids
        if age <= 10:
            action_words = ["try", "do", "make", "play", "see", "touch"]
            action_count = sum(1 for word in action_words if word in text.lower())
            engagement_factors += min(0.2, action_count * 0.05)

        # Analogies and comparisons
        comparison_words = ["like", "as", "similar", "imagine", "picture"]
        comparison_count = sum(1 for word in comparison_words if word in text.lower())
        engagement_factors += min(0.2, comparison_count * 0.1)

        return min(1.0, engagement_factors)

    async def update_effectiveness_metrics(self, result: ExplanationResult) -> None:
        """Update effectiveness metrics based on explanation result."""
        # Update cultural example effectiveness
        for example_id in result.cultural_examples_used:
            for examples_list in self.cultural_examples.values():
                for example in examples_list:
                    if example.example_id == example_id:
                        example.usage_count += 1
                        # Update effectiveness score (simple moving average)
                        alpha = 0.1  # Learning rate
                        new_score = (result.engagement_score + result.cultural_relevance_score) / 2
                        example.effectiveness_score = (1 - alpha) * example.effectiveness_score + alpha * new_score

        # Update template effectiveness
        if result.template_used in self.explanation_templates:
            template = self.explanation_templates[result.template_used]
            alpha = 0.05
            template.effectiveness_metrics["comprehension"] = (1 - alpha) * template.effectiveness_metrics.get(
                "comprehension", 0.5
            ) + alpha * result.readability_score
            template.effectiveness_metrics["engagement"] = (1 - alpha) * template.effectiveness_metrics.get(
                "engagement", 0.5
            ) + alpha * result.engagement_score

        # Store metrics for analysis
        self.effectiveness_metrics[result.concept].append(
            {
                "age": result.age,
                "region": result.region,
                "language": result.language,
                "scores": {
                    "readability": result.readability_score,
                    "cultural_relevance": result.cultural_relevance_score,
                    "engagement": result.engagement_score,
                },
                "timestamp": result.timestamp,
            }
        )

    async def get_fallback_explanation(self, concept: str, age: int, language: str) -> str:
        """Generate simple fallback explanation."""
        fallback_templates = {
            "en": f"Let me explain {concept} in a simple way for someone who is {age} years old. This is an important topic that helps us understand the world around us.",
            "es": f"Te voy a explicar {concept} de manera simple para alguien de {age} años. Este es un tema importante que nos ayuda a entender el mundo.",
            "hi": f"मैं {concept} को {age} साल के बच्चे के लिए आसान तरीके से समझाता हूँ। यह एक महत्वपूर्ण विषय है।",
        }

        return fallback_templates.get(language, fallback_templates["en"])

    async def get_explanation_analytics(self) -> dict[str, Any]:
        """Get comprehensive analytics on explanation effectiveness."""
        if not self.explanation_history:
            return {"message": "No explanations generated yet"}

        analytics = {
            "total_explanations": len(self.explanation_history),
            "languages": {},
            "regions": {},
            "age_groups": {},
            "concepts": {},
            "average_scores": {
                "readability": 0.0,
                "cultural_relevance": 0.0,
                "engagement": 0.0,
            },
            "effectiveness_trends": {},
            "top_cultural_examples": [],
        }

        # Analyze explanations
        for result in self.explanation_history:
            # Language breakdown
            if result.language not in analytics["languages"]:
                analytics["languages"][result.language] = 0
            analytics["languages"][result.language] += 1

            # Region breakdown
            if result.region not in analytics["regions"]:
                analytics["regions"][result.region] = 0
            analytics["regions"][result.region] += 1

            # Age group breakdown
            age_group = f"{(result.age // 5) * 5}-{(result.age // 5) * 5 + 4}"
            if age_group not in analytics["age_groups"]:
                analytics["age_groups"][age_group] = 0
            analytics["age_groups"][age_group] += 1

            # Concept breakdown
            if result.concept not in analytics["concepts"]:
                analytics["concepts"][result.concept] = {
                    "count": 0,
                    "avg_engagement": 0.0,
                }
            analytics["concepts"][result.concept]["count"] += 1

        # Calculate average scores
        if self.explanation_history:
            analytics["average_scores"]["readability"] = sum(
                r.readability_score for r in self.explanation_history
            ) / len(self.explanation_history)
            analytics["average_scores"]["cultural_relevance"] = sum(
                r.cultural_relevance_score for r in self.explanation_history
            ) / len(self.explanation_history)
            analytics["average_scores"]["engagement"] = sum(r.engagement_score for r in self.explanation_history) / len(
                self.explanation_history
            )

        # Top cultural examples by usage
        all_examples = []
        for examples_list in self.cultural_examples.values():
            all_examples.extend(examples_list)

        top_examples = sorted(all_examples, key=lambda x: x.usage_count, reverse=True)[:10]
        analytics["top_cultural_examples"] = [
            {
                "example_id": ex.example_id,
                "concept": ex.concept,
                "region": ex.region,
                "usage_count": ex.usage_count,
                "effectiveness_score": ex.effectiveness_score,
            }
            for ex in top_examples
        ]

        return analytics


# Global ELI5 instance
culturally_aware_eli5 = CulturallyAwareELI5()
