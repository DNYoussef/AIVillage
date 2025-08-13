"""W&B Prompt Template Tracking for AI Tutoring
Part B: Agent Forge Phase 4 - Prompt Engineering.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import logging
from typing import Any

import wandb

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Structured prompt template for tutoring interactions."""

    variant_id: str
    greeting_style: str
    hint_complexity: str
    example_type: str
    encouragement_frequency: float
    template_text: str
    performance_score: float = 0.0
    interaction_count: int = 0
    last_updated: str = ""


class TutorPromptEngineer:
    """W&B-tracked prompt optimization for tutoring."""

    def __init__(self, project_name: str = "aivillage-tutoring") -> None:
        self.project_name = project_name
        self.prompt_artifacts = []
        self.active_templates = {}
        self.performance_cache = {}

        # Initialize base prompt templates
        self.base_templates = {
            "greeting": {
                "formal": "Hello! I'm here to help you learn. What subject would you like to explore today?",
                "friendly": "Hi there! 😊 I'm excited to learn with you today! What can I help you understand?",
                "encouraging": "Hey! You're taking a great step by asking for help! I'm here to support your learning journey. What's on your mind?",
                "playful": "Greetings, fellow knowledge explorer! 🚀 Ready to unlock some amazing learning today? What adventure shall we begin?",
            },
            "hint_delivery": {
                "direct": "Here's the key concept: {concept}. Try applying it to: {problem}",
                "guided": "Let's think about this step by step. First, consider: {concept}. How might this relate to your problem?",
                "socratic": "What do you think happens when {concept}? Can you find a connection to your current challenge?",
            },
            "examples": {
                "abstract": "Consider the mathematical relationship: {example}",
                "real-world": "Imagine you're {scenario}. How would {concept} apply here?",
                "visual": "Picture this: {visual_description}. Can you see how {concept} works?",
                "story-based": "Let me tell you about {character} who faced a similar challenge with {concept}...",
            },
        }

        # Initialize W&B run
        self.initialize_wandb_tracking()

    def initialize_wandb_tracking(self) -> None:
        """Initialize W&B tracking for prompt engineering."""
        try:
            wandb.init(
                project=self.project_name,
                job_type="prompt_engineering",
                config={
                    "framework": "tutor_prompt_engineer",
                    "version": "1.0.0",
                    "base_templates": len(self.base_templates),
                    "tracking_metrics": [
                        "student_engagement",
                        "response_clarity",
                        "learning_outcome",
                        "response_time",
                        "encouragement_score",
                    ],
                },
            )

            logger.info("W&B tracking initialized for prompt engineering")

        except Exception as e:
            logger.exception(f"Failed to initialize W&B: {e}")

    async def create_prompt_sweep(self) -> str:
        """Define W&B sweep for tutoring prompts optimization."""
        sweep_config = {
            "method": "bayes",
            "metric": {"name": "student_engagement", "goal": "maximize"},
            "parameters": {
                "greeting_style": {"values": ["formal", "friendly", "encouraging", "playful"]},
                "hint_complexity": {"values": ["direct", "guided", "socratic"]},
                "example_type": {"values": ["abstract", "real-world", "visual", "story-based"]},
                "encouragement_frequency": {"min": 0.1, "max": 0.5},
                "response_length_target": {"values": ["concise", "moderate", "detailed"]},
                "personalization_level": {"values": ["generic", "adaptive", "highly_personalized"]},
                "subject_expertise": {
                    "values": [
                        "mathematics",
                        "science",
                        "programming",
                        "language_arts",
                        "history",
                        "general",
                    ]
                },
            },
            "early_terminate": {"type": "hyperband", "min_iter": 10, "eta": 2},
        }

        try:
            sweep_id = wandb.sweep(sweep_config, project=self.project_name)

            # Log sweep creation
            wandb.log(
                {
                    "sweep_created": True,
                    "sweep_id": sweep_id,
                    "parameter_combinations": len(sweep_config["parameters"]),
                    "optimization_method": sweep_config["method"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            logger.info(f"Created W&B sweep: {sweep_id}")
            return sweep_id

        except Exception as e:
            logger.exception(f"Failed to create prompt sweep: {e}")
            return ""

    async def generate_prompt_template(
        self,
        greeting_style: str,
        hint_complexity: str,
        example_type: str,
        encouragement_frequency: float,
        subject: str = "general",
        context: dict[str, Any] | None = None,
    ) -> PromptTemplate:
        """Generate a complete prompt template with specified parameters."""
        context = context or {}

        # Create variant ID for tracking
        variant_data = f"{greeting_style}_{hint_complexity}_{example_type}_{encouragement_frequency}_{subject}"
        variant_id = hashlib.md5(variant_data.encode()).hexdigest()[:12]

        # Build the complete prompt template
        template_parts = []

        # Greeting section
        greeting = self.base_templates["greeting"][greeting_style]
        template_parts.append(f"GREETING: {greeting}")

        # Subject expertise section
        if subject in ["mathematics", "science", "programming"]:
            template_parts.append(
                f"EXPERTISE: I specialize in {subject} and will provide detailed, accurate explanations."
            )

        # Hint delivery approach
        hint_style = self.base_templates["hint_delivery"][hint_complexity]
        template_parts.append(f"APPROACH: {hint_style}")

        # Example style
        example_style = self.base_templates["examples"][example_type]
        template_parts.append(f"EXAMPLES: {example_style}")

        # Encouragement integration
        if encouragement_frequency > 0.3:
            template_parts.append("ENCOURAGEMENT: Frequently praise effort and progress. Use positive reinforcement.")
        elif encouragement_frequency > 0.1:
            template_parts.append("ENCOURAGEMENT: Provide moderate encouragement and celebrate breakthroughs.")
        else:
            template_parts.append("ENCOURAGEMENT: Focus on content with minimal emotional support.")

        # Response guidelines
        template_parts.append("GUIDELINES:")
        template_parts.append("- Keep responses under 500 words for WhatsApp")
        template_parts.append("- Use appropriate language for the detected language")
        template_parts.append("- Ask one follow-up question to maintain engagement")
        template_parts.append("- Provide concrete next steps")

        # Final template assembly
        complete_template = "\n".join(template_parts)
        complete_template += "\n\nSTUDENT MESSAGE: {user_message}\nRESPONSE:"

        # Create template object
        prompt_template = PromptTemplate(
            variant_id=variant_id,
            greeting_style=greeting_style,
            hint_complexity=hint_complexity,
            example_type=example_type,
            encouragement_frequency=encouragement_frequency,
            template_text=complete_template,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

        # Store in active templates
        self.active_templates[variant_id] = prompt_template

        # Log template creation to W&B
        wandb.log(
            {
                "template_created": True,
                "variant_id": variant_id,
                "greeting_style": greeting_style,
                "hint_complexity": hint_complexity,
                "example_type": example_type,
                "encouragement_frequency": encouragement_frequency,
                "subject": subject,
                "template_length": len(complete_template),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return prompt_template

    async def evaluate_prompt_performance(
        self,
        variant_id: str,
        response_text: str,
        response_time: float,
        user_engagement_signals: dict[str, Any],
    ) -> dict[str, float]:
        """Evaluate prompt performance across multiple metrics."""
        try:
            # Calculate engagement score
            engagement_score = await self._calculate_engagement_score(response_text, user_engagement_signals)

            # Calculate clarity score
            clarity_score = await self._calculate_clarity_score(response_text)

            # Calculate encouragement score
            encouragement_score = await self._calculate_encouragement_score(response_text)

            # Calculate efficiency score (response time factor)
            efficiency_score = max(0.0, min(1.0, (5.0 - response_time) / 5.0))

            # Overall performance score (weighted average)
            performance_metrics = {
                "student_engagement": engagement_score,
                "response_clarity": clarity_score,
                "encouragement_score": encouragement_score,
                "response_efficiency": efficiency_score,
                "overall_performance": (
                    engagement_score * 0.4 + clarity_score * 0.3 + encouragement_score * 0.2 + efficiency_score * 0.1
                ),
            }

            # Update template performance
            if variant_id in self.active_templates:
                template = self.active_templates[variant_id]
                template.interaction_count += 1
                template.performance_score = performance_metrics["overall_performance"]
                template.last_updated = datetime.now(timezone.utc).isoformat()

            # Log to W&B
            wandb.log(
                {
                    "variant_id": variant_id,
                    "response_time": response_time,
                    "response_length": len(response_text),
                    **performance_metrics,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            return performance_metrics

        except Exception as e:
            logger.exception(f"Error evaluating prompt performance: {e}")
            return {"overall_performance": 0.0}

    async def _calculate_engagement_score(self, response_text: str, signals: dict[str, Any]) -> float:
        """Calculate student engagement score based on response content and signals."""
        score = 0.0

        # Question asking (encourages interaction)
        if "?" in response_text:
            score += 0.2

        # Encouraging language
        encouraging_words = [
            "great",
            "excellent",
            "good job",
            "well done",
            "keep going",
            "you can do it",
        ]
        if any(word in response_text.lower() for word in encouraging_words):
            score += 0.2

        # Interactive elements
        interactive_phrases = [
            "try this",
            "what do you think",
            "can you",
            "let's explore",
        ]
        if any(phrase in response_text.lower() for phrase in interactive_phrases):
            score += 0.3

        # Clear structure (helps engagement)
        if any(marker in response_text for marker in ["1.", "2.", "•", "-"]):
            score += 0.1

        # Examples provided
        if any(word in response_text.lower() for word in ["example", "for instance", "imagine", "picture"]):
            score += 0.2

        return min(1.0, score)

    async def _calculate_clarity_score(self, response_text: str) -> float:
        """Calculate response clarity score."""
        score = 0.0
        words = response_text.split()

        # Appropriate length (not too short, not too long)
        if 20 <= len(words) <= 150:
            score += 0.3
        elif len(words) < 20:
            score += 0.1  # Too brief
        else:
            score += 0.2  # Too long but still okay

        # Clear structure indicators
        structure_words = ["first", "second", "then", "next", "finally", "in summary"]
        if any(word in response_text.lower() for word in structure_words):
            score += 0.2

        # Explanation indicators
        explanation_words = [
            "because",
            "since",
            "therefore",
            "this means",
            "in other words",
        ]
        if any(word in response_text.lower() for word in explanation_words):
            score += 0.3

        # Concrete examples
        if any(word in response_text.lower() for word in ["example", "like", "such as"]):
            score += 0.2

        return min(1.0, score)

    async def _calculate_encouragement_score(self, response_text: str) -> float:
        """Calculate encouragement/positivity score."""
        positive_words = [
            "great",
            "excellent",
            "good",
            "well done",
            "fantastic",
            "amazing",
            "you can do it",
            "keep going",
            "nice work",
            "perfect",
            "exactly",
            "wonderful",
            "brilliant",
            "impressive",
            "outstanding",
        ]

        encouraging_phrases = [
            "don't worry",
            "it's okay",
            "that's normal",
            "everyone struggles",
            "you're learning",
            "step by step",
            "at your own pace",
            "you got this",
        ]

        score = 0.0
        text_lower = response_text.lower()

        # Count positive words
        positive_count = sum(1 for word in positive_words if word in text_lower)
        score += min(0.5, positive_count * 0.1)

        # Count encouraging phrases
        encouraging_count = sum(1 for phrase in encouraging_phrases if phrase in text_lower)
        score += min(0.3, encouraging_count * 0.15)

        # Presence of emoji (indicates friendly tone)
        if any(char in response_text for char in ["😊", "👍", "✨", "🎉", "💡", "🚀"]):
            score += 0.2

        return min(1.0, score)

    async def get_best_performing_template(
        self, subject: str = "general", min_interactions: int = 10
    ) -> PromptTemplate | None:
        """Get the best performing template for a given subject."""
        # Filter templates by interaction count
        qualified_templates = [
            template for template in self.active_templates.values() if template.interaction_count >= min_interactions
        ]

        if not qualified_templates:
            logger.warning(f"No templates with minimum {min_interactions} interactions found")
            return None

        # Sort by performance score
        best_template = max(qualified_templates, key=lambda t: t.performance_score)

        # Log selection to W&B
        wandb.log(
            {
                "best_template_selected": True,
                "variant_id": best_template.variant_id,
                "performance_score": best_template.performance_score,
                "interaction_count": best_template.interaction_count,
                "subject": subject,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return best_template

    async def run_sweep_agent(self):
        """Run W&B sweep agent for continuous optimization."""

        def train_prompt() -> None:
            """Single sweep run for prompt optimization."""
            # Initialize run with sweep parameters
            run = wandb.init()
            config = wandb.config

            # Generate template with sweep parameters
            template = asyncio.run(
                self.generate_prompt_template(
                    greeting_style=config.greeting_style,
                    hint_complexity=config.hint_complexity,
                    example_type=config.example_type,
                    encouragement_frequency=config.encouragement_frequency,
                    subject=config.get("subject_expertise", "general"),
                )
            )

            # Simulate interactions and log results
            # In production, this would be connected to real user interactions
            simulated_performance = {
                "student_engagement": 0.75 + (hash(template.variant_id) % 100) / 400,
                "response_clarity": 0.70 + (hash(template.variant_id) % 100) / 500,
                "encouragement_score": config.encouragement_frequency,
                "interaction_count": 50,  # Simulated
            }

            # Log final metrics
            wandb.log(simulated_performance)

            run.finish()

        return train_prompt

    async def save_prompt_artifacts(self) -> None:
        """Save successful prompt templates as W&B artifacts."""
        try:
            # Get top performing templates
            top_templates = sorted(
                self.active_templates.values(),
                key=lambda t: t.performance_score,
                reverse=True,
            )[
                :5
            ]  # Top 5

            for template in top_templates:
                if template.performance_score > 0.7:  # Minimum performance threshold
                    # Create artifact
                    artifact = wandb.Artifact(
                        f"prompt_template_{template.variant_id}",
                        type="prompt_template",
                        description=f"High-performing tutoring prompt (score: {template.performance_score:.3f})",
                        metadata={
                            "greeting_style": template.greeting_style,
                            "hint_complexity": template.hint_complexity,
                            "example_type": template.example_type,
                            "encouragement_frequency": template.encouragement_frequency,
                            "performance_score": template.performance_score,
                            "interaction_count": template.interaction_count,
                        },
                    )

                    # Save template content
                    with artifact.new_file(f"template_{template.variant_id}.txt", mode="w") as f:
                        f.write(template.template_text)

                    # Log artifact
                    wandb.log_artifact(artifact)
                    self.prompt_artifacts.append(artifact)

            logger.info(f"Saved {len(self.prompt_artifacts)} prompt artifacts to W&B")

        except Exception as e:
            logger.exception(f"Error saving prompt artifacts: {e}")


# Global instance for easy access
tutor_prompt_engineer = TutorPromptEngineer()
