"""W&B Prompt Tuning System for WhatsApp Tutoring
Implements A/B testing and prompt optimization
"""

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import logging
import random
from typing import Any

import numpy as np
import wandb

logger = logging.getLogger(__name__)


@dataclass
class PromptVariant:
    """Represents a prompt variant for A/B testing"""

    id: str
    template: str
    description: str
    language: str
    category: str  # 'greeting', 'tutoring', 'clarification', etc.
    parameters: dict[str, Any]
    performance_metrics: dict[str, float]
    created_at: str
    active: bool = True


@dataclass
class ABTestResult:
    """Results from A/B testing"""

    variant_id: str
    user_hash: str
    session_id: str
    response_time: float
    user_satisfaction: float  # 0-1 based on engagement
    conversion: bool  # Did user continue conversation
    language: str
    timestamp: str


class PromptTuner:
    """Manages prompt optimization using W&B experiment tracking"""

    def __init__(self):
        self.variants_cache = {}
        self.performance_history = defaultdict(deque)
        self.optimization_threshold = 0.05  # 5% improvement threshold
        self.min_samples_for_optimization = 50

        # Initialize base prompt templates
        self.initialize_base_prompts()

    def initialize_base_prompts(self):
        """Initialize base prompt templates for different scenarios"""
        base_prompts = {
            "tutoring_en": {
                "formal": """You are an expert AI tutor helping a student learn. Be encouraging, clear, and educational.

Student question: {user_message}

Provide a helpful response that:
1. Directly answers their question
2. Explains the reasoning
3. Offers additional learning resources
4. Asks a follow-up question to check understanding

Response:""",
                "conversational": """Hi! I'm here to help you learn 📚

You asked: {user_message}

Let me break this down for you step by step, and then I'll ask you a question to make sure you've got it!

""",
                "socratic": """Great question! Instead of giving you the answer directly, let me guide you to discover it yourself.

Your question: {user_message}

Think about this: {guiding_question}

What do you think? Let's work through this together!

""",
            },
            "greeting_en": {
                "enthusiastic": """🌟 Hey there! I'm your AI tutor and I'm so excited to help you learn today!

What subject or topic would you like to explore? I'm here to make learning fun and easy! 📚✨""",
                "professional": """Hello! I'm your AI tutoring assistant. I'm here to help you understand any topic you're curious about.

Please share what you'd like to learn, and I'll provide clear, comprehensive explanations tailored to your level.""",
                "friendly": """Hi! 👋 Ready to learn something new today?

I'm your AI tutor, and I love helping students discover new things. What's on your mind?""",
            },
        }

        # Store in W&B config
        wandb.config.update(
            {
                "base_prompts": base_prompts,
                "prompt_categories": list(base_prompts.keys()),
                "optimization_enabled": True,
            }
        )

        self.base_prompts = base_prompts

    async def get_optimized_prompt(
        self, message_type: str, language: str = "en", context: dict[str, Any] = None
    ) -> str:
        """Get the best-performing prompt variant for given context"""
        category_key = f"{message_type}_{language}"

        # Get variants for this category
        variants = await self.get_active_variants(category_key)

        if not variants:
            # Return default prompt if no variants available
            return self.get_default_prompt(message_type, language)

        # Select best performing variant based on recent metrics
        best_variant = await self.select_best_variant(variants)

        # Log variant selection to W&B
        wandb.log(
            {
                "prompt_selection": {
                    "category": category_key,
                    "selected_variant": best_variant.id,
                    "variant_description": best_variant.description,
                    "performance_score": best_variant.performance_metrics.get("avg_score", 0.0),
                }
            }
        )

        # Format prompt with context
        formatted_prompt = self.format_prompt(best_variant.template, context or {})

        return formatted_prompt

    async def get_active_variants(self, category: str) -> list[PromptVariant]:
        """Get all active variants for a category"""
        # Check cache first
        cache_key = f"variants_{category}"
        if cache_key in self.variants_cache:
            return self.variants_cache[cache_key]

        try:
            # In production, this would query W&B artifacts or database
            # For now, create variants from base prompts
            variants = []

            if category in self.base_prompts:
                for variant_name, template in self.base_prompts[category].items():
                    variant = PromptVariant(
                        id=f"{category}_{variant_name}",
                        template=template,
                        description=f"{variant_name.title()} {category} prompt",
                        language="en",  # TODO: Multi-language support
                        category=category,
                        parameters={},
                        performance_metrics={"avg_score": 0.5, "sample_count": 0},
                        created_at=datetime.now().isoformat(),
                    )
                    variants.append(variant)

            # Cache variants
            self.variants_cache[cache_key] = variants

            return variants

        except Exception as e:
            logger.error(f"Error getting variants for {category}: {e}")
            return []

    async def select_best_variant(self, variants: list[PromptVariant]) -> PromptVariant:
        """Select best performing variant using multi-armed bandit approach"""
        if len(variants) == 1:
            return variants[0]

        # Calculate upper confidence bounds for each variant
        total_samples = sum(v.performance_metrics.get("sample_count", 0) for v in variants)

        if total_samples < self.min_samples_for_optimization:
            # Random selection during exploration phase
            return random.choice(variants)

        # UCB1 algorithm for balanced exploration/exploitation
        ucb_scores = []
        for variant in variants:
            avg_score = variant.performance_metrics.get("avg_score", 0.0)
            sample_count = variant.performance_metrics.get("sample_count", 1)

            if sample_count == 0:
                ucb_score = float("inf")  # Prioritize unexplored variants
            else:
                confidence_bound = np.sqrt(2 * np.log(total_samples) / sample_count)
                ucb_score = avg_score + confidence_bound

            ucb_scores.append(ucb_score)

        # Select variant with highest UCB score
        best_idx = np.argmax(ucb_scores)
        selected_variant = variants[best_idx]

        # Log selection reasoning
        wandb.log(
            {
                "variant_selection": {
                    "selected_id": selected_variant.id,
                    "ucb_score": ucb_scores[best_idx],
                    "avg_score": selected_variant.performance_metrics.get("avg_score", 0.0),
                    "sample_count": selected_variant.performance_metrics.get("sample_count", 0),
                    "total_samples": total_samples,
                }
            }
        )

        return selected_variant

    def get_default_prompt(self, message_type: str, language: str = "en") -> str:
        """Get default prompt when no variants available"""
        defaults = {
            "tutoring": """You are a helpful AI tutor. Answer the student's question clearly and encouragingly.

Student question: {user_message}

Your response:""",
            "greeting": """Hello! I'm your AI tutor. How can I help you learn today?""",
        }

        return defaults.get(message_type, defaults["tutoring"])

    def format_prompt(self, template: str, context: dict[str, Any]) -> str:
        """Format prompt template with context variables"""
        try:
            # Add default context variables
            context.setdefault("user_message", "")
            context.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M"))

            # Generate guiding questions for Socratic method
            if "guiding_question" not in context and "user_message" in context:
                context["guiding_question"] = self.generate_guiding_question(context["user_message"])

            return template.format(**context)

        except KeyError as e:
            logger.warning(f"Missing context variable for prompt formatting: {e}")
            return template
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            return template

    def generate_guiding_question(self, user_message: str) -> str:
        """Generate Socratic guiding question"""
        # Simple heuristic-based question generation
        question_starters = [
            "What do you think happens when",
            "How might you approach",
            "What patterns do you notice in",
            "Can you think of a similar situation where",
            "What would you expect if",
        ]

        starter = random.choice(question_starters)
        return f"{starter} {user_message.lower()}?"

    async def record_prompt_performance(
        self,
        variant_id: str,
        session_id: str,
        user_hash: str,
        performance_metrics: dict[str, float],
    ):
        """Record performance metrics for a prompt variant"""
        try:
            # Create performance record
            result = ABTestResult(
                variant_id=variant_id,
                user_hash=user_hash,
                session_id=session_id,
                response_time=performance_metrics.get("response_time", 0.0),
                user_satisfaction=performance_metrics.get("satisfaction", 0.5),
                conversion=performance_metrics.get("conversion", False),
                language=performance_metrics.get("language", "en"),
                timestamp=datetime.now().isoformat(),
            )

            # Log to W&B
            wandb.log(
                {
                    "prompt_performance": asdict(result),
                    "metrics_summary": performance_metrics,
                }
            )

            # Update local performance tracking
            self.performance_history[variant_id].append(result)

            # Keep only recent results (last 1000)
            if len(self.performance_history[variant_id]) > 1000:
                self.performance_history[variant_id].popleft()

            # Update variant performance metrics
            await self.update_variant_metrics(variant_id)

        except Exception as e:
            logger.error(f"Error recording prompt performance: {e}")

    async def update_variant_metrics(self, variant_id: str):
        """Update aggregate metrics for a variant"""
        if variant_id not in self.performance_history:
            return

        results = list(self.performance_history[variant_id])
        if not results:
            return

        # Calculate aggregate metrics
        response_times = [r.response_time for r in results]
        satisfactions = [r.user_satisfaction for r in results]
        conversions = [r.conversion for r in results]

        metrics = {
            "avg_response_time": np.mean(response_times),
            "avg_satisfaction": np.mean(satisfactions),
            "conversion_rate": np.mean(conversions),
            "sample_count": len(results),
            "avg_score": np.mean(satisfactions) * 0.7 + (1 - np.mean(response_times) / 10) * 0.3,  # Weighted score
        }

        # Update variant in cache
        for cache_variants in self.variants_cache.values():
            for variant in cache_variants:
                if variant.id == variant_id:
                    variant.performance_metrics.update(metrics)

        # Log updated metrics
        wandb.log({"variant_metrics_update": {"variant_id": variant_id, **metrics}})


class ABTestManager:
    """Manages A/B testing for greeting messages and other interactions"""

    def __init__(self):
        self.test_assignments = {}  # user_hash -> variant mapping
        self.test_configs = self.initialize_test_configs()

    def initialize_test_configs(self):
        """Initialize A/B test configurations"""
        configs = {
            "greeting_style": {
                "variants": ["enthusiastic", "professional", "friendly"],
                "weights": [0.33, 0.33, 0.34],  # Equal distribution
                "active": True,
                "success_metric": "user_satisfaction",
            },
            "tutoring_approach": {
                "variants": ["formal", "conversational", "socratic"],
                "weights": [0.3, 0.4, 0.3],  # Favor conversational
                "active": True,
                "success_metric": "conversion_rate",
            },
        }

        # Log test configs to W&B
        wandb.config.update({"ab_test_configs": configs})

        return configs

    def get_greeting_variant(self, user_identifier: str) -> str:
        """Get consistent greeting variant for user"""
        return self.get_test_variant("greeting_style", user_identifier)

    def get_tutoring_variant(self, user_identifier: str) -> str:
        """Get consistent tutoring approach variant for user"""
        return self.get_test_variant("tutoring_approach", user_identifier)

    def get_test_variant(self, test_name: str, user_identifier: str) -> str:
        """Get consistent test variant for user using deterministic assignment"""
        if test_name not in self.test_configs:
            logger.warning(f"Unknown test: {test_name}")
            return "default"

        config = self.test_configs[test_name]
        if not config["active"]:
            return config["variants"][0]  # Return first variant if test inactive

        # Create consistent hash-based assignment
        user_hash = hashlib.md5(f"{test_name}_{user_identifier}".encode()).hexdigest()
        hash_int = int(user_hash[:8], 16)

        # Map hash to variant based on weights
        cumulative_weights = np.cumsum(config["weights"])
        random_value = (hash_int % 10000) / 10000.0  # Normalize to 0-1

        for i, threshold in enumerate(cumulative_weights):
            if random_value <= threshold:
                selected_variant = config["variants"][i]

                # Log assignment
                wandb.log(
                    {
                        "ab_test_assignment": {
                            "test_name": test_name,
                            "user_hash": user_hash[:8],
                            "variant": selected_variant,
                            "random_value": random_value,
                        }
                    }
                )

                return selected_variant

        # Fallback to last variant
        return config["variants"][-1]

    async def record_test_result(
        self,
        test_name: str,
        variant: str,
        user_identifier: str,
        session_id: str,
        success_metrics: dict[str, float],
    ):
        """Record A/B test result"""
        try:
            result_data = {
                "ab_test_result": {
                    "test_name": test_name,
                    "variant": variant,
                    "user_hash": hashlib.sha256(user_identifier.encode()).hexdigest()[:8],
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    **success_metrics,
                }
            }

            # Log to W&B
            wandb.log(result_data)

        except Exception as e:
            logger.error(f"Error recording A/B test result: {e}")

    async def analyze_test_results(self, test_name: str) -> dict[str, Any]:
        """Analyze A/B test results to determine winning variant"""
        try:
            # In production, this would query W&B API for historical results
            # For now, return mock analysis

            analysis = {
                "test_name": test_name,
                "status": "running",
                "variants": self.test_configs.get(test_name, {}).get("variants", []),
                "sample_sizes": {
                    "enthusiastic": 150,
                    "professional": 145,
                    "friendly": 155,
                },
                "conversion_rates": {
                    "enthusiastic": 0.68,
                    "professional": 0.72,
                    "friendly": 0.65,
                },
                "statistical_significance": False,
                "recommended_action": "continue_test",
                "confidence_level": 0.85,
            }

            # Log analysis
            wandb.log({"ab_test_analysis": analysis})

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing A/B test results: {e}")
            return {"error": str(e)}


# Global instances
prompt_tuner = PromptTuner()
ab_test_manager = ABTestManager()
