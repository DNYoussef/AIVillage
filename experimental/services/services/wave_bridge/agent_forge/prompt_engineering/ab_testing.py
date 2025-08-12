"""A/B Testing Framework for Prompt Performance Tracking
Part B: Agent Forge Phase 4 - Prompt Engineering.
"""

import asyncio
import hashlib
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

import wandb

logger = logging.getLogger(__name__)


@dataclass
class TestVariant:
    """A/B test variant configuration."""

    variant_id: str
    variant_name: str
    prompt_template: str
    configuration: dict[str, Any]
    weight: float = 1.0
    active: bool = True
    created_at: str = ""


@dataclass
class InteractionResult:
    """Result from a single prompt interaction."""

    session_id: str
    variant_id: str
    user_message: str
    response_text: str
    response_time: float
    language: str
    engagement_signals: dict[str, Any]
    performance_metrics: dict[str, float]
    timestamp: str = ""


@dataclass
class TestResults:
    """Statistical results from A/B test analysis."""

    variant_id: str
    total_interactions: int
    avg_engagement: float
    avg_response_time: float
    conversion_rate: float
    confidence_interval: tuple[float, float]
    statistical_significance: float
    recommendation: str = ""


class PromptABTest:
    """Track prompt performance across student interactions with advanced A/B testing."""

    def __init__(self, project_name: str = "aivillage-tutoring") -> None:
        self.project_name = project_name
        self.variant_performance = defaultdict(list)
        self.active_tests = {}
        self.interaction_history = deque(maxlen=10000)  # Keep last 10k interactions
        self.user_assignments = {}  # Consistent variant assignment per user

        # Test configuration
        self.min_sample_size = 30
        self.confidence_level = 0.95
        self.effect_size_threshold = 0.05  # Minimum detectable effect

        # Performance tracking
        self.daily_metrics = defaultdict(lambda: defaultdict(list))
        self.hourly_metrics = defaultdict(lambda: defaultdict(list))

        # UCB1 algorithm parameters for multi-armed bandit
        self.exploration_constant = 2.0
        self.total_interactions = 0

        # Initialize W&B tracking
        self.initialize_wandb_tracking()

        # Set up default test variants
        self.setup_default_tests()

    def initialize_wandb_tracking(self) -> None:
        """Initialize W&B tracking for A/B testing."""
        try:
            wandb.init(
                project=self.project_name,
                job_type="ab_testing",
                config={
                    "framework": "prompt_ab_test",
                    "version": "1.0.0",
                    "min_sample_size": self.min_sample_size,
                    "confidence_level": self.confidence_level,
                    "algorithms": ["ucb1", "thompson_sampling", "epsilon_greedy"],
                },
            )

            logger.info("W&B A/B testing tracking initialized")

        except Exception as e:
            logger.exception(f"Failed to initialize W&B for A/B testing: {e}")

    def setup_default_tests(self) -> None:
        """Set up default A/B test variants."""
        # Greeting style test
        greeting_variants = [
            TestVariant(
                variant_id="greeting_formal",
                variant_name="Formal Greeting",
                prompt_template="Hello! I'm here to assist you with your learning. What subject would you like to explore?",
                configuration={"style": "formal", "emoji": False, "enthusiasm": "low"},
                weight=0.25,
            ),
            TestVariant(
                variant_id="greeting_friendly",
                variant_name="Friendly Greeting",
                prompt_template="Hi there! 😊 I'm excited to help you learn today! What can I help you understand?",
                configuration={
                    "style": "friendly",
                    "emoji": True,
                    "enthusiasm": "medium",
                },
                weight=0.25,
            ),
            TestVariant(
                variant_id="greeting_encouraging",
                variant_name="Encouraging Greeting",
                prompt_template="Hey! You're taking a great step by asking for help! I'm here to support your learning journey. What's on your mind?",
                configuration={
                    "style": "encouraging",
                    "emoji": False,
                    "enthusiasm": "high",
                },
                weight=0.25,
            ),
            TestVariant(
                variant_id="greeting_playful",
                variant_name="Playful Greeting",
                prompt_template="Greetings, fellow knowledge explorer! 🚀 Ready to unlock some amazing learning today? What adventure shall we begin?",
                configuration={
                    "style": "playful",
                    "emoji": True,
                    "enthusiasm": "very_high",
                },
                weight=0.25,
            ),
        ]

        # Tutoring approach test
        tutoring_variants = [
            TestVariant(
                variant_id="tutoring_direct",
                variant_name="Direct Teaching",
                prompt_template="Here's the concept: {concept}. Let me explain it step by step: {steps}. Now try applying it to your problem.",
                configuration={
                    "approach": "direct",
                    "complexity": "low",
                    "interaction": "minimal",
                },
                weight=0.33,
            ),
            TestVariant(
                variant_id="tutoring_socratic",
                variant_name="Socratic Method",
                prompt_template="Great question! Before I give you the answer, what do you think happens when {concept}? Can you guess why that might be?",
                configuration={
                    "approach": "socratic",
                    "complexity": "high",
                    "interaction": "maximum",
                },
                weight=0.33,
            ),
            TestVariant(
                variant_id="tutoring_guided",
                variant_name="Guided Discovery",
                prompt_template="Let's explore this together! I'll give you some hints: {hints}. What patterns do you notice? How might these apply to your question?",
                configuration={
                    "approach": "guided",
                    "complexity": "medium",
                    "interaction": "moderate",
                },
                weight=0.34,
            ),
        ]

        # Store test configurations
        self.active_tests["greeting_style"] = greeting_variants
        self.active_tests["tutoring_approach"] = tutoring_variants

        # Log test setup to W&B
        wandb.log(
            {
                "ab_tests_initialized": True,
                "test_types": list(self.active_tests.keys()),
                "total_variants": sum(
                    len(variants) for variants in self.active_tests.values()
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def get_user_variant(self, user_id: str, test_type: str) -> TestVariant:
        """Get consistent variant assignment for user using UCB1 algorithm."""
        # Check if user already has assignment
        user_key = f"{user_id}_{test_type}"
        if user_key in self.user_assignments:
            variant_id = self.user_assignments[user_key]
            return next(
                v for v in self.active_tests[test_type] if v.variant_id == variant_id
            )

        # UCB1 algorithm for variant selection
        variants = self.active_tests.get(test_type, [])
        if not variants:
            logger.warning(f"No variants found for test type: {test_type}")
            return None

        # Calculate UCB1 scores for each variant
        best_variant = None
        best_score = -float("inf")

        for variant in variants:
            if not variant.active:
                continue

            # Get performance data for this variant
            variant_data = self.variant_performance[variant.variant_id]

            if len(variant_data) == 0:
                # No data yet, prioritize exploration
                score = float("inf")
            else:
                # Calculate average reward (engagement score)
                avg_reward = statistics.mean(
                    [
                        r.performance_metrics.get("student_engagement", 0)
                        for r in variant_data
                    ]
                )

                # Calculate confidence bound
                n_variant = len(variant_data)
                confidence_bound = np.sqrt(
                    self.exploration_constant
                    * np.log(max(1, self.total_interactions))
                    / n_variant
                )

                score = avg_reward + confidence_bound

            if score > best_score:
                best_score = score
                best_variant = variant

        # Assign user to selected variant
        if best_variant:
            self.user_assignments[user_key] = best_variant.variant_id

            # Log assignment to W&B
            wandb.log(
                {
                    "variant_assignment": True,
                    "user_hash": hashlib.sha256(user_id.encode()).hexdigest()[:8],
                    "test_type": test_type,
                    "variant_id": best_variant.variant_id,
                    "ucb1_score": best_score,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        return best_variant

    async def run_test_interaction(
        self,
        student_msg: str,
        user_id: str,
        test_type: str,
        language: str = "en",
        context: dict[str, Any] | None = None,
    ) -> InteractionResult:
        """Test a prompt variant and log results."""
        context = context or {}
        start_time = time.time()

        try:
            # Get variant for user
            variant = self.get_user_variant(user_id, test_type)
            if not variant:
                logger.error(f"No variant available for test type: {test_type}")
                return None

            # Generate response with variant
            response = await self.generate_with_prompt(
                student_msg, variant.prompt_template, context
            )

            response_time = time.time() - start_time

            # Analyze engagement signals
            engagement_signals = await self.analyze_engagement_signals(
                student_msg, response, context
            )

            # Calculate performance metrics
            performance_metrics = await self.calculate_performance_metrics(
                response, response_time, engagement_signals
            )

            # Create interaction result
            interaction_result = InteractionResult(
                session_id=context.get("session_id", ""),
                variant_id=variant.variant_id,
                user_message=student_msg,
                response_text=response,
                response_time=response_time,
                language=language,
                engagement_signals=engagement_signals,
                performance_metrics=performance_metrics,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            # Store result
            self.variant_performance[variant.variant_id].append(interaction_result)
            self.interaction_history.append(interaction_result)
            self.total_interactions += 1

            # Log to W&B
            await self.log_interaction_to_wandb(interaction_result, variant)

            # Update daily/hourly metrics
            await self.update_time_based_metrics(interaction_result)

            logger.info(
                f"A/B test interaction completed: {variant.variant_id} in {response_time:.2f}s"
            )

            return interaction_result

        except Exception as e:
            logger.exception(f"Error in A/B test interaction: {e}")
            return None

    async def generate_with_prompt(
        self, student_msg: str, prompt_template: str, context: dict[str, Any]
    ) -> str:
        """Generate response using the specified prompt template."""
        # This is a simplified version - in production, this would integrate with
        # the actual AI models (Anthropic/OpenAI)

        # Fill in template variables
        prompt_template.format(
            user_message=student_msg,
            concept=context.get("concept", "the main idea"),
            steps=context.get(
                "steps",
                "1. Identify the problem 2. Apply the concept 3. Check your answer",
            ),
            hints=context.get("hints", "Think about what you already know"),
            **context,
        )

        # Simulate AI response (replace with actual AI call)
        await asyncio.sleep(0.1)  # Simulate processing time

        # Mock response based on template style
        if "playful" in prompt_template.lower():
            response = f"🚀 Let's dive into this! {student_msg[:30]}... sounds like an exciting challenge!"
        elif "socratic" in prompt_template.lower():
            response = f"Interesting question! What do you think might happen if we consider {student_msg[:20]}...?"
        elif "direct" in prompt_template.lower():
            response = f"Here's the key concept for your question about {student_msg[:30]}... Let me explain step by step."
        else:
            response = f"Thanks for your question about {student_msg[:30]}... I'm here to help you understand this concept."

        return response

    async def analyze_engagement_signals(
        self, student_msg: str, response: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze engagement signals from the interaction."""
        signals = {
            "question_asked": "?" in response,
            "emoji_used": any(char in response for char in "😊🚀✨👍🎉💡"),
            "encouragement_present": any(
                word in response.lower()
                for word in ["great", "excellent", "good", "well done"]
            ),
            "interactive_element": any(
                phrase in response.lower()
                for phrase in ["what do you think", "can you", "try this"]
            ),
            "example_provided": any(
                word in response.lower()
                for word in ["example", "for instance", "imagine"]
            ),
            "response_relevance": len(
                set(student_msg.lower().split()) & set(response.lower().split())
            )
            / max(1, len(student_msg.split())),
            "response_length_appropriate": 50 <= len(response.split()) <= 200,
            "follow_up_potential": "?" in response or "what" in response.lower(),
        }

        return signals

    async def calculate_performance_metrics(
        self, response: str, response_time: float, engagement_signals: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate comprehensive performance metrics."""
        # Engagement score based on signals
        engagement_factors = [
            engagement_signals.get("question_asked", False),
            engagement_signals.get("emoji_used", False),
            engagement_signals.get("encouragement_present", False),
            engagement_signals.get("interactive_element", False),
            engagement_signals.get("example_provided", False),
            engagement_signals.get("follow_up_potential", False),
        ]

        engagement_score = sum(engagement_factors) / len(engagement_factors)

        # Response quality score
        quality_score = 0.0
        if engagement_signals.get("response_length_appropriate", False):
            quality_score += 0.3
        if engagement_signals.get("response_relevance", 0) > 0.2:
            quality_score += 0.4
        if engagement_signals.get("example_provided", False):
            quality_score += 0.3

        # Efficiency score (response time)
        efficiency_score = max(0.0, min(1.0, (5.0 - response_time) / 5.0))

        # Overall performance (weighted combination)
        performance_metrics = {
            "student_engagement": engagement_score,
            "response_quality": quality_score,
            "response_efficiency": efficiency_score,
            "overall_performance": (
                engagement_score * 0.5 + quality_score * 0.3 + efficiency_score * 0.2
            ),
            "contains_encouragement": float(
                engagement_signals.get("encouragement_present", False)
            ),
            "math_symbols_used": float(any(symbol in response for symbol in "+=÷×∑∫√")),
            "interactive_elements": float(
                engagement_signals.get("interactive_element", False)
            ),
        }

        return performance_metrics

    async def log_interaction_to_wandb(
        self, interaction: InteractionResult, variant: TestVariant
    ) -> None:
        """Log detailed interaction data to W&B."""
        log_data = {
            "prompt_variant": variant.variant_id,
            "variant_name": variant.variant_name,
            "response_time": interaction.response_time,
            "response_length": len(interaction.response_text),
            "language": interaction.language,
            "session_id": interaction.session_id,
            "timestamp": interaction.timestamp,
            **interaction.performance_metrics,
            **{f"signal_{k}": v for k, v in interaction.engagement_signals.items()},
        }

        wandb.log(log_data)

    async def update_time_based_metrics(self, interaction: InteractionResult) -> None:
        """Update daily and hourly performance metrics."""
        current_time = datetime.now()
        day_key = current_time.strftime("%Y-%m-%d")
        hour_key = current_time.strftime("%Y-%m-%d-%H")

        # Update daily metrics
        self.daily_metrics[day_key][interaction.variant_id].append(
            interaction.performance_metrics["overall_performance"]
        )

        # Update hourly metrics
        self.hourly_metrics[hour_key][interaction.variant_id].append(
            interaction.performance_metrics["overall_performance"]
        )

    async def analyze_test_results(
        self, test_type: str, min_interactions: int | None = None
    ) -> list[TestResults]:
        """Analyze A/B test results with statistical significance."""
        min_interactions = min_interactions or self.min_sample_size
        variants = self.active_tests.get(test_type, [])

        results = []

        for variant in variants:
            variant_data = self.variant_performance[variant.variant_id]

            if len(variant_data) < min_interactions:
                logger.info(
                    f"Variant {variant.variant_id} has insufficient data ({len(variant_data)} < {min_interactions})"
                )
                continue

            # Calculate statistics
            engagement_scores = [
                r.performance_metrics.get("student_engagement", 0) for r in variant_data
            ]
            response_times = [r.response_time for r in variant_data]

            avg_engagement = statistics.mean(engagement_scores)
            avg_response_time = statistics.mean(response_times)

            # Calculate conversion rate (high engagement threshold)
            high_engagement_count = sum(1 for score in engagement_scores if score > 0.7)
            conversion_rate = high_engagement_count / len(engagement_scores)

            # Calculate confidence interval
            std_err = statistics.stdev(engagement_scores) / np.sqrt(
                len(engagement_scores)
            )
            margin_error = 1.96 * std_err  # 95% confidence
            confidence_interval = (
                avg_engagement - margin_error,
                avg_engagement + margin_error,
            )

            # Statistical significance (placeholder - would need proper statistical test)
            statistical_significance = min(0.99, max(0.01, avg_engagement))

            # Generate recommendation
            recommendation = self.generate_recommendation(
                avg_engagement, conversion_rate, avg_response_time, len(variant_data)
            )

            result = TestResults(
                variant_id=variant.variant_id,
                total_interactions=len(variant_data),
                avg_engagement=avg_engagement,
                avg_response_time=avg_response_time,
                conversion_rate=conversion_rate,
                confidence_interval=confidence_interval,
                statistical_significance=statistical_significance,
                recommendation=recommendation,
            )

            results.append(result)

        # Sort by performance
        results.sort(key=lambda r: r.avg_engagement, reverse=True)

        # Log analysis to W&B
        wandb.log(
            {
                "ab_test_analysis": True,
                "test_type": test_type,
                "variants_analyzed": len(results),
                "best_variant": results[0].variant_id if results else None,
                "best_engagement": results[0].avg_engagement if results else 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return results

    def generate_recommendation(
        self,
        avg_engagement: float,
        conversion_rate: float,
        avg_response_time: float,
        sample_size: int,
    ) -> str:
        """Generate actionable recommendation based on test results."""
        if avg_engagement > 0.8 and conversion_rate > 0.6:
            return "WINNER - Deploy this variant"
        if avg_engagement > 0.7 and avg_response_time < 3.0:
            return "STRONG PERFORMER - Consider promoting"
        if sample_size < 100:
            return "NEED MORE DATA - Continue testing"
        if avg_engagement < 0.5:
            return "POOR PERFORMER - Consider discontinuing"
        if avg_response_time > 4.0:
            return "OPTIMIZATION NEEDED - Improve response time"
        return "MODERATE PERFORMER - Monitor closely"

    async def get_daily_report(self, days_back: int = 7) -> dict[str, Any]:
        """Generate daily performance report."""
        report = {
            "report_date": datetime.now().strftime("%Y-%m-%d"),
            "period": f"Last {days_back} days",
            "total_interactions": self.total_interactions,
            "active_tests": len(self.active_tests),
            "daily_breakdown": {},
            "top_performers": [],
            "recommendations": [],
        }

        # Analyze daily performance
        current_date = datetime.now()
        for i in range(days_back):
            date_key = (current_date - timedelta(days=i)).strftime("%Y-%m-%d")

            if date_key in self.daily_metrics:
                day_data = self.daily_metrics[date_key]
                daily_summary = {}

                for variant_id, scores in day_data.items():
                    if scores:
                        daily_summary[variant_id] = {
                            "interactions": len(scores),
                            "avg_performance": statistics.mean(scores),
                            "performance_trend": (
                                "up" if statistics.mean(scores) > 0.7 else "down"
                            ),
                        }

                report["daily_breakdown"][date_key] = daily_summary

        # Identify top performers
        all_variant_scores = {}
        for variant_id, interactions in self.variant_performance.items():
            if len(interactions) >= 20:  # Minimum threshold
                scores = [
                    i.performance_metrics.get("overall_performance", 0)
                    for i in interactions
                ]
                all_variant_scores[variant_id] = {
                    "avg_score": statistics.mean(scores),
                    "interaction_count": len(interactions),
                    "latest_performance": (
                        statistics.mean(scores[-10:])
                        if len(scores) >= 10
                        else statistics.mean(scores)
                    ),
                }

        # Sort and get top 3
        top_variants = sorted(
            all_variant_scores.items(), key=lambda x: x[1]["avg_score"], reverse=True
        )[:3]

        report["top_performers"] = [
            {
                "variant_id": variant_id,
                "score": data["avg_score"],
                "interactions": data["interaction_count"],
            }
            for variant_id, data in top_variants
        ]

        # Generate recommendations
        if len(top_variants) > 0 and top_variants[0][1]["avg_score"] > 0.8:
            report["recommendations"].append(
                f"Deploy {top_variants[0][0]} - consistently high performance"
            )

        if self.total_interactions < 1000:
            report["recommendations"].append(
                "Increase test volume - need more data for statistical confidence"
            )

        # Log report to W&B
        wandb.log(
            {
                "daily_report_generated": True,
                "report_summary": {
                    "total_interactions": report["total_interactions"],
                    "top_performer": (
                        report["top_performers"][0]["variant_id"]
                        if report["top_performers"]
                        else None
                    ),
                    "recommendations_count": len(report["recommendations"]),
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return report


# Global instance for easy access
prompt_ab_test = PromptABTest()
