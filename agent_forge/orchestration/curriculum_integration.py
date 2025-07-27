"""Integration layer for multi-model orchestration with existing curriculum learning system.

This module bridges the new OpenRouter orchestration with the existing Agent Forge
training pipeline, preserving all current functionality while adding intelligent routing.
"""

import asyncio
import logging
from typing import Any

import torch

from ..training.curriculum import CurriculumGenerator, Question
from ..training.magi_specialization import FrontierQuestionGenerator, MagiConfig
from .openrouter_client import OpenRouterClient
from .task_router import TaskContext, TaskRouter

logger = logging.getLogger(__name__)


class EnhancedFrontierQuestionGenerator(FrontierQuestionGenerator):
    """Enhanced question generator that uses OpenRouter for high-quality generation."""

    def __init__(self, config: MagiConfig, use_openrouter: bool = True):
        """Initialize enhanced question generator.

        Args:
            config: Magi configuration
            use_openrouter: Whether to use OpenRouter (falls back to local if False)
        """
        super().__init__(config)
        self.use_openrouter = use_openrouter

        if self.use_openrouter:
            self.router = TaskRouter()
            logger.info("Enhanced question generator initialized with OpenRouter")
        else:
            logger.info("Enhanced question generator using local generation")

    def _generate_single_question(self, area: str, difficulty: int) -> Question:
        """Generate a single question using OpenRouter or fallback to local generation."""
        if not self.use_openrouter:
            # Fall back to parent implementation
            return super()._generate_single_question(area, difficulty)

        # Use OpenRouter for high-quality generation
        try:
            # Check if we're already in an async context
            try:
                loop = asyncio.get_running_loop()
                # If we're in an async context, create a task
                task = loop.create_task(
                    self._generate_with_openrouter(area, difficulty)
                )
                # For now, fall back to sync generation to avoid blocking
                logger.info(
                    "Async context detected, falling back to local generation for now"
                )
                return super()._generate_single_question(area, difficulty)
            except RuntimeError:
                # Not in async context, safe to use asyncio.run
                return asyncio.run(self._generate_with_openrouter(area, difficulty))
        except Exception as e:
            logger.warning(
                f"OpenRouter generation failed: {e}. Falling back to local generation."
            )
            return super()._generate_single_question(area, difficulty)

    async def _generate_with_openrouter(self, area: str, difficulty: int) -> Question:
        """Generate a question using OpenRouter."""
        # Create context for routing
        context = TaskContext(
            difficulty_level=difficulty,
            domain=area,
            expected_length="medium",
            requires_reasoning=True,
            requires_creativity=True,
            quality_priority=difficulty >= 7,  # High quality for difficult questions
        )

        # Generate prompt based on area and difficulty
        prompt = self._create_generation_prompt(area, difficulty)

        # Route to appropriate model
        response = await self.router.route_task(prompt, context)

        # Parse response into Question format
        return self._parse_response_to_question(response.content, area, difficulty)

    def _create_generation_prompt(self, area: str, difficulty: int) -> str:
        """Create a detailed prompt for question generation."""
        # Get difficulty-specific parameters
        params = self._get_difficulty_parameters(area, difficulty)

        area_descriptions = {
            "python_programming": "Python programming and software development",
            "algorithm_design": "algorithm design and optimization",
            "mathematical_proofs": "mathematical proofs and formal reasoning",
            "computational_complexity": "computational complexity and analysis",
            "data_structures": "data structures and their implementations",
            "numerical_analysis": "numerical methods and computational mathematics",
        }

        prompt = f"""Generate a {area_descriptions.get(area, area)} problem for training a specialized AI agent.

Difficulty Level: {difficulty}/10
{"Beginner level - basic concepts" if difficulty <= 3 else ""}
{"Intermediate level - practical application" if 4 <= difficulty <= 6 else ""}
{"Advanced level - complex reasoning required" if 7 <= difficulty <= 8 else ""}
{"Expert level - cutting-edge challenges" if difficulty >= 9 else ""}

Requirements:
- The problem should test deep understanding, not just memorization
- Include edge cases and potential pitfalls
- Require {params.get("reasoning_steps", 3)} steps of reasoning
- Code complexity: {params.get("complexity", "moderate")}

Format:
Problem: [Clear problem statement]
Answer: [Detailed solution with explanation]

Make the problem specific, unambiguous, and educational."""

        return prompt

    def _parse_response_to_question(
        self, response: str, area: str, difficulty: int
    ) -> Question:
        """Parse OpenRouter response into Question object."""
        # Simple parsing - in production would use more sophisticated parsing
        parts = response.split("Answer:", 1)

        if len(parts) == 2:
            problem = parts[0].replace("Problem:", "").strip()
            answer = parts[1].strip()
        else:
            # Fallback if format is different
            problem = response.strip()
            answer = "Solution parsing required"

        return Question(text=problem, answer=answer, difficulty=difficulty, domain=area)

    def generate_curriculum_questions(self) -> list[Question]:
        """Generate curriculum questions with optional variations."""
        questions = []

        for level in range(1, self.config.curriculum_levels + 1):
            level_questions = []

            # Generate base questions
            base_questions = self._generate_level_questions(level)
            level_questions.extend(base_questions)

            # Add variations for higher difficulty levels
            if self.use_openrouter and level >= 5:
                logger.info(f"Generating variations for level {level}")
                variations = asyncio.run(
                    self._generate_variations_batch(base_questions[:10])
                )
                level_questions.extend(variations)

            questions.extend(level_questions)
            logger.info(f"Generated {len(level_questions)} questions for level {level}")

        return questions

    async def _generate_variations_batch(
        self, base_questions: list[Question]
    ) -> list[Question]:
        """Generate variations of questions in batch."""
        variations = []

        for question in base_questions:
            context = TaskContext(
                difficulty_level=question.difficulty,
                domain=question.domain,
                expected_length="medium",
                requires_reasoning=False,
                requires_creativity=True,
                cost_sensitive=True,
                quality_priority=False,
            )

            prompt = f"""Create a variation of this problem that tests the same concepts:

Original: {question.text}

The variation should:
- Test the same core concepts
- Have similar difficulty
- Use different context or values
- Not be trivially similar"""

            try:
                response = await self.router.route_task(prompt, context)
                variation_text = response.content

                # Create variation question
                variation = Question(
                    text=variation_text,
                    answer="",  # Would need separate answer generation
                    difficulty=question.difficulty,
                    domain=question.domain,
                )
                variations.append(variation)

            except Exception as e:
                logger.warning(f"Failed to generate variation: {e}")

        return variations


class EnhancedCurriculumGenerator(CurriculumGenerator):
    """Enhanced curriculum generator with OpenRouter integration."""

    def __init__(self, frontier_model: str, domain: str, use_openrouter: bool = True):
        """Initialize enhanced curriculum generator.

        Args:
            frontier_model: Model name/path (used for fallback)
            domain: Domain for curriculum
            use_openrouter: Whether to use OpenRouter
        """
        self.domain = domain
        self.use_openrouter = use_openrouter
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if use_openrouter:
            self.router = TaskRouter()
            logger.info("Enhanced curriculum generator using OpenRouter")
        else:
            # Fall back to local model
            super().__init__(frontier_model, domain)

    def _generate(self, prompt: str, max_length: int = 200) -> str:
        """Generate using OpenRouter or fall back to local model."""
        if not self.use_openrouter or not hasattr(self, "router"):
            return super()._generate(prompt, max_length)

        try:
            # Use OpenRouter
            context = TaskContext(
                difficulty_level=5,
                domain=self.domain,
                expected_length="medium",
                requires_reasoning=True,
                requires_creativity=True,
            )

            response = asyncio.run(self.router.route_task(prompt, context))
            return response.content

        except Exception as e:
            logger.warning(f"OpenRouter failed: {e}. Using local generation.")
            return super()._generate(prompt, max_length)


class MultiModelOrchestrator:
    """Main orchestrator for multi-model training integration."""

    def __init__(self, config: MagiConfig, enable_openrouter: bool = True):
        """Initialize the orchestrator.

        Args:
            config: Magi configuration
            enable_openrouter: Whether to enable OpenRouter integration
        """
        self.config = config
        self.enable_openrouter = enable_openrouter

        if enable_openrouter:
            self.client = OpenRouterClient()
            self.router = TaskRouter(self.client)
            logger.info("Multi-model orchestration enabled")

        # Initialize enhanced generators
        self.question_generator = EnhancedFrontierQuestionGenerator(
            config, use_openrouter=enable_openrouter
        )

    async def evaluate_answer_with_explanation(
        self, question: Question, generated_answer: str
    ) -> dict[str, Any]:
        """Evaluate an answer using OpenRouter for better accuracy."""
        if not self.enable_openrouter:
            # Fall back to simple evaluation
            return {
                "correct": self._simple_evaluation(question, generated_answer),
                "explanation": "Local evaluation",
                "confidence": 0.7,
            }

        # Use router for sophisticated evaluation
        result = await self.router.evaluate_with_explanation(
            question.text, generated_answer, question.answer
        )

        return result

    def _simple_evaluation(self, question: Question, answer: str) -> bool:
        """Simple keyword-based evaluation as fallback."""
        expected_keywords = question.answer.lower().split()
        answer_keywords = answer.lower().split()

        overlap = len(set(expected_keywords) & set(answer_keywords))
        return overlap / len(expected_keywords) > 0.5 if expected_keywords else False

    async def generate_research_context(
        self, topic: str, max_length: int = 4000
    ) -> str:
        """Generate research context for a topic using long-context model."""
        if not self.enable_openrouter:
            return f"Research context for {topic}"

        context = TaskContext(
            difficulty_level=7,
            domain="research",
            expected_length="long",
            requires_reasoning=True,
            requires_creativity=False,
        )

        prompt = f"""Provide comprehensive research context for training an AI agent on: {topic}

Include:
1. Historical development and key milestones
2. Current state of the art
3. Major challenges and open problems
4. Practical applications
5. Future directions

Be thorough and educational."""

        response = await self.router.route_task(prompt, context, max_tokens=max_length)

        return response.content

    def get_cost_summary(self) -> dict[str, Any]:
        """Get cost summary for the orchestration session."""
        if not self.enable_openrouter:
            return {"enabled": False}

        return {
            "enabled": True,
            "metrics": self.client.get_metrics_summary(),
            "routing_stats": self.router.get_routing_stats(),
        }

    async def close(self):
        """Clean up resources."""
        if self.enable_openrouter:
            await self.client.close()
            logger.info("Multi-model orchestrator closed")
