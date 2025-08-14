"""Mathematical Fitness Evaluator for Tutor Models
Sprint R-4+AF1: Agent Forge Phase 1 - Task B.3.
"""

import asyncio
import logging
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import torch

import wandb

logger = logging.getLogger(__name__)


@dataclass
class MathProblem:
    """Mathematical problem for evaluation."""

    problem_id: str
    category: str  # arithmetic, algebra, geometry, word_problems, etc.
    grade_level: int
    difficulty: float  # 0.0 to 1.0
    problem_text: str
    expected_answer: str
    solution_steps: list[str]
    keywords: list[str]
    scoring_criteria: dict[str, float]
    cultural_context: str = "general"
    language: str = "en"


@dataclass
class EvaluationResult:
    """Result of model evaluation on a problem."""

    problem_id: str
    model_response: str
    correctness_score: float
    step_by_step_score: float
    explanation_quality: float
    encouragement_score: float
    cultural_sensitivity: float
    response_time: float
    total_score: float
    feedback: str
    evaluation_time: float = 0.0
    timestamp: str = ""


class MathFitnessEvaluator:
    """Comprehensive evaluation system for math tutoring model fitness."""

    def __init__(
        self,
        project_name: str = "agent-forge",
        max_concurrent_evaluations: int = 5,
    ) -> None:
        self.project_name = project_name
        self.test_suite = {}
        self.evaluation_history = []
        self.model_performance_cache = {}
        self.max_concurrent_evaluations = max_concurrent_evaluations
        self.kpi_scores: dict[str, float] = {}

        # Scoring weights
        self.scoring_weights = {
            "correctness": 0.35,
            "step_by_step": 0.25,
            "explanation_quality": 0.20,
            "encouragement": 0.10,
            "cultural_sensitivity": 0.10,
        }

        # Performance thresholds
        self.performance_thresholds = {
            "excellent": 0.85,
            "good": 0.70,
            "acceptable": 0.55,
            "poor": 0.40,
        }

        # Initialize test problems
        asyncio.create_task(self.initialize_test_suite())

    async def initialize_test_suite(self) -> None:
        """Initialize comprehensive test suite for K-8 mathematics."""
        logger.info("Initializing mathematical fitness evaluation test suite")

        # Arithmetic problems (Grades K-3)
        arithmetic_problems = [
            MathProblem(
                problem_id="arith_001",
                category="arithmetic",
                grade_level=1,
                difficulty=0.2,
                problem_text="What is 5 + 3?",
                expected_answer="8",
                solution_steps=["Count 5", "Add 3 more", "Total is 8"],
                keywords=["addition", "sum", "plus"],
                scoring_criteria={"correct_answer": 0.6, "method_shown": 0.4},
            ),
            MathProblem(
                problem_id="arith_002",
                category="arithmetic",
                grade_level=2,
                difficulty=0.3,
                problem_text="If you have 12 stickers and give away 5, how many do you have left?",
                expected_answer="7",
                solution_steps=["Start with 12", "Subtract 5", "12 - 5 = 7"],
                keywords=["subtraction", "take away", "difference"],
                scoring_criteria={
                    "correct_answer": 0.5,
                    "method_shown": 0.3,
                    "word_problem": 0.2,
                },
            ),
            MathProblem(
                problem_id="arith_003",
                category="arithmetic",
                grade_level=3,
                difficulty=0.4,
                problem_text="What is 6 × 4?",
                expected_answer="24",
                solution_steps=[
                    "6 groups of 4",
                    "4 + 4 + 4 + 4 + 4 + 4 = 24",
                    "Or 6 × 4 = 24",
                ],
                keywords=["multiplication", "times", "groups"],
                scoring_criteria={"correct_answer": 0.6, "method_shown": 0.4},
            ),
        ]

        # Algebra problems (Grades 4-8)
        algebra_problems = [
            MathProblem(
                problem_id="alg_001",
                category="algebra",
                grade_level=6,
                difficulty=0.5,
                problem_text="Solve for x: 2x + 5 = 13",
                expected_answer="x = 4",
                solution_steps=[
                    "2x + 5 = 13",
                    "2x = 13 - 5",
                    "2x = 8",
                    "x = 8 ÷ 2",
                    "x = 4",
                ],
                keywords=["variable", "solve", "equation"],
                scoring_criteria={
                    "correct_answer": 0.4,
                    "steps_shown": 0.4,
                    "isolation": 0.2,
                },
            ),
            MathProblem(
                problem_id="alg_002",
                category="algebra",
                grade_level=7,
                difficulty=0.6,
                problem_text="What is the value of 3x + 2 when x = 5?",
                expected_answer="17",
                solution_steps=["Substitute x = 5", "3(5) + 2", "15 + 2", "17"],
                keywords=["substitute", "evaluate", "expression"],
                scoring_criteria={
                    "correct_answer": 0.5,
                    "substitution": 0.3,
                    "calculation": 0.2,
                },
            ),
        ]

        # Geometry problems (Grades 3-8)
        geometry_problems = [
            MathProblem(
                problem_id="geom_001",
                category="geometry",
                grade_level=4,
                difficulty=0.4,
                problem_text="What is the area of a rectangle that is 6 units long and 4 units wide?",
                expected_answer="24 square units",
                solution_steps=[
                    "Area = length × width",
                    "Area = 6 × 4",
                    "Area = 24 square units",
                ],
                keywords=["area", "rectangle", "formula"],
                scoring_criteria={"correct_answer": 0.4, "formula": 0.3, "units": 0.3},
            ),
            MathProblem(
                problem_id="geom_002",
                category="geometry",
                grade_level=6,
                difficulty=0.6,
                problem_text="Find the circumference of a circle with radius 3 units. Use π ≈ 3.14.",
                expected_answer="18.84 units",
                solution_steps=[
                    "Circumference = 2πr",
                    "C = 2 × 3.14 × 3",
                    "C = 6.28 × 3",
                    "C = 18.84 units",
                ],
                keywords=["circumference", "circle", "pi", "radius"],
                scoring_criteria={
                    "correct_answer": 0.4,
                    "formula": 0.4,
                    "calculation": 0.2,
                },
            ),
        ]

        # Word problems (Grades 2-8)
        word_problems = [
            MathProblem(
                problem_id="word_001",
                category="word_problems",
                grade_level=3,
                difficulty=0.5,
                problem_text="Sarah has 24 apples. She wants to put them in bags with 6 apples each. How many bags does she need?",
                expected_answer="4 bags",
                solution_steps=[
                    "Total apples = 24",
                    "Apples per bag = 6",
                    "Number of bags = 24 ÷ 6",
                    "Number of bags = 4",
                ],
                keywords=["division", "equal groups", "how many"],
                scoring_criteria={
                    "correct_answer": 0.3,
                    "understanding": 0.3,
                    "operation": 0.2,
                    "reasoning": 0.2,
                },
            ),
            MathProblem(
                problem_id="word_002",
                category="word_problems",
                grade_level=5,
                difficulty=0.6,
                problem_text="A recipe calls for 2.5 cups of flour. If you want to make 3 batches, how much flour do you need?",
                expected_answer="7.5 cups",
                solution_steps=[
                    "Flour per batch = 2.5 cups",
                    "Number of batches = 3",
                    "Total flour = 2.5 × 3",
                    "Total flour = 7.5 cups",
                ],
                keywords=["decimals", "multiplication", "recipe"],
                scoring_criteria={
                    "correct_answer": 0.4,
                    "setup": 0.3,
                    "calculation": 0.3,
                },
            ),
        ]

        # Fractions problems (Grades 3-8)
        fraction_problems = [
            MathProblem(
                problem_id="frac_001",
                category="fractions",
                grade_level=4,
                difficulty=0.5,
                problem_text="What is 1/2 + 1/4?",
                expected_answer="3/4",
                solution_steps=[
                    "Find common denominator: 4",
                    "1/2 = 2/4",
                    "2/4 + 1/4 = 3/4",
                ],
                keywords=["fractions", "addition", "common denominator"],
                scoring_criteria={
                    "correct_answer": 0.4,
                    "common_denominator": 0.3,
                    "steps": 0.3,
                },
            ),
            MathProblem(
                problem_id="frac_002",
                category="fractions",
                grade_level=6,
                difficulty=0.7,
                problem_text="Simplify 12/18 to lowest terms.",
                expected_answer="2/3",
                solution_steps=[
                    "Find GCD of 12 and 18",
                    "GCD = 6",
                    "12 ÷ 6 = 2",
                    "18 ÷ 6 = 3",
                    "12/18 = 2/3",
                ],
                keywords=["simplify", "lowest terms", "GCD"],
                scoring_criteria={
                    "correct_answer": 0.5,
                    "process": 0.3,
                    "explanation": 0.2,
                },
            ),
        ]

        # Store problems by category
        self.test_suite = {
            "arithmetic": arithmetic_problems,
            "algebra": algebra_problems,
            "geometry": geometry_problems,
            "word_problems": word_problems,
            "fractions": fraction_problems,
        }

        # Log test suite initialization
        total_problems = sum(len(problems) for problems in self.test_suite.values())

        wandb.log(
            {
                "test_suite_initialized": True,
                "total_problems": total_problems,
                "categories": list(self.test_suite.keys()),
                "grade_range": "K-8",
                "difficulty_range": "0.2-0.7",
            }
        )

        logger.info(f"Test suite initialized with {total_problems} problems across {len(self.test_suite)} categories")

    async def evaluate(
        self,
        model,
        tokenizer,
        individual_id: str | None = None,
        log_details: bool = True,
    ) -> float:
        """Comprehensive fitness evaluation of a math tutoring model."""
        logger.info(f"Evaluating model {individual_id or 'unknown'}")

        start_time = asyncio.get_event_loop().time()

        # Check cache
        if individual_id and individual_id in self.model_performance_cache:
            cached_result = self.model_performance_cache[individual_id]
            if (datetime.now(timezone.utc) - datetime.fromisoformat(cached_result["timestamp"])).seconds < 3600:
                logger.info(f"Using cached evaluation for {individual_id}")
                self.kpi_scores = {
                    "fitness_score": cached_result["fitness_score"],
                    "evaluation_time": cached_result["evaluation_time"],
                    "avg_response_time": cached_result["avg_response_time"],
                    **{
                        f"{category}_score": score
                        for category, score in cached_result.get("category_scores", {}).items()
                    },
                }
                return cached_result["fitness_score"]

        category_scores: dict[str, float] = {}
        all_evaluations: list[EvaluationResult] = []
        category_evaluations: defaultdict[str, list[EvaluationResult]] = defaultdict(list)
        semaphore = asyncio.Semaphore(self.max_concurrent_evaluations)

        async def evaluate_with_limit(
            problem: MathProblem,
        ) -> tuple[str, EvaluationResult]:
            async with semaphore:
                try:
                    evaluation = await self.evaluate_problem(model, tokenizer, problem)
                except Exception as e:
                    logger.exception(f"Error evaluating problem {problem.problem_id}: {e}")
                    evaluation = EvaluationResult(
                        problem_id=problem.problem_id,
                        model_response="Error in evaluation",
                        correctness_score=0.0,
                        step_by_step_score=0.0,
                        explanation_quality=0.0,
                        encouragement_score=0.0,
                        cultural_sensitivity=0.0,
                        response_time=0.0,
                        total_score=0.0,
                        feedback="Evaluation failed",
                    )
                if log_details:
                    wandb.log(
                        {
                            f"evaluation/{problem.category}/{problem.problem_id}/score": evaluation.total_score,
                            f"evaluation/{problem.category}/{problem.problem_id}/correctness": evaluation.correctness_score,
                            f"evaluation/{problem.category}/{problem.problem_id}/explanation": evaluation.explanation_quality,
                            f"evaluation/{problem.category}/{problem.problem_id}/encouragement": evaluation.encouragement_score,
                            f"evaluation/{problem.category}/{problem.problem_id}/latency": evaluation.evaluation_time,
                            "model_id": individual_id or "unknown",
                        }
                    )
                return problem.category, evaluation

        tasks = [
            asyncio.create_task(evaluate_with_limit(problem))
            for problems in self.test_suite.values()
            for problem in problems
        ]
        results = await asyncio.gather(*tasks)

        for category, evaluation in results:
            category_evaluations[category].append(evaluation)
            all_evaluations.append(evaluation)

        for category in self.test_suite.keys():
            evaluations = category_evaluations.get(category, [])
            category_scores[category] = (
                statistics.mean([eval.total_score for eval in evaluations]) if evaluations else 0.0
            )

        # Calculate weighted fitness score
        fitness_score = self.calculate_weighted_fitness(category_scores)

        # Calculate additional metrics
        evaluation_time = asyncio.get_event_loop().time() - start_time
        avg_response_time = (
            statistics.mean([eval.response_time for eval in all_evaluations]) if all_evaluations else 0.0
        )

        # Store in cache
        if individual_id:
            self.model_performance_cache[individual_id] = {
                "fitness_score": fitness_score,
                "category_scores": category_scores,
                "evaluation_time": evaluation_time,
                "avg_response_time": avg_response_time,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Expose KPI scores for external consumption
        self.kpi_scores = {
            "fitness_score": fitness_score,
            "evaluation_time": evaluation_time,
            "avg_response_time": avg_response_time,
            **{f"{category}_score": score for category, score in category_scores.items()},
        }

        # Log comprehensive results
        if log_details:
            wandb.log(
                {
                    "fitness/total_score": fitness_score,
                    "fitness/evaluation_time": evaluation_time,
                    "fitness/avg_response_time": avg_response_time,
                    **{f"fitness/{category}": score for category, score in category_scores.items()},
                    "model_id": individual_id or "unknown",
                    "problems_evaluated": len(all_evaluations),
                }
            )

        logger.info(f"Model evaluation complete: fitness={fitness_score:.3f}, time={evaluation_time:.2f}s")

        return fitness_score

    async def evaluate_problem(self, model, tokenizer, problem: MathProblem) -> EvaluationResult:
        """Evaluate model performance on a single math problem."""
        start_time = asyncio.get_event_loop().time()

        # Create tutoring prompt
        prompt = self.create_tutoring_prompt(problem)

        try:
            # Generate model response
            model_response = await self.generate_model_response(model, tokenizer, prompt)

            response_time = asyncio.get_event_loop().time() - start_time

            # Evaluate different aspects
            correctness_score = self.evaluate_correctness(model_response, problem)
            step_by_step_score = self.evaluate_step_by_step(model_response, problem)
            explanation_quality = self.evaluate_explanation_quality(model_response, problem)
            encouragement_score = self.evaluate_encouragement(model_response)
            cultural_sensitivity = self.evaluate_cultural_sensitivity(model_response, problem)

            # Calculate total score
            total_score = (
                correctness_score * self.scoring_weights["correctness"]
                + step_by_step_score * self.scoring_weights["step_by_step"]
                + explanation_quality * self.scoring_weights["explanation_quality"]
                + encouragement_score * self.scoring_weights["encouragement"]
                + cultural_sensitivity * self.scoring_weights["cultural_sensitivity"]
            )

            # Generate feedback
            feedback = self.generate_feedback(
                correctness_score,
                step_by_step_score,
                explanation_quality,
                encouragement_score,
                cultural_sensitivity,
            )

            evaluation_time = asyncio.get_event_loop().time() - start_time

            return EvaluationResult(
                problem_id=problem.problem_id,
                model_response=model_response,
                correctness_score=correctness_score,
                step_by_step_score=step_by_step_score,
                explanation_quality=explanation_quality,
                encouragement_score=encouragement_score,
                cultural_sensitivity=cultural_sensitivity,
                response_time=response_time,
                total_score=total_score,
                feedback=feedback,
                evaluation_time=evaluation_time,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        except Exception as e:
            logger.exception(f"Error in problem evaluation: {e}")
            evaluation_time = asyncio.get_event_loop().time() - start_time
            return EvaluationResult(
                problem_id=problem.problem_id,
                model_response="Error generating response",
                correctness_score=0.0,
                step_by_step_score=0.0,
                explanation_quality=0.0,
                encouragement_score=0.0,
                cultural_sensitivity=0.0,
                response_time=0.0,
                total_score=0.0,
                feedback="Evaluation failed due to error",
                evaluation_time=evaluation_time,
            )

    def create_tutoring_prompt(self, problem: MathProblem) -> str:
        """Create an appropriate tutoring prompt for the problem."""
        grade_descriptor = self.get_grade_descriptor(problem.grade_level)

        prompt = f"""You are a helpful math tutor working with {grade_descriptor}. Your goal is to help students learn by explaining concepts clearly and encouraging them.

Problem: {problem.problem_text}

Please provide a complete tutoring response that:
1. Shows the correct answer
2. Explains the steps clearly
3. Uses encouraging language
4. Makes sure the explanation is appropriate for {grade_descriptor}

Your response:"""

        return prompt

    def get_grade_descriptor(self, grade_level: int) -> str:
        """Get appropriate grade level descriptor."""
        if grade_level == 0:
            return "a kindergarten student"
        if grade_level <= 3:
            return (
                f"a {grade_level}rd grade student"
                if grade_level == 3
                else (f"a {grade_level}st grade student" if grade_level == 1 else f"a {grade_level}nd grade student")
            )
        if grade_level <= 5 or grade_level <= 8:
            return f"a {grade_level}th grade student"
        return "a student"

    async def generate_model_response(self, model, tokenizer, prompt: str, max_length: int = 200) -> str:
        """Generate response from model."""
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

            # Move to model device
            if hasattr(model, "device"):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

            # Decode response (only new tokens)
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

            return response

        except Exception as e:
            logger.exception(f"Error generating model response: {e}")
            return "I'm having trouble generating a response right now."

    def evaluate_correctness(self, response: str, problem: MathProblem) -> float:
        """Evaluate correctness of the mathematical answer."""
        response_lower = response.lower()
        expected_answer = problem.expected_answer.lower()

        # Direct answer match
        if expected_answer in response_lower:
            return 1.0

        # Extract numbers from response and expected answer
        response_numbers = re.findall(r"-?\d+\.?\d*", response)
        expected_numbers = re.findall(r"-?\d+\.?\d*", problem.expected_answer)

        if not expected_numbers:
            return 0.5  # Can't evaluate if no expected number

        # Check if any extracted number matches expected
        expected_num = expected_numbers[0]
        if expected_num in response_numbers:
            return 0.8  # Good but not perfect match

        # Try to evaluate mathematical expressions
        try:
            # Simple arithmetic evaluation
            for num_str in response_numbers:
                if abs(float(num_str) - float(expected_num)) < 0.01:
                    return 0.7
        except Exception:
            pass

        # Check for conceptual understanding even if answer is wrong
        problem_keywords = problem.keywords
        if any(keyword in response_lower for keyword in problem_keywords):
            return 0.3  # Some understanding shown

        return 0.0

    def evaluate_step_by_step(self, response: str, problem: MathProblem) -> float:
        """Evaluate quality of step-by-step explanation."""
        response_lower = response.lower()
        score = 0.0

        # Check for step indicators
        step_indicators = [
            "step",
            "first",
            "then",
            "next",
            "finally",
            "1.",
            "2.",
            "3.",
            "•",
            "-",
            "start by",
            "now",
        ]

        step_count = sum(1 for indicator in step_indicators if indicator in response_lower)
        if step_count >= 2:
            score += 0.4
        elif step_count >= 1:
            score += 0.2

        # Check for mathematical process words
        process_words = [
            "add",
            "subtract",
            "multiply",
            "divide",
            "equals",
            "solve",
            "calculate",
            "find",
            "substitute",
        ]

        process_count = sum(1 for word in process_words if word in response_lower)
        score += min(0.3, process_count * 0.1)

        # Check for explanation of why
        explanation_words = ["because", "since", "so", "therefore", "this means"]
        if any(word in response_lower for word in explanation_words):
            score += 0.2

        # Check for connection to expected solution steps
        if problem.solution_steps:
            step_matches = 0
            for expected_step in problem.solution_steps:
                # Simple keyword matching
                expected_keywords = expected_step.lower().split()
                if any(keyword in response_lower for keyword in expected_keywords):
                    step_matches += 1

            if step_matches >= len(problem.solution_steps) // 2:
                score += 0.1

        return min(1.0, score)

    def evaluate_explanation_quality(self, response: str, problem: MathProblem) -> float:
        """Evaluate overall quality of explanation."""
        response_lower = response.lower()
        score = 0.0

        # Length appropriateness (not too short, not too long)
        word_count = len(response.split())
        if 20 <= word_count <= 150:
            score += 0.3
        elif 10 <= word_count <= 200:
            score += 0.2
        elif word_count >= 5:
            score += 0.1

        # Clarity indicators
        clarity_words = [
            "let me explain",
            "here's how",
            "think of it",
            "imagine",
            "for example",
            "like this",
            "in other words",
        ]

        if any(phrase in response_lower for phrase in clarity_words):
            score += 0.2

        # Grade-appropriate language
        if problem.grade_level <= 3:
            # Simple language for young students
            simple_indicators = ["easy", "simple", "think", "count", "see"]
            if any(word in response_lower for word in simple_indicators):
                score += 0.2
        else:
            # More sophisticated language for older students
            advanced_indicators = [
                "analyze",
                "understand",
                "concept",
                "method",
                "approach",
            ]
            if any(word in response_lower for word in advanced_indicators):
                score += 0.2

        # Engagement elements
        engagement_indicators = ["you", "your", "we", "let's", "try", "can you"]
        engagement_count = sum(1 for phrase in engagement_indicators if phrase in response_lower)
        score += min(0.2, engagement_count * 0.05)

        # Completeness (addresses the problem fully)
        if len(response_lower) > 50 and problem.problem_text.lower()[:20] in response_lower:
            score += 0.1

        return min(1.0, score)

    def evaluate_encouragement(self, response: str) -> float:
        """Evaluate presence and quality of encouragement."""
        response_lower = response.lower()
        score = 0.0

        # Positive words
        positive_words = [
            "great",
            "good",
            "excellent",
            "well done",
            "nice work",
            "fantastic",
            "awesome",
            "perfect",
            "brilliant",
            "wonderful",
            "you can do it",
            "keep trying",
            "good job",
            "way to go",
        ]

        positive_count = sum(1 for word in positive_words if word in response_lower)
        score += min(0.4, positive_count * 0.15)

        # Encouraging phrases
        encouraging_phrases = [
            "don't worry",
            "it's okay",
            "that's normal",
            "everyone makes mistakes",
            "you're learning",
            "keep practicing",
            "you're doing well",
            "let's work together",
            "i'm here to help",
        ]

        encouraging_count = sum(1 for phrase in encouraging_phrases if phrase in response_lower)
        score += min(0.3, encouraging_count * 0.15)

        # Growth mindset language
        growth_words = [
            "practice",
            "learn",
            "improve",
            "grow",
            "develop",
            "progress",
            "effort",
            "try again",
            "keep going",
        ]

        growth_count = sum(1 for word in growth_words if word in response_lower)
        score += min(0.2, growth_count * 0.1)

        # Exclamation marks (enthusiasm)
        exclamation_count = response.count("!")
        score += min(0.1, exclamation_count * 0.05)

        return min(1.0, score)

    def evaluate_cultural_sensitivity(self, response: str, problem: MathProblem) -> float:
        """Evaluate cultural sensitivity and inclusiveness."""
        response_lower = response.lower()
        score = 0.8  # Start with high baseline

        # Check for potentially problematic content
        problematic_terms = [
            "stupid",
            "dumb",
            "idiot",
            "easy",
            "obvious",
            "simple",
            "everyone knows",
            "of course",
            "obviously",
        ]

        for term in problematic_terms:
            if term in response_lower:
                score -= 0.2

        # Positive inclusivity indicators
        inclusive_language = [
            "different ways",
            "various methods",
            "multiple approaches",
            "some people",
            "many students",
            "different students",
        ]

        if any(phrase in response_lower for phrase in inclusive_language):
            score += 0.1

        # Cultural context awareness
        if problem.cultural_context != "general":
            # Check if response acknowledges or respects cultural context
            # This is a simplified check - in practice would be more sophisticated
            cultural_indicators = ["culture", "different", "background", "context"]
            if any(indicator in response_lower for indicator in cultural_indicators):
                score += 0.1

        return max(0.0, min(1.0, score))

    def generate_feedback(
        self,
        correctness: float,
        steps: float,
        explanation: float,
        encouragement: float,
        cultural: float,
    ) -> str:
        """Generate detailed feedback on model performance."""
        feedback_parts = []

        # Correctness feedback
        if correctness >= 0.8:
            feedback_parts.append("✓ Provides correct mathematical answer")
        elif correctness >= 0.5:
            feedback_parts.append("⚠ Answer is partially correct or unclear")
        else:
            feedback_parts.append("✗ Mathematical answer is incorrect")

        # Step-by-step feedback
        if steps >= 0.7:
            feedback_parts.append("✓ Shows clear step-by-step process")
        elif steps >= 0.4:
            feedback_parts.append("⚠ Some steps shown but could be clearer")
        else:
            feedback_parts.append("✗ Lacks clear step-by-step explanation")

        # Explanation feedback
        if explanation >= 0.7:
            feedback_parts.append("✓ Provides clear, age-appropriate explanation")
        elif explanation >= 0.4:
            feedback_parts.append("⚠ Explanation present but could be improved")
        else:
            feedback_parts.append("✗ Explanation is unclear or inappropriate")

        # Encouragement feedback
        if encouragement >= 0.6:
            feedback_parts.append("✓ Uses encouraging, supportive language")
        elif encouragement >= 0.3:
            feedback_parts.append("⚠ Some encouragement present")
        else:
            feedback_parts.append("✗ Lacks encouraging language")

        # Cultural sensitivity feedback
        if cultural >= 0.8:
            feedback_parts.append("✓ Culturally sensitive and inclusive")
        elif cultural >= 0.6:
            feedback_parts.append("⚠ Generally appropriate with minor concerns")
        else:
            feedback_parts.append("✗ Contains potentially problematic language")

        return "; ".join(feedback_parts)

    def calculate_weighted_fitness(self, category_scores: dict[str, float]) -> float:
        """Calculate overall weighted fitness score from category scores."""
        # Category weights based on importance for math tutoring
        category_weights = {
            "arithmetic": 0.25,  # Foundational skills
            "algebra": 0.25,  # Problem-solving skills
            "geometry": 0.20,  # Spatial reasoning
            "word_problems": 0.20,  # Application skills
            "fractions": 0.10,  # Specific skill area
        }

        weighted_score = 0.0
        total_weight = 0.0

        for category, score in category_scores.items():
            weight = category_weights.get(category, 0.1)  # Default weight
            weighted_score += score * weight
            total_weight += weight

        # Normalize by total weight
        final_score = weighted_score / total_weight if total_weight > 0 else 0.0

        return min(1.0, max(0.0, final_score))

    def get_performance_level(self, fitness_score: float) -> str:
        """Get performance level description."""
        if fitness_score >= self.performance_thresholds["excellent"]:
            return "excellent"
        if fitness_score >= self.performance_thresholds["good"]:
            return "good"
        if fitness_score >= self.performance_thresholds["acceptable"]:
            return "acceptable"
        return "poor"

    def get_evaluation_analytics(self) -> dict[str, Any]:
        """Get comprehensive evaluation analytics."""
        if not self.evaluation_history:
            return {"message": "No evaluations completed yet"}

        analytics = {
            "total_evaluations": len(self.evaluation_history),
            "average_scores": {},
            "performance_distribution": {},
            "category_performance": {},
            "recent_trends": {},
            "cached_models": len(self.model_performance_cache),
        }

        # Calculate average scores
        all_scores = [eval.total_score for eval in self.evaluation_history]
        analytics["average_scores"] = {
            "overall": statistics.mean(all_scores),
            "correctness": statistics.mean([eval.correctness_score for eval in self.evaluation_history]),
            "step_by_step": statistics.mean([eval.step_by_step_score for eval in self.evaluation_history]),
            "explanation": statistics.mean([eval.explanation_quality for eval in self.evaluation_history]),
            "encouragement": statistics.mean([eval.encouragement_score for eval in self.evaluation_history]),
            "cultural_sensitivity": statistics.mean([eval.cultural_sensitivity for eval in self.evaluation_history]),
        }

        # Performance distribution
        for threshold_name, threshold_value in self.performance_thresholds.items():
            count = sum(1 for score in all_scores if score >= threshold_value)
            analytics["performance_distribution"][threshold_name] = count / len(all_scores)

        return analytics


# Global fitness evaluator instance
math_fitness_evaluator = MathFitnessEvaluator()
