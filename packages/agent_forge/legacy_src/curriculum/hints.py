"""Hints System - Generate concise hints (≤25 tokens) for wrong answers.

Provides targeted, brief hints to guide students toward correct solutions
without giving away the full answer.
"""

import logging
import random
import re
from pathlib import Path
from typing import Any

from .openrouter import OpenRouterLLM
from .schemas import HintRequest, HintResponse, HintType, PeerSummary, Problem

logger = logging.getLogger(__name__)


class HintGenerator:
    """Generates concise hints for students who provided wrong answers."""

    def __init__(
        self,
        llm_client: OpenRouterLLM,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.5,
    ):
        """Initialize HintGenerator.

        Args:
            llm_client: OpenRouter client for LLM calls
            model: Model to use for hint generation (fast, cheap model)
            temperature: Medium temperature for helpful variety
        """
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature

        # Load template
        template_path = Path(__file__).parent / "templates" / "hint_generator.jinja"
        with open(template_path, encoding="utf-8") as f:
            self.template = f.read()

        # Pre-built hint patterns for common issues
        self.hint_patterns = self._load_hint_patterns()

        logger.info(f"HintGenerator initialized with model {model}")

    def _load_hint_patterns(self) -> dict[str, dict[str, list[str]]]:
        """Load pre-built hint patterns for common programming issues."""
        return {
            "empty_input": {
                "concept": [
                    "Consider what happens with empty input",
                    "Handle the empty case first",
                ],
                "boundary": ["Check your base case", "What if the input is empty?"],
                "sanity-check": [
                    "Test with empty input",
                    "Empty input should be handled",
                ],
            },
            "off_by_one": {
                "procedure": ["Check your loop bounds", "Review your indexing"],
                "boundary": ["Start from 0 or 1?", "Inclusive or exclusive range?"],
                "sanity-check": ["Count carefully", "Check first and last elements"],
            },
            "wrong_return_type": {
                "concept": [
                    "Return the right data type",
                    "Check what you're returning",
                ],
                "units": ["Return string not print", "Return number not string"],
                "sanity-check": ["Print vs return?", "What type is expected?"],
            },
            "missing_edge_case": {
                "boundary": ["Consider edge cases", "What about minimum/maximum?"],
                "concept": ["Handle special cases", "Think about boundary conditions"],
                "procedure": [
                    "Add conditions for special cases",
                    "Check extreme values",
                ],
            },
            "algorithm_wrong": {
                "concept": ["Rethink your approach", "Try a different method"],
                "procedure": ["Break down the steps", "What's the key operation?"],
                "sanity-check": ["Does this solve the problem?", "Check your logic"],
            },
            "loop_issue": {
                "procedure": ["Check your loop logic", "What are you iterating over?"],
                "concept": ["Do you need a loop?", "Right loop type?"],
                "sanity-check": ["Loop runs correct times?", "Check loop variable"],
            },
        }

    def _classify_error_type(self, problem: Problem, wrong_answer: str) -> str:
        """Classify the type of error based on problem and wrong answer."""

        wrong_lower = wrong_answer.lower()
        problem_lower = problem.statement.lower()

        # Check for common patterns
        if "print" in wrong_lower and "return" in problem_lower:
            return "wrong_return_type"

        if ("empty" in problem_lower or "none" in problem_lower) and "if" not in wrong_lower:
            return "empty_input"

        if any(word in problem_lower for word in ["maximum", "minimum", "first", "last"]) and "range" in wrong_lower:
            return "off_by_one"

        if ("for" in wrong_lower or "while" in wrong_lower) and len(wrong_answer) < 50:
            return "loop_issue"

        if len(wrong_answer) > 200 and "def" in wrong_lower:
            return "algorithm_wrong"

        # Default fallback
        return "missing_edge_case"

    def _generate_local_hint(
        self, problem: Problem, wrong_answer: str, hint_type: HintType | None = None
    ) -> HintResponse | None:
        """Generate a hint using local patterns without LLM."""

        try:
            # Classify error
            error_type = self._classify_error_type(problem, wrong_answer)

            # Choose hint type if not specified
            if hint_type is None:
                available_types = list(self.hint_patterns[error_type].keys())
                hint_type_str = random.choice(available_types)
                hint_type = HintType(hint_type_str.replace("-", "_"))
            else:
                hint_type_str = hint_type.value.replace("_", "-")

            # Get hint patterns for this error and type
            if error_type not in self.hint_patterns:
                return None

            patterns = self.hint_patterns[error_type].get(hint_type_str, [])
            if not patterns:
                return None

            # Select and customize hint
            base_hint = random.choice(patterns)

            # Simple customization based on problem
            customized_hint = self._customize_hint(base_hint, problem)

            # Validate token count (rough estimate: 1 token ≈ 4 characters)
            if len(customized_hint) > 100:  # ~25 tokens
                customized_hint = customized_hint[:97] + "..."

            return HintResponse(ok=True, msg="hint", hint=customized_hint, hint_type=hint_type)

        except Exception as e:
            logger.warning(f"Local hint generation failed: {e}")
            return None

    def _customize_hint(self, base_hint: str, problem: Problem) -> str:
        """Customize a base hint template for the specific problem."""

        # Extract key terms from problem
        statement_lower = problem.statement.lower()

        # Simple term substitutions
        if "list" in statement_lower and "input" in base_hint:
            base_hint = base_hint.replace("input", "list")
        elif "string" in statement_lower and "input" in base_hint:
            base_hint = base_hint.replace("input", "string")

        # Add specific context
        if "maximum" in statement_lower and "case" in base_hint:
            base_hint = base_hint.replace("case", "maximum case")
        elif "empty" in statement_lower and "case" in base_hint:
            base_hint = base_hint.replace("case", "empty case")

        return base_hint

    def _validate_hint_quality(self, hint: str) -> bool:
        """Validate hint quality and token count."""

        # Check token count (rough estimate)
        estimated_tokens = len(hint.split())
        if estimated_tokens > 25:
            return False

        # Check for solution giveaways
        giveaway_patterns = [
            r"def \w+",
            r"return \w+\(",
            r"for \w+ in",
            r"while \w+",
            r"if \w+ ==",
            r"\.append\(",
            r"\.join\(",
        ]

        for pattern in giveaway_patterns:
            if re.search(pattern, hint, re.IGNORECASE):
                return False

        # Check minimum helpfulness
        if len(hint.strip()) < 10:
            return False

        # Check for encouraging language
        helpful_indicators = [
            "consider",
            "check",
            "think",
            "try",
            "what",
            "how",
            "remember",
            "handle",
            "review",
            "test",
        ]

        hint_lower = hint.lower()
        if not any(indicator in hint_lower for indicator in helpful_indicators):
            return False

        return True

    async def generate_hint(
        self,
        problem: Problem,
        wrong_answer: str,
        peer_summaries: list[PeerSummary] | None = None,
        preferred_hint_type: HintType | None = None,
        use_local_fallback: bool = True,
    ) -> HintResponse:
        """Generate a helpful hint for a wrong answer.

        Args:
            problem: Problem the student was solving
            wrong_answer: Student's incorrect solution
            peer_summaries: Analysis from other models (optional)
            preferred_hint_type: Preferred type of hint
            use_local_fallback: Use pattern-based hints if LLM fails

        Returns:
            HintResponse with concise hint (≤25 tokens)

        Raises:
            ValueError: If inputs are invalid
        """
        if not problem.statement:
            raise ValueError("Problem statement cannot be empty")

        if not wrong_answer.strip():
            raise ValueError("Wrong answer cannot be empty")

        logger.info(f"Generating hint for problem {problem.id}")

        # Try LLM generation first
        try:
            request = HintRequest(
                problem=problem,
                wrong_answer=wrong_answer,
                peer_summaries=peer_summaries or [],
            )

            # Render prompt
            prompt = self.llm_client.render_template(
                self.template,
                problem=request.problem,
                wrong_answer=request.wrong_answer,
                peer_summaries=request.peer_summaries,
            )

            # Add hint type preference if specified
            if preferred_hint_type:
                prompt += f"\n\nPrefer generating a {preferred_hint_type.value} type hint."

            response = await self.llm_client.invoke_with_schema(
                prompt=prompt,
                schema_class=HintResponse,
                model=self.model,
                temperature=self.temperature,
                max_tokens=512,
                max_schema_retries=2,
            )

            # Validate hint quality
            if self._validate_hint_quality(response.hint):
                logger.info(f"LLM hint generated: {response.hint}")
                return response
            else:
                logger.warning("LLM hint failed quality validation")

        except Exception as e:
            logger.error(f"LLM hint generation failed: {e}")

        # Fallback to local patterns
        if use_local_fallback:
            logger.info("Falling back to pattern-based hint generation")
            local_hint = self._generate_local_hint(problem, wrong_answer, preferred_hint_type)

            if local_hint:
                logger.info(f"Local hint generated: {local_hint.hint}")
                return local_hint

        # Ultimate fallback
        return HintResponse(
            ok=True,
            msg="fallback hint",
            hint="Check your logic and test with simple examples",
            hint_type=HintType.SANITY_CHECK,
        )

    async def generate_hint_batch(
        self, problems: list[Problem], wrong_answers: list[str], **kwargs
    ) -> list[HintResponse]:
        """Generate hints for multiple wrong answers efficiently.

        Args:
            problems: List of problems
            wrong_answers: Corresponding wrong answers
            **kwargs: Additional hint generation options

        Returns:
            List of HintResponse objects
        """
        if len(problems) != len(wrong_answers):
            raise ValueError("Number of problems must match number of wrong answers")

        results = []

        for problem, answer in zip(problems, wrong_answers, strict=False):
            try:
                hint = await self.generate_hint(problem, answer, **kwargs)
                results.append(hint)
            except Exception as e:
                logger.error(f"Failed to generate hint for {problem.id}: {e}")
                # Add fallback hint
                results.append(
                    HintResponse(
                        ok=False,
                        msg=f"hint_error: {e}",
                        hint="Review the problem requirements carefully",
                        hint_type=HintType.CONCEPT,
                    )
                )

        return results

    def analyze_hint_effectiveness(self, hints: list[HintResponse], follow_up_results: list[bool]) -> dict[str, Any]:
        """Analyze effectiveness of hints based on follow-up success.

        Args:
            hints: Generated hints
            follow_up_results: Whether students succeeded after hint

        Returns:
            Dictionary with effectiveness analysis
        """
        if len(hints) != len(follow_up_results):
            raise ValueError("Hints and results must have same length")

        if not hints:
            return {"error": "No data provided"}

        # Overall effectiveness
        successful = sum(follow_up_results)
        total = len(follow_up_results)
        overall_rate = successful / total if total > 0 else 0

        # By hint type
        type_stats = {}
        for hint, success in zip(hints, follow_up_results, strict=False):
            hint_type = hint.hint_type.value
            if hint_type not in type_stats:
                type_stats[hint_type] = {"total": 0, "successful": 0}

            type_stats[hint_type]["total"] += 1
            if success:
                type_stats[hint_type]["successful"] += 1

        # Calculate rates by type
        for hint_type, stats in type_stats.items():
            stats["success_rate"] = stats["successful"] / stats["total"]

        # Hint length analysis
        hint_lengths = [len(hint.hint.split()) for hint in hints]
        avg_length = sum(hint_lengths) / len(hint_lengths) if hint_lengths else 0

        return {
            "overall_success_rate": overall_rate,
            "total_hints": total,
            "successful_hints": successful,
            "by_type": type_stats,
            "average_hint_length_words": avg_length,
            "length_range": [min(hint_lengths), max(hint_lengths)] if hint_lengths else [0, 0],
        }

    def get_hint_templates_by_type(self, hint_type: HintType) -> list[str]:
        """Get available hint templates for a specific type."""

        hint_type_str = hint_type.value.replace("_", "-")
        templates = []

        for error_type, type_patterns in self.hint_patterns.items():
            if hint_type_str in type_patterns:
                templates.extend(type_patterns[hint_type_str])

        return templates


async def generate_coding_hint(
    api_key: str,
    problem: Problem,
    wrong_answer: str,
    model: str = "openai/gpt-4o-mini",
    **kwargs,
) -> HintResponse:
    """Convenience function to generate a hint with minimal setup.

    Args:
        api_key: OpenRouter API key
        problem: Problem student was solving
        wrong_answer: Student's incorrect solution
        model: Model to use for hint generation
        **kwargs: Additional arguments for HintGenerator

    Returns:
        HintResponse with helpful hint
    """
    async with OpenRouterLLM(api_key=api_key) as client:
        generator = HintGenerator(client, model=model)
        return await generator.generate_hint(problem, wrong_answer, **kwargs)


if __name__ == "__main__":
    # Demo usage
    import asyncio
    import os

    async def demo():
        # Create test problem and wrong answers
        problem = Problem(
            id="demo_hint",
            topic="list_operations",
            difficulty=0.6,
            statement="Write a function that returns the maximum value in a list of integers. Handle empty lists by returning None.",
            canonical_answer="def find_max(lst):\n    return max(lst) if lst else None",
            rubric="Function finds maximum correctly, handles empty list by returning None",
            unit_tests=[
                "assert find_max([1, 3, 2]) == 3",
                "assert find_max([]) is None",
            ],
        )

        wrong_answers = [
            # Missing empty case handling
            "def find_max(lst):\n    return max(lst)",
            # Using print instead of return
            "def find_max(lst):\n    print(max(lst))",
            # Wrong algorithm
            "def find_max(lst):\n    return sum(lst) / len(lst)",
            # Off by one error
            "def find_max(lst):\n    return lst[len(lst)]",
        ]

        api_key = os.getenv("OPENROUTER_API_KEY", "demo-key")

        if api_key == "demo-key":
            print("🔧 Demo mode: Testing local hint patterns")

            dummy_client = OpenRouterLLM(api_key="dummy")
            generator = HintGenerator(dummy_client)

            for i, answer in enumerate(wrong_answers):
                print(f"\n❌ Wrong Answer {i + 1}: {answer[:40]}...")

                local_hint = generator._generate_local_hint(problem, answer)
                if local_hint:
                    print(f"   💡 Hint: {local_hint.hint}")
                    print(f"   🏷️  Type: {local_hint.hint_type.value}")
                    print(f"   📏 Length: {len(local_hint.hint.split())} words")
                else:
                    print("   ❌ No local hint generated")

            return

        # Live API test
        print("💡 Testing live hint generation...")

        try:
            result = await generate_coding_hint(
                api_key=api_key,
                problem=problem,
                wrong_answer=wrong_answers[0],  # Test first wrong answer
            )

            print(f"\n❌ Wrong Answer: {wrong_answers[0][:50]}...")
            print(f"💡 Generated Hint: {result.hint}")
            print(f"🏷️  Hint Type: {result.hint_type.value}")
            print(f"📏 Length: {len(result.hint.split())} words")
            print(f"✅ Quality: {'Pass' if len(result.hint.split()) <= 25 else 'Fail (too long)'}")

        except Exception as e:
            print(f"❌ Demo failed: {e}")

    asyncio.run(demo())
