"""Variant Maker - Generate cosmetic variants of problems preserving underlying skill.

Creates variants that maintain the same algorithmic challenge while changing
surface details like context, variable names, and numeric values.
"""

import logging
import random
import re
import uuid
from pathlib import Path
from typing import Any

from .openrouter import OpenRouterLLM
from .schemas import NumericJitterPolicy, Problem, ProblemVariant, VariantPolicy, VariantRequest, VariantResponse

logger = logging.getLogger(__name__)


class VariantMaker:
    """Creates cosmetic variants of coding problems."""

    def __init__(
        self,
        llm_client: OpenRouterLLM,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.8,
    ):
        """Initialize VariantMaker.

        Args:
            llm_client: OpenRouter client for LLM calls
            model: Model to use for variant generation (cheaper model recommended)
            temperature: Sampling temperature (higher for variety)
        """
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature

        # Load template
        template_path = Path(__file__).parent / "templates" / "variant_synthesizer.jinja"
        with open(template_path, encoding="utf-8") as f:
            self.template = f.read()

        # Variant transformation patterns
        self.synonym_patterns = self._load_synonym_patterns()
        self.context_patterns = self._load_context_patterns()

        logger.info(f"VariantMaker initialized with model {model}")

    def _load_synonym_patterns(self) -> dict[str, list[str]]:
        """Load synonym patterns for common programming terms."""
        return {
            "find": ["locate", "identify", "search for", "discover"],
            "remove": ["eliminate", "delete", "filter out", "extract"],
            "sort": ["arrange", "order", "organize", "rank"],
            "count": ["calculate", "tally", "enumerate", "sum"],
            "maximum": ["peak", "highest", "largest", "top"],
            "minimum": ["lowest", "smallest", "bottom", "least"],
            "sum": ["total", "aggregate", "accumulate", "combine"],
            "average": ["mean", "typical", "standard", "median"],
            "unique": ["distinct", "different", "non-duplicate", "separate"],
            "reverse": ["flip", "invert", "backward", "opposite"],
            "string": ["text", "word", "phrase", "sequence"],
            "list": ["array", "sequence", "collection", "series"],
            "element": ["item", "entry", "value", "component"],
            "index": ["position", "location", "offset", "place"],
            "length": ["size", "count", "number", "total"],
            "empty": ["blank", "vacant", "null", "void"],
            "first": ["initial", "starting", "beginning", "opening"],
            "last": ["final", "ending", "closing", "terminal"],
        }

    def _load_context_patterns(self) -> list[dict[str, Any]]:
        """Load context transformation patterns."""
        return [
            {
                "from_context": "students and grades",
                "to_context": "employees and salaries",
                "transforms": {
                    "student": "employee",
                    "grade": "salary",
                    "class": "department",
                    "school": "company",
                },
            },
            {
                "from_context": "products and prices",
                "to_context": "books and ratings",
                "transforms": {
                    "product": "book",
                    "price": "rating",
                    "store": "library",
                    "customer": "reader",
                },
            },
            {
                "from_context": "tasks and times",
                "to_context": "games and scores",
                "transforms": {
                    "task": "game",
                    "time": "score",
                    "worker": "player",
                    "project": "tournament",
                },
            },
        ]

    def _apply_numeric_jitter(self, text: str, jitter_policy: NumericJitterPolicy) -> str:
        """Apply numeric jittering to text according to policy."""

        if not jitter_policy.enabled:
            return text

        def jitter_number(match):
            original = int(match.group())
            if original == 0:  # Don't jitter zero
                return str(original)

            # Calculate jitter amount
            jitter_amount = abs(original) * (jitter_policy.pct / 100.0)
            min_jitter = max(1, int(jitter_amount * 0.5))  # At least 1, up to half the jitter
            max_jitter = max(1, int(jitter_amount * 1.5))  # Up to 1.5x the jitter

            # Apply random jitter
            change = random.randint(-max_jitter, max_jitter)
            if change == 0:  # Ensure some change
                change = random.choice([-min_jitter, min_jitter])

            new_value = original + change

            # Keep positive for most contexts
            if original > 0 and new_value <= 0:
                new_value = 1

            return str(new_value)

        # Apply to standalone numbers (not part of words)
        jittered = re.sub(r"\b\d+\b", jitter_number, text)
        return jittered

    def _apply_local_transformations(
        self, base_problem: Problem, variant_policy: VariantPolicy
    ) -> ProblemVariant | None:
        """Apply local transformations without LLM for simple variants."""

        try:
            # Start with base problem
            statement = base_problem.statement
            canonical_answer = base_problem.canonical_answer
            rubric = base_problem.rubric
            unit_tests = base_problem.unit_tests[:]

            # Apply synonym substitutions if paraphrasing enabled
            if variant_policy.paraphrase:
                for original, synonyms in self.synonym_patterns.items():
                    if original.lower() in statement.lower():
                        synonym = random.choice(synonyms)
                        # Case-sensitive replacement
                        statement = re.sub(
                            r"\b" + re.escape(original) + r"\b",
                            synonym,
                            statement,
                            flags=re.IGNORECASE,
                        )
                        # Also update rubric
                        rubric = re.sub(
                            r"\b" + re.escape(original) + r"\b",
                            synonym,
                            rubric,
                            flags=re.IGNORECASE,
                        )

            # Apply numeric jittering
            statement = self._apply_numeric_jitter(statement, variant_policy.numeric_jitter)
            unit_tests = [self._apply_numeric_jitter(test, variant_policy.numeric_jitter) for test in unit_tests]

            # Simple variable name changes in canonical answer
            if variant_policy.paraphrase:
                canonical_answer = self._transform_variable_names(canonical_answer)

            # Generate new ID
            variant_id = f"{base_problem.id}_v{random.randint(1000, 9999)}"

            # Create variant
            variant = ProblemVariant(
                id=variant_id,
                statement=statement,
                canonical_answer=canonical_answer,
                rubric=rubric,
                unit_tests=unit_tests,
            )

            logger.debug(f"Created local variant: {variant_id}")
            return variant

        except Exception as e:
            logger.warning(f"Local transformation failed: {e}")
            return None

    def _transform_variable_names(self, code: str) -> str:
        """Apply simple variable name transformations."""

        # Common variable name mappings
        var_mappings = {
            r"\bn\b": random.choice(["num", "count", "size"]),
            r"\bi\b": random.choice(["idx", "pos", "counter"]),
            r"\bj\b": random.choice(["jdx", "inner", "second"]),
            r"\bx\b": random.choice(["val", "item", "elem"]),
            r"\by\b": random.choice(["other", "second", "next"]),
            r"\bstr\b": random.choice(["text", "word", "string"]),
            r"\blst\b": random.choice(["items", "values", "data"]),
            r"\barr\b": random.choice(["sequence", "collection", "series"]),
        }

        transformed = code
        for pattern, replacement in var_mappings.items():
            if random.random() < 0.3:  # 30% chance to apply each transformation
                transformed = re.sub(pattern, replacement, transformed)

        return transformed

    async def create_variants(
        self,
        base_problem: Problem,
        variant_policy: VariantPolicy,
        n_variants: int = 3,
        use_local_fallback: bool = True,
    ) -> VariantResponse:
        """Create variants of a base problem.

        Args:
            base_problem: Base problem to create variants from
            variant_policy: Variant generation policy
            n_variants: Number of variants to generate
            use_local_fallback: Use local transformations if LLM fails

        Returns:
            VariantResponse with generated variants

        Raises:
            ValueError: If parameters are invalid
        """
        if n_variants <= 0:
            raise ValueError("Number of variants must be positive")

        if n_variants > 10:
            raise ValueError("Too many variants requested (max 10)")

        logger.info(f"Creating {n_variants} variants for problem {base_problem.id}")

        variants = []
        llm_failures = 0

        # Try LLM generation first
        if n_variants > 1:  # Use LLM for multiple variants
            try:
                request = VariantRequest(base_problem=base_problem, variant_policy=variant_policy)

                # Render prompt
                prompt = self.llm_client.render_template(
                    self.template,
                    base_problem=request.base_problem,
                    variant_policy=request.variant_policy,
                )

                # Add explicit instruction for number of variants
                if n_variants > 1:
                    prompt += f"\n\nGenerate exactly {n_variants} distinct variants in the response."

                response = await self.llm_client.invoke_with_schema(
                    prompt=prompt,
                    schema_class=VariantResponse,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=2048,
                    max_schema_retries=2,
                )

                # Post-process variants
                for variant in response.variants:
                    processed = self._post_process_variant(variant, base_problem)
                    if processed:
                        variants.append(processed)

                logger.info(f"LLM generated {len(variants)} variants")

            except Exception as e:
                logger.error(f"LLM variant generation failed: {e}")
                llm_failures = n_variants

        # Fill remaining with local transformations
        remaining_needed = n_variants - len(variants)

        if remaining_needed > 0 and use_local_fallback:
            logger.info(f"Generating {remaining_needed} variants using local transformations")

            for _ in range(remaining_needed):
                local_variant = self._apply_local_transformations(base_problem, variant_policy)
                if local_variant:
                    processed = self._post_process_variant(local_variant, base_problem)
                    if processed:
                        variants.append(processed)

        if not variants:
            raise RuntimeError("Failed to generate any variants")

        # Create response
        final_response = VariantResponse(ok=True, msg=f"variants ({len(variants)} generated)", variants=variants)

        logger.info(f"Successfully created {len(variants)} variants")
        return final_response

    def _post_process_variant(self, variant: ProblemVariant, base_problem: Problem) -> ProblemVariant | None:
        """Post-process and validate a generated variant."""

        try:
            # Ensure unique ID
            if not variant.id or variant.id == base_problem.id:
                variant.id = f"{base_problem.id}_v{uuid.uuid4().hex[:6]}"

            # Validate essential fields
            if not all([variant.statement, variant.canonical_answer, variant.rubric]):
                logger.warning(f"Variant {variant.id} missing essential fields")
                return None

            # Check for reasonable changes from base
            statement_similarity = self._calculate_similarity(base_problem.statement, variant.statement)

            # Should be similar but not identical
            if statement_similarity > 0.95:
                logger.debug(f"Variant {variant.id} too similar to base")
            elif statement_similarity < 0.3:
                logger.debug(f"Variant {variant.id} very different from base")

            # Ensure unit tests are reasonable
            if not variant.unit_tests:
                logger.debug(f"Variant {variant.id} has no unit tests")

            # Basic length validation
            if len(variant.statement) < 20 or len(variant.statement) > 5000:
                logger.warning(f"Variant {variant.id} has unusual statement length")
                return None

            return variant

        except Exception as e:
            logger.error(f"Failed to post-process variant: {e}")
            return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two texts (0-1)."""

        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    async def create_variant_batch(
        self,
        problems: list[Problem],
        variant_policy: VariantPolicy,
        variants_per_problem: int = 2,
    ) -> dict[str, VariantResponse]:
        """Create variants for multiple problems efficiently.

        Args:
            problems: List of base problems
            variant_policy: Variant generation policy
            variants_per_problem: Number of variants per problem

        Returns:
            Dictionary mapping problem IDs to their variant responses
        """
        results = {}

        for problem in problems:
            try:
                logger.info(f"Creating variants for problem {problem.id}")
                response = await self.create_variants(problem, variant_policy, variants_per_problem)
                results[problem.id] = response

            except Exception as e:
                logger.error(f"Failed to create variants for {problem.id}: {e}")
                # Create empty response for consistency
                results[problem.id] = VariantResponse(ok=False, msg=f"failed: {e}", variants=[])

        logger.info(f"Completed variant generation for {len(problems)} problems")
        return results

    def assess_variant_quality(self, base_problem: Problem, variants: list[ProblemVariant]) -> dict[str, float]:
        """Assess quality of generated variants (0-1 scores)."""

        quality_scores = {}

        for variant in variants:
            score = 0.0

            # Diversity check (should be different from base)
            similarity = self._calculate_similarity(base_problem.statement, variant.statement)
            if 0.3 <= similarity <= 0.8:  # Sweet spot
                score += 0.3
            elif similarity < 0.9:  # At least some difference
                score += 0.1

            # Completeness check
            if variant.statement and variant.canonical_answer and variant.rubric:
                score += 0.3

            # Unit test preservation
            if len(variant.unit_tests) >= len(base_problem.unit_tests) * 0.5:
                score += 0.2

            # Reasonable length
            if 50 <= len(variant.statement) <= 2000:
                score += 0.1

            # Code quality (basic check)
            if "def " in variant.canonical_answer:
                score += 0.1

            quality_scores[variant.id] = min(1.0, score)

        return quality_scores


async def create_problem_variants(
    api_key: str,
    base_problem: Problem,
    variant_policy: VariantPolicy,
    n_variants: int = 3,
    model: str = "openai/gpt-4o-mini",
    **kwargs,
) -> VariantResponse:
    """Convenience function to create variants with minimal setup.

    Args:
        api_key: OpenRouter API key
        base_problem: Base problem to create variants from
        variant_policy: Variant generation policy
        n_variants: Number of variants to generate
        model: Model to use for variant generation
        **kwargs: Additional arguments for VariantMaker

    Returns:
        VariantResponse with generated variants
    """
    async with OpenRouterLLM(api_key=api_key) as client:
        maker = VariantMaker(client, model=model)
        return await maker.create_variants(base_problem, variant_policy, n_variants, **kwargs)


if __name__ == "__main__":
    # Demo usage
    import asyncio
    import os

    async def demo():
        api_key = os.getenv("OPENROUTER_API_KEY", "demo-key")

        if api_key == "demo-key":
            print("🔧 Demo mode: Set OPENROUTER_API_KEY for live testing")

            # Demo local transformations
            base_problem = Problem(
                id="demo_001",
                topic="string_manipulation",
                difficulty=0.6,
                statement="Write a function that takes a string and returns it with all vowels removed.",
                canonical_answer="def remove_vowels(s):\n    return ''.join(c for c in s if c not in 'aeiouAEIOU')",
                rubric="Function removes all vowels correctly",
                unit_tests=["assert remove_vowels('hello') == 'hll'"],
            )

            policy = VariantPolicy(
                paraphrase=True,
                numeric_jitter=NumericJitterPolicy(enabled=True, pct=10),
            )

            # Test local transformations
            dummy_client = OpenRouterLLM(api_key="dummy")
            maker = VariantMaker(dummy_client)
            variant = maker._apply_local_transformations(base_problem, policy)

            if variant:
                print("✅ Local variant generation works")
                print(f"Original: {base_problem.statement[:60]}...")
                print(f"Variant:  {variant.statement[:60]}...")

            return

        # Live API test
        base_problem = Problem(
            id="test_001",
            topic="list_operations",
            difficulty=0.65,
            statement="Write a function that finds the maximum value in a list of integers.",
            canonical_answer="def find_max(lst):\n    return max(lst) if lst else None",
            rubric="Function correctly finds maximum value, handles empty list",
            unit_tests=[
                "assert find_max([1, 3, 2]) == 3",
                "assert find_max([]) is None",
            ],
        )

        policy = VariantPolicy(paraphrase=True, numeric_jitter=NumericJitterPolicy(enabled=True, pct=15))

        try:
            result = await create_problem_variants(
                api_key=api_key,
                base_problem=base_problem,
                variant_policy=policy,
                n_variants=2,
            )

            print(f"✅ Generated {len(result.variants)} variants")
            for variant in result.variants:
                print(f"\n🔄 {variant.id}")
                print(f"   Statement: {variant.statement[:80]}...")
                print(f"   Tests: {len(variant.unit_tests)} unit tests")

        except Exception as e:
            print(f"❌ Demo failed: {e}")

    asyncio.run(demo())
