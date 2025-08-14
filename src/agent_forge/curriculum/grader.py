"""Grader - Auto-grade final answers without evaluating chain-of-thought.

Evaluates code solutions for correctness against problem requirements,
focusing only on final implementation quality and test compliance.
"""

import ast
import contextlib
import logging
from io import StringIO
from pathlib import Path
from typing import Any

from .openrouter import OpenRouterLLM
from .schemas import GradingRequest, GradingResponse, Problem

logger = logging.getLogger(__name__)


class CodeExecutionError(Exception):
    """Raised when code execution fails during testing."""

    pass


class Grader:
    """Grades coding problem solutions focusing on final answer correctness."""

    def __init__(
        self,
        llm_client: OpenRouterLLM,
        model: str = "anthropic/claude-3-5-haiku-20241022",
        temperature: float = 0.1,
        enable_code_execution: bool = False,
    ):
        """Initialize Grader.

        Args:
            llm_client: OpenRouter client for LLM calls
            model: Model to use for grading (fast, cheap model recommended)
            temperature: Low temperature for consistent grading
            enable_code_execution: Whether to execute code for testing (security risk)
        """
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.enable_code_execution = enable_code_execution

        # Load template
        template_path = Path(__file__).parent / "templates" / "grader.jinja"
        with open(template_path, encoding="utf-8") as f:
            self.template = f.read()

        # Error classification patterns
        self.error_patterns = self._load_error_patterns()

        logger.info(
            f"Grader initialized with model {model}, execution: {enable_code_execution}"
        )

    def _load_error_patterns(self) -> dict[str, list[str]]:
        """Load patterns for classifying common errors."""
        return {
            "syntax_error": [
                "SyntaxError",
                "IndentationError",
                "invalid syntax",
                "unexpected indent",
                "unmatched",
                "parentheses",
                "brackets",
            ],
            "logic_error": [
                "wrong result",
                "incorrect output",
                "algorithm",
                "logic",
                "wrong approach",
                "incorrect method",
                "bad implementation",
            ],
            "edge_case": [
                "empty",
                "null",
                "None",
                "zero",
                "boundary",
                "edge case",
                "single element",
                "minimum",
                "maximum",
                "corner case",
            ],
            "type_error": [
                "TypeError",
                "AttributeError",
                "wrong type",
                "type mismatch",
                "expected",
                "but got",
                "cannot",
                "unsupported operand",
            ],
            "incomplete": [
                "not implemented",
                "pass",
                "TODO",
                "incomplete",
                "missing",
                "not defined",
                "partial",
            ],
            "off_topic": [
                "different problem",
                "wrong task",
                "unrelated",
                "not addressing",
                "different approach",
            ],
        }

    def _classify_error_tags(self, error_description: str) -> list[str]:
        """Classify errors based on description patterns."""
        tags = []
        error_lower = error_description.lower()

        for tag, patterns in self.error_patterns.items():
            if any(pattern.lower() in error_lower for pattern in patterns):
                tags.append(tag)

        return tags[:3]  # Limit to top 3 most relevant tags

    def _extract_function_from_code(self, code: str) -> str | None:
        """Extract the main function from code for testing."""
        try:
            # Parse the code to find function definitions
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Return the first function found
                    return node.name

            return None

        except Exception:
            return None

    def _execute_code_safely(
        self, code: str, test_cases: list[str]
    ) -> tuple[bool, str]:
        """Safely execute code with test cases (if enabled)."""

        if not self.enable_code_execution:
            return False, "Code execution disabled for security"

        try:
            # Create a restricted execution environment
            namespace = {
                "__builtins__": {
                    # Allow only safe built-ins
                    "len": len,
                    "max": max,
                    "min": min,
                    "sum": sum,
                    "abs": abs,
                    "round": round,
                    "range": range,
                    "enumerate": enumerate,
                    "zip": zip,
                    "sorted": sorted,
                    "reversed": reversed,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "list": list,
                    "dict": dict,
                    "set": set,
                    "tuple": tuple,
                }
            }

            # Execute the code
            exec(code, namespace)

            # Run test cases
            failed_tests = []
            passed_tests = 0

            for test in test_cases:
                try:
                    # Capture output
                    with (
                        contextlib.redirect_stdout(StringIO()),
                        contextlib.redirect_stderr(StringIO()),
                    ):
                        result = eval(test, namespace)

                    if result:
                        passed_tests += 1
                    else:
                        failed_tests.append(f"Failed: {test}")

                except Exception as e:
                    failed_tests.append(f"Error in {test}: {str(e)}")

            success = len(failed_tests) == 0
            details = f"Passed: {passed_tests}/{len(test_cases)}"
            if failed_tests:
                details += f", Failed: {failed_tests[:3]}"  # Show first 3 failures

            return success, details

        except Exception as e:
            return False, f"Execution error: {str(e)}"

    def _perform_static_analysis(self, code: str, problem: Problem) -> dict[str, Any]:
        """Perform static analysis of the code without execution."""

        analysis = {
            "has_function": False,
            "function_name": None,
            "has_return": False,
            "has_loops": False,
            "has_conditionals": False,
            "complexity_score": 0.0,
            "syntax_valid": False,
            "imports": [],
            "estimated_correctness": 0.0,
        }

        try:
            # Parse AST
            tree = ast.parse(code)
            analysis["syntax_valid"] = True

            # Analyze structure
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis["has_function"] = True
                    if not analysis["function_name"]:  # Get first function
                        analysis["function_name"] = node.name

                elif isinstance(node, ast.Return):
                    analysis["has_return"] = True

                elif isinstance(node, (ast.For, ast.While)):
                    analysis["has_loops"] = True
                    analysis["complexity_score"] += 0.2

                elif isinstance(node, (ast.If, ast.IfExp)):
                    analysis["has_conditionals"] = True
                    analysis["complexity_score"] += 0.1

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    analysis["imports"].append(ast.dump(node))

            # Estimate correctness based on structure
            correctness = 0.0

            if analysis["has_function"]:
                correctness += 0.3
            if analysis["has_return"]:
                correctness += 0.2

            # Check if problem keywords appear in code
            problem_keywords = self._extract_problem_keywords(problem.statement)
            code_lower = code.lower()
            keyword_matches = sum(
                1 for kw in problem_keywords if kw.lower() in code_lower
            )
            if keyword_matches > 0:
                correctness += 0.3 * (keyword_matches / len(problem_keywords))

            # Reasonable length suggests thought was put in
            if 20 <= len(code) <= 1000:
                correctness += 0.2

            analysis["estimated_correctness"] = min(1.0, correctness)

        except SyntaxError as e:
            analysis["syntax_error"] = str(e)
        except Exception as e:
            analysis["parse_error"] = str(e)

        return analysis

    def _extract_problem_keywords(self, statement: str) -> list[str]:
        """Extract key algorithmic terms from problem statement."""

        keywords = []

        # Common algorithmic terms
        algorithmic_terms = [
            "sort",
            "search",
            "find",
            "maximum",
            "minimum",
            "sum",
            "count",
            "reverse",
            "remove",
            "filter",
            "unique",
            "duplicate",
            "average",
            "length",
            "index",
            "element",
            "list",
            "string",
            "array",
        ]

        statement_lower = statement.lower()
        for term in algorithmic_terms:
            if term in statement_lower:
                keywords.append(term)

        return keywords

    async def grade_solution(
        self,
        problem: Problem,
        model_answer: str,
        use_static_analysis: bool = True,
        use_llm_grading: bool = True,
    ) -> GradingResponse:
        """Grade a model's solution to a coding problem.

        Args:
            problem: Problem with statement, canonical answer, tests
            model_answer: Student's solution code
            use_static_analysis: Whether to perform static code analysis
            use_llm_grading: Whether to use LLM for grading

        Returns:
            GradingResponse with correctness assessment and error analysis

        Raises:
            ValueError: If inputs are invalid
        """
        if not problem.statement:
            raise ValueError("Problem statement cannot be empty")

        if not model_answer.strip():
            return GradingResponse(
                ok=True,
                msg="empty answer",
                correct=False,
                error_tags=["incomplete"],
                normalizer_notes="Model provided empty or whitespace-only answer",
            )

        logger.info(f"Grading solution for problem: {problem.id}")

        # Perform static analysis first
        static_analysis = None
        if use_static_analysis:
            static_analysis = self._perform_static_analysis(model_answer, problem)
            logger.debug(
                f"Static analysis: {static_analysis.get('estimated_correctness', 0):.2f} correctness"
            )

        # Try code execution if enabled
        execution_result = None
        if self.enable_code_execution and problem.unit_tests:
            try:
                success, details = self._execute_code_safely(
                    model_answer, problem.unit_tests
                )
                execution_result = {"success": success, "details": details}
                logger.debug(f"Execution result: {success}")
            except Exception as e:
                logger.warning(f"Code execution failed: {e}")

        # Use LLM grading if enabled
        llm_response = None
        if use_llm_grading:
            try:
                request = GradingRequest(problem=problem, model_answer=model_answer)

                # Render prompt
                prompt = self.llm_client.render_template(
                    self.template,
                    problem=request.problem,
                    model_answer=request.model_answer,
                )

                llm_response = await self.llm_client.invoke_with_schema(
                    prompt=prompt,
                    schema_class=GradingResponse,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=1024,
                    max_schema_retries=2,
                )

                logger.debug(f"LLM grading: {llm_response.correct}")

            except Exception as e:
                logger.error(f"LLM grading failed: {e}")

        # Synthesize final result
        final_response = self._synthesize_grading_result(
            static_analysis, execution_result, llm_response, model_answer
        )

        logger.info(
            f"Final grade: {'CORRECT' if final_response.correct else 'INCORRECT'}"
        )
        return final_response

    def _synthesize_grading_result(
        self,
        static_analysis: dict[str, Any] | None,
        execution_result: dict[str, Any] | None,
        llm_response: GradingResponse | None,
        model_answer: str,
    ) -> GradingResponse:
        """Synthesize final grading result from all available signals."""

        # Start with default failure
        correct = False
        error_tags = []
        notes = []

        # Weight different signals
        confidence_scores = []

        # Execution result (highest confidence if available)
        if execution_result:
            if execution_result["success"]:
                correct = True
                confidence_scores.append(0.9)  # Very high confidence
            else:
                confidence_scores.append(0.8)
                error_tags.extend(["logic_error"])
                notes.append(f"Execution: {execution_result['details']}")

        # LLM assessment (medium confidence)
        if llm_response:
            if llm_response.correct:
                if not confidence_scores or min(confidence_scores) > 0.3:
                    correct = True
            confidence_scores.append(0.6)
            error_tags.extend(llm_response.error_tags)
            if llm_response.normalizer_notes:
                notes.append(f"LLM: {llm_response.normalizer_notes}")

        # Static analysis (lower confidence)
        if static_analysis:
            estimated = static_analysis.get("estimated_correctness", 0.0)
            if estimated > 0.7 and not error_tags:
                correct = True
            confidence_scores.append(0.3)

            # Add syntax errors
            if not static_analysis.get("syntax_valid", True):
                error_tags.append("syntax_error")
                correct = False

            # Add structural issues
            if not static_analysis.get("has_function", False):
                error_tags.append("incomplete")
                notes.append("No function definition found")

            if not static_analysis.get("has_return", False):
                error_tags.append("incomplete")
                notes.append("No return statement found")

        # Final logic checks
        if not model_answer.strip():
            correct = False
            error_tags = ["incomplete"]

        # Detect common patterns
        model_lower = model_answer.lower()
        if any(phrase in model_lower for phrase in ["todo", "not implemented", "pass"]):
            correct = False
            error_tags.append("incomplete")

        # Remove duplicate tags and limit
        error_tags = list(dict.fromkeys(error_tags))[:4]  # Max 4 tags

        # Combine notes
        final_notes = "; ".join(notes[:3])  # Limit note length

        return GradingResponse(
            ok=True,
            msg="graded",
            correct=correct,
            error_tags=error_tags,
            normalizer_notes=final_notes,
        )

    async def grade_batch(
        self, problems: list[Problem], model_answers: list[str], **kwargs
    ) -> list[GradingResponse]:
        """Grade multiple solutions efficiently.

        Args:
            problems: List of problems
            model_answers: Corresponding model solutions
            **kwargs: Additional grading options

        Returns:
            List of GradingResponse objects
        """
        if len(problems) != len(model_answers):
            raise ValueError("Number of problems must match number of answers")

        results = []

        for problem, answer in zip(problems, model_answers, strict=False):
            try:
                result = await self.grade_solution(problem, answer, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to grade problem {problem.id}: {e}")
                # Add error response
                results.append(
                    GradingResponse(
                        ok=False,
                        msg=f"grading_error: {e}",
                        correct=False,
                        error_tags=["grading_error"],
                        normalizer_notes=f"Grader encountered error: {str(e)}",
                    )
                )

        return results

    def calculate_accuracy(
        self, grading_results: list[GradingResponse]
    ) -> dict[str, float]:
        """Calculate accuracy statistics from grading results."""

        if not grading_results:
            return {"accuracy": 0.0, "total": 0, "correct": 0}

        total = len(grading_results)
        correct = sum(1 for result in grading_results if result.correct)
        accuracy = correct / total if total > 0 else 0.0

        # Error distribution
        error_counts = {}
        for result in grading_results:
            for tag in result.error_tags:
                error_counts[tag] = error_counts.get(tag, 0) + 1

        return {
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "error_distribution": error_counts,
        }


async def grade_code_solution(
    api_key: str,
    problem: Problem,
    model_answer: str,
    model: str = "anthropic/claude-3-5-haiku-20241022",
    **kwargs,
) -> GradingResponse:
    """Convenience function to grade a solution with minimal setup.

    Args:
        api_key: OpenRouter API key
        problem: Problem to grade against
        model_answer: Model's solution
        model: Model to use for grading
        **kwargs: Additional arguments for Grader

    Returns:
        GradingResponse with grading results
    """
    async with OpenRouterLLM(api_key=api_key) as client:
        grader = Grader(client, model=model)
        return await grader.grade_solution(problem, model_answer, **kwargs)


if __name__ == "__main__":
    # Demo usage
    import asyncio
    import os

    async def demo():
        # Create test problem
        problem = Problem(
            id="demo_grading",
            topic="basic_algorithms",
            difficulty=0.6,
            statement="Write a function that returns the maximum value in a list of integers.",
            canonical_answer="def find_max(lst):\n    return max(lst) if lst else None",
            rubric="Function finds maximum correctly, handles empty list",
            unit_tests=[
                "assert find_max([1, 3, 2]) == 3",
                "assert find_max([5]) == 5",
                "assert find_max([]) is None",
            ],
        )

        # Test cases
        test_answers = [
            # Correct solution
            "def find_max(lst):\n    return max(lst) if lst else None",
            # Wrong solution
            "def find_max(lst):\n    return min(lst)",
            # Incomplete solution
            "def find_max(lst):\n    pass",
            # Syntax error
            "def find_max(lst\n    return max(lst)",
        ]

        api_key = os.getenv("OPENROUTER_API_KEY", "demo-key")

        if api_key == "demo-key":
            print("🔧 Demo mode: Testing static analysis only")

            dummy_client = OpenRouterLLM(api_key="dummy")
            grader = Grader(dummy_client)

            for i, answer in enumerate(test_answers):
                print(f"\n📝 Test {i + 1}: {answer[:30]}...")
                analysis = grader._perform_static_analysis(answer, problem)
                print(f"   Syntax valid: {analysis['syntax_valid']}")
                print(f"   Has function: {analysis['has_function']}")
                print(f"   Has return: {analysis['has_return']}")
                print(
                    f"   Estimated correctness: {analysis['estimated_correctness']:.2f}"
                )

            return

        # Live grading test
        print("🎯 Testing live grading...")

        for i, answer in enumerate(test_answers[:2]):  # Test first 2 to save API calls
            try:
                result = await grade_code_solution(
                    api_key=api_key, problem=problem, model_answer=answer
                )

                print(f"\n📝 Solution {i + 1}")
                print(f"   Correct: {'✅' if result.correct else '❌'}")
                print(f"   Error tags: {result.error_tags}")
                print(f"   Notes: {result.normalizer_notes}")

            except Exception as e:
                print(f"❌ Grading failed: {e}")

    asyncio.run(demo())
