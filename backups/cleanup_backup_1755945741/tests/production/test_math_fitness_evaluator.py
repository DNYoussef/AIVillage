import asyncio

import pytest
from src.production.evolution.evolution.math_fitness import EvaluationResult, MathFitnessEvaluator, MathProblem


class DummyEvaluator(MathFitnessEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_flight = 0
        self.max_in_flight = 0

    async def evaluate_problem(
        self, model, tokenizer, problem: MathProblem
    ) -> EvaluationResult:  # type: ignore[override]
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        return EvaluationResult(
            problem_id=problem.problem_id,
            model_response="ok",
            correctness_score=1.0,
            step_by_step_score=1.0,
            explanation_quality=1.0,
            encouragement_score=1.0,
            cultural_sensitivity=1.0,
            response_time=0.01,
            total_score=1.0,
            feedback="",
            evaluation_time=0.01,
        )


@pytest.mark.asyncio
async def test_concurrency_limit():
    evaluator = DummyEvaluator(max_concurrent_evaluations=2)
    evaluator.test_suite = {
        "algebra": [
            MathProblem("p1", "algebra", 5, 0.5, "1+1?", "2", [""], [""], {}),
            MathProblem("p2", "algebra", 5, 0.5, "1+2?", "3", [""], [""], {}),
            MathProblem("p3", "algebra", 5, 0.5, "1+3?", "4", [""], [""], {}),
        ]
    }
    await evaluator.evaluate(None, None, log_details=False)
    assert evaluator.max_in_flight <= evaluator.max_concurrent_evaluations


class SimpleEvaluator(MathFitnessEvaluator):
    async def generate_model_response(
        self, model, tokenizer, prompt: str, max_length: int = 200
    ) -> str:  # type: ignore[override]
        await asyncio.sleep(0.01)
        return "2"

    def generate_feedback(self, *args, **kwargs) -> str:  # type: ignore[override]
        return "ok"


@pytest.mark.asyncio
async def test_latency_recorded():
    evaluator = SimpleEvaluator()
    problem = MathProblem(
        problem_id="p1",
        category="arithmetic",
        grade_level=1,
        difficulty=0.1,
        problem_text="1+1?",
        expected_answer="2",
        solution_steps=[""],
        keywords=[""],
        scoring_criteria={},
    )
    result = await evaluator.evaluate_problem(None, None, problem)
    assert result.evaluation_time >= result.response_time > 0
