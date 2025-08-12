import pytest

from src.production.evolution.evolution.math_tutor_evolution import (
    EvolutionConfig,
    MathTutorEvolution,
    ModelIndividual,
)


class DummyEvaluator:
    call_counter = 0

    def __init__(self, *args, **kwargs):
        self.kpi_scores = {}

    async def evaluate(self, model, tokenizer, individual_id=None, log_details=True):
        score = 0.1 * (self.__class__.call_counter + 1)
        self.kpi_scores = {"fitness_score": score}
        self.__class__.call_counter += 1
        return score


@pytest.mark.asyncio
async def test_kpi_trend_across_generations(monkeypatch):
    monkeypatch.setattr(
        "src.production.evolution.evolution.math_fitness.MathFitnessEvaluator",
        DummyEvaluator,
    )
    monkeypatch.setattr("wandb.init", lambda *args, **kwargs: None)
    monkeypatch.setattr("wandb.log", lambda *args, **kwargs: None)

    config = EvolutionConfig(population_size=2, max_generations=2)
    evolution = MathTutorEvolution(evolution_config=config)

    ind1 = ModelIndividual("id1", "m1", None, [], 0, 0.0, {}, 0.0, 0, {})
    ind2 = ModelIndividual("id2", "m2", None, [], 0, 0.0, {}, 0.0, 0, {})
    evolution.population = [ind1, ind2]
    evolution.loaded_models = {"id1": object(), "id2": object()}
    evolution.tokenizers = {"id1": object(), "id2": object()}

    await evolution.evaluate_population_fitness()
    await evolution.evaluate_population_fitness()

    assert len(evolution.kpi_history) == 2
    gen0 = [m["fitness_score"] for m in evolution.kpi_history[0].values()]
    gen1 = [m["fitness_score"] for m in evolution.kpi_history[1].values()]
    assert sum(gen1) / len(gen1) > sum(gen0) / len(gen0)
