from datetime import datetime

import pytest

from src.production.evolution.evolution.math_tutor_evolution import (
    EvolutionConfig,
    MathTutorEvolution,
    ModelIndividual,
)


@pytest.mark.asyncio
async def test_kpi_trends_across_generations(monkeypatch):
    # Stub out wandb to avoid external calls
    import src.production.evolution.evolution.math_tutor_evolution as mte

    monkeypatch.setattr(mte.wandb, "init", lambda *a, **k: None)
    monkeypatch.setattr(mte.wandb, "log", lambda *a, **k: None)

    import src.production.evolution.evolution.math_fitness as mf

    monkeypatch.setattr(mf.wandb, "log", lambda *a, **k: None)

    # Prepare evolution instance with minimal config
    config = EvolutionConfig(
        population_size=1,
        max_generations=3,
        elitism_count=1,
        mutation_rate=0.0,
        crossover_rate=0.0,
    )
    evolution = MathTutorEvolution(evolution_config=config)

    # Create a dummy individual and populate required caches
    individual = ModelIndividual(
        individual_id="ind1",
        model_name="dummy",
        model_path=None,
        lineage=["base"],
        generation=0,
        fitness_score=0.0,
        performance_metrics={},
        model_size_mb=0.0,
        parameters_count=0,
        quantization_config={},
        created_at=datetime.utcnow().isoformat(),
    )
    evolution.population = [individual]
    evolution.loaded_models[individual.individual_id] = object()
    evolution.tokenizers[individual.individual_id] = object()

    # Set up deterministic KPI values for each generation
    kpi_sequence = [
        {"correctness": 0.1},
        {"correctness": 0.2},
        {"correctness": 0.3},
    ]
    fitness_sequence = [0.1, 0.2, 0.3]

    async def fake_evaluate(
        self, model, tokenizer, individual_id=None, log_details=False
    ):
        self.kpi_scores = kpi_sequence.pop(0)
        return fitness_sequence.pop(0)

    monkeypatch.setattr(
        mf.MathFitnessEvaluator, "evaluate", fake_evaluate, raising=False
    )

    # Run several generations to collect KPIs
    for gen in range(1, 4):
        await evolution.evolve_generation(gen)

    recorded = [
        entry["kpi_scores"]["correctness"] for entry in evolution.generation_history
    ]
    assert recorded == pytest.approx([0.1, 0.2, 0.3])
    # Ensure KPI trend is non-decreasing
    assert recorded == sorted(recorded)
