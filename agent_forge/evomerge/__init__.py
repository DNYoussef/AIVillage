from .config import Configuration, ModelReference, MergeSettings, EvolutionSettings, create_default_config
from .merging.merger import AdvancedModelMerger
from .evolutionary_tournament import EvolutionaryTournament, run_evolutionary_tournament
from .utils import (
    load_models,
    save_model,
    generate_text,
    evaluate_model,
    setup_gpu_if_available,
    clean_up_models,
    parallel_evaluate_models
)
from .merging.merge_techniques import MERGE_TECHNIQUES
from .visualization import (
    plot_fitness_over_generations,
    plot_pareto_front,
    plot_evolution_progress,
    generate_html_report,
    plot_benchmark_comparison
)
from .logging_config import setup_logging

__all__ = [
    "Configuration",
    "ModelReference",
    "MergeSettings",
    "EvolutionSettings",
    "create_default_config",
    "AdvancedModelMerger",
    "EvolutionaryTournament",
    "run_evolutionary_tournament",
    "load_models",
    "save_model",
    "generate_text",
    "evaluate_model",
    "setup_gpu_if_available",
    "clean_up_models",
    "MERGE_TECHNIQUES",
    "parallel_evaluate_models",
    "plot_fitness_over_generations",
    "plot_pareto_front",
    "plot_evolution_progress",
    "generate_html_report",
    "plot_benchmark_comparison",
    "setup_logging"
]
