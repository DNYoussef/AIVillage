from .config import Configuration, ModelReference, MergeSettings, EvolutionSettings, create_default_config
from .merger import AdvancedModelMerger
from .evolutionary_tournament import EvolutionaryMerger, run_evolutionary_tournament
from .utils import (
    load_models,
    save_model,
    generate_text,
    evaluate_model,
    setup_gpu_if_available,
    clean_up_models,
    MERGE_TECHNIQUES,
    parallel_evaluate_models
)
from .visualization import plot_fitness_over_generations, plot_benchmark_comparison, generate_html_report
from .logging_config import setup_logging

__all__ = [
    "Configuration",
    "ModelReference",
    "MergeSettings",
    "EvolutionSettings",
    "create_default_config",
    "AdvancedModelMerger",
    "EvolutionaryMerger",
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
    "plot_benchmark_comparison",
    "generate_html_report",
    "setup_logging"
]
