import time

import psutil
import torch

from .config import create_default_config
from .evolutionary_tournament import run_evolutionary_tournament
from .merger import AdvancedModelMerger
from .utils import evaluate_model, load_models
from .visualization import generate_html_report, plot_benchmark_comparison


def benchmark_merger(config):
    merger = AdvancedModelMerger(config)
    models = load_models(config.models)

    start_time = time.time()
    start_memory = psutil.virtual_memory().used

    merged_model_path = merger.merge()

    end_time = time.time()
    end_memory = psutil.virtual_memory().used

    time_taken = end_time - start_time
    memory_used = end_memory - start_memory

    evaluation_result = evaluate_model(merged_model_path)

    return {
        "time_taken": time_taken,
        "memory_used": memory_used,
        "evaluation_result": evaluation_result,
    }


def benchmark_evolutionary_tournament(config):
    start_time = time.time()
    start_memory = psutil.virtual_memory().used

    best_model_path = run_evolutionary_tournament(config)

    end_time = time.time()
    end_memory = psutil.virtual_memory().used

    time_taken = end_time - start_time
    memory_used = end_memory - start_memory

    evaluation_result = evaluate_model(best_model_path)

    return {
        "time_taken": time_taken,
        "memory_used": memory_used,
        "evaluation_result": evaluation_result,
    }


def run_benchmarks():
    config = create_default_config()

    print("Benchmarking Model Merger:")
    merger_results = benchmark_merger(config)
    print(f"Time taken: {merger_results['time_taken']:.2f} seconds")
    print(f"Memory used: {merger_results['memory_used'] / (1024 * 1024):.2f} MB")
    print(f"Evaluation result: {merger_results['evaluation_result']}")

    print("\nBenchmarking Evolutionary Tournament:")
    tournament_results = benchmark_evolutionary_tournament(config)
    print(f"Time taken: {tournament_results['time_taken']:.2f} seconds")
    print(f"Memory used: {tournament_results['memory_used'] / (1024 * 1024):.2f} MB")
    print(f"Evaluation result: {tournament_results['evaluation_result']}")

    # GPU benchmarking
    if torch.cuda.is_available():
        print("\nGPU Benchmarking:")
        start_memory = torch.cuda.memory_allocated()
        torch.cuda.synchronize()
        start_time = time.time()

        # Run a sample GPU operation
        a = torch.randn(10000, 10000, device="cuda")
        b = torch.randn(10000, 10000, device="cuda")
        c = torch.matmul(a, b)

        torch.cuda.synchronize()
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated()

        print(f"GPU operation time: {end_time - start_time:.2f} seconds")
        print(f"GPU memory used: {(end_memory - start_memory) / (1024 * 1024):.2f} MB")

    # Generate visualizations
    benchmark_data = [
        {
            "model": "Merger",
            "score": merger_results["evaluation_result"]["overall_score"],
            "time": merger_results["time_taken"],
            "memory": merger_results["memory_used"] / (1024 * 1024),
        },
        {
            "model": "Evolutionary",
            "score": tournament_results["evaluation_result"]["overall_score"],
            "time": tournament_results["time_taken"],
            "memory": tournament_results["memory_used"] / (1024 * 1024),
        },
    ]

    plot_benchmark_comparison(benchmark_data, "benchmark_comparison.png")
    generate_html_report(benchmark_data, "benchmark_report.html")


if __name__ == "__main__":
    run_benchmarks()
