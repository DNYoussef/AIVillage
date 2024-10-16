import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import seaborn as sns
import pandas as pd

def plot_fitness_over_generations(fitness_scores: List[float], output_path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_scores)
    plt.title('Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.savefig(output_path)
    plt.close()

def plot_pareto_front(scores: List[Dict[str, float]], pareto_front_indices: List[int], objectives: List[str], output_path: str):
    if len(objectives) != 2:
        raise ValueError("Pareto front visualization is only supported for 2 objectives")

    plt.figure(figsize=(10, 6))
    x = [scores[i][objectives[0]] for i in range(len(scores))]
    y = [scores[i][objectives[1]] for i in range(len(scores))]
    plt.scatter(x, y, c='blue', label='All solutions')
    
    pareto_x = [scores[i][objectives[0]] for i in pareto_front_indices]
    pareto_y = [scores[i][objectives[1]] for i in pareto_front_indices]
    plt.scatter(pareto_x, pareto_y, c='red', label='Pareto front')
    
    plt.title('Pareto Front')
    plt.xlabel(objectives[0])
    plt.ylabel(objectives[1])
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def plot_evolution_progress(all_generation_scores: List[List[Dict[str, float]]], objectives: List[str], objective_types: List[str], output_path: str):
    num_generations = len(all_generation_scores)
    num_objectives = len(objectives)

    fig, axes = plt.subplots(num_objectives, 1, figsize=(10, 5 * num_objectives), sharex=True)
    if num_objectives == 1:
        axes = [axes]

    for i, (objective, obj_type) in enumerate(zip(objectives, objective_types)):
        best_scores = []
        avg_scores = []
        for gen_scores in all_generation_scores:
            scores = [score[objective] for score in gen_scores]
            best_score = max(scores) if obj_type == 'maximize' else min(scores)
            avg_score = np.mean(scores)
            best_scores.append(best_score)
            avg_scores.append(avg_score)

        axes[i].plot(range(num_generations), best_scores, label='Best')
        axes[i].plot(range(num_generations), avg_scores, label='Average')
        axes[i].set_title(f'{objective} Progress')
        axes[i].set_ylabel('Score')
        axes[i].legend()

    axes[-1].set_xlabel('Generation')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_html_report(benchmark_results, output_path):
    html_content = """
    <html>
    <head>
        <title>EvoMerge Benchmark Results</title>
        <style>
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid black; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>EvoMerge Benchmark Results</h1>
        <table>
            <tr>
                <th>Model</th>
                <th>Score</th>
                <th>Time (s)</th>
                <th>Memory (MB)</th>
            </tr>
    """
    
    for result in benchmark_results:
        html_content += f"""
            <tr>
                <td>{result['model']}</td>
                <td>{result['score']:.4f}</td>
                <td>{result['time']:.2f}</td>
                <td>{result['memory']:.2f}</td>
            </tr>
        """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)

def plot_benchmark_comparison(benchmark_results, output_path):
    df = pd.DataFrame(benchmark_results)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model', y='score', data=df)
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Performance Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
