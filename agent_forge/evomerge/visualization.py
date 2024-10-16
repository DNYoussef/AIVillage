import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def plot_fitness_over_generations(fitness_scores, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_scores)
    plt.title('Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.savefig(output_path)
    plt.close()

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

def plot_pareto_front(objectives, objective_names, output_path):
    if len(objective_names) == 2:
        plt.figure(figsize=(10, 6))
        plt.scatter([obj[0] for obj in objectives], [obj[1] for obj in objectives])
        plt.title('Pareto Front')
        plt.xlabel(objective_names[0])
        plt.ylabel(objective_names[1])
        plt.savefig(output_path)
        plt.close()
    elif len(objective_names) == 3:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter([obj[0] for obj in objectives], [obj[1] for obj in objectives], [obj[2] for obj in objectives])
        ax.set_xlabel(objective_names[0])
        ax.set_ylabel(objective_names[1])
        ax.set_zlabel(objective_names[2])
        plt.title('Pareto Front')
        plt.savefig(output_path)
        plt.close()
    else:
        logger.warning(f"Cannot visualize Pareto front for {len(objective_names)} objectives. Skipping visualization.")
