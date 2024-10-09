import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
