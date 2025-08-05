#!/usr/bin/env python3
"""Create comprehensive visualizations for 50-generation evolution results."""

import json
from pathlib import Path
import warnings

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Set style for better visuals
plt.style.use("dark_background")
sns.set_palette("husl")


class Evolution50GenVisualizer:
    """Comprehensive visualizer for 50-generation evolution results."""

    def __init__(
        self,
        results_file: str = "D:/AgentForge/results_50gen/evolution_50gen_results.json",
    ) -> None:
        self.results_file = Path(results_file)
        self.data = self.load_results()

        # Color schemes for different merge methods
        self.method_colors = {
            "slerp": "#FF6B6B",
            "linear": "#4ECDC4",
            "task_arithmetic": "#45B7D1",
            "ties": "#96CEB4",
            "dare_ties": "#FFEAA7",
            "breadcrumbs": "#DDA0DD",
            "model_soup": "#98D8C8",
            "fisher_merging": "#F7DC6F",
        }

    def load_results(self):
        """Load 50-generation evolution results."""
        try:
            with open(self.results_file) as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Results file not found: {self.results_file}")
            return None

    def create_comprehensive_visualization(self) -> None:
        """Create comprehensive multi-panel visualization."""
        if not self.data:
            print("No data available for visualization")
            return

        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 4, figure=fig, height_ratios=[1.5, 1, 1, 1], width_ratios=[1, 1, 1, 1])

        # Main evolutionary tree (spans top row)
        ax_tree = fig.add_subplot(gs[0, :])
        self.plot_evolutionary_progression(ax_tree)

        # Fitness evolution
        ax_fitness = fig.add_subplot(gs[1, 0])
        self.plot_fitness_evolution(ax_fitness)

        # Method evolution
        ax_methods = fig.add_subplot(gs[1, 1])
        self.plot_method_evolution(ax_methods)

        # Diversity metrics
        ax_diversity = fig.add_subplot(gs[1, 2])
        self.plot_diversity_metrics(ax_diversity)

        # Best configuration
        ax_config = fig.add_subplot(gs[1, 3])
        self.plot_best_configuration(ax_config)

        # Benchmark evolution (spans width)
        ax_benchmarks = fig.add_subplot(gs[2, :2])
        self.plot_benchmark_evolution(ax_benchmarks)

        # Method performance comparison
        ax_method_perf = fig.add_subplot(gs[2, 2:])
        self.plot_method_performance(ax_method_perf)

        # Generation statistics
        ax_gen_stats = fig.add_subplot(gs[3, 0])
        self.plot_generation_statistics(ax_gen_stats)

        # Parameter evolution
        ax_params = fig.add_subplot(gs[3, 1])
        self.plot_parameter_evolution(ax_params)

        # Stagnation analysis
        ax_stagnation = fig.add_subplot(gs[3, 2])
        self.plot_stagnation_analysis(ax_stagnation)

        # Final population
        ax_final = fig.add_subplot(gs[3, 3])
        self.plot_final_population(ax_final)

        plt.tight_layout()
        plt.suptitle(
            "50-Generation Agent Forge Evolution Merge - Comprehensive Analysis",
            fontsize=20,
            fontweight="bold",
            y=0.98,
            color="white",
        )

        # Save visualization
        output_file = "D:/AgentForge/results_50gen/evolution_50gen_visualization.png"
        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
            facecolor="black",
            edgecolor="none",
        )
        print(f"Comprehensive visualization saved to: {output_file}")
        plt.show()

    def plot_evolutionary_progression(self, ax) -> None:
        """Plot the main evolutionary progression across 50 generations."""
        generations = self.data["generation_history"]

        # Sample every 5th generation for clarity (10 points total)
        sample_gens = [gen for i, gen in enumerate(generations) if i % 5 == 0 or i == len(generations) - 1]

        x_positions = []
        y_positions = []
        colors = []
        sizes = []

        for _i, gen_data in enumerate(sample_gens):
            gen_num = gen_data["generation"]
            population = gen_data["population"]

            for j, individual in enumerate(population):
                x_positions.append(gen_num)
                y_positions.append(j)

                method = individual["merge_method"]
                fitness = individual["fitness"]

                colors.append(self.method_colors.get(method, "#CCCCCC"))
                sizes.append(50 + fitness * 200)

        ax.scatter(
            x_positions,
            y_positions,
            c=colors,
            s=sizes,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
        )

        # Add fitness progression line
        best_fitness_by_gen = [gen["best_fitness"] for gen in generations]
        gen_numbers = [gen["generation"] for gen in generations]

        ax2 = ax.twinx()
        ax2.plot(
            gen_numbers,
            best_fitness_by_gen,
            "white",
            linewidth=3,
            alpha=0.8,
            label="Best Fitness",
        )
        ax2.set_ylabel("Fitness Score", color="white", fontweight="bold")
        ax2.tick_params(colors="white")

        ax.set_xlabel("Generation", color="white", fontweight="bold")
        ax.set_ylabel("Individual Index", color="white", fontweight="bold")
        ax.set_title(
            "Evolutionary Progression (50 Generations)",
            color="white",
            fontweight="bold",
            fontsize=14,
        )

        # Create method legend
        legend_elements = []
        for method, color in self.method_colors.items():
            legend_elements.append(plt.scatter([], [], c=color, s=100, label=method.replace("_", " ").title()))

        ax.legend(
            handles=legend_elements,
            loc="upper left",
            fontsize=8,
            facecolor="black",
            edgecolor="white",
        )

        ax.grid(True, alpha=0.3)
        ax.set_facecolor("black")
        ax2.set_facecolor("black")

    def plot_fitness_evolution(self, ax) -> None:
        """Plot fitness evolution over 50 generations."""
        generations = self.data["generation_history"]

        gen_numbers = [gen["generation"] for gen in generations]
        best_fitness = [gen["best_fitness"] for gen in generations]
        avg_fitness = [gen["average_fitness"] for gen in generations]

        ax.plot(
            gen_numbers,
            best_fitness,
            "o-",
            color="#6BCF7F",
            linewidth=2,
            markersize=4,
            label="Best Fitness",
            alpha=0.9,
        )
        ax.plot(
            gen_numbers,
            avg_fitness,
            "s-",
            color="#FFD93D",
            linewidth=2,
            markersize=3,
            label="Average Fitness",
            alpha=0.7,
        )

        # Add moving average
        if len(best_fitness) >= 10:
            moving_avg = pd.Series(best_fitness).rolling(window=10).mean()
            ax.plot(
                gen_numbers,
                moving_avg,
                "--",
                color="white",
                linewidth=2,
                alpha=0.6,
                label="10-Gen Moving Avg",
            )

        ax.fill_between(gen_numbers, best_fitness, avg_fitness, alpha=0.2, color="#4ECDC4")

        ax.set_xlabel("Generation", color="white", fontweight="bold")
        ax.set_ylabel("Fitness Score", color="white", fontweight="bold")
        ax.set_title("Fitness Evolution", color="white", fontweight="bold")
        ax.legend(fontsize=8, facecolor="black", edgecolor="white")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("black")

    def plot_method_evolution(self, ax) -> None:
        """Plot how merge methods evolved over generations."""
        method_evolution = self.data["performance_metrics"]["method_evolution"]

        # Convert to DataFrame for easier plotting
        methods = set()
        for gen_methods in method_evolution.values():
            methods.update(gen_methods.keys())

        methods = sorted(methods)
        generations = sorted([int(g) for g in method_evolution])

        # Create stacked area plot
        method_data = []
        for method in methods:
            method_counts = []
            for gen in generations:
                count = method_evolution[str(gen)].get(method, 0)
                method_counts.append(count)
            method_data.append(method_counts)

        ax.stackplot(
            generations,
            *method_data,
            labels=methods,
            colors=[self.method_colors.get(m, "#CCCCCC") for m in methods],
            alpha=0.8,
        )

        ax.set_xlabel("Generation", color="white", fontweight="bold")
        ax.set_ylabel("Population Count", color="white", fontweight="bold")
        ax.set_title("Method Evolution", color="white", fontweight="bold")
        ax.legend(fontsize=6, facecolor="black", edgecolor="white", loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("black")

    def plot_diversity_metrics(self, ax) -> None:
        """Plot population diversity metrics."""
        if "diversity_evolution" in self.data:
            diversity_history = self.data["diversity_evolution"]
            generations = list(range(len(diversity_history)))

            ax.plot(
                generations,
                diversity_history,
                "o-",
                color="#FF6B6B",
                linewidth=2,
                markersize=4,
                alpha=0.8,
            )

            # Add trend line
            if len(diversity_history) > 5:
                z = np.polyfit(generations, diversity_history, 1)
                p = np.poly1d(z)
                ax.plot(
                    generations,
                    p(generations),
                    "--",
                    color="white",
                    alpha=0.6,
                    linewidth=2,
                    label="Trend",
                )

            ax.set_xlabel("Generation", color="white", fontweight="bold")
            ax.set_ylabel("Diversity Score", color="white", fontweight="bold")
            ax.set_title("Population Diversity", color="white", fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.set_facecolor("black")
        else:
            ax.text(
                0.5,
                0.5,
                "Diversity data\nnot available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="white",
                fontsize=12,
            )
            ax.set_facecolor("black")

    def plot_best_configuration(self, ax) -> None:
        """Display the best configuration found."""
        best_config = self.data["evolution_summary"]["best_configuration"]

        ax.text(
            0.05,
            0.95,
            "Best Configuration",
            fontsize=12,
            fontweight="bold",
            color="#6BCF7F",
            transform=ax.transAxes,
        )

        ax.text(
            0.05,
            0.85,
            f"Method: {best_config['merge_method']}",
            fontsize=10,
            color="white",
            transform=ax.transAxes,
        )

        ax.text(
            0.05,
            0.75,
            f"Fitness: {best_config['fitness']:.4f}",
            fontsize=10,
            color="white",
            transform=ax.transAxes,
        )

        ax.text(
            0.05,
            0.65,
            f"Generation: {best_config.get('generation', 'N/A')}",
            fontsize=10,
            color="white",
            transform=ax.transAxes,
        )

        # Parameters
        params = best_config.get("parameters", {})
        if params:
            param_text = "\n".join(
                [f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in params.items()]
            )
            ax.text(
                0.05,
                0.45,
                f"Parameters:\n{param_text}",
                fontsize=9,
                color="white",
                transform=ax.transAxes,
            )

        # Benchmark scores
        benchmarks = best_config.get("benchmark_results", {})
        if benchmarks:
            ax.text(
                0.05,
                0.25,
                "Benchmarks:",
                fontsize=10,
                fontweight="bold",
                color="#FFD93D",
                transform=ax.transAxes,
            )

            y_pos = 0.15
            for metric, score in benchmarks.items():
                ax.text(
                    0.1,
                    y_pos,
                    f"{metric.upper()}: {score:.3f}",
                    fontsize=9,
                    color="white",
                    transform=ax.transAxes,
                )
                y_pos -= 0.06

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor("black")

    def plot_benchmark_evolution(self, ax) -> None:
        """Plot how benchmark scores evolved."""
        generations = self.data["generation_history"]

        # Extract best benchmark scores per generation
        benchmark_evolution = {}
        gen_numbers = []

        for gen_data in generations:
            gen_numbers.append(gen_data["generation"])

            # Get best individual's benchmarks
            best_individual = max(gen_data["population"], key=lambda x: x["fitness"])
            benchmarks = best_individual.get("benchmark_results", {})

            for metric, score in benchmarks.items():
                if metric not in benchmark_evolution:
                    benchmark_evolution[metric] = []
                benchmark_evolution[metric].append(score)

        # Plot each benchmark metric
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFD93D", "#6BCF7F"]

        for i, (metric, scores) in enumerate(benchmark_evolution.items()):
            color = colors[i % len(colors)]
            ax.plot(
                gen_numbers,
                scores,
                "o-",
                color=color,
                linewidth=2,
                markersize=3,
                label=metric.upper(),
                alpha=0.8,
            )

        # Add threshold lines
        thresholds = {
            "mmlu": 0.65,
            "gsm8k": 0.45,
            "humaneval": 0.30,
            "hellaswag": 0.70,
            "arc": 0.55,
        }
        for metric, threshold in thresholds.items():
            if metric in benchmark_evolution:
                ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.5, linewidth=1)

        ax.set_xlabel("Generation", color="white", fontweight="bold")
        ax.set_ylabel("Benchmark Score", color="white", fontweight="bold")
        ax.set_title("Benchmark Score Evolution", color="white", fontweight="bold")
        ax.legend(fontsize=8, facecolor="black", edgecolor="white")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("black")

    def plot_method_performance(self, ax) -> None:
        """Plot performance comparison by merge method."""
        generations = self.data["generation_history"]

        # Collect performance data by method
        method_performance = {}

        for gen_data in generations:
            for individual in gen_data["population"]:
                method = individual["merge_method"]
                fitness = individual["fitness"]

                if method not in method_performance:
                    method_performance[method] = []
                method_performance[method].append(fitness)

        # Create box plot
        methods = list(method_performance.keys())
        data = [method_performance[method] for method in methods]
        colors = [self.method_colors.get(method, "#CCCCCC") for method in methods]

        bp = ax.boxplot(
            data,
            labels=[m.replace("_", "\n") for m in methods],
            patch_artist=True,
            showfliers=False,
        )

        for patch, color in zip(bp["boxes"], colors, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel("Fitness Score", color="white", fontweight="bold")
        ax.set_title("Method Performance Distribution", color="white", fontweight="bold")
        ax.tick_params(colors="white", labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("black")

    def plot_generation_statistics(self, ax) -> None:
        """Plot generation-level statistics."""
        generations = self.data["generation_history"]

        # Calculate fitness standard deviation per generation
        fitness_std = []
        gen_numbers = []

        for gen_data in generations:
            gen_numbers.append(gen_data["generation"])
            fitness_scores = [ind["fitness"] for ind in gen_data["population"]]
            fitness_std.append(np.std(fitness_scores))

        ax.bar(gen_numbers[::5], fitness_std[::5], color="#FF6B6B", alpha=0.7, width=2)

        ax.set_xlabel("Generation", color="white", fontweight="bold")
        ax.set_ylabel("Fitness Std Dev", color="white", fontweight="bold")
        ax.set_title("Population Variance", color="white", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("black")

    def plot_parameter_evolution(self, ax) -> None:
        """Plot how key parameters evolved."""
        generations = self.data["generation_history"]

        # Track task_arithmetic scaling coefficient evolution
        gen_numbers = []
        scaling_coeffs = []

        for gen_data in generations:
            gen_numbers.append(gen_data["generation"])

            # Get all task_arithmetic individuals
            ta_individuals = [ind for ind in gen_data["population"] if ind["merge_method"] == "task_arithmetic"]

            if ta_individuals:
                # Average scaling coefficient
                coeffs = [ind.get("parameters", {}).get("scaling_coefficient", 1.0) for ind in ta_individuals]
                avg_coeff = np.mean(coeffs)
                scaling_coeffs.append(avg_coeff)
            else:
                scaling_coeffs.append(np.nan)

        # Remove NaN values for plotting
        valid_indices = [i for i, val in enumerate(scaling_coeffs) if not np.isnan(val)]
        valid_gens = [gen_numbers[i] for i in valid_indices]
        valid_coeffs = [scaling_coeffs[i] for i in valid_indices]

        if valid_coeffs:
            ax.plot(
                valid_gens,
                valid_coeffs,
                "o-",
                color="#45B7D1",
                linewidth=2,
                markersize=4,
                alpha=0.8,
            )

            ax.set_xlabel("Generation", color="white", fontweight="bold")
            ax.set_ylabel("Avg Scaling Coefficient", color="white", fontweight="bold")
            ax.set_title(
                "Parameter Evolution\n(Task Arithmetic)",
                color="white",
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                "No task_arithmetic\nindividuals found",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="white",
                fontsize=10,
            )

        ax.set_facecolor("black")

    def plot_stagnation_analysis(self, ax) -> None:
        """Plot stagnation analysis."""
        generations = self.data["generation_history"]

        # Calculate improvement rate
        gen_numbers = []
        improvements = []

        prev_best = 0
        for gen_data in generations:
            gen_numbers.append(gen_data["generation"])
            current_best = gen_data["best_fitness"]
            improvement = current_best - prev_best
            improvements.append(improvement)
            prev_best = current_best

        # Plot improvements
        colors = ["green" if imp > 0 else "red" if imp < 0 else "gray" for imp in improvements]
        ax.bar(gen_numbers, improvements, color=colors, alpha=0.7, width=0.8)

        ax.axhline(y=0, color="white", linestyle="-", alpha=0.5)
        ax.set_xlabel("Generation", color="white", fontweight="bold")
        ax.set_ylabel("Fitness Improvement", color="white", fontweight="bold")
        ax.set_title("Generation Improvements", color="white", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("black")

    def plot_final_population(self, ax) -> None:
        """Plot final population characteristics."""
        final_population = self.data["final_population"]

        # Method distribution in final population
        methods = [ind["merge_method"] for ind in final_population]
        method_counts = {}
        for method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1

        if method_counts:
            methods = list(method_counts.keys())
            counts = list(method_counts.values())
            colors = [self.method_colors.get(method, "#CCCCCC") for method in methods]

            wedges, texts, autotexts = ax.pie(counts, labels=methods, colors=colors, autopct="%1.0f%%", startangle=90)

            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")
                autotext.set_fontsize(8)

            for text in texts:
                text.set_color("white")
                text.set_fontweight("bold")
                text.set_fontsize(8)

        ax.set_title("Final Population\nMethod Distribution", color="white", fontweight="bold")


def main() -> None:
    """Create comprehensive 50-generation visualization."""
    print("Creating comprehensive 50-generation evolution visualization...")

    visualizer = Evolution50GenVisualizer()
    if visualizer.data:
        visualizer.create_comprehensive_visualization()
        print("Comprehensive 50-generation visualization completed!")
    else:
        print("Could not load 50-generation evolution results data")


if __name__ == "__main__":
    main()
