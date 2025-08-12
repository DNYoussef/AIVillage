#!/usr/bin/env python3
"""Create evolutionary tree visualization for Agent Forge evolution merge results."""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

warnings.filterwarnings("ignore")

# Set style
plt.style.use("dark_background")
sns.set_palette("husl")


class EvolutionVisualizer:
    """Visualize Agent Forge evolution results as evolutionary tree."""

    def __init__(
        self, results_file: str = "D:/AgentForge/results/evolution_results.json"
    ) -> None:
        self.results_file = Path(results_file)
        self.data = self.load_results()
        self.colors = {
            "slerp": "#FF6B6B",
            "linear": "#4ECDC4",
            "task_arithmetic": "#45B7D1",
            "elite": "#FFD93D",
            "best": "#6BCF7F",
        }

    def load_results(self):
        """Load evolution results from JSON file."""
        try:
            with open(self.results_file) as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Results file not found: {self.results_file}")
            return None

    def create_evolutionary_tree(self) -> None:
        """Create the main evolutionary tree visualization."""
        if not self.data:
            return

        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1])

        # Main evolutionary tree
        ax_tree = fig.add_subplot(gs[0, :])
        self.plot_evolutionary_tree(ax_tree)

        # Fitness progression
        ax_fitness = fig.add_subplot(gs[1, 0])
        self.plot_fitness_progression(ax_fitness)

        # Method distribution
        ax_methods = fig.add_subplot(gs[1, 1])
        self.plot_method_distribution(ax_methods)

        # Best configuration
        ax_config = fig.add_subplot(gs[1, 2])
        self.plot_best_configuration(ax_config)

        # Generation statistics
        ax_gen_stats = fig.add_subplot(gs[2, 0])
        self.plot_generation_statistics(ax_gen_stats)

        # Benchmark evolution
        ax_benchmarks = fig.add_subplot(gs[2, 1:])
        self.plot_benchmark_evolution(ax_benchmarks)

        plt.tight_layout()
        plt.suptitle(
            "🧬 Agent Forge Evolution Merge - Evolutionary Tree Analysis",
            fontsize=20,
            fontweight="bold",
            y=0.98,
        )

        output_file = "D:/AgentForge/results/evolution_tree.png"
        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
            facecolor="black",
            edgecolor="none",
        )
        print(f"Evolutionary tree saved to: {output_file}")
        plt.show()

    def plot_evolutionary_tree(self, ax) -> None:
        """Plot the main evolutionary tree showing parent-child relationships."""
        generations = self.data["generation_history"]

        # Create positions for each individual
        positions = {}
        y_spacing = 1.0
        x_spacing = 2.0

        # Position individuals in each generation
        for gen_idx, gen_data in enumerate(generations):
            population = gen_data["population"]
            n_individuals = len(population)

            for i, individual in enumerate(population):
                # Center individuals in each generation
                y_offset = (i - (n_individuals - 1) / 2) * y_spacing
                x_pos = gen_idx * x_spacing
                y_pos = y_offset

                ind_id = individual["id"]
                positions[ind_id] = (x_pos, y_pos)

        # Draw individuals as nodes
        for gen_idx, gen_data in enumerate(generations):
            population = gen_data["population"]

            for individual in population:
                ind_id = individual["id"]
                x, y = positions[ind_id]

                # Color by method
                method = individual["merge_method"]
                color = self.colors.get(method, "#CCCCCC")

                # Size by fitness
                fitness = individual["fitness"]
                size = 200 + (fitness * 300)

                # Mark elite and best individuals
                if fitness == max(ind["fitness"] for ind in population):
                    # Best in generation
                    ax.scatter(
                        x,
                        y,
                        s=size + 100,
                        c="gold",
                        marker="*",
                        edgecolors="black",
                        linewidth=2,
                        alpha=0.8,
                        zorder=5,
                    )

                ax.scatter(
                    x,
                    y,
                    s=size,
                    c=color,
                    alpha=0.7,
                    edgecolors="white",
                    linewidth=1,
                    zorder=3,
                )

                # Add fitness label
                ax.annotate(
                    f"{fitness:.3f}",
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color="white",
                    weight="bold",
                )

        # Draw inheritance lines (simplified - connect best to next generation)
        for gen_idx in range(len(generations) - 1):
            current_gen = generations[gen_idx]["population"]
            next_gen = generations[gen_idx + 1]["population"]

            # Find best individual in current generation
            best_current = max(current_gen, key=lambda x: x["fitness"])
            best_current_id = best_current["id"]

            if best_current_id in positions:
                x1, y1 = positions[best_current_id]

                # Connect to elite individuals in next generation
                elite_next = sorted(next_gen, key=lambda x: x["fitness"], reverse=True)[
                    :2
                ]

                for elite in elite_next:
                    elite_id = elite["id"]
                    if elite_id in positions:
                        x2, y2 = positions[elite_id]

                        # Draw inheritance line
                        ax.plot(
                            [x1, x2],
                            [y1, y2],
                            "white",
                            alpha=0.3,
                            linewidth=1,
                            linestyle="--",
                            zorder=1,
                        )

        # Customize axes
        ax.set_xlabel("Generation", fontsize=14, fontweight="bold", color="white")
        ax.set_ylabel(
            "Population Diversity", fontsize=14, fontweight="bold", color="white"
        )
        ax.set_title(
            "🌳 Evolutionary Tree - Model Evolution Across Generations",
            fontsize=16,
            fontweight="bold",
            color="white",
            pad=20,
        )

        # Add generation labels
        for gen_idx in range(len(generations)):
            ax.axvline(x=gen_idx * x_spacing, color="gray", alpha=0.3, linestyle=":")
            ax.text(
                gen_idx * x_spacing,
                ax.get_ylim()[1] + 0.2,
                f"Gen {gen_idx}",
                ha="center",
                fontsize=10,
                color="white",
                weight="bold",
            )

        # Create legend
        legend_elements = []
        for method, color in self.colors.items():
            if method not in ["elite", "best"]:
                legend_elements.append(
                    plt.scatter(
                        [], [], c=color, s=100, label=method.replace("_", " ").title()
                    )
                )

        legend_elements.append(
            plt.scatter([], [], c="gold", s=150, marker="*", label="Best in Generation")
        )

        ax.legend(
            handles=legend_elements,
            loc="upper left",
            fontsize=10,
            facecolor="black",
            edgecolor="white",
        )

        ax.grid(True, alpha=0.2)
        ax.set_facecolor("black")

    def plot_fitness_progression(self, ax) -> None:
        """Plot fitness progression over generations."""
        generations = self.data["generation_history"]

        gen_numbers = []
        best_fitness = []
        avg_fitness = []

        for gen_data in generations:
            gen_numbers.append(gen_data["generation"])
            best_fitness.append(gen_data["best_fitness"])
            avg_fitness.append(gen_data["average_fitness"])

        ax.plot(
            gen_numbers,
            best_fitness,
            "o-",
            color="#6BCF7F",
            linewidth=3,
            markersize=8,
            label="Best Fitness",
        )
        ax.plot(
            gen_numbers,
            avg_fitness,
            "s-",
            color="#FFD93D",
            linewidth=2,
            markersize=6,
            label="Average Fitness",
        )

        ax.fill_between(
            gen_numbers, best_fitness, avg_fitness, alpha=0.3, color="#4ECDC4"
        )

        ax.set_xlabel("Generation", fontweight="bold", color="white")
        ax.set_ylabel("Fitness Score", fontweight="bold", color="white")
        ax.set_title("📈 Fitness Evolution", fontweight="bold", color="white")
        ax.legend(facecolor="black", edgecolor="white")
        ax.grid(True, alpha=0.2)
        ax.set_facecolor("black")

    def plot_method_distribution(self, ax) -> None:
        """Plot distribution of merge methods across generations."""
        method_counts = {}

        for gen_data in self.data["generation_history"]:
            for individual in gen_data["population"]:
                method = individual["merge_method"]
                method_counts[method] = method_counts.get(method, 0) + 1

        methods = list(method_counts.keys())
        counts = list(method_counts.values())
        colors = [self.colors.get(method, "#CCCCCC") for method in methods]

        wedges, texts, autotexts = ax.pie(
            counts, labels=methods, colors=colors, autopct="%1.1f%%", startangle=90
        )

        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")

        for text in texts:
            text.set_color("white")
            text.set_fontweight("bold")

        ax.set_title("🔧 Merge Method Distribution", fontweight="bold", color="white")

    def plot_best_configuration(self, ax) -> None:
        """Display the best configuration found."""
        best_config = self.data["evolution_summary"]["best_configuration"]

        ax.text(
            0.1,
            0.9,
            "🏆 Best Configuration",
            fontsize=14,
            fontweight="bold",
            color="#6BCF7F",
            transform=ax.transAxes,
        )

        ax.text(
            0.1,
            0.7,
            f"Method: {best_config['merge_method']}",
            fontsize=12,
            color="white",
            transform=ax.transAxes,
        )

        ax.text(
            0.1,
            0.6,
            f"Fitness: {best_config['fitness']:.3f}",
            fontsize=12,
            color="white",
            transform=ax.transAxes,
        )

        ax.text(
            0.1,
            0.5,
            f"Generation: {best_config['generation']}",
            fontsize=12,
            color="white",
            transform=ax.transAxes,
        )

        # Parameters
        params = best_config["parameters"]
        param_text = ", ".join([f"{k}: {v}" for k, v in params.items()])
        ax.text(
            0.1,
            0.4,
            f"Parameters: {param_text}",
            fontsize=10,
            color="white",
            transform=ax.transAxes,
        )

        # Benchmark scores
        benchmarks = best_config["benchmark_results"]
        ax.text(
            0.1,
            0.2,
            "Benchmark Scores:",
            fontsize=12,
            fontweight="bold",
            color="#FFD93D",
            transform=ax.transAxes,
        )

        y_pos = 0.1
        for metric, score in benchmarks.items():
            ax.text(
                0.15,
                y_pos,
                f"{metric.upper()}: {score:.3f}",
                fontsize=10,
                color="white",
                transform=ax.transAxes,
            )
            y_pos -= 0.05

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor("black")

    def plot_generation_statistics(self, ax) -> None:
        """Plot statistics for each generation."""
        generations = self.data["generation_history"]

        gen_numbers = [gen["generation"] for gen in generations]
        fitness_std = []

        for gen_data in generations:
            population = gen_data["population"]
            fitness_scores = [ind["fitness"] for ind in population]
            fitness_std.append(np.std(fitness_scores))

        bars = ax.bar(gen_numbers, fitness_std, color="#FF6B6B", alpha=0.7)

        ax.set_xlabel("Generation", fontweight="bold", color="white")
        ax.set_ylabel("Fitness Std Dev", fontweight="bold", color="white")
        ax.set_title("📊 Population Diversity", fontweight="bold", color="white")
        ax.grid(True, alpha=0.2)
        ax.set_facecolor("black")

        # Add value labels on bars
        for bar, std_val in zip(bars, fitness_std, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.001,
                f"{std_val:.3f}",
                ha="center",
                va="bottom",
                color="white",
                fontsize=8,
            )

    def plot_benchmark_evolution(self, ax) -> None:
        """Plot how benchmark scores evolved over generations."""
        generations = self.data["generation_history"]

        # Extract best benchmark scores per generation
        benchmark_evolution = {}
        gen_numbers = []

        for gen_data in generations:
            gen_numbers.append(gen_data["generation"])
            population = gen_data["population"]

            # Get best individual's benchmarks
            best_individual = max(population, key=lambda x: x["fitness"])
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
                markersize=6,
                label=metric.upper(),
            )

        # Add threshold lines
        thresholds = {"mmlu": 0.60, "gsm8k": 0.40, "humaneval": 0.25}
        for metric, threshold in thresholds.items():
            if metric in benchmark_evolution:
                ax.axhline(
                    y=threshold,
                    color="red",
                    linestyle="--",
                    alpha=0.5,
                    label=f"{metric.upper()} Threshold",
                )

        ax.set_xlabel("Generation", fontweight="bold", color="white")
        ax.set_ylabel("Benchmark Score", fontweight="bold", color="white")
        ax.set_title("🎯 Benchmark Score Evolution", fontweight="bold", color="white")
        ax.legend(fontsize=10, facecolor="black", edgecolor="white")
        ax.grid(True, alpha=0.2)
        ax.set_facecolor("black")


def main() -> None:
    """Create evolutionary tree visualization."""
    print("Creating evolutionary tree visualization...")

    visualizer = EvolutionVisualizer()
    if visualizer.data:
        visualizer.create_evolutionary_tree()
        print("✅ Evolutionary tree visualization completed!")
    else:
        print("❌ Could not load evolution results data")


if __name__ == "__main__":
    main()
