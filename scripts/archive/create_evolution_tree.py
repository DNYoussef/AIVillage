#!/usr/bin/env python3
"""Create a simplified evolutionary tree visualization."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def create_evolutionary_tree():
    """Create evolutionary tree visualization."""
    # Load results
    results_file = Path("D:/AgentForge/results/evolution_results.json")
    try:
        with open(results_file) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Results file not found: {results_file}")
        return None

    # Set up the plot
    plt.style.use("dark_background")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("black")

    # Colors for different methods
    colors = {"slerp": "#FF6B6B", "linear": "#4ECDC4", "task_arithmetic": "#45B7D1"}

    generations = data["generation_history"]

    # 1. Evolutionary Tree
    ax1.set_title("🌳 Evolutionary Tree", fontsize=14, fontweight="bold", color="white")

    for gen_idx, gen_data in enumerate(generations):
        population = gen_data["population"]

        for i, individual in enumerate(population):
            x = gen_idx
            y = i
            fitness = individual["fitness"]
            method = individual["merge_method"]

            # Size by fitness
            size = 50 + (fitness * 200)
            color = colors.get(method, "#CCCCCC")

            # Mark best individual with star
            if fitness == max(ind["fitness"] for ind in population):
                ax1.scatter(
                    x,
                    y,
                    s=size + 50,
                    c="gold",
                    marker="*",
                    edgecolors="black",
                    linewidth=2,
                    zorder=5,
                )

            ax1.scatter(
                x, y, s=size, c=color, alpha=0.7, edgecolors="white", linewidth=1
            )

            # Add fitness label
            ax1.annotate(
                f"{fitness:.2f}",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="white",
            )

    ax1.set_xlabel("Generation", color="white", fontweight="bold")
    ax1.set_ylabel("Individual", color="white", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Add legend
    for method, color in colors.items():
        ax1.scatter([], [], c=color, s=100, label=method.replace("_", " ").title())
    ax1.scatter([], [], c="gold", s=150, marker="*", label="Best in Generation")
    ax1.legend(facecolor="black", edgecolor="white")

    # 2. Fitness Progression
    ax2.set_title("📈 Fitness Evolution", fontsize=14, fontweight="bold", color="white")

    gen_numbers = [gen["generation"] for gen in generations]
    best_fitness = [gen["best_fitness"] for gen in generations]
    avg_fitness = [gen["average_fitness"] for gen in generations]

    ax2.plot(
        gen_numbers,
        best_fitness,
        "o-",
        color="#6BCF7F",
        linewidth=3,
        markersize=8,
        label="Best Fitness",
    )
    ax2.plot(
        gen_numbers,
        avg_fitness,
        "s-",
        color="#FFD93D",
        linewidth=2,
        markersize=6,
        label="Average Fitness",
    )

    ax2.fill_between(gen_numbers, best_fitness, avg_fitness, alpha=0.3, color="#4ECDC4")
    ax2.set_xlabel("Generation", color="white", fontweight="bold")
    ax2.set_ylabel("Fitness Score", color="white", fontweight="bold")
    ax2.legend(facecolor="black", edgecolor="white")
    ax2.grid(True, alpha=0.3)

    # 3. Method Distribution
    ax3.set_title(
        "🔧 Merge Method Usage", fontsize=14, fontweight="bold", color="white"
    )

    method_counts = {}
    for gen_data in generations:
        for individual in gen_data["population"]:
            method = individual["merge_method"]
            method_counts[method] = method_counts.get(method, 0) + 1

    methods = list(method_counts.keys())
    counts = list(method_counts.values())
    pie_colors = [colors.get(method, "#CCCCCC") for method in methods]

    wedges, texts, autotexts = ax3.pie(
        counts, labels=methods, colors=pie_colors, autopct="%1.1f%%", startangle=90
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
    for text in texts:
        text.set_color("white")
        text.set_fontweight("bold")

    # 4. Benchmark Evolution
    ax4.set_title("🎯 Benchmark Scores", fontsize=14, fontweight="bold", color="white")

    # Extract benchmark evolution
    benchmark_data = {}
    for gen_data in generations:
        best_ind = max(gen_data["population"], key=lambda x: x["fitness"])
        benchmarks = best_ind.get("benchmark_results", {})

        for metric, score in benchmarks.items():
            if metric not in benchmark_data:
                benchmark_data[metric] = []
            benchmark_data[metric].append(score)

    bench_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    for i, (metric, scores) in enumerate(benchmark_data.items()):
        color = bench_colors[i % len(bench_colors)]
        ax4.plot(
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
        if metric in benchmark_data:
            ax4.axhline(
                y=threshold, color="red", linestyle="--", alpha=0.7, linewidth=1
            )

    ax4.set_xlabel("Generation", color="white", fontweight="bold")
    ax4.set_ylabel("Score", color="white", fontweight="bold")
    ax4.legend(facecolor="black", edgecolor="white")
    ax4.grid(True, alpha=0.3)

    # Style all axes
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["top"].set_color("white")
        ax.spines["right"].set_color("white")
        ax.spines["left"].set_color("white")

    plt.tight_layout()
    plt.suptitle(
        "🧬 Agent Forge Evolution Merge - Evolutionary Analysis",
        fontsize=16,
        fontweight="bold",
        color="white",
        y=0.98,
    )

    # Save the plot
    output_file = "D:/AgentForge/results/evolution_tree.png"
    plt.savefig(
        output_file, dpi=300, bbox_inches="tight", facecolor="black", edgecolor="none"
    )

    print(f"✅ Evolutionary tree saved to: {output_file}")

    # Show best configuration
    best_config = data["evolution_summary"]["best_configuration"]
    print("\n🏆 Best Configuration Found:")
    print(f"  Method: {best_config['merge_method']}")
    print(f"  Fitness: {best_config['fitness']:.3f}")
    print(f"  Generation: {best_config['generation']}")
    print(f"  Benchmarks: {best_config['benchmark_results']}")

    plt.show()
    return output_file


if __name__ == "__main__":
    print("Creating evolutionary tree visualization...")
    create_evolutionary_tree()
