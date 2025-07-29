import io
import logging
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

class AdvancedAnalytics:
    def __init__(self):
        self.metrics = {}

    def record_metric(self, metric_name: str, value: float):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def generate_performance_report(self) -> dict[str, Any]:
        report = {}
        for metric_name, values in self.metrics.items():
            report[metric_name] = {
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "latest": values[-1]
            }
        return report

    def visualize_metrics(self) -> bytes:
        fig, axs = plt.subplots(len(self.metrics), 1, figsize=(10, 5 * len(self.metrics)))
        for i, (metric_name, values) in enumerate(self.metrics.items()):
            ax = axs[i] if len(self.metrics) > 1 else axs
            ax.plot(values)
            ax.set_title(metric_name)
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        return buf.getvalue()

    def visualize_knowledge_graph(self, graph: nx.Graph) -> bytes:
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_color="lightblue",
                node_size=1500, font_size=10, font_weight="bold")
        edge_labels = nx.get_edge_attributes(graph, "type")
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        return buf.getvalue()

    def generate_heatmap(self, data: list[list[float]], labels: list[str]) -> bytes:
        plt.figure(figsize=(10, 8))
        sns.heatmap(data, annot=True, cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
        plt.title("Correlation Heatmap")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        return buf.getvalue()

    def generate_bar_chart(self, data: dict[str, float], title: str) -> bytes:
        plt.figure(figsize=(10, 6))
        plt.bar(data.keys(), data.values())
        plt.title(title)
        plt.xlabel("Categories")
        plt.ylabel("Values")
        plt.xticks(rotation=45)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        return buf.getvalue()

    def generate_scatter_plot(self, x: list[float], y: list[float], labels: list[str], title: str) -> bytes:
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y)
        for i, label in enumerate(labels):
            plt.annotate(label, (x[i], y[i]))
        plt.title(title)
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        return buf.getvalue()

    def generate_summary_statistics(self, data: dict[str, list[float]]) -> pd.DataFrame:
        summary = {}
        for key, values in data.items():
            summary[key] = {
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }
        return pd.DataFrame(summary).T

    def perform_correlation_analysis(self, data: dict[str, list[float]]) -> pd.DataFrame:
        df = pd.DataFrame(data)
        return df.corr()

    def detect_anomalies(self, data: list[float], threshold: float = 2.0) -> list[int]:
        mean = np.mean(data)
        std = np.std(data)
        return [i for i, x in enumerate(data) if abs(x - mean) > threshold * std]

    def generate_trend_analysis(self, data: list[float]) -> dict[str, Any]:
        from scipy import stats
        x = range(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "standard_error": std_err
        }
