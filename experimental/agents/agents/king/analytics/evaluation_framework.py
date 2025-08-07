from collections import deque
import io
import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class EvaluationFramework:
    def __init__(self, metrics_history_length: int = 1000) -> None:
        self.metrics_history_length = metrics_history_length
        self.metrics = {
            "response_time": deque(maxlen=metrics_history_length),
            "task_success_rate": deque(maxlen=metrics_history_length),
            "user_satisfaction": deque(maxlen=metrics_history_length),
            "eudaimonia_score": deque(maxlen=metrics_history_length),
            "knowledge_integration_rate": deque(maxlen=metrics_history_length),
            "decision_quality": deque(maxlen=metrics_history_length),
            "nlp_accuracy": deque(maxlen=metrics_history_length),
            "rag_relevance": deque(maxlen=metrics_history_length),
        }

    def record_metric(self, metric_name: str, value: float) -> None:
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
        else:
            logger.warning(f"Unknown metric: {metric_name}")

    def get_metric_average(self, metric_name: str) -> float:
        if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
            return np.mean(self.metrics[metric_name])
        return 0.0

    def get_metric_trend(self, metric_name: str, window: int = 100) -> float:
        if metric_name in self.metrics and len(self.metrics[metric_name]) >= window:
            recent_values = list(self.metrics[metric_name])[-window:]
            return (recent_values[-1] - recent_values[0]) / window
        return 0.0

    def generate_performance_report(self) -> dict[str, Any]:
        report = {}
        for metric_name in self.metrics:
            report[metric_name] = {
                "average": self.get_metric_average(metric_name),
                "trend": self.get_metric_trend(metric_name),
            }
        return report

    def visualize_metrics(self) -> bytes:
        fig, axs = plt.subplots(4, 2, figsize=(15, 20))
        fig.suptitle("Performance Metrics Visualization")

        for i, (metric_name, values) in enumerate(self.metrics.items()):
            row = i // 2
            col = i % 2
            axs[row, col].plot(list(values))
            axs[row, col].set_title(metric_name)
            axs[row, col].set_xlabel("Time")
            axs[row, col].set_ylabel("Value")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        return buf.getvalue()

    async def evaluate_response(
        self, response: dict[str, Any], user_feedback: dict[str, Any]
    ) -> dict[str, float]:
        """Evaluate the response based on various metrics."""
        evaluation = {}

        # Response time
        evaluation["response_time"] = response.get("execution_time", 0)

        # Task success rate
        evaluation["task_success_rate"] = 1.0 if response.get("success", False) else 0.0

        # User satisfaction (assuming user_feedback contains a 'satisfaction' score)
        evaluation["user_satisfaction"] = user_feedback.get("satisfaction", 0.0)

        # Eudaimonia score
        evaluation["eudaimonia_score"] = response.get("eudaimonia_score", 0.0)

        # Knowledge integration rate (assuming response contains 'new_knowledge_integrated')
        evaluation["knowledge_integration_rate"] = (
            1.0 if response.get("new_knowledge_integrated", False) else 0.0
        )

        # Decision quality (assuming response contains a 'decision_confidence' score)
        evaluation["decision_quality"] = response.get("decision_confidence", 0.0)

        # NLP accuracy (assuming response contains 'nlp_accuracy')
        evaluation["nlp_accuracy"] = response.get("nlp_accuracy", 0.0)

        # RAG relevance (assuming response contains 'rag_relevance')
        evaluation["rag_relevance"] = response.get("rag_relevance", 0.0)

        # Record all metrics
        for metric_name, value in evaluation.items():
            self.record_metric(metric_name, value)

        return evaluation

    async def analyze_performance(self) -> dict[str, Any]:
        """Analyze overall performance and provide insights."""
        report = self.generate_performance_report()
        insights = []

        for metric_name, data in report.items():
            if data["trend"] > 0.05:
                insights.append(f"{metric_name} is showing significant improvement.")
            elif data["trend"] < -0.05:
                insights.append(f"{metric_name} is declining and may need attention.")

        if report["user_satisfaction"]["average"] < 0.7:
            insights.append(
                "User satisfaction is below target. Consider reviewing and improving response quality."
            )

        if report["rag_relevance"]["average"] < 0.8:
            insights.append(
                "RAG system relevance is below expectations. Consider fine-tuning or expanding the knowledge base."
            )

        return {
            "report": report,
            "insights": insights,
            "visualization": self.visualize_metrics(),
        }

    def reset_metrics(self) -> None:
        """Reset all metrics to their initial state."""
        for metric in self.metrics:
            self.metrics[metric].clear()
        logger.info("All metrics have been reset.")

    async def save_metrics(self, path: str) -> None:
        """Save the current state of metrics to a file."""
        import json

        with open(path, "w") as f:
            json.dump({k: list(v) for k, v in self.metrics.items()}, f)
        logger.info(f"Metrics saved to {path}")

    async def load_metrics(self, path: str) -> None:
        """Load metrics from a file."""
        import json

        with open(path) as f:
            loaded_metrics = json.load(f)
        for k, v in loaded_metrics.items():
            self.metrics[k] = deque(v, maxlen=self.metrics_history_length)
        logger.info(f"Metrics loaded from {path}")
