import io
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import logging
from rag_system.utils.advanced_analytics import AdvancedAnalytics
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rag_system.utils.advanced_analytics import AdvancedAnalytics

logger = logging.getLogger(__name__)

class ComprehensiveEvaluationFramework:
    def __init__(self, advanced_analytics: AdvancedAnalytics):
        self.advanced_analytics = advanced_analytics
        self.metrics: Dict[str, List[float]] = {}
        self.timestamps: Dict[str, List[datetime]] = {}
        self.metric_descriptions: Dict[str, str] = {}

    def add_metric(self, name: str, description: str):
        if name not in self.metrics:
            self.metrics[name] = []
            self.timestamps[name] = []
            self.metric_descriptions[name] = description
            logger.info(f"Added new metric: {name} - {description}")

    def record_metric(self, name: str, value: float):
        if name not in self.metrics:
            logger.warning(f"Metric '{name}' not found. Adding it with a default description.")
            self.add_metric(name, "No description provided")

        self.metrics[name].append(value)
        self.timestamps[name].append(datetime.now())
        self.advanced_analytics.record_metric(name, value)
        logger.debug(f"Recorded value {value} for metric {name}")

    def get_metric_stats(self, name: str) -> Dict[str, float]:
        if name not in self.metrics:
            logger.error(f"Metric '{name}' not found.")
            return {}

        values = self.metrics[name]
        return {
            "latest": values[-1] if values else None,
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values)
        }

    def get_metric_trend(self, name: str, window: int = 10) -> float:
        if name not in self.metrics or len(self.metrics[name]) < window:
            logger.warning(f"Not enough data to calculate trend for metric '{name}'")
            return None

        recent_values = self.metrics[name][-window:]
        x = range(len(recent_values))
        y = recent_values
        slope, _ = np.polyfit(x, y, 1)
        return slope

    def generate_performance_report(self) -> Dict[str, Any]:
        report = {}
        for name in self.metrics:
            report[name] = {
                "description": self.metric_descriptions[name],
                "stats": self.get_metric_stats(name),
                "trend": self.get_metric_trend(name)
            }
        return report

    def generate_visualizations(self) -> Dict[str, bytes]:
        visualizations = {}

        # Time series plot for all metrics
        plt.figure(figsize=(12, 6))
        for name in self.metrics:
            plt.plot(self.timestamps[name], self.metrics[name], label=name)
        plt.legend()
        plt.title("Metrics Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("Value")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        visualizations['metrics_over_time'] = buf.getvalue()
        plt.close()

        # Correlation heatmap
        df = pd.DataFrame(self.metrics)
        correlation_matrix = df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title("Metric Correlation Heatmap")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        visualizations['correlation_heatmap'] = buf.getvalue()
        plt.close()

        return visualizations

    def evaluate_system_performance(self) -> Dict[str, Any]:
        performance_report = self.generate_performance_report()
        visualizations = self.generate_visualizations()

        # Calculate overall performance score (this is a simplified example)
        metric_scores = [stats['stats']['latest'] for stats in performance_report.values() if stats['stats']['latest'] is not None]
        overall_score = np.mean(metric_scores) if metric_scores else None

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_performance_score": overall_score,
            "metric_reports": performance_report,
            "visualizations": visualizations
        }

    def log_evaluation_results(self, results: Dict[str, Any]):
        logger.info(f"Evaluation Results at {results['timestamp']}:")
        logger.info(f"Overall Performance Score: {results['overall_performance_score']}")
        for metric, report in results['metric_reports'].items():
            logger.info(f"{metric}: Latest = {report['stats']['latest']}, Trend = {report['trend']}")
