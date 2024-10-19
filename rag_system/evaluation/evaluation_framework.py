import logging
from typing import Dict, Any, List
from datetime import datetime
import numpy as np
import pandas as pd
from rag_system.utils.advanced_analytics import AdvancedAnalytics

logger = logging.getLogger(__name__)

class EvaluationMetric:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.values = []
        self.timestamps = []

    def add_value(self, value: float):
        self.values.append(value)
        self.timestamps.append(datetime.now())

    def get_latest_value(self) -> float:
        return self.values[-1] if self.values else None

    def get_average(self) -> float:
        return np.mean(self.values) if self.values else None

    def get_trend(self, window: int = 10) -> float:
        if len(self.values) < window:
            return None
        recent_values = self.values[-window:]
        return (recent_values[-1] - recent_values[0]) / window

class EvaluationFramework:
    def __init__(self, advanced_analytics: AdvancedAnalytics):
        self.metrics: Dict[str, EvaluationMetric] = {}
        self.advanced_analytics = advanced_analytics

    def add_metric(self, name: str, description: str):
        if name not in self.metrics:
            self.metrics[name] = EvaluationMetric(name, description)

    def record_metric(self, name: str, value: float):
        if name not in self.metrics:
            logger.warning(f"Metric '{name}' not found. Adding it with a default description.")
            self.add_metric(name, "No description provided")
        self.metrics[name].add_value(value)
        self.advanced_analytics.record_metric(name, value)

    def get_metric_value(self, name: str) -> float:
        return self.metrics[name].get_latest_value() if name in self.metrics else None

    def get_metric_average(self, name: str) -> float:
        return self.metrics[name].get_average() if name in self.metrics else None

    def get_metric_trend(self, name: str, window: int = 10) -> float:
        return self.metrics[name].get_trend(window) if name in self.metrics else None

    def generate_report(self) -> Dict[str, Any]:
        report = {}
        for name, metric in self.metrics.items():
            report[name] = {
                "description": metric.description,
                "latest_value": metric.get_latest_value(),
                "average": metric.get_average(),
                "trend": metric.get_trend()
            }
        return report

    def generate_performance_summary(self) -> str:
        report = self.generate_report()
        summary = "Performance Summary:\n\n"
        for name, data in report.items():
            summary += f"{name}:\n"
            summary += f"  Description: {data['description']}\n"
            summary += f"  Latest Value: {data['latest_value']:.4f}\n"
            summary += f"  Average: {data['average']:.4f}\n"
            summary += f"  Trend: {data['trend']:.4f}\n\n"
        return summary

    def generate_visualizations(self) -> Dict[str, bytes]:
        visualizations = {}
        visualizations['metrics_over_time'] = self.advanced_analytics.visualize_metrics()
        
        # Generate correlation heatmap
        metric_values = {name: metric.values for name, metric in self.metrics.items()}
        df = pd.DataFrame(metric_values)
        correlation_matrix = df.corr()
        visualizations['correlation_heatmap'] = self.advanced_analytics.generate_heatmap(
            correlation_matrix.values.tolist(),
            correlation_matrix.index.tolist()
        )
        
        return visualizations

    def evaluate_system_performance(self) -> Dict[str, Any]:
        report = self.generate_report()
        visualizations = self.generate_visualizations()
        
        overall_performance = np.mean([data['latest_value'] for data in report.values() if data['latest_value'] is not None])
        
        return {
            "report": report,
            "visualizations": visualizations,
            "overall_performance": overall_performance,
            "summary": self.generate_performance_summary()
        }
