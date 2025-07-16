from collections import deque
import logging
from typing import Any

import numpy as np
from scipy.stats import linregress

logger = logging.getLogger(__name__)

class UnifiedAnalytics:
    def __init__(self, history_length: int = 100):
        self.metrics = {}
        self.task_history = deque(maxlen=history_length)
        self.performance_history = deque(maxlen=history_length)
        self.learning_rate = 0.01

    def record_metric(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        logger.debug(f"Recorded metric {name}: {value}")

    def get_metric_stats(self, name: str) -> dict[str, float]:
        if name not in self.metrics:
            logger.warning(f"Metric {name} not found")
            return {}
        values = self.metrics[name]
        return {
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values)
        }

    def record_task_completion(self, task_id: str, completion_time: float, success: bool):
        self.record_metric("task_completion_time", completion_time)
        self.task_history.append({"task_id": task_id, "completion_time": completion_time, "success": success})

    def update_performance_history(self, performance: float):
        self.performance_history.append(performance)

    def get_performance_trend(self) -> float:
        if len(self.performance_history) < 2:
            return 0.0
        x = np.arange(len(self.performance_history))
        y = np.array(self.performance_history)
        slope, _, _, _, _ = linregress(x, y)
        return slope

    def generate_summary_report(self) -> dict[str, Any]:
        report = {}
        for name in self.metrics:
            report[name] = self.get_metric_stats(name)
        report["performance_trend"] = self.get_performance_trend()
        report["task_success_rate"] = self.calculate_task_success_rate()
        return report

    def calculate_task_success_rate(self) -> float:
        if not self.task_history:
            return 0.0
        successful_tasks = sum(1 for task in self.task_history if task["success"])
        return successful_tasks / len(self.task_history)

    async def evolve(self):
        """Adapt internal parameters based on recent history."""
        trend = self.get_performance_trend()

        if trend > 0:
            self.learning_rate *= 0.95  # reduce step size when improving
        elif trend < 0:
            self.learning_rate *= 1.05  # try to learn faster when degrading

        self.learning_rate = float(max(0.0001, min(0.1, self.learning_rate)))

        # Remove metrics that have not changed recently to keep history small
        for name in list(self.metrics.keys()):
            values = self.metrics[name]
            if len(values) > 10 and np.std(values[-10:]) == 0:
                logger.info(f"Pruning stagnant metric {name}")
                del self.metrics[name]

    def get_info(self) -> dict[str, Any]:
        return {
            "metrics": list(self.metrics.keys()),
            "task_history_length": len(self.task_history),
            "performance_history_length": len(self.performance_history),
            "learning_rate": self.learning_rate
        }

# You can add more methods from the original files as needed
