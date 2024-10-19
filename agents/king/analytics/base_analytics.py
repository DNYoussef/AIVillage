import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class BaseAnalytics(ABC):
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}

    def record_metric(self, metric: str, value: float):
        if metric not in self.metrics:
            self.metrics[metric] = []
        self.metrics[metric].append(value)
        logger.debug(f"Recorded {metric}: {value}")

    def generate_metric_plot(self, metric: str) -> str:
        if metric not in self.metrics:
            logger.warning(f"No data for metric {metric}")
            return ""
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics[metric])
        plt.title(f"{metric} Over Time")
        plt.xlabel("Time")
        plt.ylabel(metric)
        
        filename = f"{metric}.png"
        plt.savefig(filename)
        plt.close()
        return filename

    @abstractmethod
    def generate_analytics_report(self) -> Dict[str, Any]:
        pass

    def save(self, path: str):
        raise NotImplementedError()

    def load(self, path: str):
        raise NotImplementedError()
