from abc import ABC, abstractmethod
import logging
from typing import Any, NoReturn

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class BaseAnalytics(ABC):
    def __init__(self) -> None:
        self.metrics: dict[str, list[float]] = {}

    def record_metric(self, metric: str, value: float) -> None:
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
    def generate_analytics_report(self) -> dict[str, Any]:
        """generate_analytics_report - Planned feature not yet implemented.

        This functionality is part of the Atlantis roadmap.
        """
        msg = "'generate_analytics_report' is not yet implemented. Track progress: https://github.com/DNYoussef/AIVillage/issues/feature-generate_analytics_report"
        raise NotImplementedError(msg)

    def save(self, path: str) -> NoReturn:
        raise NotImplementedError

    def load(self, path: str) -> NoReturn:
        raise NotImplementedError
