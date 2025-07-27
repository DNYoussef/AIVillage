import asyncio
import logging
from typing import Any

import matplotlib.pyplot as plt

from rag_system.error_handling.error_handler import (
    error_handler,
    safe_execute,
)

from .base_analytics import BaseAnalytics

logger = logging.getLogger(__name__)


class AnalyticsManager(BaseAnalytics):
    def __init__(self):
        super().__init__()
        self.task_success_rates = {}
        self.system_efficiency_metrics = []

    @error_handler.handle_error
    def record_task_completion(
        self, task_id: str, completion_time: float, success: bool
    ):
        self.record_metric("task_completion_time", completion_time)

        task_type = task_id.split("_")[0]  # Assuming task_id format is "type_uuid"
        if task_type not in self.task_success_rates:
            self.task_success_rates[task_type] = []
        self.task_success_rates[task_type].append(int(success))

    @error_handler.handle_error
    def update_agent_performance(self, agent: str, performance: float):
        self.record_metric(f"{agent}_performance", performance)

    @error_handler.handle_error
    def record_system_efficiency(self, metrics: dict[str, float]):
        self.system_efficiency_metrics.append(metrics)

    @error_handler.handle_error
    def generate_task_success_rate_plot(self) -> str:
        success_rates = {
            task_type: sum(rates) / len(rates)
            for task_type, rates in self.task_success_rates.items()
        }
        plt.figure(figsize=(10, 6))
        plt.bar(success_rates.keys(), success_rates.values())
        plt.title("Task Success Rates by Type")
        plt.xlabel("Task Type")
        plt.ylabel("Success Rate")
        filename = "task_success_rates.png"
        plt.savefig(filename)
        plt.close()
        return filename

    @error_handler.handle_error
    def generate_system_efficiency_plot(self) -> str:
        metrics = list(self.system_efficiency_metrics[0].keys())
        data = {
            metric: [entry[metric] for entry in self.system_efficiency_metrics]
            for metric in metrics
        }

        plt.figure(figsize=(12, 6))
        for metric, values in data.items():
            plt.plot(values, label=metric)
        plt.title("System Efficiency Metrics Over Time")
        plt.xlabel("Time")
        plt.ylabel("Metric Value")
        plt.legend()
        filename = "system_efficiency.png"
        plt.savefig(filename)
        plt.close()
        return filename

    @error_handler.handle_error
    def generate_analytics_report(self) -> dict[str, Any]:
        return {
            "task_completion_time_plot": self.generate_task_completion_time_plot(),
            "agent_performance_plot": self.generate_agent_performance_plot(),
            "task_success_rate_plot": self.generate_task_success_rate_plot(),
            "system_efficiency_plot": self.generate_system_efficiency_plot(),
            "average_task_completion_time": sum(self.task_completion_times)
            / len(self.task_completion_times)
            if self.task_completion_times
            else 0,
            "total_tasks_completed": len(self.task_completion_times),
            "average_agent_performance": {
                agent: sum(performances) / len(performances)
                for agent, performances in self.agent_performance_history.items()
            },
            "overall_task_success_rate": sum(
                sum(rates) for rates in self.task_success_rates.values()
            )
            / sum(len(rates) for rates in self.task_success_rates.values())
            if self.task_success_rates
            else 0,
        }

    @safe_execute
    async def run_analytics(self):
        while True:
            report = self.generate_analytics_report()
            logger.info(f"Generated analytics report: {report}")
            # Here you might want to send this report to a dashboard or store it for later retrieval
            await asyncio.sleep(3600)  # Run analytics every hour
