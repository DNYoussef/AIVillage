# rag_system/error_handling/adaptive_controller.py

from typing import List
from .base_controller import ErrorRateController

class AdaptiveErrorRateController(ErrorRateController):
    def __init__(self, num_steps: int, target_error_rate: float, adaptation_rate: float):
        """
        Initialize the AdaptiveErrorRateController.

        :param num_steps: The number of steps in the multi-step process.
        :param target_error_rate: The overall target error rate for the entire process.
        :param adaptation_rate: The rate at which to adapt error rates (between 0 and 1).
        """
        super().__init__(num_steps, target_error_rate)
        self.adaptation_rate = adaptation_rate

    def update_error_rates(self, observed_errors: List[float]):
        """
        Update the step error rates based on observed errors using adaptive control.

        :param observed_errors: A list of observed error rates for each step.
        """
        for i, observed_error in enumerate(observed_errors):
            error_diff = observed_error - self.step_error_rates[i]
            self.step_error_rates[i] += self.adaptation_rate * error_diff

        # Renormalize to ensure sum of error rates equals target_error_rate
        total_error = sum(self.step_error_rates)
        self.step_error_rates = [rate * self.target_error_rate / total_error for rate in self.step_error_rates]
