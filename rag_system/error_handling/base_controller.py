# rag_system/error_handling/base_controller.py

from typing import List

class ErrorRateController:
    def __init__(self, num_steps: int, target_error_rate: float):
        """
        Initialize the ErrorRateController.

        :param num_steps: The number of steps in the multi-step process.
        :param target_error_rate: The overall target error rate for the entire process.
        """
        self.num_steps = num_steps
        self.target_error_rate = target_error_rate
        self.step_error_rates = self._compute_step_error_rates()

    def _compute_step_error_rates(self) -> List[float]:
        """
        Compute individual step error rates using Bonferroni correction.

        :return: A list of error rates for each step.
        """
        # Use Bonferroni correction to split the overall error rate equally among steps
        return [self.target_error_rate / self.num_steps] * self.num_steps

    def get_step_error_rate(self, step: int) -> float:
        """
        Get the error rate allocated to a specific step.

        :param step: The step index (0-based).
        :return: The error rate for the specified step.
        """
        if 0 <= step < self.num_steps:
            return self.step_error_rates[step]
        else:
            raise ValueError("Invalid step index")

    def update_error_rates(self, observed_errors: List[float]):
        """
        Update the step error rates based on observed errors.

        :param observed_errors: A list of observed error rates for each step.
        """
        # Base implementation does not update error rates
        pass
