# rag_system/error_handling/ltt_controller.py


import numpy as np
from scipy.stats import binom

from .base_controller import ErrorRateController


class LTTErrorController(ErrorRateController):
    def __init__(
        self, num_steps: int, target_error_rate: float, confidence_level: float
    ):
        """Initialize the LTTErrorController.

        :param num_steps: The number of steps in the multi-step process.
        :param target_error_rate: The overall target error rate for the entire process.
        :param confidence_level: The confidence level for LTT calibration.
        """
        super().__init__(num_steps, target_error_rate)
        self.confidence_level = confidence_level
        self.calibrated_error_rate = None

    def calibrate(self, calibration_data: list[int]):
        """Calibrate the error rates using the Learn Then Test framework.

        :param calibration_data: A list indicating whether an error was observed in calibration (1 for error, 0 for no error).
        """
        n = len(calibration_data)
        errors = sum(calibration_data)

        def compute_p_value(lambda_param):
            return 1 - binom.cdf(errors, n, lambda_param)

        valid_lambdas = []
        lambda_range = np.linspace(0, 1, 1000)
        for lambda_param in lambda_range:
            p_value = compute_p_value(lambda_param)
            if p_value >= self.confidence_level:
                valid_lambdas.append(lambda_param)

        if valid_lambdas:
            self.calibrated_error_rate = min(valid_lambdas)
            self._compute_step_error_rates()
        else:
            raise ValueError(
                "No valid lambda found for the given confidence level and calibration data."
            )

    def _compute_step_error_rates(self):
        """Compute step error rates based on the calibrated error rate."""
        if self.calibrated_error_rate is not None:
            # Allocate the calibrated error rate across steps
            self.step_error_rates = [
                self.calibrated_error_rate / self.num_steps
            ] * self.num_steps
        else:
            # Use default error rates if not calibrated
            super()._compute_step_error_rates()
