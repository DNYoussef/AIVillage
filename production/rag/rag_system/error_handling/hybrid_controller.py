# rag_system/error_handling/hybrid_controller.py


from .adaptive_controller import AdaptiveErrorRateController
from .ltt_controller import LTTErrorController


class HybridErrorController(AdaptiveErrorRateController, LTTErrorController):
    def __init__(self, num_steps: int, target_error_rate: float, adaptation_rate: float, confidence_level: float):
        """Initialize the HybridErrorController.

        :param num_steps: The number of steps in the multi-step process.
        :param target_error_rate: The overall target error rate for the entire process.
        :param adaptation_rate: The rate at which to adapt error rates (between 0 and 1).
        :param confidence_level: The confidence level for LTT calibration.
        """
        AdaptiveErrorRateController.__init__(self, num_steps, target_error_rate, adaptation_rate)
        LTTErrorController.__init__(self, num_steps, target_error_rate, confidence_level)

    def update_error_rates(self, observed_errors: list[float], calibration_data: list[int] = None):
        """Update error rates using both adaptive control and LTT calibration.

        :param observed_errors: A list of observed error rates for each step.
        :param calibration_data: Optional calibration data for LTT calibration.
        """
        if calibration_data:
            self.calibrate(calibration_data)

        AdaptiveErrorRateController.update_error_rates(self, observed_errors)

    def _compute_step_error_rates(self):
        """Compute step error rates using both adaptive and LTT methods.
        """
        LTTErrorController._compute_step_error_rates(self)
        # Further adjust rates using adaptive method if needed
